"""
Consulta Terrestre — motor simbólico de consultas espaciales.

Sin LLM como respondedor: las preguntas en español se mapean a PLANTILLAS y se
ejecutan como operaciones deterministas sobre las máscaras de clase por frame
(``entrega/masks/``, ver ``cansat/masks.py``) y la telemetría. Si la pregunta
no mapea a ninguna plantilla, la respuesta es **"consulta no soportada"** con
sugerencias: nunca se inventa un número.

Operaciones soportadas:
  · ``area``             A o A∩B, en m² (escala por ``area_m2`` de la telemetría)
  · ``count``            componentes conexas de A∩B∩buffer(C, r metros)
  · ``length_fraction``  longitud(A∩B)/longitud(A) por esqueleto
  · ``distance``         distancia mínima/media de A a B, en metros
  · ``personas``         suma del conteo por frame (opcionalmente por zona)

Zona (opcional): polígono en lon/lat dibujado en la estación. El
georreferenciado es APROXIMADO y se declara en cada respuesta:
  · el frame es un cuadrado de lado ``sqrt(area_m2)`` centrado en ``(lat, lon)``,
  · norte arriba, sin corrección por actitud (la telemetría no trae IMU/heading).

Uso desde la estación (subproceso) o CLI:
    python tools/consulta.py --q "¿cuántos edificios con daño hay a menos de 50 m de una vía?"
"""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import masks as MK

# ══════════════════════════════════════════════════════════════════════════ #
#  Sujetos y fuentes de máscara
# ══════════════════════════════════════════════════════════════════════════ #
#: Sinónimos normalizados (minúsculas, sin acentos) → sujeto canónico.
SUJETOS: dict[str, tuple[str, ...]] = {
    "edificio": (
        "edificio", "edificios", "construccion", "construcciones", "vivienda",
        "viviendas", "casa", "casas", "estructura", "estructuras",
        # Inglés (benchmark EarthVQA). OJO: 'construction' NO va acá: en EarthVQA
        # 'construction land/area' es uso de suelo y respondíamos existencia de
        # edificios — un falso soporte medido y corregido (2026-09-20).
        "building", "buildings", "house", "houses", "structure", "structures",
    ),
    "vegetacion": (
        "vegetacion", "vegetal", "verde", "arbol", "arboles", "bosque", "forestal",
        "vegetation", "green", "tree", "trees", "forest", "woodland",
        # EarthVQA: 'agriculture' cae en vegetación en nuestro remapeo LoveDA.
        "agriculture", "agricultural",
    ),
    "agua": ("agua", "aguas", "lago", "laguna", "rio",
             "water", "lake", "river", "pond"),
    "inundacion": (
        "inundacion", "inundaciones", "inundado", "inundada", "inundados",
        "inundadas", "anegado", "anegada", "anegados", "anegadas", "flood",
        "anegamiento",
        "floods", "flooded", "flooding", "inundation",
    ),
    "via": (
        "via", "vias", "ruta", "rutas", "calle", "calles", "camino", "caminos",
        "carretera", "carreteras", "avenida", "avenidas",
        "road", "roads", "street", "streets", "route", "routes", "highway",
        "highways", "avenue", "path", "paths",
    ),
    "dano": (
        "dano", "danos", "danado", "danada", "danados", "danadas", "destruido",
        "destruida", "destruidos", "destruidas", "derrumbado", "derrumbada",
        "derrumbados", "derrumbadas",
        "damage", "damaged", "destroyed", "collapsed", "ruin", "ruins",
    ),
    "suelo": ("suelo", "suelos", "tierra", "terreno desnudo", "suelo desnudo",
              "soil", "bare soil", "bare ground", "barren",
              # EarthVQA: 'uncultivated agricultural land' es barbecho, no
              # vegetación; el match por frase gana sobre 'agricultural'.
              "uncultivated agricultural land", "uncultivated land", "fallow land",
              "fallow", "uncultivated"),
}

#: Sujeto canónico → (sufijo de máscara, valores de clase que lo forman).
FUENTES: dict[str, tuple[str, tuple[int, ...]]] = {
    "edificio": ("terreno", (1,)),
    "vegetacion": ("terreno", (0,)),
    "agua": ("terreno", (2,)),
    "suelo": ("terreno", (3,)),
    "inundacion": ("flood", (1,)),
    "dano": ("dano2", (2,)),      # preferido; fallback a "dano" si no existe
    "via": ("vias", (1,)),
}

SUFIJOS = ("terreno", "dano", "dano2", "flood", "fuego", "sev", "vias")
PREFERENCIA_DANO = ("dano2", "dano")

#: Componente mínimo para contar (filtra píxeles sueltos de ruido).
MIN_AREA_PX: int = 3

M_PER_DEG_LAT: float = 110540.0
M_PER_DEG_LON_EQ: float = 111320.0

SUGERENCIAS: list[str] = [
    "área de edificios inundados",
    "¿cuántos edificios con daño hay a menos de 50 m de una vía?",
    "¿qué fracción de las vías está inundada?",
    "distancia entre edificios y agua",
    "¿cuántas personas hay en la zona?",
]

GEOREF_NOTA = (
    "georreferenciado APROXIMADO por FOV declarado: cuadrado de lado "
    "sqrt(area_m2) centrado en (lat, lon), norte arriba, sin heading (la "
    "telemetría no trae IMU)."
)
LIMITACIONES: list[str] = [
    GEOREF_NOTA,
    "las personas se cuentan por frame: la telemetría no guarda la posición de "
    "cada detección, sólo el total.",
    "la longitud se estima por esqueletización (≈ ±10 %).",
]

_FLAG_INUNDACION = (
    r"(?:inundad|anegad|bajo agua|bajo el agua|flood|flooded|flooding|under water)"
)
_FLAG_DANO = r"(?:con dano|danad|destruid|derrumb|damaged|destroyed|collapsed)"
_ALT = "|".join(sorted({s for sins in SUJETOS.values() for s in sins},
                       key=len, reverse=True))


# ══════════════════════════════════════════════════════════════════════════ #
#  Normalización y parsing
# ══════════════════════════════════════════════════════════════════════════ #
def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, sin signos (salvo decimales), espacios simples."""
    s = unicodedata.normalize("NFD", str(texto).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9,.:%]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canon_de(palabra: str) -> str | None:
    for canon, sinonimos in SUJETOS.items():
        if palabra in sinonimos:
            return canon
    return None


def _buscar_sujeto(t: str, desde: int = 0) -> tuple[str | None, int, int]:
    m = re.search(rf"\b({_ALT})\b", t[desde:])
    if not m:
        return None, -1, -1
    return canon_de(m.group(1)), desde + m.start(), desde + m.end()


def _disponible(canon: str, disponibles: set[str]) -> bool:
    if canon == "persona":
        return True
    if canon == "dano":
        return bool(set(PREFERENCIA_DANO) & disponibles)
    return FUENTES[canon][0] in disponibles


def _spec(
    plantilla: str,
    operacion: str,
    a: str,
    b: str | None,
    texto: str,
    disponibles: set[str],
    c: str | None = None,
    buffer_m: float | None = None,
) -> dict:
    """Construye el spec validando que las máscaras necesarias existan."""
    faltan = [s for s in (a, b, c)
              if s and s != "persona" and not _disponible(s, disponibles)]
    if faltan:
        return {
            "soportada": False,
            "consulta": texto,
            "motivo": f"esta misión no tiene máscara de: {', '.join(faltan)}",
            "sugerencias": SUGERENCIAS,
        }
    return {
        "soportada": True,
        "plantilla": plantilla,
        "operacion": operacion,
        "a": a,
        "b": b,
        "c": c,
        "buffer_m": buffer_m,
        "texto": texto,
    }


def _sujeto_implicito(resto: str, a: str) -> str | None:
    """Sujeto B implícito en el modificador ('edificios inundados' → inundación)."""
    if a != "inundacion" and re.search(_FLAG_INUNDACION, resto):
        return "inundacion"
    if a != "dano" and re.search(_FLAG_DANO, resto):
        return "dano"
    return None


def _con_estado(t: str, a: str, fin: int) -> tuple[str, str | None]:
    """
    Resuelve el orden sujeto/estado.

    En español el sujeto suele ir primero ('edificios inundados'), pero en
    inglés el estado va primero ('flooded buildings'): si el primer sujeto es
    un estado (inundación/daño) y hay otro sujeto después, se intercambian.
    """
    b = _sujeto_implicito(t[fin:], a)
    if b is None and a in ("inundacion", "dano"):
        otro, _, _ = _buscar_sujeto(t, fin)
        if otro:
            return otro, a
    return a, b


def _area_es_cabeza(t: str, m: re.Match) -> bool:
    """
    ¿La palabra 'area' es el núcleo de una consulta de superficie?

    Evita el falso soporte de 'What are the road types around the residential
    area?': 'area' ahí es parte de un sintagma, no una operación. Se exige que
    detrás venga 'of/de/del…' o un estado ('area inundada'), lo que cubre
    'What is the area of roads?' y '¿cuál es el área de los edificios?'.
    """
    resto = t[m.end():]
    if re.match(r"\s*(?:of|de|del|de la|de los|de las|is|esta|estan|es)\b", resto):
        return True
    return bool(re.match(rf"\s*(?:{_FLAG_INUNDACION}|{_FLAG_DANO})", resto))


def _buffer(t: str, desde: int) -> tuple[str | None, float | None]:
    """'a menos de 50 m de una vía' / 'within 50 m of a road' → (C, metros)."""
    m = re.search(
        r"(?:a menos de|dentro de|within|less than)\s+(\d+(?:[.,]\d+)?)\s*"
        r"(km|kilometers?|kilometros?|m|meters?|metros?|metres?)\s*(?:de|of|from)",
        t[desde:])
    if not m:
        return None, None
    dist = float(m.group(1).replace(",", "."))
    if m.group(2).startswith("k"):
        dist *= 1000.0
    c, _, _ = _buscar_sujeto(t, desde + m.end())
    return (c, dist) if c else (None, None)


def parsear(consulta: str, disponibles: set[str] | None = None) -> dict:
    """Mapea la consulta a un spec; devuelve ``soportada=False`` si no mapea."""
    t = normalizar(consulta)
    disp = set(disponibles or ())

    # 1) área / superficie
    m_area = re.search(r"\b(?:area|superficie|surface)\b", t)
    if m_area and _area_es_cabeza(t, m_area):
        a, _, fin = _buscar_sujeto(t)
        if a:
            a, b = _con_estado(t, a, fin)
            return _spec("area", "area", a, b, t, disp)

    # 2) fracción / porcentaje (longitud)
    if re.search(r"\b(?:fraccion|porcentaje|proporcion|fraction|percentage|proportion)\b",
                 t):
        a, _, fin = _buscar_sujeto(t)
        if a:
            a, b = _con_estado(t, a, fin)
            if b:
                return _spec("length_fraction", "length_fraction", a, b, t, disp)

    # 3) conteo (cuántos X …)
    m = re.search(r"\b(?:cuant[oa]s?|numero de|conteo de|cantidad de"
                  r"|how many|number of|count of|amount of)\b", t)
    if m:
        if re.search(r"\b(?:personas?|people|persons?)\b", t):
            return _spec("personas", "personas", "persona", None, t, disp)
        a, _, fin = _buscar_sujeto(t, m.end())
        if a:
            a, b = _con_estado(t, a, fin)
            c, buffer_m = _buffer(t, fin)
            if c:
                return _spec("count", "count", a, b, t, disp, c=c, buffer_m=buffer_m)
            return _spec("count", "count", a, b, t, disp)

    # 4) distancia entre A y B
    if re.search(r"\b(?:distancia|distance)\b", t):
        a, _, fin = _buscar_sujeto(t)
        if a:
            b, _, _ = _buscar_sujeto(t, fin)
            if b:
                return _spec("distance", "distance", a, b, t, disp)

    # 5) existencia (preguntas Yes/No de EarthVQA: 'Is there any X?', '¿hay X?')
    if re.search(r"\b(?:hay|is there|are there|existe|existen)\b", t):
        if re.search(r"\b(?:personas?|people|persons?)\b", t):
            return _spec("personas", "personas", "persona", None, t, disp)
        a, _, fin = _buscar_sujeto(t)
        if a:
            a, b = _con_estado(t, a, fin)
            return _spec("exists", "exists", a, b, t, disp)

    # 6) personas sueltas ("¿cuántas personas hay?" / "how many people?")
    if re.search(r"\b(?:personas?|people|persons?)\b", t) and re.search(
            r"\b(?:cuant|numero|conteo|cantidad|hay|how many|number of)\b", t):
        return _spec("personas", "personas", "persona", None, t, disp)

    return {
        "soportada": False,
        "consulta": consulta,
        "motivo": "consulta no soportada: no mapea a ninguna plantilla conocida",
        "sugerencias": SUGERENCIAS,
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  Datos de la misión
# ══════════════════════════════════════════════════════════════════════════ #
def _flotante(v) -> float | None:
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


@dataclass
class DatosMision:
    """Máscaras por frame + telemetría, con cache de lectura."""

    masks_dir: Path
    telemetry_csv: Path
    filas: list[dict] = field(default_factory=list)
    _mascaras: dict = field(default_factory=dict)
    _fuentes: set[str] = field(default_factory=set)

    @classmethod
    def cargar(cls, masks_dir: str | Path, telemetry_csv: str | Path) -> DatosMision:
        p = Path(telemetry_csv)
        filas: list[dict] = []
        if p.is_file():
            with p.open("r", encoding="utf-8-sig", newline="") as fh:
                filas = [r for r in csv.DictReader(fh) if r.get("src")]
        datos = cls(Path(masks_dir), p, filas)
        mdir = datos.masks_dir
        if mdir.is_dir():
            datos._fuentes = {
                suf for suf in SUFIJOS
                if next(mdir.glob(f"*_{suf}.png"), None) is not None
            }
        return datos

    @property
    def fuentes(self) -> set[str]:
        return set(self._fuentes)

    def fila(self, src: str) -> dict:
        for r in self.filas:
            if str(r.get("src")) == src:
                return r
        return {}

    def mascara(self, src: str, fuente: str) -> np.ndarray | None:
        key = (src, fuente)
        if key not in self._mascaras:
            self._mascaras[key] = MK.load_mask(self.masks_dir / f"{src}_{fuente}.png")
        return self._mascaras[key]

    def mascara_sujeto(self, src: str, canon: str) -> np.ndarray | None:
        """Máscara binaria del sujeto canónico, o ``None`` si no está en el frame."""
        if canon == "persona":
            return None
        fuente, valores = FUENTES[canon]
        if canon == "dano" and not set(PREFERENCIA_DANO) & self._fuentes:
            return None
        if canon == "dano" and "dano2" not in self._fuentes:
            fuente = "dano"
        m = self.mascara(src, fuente)
        if m is None:
            return None
        out = np.isin(m, valores).astype(np.uint8)
        out[m == MK.NODATA] = 0
        return out

    def area_frame(self, src: str) -> float:
        return _flotante(self.fila(src).get("area_m2")) or 0.0

    def personas(self, src: str) -> int:
        v = _flotante(self.fila(src).get("people"))
        return int(round(v)) if v is not None else 0

    def tiene_geo(self) -> bool:
        for r in self.filas:
            if _flotante(r.get("lat")) is not None and _flotante(r.get("lon")) is not None:
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════ #
#  Georreferenciado aproximado
# ══════════════════════════════════════════════════════════════════════════ #
def _geometria(fila: dict) -> tuple[float, float, float] | None:
    lat0 = _flotante(fila.get("lat"))
    lon0 = _flotante(fila.get("lon"))
    area = _flotante(fila.get("area_m2"))
    if lat0 is None or lon0 is None or not area or area <= 0:
        return None
    return lat0, lon0, area


def poligono_a_pixeles(
    poligono: list | np.ndarray, fila: dict, shape: tuple[int, int]
) -> np.ndarray | None:
    """Polígono lon/lat → píxeles del frame; ``None`` si no toca la huella."""
    geo = _geometria(fila)
    if geo is None:
        return None
    lat0, lon0, area = geo
    h, w = shape
    escala = math.sqrt(area) / w                 # m/px (FOV cuadrado)
    mlon = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    pts = []
    for lon, lat in np.asarray(poligono, dtype=float):
        x = w / 2.0 + (lon - lon0) * mlon / escala
        y = h / 2.0 - (lat - lat0) * M_PER_DEG_LAT / escala
        pts.append((x, y))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if max(xs) < 0 or min(xs) >= w or max(ys) < 0 or min(ys) >= h:
        return None
    return np.round(np.array(pts)).astype(np.int32)


def pixeles_a_geo(poligono_px: np.ndarray, fila: dict,
                  shape: tuple[int, int]) -> list[list[float]]:
    """Inverso de :func:`poligono_a_pixeles` (para dibujar los resultados)."""
    geo = _geometria(fila)
    if geo is None:
        return []
    lat0, lon0, area = geo
    h, w = shape
    escala = math.sqrt(area) / w
    mlon = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    out = []
    for x, y in np.asarray(poligono_px, dtype=float):
        lon = lon0 + (x - w / 2.0) * escala / mlon
        lat = lat0 - (y - h / 2.0) * escala / M_PER_DEG_LAT
        out.append([round(lon, 7), round(lat, 7)])
    return out


def region_toca_frame(poligono: list | np.ndarray, fila: dict) -> bool:
    """True si la huella del frame (cuadrado FOV) intersecta el polígono."""
    geo = _geometria(fila)
    if geo is None:
        return False
    lat0, lon0, area = geo
    mlon = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    xs = [(float(lon) - lon0) * mlon for lon, _ in poligono]
    ys = [(float(lat) - lat0) * M_PER_DEG_LAT for _, lat in poligono]
    mitad = math.sqrt(area) / 2.0
    return (max(xs) >= -mitad and min(xs) <= mitad
            and max(ys) >= -mitad and min(ys) <= mitad)


# ══════════════════════════════════════════════════════════════════════════ #
#  Ejecución
# ══════════════════════════════════════════════════════════════════════════ #
def _poligonos_geo(mask: np.ndarray, fila: dict,
                   max_poligonos: int = 40) -> list[list[list[float]]]:
    """
    Contornos del resultado en lon/lat para dibujar en la estación.

    Se filtran componentes menores a ``MIN_AREA_PX`` y se recortan a
    ``max_poligonos``: una máscara ruidosa puede generar cientos de polígonos
    de 2 px que no aportan nada al overlay.
    """
    conts = MK.contornos(mask, min_area_px=MIN_AREA_PX)
    return [pixeles_a_geo(c, fila, mask.shape) for c in conts[:max_poligonos]]


def ejecutar(spec: dict, datos: DatosMision,
             region: list | np.ndarray | None = None) -> dict:
    """Ejecuta un spec soportado sobre las máscaras y la telemetría."""
    op = spec["operacion"]
    a_nombre = spec.get("a")
    por_frame: list[dict] = []
    acum = {"area_m2": 0.0, "count": 0, "len_a_m": 0.0, "len_ab_m": 0.0,
            "d_min": [], "d_media": [], "personas": 0}

    for fila in datos.filas:
        src = str(fila.get("src"))
        area_frame = datos.area_frame(src)

        if op == "personas":
            if region is not None and not region_toca_frame(region, fila):
                continue
            v = datos.personas(src)
            acum["personas"] += v
            por_frame.append({"src": src, "valor": v, "unidad": "personas",
                              "lat": _flotante(fila.get("lat")),
                              "lon": _flotante(fila.get("lon"))})
            continue

        a = datos.mascara_sujeto(src, a_nombre)
        if a is None:
            continue
        b = datos.mascara_sujeto(src, spec["b"]) if spec.get("b") else None
        if spec.get("b") and b is None:
            continue

        if region is not None:
            rp = poligono_a_pixeles(region, fila, a.shape)
            if rp is None:
                continue
            rm = MK.mascara_poligono(a.shape, rp)
            a = MK.interseccion(a, rm)
            if b is not None:
                b = MK.interseccion(b, rm)

        entrada = {"src": src, "area_frame_m2": round(area_frame, 2),
                   "lat": _flotante(fila.get("lat")), "lon": _flotante(fila.get("lon"))}

        if op == "area":
            objetivo = a if b is None else MK.interseccion(a, b)
            v = MK.area_m2(objetivo, area_frame)
            acum["area_m2"] += v
            entrada.update(valor=round(v, 2), unidad="m2",
                           poligonos=_poligonos_geo(objetivo, fila))

        elif op == "count":
            objetivo = a if b is None else MK.interseccion(a, b)
            if spec.get("c") and spec.get("buffer_m"):
                c = datos.mascara_sujeto(src, spec["c"])
                if c is None:
                    continue
                esc = MK.pixel_scale_m(area_frame, c.shape)
                if esc > 0:
                    objetivo = MK.interseccion(
                        objetivo, MK.buffer_mask(c, spec["buffer_m"] / esc))
            comps = MK.componentes(objetivo, min_area_px=MIN_AREA_PX,
                                   area_frame_m2=area_frame)
            acum["count"] += len(comps)
            entrada.update(valor=len(comps), unidad="componentes",
                           poligonos=_poligonos_geo(objetivo, fila),
                           areas_m2=[c.get("area_m2") for c in comps[:20]])

        elif op == "exists":
            objetivo = a if b is None else MK.interseccion(a, b)
            presencia = MK.area_px(objetivo) > 0
            if presencia:
                acum["count"] += 1
            entrada.update(valor=int(presencia), unidad="presencia",
                           poligonos=_poligonos_geo(objetivo, fila) if presencia else [])

        elif op == "length_fraction":
            interseccion = MK.interseccion(a, b)
            la, lab = MK.longitud_px(a), MK.longitud_px(interseccion)
            esc = MK.pixel_scale_m(area_frame, a.shape)
            acum["len_a_m"] += la * esc
            acum["len_ab_m"] += lab * esc
            entrada.update(valor=round(lab / la, 4) if la else 0.0,
                           unidad="fraccion",
                           longitud_a_m=round(la * esc, 2),
                           longitud_afectada_m=round(lab * esc, 2))

        elif op == "distance":
            if not a.any() or not b.any():
                continue
            d = MK.distancia_a(b)[a.astype(bool)]
            d = d[np.isfinite(d)]
            if d.size == 0:
                continue
            esc = MK.pixel_scale_m(area_frame, a.shape)
            dmin, dmed = float(d.min()) * esc, float(d.mean()) * esc
            acum["d_min"].append(dmin)
            acum["d_media"].append(dmed)
            entrada.update(valor=round(dmin, 2), unidad="m",
                           distancia_media_m=round(dmed, 2))

        por_frame.append(entrada)

    return _resultado(spec, acum, por_frame, region)


def _resultado(spec: dict, acum: dict, por_frame: list[dict],
               region) -> dict:
    op = spec["operacion"]
    if op == "area":
        total, unidades = round(acum["area_m2"], 2), "m2"
    elif op == "count":
        total, unidades = acum["count"], "componentes"
    elif op == "exists":
        total = {"frames_con_presencia": acum["count"], "de": len(por_frame)}
        unidades = "presencia"
    elif op == "length_fraction":
        total = {
            "fraccion": round(acum["len_ab_m"] / acum["len_a_m"], 4)
            if acum["len_a_m"] else 0.0,
            "longitud_a_m": round(acum["len_a_m"], 2),
            "longitud_afectada_m": round(acum["len_ab_m"], 2),
        }
        unidades = "fraccion"
    elif op == "distance":
        total = {
            "min_m": round(min(acum["d_min"]), 2) if acum["d_min"] else None,
            "media_m": round(float(np.mean(acum["d_media"])), 2)
            if acum["d_media"] else None,
        }
        unidades = "m"
    else:
        total, unidades = acum["personas"], "personas"

    return {
        "consulta": spec["texto"],
        "soportada": True,
        "plantilla": spec["plantilla"],
        "operacion": op,
        "sujetos": {"a": spec.get("a"), "b": spec.get("b"), "c": spec.get("c")},
        "buffer_m": spec.get("buffer_m"),
        "unidades": unidades,
        "total": total,
        "n_frames": len(por_frame),
        "por_frame": por_frame,
        "region_declarada": bool(region is not None),
        "georref": GEOREF_NOTA if region is not None else None,
        "limitaciones": LIMITACIONES,
    }


def responder(consulta: str, datos: DatosMision,
              region: list | np.ndarray | None = None) -> dict:
    """Parsea y ejecuta una consulta. Nunca lanza por consulta no soportada."""
    spec = parsear(consulta, datos.fuentes)
    if not spec.get("soportada"):
        return spec
    if region is not None and not datos.tiene_geo():
        return {
            "consulta": consulta,
            "soportada": False,
            "motivo": ("la telemetría no trae lat/lon por frame: no se puede "
                       "georreferenciar la zona dibujada"),
            "sugerencias": SUGERENCIAS,
        }
    return ejecutar(spec, datos, region)
