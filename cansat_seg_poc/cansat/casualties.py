"""
Estimación de pérdidas humanas a partir del daño observado — CanSat LB135.

════════════════════════════════════════════════════════════════════════════
QUÉ ES Y QUÉ NO ES (leer antes de citar un número en el DPD o ante el jurado)
════════════════════════════════════════════════════════════════════════════
El DPD pide *"detectar los daños materiales y, según su magnitud, estimar las
posibles pérdidas humanas resultantes"*. Este módulo hace exactamente eso, con
un modelo **explícito, auditable y de orden de magnitud**:

    personas_afectadas = densidad_poblacional × área_relevada × ocupación
                         × fracción_de_área_dañada

    pérdidas_estimadas = personas_afectadas × fracción_de_colapso
                         × letalidad_en_colapso

NO es una predicción de víctimas. Es una **estimación de exposición**: cuánta
gente vive (u ocupa) el área que el vuelo vio dañada, multiplicada por dos
factores de vulnerabilidad que se declaran por parámetro. Cualquier sistema
real (PAGER/USGS, GRADE del Banco Mundial) usa modelos de fragilidad por tipo
estructural, suelo, hora y calidad constructiva: acá no hay nada de eso, y el
informe debe decirlo.

SUPUESTOS POR DEFECTO (todos configurables por CLI y registrados en el JSONL)
  · densidad_poblacional = 1500 hab/km²  → urbano periurbano típico.
    Referencia de magnitud: CABA ~15 000, GBA ~5 000, localidades del interior
    ~1 000-3 000. Ajustar con el dato del predio (INDEC / WorldPop).
  · ocupación = 0.6 → fracción de la población que está en el área en el
    momento del vuelo (0.6 = mayoría en horario diurno).
  · fracción_de_colapso = 0.3 → de lo dañado, cuánto colapsó. El modelo de
    daño del vuelo tiene una sola clase "dañado" (no distingue colapso); 0.3
    es un valor conservador documentado para sismos/vientos (xBD distingue
    destroyed/major/minor y en eventos severos ~25-40 % de lo dañado colapsa).
  · letalidad_en_colapso = 0.1 → muertes entre los ocupantes de estructuras
    colapsadas. Órdenes de magnitud reportados: 1-10 % (depende de hora,
    tipo, rescate). Se usa 0.1 % NO: es 10 %.

RANGO: el resultado se publica con una banda [×0.5, ×2] porque los tres
últimos factores son incertidumbre pura, no medición.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# ── Supuestos por defecto (documentados arriba) ────────────────────────── #
POP_DENSITY_DEFAULT: float = 1500.0   # hab/km²
OCCUPANCY_DEFAULT: float = 0.6        # fracción presente
COLLAPSE_FRAC_DEFAULT: float = 0.3    # de lo dañado, fracción colapsada
FATALITY_DEFAULT: float = 0.1         # letalidad entre ocupantes de colapso
BANDA_INCERTIDUMBRE: tuple[float, float] = (0.5, 2.0)


@dataclass
class Supuestos:
    """Parámetros del modelo de estimación. Se serializan en la telemetría."""

    pop_density: float = POP_DENSITY_DEFAULT
    occupancy: float = OCCUPANCY_DEFAULT
    collapse_frac: float = COLLAPSE_FRAC_DEFAULT
    fatality: float = FATALITY_DEFAULT

    def __post_init__(self) -> None:
        if self.pop_density < 0:
            raise ValueError("pop_density no puede ser negativa")
        for nombre, v in (("occupancy", self.occupancy),
                          ("collapse_frac", self.collapse_frac),
                          ("fatality", self.fatality)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{nombre} debe estar en [0, 1], llegó {v}")


@dataclass
class EstimacionPerdidas:
    """Resultado de la estimación para UN frame o para la misión completa."""

    area_relevada_m2: float          # huella en tierra considerada
    area_danada_m2: float            # área × fracción dañada
    personas_expuestas: float        # densidad × área × ocupación
    personas_afectadas: float        # expuestas × fracción dañada
    perdidas_estimadas: float        # afectadas × colapso × letalidad
    perdidas_min: float              # banda inferior
    perdidas_max: float              # banda superior
    supuestos: Supuestos

    def to_dict(self) -> dict:
        d = asdict(self)
        d["supuestos"] = asdict(self.supuestos)
        return d


def _banda(perdidas: float) -> tuple[float, float]:
    lo, hi = BANDA_INCERTIDUMBRE
    return perdidas * lo, perdidas * hi


def estimar(
    danado_pct: float,
    area_m2: float,
    supuestos: Supuestos | None = None,
) -> EstimacionPerdidas:
    """
    Estimación para un frame.

    ``danado_pct`` es el % de superficie dañada (el ``danado_max_pct`` del
    consenso, ya normalizado por píxeles válidos). ``area_m2`` es la huella en
    tierra del frame (``mission_pipeline.ground_area_m2``).
    """
    s = supuestos or Supuestos()
    danado_pct = max(0.0, min(100.0, float(danado_pct)))
    area_m2 = max(0.0, float(area_m2))

    area_km2 = area_m2 / 1e6
    frac = danado_pct / 100.0

    expuestas = s.pop_density * area_km2 * s.occupancy
    afectadas = expuestas * frac
    perdidas = afectadas * s.collapse_frac * s.fatality
    lo, hi = _banda(perdidas)
    return EstimacionPerdidas(
        area_relevada_m2=round(area_m2, 1),
        area_danada_m2=round(area_m2 * frac, 1),
        personas_expuestas=round(expuestas, 1),
        personas_afectadas=round(afectadas, 1),
        perdidas_estimadas=round(perdidas, 2),
        perdidas_min=round(lo, 2),
        perdidas_max=round(hi, 2),
        supuestos=s,
    )


def estimar_mision(
    frames: list[dict],
    supuestos: Supuestos | None = None,
) -> dict:
    """
    Agrega la estimación de la misión sin doble conteo.

    Cada frame aporta su propia área (la huella crece al bajar) y su fracción
    dañada, así que se suman las personas afectadas/expuestas por frame. Los
    frames con ``danado_max_pct`` nulo (modelos apagados) o sin área no suman.

    Devuelve un dict con totales y el detalle de supuestos.
    """
    s = supuestos or Supuestos()
    total = EstimacionPerdidas(0, 0, 0, 0, 0, 0, 0, s)
    n_frames = 0
    for f in frames:
        dan = f.get("danado_max_pct", f.get("danado_pct"))
        if dan is None:
            continue
        area = f.get("aff_m2_area") or f.get("area_m2") or 0.0
        if not area:
            continue
        e = estimar(float(dan), float(area), s)
        total.area_relevada_m2 += e.area_relevada_m2
        total.area_danada_m2 += e.area_danada_m2
        total.personas_expuestas += e.personas_expuestas
        total.personas_afectadas += e.personas_afectadas
        total.perdidas_estimadas += e.perdidas_estimadas
        n_frames += 1
    lo, hi = _banda(total.perdidas_estimadas)
    total.perdidas_min, total.perdidas_max = lo, hi
    out = total.to_dict()
    out["n_frames_con_dano"] = n_frames
    out["nota"] = (
        "Estimación de EXPOSICIÓN con supuestos declarados, no una predicción "
        "de víctimas. La banda refleja la incertidumbre de los factores de "
        "vulnerabilidad."
    )
    return out
