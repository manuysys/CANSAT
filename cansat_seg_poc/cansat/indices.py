"""
Índices ambientales y veredictos de misión — CanSat LB135.

FUENTE ÚNICA DE VERDAD. Antes de este módulo existían TRES definiciones
distintas de USI y NDVI (``analyze_stress.py``, ``validate_int8_mission.py`` y
el glosario del frontend), con umbrales de veredicto distintos. Todo lo que
calcule un índice ambiental o un veredicto DEBE importar de acá:

    mission_pipeline.py · post_flight.py · analyze_stress.py
    validate_int8_mission.py · adaptive_sampler.py · generate_report.py
    web-app/src/lib/vocab.ts (los tooltips se redactan desde estas docstrings)

────────────────────────────────────────────────────────────────────────────
NOMENCLATURA — cambio importante respecto de versiones anteriores
────────────────────────────────────────────────────────────────────────────
Lo que antes se llamaba "NDVI" NO es NDVI. El NDVI real se calcula con bandas
multiespectrales (NIR y roja):  (NIR - Rojo) / (NIR + Rojo).  Acá no hay NIR:
hay porcentajes de superficie salidos de un clasificador RGB. Por eso se
renombra a GVI (Greenness / Verdor Index), que es lo que realmente mide.

El nombre de columna en ``telemetry.csv`` sigue siendo ``ndvi`` para no romper
el contrato ya publicado; el campo canónico nuevo es ``gvi`` y ambos se
escriben con el mismo valor. Ver ``cansat/protocol.py`` y el README.

────────────────────────────────────────────────────────────────────────────
RANGOS REALES (leer antes de citar un número en el DPD)
────────────────────────────────────────────────────────────────────────────
USI  = bui / max(USI_VEG_FLOOR, veg), con ambos en % de superficie válida.
       NO está acotado a [0, 1]. Con veg ≈ 0 el piso del denominador lo lleva
       a cientos (bui=50 % → USI=500). El frontend normaliza con
       AdaptiveSampler usando USI_NORM=300.
       → usi_norm() sí devuelve [0, 1] y es la que hay que mostrar como
         "carga antrópica 0=natural · 1=todo construido".
GVI  = (veg - bare) / max(GVI_FLOOR, veg + bare)  → siempre en [-1, 1].
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# ── Orden canónico de clases (idéntico en train, inferencia y frontend) ── #
CLASS_NAMES: tuple[str, ...] = (
    "vegetation",
    "building",
    "water",
    "bare_ground",
    "other",
)
NUM_CLASSES: int = len(CLASS_NAMES)
IGNORE_INDEX: int = 255

# Claves cortas usadas en telemetry.csv y en el frontend.
CLASS_KEYS: tuple[str, ...] = ("veg", "bui", "wat", "bare", "oth")

# ── Constantes de los índices ──────────────────────────────────────────── #
USI_VEG_FLOOR: float = 0.1  # piso % de vegetación en el denominador de USI
USI_NORM: float = 300.0  # USI de saturación para usi_norm() → [0, 1]
GVI_FLOOR: float = 0.1  # piso % de (veg + bare) en el denominador de GVI
GVI_EXPUESTO: float = -0.5  # umbral de "suelo expuesto"
VEG_EXPUESTO: float = 10.0  # % de vegetación bajo el cual aplica el umbral anterior

# ── Umbrales de veredicto (USI sin normalizar) ─────────────────────────── #
USI_ALTO: float = 3.0
USI_MODERADO: float = 1.0
AGUA_EXTENSA_PCT: float = 25.0  # a partir de acá el agua domina el diagnóstico

# ── Vocabulario controlado ─────────────────────────────────────────────── #
VERDICTS: tuple[str, ...] = (
    "ZONA SALUDABLE",
    "ESTRÉS MODERADO",
    "ALTO ESTRÉS URBANO",
    "SUELO EXPUESTO",
    "SIN DATOS",
)
VERDICT_CODE: dict[str, int] = {
    "ZONA SALUDABLE": 0,
    "ESTRÉS MODERADO": 1,
    "ALTO ESTRÉS URBANO": 2,
    "SUELO EXPUESTO": 3,
    "SIN DATOS": 4,
}

# Fracción mínima de píxeles válidos para que el veredicto signifique algo.
# Con un frame entero sin datos todos los porcentajes dan 0.0 y el veredicto
# caía en "ZONA SALUDABLE", que es exactamente lo contrario de la verdad.
MIN_VALID_FRAC: float = 0.05

DIAGNOSTICS: tuple[str, ...] = (
    "SIN DESASTRE",
    "AGUA EXTENSA (lago/rio)",
    "POSIBLE INCENDIO/EROSION",
    "POSIBLE SISMO/VIENTO",
    "INUNDACION URBANA",
    "INUNDACION SEVERA",
)
# Severidad 0..4 — misma escala que DIAG_SEV en web-app/src/lib/vocab.ts.
DIAG_SEVERITY: dict[str, int] = {
    "SIN DESASTRE": 0,
    "AGUA EXTENSA (lago/rio)": 1,
    "POSIBLE INCENDIO/EROSION": 2,
    "POSIBLE SISMO/VIENTO": 3,
    "INUNDACION URBANA": 3,
    "INUNDACION SEVERA": 4,
}
ALERT_DIAGNOSTICS: tuple[str, ...] = (
    "INUNDACION URBANA",
    "INUNDACION SEVERA",
    "POSIBLE SISMO/VIENTO",
    "POSIBLE INCENDIO/EROSION",
)


# ══════════════════════════════════════════════════════════════════════ #
#  Normalización de porcentajes de terreno
# ══════════════════════════════════════════════════════════════════════ #
def pct_by_key(pcts: Sequence[float]) -> dict[str, float]:
    """[veg, bui, wat, bare, oth] → dict con claves cortas."""
    if len(pcts) != NUM_CLASSES:
        raise ValueError(f"se esperaban {NUM_CLASSES} porcentajes, llegaron {len(pcts)}")
    return dict(zip(CLASS_KEYS, (float(v) for v in pcts), strict=True))


def pct_by_name(pcts: Sequence[float]) -> dict[str, float]:
    """[veg, bui, wat, bare, oth] → dict con nombres largos de clase."""
    if len(pcts) != NUM_CLASSES:
        raise ValueError(f"se esperaban {NUM_CLASSES} porcentajes, llegaron {len(pcts)}")
    return dict(zip(CLASS_NAMES, (float(v) for v in pcts), strict=True))


def normalize_pcts(pcts: Iterable[float]) -> list[float]:
    """Reescala a 100 % si la suma se desvió (píxeles ignore, redondeos)."""
    v = [max(0.0, float(x)) for x in pcts]
    s = sum(v)
    if s <= 0.0:
        return [0.0] * NUM_CLASSES
    return [x / s * 100.0 for x in v]


# ══════════════════════════════════════════════════════════════════════ #
#  Índices
# ══════════════════════════════════════════════════════════════════════ #
def usi(pcts: Sequence[float]) -> float:
    """
    Índice de Estrés Urbano = edificios / vegetación.

    ⚠ NO está acotado a [0, 1]: es un cociente y puede valer cientos cuando
      hay superficie construida y casi nada de verde. Para mostrar "carga
      antrópica 0…1" usar :func:`usi_norm`.
    """
    p = pct_by_key(pcts)
    return p["bui"] / max(USI_VEG_FLOOR, p["veg"])


def usi_norm(pcts: Sequence[float]) -> float:
    """USI saturado a [0, 1] — la magnitud que el frontend puede mostrar como barra."""
    return min(1.0, max(0.0, usi(pcts) / USI_NORM))


def gvi(pcts: Sequence[float]) -> float:
    """
    Índice de verdor relativo = (vegetación − suelo expuesto) / (vegetación + suelo expuesto).

    Rango [-1, 1]. **No es NDVI** (no hay banda NIR); ver la nota del módulo.
    """
    p = pct_by_key(pcts)
    return (p["veg"] - p["bare"]) / max(GVI_FLOOR, p["veg"] + p["bare"])


# Alias de transición: el contrato publicado usa la columna "ndvi".
# Mantenemos el nombre viejo apuntando a la implementación correcta para que
# ningún consumidor existente se rompa, pero el nombre canónico es gvi().
ndvi = gvi


def density(pcts: Sequence[float]) -> float:
    """Densidad urbana: edificios sobre el total de superficie no vegetal (%)."""
    p = pct_by_key(pcts)
    return p["bui"] / max(USI_VEG_FLOOR, p["bui"] + p["oth"] + p["bare"]) * 100.0


def flood_risk(pcts: Sequence[float]) -> float:
    """
    Riesgo hídrico: agua ponderada por presencia de tejido construido.

    Versión anterior: ``wat * (bui/100) if bui > 20 else wat`` — tenía un salto
    de ~5× al cruzar bui=20 (wat=50 → 50 con bui=19, 10.5 con bui=21).
    Ahora es continuo: la ponderación entra progresivamente a partir de
    FLOOD_URBAN_BUI y llega a 1 en bui=50.
    """
    p = pct_by_key(pcts)
    ramp = min(1.0, max(0.0, (p["bui"] - FLOOD_URBAN_BUI) / FLOOD_URBAN_SPAN))
    return p["wat"] * (1.0 - ramp + ramp * (p["bui"] / 100.0))


FLOOD_URBAN_BUI: float = 10.0  # a partir de acá el agua empieza a ponderarse por urbano
FLOOD_URBAN_SPAN: float = 40.0  # y acá la ponderación ya es plena


def indices(pcts: Sequence[float]) -> dict[str, float]:
    """Todos los índices de una vez, con claves estables para serializar."""
    return {
        "usi": round(usi(pcts), 4),
        "usi_norm": round(usi_norm(pcts), 4),
        "gvi": round(gvi(pcts), 4),
        "density": round(density(pcts), 2),
        "flood_risk": round(flood_risk(pcts), 4),
    }


# ══════════════════════════════════════════════════════════════════════ #
#  Veredicto ambiental
# ══════════════════════════════════════════════════════════════════════ #
def verdict(pcts: Sequence[float], valid_frac: float = 1.0) -> str:
    """
    Veredicto ambiental de la zona sobrevolada.

    ⚠ CAMBIO DE COMPORTAMIENTO respecto de ``analyze_stress.analyze()``.
      El orden anterior era::

          if usi > 3:      ALTO ESTRÉS URBANO
          elif usi > 1:    ESTRÉS MODERADO
          elif gvi < -0.5 and veg < 10:  SUELO EXPUESTO
          else:            ZONA SALUDABLE

      Con ese orden la rama SUELO EXPUESTO era **prácticamente inalcanzable**:
      un terreno de tierra desnuda con un 2 % de edificios ya daba usi > 1 y
      caía en ESTRÉS MODERADO, aunque el 96 % de la superficie fuera suelo
      pelado. Ahora SUELO EXPUESTO se evalúa PRIMERO.

      Pero hace falta el guard que el código original sólo tenía en su
      comentario —"sin verde **y sin ciudad**: cemento/tierra desnuda"—: si la
      zona ES urbana, el veredicto correcto es ALTO ESTRÉS URBANO aunque no
      haya nada de verde. Sin ese guard, una manzana céntrica 100 % construida
      (bui=60, veg=0 → gvi=−1) saldría como "SUELO EXPUESTO".

      Por eso la condición pasa a ser::

          gvi < -0.5  AND  veg < 10  AND  usi <= USI_MODERADO  AND  wat < AGUA_EXTENSA

      · el guard de USI captura el suelo pelado rural/periurbano sin pisar la
        ciudad;
      · el guard de agua corrige un bug **preexistente**: el GVI no distingue
        agua de tierra desnuda (en un lago veg=0 y bare=0, así que da −1 por el
        piso del denominador), y un espejo de agua salía como "SUELO EXPUESTO".

      Los umbrales de USI no cambiaron (3.0 / 1.0), así que los veredictos de
      zonas efectivamente urbanas son idénticos a los históricos.
    """
    # 0) Sin superficie analizable no hay veredicto posible.
    if valid_frac < MIN_VALID_FRAC:
        return "SIN DATOS"

    p = pct_by_key(pcts)
    g = gvi(pcts)
    u = usi(pcts)

    # 1) Superficie pelada SIN ciudad y SIN agua: tierra/cemento desnudo.
    if (
        g < GVI_EXPUESTO
        and p["veg"] < VEG_EXPUESTO
        and u <= USI_MODERADO
        and p["wat"] < AGUA_EXTENSA_PCT
    ):
        return "SUELO EXPUESTO"
    # 2) Carga antrópica.
    if u > USI_ALTO:
        return "ALTO ESTRÉS URBANO"
    if u > USI_MODERADO:
        return "ESTRÉS MODERADO"
    return "ZONA SALUDABLE"


def verdict_code(v: str) -> int:
    """Código numérico del veredicto para el paquete de radio ($LB135)."""
    return VERDICT_CODE[v]


# ══════════════════════════════════════════════════════════════════════ #
#  Diagnóstico de desastre
# ══════════════════════════════════════════════════════════════════════ #
def diag_severity(diag: str | None) -> int:
    """Severidad 0..4 de un diagnóstico. Desconocido no vacío → 1 (conservador)."""
    if not diag:
        return 0
    return DIAG_SEVERITY.get(diag, 1)


def is_alert(diag: str | None) -> int:
    """1 si el diagnóstico amerita alerta de misión secundaria."""
    return 1 if diag in ALERT_DIAGNOSTICS else 0


def diagnose(
    pcts: Sequence[float],
    pct_dan: float | None,
    pct_dan2: float | None = None,
    pct_siam: float | None = None,
    pct_flood: float | None = None,
    pct_flood_water: float | None = None,
    flood_available: bool = False,
    pct_fire: float | None = None,
    pct_smoke: float | None = None,
    consensus_pct: float | None = None,
    consensus_pct_two_stage: float | None = None,
) -> tuple[str, int]:
    """
    Diagnóstico de desastre + flag de alerta. Devuelve ``(diag, alert)``.

    ── CAMBIO CENTRAL: el consenso ahora es un consenso de verdad ──────────
    Regla anterior (``mission_pipeline.py``)::

        votos = (pct_dan>10) + (pct_dan2>10) + (pct_siam>10)
        if pct_dan > 10 and votos >= 2:   ...

    ``pct_dan > 10`` aparecía dos veces (como requisito y como voto), así que
    el modelo principal tenía **veto**: si decía 9 % y los otros dos 80 %, no
    había alerta. Y si decía 11 % alcanzaba con que UNO de los otros coincidiera.
    La UI, mientras tanto, anunciaba "consenso de 3 modelos".

    ── UMBRALES POR MODELO (calibración 2026-09-17) ────────────────────────
    El two-stage trabaja enmascarado por edificios, así que sus porcentajes son
    ~3× más chicos que los del principal. Medido con
    ``tools/calibrate_thresholds.py`` sobre xBD con split por desastre:

      · principal: óptimo 11.8 % (F1 0.77, FPR 4.3 %)
      · two-stage: óptimo 0.04 %; umbral para FPR ≤5 % = 3.7 % (F1 0.58)

    Con un umbral ÚNICO de 10 % el two-stage casi nunca votaba y, con el
    siamés apagado (2 votantes), el consenso exigía 2/2 → la alerta no
    disparaba nunca. Por eso hay un umbral propio para el two-stage
    (``DAMAGE_CONSENSUS_PCT_TWO_STAGE``), calibrado, y el general sigue siendo
    configurable.

    Parámetros
    ----------
    pct_dan, pct_dan2, pct_siam : % de superficie dañada de cada modelo.
        Pasar ``None`` en los que no estén activos para que no voten.
    pct_flood, pct_flood_water : % de "inundación" y de "agua normal" del
        especialista FloodNet. ``None`` si el modelo no está cargado.
    flood_available : si el especialista está disponible para desempatar.
    consensus_pct : umbral de voto del principal y del siamés (%). ``None`` =
        ``DAMAGE_CONSENSUS_PCT`` (10). El pipeline lo expone como
        ``--damage-threshold``.
    consensus_pct_two_stage : umbral del two-stage (%). ``None`` =
        ``DAMAGE_CONSENSUS_PCT_TWO_STAGE`` (3.7, calibrado).
    """
    p = pct_by_key(pcts)
    agua, veg, bare, bui = p["wat"], p["veg"], p["bare"], p["bui"]

    umbral = DAMAGE_CONSENSUS_PCT if consensus_pct is None else float(consensus_pct)
    umbral_2 = (DAMAGE_CONSENSUS_PCT_TWO_STAGE if consensus_pct_two_stage is None
                else float(consensus_pct_two_stage))
    votos_modelos: list[float] = [v for v in (pct_dan, pct_dan2, pct_siam) if v is not None]
    votos = sum(1 for v, u in ((pct_dan, umbral), (pct_dan2, umbral_2),
                               (pct_siam, umbral))
                if v is not None and v > u)
    mayoria = (len(votos_modelos) // 2) + 1
    fuerte = sum(1 for v in votos_modelos if v > DAMAGE_STRONG_PCT)

    dan_max = max(votos_modelos) if votos_modelos else 0.0

    # 0) Fuego/humo (F3 2026-09-18): alerta independiente del daño estructural.
    #    Umbrales calibrados contra imágenes normales (LoveDA/RescueNet): el
    #    modelo da fuego 0.00 % y humo ≤15 % en escenas sin incendio, así que
    #    fuego >1 % ya es señal y humo >30 % es humo extenso.
    if pct_fire is not None and pct_fire > FIRE_ALERT_PCT:
        return "INCENDIO", 1
    if pct_smoke is not None and pct_smoke > SMOKE_ALERT_PCT:
        return "HUMO EXTENSO", 1

    # 1) Daño estructural por consenso.
    if votos >= mayoria:
        if agua > AGUA_EXTENSA_PCT:
            return "INUNDACION SEVERA", 1
        return "POSIBLE SISMO/VIENTO", 1
    # Red de seguridad: el principal se quedó abajo del umbral pero los otros
    # dos ven daño fuerte. Antes esto NO alertaba (veto del principal).
    if fuerte >= 2 and dan_max > umbral:
        return "POSIBLE SISMO/VIENTO", 1

    # 2) Agua extensa: ¿inundación o lago/río?
    if agua > AGUA_EXTENSA_PCT:
        if flood_available and pct_flood is not None:
            ref = pct_flood_water if pct_flood_water is not None else 0.0
            if pct_flood > FLOOD_MIN_PCT and pct_flood > ref * FLOOD_RATIO:
                return (
                    "INUNDACION URBANA" if bui > FLOOD_URBAN_MIN_BUI else "INUNDACION SEVERA"
                ), 1
        return "AGUA EXTENSA (lago/rio)", 0

    # 3) Fuego / erosión.
    if veg < FIRE_MAX_VEG and bare > FIRE_MIN_BARE and dan_max > FIRE_MIN_DAN:
        return "POSIBLE INCENDIO/EROSION", 1

    return "SIN DESASTRE", 0


DAMAGE_CONSENSUS_PCT: float = 10.0  # umbral de voto del principal y del siamés
# Umbral propio del two-stage: trabaja enmascarado por edificios, así que sus
# porcentajes son ~3× menores que los del principal.
#   · 2026-09-17: calibrado en xBD val por desastre → 3.7 % (FPR ≤ 5 %).
#   · 2026-09-18: el modelo de vuelo pasó a ser el two-stage adaptado a UAV
#     (RescueNet, ver MODELS.yaml dano_f2_rescuenet) y se recalibró en el
#     dominio de vuelo (RescueNet val, 600 tiles, etiqueta ≥10 % de píxeles
#     dañados): F1-óptimo 10.18 % (F1 0.828, recall 0.887, FPR 41.5 % sobre
#     val de zona de desastre). Se prioriza DETECCIÓN: 10.2 %.
#     Ojo: el principal xBD satura en UAV (mediana 62.7 % de "daño" en tiles
#     sin daño) — en vuelo el que discrimina es este two-stage.
DAMAGE_CONSENSUS_PCT_TWO_STAGE: float = 10.2
DAMAGE_STRONG_PCT: float = 25.0  # "daño fuerte" para la red de seguridad
FLOOD_MIN_PCT: float = 20.0
FLOOD_RATIO: float = 1.5
FLOOD_URBAN_MIN_BUI: float = 5.0
FIRE_MAX_VEG: float = 5.0
FIRE_MIN_BARE: float = 40.0
FIRE_MIN_DAN: float = 5.0
# Fuego/humo (modelo cansat_fire_smoke, F3 2026-09-18). Medido en escenas
# normales: fuego 0.00 % (LoveDA urbano/rural, RescueNet) y humo ≤15 % en
# rural; los umbrales dejan margen.
FIRE_ALERT_PCT: float = 1.0
SMOKE_ALERT_PCT: float = 30.0


# ══════════════════════════════════════════════════════════════════════ #
#  Paquete completo (lo que consumen pipeline y post-vuelo)
# ══════════════════════════════════════════════════════════════════════ #
def environment(
    pcts: Sequence[float],
    n_green_patches: int = 0,
    valid_frac: float = 1.0,
    haze_pct: float | None = None,
    humidex: float | None = None,
    exg_pct: float | None = None,
    shadow_pct: float | None = None,
    fire_pct: float | None = None,
) -> dict[str, object]:
    """
    Diccionario ambiental completo, con las mismas claves que devolvía
    ``analyze_stress.analyze()`` para no romper consumidores existentes.

    ``pcts`` son porcentajes ya calculados sobre píxeles válidos.
    ``haze_pct`` (bruma por dark channel, ``cansat/stress.py``) y ``humidex``
    (calor con T y humedad del sensor) agregan el bloque de estrés ambiental
    por contaminación que pide el DPD; si son ``None`` no aparecen.
    """
    from . import stress as ST

    p = pct_by_key(pcts)
    by_name = pct_by_name(pcts)
    v = verdict(pcts, valid_frac=valid_frac)
    veg_frac = p["veg"] / max(1e-9, sum(pcts)) if sum(pcts) > 0 else 0.0
    out: dict[str, object] = {
        "pcts": by_name,
        "usi": usi(pcts),
        "usi_norm": usi_norm(pcts),
        "ndvi": gvi(pcts),  # clave legacy del contrato
        "gvi": gvi(pcts),  # clave canónica
        "density": density(pcts),
        "n_green_patches": int(n_green_patches),
        "frag_per_patch": (p["veg"] / n_green_patches) if n_green_patches else 0.0,
        "flood_risk": flood_risk(pcts),
        "valid_frac": round(float(valid_frac), 5),
        "verdict": v,
        "vcode": verdict_code(v),
        "_veg_frac": veg_frac,
    }
    if haze_pct is not None:
        out["haze_pct"] = round(float(haze_pct), 2)
        out["contam"] = ST.contam_verdict(float(haze_pct))
    if humidex is not None:
        out["humidex"] = round(float(humidex), 1)
        out["heat"] = ST.heat_verdict(float(humidex))
    if exg_pct is not None:
        out["veg_exg_pct"] = round(float(exg_pct), 2)
    if shadow_pct is not None:
        out["shadow_pct"] = round(float(shadow_pct), 2)
    if (haze_pct is not None or humidex is not None or fire_pct is not None):
        out["stress_idx"] = ST.stress_score(
            haze_pct if haze_pct is not None else 0.0,
            humidex if humidex is not None else 25.0,
            usi_norm(pcts),
            fire_pct=fire_pct if fire_pct is not None else 0.0)
    return out
