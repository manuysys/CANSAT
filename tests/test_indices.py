"""
Tests de ``cansat.indices`` — la fuente única de USI/GVI/veredicto/diagnóstico.

Estos son los tests que NO existían y que habrían atrapado, antes del vuelo,
los tres problemas más graves del proyecto:
  · tres definiciones distintas de USI y NDVI con umbrales incompatibles;
  · un veredicto "SUELO EXPUESTO" inalcanzable por el orden de las ramas;
  · un "consenso de 3 modelos" donde el modelo principal tenía veto.
"""
import pytest

from cansat import indices as I


# ── USI ──────────────────────────────────────────────────────────────────── #
def test_usi_es_cociente_edificios_vegetacion():
    # bui=50, veg=25 → 2.0
    assert I.usi([25, 50, 0, 25, 0]) == pytest.approx(2.0)


def test_usi_no_esta_acotado_a_0_1():
    """
    DOCUMENTA el comportamiento real: USI puede valer cientos.

    El tooltip del frontend decía "0 = natural, 1 = todo construido", lo cual
    era falso. Este test fija la verdad para que nadie vuelva a asumirlo.
    """
    u = I.usi([0.0, 50.0, 0.0, 50.0, 0.0])
    assert u == pytest.approx(500.0)      # 50 / max(0.1, 0) = 500
    assert u > 1.0


def test_usi_norm_si_esta_acotado():
    for p in ([0, 100, 0, 0, 0], [50, 50, 0, 0, 0], [90, 1, 5, 3, 1]):
        assert 0.0 <= I.usi_norm(p) <= 1.0


def test_usi_con_vegetacion_cero_no_divide_por_cero():
    assert I.usi([0, 10, 0, 90, 0]) == pytest.approx(100.0)   # 10 / 0.1


# ── GVI (antes mal llamado NDVI) ────────────────────────────────────────── #
def test_gvi_acotado_entre_menos_1_y_1():
    for p in ([100, 0, 0, 0, 0], [0, 0, 0, 100, 0], [50, 0, 0, 50, 0]):
        assert -1.0 <= I.gvi(p) <= 1.0


def test_gvi_positivo_con_mas_vegetacion_que_suelo():
    assert I.gvi([60, 0, 0, 20, 20]) > 0


def test_limites_de_veredicto_por_usi():
    """Fija los umbrales exactos para que nadie los mueva sin que salte un test."""
    # USI justo en 1.0 → no es MODERADO (la condición es >)
    assert I.verdict([20.0, 20.0, 0.0, 40.0, 20.0]) == "ZONA SALUDABLE"
    # USI apenas arriba de 1.0 → MODERADO
    assert I.verdict([19.0, 20.0, 0.0, 41.0, 20.0]) == "ESTRÉS MODERADO"
    # USI apenas arriba de 3.0 → ALTO
    assert I.verdict([10.0, 31.0, 0.0, 40.0, 19.0]) == "ALTO ESTRÉS URBANO"


def test_gvi_negativo_con_mas_suelo_que_vegetacion():
    assert I.gvi([5, 0, 0, 60, 35]) < 0


def test_gvi_no_distingue_agua_de_suelo_desnudo():
    """
    LIMITACIÓN CONOCIDA y documentada: en ambos casos veg=0, así que GVI da ~-1.

    Por eso el veredicto SUELO EXPUESTO exige además agua < 25 %. Sin ese guard
    un lago salía como "SUELO EXPUESTO" (bug preexistente detectado en la
    auditoría).
    """
    assert I.gvi([0, 0, 90, 5, 5]) == pytest.approx(I.gvi([0, 0, 0, 90, 10]), abs=0.05)


def test_ndvi_es_alias_de_gvi():
    """El contrato publicado usa la columna 'ndvi'; debe seguir funcionando."""
    p = [30, 12, 5, 40, 13]
    assert I.ndvi(p) == I.gvi(p)


# ── Veredictos ──────────────────────────────────────────────────────────── #
CAMPO = [84.77, 0.0, 7.71, 7.52, 0.0]
CIUDAD = [0.0, 60.0, 0.0, 20.0, 20.0]
# USI = bui/veg = 18/12 = 1.5 → cae entre MODERADO(1.0) y ALTO(3.0).
# (El primer valor que puse, [35,12,...], daba USI 0.34 y el test fallaba: el
#  caso de prueba estaba mal calculado, no la función.)
SUBURBIO = [12.0, 18.0, 5.0, 45.0, 20.0]
PELADO_RURAL = [4.0, 0.4, 0.0, 90.0, 5.6]
LAGO = [0.0, 0.0, 90.0, 5.0, 5.0]


def test_campo_verde_es_zona_saludable():
    assert I.verdict(CAMPO) == "ZONA SALUDABLE"


def test_ciudad_densa_es_alto_estres_aunque_no_tenga_verde():
    """
    REGRESIÓN que introduje y corregí durante la auditoría: al mover
    SUELO EXPUESTO al primer lugar, una manzana 100 % construida (veg=0 →
    gvi=-1) caía en "SUELO EXPUESTO". El guard `usi <= USI_MODERADO` lo impide.
    """
    assert I.verdict(CIUDAD) == "ALTO ESTRÉS URBANO"


def test_suburbio_es_estres_moderado():
    assert I.verdict(SUBURBIO) == "ESTRÉS MODERADO"


def test_suelo_pelado_rural_es_suelo_expuesto():
    """
    ANTES ERA INALCANZABLE. El orden viejo evaluaba USI primero, y un terreno
    pelado con 2 % de edificios ya daba USI > 1 → caía en ESTRÉS MODERADO.
    """
    assert I.verdict(PELADO_RURAL) == "SUELO EXPUESTO"


def test_lago_no_es_suelo_expuesto():
    """BUG PREEXISTENTE: GVI no distingue agua de tierra, y un lago salía pelado."""
    assert I.verdict(LAGO) != "SUELO EXPUESTO"


def test_frame_sin_datos_no_es_zona_saludable():
    """
    BUG PREEXISTENTE: con 0 píxeles válidos todos los porcentajes dan 0.0 y el
    veredicto caía en "ZONA SALUDABLE", que es lo contrario de la verdad.
    """
    assert I.verdict([0.0] * 5, valid_frac=0.0) == "SIN DATOS"
    assert I.verdict([0.0] * 5, valid_frac=0.01) == "SIN DATOS"


def test_todo_veredicto_es_conocido_y_tiene_codigo():
    for p in (CAMPO, CIUDAD, SUBURBIO, PELADO_RURAL, LAGO, [0.0] * 5):
        v = I.verdict(p)
        assert v in I.VERDICTS
        assert v in I.VERDICT_CODE


# ── Riesgo hídrico ──────────────────────────────────────────────────────── #
def test_flood_risk_es_continuo():
    """
    BUG PREEXISTENTE: ``wat * (bui/100) if bui > 20 else wat`` saltaba ~5× al
    cruzar bui=20 (wat=50 → 50 con bui=19, pero 10.5 con bui=21).
    """
    a = I.flood_risk([0, 19.0, 50, 0, 31])
    b = I.flood_risk([0, 21.0, 50, 0, 29])
    assert abs(a - b) < 10.0, f"salto discontinuo: {a} vs {b}"


def test_flood_risk_monotonico_en_agua():
    r = [I.flood_risk([0, 25, w, 0, 75 - w]) for w in (10, 30, 50, 70)]
    assert r == sorted(r)


# ── Diagnóstico / consenso de daño ──────────────────────────────────────── #
TERRENO = [10.0, 10.0, 20.0, 30.0, 30.0]


def test_sin_dano_no_hay_alerta():
    diag, alert = I.diagnose(TERRENO, 0.0, 0.0, 0.0)
    assert diag == "SIN DESASTRE"
    assert alert == 0


def test_mayoria_de_modelos_genera_alerta():
    diag, alert = I.diagnose(TERRENO, 30.0, 30.0, 0.0)
    assert alert == 1
    assert diag in ("POSIBLE SISMO/VIENTO", "INUNDACION SEVERA")


def test_el_modelo_principal_ya_no_tiene_veto():
    """
    BUG CENTRAL. La regla vieja era::

        votos = (pct_dan>10) + (pct_dan2>10) + (pct_siam>10)
        if pct_dan > 10 and votos >= 2: ...

    Como ``pct_dan > 10`` aparecía dos veces (requisito y voto), si el principal
    decía 9 % y los otros dos 80 %, NO había alerta. Y la UI anunciaba
    "consenso de 3 modelos".
    """
    _diag, alert = I.diagnose(TERRENO, pct_dan=9.0, pct_dan2=80.0, pct_siam=80.0)
    assert alert == 1, "dos de tres modelos ven daño fuerte: debe alertar"


def test_un_solo_modelo_no_alcanza():
    _diag, alert = I.diagnose(TERRENO, pct_dan=60.0, pct_dan2=0.0, pct_siam=0.0)
    assert alert == 0, "un solo voto de tres no es consenso"


def test_agua_extensa_sin_dano_no_es_inundacion():
    """Un lago no es una inundación: es la distinción clave del proyecto."""
    diag, alert = I.diagnose([5.0, 10.0, 40.0, 25.0, 20.0], 0.0, 0.0, 0.0)
    assert diag == "AGUA EXTENSA (lago/rio)"
    assert alert == 0


def test_agua_extensa_con_dano_es_inundacion_severa():
    diag, alert = I.diagnose([5.0, 10.0, 40.0, 25.0, 20.0], 30.0, 30.0, 0.0)
    assert diag == "INUNDACION SEVERA"
    assert alert == 1


def test_flood_specialist_distingue_inundacion_de_lago():
    pcts = [5.0, 10.0, 40.0, 25.0, 20.0]
    # Especialista dice "inundación" claramente por encima del agua normal.
    diag, _ = I.diagnose(pcts, 0.0, 0.0, 0.0,
                         pct_flood=35.0, pct_flood_water=8.0, flood_available=True)
    assert "INUNDACION" in diag
    # Especialista dice que es agua normal.
    diag2, _ = I.diagnose(pcts, 0.0, 0.0, 0.0,
                          pct_flood=5.0, pct_flood_water=40.0, flood_available=True)
    assert diag2 == "AGUA EXTENSA (lago/rio)"


def test_fuego_o_erosion():
    diag, alert = I.diagnose([3.0, 2.0, 2.0, 60.0, 33.0], 8.0, 4.0, 2.0)
    assert diag == "POSIBLE INCENDIO/EROSION"
    assert alert == 1


def test_modelos_ausentes_no_votan():
    """Con sólo el principal disponible, mayoria = 1 y su voto basta."""
    _diag, alert = I.diagnose(TERRENO, pct_dan=40.0, pct_dan2=None, pct_siam=None)
    assert alert == 1


def test_todo_diagnostico_es_conocido():
    for args in [(0, 0, 0), (40, 40, 40), (40, 0, 0)]:
        diag, _ = I.diagnose(TERRENO, *args)
        assert diag in I.DIAGNOSTICS
        assert diag in I.DIAG_SEVERITY


def test_severidad_ordena_los_diagnosticos():
    assert I.diag_severity("SIN DESASTRE") == 0
    assert I.diag_severity("AGUA EXTENSA (lago/rio)") == 1
    assert I.diag_severity("INUNDACION SEVERA") == 4
    assert I.diag_severity("INUNDACION SEVERA") > I.diag_severity("POSIBLE SISMO/VIENTO")


def test_diag_desconocido_no_es_cero():
    """Conservador: un diagnóstico nuevo arranca en severidad 1, no en 0."""
    assert I.diag_severity("ALGO NUEVO") == 1
    assert I.diag_severity(None) == 0


# ── Normalización y utilidades ──────────────────────────────────────────── #
def test_pct_by_key_rechaza_largo_incorrecto():
    with pytest.raises(ValueError):
        I.pct_by_key([1, 2, 3])


def test_normalize_pcts_reescala_a_100():
    out = I.normalize_pcts([10, 10, 10, 10, 10])
    assert sum(out) == pytest.approx(100.0)
    assert out == [20.0] * 5


def test_normalize_pcts_con_todo_cero_no_divide_por_cero():
    assert I.normalize_pcts([0, 0, 0, 0, 0]) == [0.0] * 5


def test_environment_devuelve_las_claves_del_contrato_legacy():
    """Los consumidores viejos esperan estas claves exactas."""
    env = I.environment(CAMPO)
    for k in ("pcts", "usi", "ndvi", "density", "n_green_patches",
              "frag_per_patch", "flood_risk", "verdict"):
        assert k in env, f"falta la clave legacy '{k}'"
    # y las nuevas
    for k in ("gvi", "usi_norm", "vcode", "valid_frac"):
        assert k in env, f"falta la clave canónica '{k}'"


def test_environment_ndvi_y_gvi_coinciden():
    env = I.environment(SUBURBIO)
    assert env["ndvi"] == env["gvi"]
