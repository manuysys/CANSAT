"""
Tests de ``cansat.casualties`` — estimación de pérdidas humanas (DPD).

Lo importante que fijan estos tests: el modelo es lineal y auditable, los
supuestos se respetan, y los casos borde (sin daño, sin área, densidad 0) no
producen NaN ni números absurdos.
"""
import pytest

from cansat.casualties import (Supuestos, estimar, estimar_mision,
                               BANDA_INCERTIDUMBRE, ocupacion_por_hora)


def test_sin_dano_no_hay_perdidas():
    e = estimar(danado_pct=0.0, area_m2=20_000)
    assert e.personas_afectadas == 0.0
    assert e.perdidas_estimadas == 0.0
    assert e.personas_expuestas > 0          # hay gente expuesta aunque no daño


def test_formula_lineal_documentada():
    """personas_afectadas = densidad · área_km² · ocupación · fracción."""
    s = Supuestos(pop_density=1000, occupancy=1.0, collapse_frac=1.0, fatality=1.0)
    e = estimar(danado_pct=50.0, area_m2=2_000_000, supuestos=s)   # 2 km²
    # expuestas = 1000 · 2 · 1 = 2000 ; afectadas = 50 % = 1000 ; perdidas = 1000
    assert e.personas_expuestas == pytest.approx(2000.0)
    assert e.personas_afectadas == pytest.approx(1000.0)
    assert e.perdidas_estimadas == pytest.approx(1000.0)
    assert e.area_danada_m2 == pytest.approx(1_000_000.0)


def test_banda_de_incertidumbre():
    e = estimar(danado_pct=40.0, area_m2=100_000)
    lo, hi = BANDA_INCERTIDUMBRE
    assert e.perdidas_min == pytest.approx(e.perdidas_estimadas * lo, rel=1e-6)
    assert e.perdidas_max == pytest.approx(e.perdidas_estimadas * hi, rel=1e-6)


def test_colapso_medido_reemplaza_el_supuesto():
    """F2b: con severidad disponible, la fracción de colapso es medida."""
    s = Supuestos()
    e_sup = estimar(danado_pct=50.0, area_m2=10_000.0, supuestos=s)
    e_med = estimar(danado_pct=50.0, area_m2=10_000.0, supuestos=s,
                    collapse_frac_medido=0.6)
    assert e_sup.colapso_fuente == "supuesto"
    assert e_sup.colapso_usado == pytest.approx(s.collapse_frac)
    assert e_med.colapso_fuente == "medido"
    assert e_med.colapso_usado == pytest.approx(0.6)
    # 0.6 / 0.3 = 2 → el doble de pérdidas con el colapso medido (con redondeo)
    assert e_med.perdidas_estimadas == pytest.approx(
        e_sup.perdidas_estimadas * 2.0, abs=0.02)


def test_colapso_medido_se_acota_a_0_1():
    e = estimar(danado_pct=50.0, area_m2=10_000.0, collapse_frac_medido=1.7)
    assert e.colapso_usado == pytest.approx(1.0)
    e2 = estimar(danado_pct=50.0, area_m2=10_000.0, collapse_frac_medido=-0.2)
    assert e2.colapso_usado == pytest.approx(0.0)


def test_ocupacion_por_hora_franjas_pager():
    """PAGER: de noche hay más gente presente (los sismos nocturnos matan más)."""
    occ_dia, f_dia = ocupacion_por_hora(12.0)
    occ_noche, f_noche = ocupacion_por_hora(23.0)
    occ_trans, f_trans = ocupacion_por_hora(7.0)
    assert f_dia == "dia" and f_noche == "noche" and f_trans == "transito"
    assert occ_noche > occ_trans > occ_dia


def test_vulnerabilidad_escala_el_colapso():
    s = Supuestos(vulnerabilidad=2.0)
    e = estimar(50.0, 10_000.0, s, collapse_frac_medido=0.3)
    assert e.colapso_usado == pytest.approx(0.6)      # 0.3 × 2.0
    assert e.supuestos.vulnerabilidad == 2.0
    assert e.perdidas_estimadas > estimar(50.0, 10_000.0,
                                          collapse_frac_medido=0.3).perdidas_estimadas


def test_ocupacion_fuente_se_registra_en_el_resultado():
    e = estimar(10.0, 1000.0, ocupacion_fuente="noche")
    assert e.ocupacion_fuente == "noche"
    assert e.ocupacion_usada == pytest.approx(0.6)    # default del dataclass


def test_area_cero_no_divide_ni_explota():
    e = estimar(danado_pct=80.0, area_m2=0.0)
    assert e.personas_afectadas == 0.0 and e.perdidas_estimadas == 0.0


def test_densidad_cero_es_estimacion_valida():
    """Sin dato de población, el modelo reporta 0 afectados (no inventa gente)."""
    e = estimar(danado_pct=90.0, area_m2=50_000, supuestos=Supuestos(pop_density=0))
    assert e.personas_afectadas == 0.0
    assert e.area_danada_m2 > 0


def test_danado_pct_se_acota_a_0_100():
    e1 = estimar(danado_pct=150.0, area_m2=10_000)
    e2 = estimar(danado_pct=100.0, area_m2=10_000)
    assert e1.personas_afectadas == e2.personas_afectadas
    e3 = estimar(danado_pct=-5.0, area_m2=10_000)
    assert e3.personas_afectadas == 0.0


def test_supuestos_invalidos_lanzan():
    with pytest.raises(ValueError):
        Supuestos(occupancy=1.5)
    with pytest.raises(ValueError):
        Supuestos(fatality=-0.1)
    with pytest.raises(ValueError):
        Supuestos(pop_density=-1)


def test_estimar_mision_suma_frames_y_omite_sin_dato():
    frames = [
        {"danado_max_pct": 40.0, "aff_m2_area": 100_000},   # dañado
        {"danado_max_pct": None, "aff_m2_area": 100_000},   # modelos apagados
        {"danado_max_pct": 0.0, "aff_m2_area": 200_000},    # sin daño
        {"danado_max_pct": 20.0, "aff_m2_area": 0},         # sin área
    ]
    r = estimar_mision(frames)
    assert r["n_frames_con_dano"] == 2      # el nulo y el sin-área no cuentan
    assert r["perdidas_estimadas"] > 0
    assert r["perdidas_min"] < r["perdidas_estimadas"] < r["perdidas_max"]
    assert "EXPOSICIÓN" in r["nota"]


def test_estimar_mision_vacia():
    r = estimar_mision([])
    assert r["perdidas_estimadas"] == 0.0
    assert r["n_frames_con_dano"] == 0
