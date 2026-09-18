"""
Tests del muestreo adaptativo — calibración de la incertidumbre.

Cubren el hallazgo de la auditoría: ``uncert_max=0.03`` y ``uncert/0.06``
estaban calibrados para la varianza cruda de TTA (0.001-0.01), pero el camino
de vuelo usa entropía normalizada (0.3-0.7). Resultado: ``unc=1.0`` siempre y
``review=True`` casi siempre — el sampler mandaba todo a HIGH.
"""
import numpy as np
import pytest

from adaptive_sampler import AdaptiveSampler


def test_incertidumbre_calibrada_no_satura_en_entropia_tipica():
    """Entropía normalizada típica (0.3) no debe valer como 1.0 en el score."""
    s = AdaptiveSampler()
    base = s.interest_score([50, 10, 10, 15, 15], usi=1.0, uncert=0.0)
    con_unc = s.interest_score([50, 10, 10, 15, 15], usi=1.0, uncert=0.3)
    aporte = con_unc - base
    # 0.3/0.5 * w4 = 0.6 * 0.10 = 0.06, no el w4 completo (0.10).
    assert aporte == pytest.approx(0.06, abs=0.005)
    assert aporte < s.w[3]


def test_review_solo_con_incertidumbre_alta():
    s = AdaptiveSampler()
    assert s.decide(0.0, uncert=0.2)["review"] is False
    assert s.decide(0.0, uncert=0.2)["priority"] == "LOW"
    d = s.decide(0.0, uncert=0.6)
    assert d["review"] is True
    assert d["priority"] == "HIGH" and d["action"] == "high_res"


def test_uncert_max_es_el_umbral_de_review():
    s = AdaptiveSampler(uncert_max=0.25)
    assert s.decide(0.0, uncert=0.24)["review"] is False
    assert s.decide(0.0, uncert=0.26)["review"] is True


def test_score_satura_en_el_umbral_de_review():
    """El término de incertidumbre del score no puede saturar antes del review."""
    s = AdaptiveSampler()
    u = s.uncert_max
    s1 = s.interest_score([100, 0, 0, 0, 0], usi=0.0, uncert=u)
    s2 = s.interest_score([100, 0, 0, 0, 0], usi=0.0, uncert=10 * u)
    # Ya saturado en u: incertidumbre mayor no cambia el score.
    assert s1 == pytest.approx(s2)
    assert s1 - s.interest_score([100, 0, 0, 0, 0], usi=0.0, uncert=0.0) \
        == pytest.approx(s.w[3], abs=1e-6)


def test_score_acotado_y_sin_nan_con_entradas_raras():
    s = AdaptiveSampler()
    for unc in (0.0, 0.3, 1.0, 100.0):
        v = s.interest_score([0, 0, 0, 0, 0], usi=0.0, uncert=unc)
        assert np.isfinite(v) and 0.0 <= v <= 1.0


def test_coverage_boost_premia_clases_poco_vistas():
    s = AdaptiveSampler()
    # El check usa la cobertura DESPUÉS del update (0.9·cov + 0.1·frac), así que
    # hace falta que 0.1·frac no cruce el umbral por sí solo.
    pcts_bui = [0, 40, 0, 30, 30]
    b1 = s.coverage_boost(pcts_bui)      # primera vez: bui poco cubierto
    assert b1 > 0
    for _ in range(30):
        s.coverage_boost(pcts_bui)
    b2 = s.coverage_boost(pcts_bui)      # ya cubierto: no premia más
    assert b2 == 0.0
