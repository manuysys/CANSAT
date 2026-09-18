"""
Tests del consenso de daño con UMBRALES POR MODELO (calibración 2026-09-17).

Contexto: el two-stage va enmascarado por edificios y sus porcentajes son ~3×
menores que los del principal (calibrado: principal 11.8 %, two-stage 3.7 %).
Con un umbral único de 10 % y el siamés apagado (2 votantes), el consenso
exigía 2/2 y la alerta no disparaba nunca. Estos tests fijan el
comportamiento nuevo.
"""
import pytest

from cansat import indices as I

TERRENO = [30.0, 20.0, 10.0, 20.0, 20.0]


def test_two_stage_vota_con_su_umbral_calibrado():
    """Principal 12 % y two-stage 5 %: los dos votan (5 > 3.7) â†’ alerta."""
    diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=5.0, pct_siam=None)
    assert alert == 1 and diag == "POSIBLE SISMO/VIENTO"


def test_con_umbral_unico_el_mismo_caso_no_alertaba():
    """Regresión: con 10 % para todo, el two-stage no votaba y no había alerta."""
    _diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=5.0, pct_siam=None,
                              consensus_pct=10.0, consensus_pct_two_stage=10.0)
    assert alert == 0


def test_two_stage_debajo_de_su_umbral_no_vota():
    _diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=2.0, pct_siam=None)
    assert alert == 0, "2 % está debajo del umbral calibrado del two-stage (3.7 %)"


def test_principal_debajo_no_activa_el_consenso():
    _diag, alert = I.diagnose(TERRENO, pct_dan=5.0, pct_dan2=20.0, pct_siam=None)
    assert alert == 0, "1 de 2 votos no es mayoría"


def test_umbrales_override_por_cli():
    _diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=5.0, pct_siam=None,
                             consensus_pct=20.0, consensus_pct_two_stage=1.0)
    assert alert == 0, "con principal exigido en 20 %, 12 % no vota"
    _diag, alert = I.diagnose(TERRENO, pct_dan=21.0, pct_dan2=1.5, pct_siam=None,
                              consensus_pct=20.0, consensus_pct_two_stage=1.0)
    assert alert == 1


def test_tres_modelos_mayoria_simple():
    """Con 3 votantes, 2 alcanzan."""
    _diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=5.0, pct_siam=0.5)
    assert alert == 1


def test_umbrales_calibrados_publicados():
    assert pytest.approx(10.0) == I.DAMAGE_CONSENSUS_PCT
    assert pytest.approx(3.7) == I.DAMAGE_CONSENSUS_PCT_TWO_STAGE
