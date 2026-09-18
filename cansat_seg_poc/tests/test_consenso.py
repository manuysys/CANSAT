"""
Tests del consenso de daño con UMBRALES POR MODELO.

Contexto: el two-stage va enmascarado por edificios y sus porcentajes son ~3×
menores que los del principal. Umbrales calibrados:
  · xBD (2026-09-17): principal 11.8 %, two-stage 3.7 %.
  · UAV/RescueNet (2026-09-18, modelo de vuelo): two-stage 10.2 %
    (F1 0.828, recall 0.887 — se prioriza detección).
Con un umbral ÚNICO de 10 % y el siamés apagado (2 votantes), el consenso
exigía 2/2 y la alerta no disparaba nunca. Estos tests fijan el
comportamiento nuevo.
"""
import pytest

from cansat import indices as I

TERRENO = [30.0, 20.0, 10.0, 20.0, 20.0]


def test_two_stage_vota_con_su_umbral_calibrado():
    """Principal 12 % y two-stage 12 %: los dos votan (12 > 10.2) → alerta."""
    diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=12.0, pct_siam=None)
    assert alert == 1 and diag == "POSIBLE SISMO/VIENTO"


def test_con_umbral_unico_el_mismo_caso_no_alertaba():
    """Regresión: con 10 % para todo, el two-stage no votaba y no había alerta."""
    _diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=5.0, pct_siam=None,
                              consensus_pct=10.0, consensus_pct_two_stage=10.0)
    assert alert == 0


def test_two_stage_debajo_de_su_umbral_no_vota():
    _diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=5.0, pct_siam=None)
    assert alert == 0, "5 % está debajo del umbral calibrado del two-stage (10.2 %)"


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
    _diag, alert = I.diagnose(TERRENO, pct_dan=12.0, pct_dan2=12.0, pct_siam=0.5)
    assert alert == 1


def test_umbrales_calibrados_publicados():
    assert pytest.approx(10.0) == I.DAMAGE_CONSENSUS_PCT
    assert pytest.approx(10.2) == I.DAMAGE_CONSENSUS_PCT_TWO_STAGE


def test_fuego_dispara_alerta_independiente_del_dano():
    """F3: fuego >1 % alerta aunque no haya daño estructural."""
    diag, alert = I.diagnose(TERRENO, pct_dan=0.0, pct_dan2=0.0, pct_siam=None,
                             pct_fire=3.0)
    assert alert == 1 and diag == "INCENDIO"


def test_humo_extenso_alerta():
    diag, alert = I.diagnose(TERRENO, pct_dan=0.0, pct_dan2=0.0, pct_siam=None,
                             pct_smoke=40.0)
    assert alert == 1 and diag == "HUMO EXTENSO"


def test_humo_leve_no_alerta():
    """Hasta 15 % de humo aparece en escenas rurales normales: no alerta."""
    _diag, alert = I.diagnose(TERRENO, pct_dan=0.0, pct_dan2=0.0, pct_siam=None,
                              pct_smoke=10.0)
    assert alert == 0
