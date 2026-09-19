"""Tests de cansat/stress.py — estrés ambiental (bruma + humidex) y su
integración en cansat.indices.environment()."""
import numpy as np

from cansat import indices as IDX
from cansat import stress as ST


def _brumosa(base: np.ndarray, alpha: float = 0.75, aire: float = 0.85) -> np.ndarray:
    """Bruma sintética: mezcla con aire blanco."""
    return np.clip(base.astype(np.float32) * (1 - alpha) + 255 * alpha * aire,
                   0, 255).astype(np.uint8)


def test_humidex_valores_conocidos():
    # T=25 °C, RH=50 % → humidex ≈ 28.3 (Environment Canada)
    hx = ST.humidex(25.0, 50.0)
    assert 27.5 < hx < 29.5
    # Con la misma temperatura, más humedad = más calor percibido.
    assert ST.humidex(35.0, 80.0) > ST.humidex(35.0, 20.0)


def test_heat_verdict_categorias():
    assert ST.heat_verdict(25.0) == "CONFORTABLE"
    assert ST.heat_verdict(35.0) == "DISCONFORT"
    assert ST.heat_verdict(42.0) == "ESTRÉS TÉRMICO"
    assert ST.heat_verdict(50.0) == "PELIGROSO"


def test_bruma_detecta_imagen_brumosa():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    clara = ST.haze_metrics(base)
    con_bruma = ST.haze_metrics(_brumosa(base))
    assert con_bruma["haze_pct"] > clara["haze_pct"]
    assert con_bruma["visibility"] < clara["visibility"]


def test_contam_verdict_categorias():
    assert ST.contam_verdict(1.0) == "AIRE LIMPIO"
    assert ST.contam_verdict(10.0) == "BRUMA LEVE"
    assert ST.contam_verdict(30.0) == "BRUMA MODERADA"
    assert ST.contam_verdict(60.0) == "BRUMA DENSA (SMOG)"


def test_stress_score_rango():
    assert ST.stress_score(0.0, 20.0, 0.0) == 0.0
    assert ST.stress_score(50.0, 50.0, 1.0) == 100.0
    medio = ST.stress_score(20.0, 30.0, 0.5)
    assert 0.0 < medio < 100.0


def test_environment_incluye_estres_cuando_hay_datos():
    env = IDX.environment([40, 30, 10, 10, 10], haze_pct=12.0, humidex=33.0)
    assert env["contam"] == "BRUMA LEVE"
    assert env["heat"] == "DISCONFORT"
    assert "stress_idx" in env and 0 <= env["stress_idx"] <= 100


def test_environment_sin_datos_no_inventa_estres():
    env = IDX.environment([40, 30, 10, 10, 10])
    assert "stress_idx" not in env
    assert "haze_pct" not in env


def test_exg_detecta_vegetacion():
    verde = np.zeros((64, 64, 3), np.uint8)
    verde[:, :, 1] = 180                       # BGR: canal verde alto
    gris = np.full((64, 64, 3), 128, np.uint8)
    v, g = ST.exg_metrics(verde), ST.exg_metrics(gris)
    assert v["veg_exg_pct"] > 90 and g["veg_exg_pct"] < 10
    assert v["exg_medio"] > g["exg_medio"]


def test_shadow_pct_extremos():
    assert ST.shadow_pct(np.zeros((32, 32, 3), np.uint8)) == 100.0
    assert ST.shadow_pct(np.full((32, 32, 3), 200, np.uint8)) == 0.0


def test_stress_score_con_fuego_pisa_el_indice():
    base = ST.stress_score(0.0, 20.0, 0.0)
    con_fuego = ST.stress_score(0.0, 20.0, 0.0, fire_pct=10.0)
    assert con_fuego >= 80.0 > base
