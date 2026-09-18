"""
Tests de geometría y lógica del pipeline de misión.

Cubren:
  · ``ground_area_m2`` — la huella en tierra usaba ``2·alt·tan(FOV)`` (FOV
    doble) en vez de ``2·alt·tan(FOV/2)``: sobreestimaba el área ×4.6.
  · ``entropy_uncertainty`` — normalizada a [0,1] y en la misma escala que la
    varianza de TTA.
  · ``positions`` de ``eval_sliding`` — el bug ``H < win`` con índice negativo.
"""
import numpy as np
import pytest

from mission_pipeline import FOV_H_DEG, FOV_V_DEG, entropy_uncertainty, ground_area_m2


# ── Huella en tierra ────────────────────────────────────────────────────── #
def test_area_en_tierra_formula_correcta():
    import math
    alt = 250.0
    esperado = (2 * alt * math.tan(math.radians(FOV_H_DEG / 2))) * \
               (2 * alt * math.tan(math.radians(FOV_V_DEG / 2)))
    assert ground_area_m2(alt) == pytest.approx(esperado, rel=1e-9)
    assert ground_area_m2(250.0) == pytest.approx(17198.7, rel=1e-3)


def test_area_no_es_la_formula_vieja_de_fov_doble():
    import math
    alt = 250.0
    vieja = (2 * alt * math.tan(math.radians(FOV_H_DEG))) * \
            (2 * alt * math.tan(math.radians(FOV_V_DEG)))
    assert ground_area_m2(alt) < vieja / 4.0


def test_area_escala_con_el_cuadrado_de_la_altitud():
    assert ground_area_m2(500.0) == pytest.approx(4.0 * ground_area_m2(250.0))


def test_area_en_el_suelo_es_cero():
    assert ground_area_m2(0.0) == 0.0
    assert ground_area_m2(-10.0) == 0.0


# ── Incertidumbre por entropía ──────────────────────────────────────────── #
def test_entropia_uniforme_es_maxima():
    logits = np.zeros((1, 5, 4, 4), np.float32)      # softmax uniforme
    assert entropy_uncertainty(logits) == pytest.approx(1.0, abs=1e-6)


def test_entropia_de_un_pico_es_casi_cero():
    logits = np.full((1, 5, 4, 4), -20.0, np.float32)
    logits[:, 0] = 20.0
    assert entropy_uncertainty(logits) < 0.01


def test_entropia_acotada_entre_cero_y_uno():
    rng = np.random.default_rng(3)
    for _ in range(5):
        logits = rng.normal(0, 5, (1, 5, 8, 8)).astype(np.float32)
        u = entropy_uncertainty(logits)
        assert 0.0 <= u <= 1.0


# ── positions() de eval_sliding ─────────────────────────────────────────── #
pytest.importorskip("torch")
from eval_sliding import positions  # noqa: E402  (tras el importorskip)


def test_positions_nunca_negativas():
    for total, win, stride in [(100, 320, 160), (320, 320, 160), (319, 320, 160)]:
        pos = positions(total, win, stride)
        assert all(p >= 0 for p in pos), (total, win, stride, pos)


def test_positions_cubre_el_frame_exacto():
    pos = positions(100, 320, 160)
    assert pos == [0]


def test_positions_coincide_con_la_grilla_cuando_entra():
    pos = positions(1000, 320, 320)
    assert pos == [0, 320, 640, 680]
