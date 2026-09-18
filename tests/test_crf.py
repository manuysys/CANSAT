"""
Tests de ``cansat.crf`` — refinamiento edge-aware guiado por la imagen.

Cubren el hallazgo: el "Dense CRF" original no usaba la imagen de guía (``I``
se calculaba y nunca se leía); el fallback del pipeline era un bilateral
aplicado sobre las propias probabilidades. Ahora el paso de mensajes es un
filtro guiado con box filters core.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cansat.crf import _guided_filter, dense_crf  # noqa: E402  (tras el importorskip)


def test_filtro_guiado_respeta_el_borde_de_la_guia():
    guide = np.zeros((64, 64), np.float32)
    guide[:, 32:] = 1.0
    src = np.zeros((64, 64), np.float32)
    src[:, :32] = 0.9
    src[:, 32:] = 0.1
    out = _guided_filter(guide, src, radius=8, eps=(25 / 255.0) ** 2)
    # Lejos del borde, cada lado conserva su valor: no hay sangrado.
    assert out[32, 6] == pytest.approx(0.9, abs=0.05)
    assert out[32, 57] == pytest.approx(0.1, abs=0.05)


def test_filtro_guiado_suaviza_donde_la_guia_es_plana():
    guide = np.full((64, 64), 0.5, np.float32)
    src = np.zeros((64, 64), np.float32)
    src[30:34, 30:34] = 1.0
    out = _guided_filter(guide, src, radius=8, eps=(25 / 255.0) ** 2)
    assert out[32, 32] < 0.5            # el pico se promedia con el entorno


def test_dense_crf_conserva_forma_y_normaliza():
    rng = np.random.default_rng(0)
    probs = rng.random((5, 32, 32)).astype(np.float32)
    probs /= probs.sum(axis=0, keepdims=True)
    img = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    out = dense_crf(probs, img, iters=3)
    assert out.shape == probs.shape
    assert np.allclose(out.sum(axis=0), 1.0, atol=1e-4)
    assert np.all(out > 0)


def test_dense_crf_iters_cero_es_identidad_normalizada():
    probs = np.abs(np.random.default_rng(1).random((5, 16, 16))).astype(np.float32)
    img = np.zeros((16, 16, 3), np.uint8)
    out = dense_crf(probs, img, iters=0)
    assert np.allclose(out.sum(axis=0), 1.0, atol=1e-4)


def test_dense_crf_guia_de_otro_tamano_se_reescala():
    """Caso típico: probs a 512 y frame nativo más grande."""
    probs = np.full((5, 32, 32), 0.2, np.float32)
    img = np.zeros((64, 64, 3), np.uint8)
    out = dense_crf(probs, img, iters=1)
    assert out.shape == probs.shape
