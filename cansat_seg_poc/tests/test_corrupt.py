"""Tests de cansat/corrupt.py — corrupciones sintéticas de la stress suite.

Son rápidos (numpy/cv2 puro, sin ONNX): en CI corren siempre. La suite real
que evalúa modelos vive en tools/stress_suite.py (marcada slow en su momento).
"""

import numpy as np
import pytest

from cansat import corrupt as COR
from cansat import stress as ST


def _texturada(n: int = 96) -> np.ndarray:
    """Ajedrez de alto contraste: sensible al blur y con estadísticas estables."""
    y, x = np.mgrid[0:n, 0:n]
    tab = (((x // 6 + y // 6) % 2) * 255).astype(np.uint8)
    return np.dstack([tab, tab, tab])


def _mascara(n: int = 96) -> np.ndarray:
    mask = np.zeros((n, n), np.uint8)
    mask[20:60, 20:60] = 1
    mask[60:80, 60:80] = 2
    return mask


def test_corrupciones_declaradas():
    assert set(COR.CORRUPCIONES) == {
        "lluvia", "niebla", "motion_blur",
        "subexposicion", "sobreexposicion", "escala",
    }


def test_aplicar_es_determinista():
    img, mask = _texturada(), _mascara()
    for nombre in COR.CORRUPCIONES:
        a = COR.aplicar(nombre, img, mask, np.random.default_rng(7))
        b = COR.aplicar(nombre, img, mask, np.random.default_rng(7))
        assert np.array_equal(a[0], b[0]), nombre
        assert np.array_equal(a[1], b[1]), nombre


def test_formas_y_dtypes():
    img, mask = _texturada(), _mascara()
    for nombre in COR.CORRUPCIONES:
        out, mask_out, meta = COR.aplicar(nombre, img, mask, np.random.default_rng(1))
        assert out.shape == img.shape and out.dtype == np.uint8, nombre
        assert mask_out is not None
        assert mask_out.shape == mask.shape and mask_out.dtype == np.uint8, nombre
        assert isinstance(meta, dict) and meta, nombre


def test_solo_escala_toca_la_mascara():
    img, mask = _texturada(), _mascara()
    for nombre in COR.CORRUPCIONES:
        if nombre == "escala":
            continue
        _out, mask_out, _meta = COR.aplicar(nombre, img, mask,
                                            np.random.default_rng(2))
        assert np.array_equal(mask_out, mask), nombre


def test_escala_conserva_clases_y_factor():
    img, mask = _texturada(), _mascara()
    _out, mask_out, meta = COR.aplicar("escala", img, mask, np.random.default_rng(3))
    assert np.array_equal(np.unique(mask_out), np.array([0, 1, 2], np.uint8))
    assert mask_out[40, 40] == 1 and mask_out[70, 70] == 2
    assert COR.ESCALA_RANGO[0] <= meta["factor"] <= COR.ESCALA_RANGO[1]


def test_niebla_registra_bruma():
    img = _texturada()
    _out, meta = COR.niebla(img, np.random.default_rng(0))
    limpio = ST.haze_metrics(img)
    assert meta["haze_pct"] > limpio["haze_pct"]
    assert meta["visibility"] < limpio["visibility"]
    assert 0.0 < meta["visibility"] <= 1.0


def test_motion_blur_baja_nitidez():
    img = _texturada()
    out, meta = COR.motion_blur(img, np.random.default_rng(0))
    assert COR.nitidez(out) < COR.nitidez(img)
    assert meta["k"] == COR.MOTION_K


def test_exposiciones_mueven_la_media():
    # Gris medio (no un ajedrez 0/255): la ganancia sí mueve la media.
    img = np.random.default_rng(9).integers(60, 181, (96, 96, 3), dtype=np.uint8)
    sub, _ = COR.exposicion(img, -COR.EXPOSICION_EV)
    sobre, _ = COR.exposicion(img, +COR.EXPOSICION_EV)
    assert sub.mean() < img.mean() < sobre.mean()
    assert sobre.max() == 255


def test_lluvia_anade_trazos():
    img = np.full((96, 96, 3), 40, np.uint8)
    out, meta = COR.lluvia(img, np.random.default_rng(4))
    assert meta["gotas"] > 0
    assert out.max() > img.max()          # hay píxeles brillantes de lluvia


def test_corrupcion_desconocida():
    with pytest.raises(ValueError, match="desconocida"):
        COR.aplicar("granizo", _texturada(), None, np.random.default_rng(0))
