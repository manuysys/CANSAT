"""Tests del preprocesamiento (fuente única) y la corrección gray-world."""
from __future__ import annotations

import numpy as np
import pytest

from cansat import preprocess as PP


def test_gray_world_iguala_medias_uint8():
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = 40    # R
    rgb[..., 1] = 80    # G
    rgb[..., 2] = 160   # B (dominante azul)
    out = PP.gray_world(rgb)
    assert out.dtype == np.uint8
    medias = out.astype(np.float32).mean(axis=(0, 1))
    assert float(medias.max() - medias.min()) < 2.0


def test_gray_world_uniforme_no_cambia():
    rgb = np.full((4, 4, 3), 0.5, np.float32)
    out = PP.gray_world(rgb)
    assert out.dtype == np.float32 and np.allclose(out, 0.5)


def test_preprocess_bgr_color_norm_cambia_el_tensor():
    pytest.importorskip("cv2")
    bgr = np.zeros((16, 16, 3), np.uint8)
    bgr[..., 0] = 160   # B
    bgr[..., 2] = 40    # R
    t_sin = PP.preprocess_bgr(bgr, 8, color_norm=False)
    t_con = PP.preprocess_bgr(bgr, 8, color_norm=True)
    assert t_sin.shape == t_con.shape == (1, 3, 8, 8)
    assert not np.allclose(t_sin, t_con)
