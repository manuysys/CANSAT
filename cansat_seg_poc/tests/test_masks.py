"""Tests de cansat/masks.py — persistencia y geometría de máscaras de consulta."""

import cv2
import numpy as np
import pytest

from cansat import masks as MK


def test_save_load_roundtrip(tmp_path):
    m = np.zeros((8, 8), np.uint8)
    m[2:5, 3:7] = 1
    m[m == 0] = MK.NODATA
    p = MK.save_mask(tmp_path / "sub" / "m.png", m)
    out = MK.load_mask(p)
    assert out is not None and np.array_equal(out, m)
    assert MK.load_mask(tmp_path / "no_existe.png") is None


def test_escala_metrica():
    assert MK.pixel_scale_m(100.0, (10, 10)) == pytest.approx(1.0)
    assert MK.pixel_scale_m(400.0, (20, 20)) == pytest.approx(1.0)
    assert MK.pixel_scale_m(0.0, (10, 10)) == 0.0


def test_area_ignora_nodata():
    m = np.zeros((10, 10), np.uint8)
    m[0:2, :] = 1
    m[5, 5] = MK.NODATA
    assert MK.area_px(m) == 20
    assert MK.area_m2(m, 100.0) == pytest.approx(20.0)


def test_interseccion():
    a = np.zeros((4, 4), np.uint8)
    a[0:3, 0:3] = 1
    b = np.zeros((4, 4), np.uint8)
    b[2:4, 2:4] = 1
    assert MK.interseccion(a, b).sum() == 1
    assert MK.interseccion(a).sum() == 9


def test_mascara_poligono():
    poly = np.array([[1, 1], [5, 1], [5, 5], [1, 5]], np.int32)
    m = MK.mascara_poligono((8, 8), poly)
    assert m[3, 3] == 1
    assert m[0, 0] == 0
    # fillPoly incluye el borde del polígono: 5×5 píxeles, no el interior 4×4.
    assert MK.area_px(m) == 25


def test_buffer_metrica():
    m = np.zeros((15, 15), np.uint8)
    m[7, 7] = 1
    b = MK.buffer_mask(m, 2)
    assert b[7, 9] == 1 and b[9, 7] == 1
    assert b[7, 10] == 0
    assert MK.area_px(b) > MK.area_px(m)


def test_distancia_a():
    m = np.zeros((7, 7), np.uint8)
    m[0, 0] = 1
    d = MK.distancia_a(m)
    assert d[0, 0] == 0.0
    assert d[2, 0] == pytest.approx(2.0)
    assert d[0, 3] == pytest.approx(3.0)
    vacia = MK.distancia_a(np.zeros((4, 4), np.uint8))
    assert np.isinf(vacia).all()


def test_componentes_con_min_area():
    m = np.zeros((20, 20), np.uint8)
    m[2:6, 2:6] = 1          # 16 px
    m[10:12, 10:11] = 1      # 2 px (ruido)
    comps = MK.componentes(m, min_area_px=5, area_frame_m2=400.0)
    assert len(comps) == 1
    assert comps[0]["area_px"] == 16
    assert comps[0]["area_m2"] == pytest.approx(16.0)   # pixel = 1 m²
    assert comps[0]["centroide"] == [3.5, 3.5]


def test_contornos_simplificados():
    m = np.zeros((30, 30), np.uint8)
    m[5:25, 5:15] = 1
    conts = MK.contornos(m, min_area_px=10)
    assert len(conts) == 1
    x, y, w, h = cv2.boundingRect(conts[0])
    assert (x, y, w, h) == (5, 5, 10, 20)


def test_post_flight_persiste_mascaras_nativas(tmp_path):
    """El helper de post_flight remuestrea a resolución nativa y marca NODATA."""
    from post_flight import _guardar_mascaras

    m320 = np.zeros((4, 4), np.uint8)
    m320[0, 0] = 2
    val = np.ones((4, 4), bool)
    val[1, 1] = False
    n = _guardar_mascaras(tmp_path, "cam_001", (8, 8), {
        "dano": (m320, val),
        "terreno": (None, None),
    })
    assert n == 1
    out = MK.load_mask(tmp_path / "cam_001_dano.png")
    assert out is not None and out.shape == (8, 8)
    assert out[0, 0] == 2
    assert out[2, 2] == MK.NODATA
