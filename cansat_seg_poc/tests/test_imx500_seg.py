"""Tests de la segmentación NPU (formato medido en la placa 2026-10-04)."""
from __future__ import annotations

import numpy as np

from cansat.imx500_seg import (VOC_CLASSES, VOC_PERSON, cobertura, dibujar,
                               resumen)


def test_voc_persona_es_15():
    assert VOC_CLASSES[VOC_PERSON] == "persona"
    assert VOC_PERSON == 15


def test_cobertura_y_resumen_persona():
    mask = np.zeros((10, 10), np.uint8)
    mask[:5, :] = VOC_PERSON                    # 50 % persona
    cob = cobertura(mask)
    assert cob[0] == 50.0 and cob[VOC_PERSON] == 50.0
    res = resumen(mask)
    assert res["persona_pct"] == 50.0
    assert res["vehiculos_pct"] == 0.0
    assert res["top"][0] == {"clase": "persona", "pct": 50.0}


def test_resumen_vehiculos_suma_clases_voc():
    mask = np.zeros((10, 10), np.uint8)
    mask[0:2] = 7                               # auto 20 %
    mask[2:3] = 6                               # bus 10 %
    res = resumen(mask)
    assert res["vehiculos_pct"] == 30.0
    assert res["persona_pct"] == 0.0


def test_dibujar_pinta_solo_la_mascara():
    frame = np.zeros((8, 8, 3), np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    mask[:4] = VOC_PERSON
    out = dibujar(frame, mask)
    assert out.shape == frame.shape
    assert out[:4].sum() > 0                    # persona pintada (rojo)
    assert out[4:].sum() == 0                   # fondo intacto


def test_dibujar_reescala_al_frame():
    frame = np.zeros((20, 30, 3), np.uint8)
    mask = np.full((10, 10), VOC_PERSON, np.uint8)
    out = dibujar(frame, mask)
    assert out.sum() > 0
