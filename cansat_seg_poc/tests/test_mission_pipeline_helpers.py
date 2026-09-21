"""Tests de helpers puros de mission_pipeline.py (sin ONNX, sin YOLO, sin torch).

Cubre piezas que la auditoría dejó sin tests: el softmax/terreno, la evidencia
visual (donde vivía el overlay), el conteo de parches verdes y `detect_objects`
con un YOLO de mentira (el conteo de personas/vehículos y sus cajas).
"""

import numpy as np

from mission_pipeline import (
    _disp,
    _green_patches,
    _rnum,
    build_evidence,
    detect_objects,
    softmax_np,
    terrain_percentages,
    valid_mask_for,
)


def test_softmax_np_normaliza_por_clase():
    # En el pipeline los logits son CHW: la clase es el eje 0, así que la
    # normalización es por columna (cada posición suma 1 sobre las clases).
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], np.float32)
    p = softmax_np(x)
    assert p.shape == x.shape
    assert np.allclose(p.sum(axis=0), 1.0, atol=1e-6)


def test_valid_mask_for_thresh_cero_es_todo_valido():
    bgr = np.zeros((32, 32, 3), np.uint8)
    assert valid_mask_for(bgr, 16, thresh=0).all()


def test_terrain_percentages_dominante_y_suma():
    logits = np.zeros((1, 5, 8, 8), np.float32)
    logits[0, 2] = 10.0                     # water domina
    bgr = np.full((8, 8, 3), 120, np.uint8)  # sin borde negro
    pcts, dom, seg, valid = terrain_percentages(logits, bgr, 8)
    assert dom == 2
    assert seg.shape == (8, 8) and valid.shape == (8, 8)
    assert sum(pcts) == 100.0


def test_green_patches_filtra_por_tamano():
    # Fondo clase 3 (bare_ground), parche de vegetación de 12x12 con una esquina
    # cambiada de clase: 143 px de vegetación.
    seg = np.full((40, 40), 3, np.uint8)
    seg[0:12, 0:12] = 0
    seg[0, 0] = 1
    valid = np.ones((40, 40), bool)
    assert _green_patches(seg, valid, min_px=100) == 1
    assert _green_patches(seg, valid, min_px=200) == 0
    assert _green_patches(np.ones((10, 10), np.uint8), np.ones((10, 10), bool)) == 0


def test_build_evidence_devuelve_imagen_del_mismo_tamano():
    bgr = np.full((48, 64, 3), 120, np.uint8)
    seg = np.zeros((48, 64), np.uint8)
    seg[:, :32] = 1
    boxes = [(4, 4, 20, 20, 0), (30, 6, 44, 18, 1)]
    env = {"verdict": "ZONA SALUDABLE", "usi": 1.2, "gvi": 0.4}
    out = build_evidence(bgr, seg, boxes, env, people=1, veh=2, alt=250.0,
                         p=1013.0, temp=23.5, t=12.0, diag="SIN DESASTRE",
                         sharp=10.0, sharp_ok=True)
    assert out.shape == bgr.shape and out.dtype == np.uint8
    # El cartel negro superior se dibuja: la esquina queda oscura.
    assert int(out[2, 2].sum()) == 0


class _Box:
    def __init__(self, cls_id, xyxy, conf=0.9):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=np.float32)]


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class _YoloStub:
    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, bgr, conf, imgsz, verbose):
        return [_Res(self._boxes)]


def test_detect_objects_cuenta_y_devuelve_cajas():
    boxes = [_Box(0, [1, 2, 11, 22]),      # persona
             _Box(2, [30, 5, 50, 20]),     # vehículo
             _Box(7, [60, 5, 70, 15])]     # clase fuera de los sets: se ignora
    people, veh, cajas = detect_objects(_YoloStub(boxes), np.zeros((64, 64, 3), np.uint8),
                                        conf=0.25, imgsz=320,
                                        person_ids={0}, veh_ids={2})
    assert (people, veh) == (1, 1)
    assert cajas == [(1, 2, 11, 22, 0), (30, 5, 50, 20, 2)]


def test_rnum_y_disp_toleran_none():
    # Modelo de daño apagado → None no debe convertirse en 0.0 silencioso.
    assert _rnum(None) is None
    assert _rnum(3.14159, 2) == 3.14
    assert _disp(None) == "off"
    assert _disp(7.25, 1) == "7.2"
