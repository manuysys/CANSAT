"""
Tests de ``cansat.imx500`` — parser de detecciones on-sensor (DPD: personas).

No requieren hardware: prueban el parseo del tensor SSD, el reescalado y el
conteo de personas/vehículos con datos sintéticos. La captura real con
picamera2 sólo corre en la Pi (ver ``_demo``).
"""
import numpy as np
import pytest

from cansat.imx500 import (
    COCO_PERSON,
    COCO_VEHICLES,
    SSD_PERSON,
    SSD_VEHICLES,
    Deteccion,
    contar,
    parse_ssd_output,
    parse_ssd_outputs,
    rescale,
)


def _fila_ssd(y0, x0, y1, x1, score, cls):
    return [y0, x0, y1, x1, score, cls, -1.0]


def test_parsea_formato_ssd_7_columnas():
    out = np.array([_fila_ssd(0.1, 0.2, 0.3, 0.4, 0.9, 1)], np.float32)
    dets = parse_ssd_output(out, score_thresh=0.5)
    assert len(dets) == 1
    d = dets[0]
    # (y0,x0,y1,x1) → (x1,y1,x2,y2)
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((0.2, 0.1, 0.4, 0.3))
    assert d.score == pytest.approx(0.9) and d.cls == 1


def test_parsea_formato_xyxy_6_columnas():
    out = np.array([[10, 20, 30, 40, 0.8, 2]], np.float32)
    dets = parse_ssd_output(out)
    assert len(dets) == 1
    assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == (10, 20, 30, 40)


def test_descarta_baja_confianza_y_acepta_batch():
    out = np.array([[[0.1, 0.2, 0.3, 0.4, 0.2, 1, -1],     # score bajo
                     [0.1, 0.2, 0.3, 0.4, 0.7, 1, -1]]], np.float32)
    dets = parse_ssd_output(out, score_thresh=0.5)
    assert len(dets) == 1 and dets[0].score == pytest.approx(0.7)


def test_entradas_invalidas_no_explotan():
    assert parse_ssd_output(None) == []
    assert parse_ssd_output(np.zeros((0,), np.float32)) == []
    assert parse_ssd_output(np.zeros((3, 3), np.float32)) == []   # <6 columnas


def test_rescale_a_pixeles():
    dets = [Deteccion(0.5, 0.5, 1.0, 1.0, 0.9, 1)]
    r = rescale(dets, width=640, height=480)
    assert (r[0].x1, r[0].y1, r[0].x2, r[0].y2) == (320.0, 240.0, 640.0, 480.0)
    assert rescale(dets, 640, 480, normalized=False) is dets


def test_conteo_personas_y_vehiculos_coco():
    dets = [
        Deteccion(0, 0, 10, 10, 0.9, 0),    # persona (COCO)
        Deteccion(0, 0, 10, 10, 0.9, 0),    # persona
        Deteccion(0, 0, 10, 10, 0.9, 2),    # auto
        Deteccion(0, 0, 10, 10, 0.9, 7),    # camión
        Deteccion(0, 0, 10, 10, 0.9, 16),   # gato: se ignora
    ]
    per, veh, cajas = contar(dets, person_ids=COCO_PERSON, veh_ids=COCO_VEHICLES)
    assert per == 2 and veh == 2
    assert len(cajas) == 4
    assert cajas[0] == (0, 0, 10, 10, 0)     # formato del pipeline


def test_ssd_usa_coco_80_persona_cero():
    """Medido en hardware 2026-10-03: el SSD preempaquetado es COCO-80 (person=0)."""
    assert SSD_PERSON == COCO_PERSON == frozenset({0})
    assert SSD_VEHICLES == COCO_VEHICLES == frozenset({1, 2, 3, 5, 7})


def test_controles_camara_shutter_y_gain():
    """La captura puede fijar obturación/ganancia (anti motion-blur)."""
    from cansat.imx500 import Imx500Camera, controles_camara

    assert controles_camara(10) == {"FrameRate": 10}
    c = controles_camara(30, shutter_us=8000, gain=16)
    assert c["ExposureTime"] == 8000 and c["AnalogueGain"] == 16
    cam = Imx500Camera("modelo.rpk", shutter_us=8000, gain=16)
    assert cam.shutter_us == 8000 and cam.gain == 16


def test_parsea_formato_real_lista_3_tensores():
    """El formato REAL de get_outputs(): [boxes, scores, classes]."""
    boxes = np.array([[[0.1, 0.2, 0.6, 0.7]]], np.float32)   # (1,1,4) y0,x0,y1,x1
    scores = np.array([[0.77]], np.float32)
    classes = np.array([[0]], np.float32)                    # persona COCO-80
    dets = parse_ssd_outputs([boxes, scores, classes], score_thresh=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((0.2, 0.1, 0.7, 0.6))
    assert d.cls == 0 and d.score == pytest.approx(0.77)


def test_parsea_lista_descarta_baja_confianza():
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.1, 0.1, 0.2, 0.2]], np.float32)
    scores = np.array([0.2, 0.9], np.float32)
    classes = np.array([0, 2], np.float32)
    dets = parse_ssd_outputs([boxes, scores, classes], score_thresh=0.5)
    assert len(dets) == 1 and dets[0].cls == 2


def test_parsea_lista_vacia_o_corta_no_explota():
    assert parse_ssd_outputs(None) == []
    assert parse_ssd_outputs([np.zeros((1, 4)), np.zeros((1,))]) == []   # <3 tensores
    assert parse_ssd_outputs([np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))]) == []
