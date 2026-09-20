"""Tests de tools/stress_suite.py — cabeceras de métrica por tarea (sin ONNX).

Importa el script por ruta (tools/ no es paquete) y verifica que flood/fuego
tengan su métrica objetivo y que daño siga con sus métricas binarias.
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "stress_suite", ROOT / "tools" / "stress_suite.py")
SS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SS)


def test_nombres_por_tarea():
    assert SS.nombres_clases("flood", 3) == ["other", "inundacion", "agua"]
    assert SS.nombres_clases("fuego", 3) == ["other", "fuego", "humo"]
    assert SS.nombres_clases("dano", 3) == ["other", "intacto", "danado"]
    assert len(SS.nombres_clases("terreno", 5)) == 5


def test_cabecera_flood():
    conf = np.array([[8, 1, 1], [0, 3, 1], [0, 1, 4]], dtype=np.int64)
    out = SS.resumen_metricas(conf, 3, [10.0, 8.0], tarea="flood")
    assert out["iou_clase_objetivo"] == round(3 / 6, 6)      # inundación
    assert out["iou_agua"] == round(4 / 7, 6)
    assert "iou_bin_dano" not in out


def test_cabecera_fuego():
    conf = np.array([[8, 1, 1], [1, 4, 0], [0, 2, 3]], dtype=np.int64)
    out = SS.resumen_metricas(conf, 3, [], tarea="fuego")
    assert out["iou_fuego"] == out["iou_clase_objetivo"]
    assert out["iou_humo"] == round(SS._iou(conf, 2), 6)
    assert out["nitidez_media"] is None


def test_cabecera_dano_sigue_igual():
    conf = np.array([[10, 0, 0], [0, 2, 3], [0, 1, 6]], dtype=np.int64)
    out = SS.resumen_metricas(conf, 3, [5.0], tarea="dano2")
    assert out["iou_bin_dano"] == round(6 / 10, 6)
    assert out["iou_dano_two_stage"] == round(6 / 10, 6)
    assert 0.7 < out["f1_bin_dano"] < 0.8
    assert "iou_clase_objetivo" not in out


def test_delta_incluye_objetivo():
    estresado = {"miou": 0.3, "iou_clase_objetivo": 0.2, "nitidez_media": 5.0}
    limpio = {"miou": 0.5, "iou_clase_objetivo": 0.4, "nitidez_media": 10.0}
    d = SS._delta(estresado, limpio)
    assert d["delta_miou"] == -0.2
    assert d["delta_iou_clase_objetivo"] == -0.2
    assert d["delta_nitidez"] == -5.0
