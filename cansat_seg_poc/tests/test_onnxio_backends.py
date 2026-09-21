"""
Tests del backend ``cv2.dnn`` de ``cansat.onnxio``.

Contexto: la Raspberry Pi Zero v1 (ARMv6) no tiene wheels de onnxruntime, así
que el pipeline de vuelo necesita un backend alternativo. ``cv2.dnn`` ya viene
con OpenCV (dependencia existente) y da resultados equivalentes, con dos
limitaciones: no lee pesos externos (``.onnx.data``) y no expone shapes.

Requieren los modelos exportados en ``outputs/`` (no están en git): los tests
se saltean si no existen.
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from cansat import onnxio

TERRENO = Path("outputs/cansat_seg_terrain_v2.onnx")              # autocontenido
CON_EXTERNO = Path("outputs/cansat_damage3_mobilenetv2.onnx")     # usa .onnx.data


def _tensor(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (1, 3, 320, 320)).astype(np.float32)


@pytest.mark.slow
@pytest.mark.skipif(not TERRENO.is_file(), reason="falta outputs/cansat_seg_terrain_v2.onnx")
def test_backend_cv2_coincide_con_onnxruntime():
    # ⚠ Medición 2026-09-21: con OpenCV 4.13 el acuerdo es >0.99, pero con
    # OpenCV 5.0.0 el backend cv2.dnn devuelve valores absurdos (logits ~1e14,
    # acuerdo 0.01) sobre el ONNX de vuelo — ver decisiones.yaml `opencv-5-dnn`.
    # Por eso va marcado slow y requirements-flight.txt fija opencv<5.
    m_ort = onnxio.OnnxModel(TERRENO, label="terreno", backend="onnxruntime")
    m_cv = onnxio.OnnxModel(TERRENO, label="terreno", backend="cv2")
    assert m_ort.backend == "onnxruntime"
    assert m_cv.backend == "opencv-dnn"
    x = _tensor()
    o1, o2 = m_ort.run({"input": x}), m_cv.run({"input": x})
    assert o1.shape == o2.shape == (1, 5, 320, 320)
    acuerdo = float((o1[0].argmax(0) == o2[0].argmax(0)).mean())
    assert acuerdo > 0.99, f"argmax difiere entre backends: {acuerdo:.3f}"


@pytest.mark.skipif(not TERRENO.is_file(), reason="falta outputs/cansat_seg_terrain_v2.onnx")
def test_backend_cv2_no_expone_shape_y_describe_lo_dice():
    m_cv = onnxio.OnnxModel(TERRENO, label="terreno", backend="cv2")
    assert m_cv.size_px is None           # OpenCV no expone el shape de entrada
    assert m_cv.n_classes is None
    assert "opencv-dnn" in m_cv.describe()


def _modelo_con_pesos_externos() -> Path | None:
    """
    Primer ONNX de outputs/ que use pesos externos (.onnx.data).

    ⚠ Los de daño/flood se re-exportaron autocontenidos el 2026-09-17, así que
    el candidato cambia con el tiempo: se busca en vez de hardcodear.
    """
    for p in sorted(Path("outputs").glob("*.onnx")):
        if p.with_name(p.name + ".data").is_file():
            return p
    return None


def test_backend_cv2_no_lee_pesos_externos_y_lo_explica(tmp_path):
    """El error debe decir QUÉ hacer (tools/onnx_inline.py), no un traceback crudo."""
    modelo = _modelo_con_pesos_externos()
    if modelo is None:
        pytest.skip("no hay ningún ONNX con pesos externos en outputs/ (todos inline)")
    # Copia del modelo SIN su .onnx.data: el caso de copiar a la Pi a medias.
    copia = tmp_path / modelo.name
    copia.write_bytes(modelo.read_bytes())
    with pytest.raises(RuntimeError) as exc:
        onnxio.OnnxModel(copia, label="dano", backend="cv2")
    msg = str(exc.value)
    assert "onnx_inline" in msg


def test_auto_prefiere_onnxruntime():
    if not TERRENO.is_file():
        pytest.skip("falta outputs/cansat_seg_terrain_v2.onnx")
    m = onnxio.OnnxModel(TERRENO, label="terreno")
    assert m.backend == "onnxruntime"
