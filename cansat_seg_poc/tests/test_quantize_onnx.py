"""Tests de quantize_onnx.py — cuantización INT8 estática (QDQ) con calibración.

El QDQ estático nunca se había ejecutado en el proyecto (decisión INT8 pendiente
de re-medición). Acá se cubre de punta a punta sobre un modelo de JUGUETE
(Conv 1×1 + ReLU), sin dataset ni torch: dataset sintético en tmp y el mismo
`quantize_static` que usa el script real.

Requiere `onnx` + `onnxruntime` (los instala el CI); si faltan, se saltea.
"""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import cv2
import onnx
import quantize_onnx as Q
from onnx import TensorProto, helper


def _dataset_sintetico(root, split: str, env: str, n: int = 3, size: int = 16,
                       rng: np.random.Generator | None = None) -> None:
    rng = rng or np.random.default_rng(0)
    d = root / "loveda_raw" / split / env / "images_png"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"img_{i:02d}.png"), img)


def _toy_onnx(path, size: int = 16) -> None:
    """Conv 1×1 (3→3) + ReLU con los nombres del contrato ('input'/'logits')."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3, size, size])
    w = helper.make_tensor(
        "w", TensorProto.FLOAT, [3, 3, 1, 1],
        np.random.default_rng(1).standard_normal((3, 3, 1, 1)).astype(np.float32).tobytes(),
        raw=True)
    conv = helper.make_node("Conv", ["input", "w"], ["c"], kernel_shape=[1, 1],
                            pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["c"], ["logits"])
    graph = helper.make_graph([conv, relu], "toy", [inp], [out], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # `onnx` nuevo genera IR 14 por default y el ORT del CI (1.30) solo soporta
    # hasta IR 13: sin fijarlo, quantize_static falla al recargar el modelo.
    model.ir_version = 9
    onnx.save(model, str(path))


# ── Datos ───────────────────────────────────────────────────────────────── #
def test_split_dirs_y_collect_dedupe(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    for split, env in (("Train", "Urban"), ("Train", "Rural")):
        d = root / "loveda_remapped" / split / env / "images_png"
        d.mkdir(parents=True)
        (d / "misma.png").write_bytes(b"")   # mismo nombre en dos entornos
    monkeypatch.setattr(Q, "DATASET_ROOT", root)
    # En Windows el filesystem es case-insensitive: Urban y urban resuelven al
    # mismo directorio y split_dirs lo lista dos veces. La garantía que importa
    # es que collect() deduplica por nombre.
    assert len({str(d).lower() for d in Q.split_dirs("Train")}) == 2
    assert len(Q.collect("Train")) == 1      # deduplicado por nombre
    assert Q.split_dirs("Val") == []


def test_calib_reader_normaliza_y_termina(tmp_path):
    img = tmp_path / "i.png"
    cv2.imwrite(str(img), np.full((8, 8, 3), 128, np.uint8))
    reader = Q.LovedaCalibrationDataReader([img], img_size=16)
    x = reader.get_next()
    assert x is not None and x["input"].shape == (1, 3, 16, 16)
    assert x["input"].dtype == np.float32
    # Gris 128 ≈ media ImageNet: la normalización queda cerca de 0.
    assert abs(float(x["input"].mean())) < 3.0
    assert reader.get_next() is None


def test_preprocess_shape(tmp_path):
    img = tmp_path / "i.png"
    cv2.imwrite(str(img), np.zeros((32, 32, 3), np.uint8))
    x = Q.preprocess(img, 16)
    assert x.shape == (1, 3, 16, 16) and x.dtype == np.float32


# ── CLI ─────────────────────────────────────────────────────────────────── #
def test_cli_rechaza_calib_y_val_iguales(tmp_path):
    with pytest.raises(SystemExit):
        Q.main(["--calib-split", "Train", "--val-split", "Train"])


def test_cli_modelo_inexistente(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "DATASET_ROOT", tmp_path / "dataset")
    with pytest.raises(SystemExit):
        Q.main(["--in", str(tmp_path / "no.onnx"), "--out", str(tmp_path / "o.onnx")])


# ── QDQ end-to-end con modelo de juguete ────────────────────────────────── #
def test_qdq_end_to_end_modelo_de_juguete(tmp_path, monkeypatch, capsys):
    toy = tmp_path / "toy.onnx"
    _toy_onnx(toy)
    _dataset_sintetico(tmp_path / "dataset", "Train", "Urban")
    _dataset_sintetico(tmp_path / "dataset", "Val", "Urban")
    monkeypatch.setattr(Q, "DATASET_ROOT", tmp_path / "dataset")

    out = tmp_path / "toy_int8_qdq.onnx"
    rc = Q.main(["--in", str(toy), "--out", str(out),
                 "--calib-images", "2", "--val-images", "2", "--img-size", "16"])
    assert rc is None or rc == 0
    assert out.is_file() and out.stat().st_size > 0

    texto = capsys.readouterr().out
    assert "INT8 estático QDQ" in texto
    assert "Coincidencia de predicción por píxel" in texto

    # El modelo cuantizado conserva nombres de entrada/salida del contrato.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out))
    assert sess.get_inputs()[0].name == "input"
    assert sess.get_outputs()[0].name == "logits"
    x = Q.preprocess(sorted((tmp_path / "dataset" / "loveda_raw" / "Val"
                             / "Urban" / "images_png").glob("*.png"))[0], 16)
    y = sess.run(["logits"], {"input": x})[0]
    assert y.shape == (1, 3, 16, 16)
