"""Tests de enhance_image.py — super-resolución (ESPCM/FSRCNN/LapSRN/EDSR).

Sin red y sin opencv-contrib: cubre `ensure_model` (no descarga en runtime,
checksum) y el guard del wheel NO-contrib que exponía `cv2.dnn_superres` vacío.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np

import enhance_image as ENH


def test_sha256_conocido(tmp_path):
    p = tmp_path / "dato.bin"
    p.write_bytes(b"cansat")
    assert ENH._sha256(p) == hashlib.sha256(b"cansat").hexdigest()


def test_ensure_model_no_descarga_en_runtime(tmp_path, monkeypatch):
    falta = tmp_path / "no_existe.pb"
    monkeypatch.setitem(ENH.MODELS, "test_x2", ("espcn", str(falta), "http://nope"))
    try:
        ENH.ensure_model("test_x2")
        raise AssertionError("debía lanzar FileNotFoundError")
    except FileNotFoundError as e:
        assert "No se descarga en runtime" in str(e)
        assert "opencv-CONTRIB" in str(e)


def test_ensure_model_existente_sin_hash(tmp_path, monkeypatch):
    p = tmp_path / "modelo.pb"
    p.write_bytes(b"pesos")
    monkeypatch.setitem(ENH.MODELS, "test_x2", ("espcn", str(p), "http://nope"))
    algo, ruta = ENH.ensure_model("test_x2", allow_download=False)
    assert algo == "espcn" and Path(ruta) == p


def test_ensure_model_hash_incorrecto(tmp_path, monkeypatch):
    p = tmp_path / "modelo.pb"
    p.write_bytes(b"pesos")
    monkeypatch.setitem(ENH.MODELS, "test_x2", ("espcn", str(p), "http://nope"))
    monkeypatch.setitem(ENH.SHA256, str(p), "0" * 64)
    try:
        ENH.ensure_model("test_x2")
        raise AssertionError("debía lanzar RuntimeError por checksum")
    except RuntimeError as e:
        assert "checksum SHA-256 incorrecto" in str(e)


def test_make_compare_imagen_grande():
    a = np.zeros((500, 600, 3), np.uint8)
    b = np.zeros((500, 600, 3), np.uint8)
    out = ENH.make_compare(a, b, size=420)
    assert out.shape == (420, 840, 3) and out.dtype == np.uint8


def test_make_compare_imagen_chica_no_revienta():
    # Antes asumía h,w >= size y con una imagen chica el slice quedaba invertido.
    a = np.zeros((100, 100, 3), np.uint8)
    b = np.zeros((100, 100, 3), np.uint8)
    out = ENH.make_compare(a, b, size=420)
    assert out.shape[0] == 100 and out.shape[1] == 200


def _imagen(tmp_path) -> Path:
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.full((24, 24, 3), 128, np.uint8))
    return p


def test_main_sin_contrib_avisa_y_devuelve_1(tmp_path, monkeypatch, capsys):
    img = _imagen(tmp_path)
    monkeypatch.setattr(ENH, "ensure_model",
                        lambda key, allow_download=False: ("espcn", "fake.pb"))
    # Simula el wheel NO-contrib: el módulo existe pero sin la clase.
    monkeypatch.setattr(ENH.cv2, "dnn_superres", object(), raising=False)
    assert ENH.main(["--image", str(img)]) == 1
    assert "CONTRIB" in capsys.readouterr().out


def test_main_modelo_faltante_devuelve_1(tmp_path, monkeypatch, capsys):
    img = _imagen(tmp_path)

    def _falla(*_a, **_k):
        raise FileNotFoundError("falta el modelo")

    monkeypatch.setattr(ENH, "ensure_model", _falla)
    assert ENH.main(["--image", str(img)]) == 1
    assert "falta el modelo" in capsys.readouterr().out


def test_main_imagen_inexistente(tmp_path, capsys):
    assert ENH.main(["--image", str(tmp_path / "nope.png")]) == 1
    assert "No se pudo cargar" in capsys.readouterr().out
