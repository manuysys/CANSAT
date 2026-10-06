"""Tests de cansat.paths (rutas resueltas desde el archivo, no del CWD)."""
from __future__ import annotations

import pytest

from cansat import paths as P


def test_p_resuelve_contra_la_raiz():
    ruta = P.p("outputs", "x.onnx")
    assert ruta.is_absolute()
    assert ruta.parent.parent == P.ROOT or P.ROOT in ruta.parents
    assert ruta.name == "x.onnx"


def test_require_falla_con_hint(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        P.require(tmp_path / "no_existe.bin", hint="regeneralo con tools/x.py")
    assert "regeneralo" in str(e.value)
    f = tmp_path / "si.bin"
    f.write_bytes(b"x")
    assert P.require(f) == f


def test_model_path_opcional_devuelve_none(tmp_path):
    assert P.model_path(str(tmp_path / "nada.onnx"), required=False) is None
    with pytest.raises(FileNotFoundError):
        P.model_path(str(tmp_path / "nada.onnx"), required=True)


def test_ensure_dirs_crea(tmp_path):
    d = tmp_path / "a" / "b"
    P.ensure_dirs(d)
    assert d.is_dir()
