"""Tests de export_onnx.py — exportador único de ONNX.

Requieren **torch** (no está en el CI): van marcados ``slow`` y con
``importorskip``, así que se saltean en CI y corren localmente con el venv.
Cubren la puerta de entrada (opset fijo, argparse y error claro si falta el
checkpoint), que es donde la auditoría encontró exports silenciosamente rotos.
"""

import pytest

pytest.importorskip("torch")

import export_onnx as EX

pytestmark = pytest.mark.slow


def test_opset_fijo_en_17():
    # Requisito del conversor IMX500: no debe degradarse con el default de torch.
    assert EX.OPSET == 17


def test_main_checkpoint_inexistente_devuelve_1(tmp_path, capsys):
    rc = EX.main(["--checkpoint", str(tmp_path / "no_existe.pth"),
                  "--output", str(tmp_path / "o.onnx")])
    assert rc == 1
    assert "No se encontró el checkpoint" in capsys.readouterr().out


def test_main_arch_invalida_falla_en_argparse():
    with pytest.raises(SystemExit):
        EX.main(["--arch", "arquitectura_que_no_existe"])
