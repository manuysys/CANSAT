"""Tests del preset de perfiles de vuelo de mission_pipeline (--perfil)."""

from types import SimpleNamespace

from mission_pipeline import PERFILES, aplicar_perfil


def test_perfil_completo_no_cambia_nada():
    assert aplicar_perfil(SimpleNamespace(perfil="completo")) == {}


def test_perfil_rapido_apaga_dano_tta_crf_y_vis():
    ov = aplicar_perfil(SimpleNamespace(perfil="rapido"))
    assert ov["no_damage"] is True       # NO emite veredicto de daño
    assert ov["tta"] is False
    assert ov["crf_iters"] == 0
    assert ov["no_vis"] is True
    assert ov["onnx_224"] is True        # terreno @224 (mitad de cómputo)


def test_aplicar_perfil_no_muta_el_objeto():
    args = SimpleNamespace(perfil="rapido", no_damage=False, crf_iters=5)
    aplicar_perfil(args)
    assert args.no_damage is False and args.crf_iters == 5


def test_perfiles_declarados():
    assert set(PERFILES) == {"completo", "rapido"}
    assert PERFILES["completo"] == {}
