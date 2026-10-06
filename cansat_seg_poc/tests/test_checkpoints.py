"""Tests de cansat.checkpoints (guardado con metadata y carga tolerante)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cansat.checkpoints import (ckpt_meta, load_into,  # noqa: E402
                                load_model_state, save_ckpt)


def _model():
    from torch import nn

    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_save_load_roundtrip_con_metadata(tmp_path):
    m = _model()
    pth = save_ckpt(tmp_path / "ck.pth", m, num_classes=2, img_size=8,
                    miou=0.5, script="test")
    state = load_model_state(pth)
    m2 = _model()
    m2.load_state_dict(state)
    for (n1, w1), (n2, w2) in zip(m.state_dict().items(),
                                  m2.state_dict().items(), strict=True):
        assert n1 == n2 and torch.allclose(w1, w2)
    meta = ckpt_meta(pth)
    assert meta.get("miou") == 0.5 and meta.get("script") == "test"


def test_load_into_tolerante(tmp_path):
    m = _model()
    pth = save_ckpt(tmp_path / "ck.pth", m, num_classes=2)
    m2 = _model()
    _state, frac = load_into(m2, pth, strict=False, min_loaded_frac=0.3)
    assert 0.0 < frac <= 1.0


def test_load_model_state_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_state(tmp_path / "no_existe.pth")
