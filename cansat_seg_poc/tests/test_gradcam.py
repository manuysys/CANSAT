"""Tests de tools/gradcam.py — Grad-CAM mínimo (requieren torch: `slow`)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

import numpy as np
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("gradcam", ROOT / "tools" / "gradcam.py")
GR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GR)

pytestmark = pytest.mark.slow


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.classifier(self.features(x))


def _tensor(n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, (1, 3, n, n)).astype(np.float32)


def test_capa_objetivo_ignora_el_classifier():
    capa = GR._capa_objetivo(_Toy())
    assert isinstance(capa, nn.Conv2d) and capa.out_channels == 4


def test_gradcam_forma_y_rango():
    cam = GR.gradcam(_Toy(), _tensor(), clase=1)
    assert cam.shape == (8, 8)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    assert cam.max() > 0.0


def test_gradcam_clase_por_punto():
    cam = GR.gradcam(_Toy(), _tensor(), punto=(4, 4))
    assert cam.shape == (8, 8)


def test_superponer_mantiene_tamano():
    bgr = np.full((16, 24, 3), 120, np.uint8)
    cam = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    out = GR.superponer(bgr, cam)
    assert out.shape == bgr.shape and out.dtype == np.uint8
