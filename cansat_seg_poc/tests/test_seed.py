"""Tests de cansat.seed (reproducibilidad de random/numpy/torch)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from cansat.seed import set_seed


def test_set_seed_determinista_random_y_numpy():
    set_seed(42)
    a = [random.random() for _ in range(3)]
    b = np.random.rand(3)
    set_seed(42)
    assert [random.random() for _ in range(3)] == a
    assert np.allclose(np.random.rand(3), b)


def test_set_seed_distinto_cambia():
    set_seed(1)
    a = random.random()
    set_seed(2)
    assert random.random() != a


def test_device_devuelve_algo():
    pytest.importorskip("torch")
    from cansat.seed import device

    dev = device(prefer="cuda", allow_cpu=True)
    assert str(dev) in ("cpu", "cuda", "cuda:0") or "cuda" in str(dev)
