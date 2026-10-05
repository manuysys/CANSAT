"""Tests del MoE de daño (gate serializado + features), integración V11 4.2."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cansat import moe


def _gate_json(path, coef, intercept, mean=None, scale=None):
    path.write_text(json.dumps({
        "scaler": {"mean": mean or [0.0] * 7, "scale": scale or [1.0] * 7},
        "coef": coef, "intercept": intercept, "meta": {"test": True},
    }), encoding="utf-8")


def test_cargar_gate_inexistente_o_corrupto(tmp_path):
    assert moe.cargar_gate(tmp_path / "no_existe.json") is None
    p = tmp_path / "malo.json"
    p.write_text("{no json", encoding="utf-8")
    assert moe.cargar_gate(p) is None
    p.write_text('{"scaler": {"mean": [0], "scale": [1]}, "coef": [0]}',
                 encoding="utf-8")
    assert moe.cargar_gate(p) is None      # longitudes distintas a FEATURES


def test_gate_prob_y_eleccion(tmp_path):
    p = tmp_path / "gate.json"
    # Coef positivo en p_b_media (índice 2): más p_b ⇒ elige B.
    coef = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    _gate_json(p, coef, intercept=-5.0)
    g = moe.cargar_gate(p)
    assert g is not None and g.meta["test"] is True
    feats_bajo = [0.0] * 7
    feats_alto = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert g.prob_b(feats_bajo) < 0.5 and not g.elige_b(feats_bajo)
    assert g.prob_b(feats_alto) > 0.5 and g.elige_b(feats_alto)


def test_features_del_frame():
    pa = np.full((4, 4), 0.5, np.float32)
    pb = np.full((4, 4), 0.5, np.float32)
    tensor = np.zeros((1, 3, 4, 4), np.float32)
    f = moe.features(pa, pb, tensor)
    assert len(f) == len(moe.FEATURES)
    assert f[0] == pytest.approx(0.5) and f[1] == pytest.approx(0.5)
    assert f[4] == pytest.approx(0.0)      # desacuerdo nulo
    pb2 = np.full((4, 4), 0.9, np.float32)
    f2 = moe.features(pa, pb2, tensor)
    assert f2[4] == pytest.approx(0.4)


def test_elegir_sin_gate_es_a():
    pa = np.zeros((4, 4), np.float32)
    pb = np.ones((4, 4), np.float32)
    experto, prob, feats = moe.elegir(None, pa, pb, np.zeros((1, 3, 4, 4)))
    assert experto == "a" and prob == 0.5 and len(feats) == 7
