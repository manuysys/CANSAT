"""Tests de cansat/calibracion.py — ECE, temperatura y sets APS (sin datos reales)."""

import numpy as np
import pytest

from cansat import calibracion as CAL


def _probs_desde_logits(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_dominio():
    assert CAL.dominio("xbd:joplin-tornado") == "xbd"
    assert CAL.dominio("crasar:Steinhatchee") == "crasar"
    assert CAL.dominio("suelto") == "suelto"


def test_temperatura_uno_no_cambia():
    p = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(CAL.aplicar_temperatura(p, 1.0), p)


def test_temperatura_mayor_suaviza():
    p = np.array([[0.95, 0.03, 0.02]])
    q = CAL.aplicar_temperatura(p, 2.0)
    assert q.max() < p.max() and np.isclose(q.sum(), 1.0)


def test_ece_perfecto_es_cero():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    assert CAL.ece(p, y) == pytest.approx(0.0, abs=1e-9)


def test_ece_confiado_y_equivocado_es_alto():
    p = np.array([[0.99, 0.01], [0.99, 0.01]])
    y = np.array([0, 1])                      # 50 % de acierto con 99 % de conf
    assert CAL.ece(p, y) > 0.3


def test_ajustar_temperatura_desagranda_y_baja_nll():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    # Señal débil (margen chico respecto del ruido): acá la sobre-confianza
    # perjudica la NLL y la temperatura óptima debe suavizar (T > 1).
    z = rng.normal(0, 1.0, (500, 5))
    z[np.arange(500), y] += 0.6
    p_calibrada = _probs_desde_logits(z)
    p_over = _probs_desde_logits(z * 3.0)     # sobre-confiada
    t = CAL.ajustar_temperatura(p_over, y)
    assert t > 1.2
    assert CAL.nll(CAL.aplicar_temperatura(p_over, t), y) < CAL.nll(p_over, y)
    assert CAL.ece(CAL.aplicar_temperatura(p_over, t), y) <= CAL.ece(p_over, y) + 1e-9
    assert CAL.ece(p_calibrada, y) < CAL.ece(p_over, y)


def test_aps_cobertura_minima():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 500)
    p = _probs_desde_logits(rng.normal(0, 1, (500, 4)))
    tau = CAL.aps_umbral(p[:250], y[:250], alpha=0.10)
    cob, tam = CAL.aps_cobertura_tamano(p[250:], y[250:], tau)
    assert cob >= 0.88                    # cerca de 1-alpha (tolera varianza)
    assert 1.0 <= tam <= 4.0


def test_aps_tau_alto_cubre_todo():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 30)
    p = _probs_desde_logits(rng.normal(0, 1, (30, 3)))
    cob, tam = CAL.aps_cobertura_tamano(p, y, tau=1.0)
    assert cob == pytest.approx(1.0) and tam == pytest.approx(3.0)


def test_analizar_con_split_por_eventos():
    rng = np.random.default_rng(3)
    eventos, probs, y = [], [], []
    for i, ev in enumerate(["xbd:a", "xbd:b", "crasar:c", "crasar:d"]):
        n = 60
        yy = rng.integers(0, 5, n)
        p = _probs_desde_logits(rng.normal(0, 1, (n, 5)) * 2.0)
        eventos += [ev] * n
        probs.append(p)
        y.append(yy)
    probs = np.vstack(probs)
    y = np.concatenate(y)
    res = CAL.analizar(probs, y, eventos)
    assert res["n_total"] == 240
    assert res["eventos_calibracion"] and res["eventos_evaluacion"]
    assert "alpha_0.05" in res["aps"]
    assert 0.0 <= res["ece_evaluacion_antes"] <= 1.0
    assert "xbd" in res["por_dominio"]
