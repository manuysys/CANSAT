"""Tests de cansat/active.py — priorización de tiles para anotar."""

import numpy as np
import pytest

from cansat import active as ACT


def test_entropia_uniforme_es_uno():
    p = np.full((5, 4, 4), 1 / 5)
    assert ACT.entropia_normalizada(p) == pytest.approx(1.0, abs=1e-6)


def test_entropia_one_hot_es_cero():
    p = np.zeros((5, 4, 4))
    p[2] = 1.0
    assert ACT.entropia_normalizada(p) == pytest.approx(0.0, abs=1e-6)


def test_entropia_intermedia():
    p = np.array([0.7, 0.1, 0.1, 0.1])
    e = ACT.entropia_normalizada(p)
    assert 0.3 < e < 0.7


def test_entropia_shape_invalida():
    with pytest.raises(ValueError):
        ACT.entropia_normalizada(np.zeros((2, 2)))


def test_rareza_prefiere_clases_poco_frecuentes():
    prom = [0.36, 0.07, 0.12, 0.09, 0.36]
    comun = ACT.rareza([0.9, 0.02, 0.02, 0.03, 0.03], prom)
    raro = ACT.rareza([0.05, 0.35, 0.35, 0.20, 0.05], prom)
    assert raro > comun


def test_rareza_formas():
    with pytest.raises(ValueError):
        ACT.rareza([0.5, 0.5], [0.3, 0.3, 0.4])


def test_puntaje_pesos_y_rango():
    assert ACT.puntaje(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert ACT.puntaje(0.0, 0.0, 0.0) == 0.0
    # Recorta fuera de rango en vez de explotar.
    assert ACT.puntaje(2.0, -1.0, 5.0) == pytest.approx(ACT.PESOS[0] + ACT.PESOS[2])


def test_motivos_lista_y_fallback():
    m = ACT.motivos(0.9, 0.9, 0.9)
    assert "entropia alta" in m and "clases raras" in m and "estres ambiental" in m
    assert ACT.motivos(0.1, 0.1, 0.1) == ["score de cola"]
