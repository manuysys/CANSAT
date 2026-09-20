"""
Política fire_only_v1 del clasificador de tipo (cansat/tipos.py).

Contexto: el 7-clases no generaliza a eventos nuevos (LOEO 0.248); solo
incendio se confirma con umbral. Estos tests fijan la política para que un
cambio futuro no la relaje sin querer.
"""
from __future__ import annotations

from cansat import tipos


def _probs(**kw) -> list[float]:
    p = [0.0] * len(tipos.DISASTER_CLASSES)
    for k, v in kw.items():
        p[tipos.DISASTER_CLASSES.index(k)] = v
    return p


def test_incendio_conf_alta_confirma():
    d = tipos.decidir_tipo(_probs(incendio=0.90, otro=0.10), 0.70)
    assert d["estado"] == "confirmado_por_modelo"
    assert d["clase"] == "incendio"
    assert d["confiable"] is True
    assert d["nota"] == ""


def test_incendio_conf_baja_abstiene():
    d = tipos.decidir_tipo(_probs(incendio=0.55, otro=0.45), 0.70)
    assert d["estado"] == "abstencion_por_confianza"
    assert d["confiable"] is False
    assert "umbral" in d["nota"]


def test_umbral_es_inclusivo():
    d = tipos.decidir_tipo(_probs(incendio=0.70), 0.70)
    assert d["estado"] == "confirmado_por_modelo"


def test_inundacion_alta_abstiene():
    d = tipos.decidir_tipo(_probs(inundacion=0.95, incendio=0.05), 0.70)
    assert d["estado"] == "clase_no_habilitada"
    assert d["clase"] == "inundacion"
    assert d["confiable"] is False


def test_huracan_abstiene():
    d = tipos.decidir_tipo(_probs(huracan=0.99), 0.70)
    assert d["estado"] == "clase_no_habilitada"
    assert d["confiable"] is False


def test_tornado_abstiene():
    # Clase sin muestras de train: jamás debe afirmarse.
    d = tipos.decidir_tipo(_probs(tornado=0.99), 0.70)
    assert d["estado"] == "clase_no_habilitada"
    assert d["confiable"] is False


def test_volcan_abstiene():
    d = tipos.decidir_tipo(_probs(volcan=0.99), 0.70)
    assert d["estado"] == "clase_no_habilitada"
    assert d["confiable"] is False


def test_politica_desconocida_abstiene_todo():
    d = tipos.decidir_tipo(_probs(incendio=0.99), 0.70, politica="inexistente")
    assert d["estado"] == "clase_no_habilitada"
    assert d["confiable"] is False


def test_sin_probs_no_explota():
    d = tipos.decidir_tipo([], 0.70)
    assert d["estado"] == "modelo_deshabilitado"
    assert d["clase"] is None
