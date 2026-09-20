"""Tests de cansat/consultas.py — motor simbólico de la Consulta Terrestre.

Misión sintética de 2 frames de 8×8 con pixel = 1 m² (area_m2 = 64):
  · terreno: cols 0-3 edificio, cols 4-5 agua, cols 6-7 vegetación
  · flood:   rectángulo rows 2-5, cols 1-3 (12 m² dentro de edificios)
  · dano2:   bloque 2×2 en (rows 0-1, cols 0-1)
  · vías:    fila 3 completa (8 px = 8 m)
"""

import csv

import numpy as np
import pytest

from cansat import consultas as CO
from cansat import masks as MK

DISPONIBLES = {"terreno", "flood", "dano2", "vias"}


def _mision(tmp_path, con_vias=True):
    masks = tmp_path / "masks"
    masks.mkdir()
    terreno = np.zeros((8, 8), np.uint8)
    terreno[:, 0:4] = 1
    terreno[:, 4:6] = 2
    flood = np.zeros((8, 8), np.uint8)
    flood[2:6, 1:4] = 1
    dano2 = np.zeros((8, 8), np.uint8)
    dano2[1:3, 0:2] = 2
    vias = np.zeros((8, 8), np.uint8)
    vias[3, :] = 1

    for src in ("cam_001", "cam_002"):
        MK.save_mask(masks / f"{src}_terreno.png", terreno)
        MK.save_mask(masks / f"{src}_flood.png", flood)
        MK.save_mask(masks / f"{src}_dano2.png", dano2)
        if con_vias:
            MK.save_mask(masks / f"{src}_vias.png", vias)

    csv_path = tmp_path / "telemetry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["src", "lat", "lon", "area_m2", "people"])
        wr.writerow(["cam_001", -34.6, -58.6, 64.0, 3])
        wr.writerow(["cam_002", -34.6, -58.6, 64.0, 5])
    return CO.DatosMision.cargar(masks, csv_path)


def _region_izquierda():
    """Cuadrante superior izquierdo (filas 0-3, cols 0-3) en lon/lat."""
    lat0, lon0 = -34.6, -58.6
    mlon = 111320.0 * np.cos(np.radians(lat0))
    dlon4 = 4.0 / mlon
    dlat4 = 4.0 / 110540.0
    return [[lon0 - dlon4, lat0], [lon0, lat0],
            [lon0, lat0 - dlat4], [lon0 - dlon4, lat0 - dlat4]]


# ── Parser ──────────────────────────────────────────────────────────────── #
@pytest.mark.parametrize(("texto", "plantilla", "a", "b"), [
    ("área de edificios inundados", "area", "edificio", "inundacion"),
    ("¿cuánta superficie está inundada?", "area", "inundacion", None),
    ("¿cuántos edificios con daño hay?", "count", "edificio", "dano"),
    ("¿qué fracción de las vías está inundada?", "length_fraction", "via", "inundacion"),
    ("distancia entre agua y vegetación", "distance", "agua", "vegetacion"),
    ("¿cuántas personas hay?", "personas", "persona", None),
    # Inglés: mismo parser para el benchmark EarthVQA (QA en inglés).
    ("What is the area of the flooded buildings?", "area", "edificio", "inundacion"),
    ("How many damaged buildings are there?", "count", "edificio", "dano"),
    ("What percentage of roads are flooded?", "length_fraction", "via", "inundacion"),
    ("What is the distance between water and buildings?", "distance", "agua", "edificio"),
    ("How many people are in this scene?", "personas", "persona", None),
    ("Is there any water in this scene?", "exists", "agua", None),
    ("¿hay agua en la zona?", "exists", "agua", None),
    ("What is the area of agriculture?", "area", "vegetacion", None),
])
def test_parser_plantillas(texto, plantilla, a, b):
    spec = CO.parsear(texto, DISPONIBLES)
    assert spec["soportada"], spec
    assert spec["plantilla"] == plantilla
    assert spec["a"] == a and spec["b"] == b


def test_parser_buffer():
    spec = CO.parsear("¿cuántos edificios con daño hay a menos de 50 m de una vía?",
                      DISPONIBLES)
    assert spec["c"] == "via" and spec["buffer_m"] == 50.0


def test_parser_no_soportada():
    spec = CO.parsear("¿cuántas heladerías hay?", DISPONIBLES)
    assert not spec["soportada"]
    assert spec["sugerencias"]


def test_parser_juicio_con_sujeto_conocido():
    # Yes/No de un sujeto que existe → operación de existencia (no se adivina).
    spec = CO.parsear("Are there any buildings in this scene?", DISPONIBLES)
    assert spec["soportada"] and spec["plantilla"] == "exists"
    assert spec["a"] == "edificio"


def test_parser_juicio_sin_sujeto_soportado():
    # 'playground' no existe en nuestra ontología de 5 clases: se rechaza.
    spec = CO.parsear("Are there any playgrounds in this scene?", DISPONIBLES)
    assert not spec["soportada"]


def test_parser_uncultivated_es_suelo():
    # Barbecho = suelo desnudo, no vegetación (falso soporte medido y corregido).
    spec = CO.parsear("Is there any uncultivated agricultural land in this scene?",
                      DISPONIBLES)
    assert spec["soportada"] and spec["a"] == "suelo"


@pytest.mark.parametrize("texto", [
    "Is there any construction land in this scene?",
    "Is there a construction area near the residential area?",
])
def test_parser_uso_de_suelo_no_se_responde(texto):
    # 'construction land/area' es uso de suelo: no lo respondemos con edificios.
    assert not CO.parsear(texto, DISPONIBLES)["soportada"]


def test_parser_no_confunde_area_de_sintagma():
    # 'residential area' es un sintagma, no una consulta de superficie.
    spec = CO.parsear("What are the road types around the residential area?",
                      DISPONIBLES)
    assert not spec["soportada"]


def test_existencia_responde_presencia(tmp_path):
    res = CO.responder("¿hay agua en la zona?", _mision(tmp_path))
    assert res["operacion"] == "exists"
    assert res["total"]["frames_con_presencia"] == 2
    assert res["total"]["de"] == 2


def test_parser_sujeto_sin_mascara():
    spec = CO.parsear("¿qué fracción de las vías está inundada?",
                      DISPONIBLES - {"vias"})
    assert not spec["soportada"]
    assert "via" in spec["motivo"]


# ── Ejecución ───────────────────────────────────────────────────────────── #
def test_area_edificios_inundados(tmp_path):
    res = CO.responder("área de edificios inundados", _mision(tmp_path))
    assert res["soportada"] and res["unidades"] == "m2"
    assert res["total"] == pytest.approx(24.0)      # 12 m² × 2 frames
    assert all(f["poligonos"] for f in res["por_frame"])


def test_area_total_y_region(tmp_path):
    datos = _mision(tmp_path)
    total = CO.responder("área de edificios", datos)
    assert total["total"] == pytest.approx(64.0)    # 32 m² × 2
    region = CO.responder("área de edificios", datos, _region_izquierda())
    assert 0 < region["total"] < total["total"]
    assert region["georref"] is not None and region["region_declarada"]


def test_conteo_con_buffer(tmp_path):
    datos = _mision(tmp_path)
    lejos = CO.responder(
        "¿cuántos edificios con daño hay a menos de 1 m de una vía?", datos)
    cerca = CO.responder(
        "¿cuántos edificios con daño hay a menos de 2 m de una vía?", datos)
    assert lejos["total"] == 0
    assert cerca["total"] == 2                      # 1 componente × 2 frames


def test_fraccion_vias_inundadas(tmp_path):
    res = CO.responder("¿qué fracción de las vías está inundada?", _mision(tmp_path))
    assert res["total"]["fraccion"] == pytest.approx(0.375)
    assert res["total"]["longitud_a_m"] == pytest.approx(16.0)   # 8 m × 2


def test_distancia_agua_vegetacion(tmp_path):
    res = CO.responder("distancia entre agua y vegetación", _mision(tmp_path))
    assert res["total"]["min_m"] == pytest.approx(1.0)
    assert res["total"]["media_m"] is not None


def test_personas(tmp_path):
    res = CO.responder("¿cuántas personas hay?", _mision(tmp_path))
    assert res["total"] == 8


def test_region_sin_geo(tmp_path):
    datos = _mision(tmp_path)
    for fila in datos.filas:
        fila["lat"] = fila["lon"] = ""
    res = CO.responder("área de edificios", datos, _region_izquierda())
    assert not res["soportada"] and "lat/lon" in res["motivo"]
