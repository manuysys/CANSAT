"""Tests de la salud de la Pi (U2): lecturas inyectables, sin placa real."""
from __future__ import annotations

from pathlib import Path

from cansat.salud_pi import (agregar, leer_temp_c, leer_throttled,
                             leer_uptime_s, parse_throttled)


def test_leer_temp_c(tmp_path: Path):
    p = tmp_path / "temp"
    p.write_text("48750\n")
    assert leer_temp_c(p) == 48.75
    assert leer_temp_c(tmp_path / "no_existe") is None
    p.write_text("basura")
    assert leer_temp_c(p) is None


def test_leer_uptime_s(tmp_path: Path):
    p = tmp_path / "uptime"
    p.write_text("1234.5 9999.0\n")
    assert leer_uptime_s(p) == 1234.5
    assert leer_uptime_s(tmp_path / "no_existe") is None


def test_parse_throttled():
    flags = parse_throttled("throttled=0x0")
    assert flags is not None
    assert not any(flags.values())
    flags = parse_throttled("throttled=0x50000")   # bits 16 y 18
    assert flags["subvoltaje_ocurrio"] is True
    assert flags["throttled_ocurrio"] is True
    assert flags["freq_capeada_ocurrio"] is False
    assert flags["subvoltaje_actual"] is False
    assert parse_throttled("sin el formato") is None


def test_leer_throttled_con_runner_falso():
    class R:
        stdout = "throttled=0x0\n"

    assert leer_throttled(runner=lambda *a, **k: R()) is not None

    def rompe(*a, **k):
        raise OSError("vcgencmd no está")

    assert leer_throttled(runner=rompe) is None


def test_agregar_resumen():
    muestras = [
        {"temp_c": 40.0, "uptime_s": 100.0, "throttled": None},
        {"temp_c": 50.0, "uptime_s": 110.0,
         "throttled": {"subvoltaje_ocurrio": True, "throttled_actual": False}},
        {"temp_c": 60.0, "uptime_s": 120.0,
         "throttled": {"subvoltaje_ocurrio": False, "throttled_actual": True}},
    ]
    s = agregar(muestras)
    assert s["n_muestras"] == 3
    assert s["temp_c_min"] == 40.0 and s["temp_c_max"] == 60.0
    assert s["temp_c_prom"] == 50.0
    assert s["uptime_inicio_s"] == 100.0 and s["uptime_fin_s"] == 120.0
    # OR lógico de los flags: subvoltaje ocurrió en una muestra; throttled
    # actual en otra → ambos True en el acumulado.
    assert s["throttled"]["subvoltaje_ocurrio"] is True
    assert s["throttled"]["throttled_actual"] is True
    assert s["throttled_alguna_vez"] is True


def test_agregar_sin_datos():
    s = agregar([{"temp_c": None, "uptime_s": None, "throttled": None}])
    assert s["temp_c_min"] is None and s["temp_c_max"] is None
    assert s["throttled"] is None
    assert s["throttled_alguna_vez"] is None
