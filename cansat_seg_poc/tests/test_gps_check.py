"""Tests de tools/gps_check.py — discriminación fix real vs perfil simulado.

Importa el script por ruta (tools/ no es paquete, misma convención que
tests/test_stress_suite.py) y valida el análisis puro con paquetes sintéticos:
el perfil simulado del firmware no debe contar como fix y una posición real
lejos del perfil sí, con TTFF medido desde pkt=0.
"""

import importlib.util
from pathlib import Path

from cansat import protocol as PROTO

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("gps_check", ROOT / "tools" / "gps_check.py")
GC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GC)


def _sim(pkt: int) -> str:
    lat, lon = GC.sim_esperada(float(pkt))
    return PROTO.format_packet(PROTO.Packet(pkt=pkt, t_s=float(pkt), lat=lat, lon=lon))


def _real(pkt: int, lat: float = -34.63890, lon: float = -58.59213) -> str:
    return PROTO.format_packet(PROTO.Packet(pkt=pkt, t_s=float(pkt), lat=lat, lon=lon))


def test_perfil_simulado_no_confirma_fix():
    lineas = [(3.5 + i, _sim(i)) for i in range(10)]
    res = GC.analizar(lineas)
    assert res["conclusion"]["veredicto"] == "simulado"
    m = res["mediciones"]
    assert m["paquetes"] == 10
    assert m["paquetes_reales"] == 0
    assert m["fix_pct"] == 0.0
    assert m["dist_sim_mediana_m"] < 1.0


def test_fix_real_tras_ttff():
    lineas = [(3.5 + i, _sim(i) if i < 3 else _real(i)) for i in range(10)]
    lineas.append((6.0, "# GPS fix=1 n=4 lat=-34.63890 lon=-58.59213"))
    res = GC.analizar(lineas)
    assert res["conclusion"]["veredicto"].startswith("real")
    m = res["mediciones"]
    assert m["paquetes_reales"] == 7
    assert m["primer_pkt_fix"] == 3
    assert m["ttff_s"] == 3.0
    assert m["debug_fix1"] == 1
    assert m["lat_media"] == -34.63890
    assert m["dispersion_m"] < 1.0
    assert m["dist_sim_mediana_m"] > GC.UMBRAL_REAL_M


def test_sin_datos():
    res = GC.analizar([(0.1, "# GPS: UART2 9600 listo"), (0.2, "# GPS fix=0 n=0")])
    assert res["conclusion"]["veredicto"] == "sin datos"
    assert res["mediciones"]["paquetes"] == 0
    assert res["mediciones"]["debug_gps"] == 1


def test_perdidos_y_checksum_mal():
    lineas = [
        (3.5, _sim(0)),
        (4.5, _sim(1)),
        (6.5, _sim(3)),
        (7.5, _sim(4).rsplit("*", 1)[0] + "*00"),
    ]
    res = GC.analizar(lineas)
    m = res["mediciones"]
    assert m["paquetes"] == 4
    assert m["perdidos_por_contador"] == 1
    assert m["checksum_mal"] == 1
