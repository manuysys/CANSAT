"""Tests de tools/lora_check.py — medición del aire (TX vs RX) sin hardware.

Importa el script por ruta (misma convención que test_stress_suite.py) y valida
la pérdida por contador, la racha máxima y las stats de RSSI/SNR con líneas
sintéticas del protocolo LB135 y del debug LoRa del firmware.
"""

import importlib.util
from pathlib import Path

from cansat import protocol as PROTO

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("lora_check", ROOT / "tools" / "lora_check.py")
LC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LC)


def _pkt(pkt: int) -> str:
    return PROTO.format_packet(PROTO.Packet(pkt=pkt, t_s=float(pkt)))


def _tx_lineas(n: int = 10):
    lineas = [(1.0 + i, _pkt(i)) for i in range(n)]
    lineas.append((0.5, "# LoRa: TX @ 915.0 MHz ... OK SF9 BW125 14 dBm"))
    lineas.append((1.0 + n, f"# LORA tx={n} rssi=-61.00"))
    return lineas


def test_perdida_por_contador_y_racha():
    tx = _tx_lineas(10)
    rx = [(1.0 + i, _pkt(i)) for i in range(10) if i != 4]
    rx += [(2.0, "# LORA rx=9 rssi=-29.50 snr=11.10")]
    res = LC.analizar(tx, rx)
    m = res["mediciones"]
    assert m["emitidos_en_rango"] == 10
    assert m["recibidos"] == 9
    assert m["perdidos"] == 1
    assert m["perdida_pct"] == 10.0
    assert m["racha_max_faltantes"] == 1
    assert m["faltantes_muestra"] == [4]
    assert m["rssi_dbm"]["mean"] == -29.5
    assert m["snr_db"]["mean"] == 11.1
    assert res["conclusion"]["veredicto"] == "con pérdidas"


def test_sin_perdidas_es_ok():
    tx = _tx_lineas(8)
    rx = [(1.0 + i, _pkt(i)) for i in range(8)]
    rx += [(9.0, "# LORA rx=8 rssi=-30.00 snr=11.25")]
    res = LC.analizar(tx, rx)
    assert res["mediciones"]["perdidos"] == 0
    assert res["mediciones"]["perdida_pct"] == 0.0
    assert res["conclusion"]["veredicto"] == "ok"


def test_racha_consecutiva():
    tx = _tx_lineas(10)
    rx = [(1.0 + i, _pkt(i)) for i in range(10) if i not in (3, 4, 5)]
    res = LC.analizar(tx, rx)
    assert res["mediciones"]["perdidos"] == 3
    assert res["mediciones"]["racha_max_faltantes"] == 3


def test_sin_paquetes():
    res = LC.analizar([(0.1, "# LoRa: TX ... OK")], [(0.2, "# LoRa: RX ... OK")])
    assert res["conclusion"]["veredicto"] == "sin datos"


def test_solo_rx_deduce_emitidos_del_contador():
    rx = [(1.0 + i, _pkt(i)) for i in range(10) if i != 3]
    rx += [(9.0, "# LORA rx=9 rssi=-95.00 snr=-2.50")]
    res = LC.analizar(None, rx)
    m = res["mediciones"]
    assert m["modo"] == "solo_rx"
    assert m["emitidos_en_rango"] == 10
    assert m["perdidos"] == 1
    assert m["racha_max_faltantes"] == 1
    assert m["rssi_dbm"]["mean"] == -95.0
    assert res["conclusion"]["veredicto"] == "con pérdidas"
    assert any("modo campo" in c for c in res["conclusion"]["criterios"])
