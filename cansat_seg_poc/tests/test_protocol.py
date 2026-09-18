"""
Tests de ``cansat.protocol`` — el paquete de radio/UART LB135.

Cubren el hallazgo 2.2 de la auditoría: había TRES formatos incompatibles y el
listener no podía leer lo que emitía el pipeline. Estos tests fijan que los tres
se parsean, que el round-trip v2 es exacto y que la basura no rompe nada.
"""
from dataclasses import fields

import pytest

from cansat import protocol as P

META_FIELDS = {"raw", "checksum_ok", "rx_wall"}


def payload_equal(a: P.Packet, b: P.Packet) -> bool:
    return all(getattr(a, f.name) == getattr(b, f.name)
               for f in fields(P.Packet) if f.name not in META_FIELDS)


@pytest.fixture
def pkt() -> P.Packet:
    return P.Packet(pkt=7, t_s=32.8, alt_m=180.4, p_hPa=992.1, temp_C=18.3,
                    veg=49.2, bui=0.6, wat=25.8, bare=14.7, oth=9.7,
                    dom=2, usi=0.01, gvi=0.541, vcode=0,
                    personas=3, vehiculos=5, alert=1)


# ── Emisión / round-trip ────────────────────────────────────────────────── #
def test_format_v2_empieza_con_dolar(pkt):
    assert P.format_packet(pkt).startswith("$LB135,2,")


def test_format_v2_trae_checksum(pkt):
    line = P.format_packet(pkt)
    assert "*" in line
    assert len(line.rsplit("*", 1)[1]) == 2


def test_roundtrip_v2_es_exacto(pkt):
    assert payload_equal(P.parse(P.format_packet(pkt)), pkt)


def test_roundtrip_v2_sin_checksum(pkt):
    line = P.format_packet(pkt, with_checksum=False)
    parsed = P.parse(line)
    assert payload_equal(parsed, pkt)
    assert parsed.checksum_ok is None      # no traía, así que no se opina


def test_paquete_dentro_del_largo_maximo(pkt):
    assert len(P.format_packet(pkt)) <= P.MAX_LEN


def test_format_legacy_v1_tiene_18_campos(pkt):
    body, _ = P.verify(P.format_legacy_v1(pkt))
    assert len(body.split(",")) == 17       # LB135 + 16 valores


# ── Checksum ────────────────────────────────────────────────────────────── #
def test_checksum_detecta_un_bit_cambiado(pkt):
    line = P.format_packet(pkt)
    body, cs = P.verify(line)
    assert cs is True
    corrupto = body[:-1] + ("0" if body[-1] != "0" else "1")
    assert P.verify(f"${corrupto}*{line.rsplit('*', 1)[1]}")[1] is False


def test_checksum_es_determinista():
    assert P.checksum("LB135,2,1") == P.checksum("LB135,2,1")


# ── Parseo de los tres formatos históricos ──────────────────────────────── #
def test_parsea_v1_el_que_emitia_el_pipeline_viejo():
    """18 campos, con $, sin número de paquete ni versión."""
    line = ("$LB135,00032.8,0180.4,0992.1,018.3,"
            " 49.2,  0.6, 25.8, 14.7,  9.7,2, 0.01,+0.541,0,3,5,1")
    p = P.parse(line)
    assert p is not None
    assert p.version == 1
    assert p.alt_m == pytest.approx(180.4)
    assert p.wat == pytest.approx(25.8)
    assert p.alert == 1


def test_parsea_el_atmosferico_de_6_campos_de_la_esp32():
    """El formato que parseaba uart_listener.py antes."""
    p = P.parse("LB135,00007,1013.25,992.10,180.40,18.3")
    assert p is not None
    assert p.version == 0
    assert p.pkt == 7
    assert p.p_hPa == pytest.approx(992.10)
    assert p.alt_m == pytest.approx(180.40)
    assert p.temp_C == pytest.approx(18.3)


def test_el_listener_ahora_lee_lo_que_emite_el_pipeline(pkt):
    """
    EL BUG CENTRAL. ``uart_listener.parse_lb135`` exigía
    ``parts[0].startswith("LB135")`` — sin ``$`` — y esperaba exactamente 6
    campos. Lo que emitía ``mission_pipeline.py`` tenía ``$`` y 18 campos, así
    que devolvía ``None``: el listener NO podía leer el pipeline.
    """
    emitted = P.format_packet(pkt)               # lo que emite el pipeline v2
    assert emitted.startswith("$")
    assert P.parse(emitted) is not None          # y el listener lo entiende

    legacy = P.format_legacy_v1(pkt)             # lo que emitía el pipeline v1
    assert P.parse(legacy) is not None


# ── Tolerancia a basura ─────────────────────────────────────────────────── #
@pytest.mark.parametrize("linea", [
    "", None, "\n", "   ",
    "# boot",                                     # comentario de arranque ESP32
    "random noise",
    "LB135",                                      # sólo el prefijo
    "LB135,",                                     # prefijo + coma
    "$LB135,2,1",                                 # truncado a mitad
    "NMEA,4404.14,N",                             # otro protocolo
    "\x00\x01\x02",                               # binario
    "LB135,abc,def,ghi,jkl,mno",                  # campos no numéricos
])
def test_basura_devuelve_none_sin_lanzar(linea):
    """El listener corre en vuelo: ninguna línea rara puede tirar una excepción."""
    assert P.parse(linea) is None


def test_todos_los_campos_invalidos_se_rechazan():
    """
    endurecido en la auditoría: antes esto devolvía un paquete con todo en cero,
    que el listener registraba como dato VÁLIDO. Ahora devuelve None.
    """
    assert P.parse("$LB135,2,7,xx,yy,zz,ww,1,2,3,4,5,6,7,8,9,10,11,12") is None


def test_nan_e_inf_se_rechazan():
    assert P.parse("$LB135,2,1,nan,inf,900,20,0,0,0,0,0,0,0,0,0,0,0,0") is None


def test_un_campo_podrido_no_tira_todo_el_paquete():
    """
    Tolerancia parcial: si la mayoría de los campos clave son numéricos, el
    paquete se acepta y el campo podrido cae a su default. En vuelo es mejor
    recibir telemetría con un hueco que no recibir nada.
    """
    p = P.parse("$LB135,2,7,00032.8,0180.4,xx,018.3,"
                "49.2,0.6,25.8,14.7,9.7,2,0.01,0.541,0,3,5,1")
    assert p is not None
    assert p.alt_m == pytest.approx(180.4)
    assert p.t_s == pytest.approx(32.8)
    assert p.p_hPa == 0.0                      # el campo podrido cayó al default


def test_prefijo_con_suficiente_comas_pero_sin_numeros():
    assert P.parse("LB135,a,b,c,d,e,f") is None


# ── verify() ────────────────────────────────────────────────────────────── #
def test_verify_quita_dolar_y_checksum(pkt):
    body, ok = P.verify(P.format_packet(pkt))
    assert not body.startswith("$")
    assert "*" not in body
    assert ok is True


def test_verify_sin_dolar(pkt):
    body, _ = P.verify(P.format_packet(pkt)[1:])
    assert body.startswith("LB135,2,")


def test_verify_preserva_cr_lf():
    body, _ = P.verify("$LB135,2,1\r\n")
    assert "\r" not in body and "\n" not in body


# ── Atmósfera ───────────────────────────────────────────────────────────── #
def test_sim_atmo_en_t_cero_es_el_apogeo():
    alt, p, t = P.sim_atmo(0.0, apogee=250.0)
    assert alt == pytest.approx(250.0)
    assert p < 1013.25 and p > 950.0
    assert t < 15.0


def test_sim_atmo_desciende_linealmente():
    a0, _, _ = P.sim_atmo(0.0, apogee=250.0, speed=3.0)
    a1, _, _ = P.sim_atmo(10.0, apogee=250.0, speed=3.0)
    assert a0 - a1 == pytest.approx(30.0)


def test_sim_atmo_no_atraviesa_el_suelo():
    alt, _, _ = P.sim_atmo(1e6, apogee=250.0, speed=3.0)
    assert alt == 0.0


def test_sim_atmo_no_esta_duplicada():
    """
    ANTES: la misma función estaba copiada idéntica en mission_pipeline.py y
    sim_uart.py. Ahora los dos importan de acá.
    """
    import inspect
    import sim_uart
    src = inspect.getsource(sim_uart)
    assert "def sim_atmo" not in src, "sim_uart no debe redefinir sim_atmo"


def test_presion_a_altitud_es_inversa_de_altitud_a_presion():
    p0 = 1013.25
    for alt in (0.0, 100.0, 250.0, 1000.0):
        p = p0 * (1 - 2.25577e-5 * alt) ** 5.25588
        assert P.pressure_to_altitude(p, p0) == pytest.approx(alt, abs=0.5)


def test_qnh_desde_lectura_local():
    """Calibrar con la altitud conocida del predio debe devolver ~la presión local a nivel del mar."""
    lectura = 990.0
    alt_predio = 200.0
    qnh = P.sea_level_pressure(lectura, alt_predio)
    assert qnh > lectura
    # Y al revés: con ese QNH, la altitud en el predio da la que conocíamos.
    assert P.pressure_to_altitude(lectura, qnh) == pytest.approx(alt_predio, abs=1.0)


def test_qnh_no_acepta_presion_invalida():
    assert P.sea_level_pressure(0.0, 100.0) == 1013.25
    assert P.pressure_to_altitude(-5.0, 1013.25) == 0.0


def test_altitud_barometrica_con_p0_local_arranca_en_cero():
    """
    DOCUMENTA el bug corregido: si p0 = la presión local (lo que hacía
    mission_pipeline.py), la altitud arranca en 0 m aunque el CanSat esté a
    200 m. Para altitud absoluta hace falta el QNH (--p0-alt).
    """
    p_local = 990.0
    assert P.pressure_to_altitude(p_local, p_local) == pytest.approx(0.0)


# ── Campo de daño (v2, agregado al final) ───────────────────────────────── #
def test_roundtrip_v2_incluye_danado(pkt):
    pkt.danado_pct = 37.5
    parsed = P.parse(P.format_packet(pkt))
    assert parsed is not None
    assert parsed.danado_pct == pytest.approx(37.5)
    assert payload_equal(parsed, pkt)


def test_v2_de_19_campos_sin_danado_sigue_parseando(pkt):
    """Un receptor/emisor viejo (sin el campo 20) no se desalinea."""
    body, _ = P.verify(P.format_packet(pkt, with_checksum=False))
    partes = body.split(",")
    assert len(partes) == 23          # 19 base + daño + lat + lon + humedad
    parsed = P.parse(",".join(partes[:19]))
    assert parsed is not None
    assert parsed.danado_pct == 0.0
    assert parsed.alert == pkt.alert
    assert payload_equal(parsed, pkt)


def test_version_futura_no_se_interpreta_como_v1(pkt):
    """
    Antes, una línea v3 de 20 campos caía en la rama v1 (``n >= 17``) y todos
    los campos quedaban corridos un lugar. Ahora devuelve None.
    """
    body, _ = P.verify(P.format_packet(pkt, with_checksum=False))
    partes = body.split(",")
    partes[1] = "3"
    assert P.parse(",".join(partes)) is None


# ── Estado del listener (uart_state.json) ───────────────────────────────── #
def test_read_uart_state_fresco(tmp_path):
    f = tmp_path / "uart_state.json"
    f.write_text('{"p_hPa": 990.0, "temp_C": 18.5, "alt_m": 200.0, '
                 '"lat": -34.6075, "lon": -58.6126}', encoding="utf-8")
    st = P.read_uart_state(f)
    assert st is not None
    assert st["p_hPa"] == 990.0 and st["temp_C"] == 18.5
    assert st["lat"] == pytest.approx(-34.6075)
    assert st["lon"] == pytest.approx(-58.6126)


def test_read_uart_state_sin_gps_devuelve_ceros(tmp_path):
    """Un estado viejo (sin lat/lon) sigue siendo válido: GPS = 0/0."""
    f = tmp_path / "uart_state.json"
    f.write_text('{"p_hPa": 990.0, "temp_C": 18.5}', encoding="utf-8")
    st = P.read_uart_state(f)
    assert st is not None and st["lat"] == 0.0 and st["lon"] == 0.0


def test_read_uart_state_viejo_no_sirve(tmp_path):
    import os
    import time
    f = tmp_path / "uart_state.json"
    f.write_text('{"p_hPa": 990.0, "temp_C": 18.5}', encoding="utf-8")
    viejo = time.time() - 60
    os.utime(f, (viejo, viejo))
    assert P.read_uart_state(f, max_age_s=10.0) is None


def test_read_uart_state_corrupto_o_inexistente(tmp_path):
    assert P.read_uart_state(tmp_path / "no_existe.json") is None
    f = tmp_path / "mala.json"
    f.write_text("esto no es json", encoding="utf-8")
    assert P.read_uart_state(f) is None
    f.write_text('{"p_hPa": 5.0, "temp_C": 20.0}', encoding="utf-8")   # fuera de rango
    assert P.read_uart_state(f) is None
    f.write_text('{"temp_C": 20.0}', encoding="utf-8")                  # falta p_hPa
    assert P.read_uart_state(f) is None


# ── GPS en el paquete v2 ────────────────────────────────────────────────── #
def test_roundtrip_v2_incluye_gps(pkt):
    pkt.lat, pkt.lon = -34.60751, -58.61263
    parsed = P.parse(P.format_packet(pkt))
    assert parsed is not None
    assert parsed.lat == pytest.approx(-34.60751, abs=1e-5)
    assert parsed.lon == pytest.approx(-58.61263, abs=1e-5)
    assert parsed.has_fix() is True
    assert payload_equal(parsed, pkt)


def test_paquete_sin_gps_no_tiene_fix(pkt):
    parsed = P.parse(P.format_packet(pkt))          # lat/lon en 0 por defecto
    assert parsed is not None and parsed.has_fix() is False


def test_v2_de_20_campos_sin_gps_sigue_parseando(pkt):
    """Un emisor intermedio (con daño pero sin GPS) no se desalinea."""
    body, _ = P.verify(P.format_packet(pkt, with_checksum=False))
    partes = body.split(",")
    assert len(partes) == 23
    parsed = P.parse(",".join(partes[:20]))
    assert parsed is not None
    assert parsed.lat == 0.0 and parsed.lon == 0.0
    assert payload_equal(parsed, pkt)


def test_paquete_dentro_del_largo_maximo_con_gps(pkt):
    pkt.lat, pkt.lon = -34.60751, -58.61263
    pkt.danado_pct = 100.0
    assert len(P.format_packet(pkt)) <= P.MAX_LEN


# ── Humedad del BME280 (DPD: se medía y no viajaba) ─────────────────────── #
def test_roundtrip_v2_incluye_humedad(pkt):
    pkt.hum_pct = 63.5
    parsed = P.parse(P.format_packet(pkt))
    assert parsed is not None
    assert parsed.hum_pct == pytest.approx(63.5)
    assert payload_equal(parsed, pkt)


def test_v2_sin_humedad_queda_en_cero(pkt):
    """Un emisor que todavía no manda humedad no desalinea nada."""
    parsed = P.parse(P.format_packet(pkt))       # hum_pct=0 por defecto
    assert parsed is not None and parsed.hum_pct == 0.0


def test_read_uart_state_incluye_humedad(tmp_path):
    f = tmp_path / "uart_state.json"
    f.write_text('{"p_hPa": 990.0, "temp_C": 18.5, "hum_pct": 71.0}',
                 encoding="utf-8")
    st = P.read_uart_state(f)
    assert st is not None and st["hum_pct"] == pytest.approx(71.0)
    # Humedad fuera de rango se descarta (0), no rompe el estado.
    f.write_text('{"p_hPa": 990.0, "temp_C": 18.5, "hum_pct": 150.0}',
                 encoding="utf-8")
    st = P.read_uart_state(f)
    assert st is not None and st["hum_pct"] == 0.0
