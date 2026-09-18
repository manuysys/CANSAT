"""
Protocolo de telemetría LB135 — FUENTE ÚNICA.

════════════════════════════════════════════════════════════════════════════
POR QUÉ EXISTE ESTE MÓDULO
════════════════════════════════════════════════════════════════════════════
Había **tres formatos incompatibles** que nunca se pudieron hablar entre sí:

  ``mission_pipeline.py`` emitía
      $LB135,t,alt,p,temp,veg,bui,wat,bare,oth,dom,usi,ndvi,vcode,personas,vehiculos,alert
      → 18 campos, con ``$``, **sin número de paquete**.

  ``uart_listener.py`` parseaba
      LB135,pkt,p_base_hPa,p_abs_hPa,alt_m,temp_C
      → 6 campos, **sin** ``$``, y exigía ``startswith("LB135")``.
      O sea: era incapaz de leer lo que emitía el pipeline.

  ``sim_uart.py`` emitía
      LB135,i,p,p,alt,temp
      → 6 campos, y calculaba veg/bui/wat/bare/oth/dom/usi/ndvi/vcode
        para **no usarlos nunca**.

El firmware de la ESP32 no está en ninguno de los dos ZIP, así que el formato
que realmente viaja por el aire es un cuarto desconocido. Este módulo define el
contrato y **parsea tolerante** para aceptar los tres históricos.

════════════════════════════════════════════════════════════════════════════
CONTRATO v2 (el que hay que usar de acá en adelante)
════════════════════════════════════════════════════════════════════════════
::

    $LB135,2,<pkt>,<t_s>,<alt_m>,<p_hPa>,<temp_C>,<veg>,<bui>,<wat>,<bare>,
           <oth>,<dom>,<usi>,<gvi>,<vcode>,<personas>,<vehiculos>,<alert>,
           [<danado_pct>],[<lat>],[<lon>],[<hum_pct>]

    · ``$`` inicial y ``*hh`` de checksum opcionales al parsear; el emisor los pone.
    · campo 2 = versión del protocolo. Un receptor viejo que no lo entienda
      puede detectar el ``2`` y avisar en vez de desalinearse en silencio.
    · ``pkt`` = número de paquete monótono (faltaba en v1: sin él no se puede
      detectar pérdida de paquetes por radio, que es justo lo que importa).
    · ``gvi`` = el índice de verdor que antes se llamaba ``ndvi`` (ver
      ``cansat/indices.py``). El nombre de campo en el CSV sigue siendo ``ndvi``.
    · campos 20-23 (opcionales): ``danado_pct`` (% máximo de daño del
      consenso), ``lat`` y ``lon`` (posición GPS, 5 decimales ≈ 1 m) y
      ``hum_pct`` (humedad relativa del BME280; 0 = sin dato). Se agregan AL
      FINAL a propósito: un receptor viejo los ignora, y el parser tolera
      paquetes de 19 a 23 campos. La posición viene del GPS de la ESP32
      (ATGM336H) y es la que permite georreferenciar cada frame (requisito del
      DPD: "hora, posición, altitud, presión, temperatura"); la humedad cierra
      la trazabilidad del BME280 del DPD, que la mide y no viajaba.

Longitud máxima del paquete: 160 bytes. A 115200 8N1 son ~14 ms por paquete,
cómodo para 1 Hz con margen (antes eran 120 bytes con 19 campos).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PREFIX: str = "LB135"
VERSION: int = 2
BAUD: int = 115200
MAX_LEN: int = 160

# Nombres canónicos de los campos de terreno, en orden de línea.
TERRAIN_FIELDS: tuple[str, ...] = ("veg", "bui", "wat", "bare", "oth")


@dataclass
class Packet:
    """Un paquete de telemetría de misión secundaria."""

    pkt: int = 0
    t_s: float = 0.0
    alt_m: float = 0.0
    p_hPa: float = 0.0
    temp_C: float = 0.0
    veg: float = 0.0
    bui: float = 0.0
    wat: float = 0.0
    bare: float = 0.0
    oth: float = 0.0
    dom: int = -1
    usi: float = 0.0
    gvi: float = 0.0
    vcode: int = 0
    personas: int = 0
    vehiculos: int = 0
    alert: int = 0
    # Campos opcionales (van al final de la línea para que un receptor viejo
    # los ignore sin desalinearse).
    danado_pct: float = 0.0
    lat: float = 0.0        # grados; 0/0 = sin fix GPS
    lon: float = 0.0
    hum_pct: float = 0.0    # humedad relativa (BME280); 0 = sin dato
    version: int = VERSION
    # Metadatos que NO viajan por radio; los agrega el receptor.
    rx_wall: str | None = None
    raw: str | None = None
    checksum_ok: bool | None = None

    def terrain(self) -> list[float]:
        return [self.veg, self.bui, self.wat, self.bare, self.oth]

    def has_fix(self) -> bool:
        """True si el paquete trae una posición GPS utilizable."""
        return not (self.lat == 0.0 and self.lon == 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════ #
#  Checksum
# ══════════════════════════════════════════════════════════════════════ #
def checksum(body: str) -> str:
    """XOR de los bytes entre ``$`` y ``*``, en hex de 2 dígitos (estilo NMEA)."""
    x = 0
    for ch in body:
        x ^= ord(ch)
    return f"{x:02X}"


def verify(line: str) -> tuple[str, bool | None]:
    """
    Quita ``$`` inicial y ``*hh`` final. Devuelve ``(cuerpo, checksum_ok)``.
    ``checksum_ok`` es ``None`` si la línea no traía checksum.
    """
    s = line.strip().strip("\r\n")
    if s.startswith("$"):
        s = s[1:]
    ok: bool | None = None
    if "*" in s:
        s, _, cs = s.rpartition("*")
        ok = cs.strip().upper() == checksum(s)
    return s, ok


# ══════════════════════════════════════════════════════════════════════ #
#  Emisión
# ══════════════════════════════════════════════════════════════════════ #
def format_packet(p: Packet, with_checksum: bool = True, version: int | None = None) -> str:
    """Serializa a la línea v2 (con ``$`` y checksum)."""
    v = VERSION if version is None else version
    body = ",".join(
        [
            PREFIX,
            str(v),
            str(int(p.pkt)),
            f"{p.t_s:07.1f}",
            f"{p.alt_m:06.1f}",
            f"{p.p_hPa:06.1f}",
            f"{p.temp_C:05.1f}",
            *(f"{x:5.1f}" for x in p.terrain()),
            str(int(p.dom)),
            f"{p.usi:6.2f}",
            f"{p.gvi:+6.3f}",
            str(int(p.vcode)),
            str(int(p.personas)),
            str(int(p.vehiculos)),
            str(int(p.alert)),
            f"{p.danado_pct:5.1f}",
            f"{p.lat:.5f}",
            f"{p.lon:.5f}",
            f"{p.hum_pct:4.1f}",
        ]
    )
    if with_checksum:
        return f"${body}*{checksum(body)}"
    return f"${body}"


def format_legacy_v1(p: Packet) -> str:
    """
    Formato v1 (18 campos, sin versión ni número de paquete).

    Sólo para no romper un receptor ya desplegado. Preferir :func:`format_packet`.
    """
    return (
        f"$LB135,{p.t_s:07.1f},{p.alt_m:06.1f},{p.p_hPa:06.1f},{p.temp_C:05.1f},"
        + ",".join(f"{x:5.1f}" for x in p.terrain())
        + f",{int(p.dom)},{p.usi:5.2f},{p.gvi:+6.3f},{int(p.vcode)}"
        + f",{int(p.personas)},{int(p.vehiculos)},{int(p.alert)}"
    )


# ══════════════════════════════════════════════════════════════════════ #
#  Parseo tolerante
# ══════════════════════════════════════════════════════════════════════ #
def _f(s: str, default: float = 0.0) -> float:
    try:
        v = float(s)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def _i(s: str, default: int = 0) -> int:
    return int(_f(s, float(default)))


def _numeric_ok(parts: list[str], indices: tuple[int, ...], minimo: int = 2) -> bool:
    """
    Verifica que al menos ``minimo`` de los campos indicados sean numéricos.

    Sin esto, ``parse`` aceptaba cualquier línea que empezara con ``LB135`` y
    tuviera suficientes comas: ``LB135,abc,def,ghi,jkl,mno`` devolvía un paquete
    con todo en cero, que es peor que devolver ``None`` porque el listener lo
    registra como dato válido.
    """
    ok = 0
    for i in indices:
        if i >= len(parts):
            continue
        try:
            v = float(parts[i])
        except (TypeError, ValueError):
            continue
        if not (math.isnan(v) or math.isinf(v)):
            ok += 1
    return ok >= minimo


def parse(line: str) -> Packet | None:
    """
    Parsea cualquier variante conocida del protocolo LB135.

    Acepta:
      · v2  — 19 campos, con versión (lo que emite :func:`format_packet`)
      · v1  — 18 campos, sin versión (lo que emitía ``mission_pipeline.py``)
      · atm — 6 campos ``pkt,p_base,p_abs,alt,temp`` (lo que parseaba
              ``uart_listener.py``, presumiblemente el firmware de la ESP32)

    Devuelve ``None`` si la línea no es un paquete LB135 (comentario ``#``,
    basura de arranque de la ESP32, línea partida). Nunca lanza.
    """
    if not line:
        return None
    body, cs_ok = verify(line)
    parts = [p.strip() for p in body.split(",")]
    if len(parts) < 2 or not parts[0].upper().lstrip("$").startswith(PREFIX):
        return None

    try:
        n = len(parts)

        # ── v2: LB135,2,pkt,t,alt,p,temp,5×terrain,dom,usi,gvi,vcode,per,veh,alert
        if n >= 19 and parts[1] == str(VERSION):
            if not _numeric_ok(parts, (3, 4, 5, 6), minimo=3):
                return None
            return Packet(
                version=VERSION,
                pkt=_i(parts[2]),
                t_s=_f(parts[3]),
                alt_m=_f(parts[4]),
                p_hPa=_f(parts[5]),
                temp_C=_f(parts[6]),
                veg=_f(parts[7]),
                bui=_f(parts[8]),
                wat=_f(parts[9]),
                bare=_f(parts[10]),
                oth=_f(parts[11]),
                dom=_i(parts[12], -1),
                usi=_f(parts[13]),
                gvi=_f(parts[14]),
                vcode=_i(parts[15]),
                personas=_i(parts[16]),
                vehiculos=_i(parts[17]),
                alert=_i(parts[18]),
                danado_pct=_f(parts[19]) if n > 19 else 0.0,
                lat=_f(parts[20]) if n > 20 else 0.0,
                lon=_f(parts[21]) if n > 21 else 0.0,
                hum_pct=_f(parts[22]) if n > 22 else 0.0,
                checksum_ok=cs_ok,
                raw=line.rstrip(),
            )

        # Versión futura con más campos: NO interpretarla como v1 (lo que
        # desalineaba silenciosamente todos los índices).
        if n > 18 and parts[1] != str(VERSION):
            return None

        # ── v1: LB135,t,alt,p,temp,5×terrain,dom,usi,ndvi,vcode,per,veh,alert
        if n >= 17:
            if not _numeric_ok(parts, (1, 2, 3, 4), minimo=3):
                return None
            return Packet(
                version=1,
                t_s=_f(parts[1]),
                alt_m=_f(parts[2]),
                p_hPa=_f(parts[3]),
                temp_C=_f(parts[4]),
                veg=_f(parts[5]),
                bui=_f(parts[6]),
                wat=_f(parts[7]),
                bare=_f(parts[8]),
                oth=_f(parts[9]),
                dom=_i(parts[10], -1),
                usi=_f(parts[11]),
                gvi=_f(parts[12]),
                vcode=_i(parts[13]),
                personas=_i(parts[14]),
                vehiculos=_i(parts[15]),
                alert=_i(parts[16]) if n > 16 else 0,
                checksum_ok=cs_ok,
                raw=line.rstrip(),
            )

        # ── atmosférico ESP32: LB135,pkt,p_base,p_abs,alt,temp
        if n >= 6:
            if not _numeric_ok(parts, (1, 2, 3, 4, 5), minimo=3):
                return None
            return Packet(
                version=0,
                pkt=_i(parts[1]),
                p_hPa=_f(parts[3]),
                alt_m=_f(parts[4]),
                temp_C=_f(parts[5]),
                checksum_ok=cs_ok,
                raw=line.rstrip(),
            )
    except Exception:  # nunca romper el listener por una línea rara
        return None
    return None


# ══════════════════════════════════════════════════════════════════════ #
#  Atmosfera
# ══════════════════════════════════════════════════════════════════════ #
def sim_atmo(t: float, apogee: float = 250.0, speed: float = 3.0) -> tuple[float, float, float]:
    """
    Perfil del DPD: eyección ~250 m, descenso 2-4 m/s bajo paracaídas.

    Estaba duplicada idéntica en ``mission_pipeline.py`` y ``sim_uart.py``.
    """
    alt = max(0.0, apogee - speed * t)
    p = 1013.25 * (1 - 2.25577e-5 * alt) ** 5.25588
    temp = 15.0 - 0.0065 * alt
    return alt, p, temp


def pressure_to_altitude(p_hPa: float, p0_hPa: float) -> float:
    """Altitud barométrica. ``p0`` es la presión de referencia **calibrada**."""
    if p0_hPa <= 0 or p_hPa <= 0:
        return 0.0
    return 44330.0 * (1.0 - (p_hPa / p0_hPa) ** 0.1903)


def sea_level_pressure(p_hPa: float, alt_m: float) -> float:
    """
    Presión reducida al nivel del mar (QNH) a partir de una lectura en sitio.

    ⚠ ``mission_pipeline.py`` hacía ``p0 = bmp.pressure``, o sea usaba la
      presión **local** como referencia: la altitud arrancaba en ~0 m y derivaba
      con el clima. Lo correcto es una de estas dos:

      1. Calibrar en el punto de lanzamiento con la altitud conocida del predio
         (``p0 = sea_level_pressure(lectura, altitud_predio)``), o
      2. Tomar el QNH del informe meteorológico más cercano.

      Con cualquiera de las dos, la altitud reportada es absoluta y comparable
      entre vuelos.
    """
    if p_hPa <= 0:
        return 1013.25
    return p_hPa / (1 - 2.25577e-5 * max(0.0, alt_m)) ** 5.25588


# ══════════════════════════════════════════════════════════════════════ #
#  Estado del listener UART
# ══════════════════════════════════════════════════════════════════════ #
def read_uart_state(path: str | Path, max_age_s: float = 10.0) -> dict | None:
    """
    Última lectura fresca que dejó ``uart_listener.py`` en ``--state``.

    Devuelve un dict ``{"p_hPa", "temp_C", "lat", "lon"}`` o ``None`` si el
    archivo no existe, está corrupto o es más viejo que ``max_age_s``.

    ⚠ Antes este archivo lo escribía el listener y **nadie lo leía**: el
      pipeline simulaba la atmósfera aunque hubiera telemetría real de la
      ESP32 disponible. Ahora ``mission_pipeline.py --uart-state`` lo consume
      cuando no hay BMP280, y además toma la posición GPS de la ESP32 (el DPD
      pide asociar hora/posición/altitud a cada imagen).
    """
    try:
        p = Path(path)
        if not p.is_file():
            return None
        if (time.time() - p.stat().st_mtime) > max(0.0, max_age_s):
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        p_hpa = float(data["p_hPa"])
        temp = float(data["temp_C"])
        lat = float(data.get("lat") or 0.0)
        lon = float(data.get("lon") or 0.0)
        hum = float(data.get("hum_pct") or 0.0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not (300.0 < p_hpa < 1200.0) or math.isnan(temp):
        return None
    if math.isnan(lat) or math.isnan(lon) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        lat = lon = 0.0
    if math.isnan(hum) or not (0.0 <= hum <= 100.0):
        hum = 0.0
    return {"p_hPa": p_hpa, "temp_C": temp, "lat": lat, "lon": lon, "hum_pct": hum}
