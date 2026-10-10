"""
Check GPS (LB135): ¿el ATGM336H de la Heltec de vuelo tiene fix REAL o cae al
perfil simulado declarado?

Captura el serial de la Heltec durante N segundos (default 120), guarda las
líneas crudas en un .txt y calcula lo que importa:

  · paquetes válidos, checksum malo, líneas debug y basura;
  · fracción de paquetes con posición REAL y TTFF (medible sólo si la placa se
    reinicia al abrir el puerto: así el primer paquete visto es pkt=0);
  · dispersión de la posición (m) y velocidad entre fixes consecutivos;
  · DERIVA medida (m/s) contra la que produce el simulador.

Discriminación real vs simulado
───────────────────────────────
El firmware (firmware/heltec_lb135_uart/src/main.cpp) sin fix usa un perfil en
línea recta: LAT_BASE/LON_BASE (Obelisco CABA) con deriva 0.6 m/s al sur y
0.4 m/s al este. Este script reconstruye, para CADA paquete, la posición
simulada esperada a partir de su ``t_s`` y mide la distancia a la posición
reportada. Una posición real del predio está a kilómetros de esa línea; el
perfil simulado cae a ~0 m por construcción. La línea debug ``# GPS fix=1
n=…`` que el firmware imprime cada 30 paquetes es evidencia directa adicional.

Criterios del veredicto (explícitos en el JSON):
  · ``simulado``  : ningún paquete se aparta > UMBRAL_REAL_M del perfil.
  · ``real``      : hay paquetes apartados > UMBRAL_REAL_M (y estable).
  · ``sin datos`` : no llegó ningún paquete LB135.

Uso::

    python tools/gps_check.py --port COM4 --seconds 120
    python tools/gps_check.py --from-log outputs/gps_capture_raw.txt
    python tools/gps_check.py --port /dev/serial0 --seconds 120   # en la Pi
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cansat import protocol as PROTO                          # noqa: E402

# Perfil simulado del firmware (main.cpp): copiado acá para discriminar.
SIM_LAT0 = -34.60372
SIM_LON0 = -58.38159
SIM_DERIVA_LAT_M_S = 0.6
SIM_DERIVA_LON_M_S = 0.4
M_POR_DEG_LAT = 111320.0
M_POR_DEG_LON = 111320.0 * 0.82    # el firmware usa este factor para el perfil

# Distancia a la posición simulada por encima de la cual la posición es REAL.
UMBRAL_REAL_M = 1000.0

GPS_DBG = re.compile(
    r"#\s*GPS\s+fix=(\d)\s+n=(\d+)(?:\s+lat=([-\d.]+)\s+lon=([-\d.]+))?")


def sim_esperada(t_s: float) -> tuple[float, float]:
    """Posición que reportaría el firmware SIN fix para un ``t_s`` dado."""
    lat = SIM_LAT0 - (SIM_DERIVA_LAT_M_S * t_s) / M_POR_DEG_LAT
    lon = SIM_LON0 + (SIM_DERIVA_LON_M_S * t_s) / M_POR_DEG_LON
    return lat, lon


def dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia equirectangular: sobra para decenas de km a escala local."""
    m_lon = M_POR_DEG_LAT * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot((lon2 - lon1) * m_lon, (lat2 - lat1) * M_POR_DEG_LAT)


def mediana(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _pendiente(ts: list[float], ys: list[float]) -> tuple[float, float]:
    """Regresión lineal y = a·t + b → (a, RMS de residuos)."""
    n = len(ts)
    mt = sum(ts) / n
    my = sum(ys) / n
    num = sum((t - mt) * (y - my) for t, y in zip(ts, ys, strict=True))
    den = sum((t - mt) ** 2 for t in ts)
    a = num / den if den else 0.0
    b = my - a * mt
    rms = math.sqrt(sum((y - (a * t + b)) ** 2 for t, y in zip(ts, ys, strict=True)) / n)
    return a, rms


def analizar(lineas) -> dict:
    """Analiza ``(t_rel_s, línea_cruda)`` y devuelve mediciones + veredicto."""
    paquetes: list[dict] = []
    dbg_gps: list[dict] = []
    n_lineas = n_basura = n_debug_otras = n_checksum_mal = 0
    for t_rel, linea in lineas:
        texto = (linea or "").strip()
        if not texto:
            continue
        n_lineas += 1
        if texto.startswith("#"):
            m = GPS_DBG.search(texto)
            if m:
                dbg_gps.append({
                    "t_rel_s": round(t_rel, 3),
                    "fix": int(m.group(1)),
                    "n": int(m.group(2)),
                    "lat": float(m.group(3)) if m.group(3) else None,
                    "lon": float(m.group(4)) if m.group(4) else None,
                })
            else:
                n_debug_otras += 1
            continue
        pkt = PROTO.parse(texto)
        if pkt is None:
            n_basura += 1
            continue
        if pkt.checksum_ok is False:
            n_checksum_mal += 1
        paquetes.append({
            "t_rel_s": round(t_rel, 3),
            "pkt": pkt.pkt,
            "t_s": pkt.t_s,
            "lat": pkt.lat,
            "lon": pkt.lon,
            "checksum_ok": pkt.checksum_ok,
        })

    ordenados = sorted(paquetes, key=lambda p: p["t_rel_s"])
    n_paquetes = len(ordenados)

    # Discriminación real vs simulado por paquete.
    for p in ordenados:
        s_lat, s_lon = sim_esperada(p["t_s"])
        p["dist_sim_m"] = round(dist_m(p["lat"], p["lon"], s_lat, s_lon), 1)
        p["real"] = p["dist_sim_m"] > UMBRAL_REAL_M

    reales = [p for p in ordenados if p["real"]]
    sim_paquetes = [p for p in ordenados if not p["real"]]
    dist_sim = [p["dist_sim_m"] for p in ordenados]

    perdidos = 0
    prev = None
    for p in ordenados:
        if prev is not None and p["pkt"] > prev + 1:
            perdidos += p["pkt"] - prev - 1
        prev = p["pkt"]

    periodos = [b["t_rel_s"] - a["t_rel_s"] for a, b in itertools.pairwise(ordenados)]

    pkt_min = min((p["pkt"] for p in ordenados), default=None)
    pkt_max = max((p["pkt"] for p in ordenados), default=None)

    primer_fix = reales[0] if reales else None
    ttff_s = ttff_nota = None
    if primer_fix is not None:
        if pkt_min == 0:
            t_pkt0 = min(p["t_rel_s"] for p in ordenados if p["pkt"] == 0)
            ttff_s = round(primer_fix["t_rel_s"] - t_pkt0, 2)
        else:
            ttff_nota = (f"la placa no arrancó de cero al abrir el puerto "
                         f"(primer pkt={pkt_min}): TTFF no medible en esta corrida")

    mediciones: dict = {
        "lineas": n_lineas,
        "paquetes": n_paquetes,
        "checksum_mal": n_checksum_mal,
        "basura": n_basura,
        "debug_otras": n_debug_otras,
        "debug_gps": len(dbg_gps),
        "debug_fix1": sum(1 for d in dbg_gps if d["fix"] == 1),
        "debug_lineas": dbg_gps,
        "pkt_min": pkt_min,
        "pkt_max": pkt_max,
        "perdidos_por_contador": perdidos,
        "periodo_mediano_s": round(mediana(periodos), 3) if periodos else None,
        "paquetes_reales": len(reales),
        "paquetes_simulados": len(sim_paquetes),
        "fix_pct": round(100.0 * len(reales) / n_paquetes, 2) if n_paquetes else 0.0,
        "primer_pkt_fix": primer_fix["pkt"] if primer_fix else None,
        "ttff_s": ttff_s,
        "ttff_nota": ttff_nota,
        "dist_sim_mediana_m": mediana(dist_sim),
        "dist_sim_min_m": min(dist_sim) if dist_sim else None,
        "dist_sim_max_m": max(dist_sim) if dist_sim else None,
    }

    if reales:
        lats = [p["lat"] for p in reales]
        lons = [p["lon"] for p in reales]
        lat_med, lon_med = mediana(lats), mediana(lons)
        dispersion = max(dist_m(p["lat"], p["lon"], lat_med, lon_med) for p in reales)
        velocidades = [
            dist_m(a["lat"], a["lon"], b["lat"], b["lon"]) / max(b["t_s"] - a["t_s"], 1e-9)
            for a, b in itertools.pairwise(reales)
        ]
        y_n = [(p["lat"] - reales[0]["lat"]) * M_POR_DEG_LAT for p in reales]
        y_e = [(p["lon"] - reales[0]["lon"]) * M_POR_DEG_LAT
               * math.cos(math.radians(lat_med)) for p in reales]
        ts = [p["t_s"] for p in reales]
        pend_n, rms_n = _pendiente(ts, y_n) if len(reales) >= 2 else (0.0, 0.0)
        pend_e, rms_e = _pendiente(ts, y_e) if len(reales) >= 2 else (0.0, 0.0)
        mediciones.update({
            "lat_media": round(lat_med, 6),
            "lon_media": round(lon_med, 6),
            "lat_min": min(lats), "lat_max": max(lats),
            "lon_min": min(lons), "lon_max": max(lons),
            "dispersion_m": round(dispersion, 1),
            "velocidad_mediana_m_s": (round(mediana(velocidades), 2)
                                      if velocidades else None),
            "deriva_n_m_s": round(pend_n, 3),
            "deriva_e_m_s": round(pend_e, 3),
            "deriva_rms_m": round(math.hypot(rms_n, rms_e), 1),
            "sim_deriva_n_m_s": SIM_DERIVA_LAT_M_S,
            "sim_deriva_e_m_s": SIM_DERIVA_LON_M_S,
        })

    criterios: list[str] = []
    if n_paquetes == 0:
        veredicto = "sin datos"
        criterios.append("no llegó ningún paquete LB135 en la ventana")
    elif not reales:
        veredicto = "simulado"
        criterios.append(
            f"ningún paquete se aparta más de {UMBRAL_REAL_M:.0f} m del perfil "
            f"simulado (dist mediana {mediciones['dist_sim_mediana_m']} m)")
        if mediciones["debug_fix1"]:
            criterios.append("PERO el debug del firmware dice fix=1: contradicción, revisar")
    else:
        veredicto = "real" if len(reales) == n_paquetes else "real (tras TTFF)"
        criterios.append(
            f"{len(reales)}/{n_paquetes} paquetes a >{UMBRAL_REAL_M:.0f} m del "
            f"perfil simulado (dist mediana {mediciones['dist_sim_mediana_m']} m)")
        if ttff_s is not None:
            criterios.append(f"TTFF medido: {ttff_s} s desde pkt=0")
        if mediciones["debug_fix1"]:
            criterios.append(f"debug del firmware con fix=1 en "
                             f"{mediciones['debug_fix1']} línea(s)")
        if len(reales) >= 2:
            criterios.append(
                f"deriva medida N={mediciones['deriva_n_m_s']} m/s / "
                f"E={mediciones['deriva_e_m_s']} m/s vs simulada 0.6/0.4")

    return {
        "mediciones": mediciones,
        "conclusion": {"veredicto": veredicto, "criterios": criterios},
        "paquetes": ordenados,
    }


def capturar(port: str, baud: int, seconds: float, raw_path: Path) -> list[tuple[float, str]]:
    """Lee ``seconds`` del puerto y escribe las líneas crudas a ``raw_path``."""
    import serial
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    lineas: list[tuple[float, str]] = []
    with serial.Serial(port, baud, timeout=1) as ser:
        t0 = time.monotonic()
        with raw_path.open("w", encoding="utf-8") as raw:
            try:
                while time.monotonic() - t0 < seconds:
                    data = ser.readline()
                    if not data:
                        continue
                    t_rel = time.monotonic() - t0
                    texto = data.decode("utf-8", "replace").strip()
                    lineas.append((t_rel, texto))
                    raw.write(f"{t_rel:.3f}\t{texto}\n")
                    raw.flush()
            except KeyboardInterrupt:
                print("\n[gps_check] interrumpido: analizo lo capturado igual")
    return lineas


def leer_raw(path: Path) -> list[tuple[float, str]]:
    lineas = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t, tab, texto = ln.partition("\t")
        with contextlib.suppress(ValueError):
            lineas.append((float(t), texto if tab else ln))
    return lineas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check del GPS real (LB135)")
    ap.add_argument("--port", default=None, help="COM4 (PC) o /dev/serial0 (Pi)")
    ap.add_argument("--baud", type=int, default=PROTO.BAUD)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default="docs/benchmarks/gps_fix_120s.json")
    ap.add_argument("--raw", default="outputs/gps_capture_raw.txt")
    ap.add_argument("--from-log", default=None,
                    help="reanalizar un crudo ya capturado (no abre el puerto)")
    ap.add_argument("--nota", default="", help="contexto del operador (antena, lugar)")
    args = ap.parse_args(argv)

    if args.from_log:
        raw_path = Path(args.from_log)
        if not raw_path.is_file():
            print(f"[ERROR] no existe {raw_path}")
            return 2
        lineas = leer_raw(raw_path)
        dur = lineas[-1][0] if lineas else 0.0
        print(f"[gps_check] reanalizando {raw_path} ({len(lineas)} líneas)")
    elif args.port:
        print(f"[gps_check] {args.port}@{args.baud} durante {args.seconds:.0f} s")
        t0 = time.monotonic()
        lineas = capturar(args.port, args.baud, args.seconds, Path(args.raw))
        dur = time.monotonic() - t0
    else:
        print("[ERROR] especificá --port o --from-log")
        return 2

    res = analizar(lineas)
    m, c = res["mediciones"], res["conclusion"]
    salida = {
        "prueba": "GPS ATGM336H en la Heltec de vuelo: fix real vs perfil simulado",
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "puerto": args.port,
        "baud": args.baud,
        "ventana_s": round(dur, 1),
        "raw": (args.from_log or args.raw),
        "nota": args.nota,
        "hardware": ("Heltec WiFi LoRa 32 V3 (vuelo): ATGM336H por UART2 "
                     "RX=GPIO6 TX=GPIO5 @9600 8N1"),
        "umbral_real_m": UMBRAL_REAL_M,
        "mediciones": m,
        "conclusion": c,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 62)
    print(f"  {m['paquetes']} paquetes en {dur:.0f} s · chk_mal {m['checksum_mal']} · "
          f"basura {m['basura']} · perdidos {m['perdidos_por_contador']}")
    print(f"  periodo entre paquetes: {m['periodo_mediano_s']} s (mediana)")
    print(f"  reales {m['paquetes_reales']} ({m['fix_pct']} %) · "
          f"primer fix pkt={m['primer_pkt_fix']} · TTFF {m['ttff_s']} s")
    print(f"  dist al perfil simulado: mediana {m['dist_sim_mediana_m']} m · "
          f"min {m['dist_sim_min_m']} m · max {m['dist_sim_max_m']} m")
    if "lat_media" in m:
        print(f"  lat/lon media {m['lat_media']}, {m['lon_media']} · "
              f"dispersión {m['dispersion_m']} m · v mediana "
              f"{m['velocidad_mediana_m_s']} m/s")
        print(f"  deriva medida N={m['deriva_n_m_s']} / E={m['deriva_e_m_s']} m/s "
              f"(RMS {m['deriva_rms_m']} m) vs simulada 0.6/0.4")
    print(f"  VEREDICTO: {c['veredicto'].upper()}")
    for criterio in c["criterios"]:
        print(f"    · {criterio}")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
