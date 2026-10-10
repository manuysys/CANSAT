"""
Check del enlace LoRa (LB135): TX de vuelo → RX de estación por aire.

Abre A LA VEZ el puerto de la Heltec de vuelo (emite la misma línea LB135 por
UART USB y por LoRa) y el de la RX de estación (recibe por LoRa y la re-emite
por USB), captura N segundos, guarda los crudos y mide:

  · paquetes emitidos (TX) vs recibidos (RX) en el rango común de contadores;
  · pérdida % y la racha máxima de contadores faltantes (pérdida por aire);
  · RSSI/SNR del receptor (líneas ``# LORA rx=… rssi=… snr=…`` del firmware);
  · cadencia del TX (periodo mediano entre paquetes) y gaps de recepción.

Veredicto explícito: ``ok`` (0 pérdidas en el rango común) o ``con pérdidas``.

Modo de campo (prueba a distancia): la TX va en el CanSat con batería y sin
USB, así que sólo se captura la RX. En ese caso los paquetes emitidos se
deducen del rango de contadores recibidos (``--solo-rx``).

Uso::

    python tools/lora_check.py --tx-port COM4 --rx-port COM5 --seconds 120
    python tools/lora_check.py --solo-rx --rx-port COM5 --seconds 60 --nota "100 m"
    python tools/lora_check.py --from-logs outputs/lora_tx_raw.txt outputs/lora_rx_raw.txt
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cansat import protocol as PROTO                          # noqa: E402


@contextlib.contextmanager
def sin_dormir(activo: bool):
    """Evita que Windows suspenda la máquina durante una captura larga.

    Un soak de 10 min quedó contaminado por una suspensión de la notebook
    (gap de recepción de 3350 s, medido 2026-10-09): con esto la ventana
    capturada es continua.
    """
    if not (activo and sys.platform == "win32"):
        yield
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    k = ctypes.windll.kernel32
    k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        k.SetThreadExecutionState(ES_CONTINUOUS)


def mediana(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def percentil(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def capturar(port: str, baud: int, seconds: float, raw_path: Path,
             nombre: str = "", progreso_s: float = 30.0) -> list[tuple[float, str]]:
    """Lee ``seconds`` del puerto y escribe el crudo ``t\\tlinea``."""
    import serial
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    lineas: list[tuple[float, str]] = []
    with serial.Serial(port, baud, timeout=1) as ser:
        t0 = time.monotonic()
        prox = t0 + progreso_s
        with raw_path.open("w", encoding="utf-8") as raw:
            while time.monotonic() - t0 < seconds:
                data = ser.readline()
                if not data:
                    continue
                t_rel = time.monotonic() - t0
                texto = data.decode("utf-8", "replace").strip()
                lineas.append((t_rel, texto))
                raw.write(f"{t_rel:.3f}\t{texto}\n")
                raw.flush()
                if progreso_s and time.monotonic() >= prox and nombre:
                    print(f"[{nombre}] {t_rel:.0f}/{seconds:.0f} s · "
                          f"{len(lineas)} líneas", flush=True)
                    prox = time.monotonic() + progreso_s
    return lineas


def leer_raw(path: Path) -> list[tuple[float, str]]:
    lineas = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t, tab, texto = ln.partition("\t")
        with contextlib.suppress(ValueError):
            lineas.append((float(t), texto if tab else ln))
    return lineas


def _extraer(lineas) -> tuple[list[dict], list[dict]]:
    """Separa paquetes LB135 de las líneas debug LORA (con su timestamp)."""
    pkts: list[dict] = []
    dbg: list[dict] = []
    for t, ln in lineas:
        if ln.startswith("#"):
            d = PROTO.parse_lora_dbg(ln)
            if d and "n" in d:
                dbg.append({"t_rel_s": round(t, 3), **d})
            continue
        p = PROTO.parse(ln)
        if p is not None:
            pkts.append({"t_rel_s": round(t, 3), "pkt": p.pkt,
                         "checksum_ok": p.checksum_ok})
    return pkts, dbg


def _stats(vals: list[float]) -> dict | None:
    if not vals:
        return None
    return {"min": round(min(vals), 2), "mean": round(sum(vals) / len(vals), 2),
            "max": round(max(vals), 2)}


def analizar(tx_lineas, rx_lineas) -> dict:
    """Mediciones del aire a partir de las líneas crudas de ambos lados.

    ``tx_lineas=None`` activa el modo campo (solo RX): lo emitido se deduce del
    rango de contadores recibidos.
    """
    rx_pkts, rx_dbg = _extraer(rx_lineas)
    if tx_lineas is None:
        tx_pkts, tx_dbg, modo = [], [], "solo_rx"
    else:
        tx_pkts, tx_dbg, modo = (*_extraer(tx_lineas), "tx+rx")

    tx_por_pkt = {p["pkt"]: p for p in tx_pkts}
    rx_por_pkt = {p["pkt"]: p for p in rx_pkts}
    r_min = min(rx_por_pkt, default=None)
    r_max = max(rx_por_pkt, default=None)

    if r_min is None:
        emitidos: list[int] = []
    elif tx_por_pkt:
        emitidos = sorted(p for p in tx_por_pkt if r_min <= p <= r_max)
    else:
        # Modo campo: el rango de contadores recibidos ES lo emitido.
        emitidos = list(range(r_min, r_max + 1))
    faltantes = [p for p in emitidos if p not in rx_por_pkt]
    recibidos = len(emitidos) - len(faltantes)

    # Racha máxima de contadores faltantes consecutivos.
    racha = racha_max = 0
    for p in emitidos:
        racha = racha + 1 if p in faltantes else 0
        racha_max = max(racha_max, racha)

    periodos_tx = [b["t_rel_s"] - a["t_rel_s"]
                   for a, b in itertools.pairwise(sorted(tx_pkts, key=lambda p: p["t_rel_s"]))]
    gaps_rx = [b["t_rel_s"] - a["t_rel_s"]
               for a, b in itertools.pairwise(sorted(rx_pkts, key=lambda p: p["t_rel_s"]))]

    rssi = [d["rssi"] for d in rx_dbg if "rssi" in d]
    snr = [d["snr"] for d in rx_dbg if "snr" in d]
    ultimo_tx_n = max((d["n"] for d in tx_dbg), default=None)
    ultimo_rx_n = max((d["n"] for d in rx_dbg), default=None)

    criterios: list[str] = []
    if not emitidos:
        veredicto = "sin datos"
        criterios.append("no hubo paquetes en el rango común (¿ambas placas encendidas?)")
    elif not faltantes:
        veredicto = "ok"
        criterios.append(f"{recibidos}/{len(emitidos)} paquetes en el rango común, 0 pérdidas")
    else:
        veredicto = "con pérdidas"
        criterios.append(f"{recibidos}/{len(emitidos)} paquetes en el rango común "
                         f"({len(faltantes)} faltantes, racha máx {racha_max})")
    if rssi:
        criterios.append(f"RSSI medio {sum(rssi) / len(rssi):.2f} dBm y SNR medio "
                         f"{sum(snr) / len(snr):.2f} dB" if snr else
                         f"RSSI medio {sum(rssi) / len(rssi):.2f} dBm")
    if modo == "solo_rx" and emitidos:
        criterios.append("modo campo (solo RX): lo emitido se deduce del rango "
                         "de contadores recibidos")
    if gaps_rx and max(gaps_rx) > 5.0:
        criterios.append(f"hueco de recepción de {max(gaps_rx):.0f} s "
                         f"(¿suspensión de la PC o corte de energía?)")

    return {
        "mediciones": {
            "modo": modo,
            "tx_paquetes": len(tx_pkts),
            "tx_pkt_min": min(tx_por_pkt, default=None),
            "tx_pkt_max": max(tx_por_pkt, default=None),
            "tx_periodo_mediano_s": (round(mediana(periodos_tx), 3)
                                     if periodos_tx else None),
            "ultimo_tx_n": ultimo_tx_n,
            "rx_paquetes": len(rx_pkts),
            "rx_pkt_min": r_min,
            "rx_pkt_max": r_max,
            "rx_ultimo_rx_n": ultimo_rx_n,
            "emitidos_en_rango": len(emitidos),
            "recibidos": recibidos,
            "perdidos": len(faltantes),
            "perdida_pct": (round(100.0 * len(faltantes) / len(emitidos), 3)
                            if emitidos else None),
            "racha_max_faltantes": racha_max,
            "faltantes_muestra": faltantes[:20],
            "rssi_dbm": _stats(rssi),
            "snr_db": _stats(snr),
            "gap_rx_p95_s": (round(percentil(gaps_rx, 0.95), 2) if gaps_rx else None),
            "gap_rx_max_s": round(max(gaps_rx), 2) if gaps_rx else None,
            "checksum_mal_tx": sum(1 for p in tx_pkts if p["checksum_ok"] is False),
            "checksum_mal_rx": sum(1 for p in rx_pkts if p["checksum_ok"] is False),
        },
        "conclusion": {"veredicto": veredicto, "criterios": criterios},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check del enlace LoRa (LB135)")
    ap.add_argument("--tx-port", default="COM4", help="puerto de la Heltec de vuelo")
    ap.add_argument("--rx-port", default="COM5", help="puerto de la Heltec de estación")
    ap.add_argument("--baud", type=int, default=PROTO.BAUD)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default="docs/benchmarks/lora_rf_1hz.json")
    ap.add_argument("--dir", default="outputs",
                    help="carpeta de los crudos (tx/rx_raw.txt)")
    ap.add_argument("--from-logs", nargs=2, default=None,
                    metavar=("TX_RAW", "RX_RAW"),
                    help="reanalizar dos crudos ya capturados (no abre puertos)")
    ap.add_argument("--solo-rx", action="store_true",
                    help="modo campo: captura sólo la RX (TX con batería, sin USB)")
    ap.add_argument("--progreso-s", type=float, default=30.0,
                    help="imprime progreso cada N s (0 = sin progreso)")
    ap.add_argument("--nota", default="")
    args = ap.parse_args(argv)

    if args.from_logs:
        tx_lineas = leer_raw(Path(args.from_logs[0]))
        rx_lineas = leer_raw(Path(args.from_logs[1]))
        dur = max((lineas[-1][0] for lineas in (tx_lineas, rx_lineas) if lineas),
                  default=0.0)
        tx_raw = args.from_logs[0]
        rx_raw = args.from_logs[1]
        print(f"[lora_check] reanalizando {tx_raw} ({len(tx_lineas)} líneas) + "
              f"{rx_raw} ({len(rx_lineas)} líneas)")
    elif args.solo_rx:
        rx_raw = str(Path(args.dir) / "lora_rx_raw.txt")
        tx_raw = None
        print(f"[lora_check] modo campo: sólo {args.rx_port} durante "
              f"{args.seconds:.0f} s (la TX va con batería)")
        with sin_dormir(True):
            rx_lineas = capturar(args.rx_port, args.baud, args.seconds,
                                 Path(rx_raw), nombre="rx",
                                 progreso_s=args.progreso_s)
        tx_lineas = None
        dur = rx_lineas[-1][0] if rx_lineas else 0.0
    else:
        raw_dir = Path(args.dir)
        tx_raw = str(raw_dir / "lora_tx_raw.txt")
        rx_raw = str(raw_dir / "lora_rx_raw.txt")
        print(f"[lora_check] {args.tx_port} y {args.rx_port} durante "
              f"{args.seconds:.0f} s")
        resultados: dict[str, list[tuple[float, str]]] = {}

        def _run(nombre: str, puerto: str, raw: str) -> None:
            try:
                resultados[nombre] = capturar(puerto, args.baud, args.seconds,
                                              Path(raw), nombre=nombre,
                                              progreso_s=args.progreso_s)
            except Exception as e:
                print(f"[ERROR] {nombre} ({puerto}): {type(e).__name__}: {e}")
                resultados[nombre] = []

        hilos = [threading.Thread(target=_run, args=("tx", args.tx_port, tx_raw)),
                 threading.Thread(target=_run, args=("rx", args.rx_port, rx_raw))]
        with sin_dormir(True):
            for h in hilos:
                h.start()
            for h in hilos:
                h.join()
        tx_lineas = resultados.get("tx", [])
        rx_lineas = resultados.get("rx", [])
        # Ventana real de datos capturados (no el wall time, que incluiría
        # suspensiones de la máquina).
        dur = max((lineas[-1][0] for lineas in (tx_lineas, rx_lineas) if lineas),
                  default=0.0)
        if not tx_lineas or not rx_lineas:
            print("[ERROR] faltó capturar uno de los dos lados")
            return 2

    res = analizar(tx_lineas, rx_lineas)
    m, c = res["mediciones"], res["conclusion"]
    salida = {
        "prueba": "enlace LoRa RF end-to-end (TX vuelo -> RX estacion)",
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tx": ("modo campo: no capturada (la TX va en el CanSat con batería)"
               if tx_lineas is None else
               f"{args.tx_port} (Heltec V3 vuelo, firmware LORA_MODO=1)"),
        "rx": f"{args.rx_port} (Heltec V3 estacion, LORA_MODO=2)",
        "config_radio": "915.0 MHz SF9 BW125 sync 0x12 14 dBm",
        "distancia": "banco: placas contiguas (~10 cm), antenas IPEX conectadas",
        "ventana_s": round(dur, 1),
        "raw_tx": tx_raw,
        "raw_rx": rx_raw,
        "nota": args.nota,
        "mediciones": m,
        "conclusion": c,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(salida, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("=" * 62)
    if m["tx_paquetes"]:
        print(f"  ventana {dur:.0f} s · TX {m['tx_paquetes']} paquetes "
              f"(periodo mediano {m['tx_periodo_mediano_s']} s) · RX {m['rx_paquetes']}")
    else:
        print(f"  ventana {dur:.0f} s · solo RX: {m['rx_paquetes']} paquetes")
    print(f"  rango común: emitidos {m['emitidos_en_rango']} · recibidos "
          f"{m['recibidos']} · perdidos {m['perdidos']} ({m['perdida_pct']} %) · "
          f"racha máx {m['racha_max_faltantes']}")
    if m["rssi_dbm"]:
        print(f"  RSSI {m['rssi_dbm']['min']} / {m['rssi_dbm']['mean']} / "
              f"{m['rssi_dbm']['max']} dBm · SNR "
              f"{m['snr_db']['min']} / {m['snr_db']['mean']} / {m['snr_db']['max']} dB"
              if m["snr_db"] else f"  RSSI {m['rssi_dbm']['mean']} dBm")
    print(f"  gaps RX p95 {m['gap_rx_p95_s']} s · max {m['gap_rx_max_s']} s · "
          f"checksum malos TX/RX {m['checksum_mal_tx']}/{m['checksum_mal_rx']}")
    print(f"  VEREDICTO: {c['veredicto'].upper()}")
    for criterio in c["criterios"]:
        print(f"    · {criterio}")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
