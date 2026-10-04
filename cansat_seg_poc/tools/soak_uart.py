"""
Soak UART (U1): prueba prolongada de estabilidad del contrato LB135 v2.

Lee el puerto serie (``COM4`` en la PC con la Heltec, ``/dev/serial0`` en la
Pi), verifica checksum, parsea con ``cansat.protocol`` y reporta: paquetes/s,
pérdidas por contador, gaps máximos, checksums malos, basura y rangos de
sensores. No modifica nada del sistema: sólo escucha.

Uso::

    python tools/soak_uart.py --port COM4 --minutes 10
    python tools/soak_uart.py --port /dev/serial0 --minutes 30 \
        --out outputs/soak_uart.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cansat import protocol as PROTO                          # noqa: E402


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return round(s[idx], 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Soak UART del contrato LB135 v2")
    ap.add_argument("--port", required=True, help="COM4 (PC) o /dev/serial0 (Pi)")
    ap.add_argument("--baud", type=int, default=PROTO.BAUD)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/soak_uart.json")
    ap.add_argument("--progreso-s", type=float, default=30.0)
    args = ap.parse_args()

    import serial

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] no se pudo abrir {args.port}@{args.baud}: {e}")
        return 1

    print(f"[soak] {args.port}@{args.baud} durante {args.minutes:.0f} min "
          f"(Ctrl+C corta y reporta)")
    t0 = time.time()
    fin = t0 + args.minutes * 60.0
    n_lineas = n_paq = n_chk = n_parse = n_basura = n_perdidos = n_sin_fix = 0
    temps: list[float] = []
    hums: list[float] = []
    gaps: list[float] = []
    basura_muestras: list[str] = []
    pkt_prev: int | None = None
    t_prev: float | None = None
    ultimo: dict | None = None
    prox_prog = t0 + args.progreso_s

    try:
        while time.time() < fin:
            raw = ser.readline()
            if not raw:
                continue
            linea = raw.decode("utf-8", "replace").strip()
            if not linea:
                continue
            n_lineas += 1
            if not linea.startswith("$LB135"):
                n_basura += 1
                if len(basura_muestras) < 3:
                    basura_muestras.append(linea[:120])
                continue
            _, ok = PROTO.verify(linea)
            if ok is False:
                n_chk += 1
                continue
            p = PROTO.parse(linea)
            if p is None:
                n_parse += 1
                continue
            ahora = time.time()
            n_paq += 1
            if pkt_prev is not None:
                d = (p.pkt - pkt_prev) & 0xFFFF   # el contador puede dar la vuelta
                if 1 < d < 1000:
                    n_perdidos += d - 1
            pkt_prev = p.pkt
            if t_prev is not None:
                gaps.append(ahora - t_prev)
            t_prev = ahora
            if p.temp_C:
                temps.append(p.temp_C)
            if p.hum_pct:
                hums.append(p.hum_pct)
            if not p.has_fix():
                n_sin_fix += 1
            ultimo = p.to_dict()
            if time.time() >= prox_prog:
                dur = time.time() - t0
                print(f"  [{dur:6.0f} s] paquetes={n_paq} "
                      f"({n_paq / dur:.1f}/s) perdidos={n_perdidos} "
                      f"chk_mal={n_chk} basura={n_basura}", flush=True)
                prox_prog = time.time() + args.progreso_s
    except KeyboardInterrupt:
        print("  [Ctrl+C] cortando y reportando…")
    finally:
        ser.close()

    dur = max(time.time() - t0, 1e-6)
    reporte = {
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "puerto": args.port,
        "baud": args.baud,
        "duracion_s": round(dur, 1),
        "lineas": n_lineas,
        "paquetes_validos": n_paq,
        "paquetes_por_s": round(n_paq / dur, 2),
        "checksum_mal": n_chk,
        "parse_fail": n_parse,
        "basura": n_basura,
        "basura_muestras": basura_muestras,
        "perdidos_por_contador": n_perdidos,
        "tasa_perdida_pct": (round(100.0 * n_perdidos /
                                   max(1, n_paq + n_perdidos), 3)),
        "gap_s_p50": _pct(gaps, 0.50),
        "gap_s_p95": _pct(gaps, 0.95),
        "gap_s_max": (round(max(gaps), 2) if gaps else None),
        "temp_c_min": (round(min(temps), 2) if temps else None),
        "temp_c_max": (round(max(temps), 2) if temps else None),
        "hum_pct_min": (round(min(hums), 1) if hums else None),
        "hum_pct_max": (round(max(hums), 1) if hums else None),
        "paquetes_sin_fix": n_sin_fix,
        "ultimo": ultimo,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reporte, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("=" * 60)
    print(f"  {n_paq} paquetes en {dur:.0f} s ({reporte['paquetes_por_s']}/s) · "
          f"perdidos {n_perdidos} ({reporte['tasa_perdida_pct']}%) · "
          f"chk_mal {n_chk} · basura {n_basura}")
    print(f"  gaps p95 {reporte['gap_s_p95']} s · max {reporte['gap_s_max']} s · "
          f"temp {reporte['temp_c_min']}-{reporte['temp_c_max']} °C · "
          f"hum {reporte['hum_pct_min']}-{reporte['hum_pct_max']} %")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
