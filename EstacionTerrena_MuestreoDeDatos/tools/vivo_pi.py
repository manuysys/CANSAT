"""
Pull en vivo (U5): sincroniza el último vuelo de la Pi al ``outputs/mission``
de la estación. La web ya auto-refresca ``/api/mission`` cada 3 s, así que con
esto el panel queda EN VIVO sin tocar el frontend.

Uso (en la PC, con el servidor de la Pi corriendo):

    python tools/vivo_pi.py                    # loop cada 2 s
    python tools/vivo_pi.py --once             # una pasada (para probar)
    python tools/vivo_pi.py --pi-url http://192.168.68.240:8080

En la Pi: ``python pi/servidor_vivo.py --port 8080`` (ver pi/guia_pi.md §7).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# Consola de Windows (cp1252) no soporta "→"/"°"/"·" en los prints.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

#: Archivos del contrato que se copian siempre (si la Pi los tiene).
ARCHIVOS = ("telemetry.csv", "telemetry.jsonl", "pi_health.json",
            "summary.json")


def fetch(url: str, timeout: float = 4.0, opener=urlopen) -> bytes | None:
    """GET simple; ``None`` si la Pi no responde (no levanta excepción)."""
    try:
        with opener(url, timeout=timeout) as r:
            return r.read()
    except (URLError, OSError, ValueError):
        return None


def sync_once(pi_url: str, out: Path, opener=urlopen,
              timeout: float = 4.0) -> dict | None:
    """
    Una pasada de sincronización. Devuelve el ``/status`` de la Pi (o ``None``
    si no respondió). ``opener`` es inyectable para tests.
    """
    base = pi_url.rstrip("/")
    raw = fetch(f"{base}/status", timeout, opener)
    if raw is None:
        return None
    try:
        st = json.loads(raw)
    except ValueError:
        return None
    if not st.get("ok"):
        return st
    out.mkdir(parents=True, exist_ok=True)
    (out / "vis").mkdir(exist_ok=True)
    # Sólo los archivos que EXISTEN en la Pi; los locales que ya no están se
    # borran (si no, un summary.json de un vuelo anterior queda pegado y
    # contradice los frames en vivo).
    presentes = st.get("archivos", {n: 1 for n in ARCHIVOS})
    for nombre in ARCHIVOS:
        dest = out / nombre
        if nombre not in presentes:
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            continue
        data = fetch(f"{base}/{nombre}", timeout, opener)
        if data is not None:
            dest.write_bytes(data)
    # Sólo las evidencias nuevas (por nombre + tamaño).
    for item in st.get("vis", []):
        dest = out / "vis" / item["n"]
        try:
            if dest.exists() and dest.stat().st_size == item["size"]:
                continue
        except OSError:
            pass
        data = fetch(f"{base}/vis/{item['n']}", timeout, opener)
        if data is not None:
            dest.write_bytes(data)
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pi-url", default="http://192.168.68.240:8080")
    ap.add_argument("--out", type=Path, default=Path("outputs/mission"))
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    print(f"[vivo] Pi: {args.pi_url} → {args.out} (Ctrl+C para salir)")
    while True:
        st = sync_once(args.pi_url, args.out)
        if st is None:
            print(f"[WARN] {time.strftime('%H:%M:%S')} sin conexión con la Pi "
                  f"({args.pi_url}); reintento en {args.interval:.0f} s",
                  flush=True)
        elif not st.get("ok"):
            print(f"[WARN] la Pi responde pero: {st.get('error')}", flush=True)
        else:
            u = st.get("ultimo") or {}
            salud = st.get("salud") or {}
            print(f"[{time.strftime('%H:%M:%S')}] frames={st.get('n_frames')} "
                  f"| {u.get('verdict')} · {u.get('people')} pers · "
                  f"{u.get('diag')} | temp {salud.get('temp_c')} °C "
                  f"| vis={len(st.get('vis', []))}", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
