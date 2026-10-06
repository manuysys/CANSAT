"""
Servidor HTTP mínimo (stdlib, sin dependencias) con el ÚLTIMO vuelo de la Pi
para el modo **pull en vivo** de la estación terrena (U5).

Sirve, del vuelo más reciente en ``/home/pi/vuelos/*/mission``:

  GET /status          → JSON: dir, n_frames, mtime, salud de la Pi, archivos
  GET /telemetry.csv   → CSV del contrato (34 columnas)
  GET /telemetry.jsonl → extras por frame
  GET /pi_health.json  → salud de la Pi (si existe)
  GET /summary.json    → resumen (si existe)
  GET /vis/<archivo>   → evidencia visual (jpg)
  GET /frame.jpg       → atajo a la ÚLTIMA evidencia visual

Uso (en la Pi)::

    python pi/servidor_vivo.py --port 8080
    # y en la PC:
    python tools/vivo_pi.py --pi-url http://192.168.68.240:8080

Nota: es de sólo lectura y sin autenticación: pensado para la red local del
predio. No exponerlo a Internet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# El script vive en pi/: agregar la raíz del repo para importar `cansat`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DEFAULT = Path("/home/pi/vuelos")
ARCHIVOS_FIJOS = ("telemetry.csv", "telemetry.jsonl", "pi_health.json",
                  "summary.json")


def vuelo_mas_nuevo(base: Path) -> Path | None:
    """Carpeta ``mission`` del vuelo más reciente (por mtime del CSV)."""
    candidatos = []
    for csv in base.glob("*/mission/telemetry.csv"):
        try:
            candidatos.append((csv.stat().st_mtime, csv.parent))
        except OSError:
            continue
    if not candidatos:
        return None
    return max(candidatos)[1]


def _salud() -> dict | None:
    try:
        from cansat.salud_pi import leer_temp_c, leer_throttled, leer_uptime_s

        temp = leer_temp_c()
        thr = leer_throttled()
        return {
            "temp_c": (round(temp, 1) if temp is not None else None),
            "uptime_s": (round(leer_uptime_s() or 0.0, 1)
                         if leer_uptime_s() is not None else None),
            "throttled": thr,
        }
    except Exception:
        return None


def _ultimo_paquete(mission: Path) -> dict | None:
    """Última línea del JSONL (veredicto/personas/temp) para el status."""
    jl = mission / "telemetry.jsonl"
    try:
        with jl.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            lineas = fh.read().decode("utf-8", "replace").strip().splitlines()
        if not lineas:
            return None
        pkt = json.loads(lineas[-1])
        return {k: pkt.get(k) for k in
                ("t_s", "alt_m", "verdict", "people", "vehicles", "diag",
                 "temp_c", "pi_temp_c")}
    except (OSError, ValueError):
        return None


def status(base: Path) -> dict:
    mission = vuelo_mas_nuevo(base)
    if mission is None:
        return {"ok": False, "error": "sin vuelos en la Pi"}
    try:
        st = (mission / "telemetry.csv").stat()
        n_frames = sum(1 for _ in (mission / "telemetry.jsonl").open(
            encoding="utf-8"))
    except OSError as e:
        return {"ok": False, "error": str(e)}
    vis = []
    for p in sorted((mission / "vis").glob("*.jpg"))[-80:]:
        try:
            s = p.stat()
            vis.append({"n": p.name, "size": s.st_size, "mtime": int(s.st_mtime)})
        except OSError:
            continue
    # Qué archivos fijos EXISTEN (para que el puller borre los locales que ya
    # no están en la Pi: p. ej. un summary.json viejo de otro vuelo).
    archivos = {}
    for n in ARCHIVOS_FIJOS:
        try:
            archivos[n] = (mission / n).stat().st_size
        except OSError:
            continue
    return {
        "ok": True,
        "dir": str(mission),
        "n_frames": n_frames,
        "mtime": int(st.st_mtime),
        "salud": _salud(),
        "ultimo": _ultimo_paquete(mission),
        "archivos": archivos,
        "vis": vis,
    }


def _leer_seguro(mission: Path, rel: str) -> bytes | None:
    """Lee un archivo de la misión evitando path traversal."""
    p = (mission / rel).resolve()
    if not str(p).startswith(str(mission.resolve())):
        return None
    try:
        return p.read_bytes()
    except OSError:
        return None


def _handler_clase(base: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CanSatVivo/1.0"

        def log_message(self, fmt, *args):   # silencioso (lo pide el journal)
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/status", "/"):
                body = json.dumps(status(base), ensure_ascii=False).encode()
                self._send(200, body, "application/json; charset=utf-8")
                return
            mission = vuelo_mas_nuevo(base)
            if mission is None:
                self._send(404, b'{"ok": false, "error": "sin vuelos"}',
                           "application/json")
                return
            if path == "/frame.jpg":
                js = sorted((mission / "vis").glob("*.jpg"))
                if not js:
                    self._send(404, b"sin frames", "text/plain")
                    return
                self._send(200, js[-1].read_bytes(), "image/jpeg")
                return
            rel = path.lstrip("/")
            if rel in ARCHIVOS_FIJOS or rel.startswith("vis/"):
                data = _leer_seguro(mission, rel)
                if data is None:
                    self._send(404, b"no existe", "text/plain")
                    return
                ctype = ("application/json" if rel.endswith(".json")
                         else "text/csv" if rel.endswith(".csv")
                         else "application/x-ndjson" if rel.endswith(".jsonl")
                         else "image/jpeg")
                self._send(200, data, ctype)
                return
            self._send(404, b"ruta desconocida", "text/plain")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description="Servidor del vuelo en vivo (U5)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--base", type=Path, default=BASE_DEFAULT)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), _handler_clase(args.base))
    print(f"[vivo] sirviendo {args.base} en :{args.port} "
          f"({time.strftime('%H:%M:%S')})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
