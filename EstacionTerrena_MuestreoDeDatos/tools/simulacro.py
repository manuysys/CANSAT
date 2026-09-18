#!/usr/bin/env python3
"""simulacro.py — Ensayo end-to-end SIN hardware de la Estación Terrena.

Reproduce una misión (grabada o sintética) sobre una carpeta "viva" que
web_server.py sirve como si fuera el pipeline escribiendo en tiempo real:
copia imágenes/contrato y hace stream de telemetry.csv a la cadencia real
(o acelerada), con anomalías inyectables para entrenar al operador y
probar la degradación del dashboard.

Uso típico (dos terminales):
    python tools/simulacro.py --dst /tmp/sim-live --velocidad 4
    python web_server.py --root /tmp/sim-live --port 8004
    # abrir http://localhost:8004 → botón Live de la escena 3D activado

Anomalías:
    --dropout SEGS     pausa de escritura (simula corte del enlace TLM)
    --rafaga N         inyecta N filas extra con alert=1 y daño alto al 60%
    --trunco-en FILA   borra el CSV en esa fila (prueba recuperación)
    --sin-postvuelo    no publica entrega/summary.json (vista Post degradada)
"""
import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[simulacro {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preparar_dst(src: Path, dst: Path, sin_postvuelo: bool) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ["outputs/mission", "entrega", "web", "web-app"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)
    miss = src / "outputs" / "mission"
    if not (miss / "telemetry.csv").exists():
        sys.exit(f"[simulacro] FALTA {miss / 'telemetry.csv'} — generá una con tools/make_demo_mission.py")
    # todo el material menos el CSV de telemetría (ese se stream-ea)
    for p in miss.rglob("*"):
        if p.is_file() and p.name != "telemetry.csv":
            rel = p.relative_to(src)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)
    for p in (src / "entrega").rglob("*"):
        if p.is_file() and p.name != "summary.json":
            rel = p.relative_to(src)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)
    log(f"destino preparado en {dst} (post-vuelo: {'NO' if sin_postvuelo else 'se publica al aterrizar'})")


def leer_filas(src: Path):
    with open(src / "outputs" / "mission" / "telemetry.csv", newline="") as f:
        rdr = csv.reader(f)
        head = next(rdr)
        rows = [r for r in rdr if r]
    return head, rows


def fila_rafaga(base: list, head: list, rng: random.Random, k: int) -> list:
    idx = {c: i for i, c in enumerate(head)}
    r = list(base)
    r[idx["src"]] = f"cap_{9000 + k:04d}"
    r[idx["alert"]] = "1"
    r[idx["danado_pct"]] = f"{rng.uniform(55, 90):.1f}"
    r[idx["diag"]] = "INUNDACION SEVERA"
    r[idx["verdict"]] = "ALTO ESTRÉS URBANO"
    r[idx["sample_pri"]] = "HIGH"
    r[idx["sample_score"]] = f"{rng.uniform(0.7, 0.95):.2f}"
    r[idx["t_s"]] = f"{float(base[idx['t_s']]) + 1.5 * (k + 1):.1f}"
    r[idx["alt_m"]] = f"{max(1.0, float(base[idx['alt_m']]) - 2 * (k + 1)):.1f}"
    return r


def escribir(path: Path, head: list, rows: list, append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        w = csv.writer(f)
        if not append:
            w.writerow(head)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT), help="raíz del proyecto con la misión grabada")
    ap.add_argument("--dst", default="/tmp/sim-live", help="carpeta viva que sirve web_server.py")
    ap.add_argument("--velocidad", type=float, default=1.0, help="multiplicador de cadencia (4 = 4×)")
    ap.add_argument("--seed", type=int, default=135)
    ap.add_argument("--dropout", type=float, default=0, help="segundos de corte TLM a mitad del stream")
    ap.add_argument("--rafaga", type=int, default=0, help="filas de alerta extra inyectadas al 60%%")
    ap.add_argument("--trunco-en", type=int, default=-1, help="borra el CSV en esta fila (recuperación)")
    ap.add_argument("--sin-postvuelo", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    preparar_dst(src, dst, args.sin_postvuelo)
    head, rows = leer_filas(src)
    ix = {c: i for i, c in enumerate(head)}
    csv_live = dst / "outputs" / "mission" / "telemetry.csv"

    if args.rafaga:
        punto = int(len(rows) * 0.6)
        extra = [fila_rafaga(rows[min(punto + k, len(rows) - 1)], head, rng, k) for k in range(args.rafaga)]
        rows = rows[:punto] + extra + rows[punto:]
        log(f"ráfaga inyectada: {args.rafaga} filas alert=1 desde la posición {punto}")

    escribir(csv_live, head, [])
    prev_alt = -1.0
    apogeo = False
    escritos = 0
    for i, row in enumerate(rows):
        if i > 0:
            dt = max(0.2, float(row[ix["t_s"]]) - float(rows[i - 1][ix["t_s"]]))
            time.sleep(dt / max(0.1, args.velocidad))
        escribir(csv_live, head, [row], append=True)
        escritos += 1
        alt = float(row[ix["alt_m"]])
        if not apogeo and prev_alt >= 0 and alt < prev_alt:
            apogeo = True
            log(f"APOGEO: fin del ascenso en {row[ix['src']]} ({prev_alt:.0f} m)")
        prev_alt = alt
        if args.dropout and i == len(rows) // 2:
            log(f"DROPOUT TLM: {args.dropout:.0f} s sin escribir…")
            time.sleep(args.dropout)
            log("enlace restablecido")
        if args.trunco_en == i:
            log("TRUNCO: CSV borrado (el dashboard debe degradar y recuperar)")
            csv_live.unlink(missing_ok=True)
            time.sleep(3)
            escribir(csv_live, head, rows[: i + 1])
            log("CSV restaurado con historial completo, reanudando stream")
        log(f"frame {row[ix['src']]} · t={row[ix['t_s']]} s · alt={alt:.1f} m · alert={row[ix['alert']]}")

    if not args.sin_postvuelo and (src / "entrega" / "summary.json").exists():
        shutil.copy2(src / "entrega" / "summary.json", dst / "entrega" / "summary.json")
        log("summary.json publicado → post-vuelo disponible")
    log(f"SIMULACRO COMPLETO: {escritos} frames escritos en {dst}")


if __name__ == "__main__":
    main()
