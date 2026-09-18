"""
CanSat La Base — Simulador UART LB135 (vuelo fake sin hardware).

Emite paquetes con el formato del contrato (``cansat/protocol.py``), que es el
mismo que parsea ``uart_listener.py`` y el mismo que emite
``mission_pipeline.py``. Antes los tres tenían formatos incompatibles.

⚠ La versión anterior calculaba ``veg, bui, wat, bare, oth, dom, usi, ndvi,
  vcode`` y **no los usaba**: el paquete que imprimía tenía sólo 6 campos. Todo
  ese cálculo era código muerto. Acá los campos viajan de verdad, y los
  porcentajes de terreno salen de una historia coherente (no de ``random``
  suelto), así el simulacro sirve para probar la estación terrena.

Uso:
    python sim_uart.py --frames 30                      # a stdout
    python sim_uart.py --frames 30 --format v1          # formato legacy
    python sim_uart.py --frames 30 | python uart_listener.py --stdin
    python sim_uart.py --frames 30 --port COM3          # a un puerto real
    python sim_uart.py --frames 30 --escenario inundacion
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cansat import indices as IDX
from cansat import protocol as PROTO

# Historias de terreno por escenario: (veg, bui, wat, bare, oth) en %.
# Sumando 100. Sirven para ejercitar cada rama del diagnóstico.
ESCENARIOS: dict[str, list[tuple[float, float, float, float, float]]] = {
    "rural": [(78, 1, 6, 12, 3), (72, 2, 8, 15, 3), (65, 3, 10, 18, 4),
              (58, 4, 12, 22, 4), (50, 6, 14, 26, 4)],
    "urbano": [(12, 55, 2, 20, 11), (8, 62, 2, 18, 10), (5, 68, 1, 16, 10),
               (15, 48, 3, 24, 10), (10, 58, 2, 20, 10)],
    "inundacion": [(40, 8, 25, 20, 7), (25, 10, 40, 18, 7), (10, 12, 58, 14, 6),
                   (4, 14, 68, 10, 4), (2, 16, 74, 6, 2)],
    "sismo": [(30, 25, 3, 30, 12), (18, 30, 3, 36, 13), (8, 34, 2, 44, 12),
              (4, 30, 2, 54, 10), (2, 26, 2, 62, 8)],
    "mixto": [(70, 3, 8, 15, 4), (45, 15, 6, 26, 8), (20, 38, 4, 28, 10),
              (8, 58, 3, 20, 11), (35, 20, 25, 15, 5)],
}


# Daño estructural máximo (%) que alcanza cada escenario al final del descenso.
DANIO_POR_ESCENARIO: dict[str, float] = {
    "inundacion": 55.0,   # → INUNDACION SEVERA (agua > 25 % + consenso de daño)
    "sismo": 45.0,        # → POSIBLE SISMO/VIENTO
    "rural": 0.0,
    "urbano": 0.0,
    "mixto": 0.0,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Simulador UART LB135")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--port", default=None, help="puerto serial real (default: stdout)")
    ap.add_argument("--baud", type=int, default=PROTO.BAUD)
    ap.add_argument("--format", choices=("v2", "v1"), default="v2")
    ap.add_argument("--escenario", choices=sorted(ESCENARIOS), default="mixto")
    ap.add_argument("--apogee", type=float, default=250.0)
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="segundos reales entre paquetes (0 = lo más rápido posible)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lat", type=float, default=-34.6075,
                    help="latitud del punto de eyección (default: El Palomar)")
    ap.add_argument("--lon", type=float, default=-58.6126,
                    help="longitud del punto de eyección")
    ap.add_argument("--drift-deg", type=float, default=0.002,
                    help="deriva total del descenso en grados (~220 m hacia el E)")
    ap.add_argument("--no-checksum", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="fracción de paquetes que se pierden (prueba de enlace)")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    historia = ESCENARIOS[args.escenario]

    out = None
    if args.port:
        try:
            import serial
        except ImportError:
            sys.stderr.write("[ERROR] Falta pyserial: pip install pyserial\n")
            return 1
        out = serial.Serial(args.port, args.baud, timeout=1)
        print(f"[UART] abierto {args.port} @ {args.baud}", file=sys.stderr)

    enviados = perdidos = 0
    try:
        for i in range(args.frames):
            t = i * max(args.interval, 0.1)
            alt, p, temp = PROTO.sim_atmo(t, args.apogee, args.speed)

            # Interpola la historia según el progreso del descenso.
            frac = min(1.0, i / max(1, args.frames - 1))
            pos = frac * (len(historia) - 1)
            lo, hi = math.floor(pos), min(len(historia) - 1, math.ceil(pos))
            w = pos - lo
            pcts = [a * (1 - w) + b * w for a, b in zip(historia[lo], historia[hi], strict=False)]
            # Ruido pequeño, manteniendo la suma en 100.
            pcts = [max(0.0, v + rng.uniform(-1.5, 1.5)) for v in pcts]
            pcts = IDX.normalize_pcts(pcts)

            env_usi = IDX.usi(pcts)
            env_gvi = IDX.gvi(pcts)
            v = IDX.verdict(pcts)
            # Los escenarios de desastre inyectan daño estructural creciente con
            # el descenso, para ejercitar la ruta de alerta del listener y de la
            # estación terrena. Sin esto "inundacion" nunca alertaba (el agua
            # extensa sin daño es "AGUA EXTENSA (lago/rio)", que no es alerta).
            dan = DANIO_POR_ESCENARIO.get(args.escenario, 0.0) * frac
            _diag, alert = IDX.diagnose(pcts, dan, dan * 0.9, dan * 0.8)

            pkt = PROTO.Packet(
                pkt=i, t_s=t, alt_m=alt, p_hPa=p, temp_C=temp,
                veg=pcts[0], bui=pcts[1], wat=pcts[2], bare=pcts[3], oth=pcts[4],
                dom=int(max(range(5), key=lambda k: pcts[k])),
                usi=env_usi, gvi=env_gvi, vcode=IDX.verdict_code(v),
                personas=rng.randint(0, 3), vehiculos=rng.randint(0, 5),
                alert=alert,
                danado_pct=dan,
                # Trayectoria GPS simulada: el descenso deriva hacia el este
                # (viento típico) desde el punto de eyección. Con --lat/--lon se
                # fija el origen; sin ellos, un punto genérico de El Palomar.
                lat=args.lat + frac * args.drift_deg,
                lon=args.lon + frac * args.drift_deg * 1.3,
                # Humedad relativa simulada: sube con la altura y con el
                # escenario de inundacion (aire saturado cerca del agua).
                hum_pct=min(98.0, 45.0 + 18.0 * frac
                            + (12.0 if args.escenario == "inundacion" else 0.0)
                            + rng.uniform(-2.0, 2.0)),
            )
            line = (PROTO.format_packet(pkt, with_checksum=not args.no_checksum)
                    if args.format == "v2" else PROTO.format_legacy_v1(pkt))

            if args.dropout > 0 and rng.random() < args.dropout:
                perdidos += 1
                continue

            data = (line + "\n").encode()
            if out:
                out.write(data)
            else:
                print(line, flush=True)
            enviados += 1

            if args.interval > 0:
                time.sleep(args.interval)
    finally:
        if out:
            out.close()

    print(f"[OK] simulación terminada: {enviados} paquetes enviados"
          + (f", {perdidos} perdidos a propósito (--dropout)" if perdidos else "")
          + f" · escenario {args.escenario}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
