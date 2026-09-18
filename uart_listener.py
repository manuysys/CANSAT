"""
CanSat La Base — Listener UART LB135 (lado Pi).

Lee paquetes de la ESP32 y los loguea a JSONL. El formato del paquete está
definido en ``cansat/protocol.py`` (FUENTE ÚNICA).

════════════════════════════════════════════════════════════════════════════
POR QUÉ SE REESCRIBIÓ
════════════════════════════════════════════════════════════════════════════
Había **tres formatos incompatibles** en el repo:

  · ``mission_pipeline.py`` emitía ``$LB135,t,alt,p,temp,<5 terreno>,dom,usi,
    ndvi,vcode,personas,vehiculos,alert`` — 18 campos, con ``$``.
  · Este listener parseaba ``LB135,pkt,p_base,p_abs,alt,temp`` — 6 campos,
    **sin** ``$``, y exigía ``parts[0].startswith("LB135")``.
  · ``sim_uart.py`` emitía el de 6 campos y calculaba 9 variables que no usaba.

O sea: **el listener no podía leer lo que emitía el pipeline**. Y el firmware
de la ESP32 no está en el repo, así que el formato que realmente viaja por el
aire era un cuarto desconocido.

Ahora los tres usan ``cansat/protocol.py``, que además parsea tolerante: acepta
v2 (19 campos con versión y checksum), v1 (18 campos, el histórico del
pipeline) y el atmosférico de 6 campos (el del firmware). Nunca lanza.

Se agregan cosas que faltaban y en vuelo importan:
  · log en modo **append** (antes ``"w"``: cada corrida borraba la anterior);
  · reconexión si el puerto serial se cae (la ESP32 se resetea al eyectar);
  · contador de paquetes perdidos por número de paquete y de checksums malos;
  · escritura de la última altitud conocida a un archivo de estado, para que
    ``mission_pipeline.py`` pueda usarla en vez de simular la atmósfera.

Uso:
    python uart_listener.py --port /dev/serial0        # Raspberry
    python uart_listener.py --port COM3                # Windows (prueba)
    python sim_uart.py --frames 10 | python uart_listener.py --stdin
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cansat import protocol as PROTO


def open_serial(port: str, baud: int):
    try:
        import serial
    except ImportError:
        sys.stderr.write(
            "[ERROR] Falta pyserial:  pip install pyserial\n"
            "        (no estaba declarado en requirements.txt)\n")
        raise SystemExit(1) from None
    return serial.Serial(port, baud, timeout=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Listener UART LB135")
    ap.add_argument("--port", default=None, help="/dev/serial0 en la Pi, COM3 en Windows")
    ap.add_argument("--baud", type=int, default=PROTO.BAUD)
    ap.add_argument("--stdin", action="store_true", help="leer de stdin (simulador)")
    ap.add_argument("--out", default="uart_log.jsonl")
    ap.add_argument("--state", default="outputs/uart_state.json",
                    help="última lectura (escrita a 1 Hz), para que "
                         "mission_pipeline.py la use con --uart-state")
    ap.add_argument("--append", action="store_true", default=True,
                    help="no truncar el log de una corrida anterior (default)")
    ap.add_argument("--overwrite", action="store_true",
                    help="truncar el log en vez de appendear")
    ap.add_argument("--quiet", action="store_true", help="no imprimir cada paquete")
    ap.add_argument("--max-reconnect", type=int, default=30,
                    help="reintentos si el puerto serial se cae (0 = no reintentar)")
    args = ap.parse_args(argv)

    if not args.port and not args.stdin:
        print("[ERROR] Especificá --port o --stdin")
        return 1

    mode = "w" if args.overwrite else "a"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if mode == "a" and Path(args.out).is_file():
        with Path(args.out).open(encoding="utf-8") as fh:
            prev = sum(1 for _ in fh)
        print(f"[UART] appendeando a {args.out} ({prev} paquetes previos)")

    stream = sys.stdin if args.stdin else open_serial(args.port, args.baud)
    if not args.stdin:
        print(f"[UART] escuchando {args.port} @ {args.baud} 8N1")

    n_ok = n_bad = n_cs_bad = 0
    ultimo_pkt: int | None = None
    perdidos = 0
    reconnects = 0
    last_state_write = 0.0

    with open(args.out, mode, encoding="utf-8") as log:
        try:
            while True:
                try:
                    raw = stream.readline() if hasattr(stream, "readline") else None
                    if raw is None:
                        line = next(iter(stream), None)
                    else:
                        line = raw
                except Exception as e:
                    if args.stdin or args.max_reconnect <= 0:
                        print(f"[UART] fin del stream ({type(e).__name__}: {e})")
                        break
                    reconnects += 1
                    if reconnects > args.max_reconnect:
                        print(f"[UART] {reconnects - 1} reintentos agotados. Saliendo.")
                        break
                    print(f"[UART] puerto caído ({e}); reintento "
                          f"{reconnects}/{args.max_reconnect} en 1 s")
                    time.sleep(1.0)
                    try:
                        stream = open_serial(args.port, args.baud)
                    except Exception as e2:
                        print(f"[UART] no se pudo reabrir: {e2}")
                    continue

                if not line:
                    if args.stdin:
                        break
                    continue

                text = line if isinstance(line, str) else line.decode("utf-8", "ignore")
                text = text.strip()
                if not text:
                    continue
                if text.startswith("#"):
                    if not args.quiet:
                        print(f"[DBG] {text}")
                    continue

                pkt = PROTO.parse(text)
                if pkt is None:
                    n_bad += 1
                    if not args.quiet:
                        print(f"[?] ignorado ({len(text)} bytes): {text[:80]}")
                    continue

                if pkt.checksum_ok is False:
                    n_cs_bad += 1
                    if not args.quiet:
                        print(f"[!] checksum inválido: {text[:80]}")
                    # Se registra igual: en vuelo preferimos datos dudosos a nada,
                    # pero queda marcado para filtrarlo en post-vuelo.

                # Detección de pérdida por número de paquete.
                # ⚠ ARREGLADO: antes era ``pkt.pkt > 0`` y
                # ``ultimo_pkt = pkt.pkt if pkt.pkt else ultimo_pkt``, así que
                # el pkt=0 (válido, primer paquete) quedaba como "sin estado" y
                # la secuencia 0,2 no reportaba el paquete 1 perdido.
                if ultimo_pkt is not None and pkt.pkt > ultimo_pkt + 1:
                    gap = pkt.pkt - ultimo_pkt - 1
                    perdidos += gap
                    print(f"[!] {gap} paquete(s) perdido(s) entre "
                          f"{ultimo_pkt} y {pkt.pkt} (enlace de radio)")

                pkt.rx_wall = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                pkt.raw = text
                log.write(json.dumps(pkt.to_dict(), ensure_ascii=False) + "\n")
                log.flush()
                n_ok += 1
                ultimo_pkt = pkt.pkt

                if not args.quiet:
                    extra = "" if pkt.checksum_ok is not False else " [CS!]"
                    if pkt.version == 0:
                        print(f"[{n_ok}] pkt={pkt.pkt} alt={pkt.alt_m:.1f} m "
                              f"T={pkt.temp_C:.1f} °C (atmosférico ESP32){extra}")
                    else:
                        dano = (f" dano={pkt.danado_pct:.1f}%"
                                if pkt.version >= 2 and pkt.danado_pct > 0 else "")
                        gps = (f" gps={pkt.lat:.5f},{pkt.lon:.5f}"
                               if pkt.has_fix() else "")
                        hum = (f" hum={pkt.hum_pct:.0f}%"
                               if pkt.version >= 2 and pkt.hum_pct > 0 else "")
                        print(f"[{n_ok}] pkt={pkt.pkt} t={pkt.t_s:.1f} "
                              f"alt={pkt.alt_m:.1f} m vcode={pkt.vcode} "
                              f"alert={pkt.alert}{dano}{gps}{hum}{extra}")

                # Estado para que el pipeline lo lea sin acoplarse al serial.
                # Throttle a 1 Hz: reescribirlo en cada paquete desgasta la SD
                # sin aportar nada (el fallback tolera hasta 10 s de antigüedad).
                now = time.monotonic()
                if args.state and (now - last_state_write) >= 1.0:
                    last_state_write = now
                    try:
                        Path(args.state).parent.mkdir(parents=True, exist_ok=True)
                        Path(args.state).write_text(json.dumps({
                            "alt_m": pkt.alt_m, "p_hPa": pkt.p_hPa,
                            "temp_C": pkt.temp_C, "pkt": pkt.pkt,
                            "t_s": pkt.t_s, "rx_wall": pkt.rx_wall,
                            "version": pkt.version,
                            "lat": pkt.lat, "lon": pkt.lon, "hum_pct": pkt.hum_pct,
                        }, ensure_ascii=False), encoding="utf-8")
                    except OSError:
                        pass
        except KeyboardInterrupt:
            print("\n[UART] interrumpido por el operador.")
        finally:
            if hasattr(stream, "close") and stream is not sys.stdin:
                with contextlib.suppress(Exception):
                    stream.close()

    print(f"[OK] {n_ok} paquetes válidos → {args.out}")
    if n_bad:
        print(f"     {n_bad} línea(s) no reconocida(s)")
    if n_cs_bad:
        print(f"     {n_cs_bad} con checksum inválido (registrados igual, marcados)")
    if perdidos:
        print(f"     {perdidos} paquete(s) perdido(s) por radio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
