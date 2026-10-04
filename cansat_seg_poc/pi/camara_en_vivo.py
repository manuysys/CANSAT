#!/usr/bin/env python3
"""
Vista en vivo de la AI Camera para AJUSTAR EL FOCO a mano — MJPEG por HTTP.

Sirve en http://<ip-de-la-pi>:8080 una página con el stream de la cámara y el
número de nitidez (varianza del Laplaciano) dibujado encima: girás el anillo
del lente hasta que ese número sea máximo.

Usa obturación corta (8 ms) para que no haya motion blur al mover la cámara;
si la escena queda oscura, subí --gain.

Uso (en la Pi):
    venv/bin/python pi/camara_en_vivo.py            # puerto 8080
    venv/bin/python pi/camara_en_vivo.py --port 8081
"""
from __future__ import annotations

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from picamera2 import Picamera2


def nitidez(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main() -> int:
    ap = argparse.ArgumentParser(description="Vista en vivo para ajustar el foco")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--shutter", type=int, default=8000, help="µs (anti motion-blur)")
    ap.add_argument("--gain", type=float, default=16.0)
    args = ap.parse_args()

    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "BGR888"},
        controls={"FrameRate": 10, "ExposureTime": args.shutter,
                  "AnalogueGain": args.gain},
        buffer_count=4,
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.5)                       # primer frame

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:         # noqa: N802
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = cam.capture_array("main")
                        n = nitidez(frame)
                        color = ((0, 220, 0) if n >= 50 else
                                 (0, 200, 255) if n >= 15 else (0, 0, 255))
                        cv2.putText(frame, f"nitidez {n:5.1f}",
                                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, color, 2, cv2.LINE_AA)
                        ok, jpg = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not ok:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                         + jpg.tobytes() + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path in ("/", "/index.html"):
                html = (
                    "<html><head><title>Foco AI Camera</title></head>"
                    "<body style='background:#0b0e14;color:#eee;"
                    "font-family:monospace;text-align:center'>"
                    "<h3>Ajuste de foco — girá el anillo del lente hasta que "
                    "'nitidez' sea MÁXIMA (verde &gt;50)</h3>"
                    "<img src='/stream.mjpg' style='max-width:95vw;border:1px "
                    "solid #333;border-radius:8px'>"
                    "<p style='color:#8b9aab'>rojo &lt;15 · amarillo 15-50 · "
                    "verde &gt;50 · obturación 8 ms (anti blur)</p>"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a) -> None:   # silencio
            pass

    print(f"Vista en vivo: http://0.0.0.0:{args.port}  (Ctrl+C para salir)")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
