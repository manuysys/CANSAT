"""
Detección de personas/vehículos con el NPU del AI Camera (IMX500) — CanSat LB135.

════════════════════════════════════════════════════════════════════════════
POR QUÉ EXISTE
════════════════════════════════════════════════════════════════════════════
El DPD pide reconocer "personas" entre los elementos del terreno. En la
Raspberry Pi Zero v1 (ARMv6) **no hay PyTorch ni ultralytics**, así que YOLO en
CPU no es una opción. El AI Camera (Sony IMX500) resuelve el problema por la
vía opuesta: **la red corre EN EL SENSOR** y la Pi sólo recibe las detecciones
ya calculadas (metadata de libcamera). Costo para la CPU: casi nulo.

Cómo se usa (lado pipeline):
    from cansat.imx500 import Imx500Camera
    cam = Imx500Camera(model="/usr/share/imx500-models/imx500_network_ssd_...rpk")
    cam.start()
    bgr, personas, vehiculos, cajas = cam.capture(person_ids={0}, veh_ids={1,2,3,5,7})
    ...
    cam.close()

Requisitos en la Pi: ``sudo apt install imx500-all`` (firmware + modelos
preempaquetados + post-proceso de rpicam-apps). Ver pi/guia_pi.md.

⚠ ESTADO: implementado contra la API oficial de picamera2 (IMX500 + metadata) y
  con el parser del tensor SSD cubierto por tests sintéticos, pero **no fue
  validado en hardware todavía**. La primera prueba en la Pi debe ser:
      python -m cansat.imx500 --model /usr/share/imx500-models/<modelo>.rpk
  que imprime las detecciones por frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Clases COCO (redes preempaquetadas del AI Camera: SSD MobileNet / YOLO).
COCO_PERSON: frozenset[int] = frozenset({0})
COCO_VEHICLES: frozenset[int] = frozenset({1, 2, 3, 5, 7})   # bicycle, car, moto, bus, truck
# El SSD del IMX500 usa 91 clases COCO (indexadas 1..90 con 0=background).
SSD_PERSON: frozenset[int] = frozenset({1})                  # 'person' en el dataset COCO-91
SSD_VEHICLES: frozenset[int] = frozenset({2, 4, 6, 8})       # bicycle, car, motorcycle, bus


@dataclass
class Deteccion:
    """Una detección en píxeles de la imagen final (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    def as_box(self) -> tuple[int, int, int, int, int]:
        """Formato que consume ``mission_pipeline.build_evidence``."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2), int(self.cls))


def parse_ssd_output(output: np.ndarray, score_thresh: float = 0.5,
                     max_det: int = 25) -> list[Deteccion]:
    """
    Parsea la salida típica del SSD MobileNet FPNLite del IMX500.

    Formatos aceptados (el RPK los emite como tensor plano):
      · ``(N, 7)`` o ``(1, N, 7)`` con ``[y0, x0, y1, x1, score, clase, -1]``
        en coordenadas NORMALIZADAS [0, 1] (formato del demo oficial de
        picamera2 para el SSD).
      · ``(N, 6)`` con ``[x1, y1, x2, y2, score, clase]`` ya en píxeles: se
        acepta también para no romper con re-exports distintos.

    No conoce el tamaño de la imagen: devuelve coordenadas tal cual vienen
    (normalizadas) y ``rescale()`` las pasa a píxeles.
    """
    if output is None:
        return []
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 6:
        return []

    dets: list[Deteccion] = []
    n_cols = arr.shape[1]
    for row in arr[: max(0, max_det * 4)]:
        if n_cols >= 7:
            y0, x0, y1, x1, score, cls = row[:6]
            box = (x0, y0, x1, y1)
        else:
            x1_, y1_, x2_, y2_, score, cls = row[:6]
            box = (x1_, y1_, x2_, y2_)
        if float(score) < score_thresh:
            continue
        dets.append(Deteccion(box[0], box[1], box[2], box[3],
                              float(score), int(cls)))
        if len(dets) >= max_det:
            break
    return dets


def rescale(dets: list[Deteccion], width: int, height: int,
            normalized: bool = True) -> list[Deteccion]:
    """Pasa las coordenadas a píxeles si vienen normalizadas."""
    if not normalized:
        return dets
    out = []
    for d in dets:
        out.append(Deteccion(d.x1 * width, d.y1 * height,
                             d.x2 * width, d.y2 * height, d.score, d.cls))
    return out


def contar(dets: list[Deteccion],
           person_ids: frozenset[int] = SSD_PERSON,
           veh_ids: frozenset[int] = SSD_VEHICLES,
           ) -> tuple[int, int, list[tuple[int, int, int, int, int]]]:
    """Cuenta personas/vehículos y devuelve los recuadros (formato del pipeline)."""
    personas = vehiculos = 0
    cajas: list[tuple[int, int, int, int, int]] = []
    for d in dets:
        if d.cls in person_ids:
            personas += 1
        elif d.cls in veh_ids:
            vehiculos += 1
        else:
            continue
        cajas.append(d.as_box())
    return personas, vehiculos, cajas


class Imx500Camera:
    """
    Cámara AI (IMX500) con inferencia on-sensor.

    La red se elige con ``model`` (un ``.rpk``). Los preempaquetados vienen en
    ``/usr/share/imx500-models/`` al instalar ``imx500-all``.
    """

    def __init__(self, model: str, width: int = 640, height: int = 480,
                 fps: int = 10, person_ids: frozenset[int] = SSD_PERSON,
                 veh_ids: frozenset[int] = SSD_VEHICLES):
        self.model = str(model)
        self.width, self.height, self.fps = width, height, fps
        self.person_ids, self.veh_ids = person_ids, veh_ids
        self._picam = None
        self._imx500 = None
        self._normalized = True

    # ── ciclo de vida ─────────────────────────────────────────────────── #
    def start(self) -> None:
        try:
            from picamera2 import Picamera2
            from picamera2.devices import IMX500
            from picamera2.devices.imx500 import NetworkIntrinsics
        except ImportError as e:  # pragma: no cover - sólo en la Pi
            raise ImportError(
                "El modo IMX500 necesita picamera2 con soporte AI Camera:\n"
                "  sudo apt install imx500-all python3-picamera2\n"
                "  (ver pi/guia_pi.md)"
            ) from e

        self._imx500 = IMX500(self.model)
        intr = self._imx500.network_intrinsics
        if not intr:
            intr = NetworkIntrinsics()
            intr.task = "object detection"
        self._intr = intr
        self._picam = Picamera2(self._imx500.camera_num)
        cfg = self._picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"},
            controls={"FrameRate": min(self.fps, getattr(intr, "inference_rate", 30))},
            buffer_count=8,
        )
        self._picam.configure(cfg)
        import contextlib

        with contextlib.suppress(Exception):
            self._imx500.show_network_fw_progress_bar()
        self._picam.start()
        if getattr(intr, "preserve_aspect_ratio", False):
            self._imx500.set_auto_aspect_ratio()

    def close(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None

    def stop(self) -> None:
        """Alias de ``close()`` para ser compatible con el resto del pipeline."""
        self.close()

    # ── captura ───────────────────────────────────────────────────────── #
    def capture(self) -> tuple[np.ndarray, int, int, list[tuple[int, int, int, int, int]]]:
        """
        Un frame BGR + detecciones ya contadas.

        Devuelve ``(bgr, personas, vehiculos, cajas)`` donde ``cajas`` usa el
        formato ``(x1, y1, x2, y2, clase)`` del pipeline.
        """
        if self._picam is None:
            raise RuntimeError("Imx500Camera.start() no fue llamado.")
        req = self._picam.capture_request()
        try:
            bgr = req.make_array("main")
            meta = req.get_metadata()
            personas = vehiculos = 0
            cajas: list[tuple[int, int, int, int, int]] = []
            try:
                out = self._imx500.get_outputs(meta, add_batch=True)
                dets = parse_ssd_output(out)
                if dets:
                    h, w = bgr.shape[:2]
                    # El SSD del IMX500 entrega cajas normalizadas; si algún
                    # valor supera 1.5 se asume que ya están en píxeles.
                    norm = max(d.x2 for d in dets) <= 1.5
                    self._normalized = norm
                    dets = rescale(dets, w, h, normalized=norm)
                    personas, vehiculos, cajas = contar(
                        dets, self.person_ids, self.veh_ids)
            except Exception:
                # Sin metadata de inferencia (primer frame, firmware cargando):
                # devolver el frame sin detecciones en vez de caer el vuelo.
                pass
            return np.ascontiguousarray(bgr), personas, vehiculos, cajas
        finally:
            req.release()


def _demo(argv=None) -> int:
    """Prueba de humo en la Pi: imprime detecciones durante N segundos."""
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Prueba del IMX500 on-sensor")
    ap.add_argument("--model", default="/usr/share/imx500-models/"
                                       "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="guardar un frame anotado")
    args = ap.parse_args(argv)

    cam = Imx500Camera(args.model)
    cam.start()
    print(f"[IMX500] modelo {args.model} — capturando {args.seconds:.0f} s...")
    t0 = time.time()
    n = 0
    try:
        while time.time() - t0 < args.seconds:
            bgr, per, veh, cajas = cam.capture()
            n += 1
            print(f"  frame {n}: {per} persona(s), {veh} vehículo(s), "
                  f"{len(cajas)} caja(s)")
            if args.out and n == 1:
                import cv2

                for (x1, y1, x2, y2, _c) in cajas:
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(args.out, bgr)
                print(f"  frame anotado → {args.out}")
    finally:
        cam.close()
    print(f"[OK] {n} frames capturados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
