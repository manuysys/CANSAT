"""
Segmentación semántica en el NPU del IMX500 (DeepLabV3+, 21 clases VOC).

Medido en la placa (2026-10-04) con ``imx500_network_deeplabv3plus.rpk``:
``task=segmentation`` y UN solo tensor de salida ``(1, 320, 320)`` float32 que
YA es el mapa de clases (argmax) con índices VOC 0-20 (persona = 15). En una
escena con una persona al frente: 80.7 % fondo / 19.3 % persona, sin gastar
CPU (la inferencia corre en el sensor).

El mIoU oficial del modelo es 0.7214 en VOC2012 (se declara como referencia;
NO es una medición nuestra).

Uso::

    python -m cansat.imx500_seg --seconds 10
    python -m cansat.imx500_seg --seconds 5 --guardar salida.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cansat.imx500 import controles_camara

MODELO_DEFAULT = ("/usr/share/imx500-models/"
                  "imx500_network_deeplabv3plus.rpk")

#: 21 clases de VOC2012 (orden del dataset; persona = 15).
VOC_CLASSES: tuple[str, ...] = (
    "fondo", "avion", "bicicleta", "pajaro", "bote", "botella", "bus",
    "auto", "gato", "silla", "vaca", "mesa", "perro", "caballo", "moto",
    "persona", "planta", "oveja", "sofa", "tren", "tv",
)
VOC_PERSON: int = 15
#: Vehículos de VOC (los que interesan para la misión).
VOC_VEHICLES: frozenset[int] = frozenset({1, 2, 4, 6, 7, 14, 19})

#: Colores BGR por clase (el fondo no se pinta).
VOC_COLORS: dict[int, tuple[int, int, int]] = {
    1: (255, 128, 0), 2: (0, 200, 255), 3: (0, 128, 255), 4: (255, 0, 128),
    5: (128, 0, 255), 6: (0, 0, 255), 7: (0, 64, 255), 8: (255, 0, 0),
    9: (160, 160, 0), 10: (128, 255, 0), 11: (0, 255, 128), 12: (255, 255, 0),
    13: (64, 0, 255), 14: (255, 0, 64), 15: (0, 0, 255), 16: (0, 255, 0),
    17: (255, 200, 200), 18: (200, 200, 0), 19: (255, 0, 255),
    20: (128, 128, 128),
}
# La persona se resalta en rojo puro (se pisa la entrada 15 de arriba).
VOC_COLORS[VOC_PERSON] = (0, 0, 255)

#: 5 clases del terreno de vuelo (LoveDA remapeado) — las del rpk PROPIO.
LOVE_CLASSES: tuple[str, ...] = ("vegetacion", "edificio", "agua", "suelo",
                                 "otro")
LOVE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (60, 180, 60),     # vegetacion
    1: (0, 0, 255),       # edificio (rojo)
    2: (255, 120, 0),     # agua (azul)
    3: (160, 160, 160),   # suelo
    4: (70, 70, 70),      # otro
}

PALETAS: dict[str, dict[int, tuple[int, int, int]]] = {
    "voc": VOC_COLORS, "love": LOVE_COLORS,
}


def mask_desde_salida(outputs) -> np.ndarray | None:
    """
    Salida del rpk → máscara 2D de clases.

    Acepta:
      · ``(1, H, W)``/``(H, W)``: mapa de clases ya en argmax (DeepLabV3+ del
        apt);
      · ``(C, H, W)``/``(1, C, H, W)``: logits por clase (nuestro rpk del
        tiny) → argmax sobre el eje de canales.
    """
    if not outputs:
        return None
    arr = np.asarray(outputs[0])
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.argmax(0)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    return None


def cobertura(mask: np.ndarray) -> dict[int, float]:
    """Porcentaje del frame por clase (sólo las clases presentes)."""
    m = np.asarray(mask).astype(np.int32)
    if m.ndim != 2 or m.size == 0:
        return {}
    vals, counts = np.unique(m, return_counts=True)
    total = float(m.size)
    return {int(v): 100.0 * float(c) / total
            for v, c in zip(vals, counts, strict=True)}


def resumen(mask: np.ndarray, top: int = 3, clases: str = "voc") -> dict:
    """
    Resumen del frame. Modo ``voc``: % persona/vehículos (clases VOC del
    DeepLabV3+ del apt). Modo ``love``: cobertura de las 5 clases de terreno
    (nuestro rpk del tiny).
    """
    cob = cobertura(mask)
    if clases == "love":
        orden = sorted(((pct, c) for c, pct in cob.items()), reverse=True)
        return {
            "clases": {LOVE_CLASSES[c]: round(pct, 1)
                       for c, pct in sorted(cob.items())},
            "top": [{"clase": LOVE_CLASSES[c], "pct": round(pct, 1)}
                    for pct, c in orden[:top]],
        }
    persona = cob.get(VOC_PERSON, 0.0)
    vehiculos = sum(cob.get(c, 0.0) for c in VOC_VEHICLES)
    orden = sorted(((pct, c) for c, pct in cob.items() if c != 0), reverse=True)
    top_clases = [{"clase": VOC_CLASSES[c], "pct": round(pct, 1)}
                  for pct, c in orden[:top]]
    return {
        "persona_pct": round(persona, 1),
        "vehiculos_pct": round(vehiculos, 1),
        "top": top_clases,
    }


def dibujar(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
            colores: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    """Overlay de la máscara (reescalada al frame) sobre el frame BGR."""
    import cv2

    h, w = frame.shape[:2]
    m = np.asarray(mask).astype(np.uint8)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(frame)
    for cls, bgr in (colores or VOC_COLORS).items():
        color[m == cls] = bgr
    pintable = m != 0
    out = frame.copy()
    out[pintable] = (frame[pintable].astype(np.float32) * (1 - alpha)
                     + color[pintable].astype(np.float32) * alpha).astype(
                         np.uint8)
    return out


class Imx500Segmenter:
    """Cámara AI (IMX500) con DeepLabV3+ on-sensor."""

    def __init__(self, model: str = MODELO_DEFAULT, width: int = 640,
                 height: int = 480, fps: int = 10,
                 shutter_us: int | None = None, gain: float | None = None,
                 clases: str = "voc"):
        self.model = str(model)
        self.width, self.height, self.fps = width, height, fps
        self.shutter_us, self.gain = shutter_us, gain
        # "voc": DeepLabV3+ del apt (máscara en argmax); "love": nuestro rpk
        # del tiny (logits de 5 clases de terreno).
        self.clases = clases if clases in PALETAS else "voc"
        self._picam = None
        self._imx500 = None
        self._intr = None

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
            from picamera2.devices import IMX500
            from picamera2.devices.imx500 import NetworkIntrinsics
        except ImportError as e:  # pragma: no cover - sólo en la Pi
            raise ImportError(
                "La segmentación IMX500 necesita picamera2 con AI Camera:\n"
                "  sudo apt install imx500-all python3-picamera2\n"
                "  (ver pi/guia_pi.md)"
            ) from e

        self._imx500 = IMX500(self.model)
        intr = self._imx500.network_intrinsics
        if not intr:
            intr = NetworkIntrinsics()
            intr.task = "segmentation"
        self._intr = intr
        self._picam = Picamera2(self._imx500.camera_num)
        cfg = self._picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"},
            controls=controles_camara(
                min(self.fps, getattr(intr, "inference_rate", 30) or 30),
                self.shutter_us, self.gain),
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
        """Alias de ``close()``."""
        self.close()

    def capture(self) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """
        Un frame BGR + la máscara VOC (o ``None`` si aún no hay inferencia).

        Devuelve ``(bgr, mask, resumen)``; ``resumen`` es el de ``resumen()``
        (o vacío si no hay máscara todavía).
        """
        if self._picam is None:
            raise RuntimeError("Imx500Segmenter.start() no fue llamado.")
        req = self._picam.capture_request()
        try:
            bgr = req.make_array("main")
            meta = req.get_metadata()
            mask = None
            try:
                out = self._imx500.get_outputs(meta, add_batch=True)
                mask = mask_desde_salida(out)
            except Exception:
                # Primer frame / firmware cargando: frame sin máscara.
                mask = None
            return bgr, mask, (resumen(mask, clases=self.clases)
                               if mask is not None else {})
        finally:
            req.release()


def _demo() -> int:
    ap = argparse.ArgumentParser(
        description="DeepLabV3+ on-sensor: % persona/vehículos por frame")
    ap.add_argument("--model", default=MODELO_DEFAULT)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--shutter", type=int, default=None,
                    help="ExposureTime en µs (p. ej. 8000 con luz baja)")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--clases", choices=("voc", "love"), default="voc",
                    help="voc = DeepLabV3+ del apt; love = nuestro rpk del tiny")
    ap.add_argument("--guardar", default=None,
                    help="PNG del primer frame con overlay (opcional)")
    args = ap.parse_args()

    if args.guardar:
        # Preload: el primer `import cv2` en la Zero tarda ~20 s (¡no debe
        # caer dentro de la ventana de medición de fps!).
        import cv2

    seg = Imx500Segmenter(args.model, shutter_us=args.shutter, gain=args.gain,
                          clases=args.clases)
    print(f"[IMX500-seg] modelo: {Path(seg.model).name}")
    seg.start()
    # Warm-up: el primer frame espera la carga de la red en el sensor (medido:
    # ~25 s la primera vez). No se cuenta en el fps.
    t_warm = time.time()
    mask = None
    while mask is None and time.time() - t_warm < 60:
        _, mask, _ = seg.capture()
    print(f"  [warm-up] primera máscara en {time.time() - t_warm:.1f} s")
    t0 = time.time()
    frames = 0
    con_mask = 0
    guardado = False
    try:
        while time.time() - t0 < args.seconds:
            bgr, mask, res = seg.capture()
            frames += 1
            if mask is not None:
                con_mask += 1
                if frames <= 2 or frames % 10 == 0:
                    if args.clases == "love":
                        print(f"  frame {frames}: {res['clases']}")
                    else:
                        print(f"  frame {frames}: persona {res['persona_pct']}% "
                              f"· vehículos {res['vehiculos_pct']}% · "
                              f"top {res['top']}")
                if args.guardar and not guardado:
                    import cv2

                    cv2.imwrite(args.guardar,
                                dibujar(bgr, mask, colores=PALETAS[args.clases]))
                    print(f"  [guardado] {args.guardar}")
                    guardado = True
    finally:
        seg.close()
    dur = time.time() - t0
    fps = frames / dur if dur > 0 else 0.0
    print(f"[OK] {frames} frames en {dur:.1f} s ({fps:.1f} fps), "
          f"{con_mask} con máscara.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_demo())
