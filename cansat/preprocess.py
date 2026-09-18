"""
Preprocesado de imagen — FUENTE ÚNICA.

Antes había **62 definiciones** de ``prep``/``preprocess``/``prep_tile``
repartidas en 24 archivos, con ``MEAN``/``STD`` de ImageNet copiados
literalmente y variantes sutiles (320 vs 512, con/sin ``np.newaxis``,
BGR vs RGB, ``IMREAD_UNCHANGED`` o no). Cualquier cambio de normalización
había que hacerlo a mano en 24 lugares.

Todo lo que alimente un ONNX de este proyecto DEBE pasar por acá.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - permite importar sin OpenCV instalado
    cv2 = None

# ── Normalización ImageNet (la que usó TODO el entrenamiento) ─────────── #
MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MEAN_LIST = [0.485, 0.456, 0.406]
STD_LIST = [0.229, 0.224, 0.225]

# Tamaño por defecto del modelo de vuelo (DeepLabV3+ MobileNetV2).
DEFAULT_IMG_SIZE: int = 320


def _require_cv2():
    if cv2 is None:
        raise ImportError(
            "opencv-python no está instalado. En la Raspberry Pi:\n"
            "    pip install opencv-python-headless\n"
            "NOTA: si vas a usar super-resolución (cv2.dnn_superres) necesitás\n"
            "      opencv-contrib-python-headless, no el headless común."
        )
    return cv2


def normalize(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB [0,255] → float32 normalizado, mismo layout HWC."""
    return (rgb.astype(np.float32) / 255.0 - MEAN) / STD


def to_chw(arr: np.ndarray, batch: bool = True) -> np.ndarray:
    """HWC → CHW, opcionalmente con dimensión de batch adelante (NCHW)."""
    out = arr.transpose(2, 0, 1)
    return out[np.newaxis, ...] if batch else out


def preprocess_bgr(
    bgr: np.ndarray, img_size: int = DEFAULT_IMG_SIZE, batch: bool = True
) -> np.ndarray:
    """
    BGR uint8 → tensor NCHW float32 listo para ``sess.run``.

    Equivalente exacto al ``preprocess_bgr``/``prep``/``prep_tile`` que estaba
    duplicado en 24 archivos.
    """
    cv2 = _require_cv2()
    resized = cv2.resize(bgr, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return to_chw(normalize(rgb), batch=batch)


def preprocess_path(
    path: str | Path, img_size: int = DEFAULT_IMG_SIZE, batch: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee un archivo y devuelve ``(tensor, bgr_original)``.

    Lanza ``FileNotFoundError`` con mensaje claro si no se pudo leer, en vez de
    propagar un ``None`` que revienta dos llamadas más tarde.
    """
    cv2 = _require_cv2()
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {p}")
    return preprocess_bgr(img, img_size, batch=batch), img


# ── torch ─────────────────────────────────────────────────────────────── #
def to_tensor_transform(img_size: int):
    """
    Devuelve un ``torchvision.transforms.Compose`` equivalente a
    :func:`preprocess_bgr`, para los Dataset de entrenamiento.

    Centraliza la normalización que antes estaba repetida en cada ``Dataset``.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_LIST, std=STD_LIST),
        ]
    )
