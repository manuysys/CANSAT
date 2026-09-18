"""
Máscara de píxeles "sin datos".

El pipeline original marcaba como sin-datos TODO píxel con
``bgr.max(axis=2) <= 15``. Eso es correcto para **tiles de LoveDA**, que vienen
con bordes negros de recorte, pero en ``--camera`` (vuelo real) descarta
sombras, asfalto oscuro, agua profunda y cualquier frame subexpuesto — y sobre
ese remanente se calculan porcentajes, USI, veredicto, sampler y alerta.

Acá hay dos estrategias:

``border_mask``   la correcta para tiles: un píxel es "sin datos" sólo si es
                  casi negro **Y** pertenece a una mancha negra que toca el
                  borde de la imagen. Una sombra en el medio del frame no.
``threshold_mask`` la original, por compatibilidad y para depurar.

Regla práctica: ``--camera`` → umbral 0 (desactivado) o ``border_mask``.
                ``--folder`` de tiles → ``border_mask``.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DEFAULT_THRESHOLD: int = 15


def threshold_mask(bgr: np.ndarray, thresh: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """bool HWC-independent: True = píxel con datos. Umbral global de luminancia."""
    if thresh <= 0:
        return np.ones(bgr.shape[:2], dtype=bool)
    return bgr.max(axis=2) > thresh


def border_mask(
    bgr: np.ndarray, thresh: int = DEFAULT_THRESHOLD, min_area_frac: float = 0.0
) -> np.ndarray:
    """
    True = píxel con datos. Sólo descarta el negro **contiguo al borde**.

    ``min_area_frac`` descarta además manchas negras internas mayores a esa
    fracción del frame (útil si el tile tiene un recorte negro interior grande).
    """
    if cv2 is None:
        raise ImportError("border_mask necesita opencv-python.")
    if thresh <= 0:
        return np.ones(bgr.shape[:2], dtype=bool)

    dark = (bgr.max(axis=2) <= thresh).astype(np.uint8)
    if not dark.any():
        return np.ones(bgr.shape[:2], dtype=bool)

    h, w = dark.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    total = float(h * w)
    nodata = np.zeros((h, w), dtype=bool)

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        toca_borde = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
        grande = (area / total) >= min_area_frac > 0.0
        if toca_borde or grande:
            nodata[labels == i] = True

    return ~nodata


def valid_at_size(mask: np.ndarray, img_size: int) -> np.ndarray:
    """Reescala una máscara bool al tamaño del modelo con vecino más cercano."""
    if cv2 is None:
        raise ImportError("valid_at_size necesita opencv-python.")
    small = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return small.astype(bool)


def terrain_percentages(seg: np.ndarray, valid: np.ndarray, num_classes: int = 5):
    """
    Porcentajes por clase sobre píxeles válidos + clase dominante.

    Devuelve ``(pcts, dominante, n_validos)``. Si no hay píxeles válidos
    devuelve ceros y ``dominante=-1`` en vez de inventar una clase.
    """
    total = int(valid.sum())
    if total == 0:
        return [0.0] * num_classes, -1, 0
    pcts = [float(((seg == c) & valid).sum()) / total * 100.0 for c in range(num_classes)]
    return pcts, int(np.argmax(pcts)), total
