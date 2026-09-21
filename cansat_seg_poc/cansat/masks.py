"""
Máscaras de clases por frame — persistencia y geometría (consulta terrestre).

El post-vuelo YA calculaba las máscaras de terreno, daño, flood, fuego y
severidad, pero sólo persistía el overlay de terreno COLOREADO
(``ens_seg/``). Para responder consultas espaciales sobre las máscaras hace
falta el mapa de índices de clase, no el color.

Formato: PNG gris de 1 byte por píxel con el índice de clase; ``255`` = sin
dato. Lossless, legible por cualquier herramienta y de pocos KB por frame.
No se usa ``.npz``/``np.savez``: en el repo no existía y el PNG gris ya lo
leen OpenCV, PIL y el navegador.

Las operaciones de este módulo (intersección, área, buffer métrico, distancia,
componentes conexas, contornos) son las primitivas del motor simbólico
``cansat/consultas.py``. Todas trabajan con máscaras ``uint8``/``bool`` y
devuelven píxeles; la conversión a metros se hace con ``area_frame_m2``
(la huella de FOV que la telemetría ya trae como ``area_m2``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

NODATA: int = 255


# ══════════════════════════════════════════════════════════════════════════ #
#  Persistencia
# ══════════════════════════════════════════════════════════════════════════ #
def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    """Guarda una máscara de índices de clase como PNG gris (crea carpetas)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.ascontiguousarray(mask.astype(np.uint8)))
    return p


def load_mask(path: str | Path) -> np.ndarray | None:
    """Carga una máscara (H, W) uint8; ``None`` si no existe o no es gris."""
    p = Path(path)
    if not p.is_file():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════ #
#  Escala métrica
# ══════════════════════════════════════════════════════════════════════════ #
def pixel_scale_m(area_frame_m2: float, shape: tuple[int, ...]) -> float:
    """
    Lado de un píxel en metros, asumiendo frame cuadrado y FOV cuadrado.

    ``area_frame_m2`` es la huella de la telemetría (``2·alt·tan(FOV/2)``).
    Es una aproximación declarada: sin IMU no hay corrección por inclinación.
    """
    h, w = shape[0], shape[1]
    if area_frame_m2 <= 0 or h <= 0 or w <= 0:
        return 0.0
    return float(np.sqrt(area_frame_m2 / (h * w)))


def area_px(mask: np.ndarray) -> int:
    """Píxeles activos (distintos de 0 y de NODATA)."""
    m = np.asarray(mask)
    return int(((m != 0) & (m != NODATA)).sum())


def area_m2(mask: np.ndarray, area_frame_m2: float) -> float:
    """Área de la máscara en m² (aproximada con ``pixel_scale_m``)."""
    return area_px(mask) * pixel_scale_m(area_frame_m2, mask.shape) ** 2


# ══════════════════════════════════════════════════════════════════════════ #
#  Operaciones geométricas (píxeles)
# ══════════════════════════════════════════════════════════════════════════ #
def interseccion(*mascaras: np.ndarray) -> np.ndarray:
    """AND lógico de N máscaras (uint8 0/1)."""
    if not mascaras:
        raise ValueError("interseccion() necesita al menos una máscara")
    out = np.asarray(mascaras[0]).astype(bool)
    for m in mascaras[1:]:
        out &= np.asarray(m).astype(bool)
    return out.astype(np.uint8)


def mascara_poligono(shape: tuple[int, int], poligono_px: np.ndarray) -> np.ndarray:
    """Máscara binaria del interior de un polígono (N, 2) en píxeles."""
    m = np.zeros(shape, np.uint8)
    pts = np.asarray(poligono_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m


def buffer_mask(mask: np.ndarray, radio_px: float) -> np.ndarray:
    """
    Dilatación métrica: todo píxel a ≤ ``radio_px`` de la máscara.

    Kernel elíptico de lado ``2·radio+1`` (la aproximación estándar de buffer
    en imagen; el error es subpíxel para radios ≥ 2).

    ⚠ El radio se acota a la diagonal del frame: una consulta de "200 m" sobre
    un frame cuya huella mide 8 m daría un kernel de cientos de miles de píxeles
    (memoria/tiempo irrecuperables). Si el radio alcanza para cubrir el frame
    completo desde cualquier píxel, el buffer ES el frame entero.
    """
    m = (np.asarray(mask) != 0).astype(np.uint8)
    if radio_px <= 0:
        return m
    h, w = m.shape
    diagonal = float(np.hypot(h, w))
    if radio_px >= diagonal:
        return np.ones_like(m)
    r = max(1, int(round(radio_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m, k)


def distancia_a(mask: np.ndarray) -> np.ndarray:
    """
    Distancia euclídea (px) de cada píxel al píxel activo más cercano de ``mask``.

    Si la máscara está vacía devuelve ``inf`` (una consulta a distancia de algo
    que no está en el frame no tiene respuesta métrica).
    """
    m = (np.asarray(mask) != 0).astype(np.uint8)
    if not m.any():
        return np.full(m.shape, np.inf, np.float32)
    # DIST_MASK_PRECISE: la máscara 3×3 aproxima y da 1.91 en vez de 2.00,
    # y las consultas de distancia se reportan en metros.
    return cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def componentes(
    mask: np.ndarray,
    min_area_px: int = 0,
    area_frame_m2: float | None = None,
) -> list[dict]:
    """
    Componentes conexas (8-conectividad) con área, bbox y centroide.

    ``min_area_px`` filtra ruido (p.ej. un edificio de 1 píxel). Si se pasa
    ``area_frame_m2`` cada componente incluye ``area_m2``.
    """
    m = (np.asarray(mask) != 0).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    esc = (pixel_scale_m(area_frame_m2, m.shape) ** 2
           if area_frame_m2 and area_frame_m2 > 0 else None)
    out = []
    for i in range(1, n):                      # 0 = fondo
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        c = {
            "area_px": area,
            "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
            "centroide": [round(float(cent[i][0]), 2), round(float(cent[i][1]), 2)],
        }
        if esc is not None:
            c["area_m2"] = round(area * esc, 2)
        out.append(c)
    out.sort(key=lambda c: -c["area_px"])
    return out


def contornos(
    mask: np.ndarray,
    min_area_px: int = 0,
    epsilon_px: float = 1.5,
) -> list[np.ndarray]:
    """
    Contornos simplificados de una máscara binaria, como polígonos (N, 2) int32.

    ``epsilon_px`` es la tolerancia de ``approxPolyDP``: 1.5 px conserva la
    forma y elimina el ruido de escalera del pixelado en las consultas de UI.
    """
    m = (np.asarray(mask) != 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_px:
            continue
        if epsilon_px > 0:
            c = cv2.approxPolyDP(c, epsilon_px, True)
        out.append(c.reshape(-1, 2).astype(np.int32))
    out.sort(key=lambda p: -abs(float(cv2.contourArea(p.reshape(-1, 1, 2)))))
    return out


# ══════════════════════════════════════════════════════════════════════════ #
#  Longitud (esqueleto)
# ══════════════════════════════════════════════════════════════════════════ #
def _vecinos8(m: np.ndarray):
    """P2..P9 (N, E, S, O y diagonales) alrededor de cada píxel, por padding."""
    p = np.pad(m, 1)
    return (p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
            p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2])


def _zhang_suen(m: np.ndarray) -> np.ndarray:
    """
    Esqueletización Zhang-Suen vectorizada.

    Fallback para entornos sin ``cv2.ximgproc`` (p.ej. opencv-python-headless
    del CI). Mismo resultado que ``ximgproc.thinning(THINNING_ZHANGSUEN)``.
    """
    m = (m != 0).astype(np.uint8)
    while True:
        cambio = False
        for paso in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = _vecinos8(m)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
                    for i in range(8))
            if paso == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            borrar = (m == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if borrar.any():
                m[borrar] = 0
                cambio = True
        if not cambio:
            return m


def esqueleto(mask: np.ndarray) -> np.ndarray:
    """Esqueleto de 1 píxel de ancho de una máscara binaria."""
    m = (np.asarray(mask) != 0).astype(np.uint8)
    if not m.any():
        return m
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return (cv2.ximgproc.thinning(m * 255) > 0).astype(np.uint8)
    return _zhang_suen(m)


def longitud_px(mask: np.ndarray) -> float:
    """
    Longitud en píxeles estimada por esqueletización.

    Cada píxel del esqueleto cuenta 1: los tramos diagonales quedan
    subestimados ~10 %. Es la convención estándar cuando no hay una polilínea
    vectorial; se declara en las respuestas de consulta.
    """
    return float(esqueleto(mask).sum())
