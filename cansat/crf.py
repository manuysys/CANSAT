"""
Refinamiento edge-aware de probabilidades de segmentación.

════════════════════════════════════════════════════════════════════════════
POR QUÉ SE REESCRIBIÓ
════════════════════════════════════════════════════════════════════════════
El ``crf_refine.py`` original se presentaba como un "Dense CRF mean-field
implementado en numpy (sin pydensecrf)" que *"suaviza probs respetando bordes
de color de la imagen"*. No hacía ninguna de las dos cosas:

  · La imagen **no participaba**. ``I = img.astype(np.float32) / 255.0`` se
    calculaba y nunca se usaba — código muerto. El filtro bilateral se aplicaba
    sobre ``q`` (las probabilidades de la propia clase), o sea que era
    **auto-suavizado**, no refinamiento guiado por color. Un dense CRF real
    define su kernel sobre la *imagen guía*.

  · ``sigmaColor = srgb / 255.0 = 0.051`` sobre valores en [0, 1].
    ``cv2.bilateralFilter`` interpreta ``sigmaColor`` en la escala de los datos
    de entrada; con 0.051 prácticamente no promedia nada. El parámetro ``srgb=13``
    tiene sentido en escala 0-255, no en 0-1.

  · ``Q = probs * np.exp(-compat * (1.0 - msg))`` usaba ``probs`` (el unary
    original) en cada iteración, así que el "mean-field" no acumulaba: la
    iteración 5 partía de la misma base que la 1.

Además existía **otro** CRF en ``mission_pipeline.crf_lite()`` (guided filter
de ``cv2.ximgproc``), con lo que había dos refinamientos distintos que no se
parecían. Éste es el único.

════════════════════════════════════════════════════════════════════════════
QUÉ HACE AHORA
════════════════════════════════════════════════════════════════════════════
Mean-field aproximado donde el paso de mensajes es un **filtro guiado**
(He, Sun & Tang) sobre las probabilidades de cada clase, usando la imagen como
guía. El filtro guiado es edge-aware por construcción y es la aproximación
estándar al kernel de apariencia de Krähenbühl & Koltun cuando no se dispone de
``pydensecrf``. El término de compatibilidad de Potts se aplica sobre ``Q``
acumulada, no sobre el unary original.

Ventaja concreta sobre ``jointBilateralFilter``: el filtro guiado se implementa
con box filters de OpenCV **core**, así que funciona igual en la PC (con contrib)
y en la Pi de vuelo (``opencv-python-headless``, sin contrib). Antes el pipeline
caía a un bilateral sobre las propias probabilidades cuando faltaba contrib: eso
no usaba la imagen y no respetaba bordes.

No es un dense CRF exacto —no hay kernel de suavizado espacial de largo alcance
ni normalización simétrica— y no se presenta como tal. Es un refinamiento
edge-aware, que es lo que el pipeline necesita y lo que el nombre debería decir.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _guided_filter(guide_gray: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Filtro guiado "rápido" (He et al.) sobre un canal, con box filters de core.

    ``guide_gray`` (H,W) float32 en [0,1] guía el suavizado de ``src`` (H,W).
    La clave es que ``a`` y ``b`` se estiman por ventana: donde la guía tiene un
    borde (varianza alta) el filtro se vuelve casi identidad, y donde es plana
    promedia. Eso es lo que lo hace edge-aware y lo que el bilateral
    auto-aplicado no hacía.
    """
    r = max(1, int(radius))
    ksize = (2 * r + 1, 2 * r + 1)
    guide_f = guide_gray.astype(np.float32)
    p = src.astype(np.float32)
    mean_I = cv2.blur(guide_f, ksize)
    mean_p = cv2.blur(p, ksize)
    cov_Ip = cv2.blur(guide_f * p, ksize) - mean_I * mean_p
    var_I = cv2.blur(guide_f * guide_f, ksize) - mean_I * mean_I
    a = cov_Ip / (var_I + max(1e-8, eps))
    b = mean_p - a * mean_I
    return cv2.blur(a, ksize) * guide_f + cv2.blur(b, ksize)


def _smooth_with_guide(
    probs: np.ndarray, guide_u8: np.ndarray, d: int, sigma_color: float
) -> np.ndarray:
    """
    Paso de mensajes: suaviza ``probs`` (C,H,W en [0,1]) usando la imagen como
    guía. ``sigma_color`` viene en escala 0-255 (como en el bilateral clásico) y
    se traduce a ``eps`` del filtro guiado con ``(sigma_color/255)^2``.
    """
    guide = np.asarray(guide_u8)
    if guide.ndim == 2:
        gray = guide
    else:
        gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    eps = (max(1.0, float(sigma_color)) / 255.0) ** 2
    radius = max(1, int(d) // 2)
    out = np.empty_like(probs)
    for c in range(probs.shape[0]):
        out[c] = _guided_filter(gray, probs[c], radius=radius, eps=eps)
    return out


def dense_crf(
    probs: np.ndarray,
    img: np.ndarray,
    iters: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 3.0,
    compat: float = 5.0,
    d: int = 9,
) -> np.ndarray:
    """
    Refinamiento edge-aware de probabilidades, guiado por la imagen.

    Parámetros
    ----------
    probs : (C, H, W) probabilidades por clase. Se normalizan internamente.
    img   : (H, W, 3) BGR uint8 — **la guía**. Si su tamaño no coincide con el
            de ``probs`` se reescala (caso típico: probs a 512 y frame nativo).
    iters : iteraciones de mean-field. 0 desactiva el refinamiento.
    sigma_color : ancho del kernel en el espacio de color, **en escala 0-255**.
    sigma_space : se conserva por compatibilidad de firma; el radio espacial del
            filtro guiado sale de ``d`` (``radius = d//2``).
    compat  : fuerza del modelo de Potts (penaliza clases distintas).
    d       : diámetro del filtro (``d//2`` = radio del box filter guiado).

    Devuelve
    --------
    (C, H, W) probabilidades refinadas y renormalizadas.
    """
    if cv2 is None:
        raise ImportError("dense_crf necesita opencv-python.")

    Q = np.asarray(probs, dtype=np.float32).copy()
    if Q.ndim != 3:
        raise ValueError(f"probs debe ser (C,H,W); llegó {Q.shape}")
    Q = np.clip(Q, 1e-6, None)
    Q /= Q.sum(axis=0, keepdims=True)

    if iters <= 0 or compat <= 0:
        return Q

    guide = np.asarray(img)
    if guide.ndim == 2:
        guide = cv2.cvtColor(guide, cv2.COLOR_GRAY2BGR)
    if guide.shape[:2] != Q.shape[1:]:
        guide = cv2.resize(guide, (Q.shape[2], Q.shape[1]), interpolation=cv2.INTER_LINEAR)
    if guide.dtype != np.uint8:
        guide = np.clip(guide, 0, 255).astype(np.uint8)

    # Unary fijo (el dato); Q es lo que itera.
    unary = Q.copy()

    for _ in range(int(iters)):
        msg = _smooth_with_guide(Q, guide, d=d, sigma_color=sigma_color)
        msg = np.clip(msg, 1e-6, None)
        msg /= msg.sum(axis=0, keepdims=True)
        # Potts: refuerza donde el vecindario de color coincide.
        Q = unary * np.exp(-compat * (1.0 - msg))
        Q = np.clip(Q, 1e-6, None)
        Q /= Q.sum(axis=0, keepdims=True)

    return Q


def refine_argmax(probs: np.ndarray, img: np.ndarray, **kw) -> np.ndarray:
    """Atajo: refina y devuelve el mapa de clases."""
    return dense_crf(probs, img, **kw).argmax(axis=0)
