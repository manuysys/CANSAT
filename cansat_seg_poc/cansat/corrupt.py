"""
Corrupciones sintéticas — stress suite de aptitud de vuelo.

El DPD pide números honestos de cómo se degrada el pipeline fuera del
laboratorio. Este módulo genera las degradaciones que un CanSat puede
encontrar en vuelo y que NO están en los sets de validación (LoveDA/xBD/
RescueNet son nadir y con buena exposición):

  · ``lluvia``        trazos de lluvia + reducción de contraste + micoborroso
  · ``niebla``        síntesis física ``I = J·t + A·(1−t)`` con profundidad
                      vertical y ``A``/``t`` estimadas con el *dark channel
                      prior* de ``cansat/stress.py`` (el mismo que MIDE bruma)
  · ``motion_blur``   kernel lineal rotado + ruido (manoseo/velocidad)
  · ``subexposicion`` ganancia 2^EV con EV negativo (contraluz/sombra)
  · ``sobreexposicion`` ganancia 2^EV con EV positivo (cielo lavado)
  · ``escala``        remuestreo down→up (cambio de altitud efectiva)

⚠ Honestidad: son degradaciones SINTÉTICAS con severidad fija declarada acá.
No reemplazan una validación de vuelo real; sirven para comparar el mismo
modelo contra sí mismo (limpio vs estresado) en condiciones reproducibles.
El resultado se reporta con ``tools/stress_suite.py``.

Las constantes están todas juntas para que el informe las cite: si se cambia
una severidad, cambia el número y hay que regenerar ``outputs/stress_suite.json``.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import stress as ST

#: Corrupciones soportadas (orden estable para reportes).
CORRUPCIONES: tuple[str, ...] = (
    "lluvia",
    "niebla",
    "motion_blur",
    "subexposicion",
    "sobreexposicion",
    "escala",
)

# ── Severidad declarada (única, "moderada") ─────────────────────────────── #
LLUVIA_DENSIDAD: float = 0.0009   # gotas por píxel del frame
LLUVIA_ANGULO: float = 12.0       # grados respecto de la vertical
LLUVIA_LARGO: tuple[int, int] = (14, 40)
LLUVIA_ALPHA: float = 0.45        # opacidad de los trazos
LLUVIA_CONTRASTE: float = 0.85    # factor de contraste alrededor de 128

NIEBLA_BETA: float = 1.4          # atenuación por unidad de profundidad (0–1)
NIEBLA_PROFUNDIDAD_MIN: float = 0.15
NIEBLA_RUIDO: float = 0.35        # variación espacial de la profundidad

MOTION_K: int = 15
MOTION_ANGULO: float = 35.0
MOTION_SIGMA: float = 4.0

EXPOSICION_EV: float = 1.2        # EV de la sub/sobre-exposición

ESCALA_RANGO: tuple[float, float] = (0.45, 0.60)   # factor down→up


def motion_kernel(size: int, angle: float) -> np.ndarray:
    """Kernel lineal normalizado rotado ``angle`` grados (bench_degradados)."""
    kern = np.zeros((size, size), np.float32)
    kern[size // 2, :] = 1.0 / size
    m = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
    return cv2.warpAffine(kern, m, (size, size))


def nitidez(bgr: np.ndarray) -> float:
    """
    Varianza del Laplaciano tras un suavizado leve.

    A diferencia de ``bench_degradados.blur_score`` (sin suavizado), acá se
    aplica un Gaussian de σ=1 antes del Laplaciano: si no, el ruido gaussiano
    de ``motion_blur`` domina la varianza y una imagen MÁS borrosa parece más
    nítida. Con el suavizado la métrica mide estructura, no ruido.
    """
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (0, 0), 1.0)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


# ══════════════════════════════════════════════════════════════════════════ #
#  Corrupciones individuales: (bgr, rng) → (bgr', meta)  [escala: + mask]
# ══════════════════════════════════════════════════════════════════════════ #
def lluvia(bgr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Trazos de lluvia + contraste reducido + micoborroso.

    La lluvia real emborrona la escena y baja el contraste global; los trazos
    son brillantes contra el fondo. Se dibujan sobre una capa y se mezclan.
    """
    h, w = bgr.shape[:2]
    img = (bgr.astype(np.float32) - 128.0) * LLUVIA_CONTRASTE + 128.0
    capa = np.zeros_like(img)

    n = max(1, int(h * w * LLUVIA_DENSIDAD))
    x0 = rng.integers(0, w, n)
    y0 = rng.integers(0, h, n)
    largo = rng.integers(LLUVIA_LARGO[0], LLUVIA_LARGO[1] + 1, n)
    brillo = rng.integers(200, 256, n)
    ang = np.deg2rad(LLUVIA_ANGULO)
    for i in range(n):
        dx = round(float(largo[i]) * np.sin(ang))
        dy = round(float(largo[i]) * np.cos(ang))
        color = (float(brillo[i]),) * 3
        cv2.line(capa, (int(x0[i]), int(y0[i])),
                 (int(x0[i]) + dx, int(y0[i]) + dy), color, 1, cv2.LINE_AA)

    out = img * (1.0 - LLUVIA_ALPHA) + capa * LLUVIA_ALPHA
    out = cv2.GaussianBlur(out, (3, 3), 0)          # estela de movimiento
    meta = {
        "gotas": n,
        "angulo": LLUVIA_ANGULO,
        "contraste": LLUVIA_CONTRASTE,
        "alpha": LLUVIA_ALPHA,
    }
    return np.clip(out, 0, 255).astype(np.uint8), meta


def niebla(bgr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Niebla/bruma sintética con el modelo atmosférico ``I = J·t + A·(1−t)``.

    ``A`` se estima del propio frame con el *dark channel prior* de
    ``cansat/stress.py``; la profundidad crece hacia el horizonte (arriba) con
    ruido de baja frecuencia. La transmisión resultante se MIDE con
    ``haze_metrics`` y queda en ``meta`` como verificación.
    """
    img = bgr.astype(np.float32) / 255.0
    a = ST._atmospheric_light(img, ST.dark_channel(img)).astype(np.float32)

    h, w = bgr.shape[:2]
    filas = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    ruido = cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 24.0)
    prof = NIEBLA_PROFUNDIDAD_MIN + (1.0 - NIEBLA_PROFUNDIDAD_MIN) * filas
    prof = np.clip(prof + NIEBLA_RUIDO * (ruido - 0.5), 0.0, 1.0)

    t = np.exp(-NIEBLA_BETA * prof)[..., None]
    out = img * t + a[None, None, :] * (1.0 - t)
    out8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    meta = {
        "beta": NIEBLA_BETA,
        "atm_light": [round(float(v), 4) for v in a],
        **ST.haze_metrics(out8),
    }
    return out8, meta


def motion_blur(bgr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Desenfoque de movimiento lineal + ruido gaussiano."""
    blur = cv2.filter2D(bgr, -1, motion_kernel(MOTION_K, MOTION_ANGULO))
    out = blur.astype(np.float32) + rng.normal(0.0, MOTION_SIGMA, blur.shape)
    meta = {"k": MOTION_K, "angulo": MOTION_ANGULO, "sigma": MOTION_SIGMA}
    return np.clip(out, 0, 255).astype(np.uint8), meta


def exposicion(bgr: np.ndarray, ev: float) -> tuple[np.ndarray, dict]:
    """Sub/sobre-exposición por ganancia fotográfica ``2**ev`` saturada a 0–255."""
    out = bgr.astype(np.float32) * (2.0**ev)
    return np.clip(out, 0, 255).astype(np.uint8), {"ev": round(float(ev), 3)}


def escala(
    bgr: np.ndarray, mask: np.ndarray | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """
    Jitter de escala: down→up remuestreo (simula volar más alto).

    La máscara se remuestrea con vecino más cercano para conservar las clases.
    """
    h, w = bgr.shape[:2]
    f = float(rng.uniform(*ESCALA_RANGO))
    pw, ph = max(1, round(w * f)), max(1, round(h * f))
    chico = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    out = cv2.resize(chico, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_out = mask
    if mask is not None:
        mk = cv2.resize(mask.astype(np.uint8), (pw, ph),
                        interpolation=cv2.INTER_NEAREST)
        mask_out = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
    return out, mask_out, {"factor": round(f, 3)}


# ══════════════════════════════════════════════════════════════════════════ #
#  API de la suite
# ══════════════════════════════════════════════════════════════════════════ #
def aplicar(
    nombre: str,
    bgr: np.ndarray,
    mask: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """
    Aplica una corrupción y devuelve ``(bgr, mask, meta)``.

    Sólo ``escala`` modifica la máscara (los objetos cambian de tamaño); el
    resto la deja intacta para que la GT siga alineada píxel a píxel.
    """
    if nombre == "lluvia":
        out, meta = lluvia(bgr, rng)
    elif nombre == "niebla":
        out, meta = niebla(bgr, rng)
    elif nombre == "motion_blur":
        out, meta = motion_blur(bgr, rng)
    elif nombre == "subexposicion":
        out, meta = exposicion(bgr, -EXPOSICION_EV)
    elif nombre == "sobreexposicion":
        out, meta = exposicion(bgr, +EXPOSICION_EV)
    elif nombre == "escala":
        return escala(bgr, mask, rng)
    else:
        raise ValueError(
            f"corrupción desconocida: {nombre!r} (soportadas: {', '.join(CORRUPCIONES)})"
        )
    return out, mask, meta
