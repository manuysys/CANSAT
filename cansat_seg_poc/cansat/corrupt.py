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
  · ``sombras``       parches poligonales oscurecidos, borde suavizado (sol de
                      mediodía; solo fotométrica, la GT no se toca)
  · ``vibracion``     traslación ±3 px + rotación ±0.4° (jitter del motor;
                      geométrica: la máscara se transforma igual, borde replicate)

El mismo módulo se usa como **augmentación de entrenamiento** (train_damage_v3.py
``--aug-uav`` aplica motion_blur/escala/sombras/vibracion): entrenar y evaluar
con la misma fuente evita que el "mejora" sea un artefacto de implementaciones
distintas.

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

#: Corrupciones soportadas (orden estable para reportes: al final van las
#: nuevas para no cambiar el stream aleatorio de las que ya existen).
CORRUPCIONES: tuple[str, ...] = (
    "lluvia",
    "niebla",
    "motion_blur",
    "subexposicion",
    "sobreexposicion",
    "escala",
    "sombras",
    "vibracion",
)

#: Subconjunto UAV para augmentación de entrenamiento (movimiento/altitud/luz).
AUG_UAV: tuple[str, ...] = ("motion_blur", "escala", "sombras", "vibracion")

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

SOMBRAS_N: int = 2                # parches de sombra (1..N por frame)
SOMBRAS_FACTOR: float = 0.45      # multiplicador de brillo dentro de la sombra
SOMBRAS_BLUR: int = 21            # suavizado del borde (impar)

VIBRACION_DX: int = 3             # traslación máxima en píxeles
VIBRACION_ANG: float = 0.4        # rotación máxima en grados


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


def sombras(bgr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Sombras duras: 1..N polígonos convexos oscurecidos, borde suavizado.

    Solo fotométrica (la geometría no cambia, la GT sigue alineada): una sombra
    real no mueve los edificios, solo les baja el brillo.
    """
    h, w = bgr.shape[:2]
    n = int(rng.integers(1, SOMBRAS_N + 1))
    mascara = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.08, 0.25) * max(h, w)
        k = int(rng.integers(3, 7))
        angs = np.sort(rng.uniform(0, 2 * np.pi, k))
        radios = rng.uniform(0.5 * r, r, k)
        pts = np.stack([cx + radios * np.cos(angs),
                        cy + radios * np.sin(angs)]).T.astype(np.int32)
        cv2.fillConvexPoly(mascara, pts, 1.0)
    mascara = cv2.GaussianBlur(mascara, (SOMBRAS_BLUR, SOMBRAS_BLUR), 0)
    factor = 1.0 - (1.0 - SOMBRAS_FACTOR) * mascara[..., None]
    out = bgr.astype(np.float32) * factor
    meta = {"n": n, "factor": SOMBRAS_FACTOR, "borde": SOMBRAS_BLUR}
    return np.clip(out, 0, 255).astype(np.uint8), meta


def vibracion(
    bgr: np.ndarray, mask: np.ndarray | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """
    Sacudida de alta frecuencia: traslación ±3 px + rotación ±0.4°.

    Geométrica como ``escala``: la máscara se transforma con la misma matriz
    (vecino más cercano) y el borde se replica (jitter, no recorte).
    """
    h, w = bgr.shape[:2]
    ang = float(rng.uniform(-VIBRACION_ANG, VIBRACION_ANG))
    dx = float(rng.uniform(-VIBRACION_DX, VIBRACION_DX))
    dy = float(rng.uniform(-VIBRACION_DX, VIBRACION_DX))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    m[:, 2] += (dx, dy)
    out = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    mask_out = mask
    if mask is not None:
        mask_out = cv2.warpAffine(mask.astype(np.uint8), m, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REPLICATE)
    return out, mask_out, {"dx": round(dx, 2), "dy": round(dy, 2),
                           "ang": round(ang, 3)}


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

    ``escala`` y ``vibracion`` son geométricas y transforman la máscara igual;
    el resto es fotométrico y la deja intacta para que la GT siga alineada.
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
    elif nombre == "sombras":
        out, meta = sombras(bgr, rng)
    elif nombre == "vibracion":
        return vibracion(bgr, mask, rng)
    else:
        raise ValueError(
            f"corrupción desconocida: {nombre!r} (soportadas: {', '.join(CORRUPCIONES)})"
        )
    return out, mask, meta
