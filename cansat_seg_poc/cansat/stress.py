"""
Estrés ambiental — CanSat LB135.

Dos señales que el DPD pide ("estudiar el estrés ambiental generado por la
contaminación en las ciudades… junto con la información de los sensores") y que
los índices urbanos (USI/GVI, `cansat/indices.py`) NO cubrían:

  · **Bruma/contaminación por imagen**: el *dark channel prior* (He et al.,
    2009) estima la transmisión atmosférica de la escena. Con bruma/aerosoles
    la transmisión baja y el canal oscuro se aclara. Se reporta:
        ``haze_pct``   % de píxeles con transmisión < 0.7 (bruma densa)
        ``visibility`` transmisión media (1 = aire cristalino)
        ``atm_light``  luz atmosférica estimada (0–1)
    ⚠ Es un proxy **relativo**: una escena nevada o un cielo blanco pueden dar
    bruma alta. Sirve para comparar frames del mismo vuelo y detectar smog.

  · **Calor (sensores)**: ``humidex = T + 0.5555·(e − 10)`` con ``e`` la
    presión de vapor (Magnus). Combina temperatura y humedad del BME280 o del
    UART — el "junto con la información de los sensores" del DPD.

Referencias: He, Sun & Tang (CVPR 2009) para el dark channel; Environment
Canada para el humidex (categorías: <30 confortable, 30–39 disconfort,
40–45 gran disconfort, >45 peligroso).
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Bruma (dark channel prior) ────────────────────────────────────────── #
HAZE_DENSE_T: float = 0.7      # transmisión por debajo de esto = bruma densa
CONTAM_LEVE: float = 5.0       # % de frame con bruma densa
CONTAM_MODERADA: float = 20.0
CONTAM_DENSA: float = 45.0

# ── Calor (humidex) ───────────────────────────────────────────────────── #
HEAT_DISCONFORT: float = 30.0
HEAT_ESTRES: float = 40.0
HEAT_PELIGRO: float = 46.0


def dark_channel(img: np.ndarray, size: int = 15) -> np.ndarray:
    """Canal oscuro: mínimo por canal + erosión (mínimo local). ``img`` en [0,1]."""
    dc = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(dc.astype(np.float32), kernel)


def _atmospheric_light(img: np.ndarray, dark: np.ndarray,
                       top: float = 0.001) -> np.ndarray:
    """Luz atmosférica: píxel más brillante entre el ``top`` % del canal oscuro."""
    h, w = dark.shape
    n = max(1, int(h * w * top))
    flat = dark.ravel()
    idx = np.argpartition(flat, -n)[-n:]
    ys, xs = np.unravel_index(idx, dark.shape)
    candidates = img[ys, xs]
    return np.clip(candidates[candidates.sum(axis=1).argmax()], 1e-3, 1.0)


def haze_metrics(bgr: np.ndarray, size: int = 15, omega: float = 0.95) -> dict:
    """
    Métricas de bruma/aerosoles de un frame BGR (uint8).

    Devuelve ``haze_pct`` (%), ``visibility`` (0–1) y ``atm_light`` (0–1).
    """
    img = bgr.astype(np.float32) / 255.0
    dark = dark_channel(img, size)
    a = _atmospheric_light(img, dark)
    t = 1.0 - omega * dark_channel(img / a, size)
    t = np.clip(t, 0.1, 1.0)
    return {
        "haze_pct": round(float((t < HAZE_DENSE_T).mean() * 100.0), 2),
        "visibility": round(float(t.mean()), 4),
        "atm_light": round(float(a.mean()), 4),
    }


def contam_verdict(haze_pct: float) -> str:
    """Categoría de contaminación/bruma a partir de ``haze_pct``."""
    if haze_pct < CONTAM_LEVE:
        return "AIRE LIMPIO"
    if haze_pct < CONTAM_MODERADA:
        return "BRUMA LEVE"
    if haze_pct < CONTAM_DENSA:
        return "BRUMA MODERADA"
    return "BRUMA DENSA (SMOG)"


# ── Calor ─────────────────────────────────────────────────────────────── #
def humidex(temp_c: float, hum_pct: float) -> float:
    """
    Humidex (Environment Canada): ``T + 0.5555·(e − 10)``.

    ``e`` = presión de vapor (hPa) por Magnus:
    ``6.11 · 10^(7.5·T/(237.7+T)) · RH/100``.
    """
    t = float(temp_c)
    rh = max(0.0, min(100.0, float(hum_pct)))
    e = 6.11 * 10.0 ** (7.5 * t / (237.7 + t)) * rh / 100.0
    return round(t + 0.5555 * (e - 10.0), 2)


def heat_verdict(hx: float) -> str:
    """Categoría de estrés térmico a partir del humidex."""
    if hx < HEAT_DISCONFORT:
        return "CONFORTABLE"
    if hx < HEAT_ESTRES:
        return "DISCONFORT"
    if hx < HEAT_PELIGRO:
        return "ESTRÉS TÉRMICO"
    return "PELIGROSO"


# ── Índice agregado ───────────────────────────────────────────────────── #
def stress_score(haze_pct: float, hx: float, usi_norm: float,
                 fire_pct: float = 0.0) -> float:
    """
    Índice de estrés ambiental 0–100: bruma (50 %), calor (25 %) y carga
    antrópica (25 %). Un incendio detectado pisa el índice (mínimo 60) porque
    es estrés agudo, no crónico. Pesos declarados acá para que el informe los
    cite.
    """
    haze_n = min(1.0, max(0.0, float(haze_pct) / 50.0))
    heat_n = min(1.0, max(0.0, (float(hx) - 25.0) / 25.0))
    urb_n = min(1.0, max(0.0, float(usi_norm)))
    base = 100.0 * (0.50 * haze_n + 0.25 * heat_n + 0.25 * urb_n)
    if fire_pct and fire_pct > 1.0:
        base = max(base, min(100.0, 60.0 + float(fire_pct) * 2.0))
    return round(base, 1)


# ── Vegetación y sombras (índices RGB sin segmentación) ───────────────── #
EXG_UMBRAL: float = 0.05      # ExG por encima de esto = píxel vegetal


def exg_metrics(bgr: np.ndarray) -> dict:
    """
    Excess Green (ExG = 2G − R − B, normalizado) — índice de vegetación RGB
    robusto a la iluminación, independiente del modelo de segmentación.

    Devuelve ``exg_medio`` (media en [-2, 2]) y ``veg_exg_pct`` (% de píxeles
    vegetales por ExG).
    """
    img = bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    s = np.maximum(1.0, b + g + r)
    exg = (2.0 * g - r - b) / s
    return {
        "exg_medio": round(float(exg.mean()), 4),
        "veg_exg_pct": round(float((exg > EXG_UMBRAL).mean() * 100.0), 2),
    }


SHADOW_UMBRAL: int = 55       # V por debajo = sombra/objeto oscuro


def shadow_pct(bgr: np.ndarray) -> float:
    """
    % de píxeles muy oscuros (V < umbral): proxy de sombras proyectadas —
    más sombra ⇒ más volumen construido/vertical (proxy de densidad urbana).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return round(float((hsv[:, :, 2] < SHADOW_UMBRAL).mean() * 100.0), 2)
