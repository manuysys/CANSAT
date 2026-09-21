"""
Selección de tiles para anotación (active learning) — cola del próximo vuelo.

Hoy no hay frames de vuelo reales anotados, así que no se puede re-entrenar; lo
que **sí** se puede es no anotar al azar cuando lleguen: esta cola prioriza qué
tiles revisar primero. Tres señales, todas medibles con lo que ya existe:

  · **incertidumbre** del modelo de terreno (entropía media normalizada),
  · **rareza** de la mezcla de clases del tile contra la referencia de LoveDA
    Val (un tile con clases poco frecuentes enseña más que uno de pasto),
  · **estrés ambiental** (bruma por dark channel) — los frames degradados son
    los que más revelan fallas.

Score 0-1 = 0.50·incertidumbre + 0.30·rareza + 0.20·estrés (pesos declarados
acá). La cola es una lista ordenada de rutas + motivos; **no se copian** los
frames. Referencia: uncertainty sampling (Lewis & Catlett; ver también el uso
de entropía en active learning para detección UAV).
"""

from __future__ import annotations

import numpy as np

#: Pesos del score (declarados para el informe).
PESOS: tuple[float, float, float] = (0.50, 0.30, 0.20)


def entropia_normalizada(probs) -> float:
    """
    Entropía media de un mapa de probabilidades, normalizada a [0, 1].

    ``probs`` es (C, H, W) —como la salida softmax del modelo— o un vector (C,).
    1 = el modelo no sabe (uniforme), 0 = one-hot seguro.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 3:
        p = p.reshape(p.shape[0], -1).mean(axis=1)
    if p.ndim != 1 or p.size < 2:
        raise ValueError(f"se esperaba (C,H,W) o (C,), llegó {p.shape}")
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    h = float(-np.sum(p * np.log(p)))
    return float(h / np.log(p.size))


def rareza(pcts, clases_prom) -> float:
    """
    Rareza de la mezcla de clases contra la referencia, en [0, 1].

    Cada clase pesa ``min(ref)/ref_i`` (las raras pesan más), normalizado por el
    máximo teórico. Es un heurístico declarado, no una probabilidad.
    """
    p = np.asarray(pcts, dtype=np.float64)
    ref = np.asarray(clases_prom, dtype=np.float64)
    if p.shape != ref.shape:
        raise ValueError(f"mezclas de distinto tamaño: {p.shape} vs {ref.shape}")
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    w = ref.min() / np.maximum(ref, 1e-9)
    w = w / max(w.max(), 1e-9)          # la más común pesa ~min/ref, la rara ~1
    return float(np.clip(np.sum(p * w) / max(w.sum(), 1e-9), 0.0, 1.0))


def puntaje(incertidumbre: float, rareza_v: float, stress: float,
            pesos: tuple[float, float, float] = PESOS) -> float:
    """Score de prioridad 0-1 con los tres componentes recortados a [0, 1]."""
    a, b, c = (float(np.clip(v, 0.0, 1.0)) for v in (incertidumbre, rareza_v, stress))
    return round(a * pesos[0] + b * pesos[1] + c * pesos[2], 4)


def motivos(incertidumbre: float, rareza_v: float, stress: float,
            umbral: float = 0.55) -> list[str]:
    """Motivos legibles de por qué el tile entró a la cola."""
    out = []
    if incertidumbre >= umbral:
        out.append("entropia alta")
    if rareza_v >= umbral:
        out.append("clases raras")
    if stress >= umbral:
        out.append("estres ambiental")
    return out or ["score de cola"]
