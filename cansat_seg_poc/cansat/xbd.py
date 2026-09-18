"""
Utilidades del dataset xBD / xView2 — CanSat LB135.

Acá vive lo que **comparten** los scripts de daño y el evaluador:

  · :func:`grupo` — el nombre del desastre de un tile (``hurricane-harvey``).
  · :func:`split_por_desastre` — train/val **por desastre**, no por fila.
  · :func:`iou_dano_edificios` — la métrica de selección que usa la misión
    (daño contado sólo sobre edificios, receta xView2).

════════════════════════════════════════════════════════════════════════════
POR QUÉ EL SPLIT POR DESASTRE
════════════════════════════════════════════════════════════════════════════
Los cuatro scripts de daño partían el manifest con ``random.Random(42).shuffle``
y 90/10 **por fila**. Como los tiles de un mismo evento son geográficamente
vecinos, train y val terminaban compartiendo desastres: el "IoU de val" medía
en gran parte datos vistos. Medición del 2026-09-17: con split por fila el IoU
de daño daba 0.32; con desastres NO vistos, 0.10. La diferencia era la fuga.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np


def grupo(nombre: str) -> str:
    """
    Nombre del desastre de un tile xBD.

    ``hurricane-harvey_00000426`` → ``hurricane-harvey``. Los nombres del
    manifest pueden venir con sufijos (``_post_disaster``): se toma lo anterior
    al primer ``_``.
    """
    return str(nombre).split("_", 1)[0]


def split_por_desastre(
    rows: Sequence[dict],
    holdout_frac: float = 0.2,
    seed: int = 42,
    name_key: str = "name",
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Divide las filas del manifest en train/val **agrupando por desastre**.

    Devuelve ``(train_rows, val_rows, val_grupos)``. El holdout se hace por
    desastre completo, así ningún evento aparece en los dos lados.
    """
    grupos = sorted({grupo(r[name_key]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(grupos)
    n_val = max(1, round(len(grupos) * float(holdout_frac)))
    val_g = set(grupos[:n_val])
    train_rows = [r for r in rows if grupo(r[name_key]) not in val_g]
    val_rows = [r for r in rows if grupo(r[name_key]) in val_g]
    return train_rows, val_rows, sorted(val_g)


def iou_dano_edificios(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    IoU de "dañado" contado sobre edificios (métrica de la misión).

    ``pred``: mapa de clases 0/1/2 (other/intacto/dañado).
    ``gt``:   máscara 0..4 de xBD (2-4 = dañado).

    Es la métrica con la que se selecciona el mejor checkpoint y la que mide
    ``eval_sliding``: el daño sólo tiene sentido sobre construcciones.
    """
    pd = pred == 2
    gd = gt >= 2
    gb = gt >= 1
    union = ((pd & gb) | gd).sum()
    return float((pd & gd).sum() / union) if union else 0.0
