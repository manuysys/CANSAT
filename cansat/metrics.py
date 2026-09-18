"""
Métricas de segmentación — FUENTE ÚNICA.

Antes había TRES implementaciones de la matriz de confusión:
  · ``train.py``             vectorizada con ``np.bincount``      (rápida)
  · ``eval_onnx_gap.py``,
    ``eval_int8_cpu.py``,
    ``evaluate_val.py``      doble loop de Python sobre 5×5       (~25× más lenta)
  · ``train_siamese_damage`` doble loop sobre clases              (la más lenta)

Acá queda solo la vectorizada.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

IGNORE_INDEX = 255


def confusion(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, ignore_index: int = IGNORE_INDEX
) -> np.ndarray:
    """
    Matriz de confusión ``conf[gt, pred]`` de forma vectorizada.

    Acepta arrays de cualquier forma (se aplanan) y enmascara ``ignore_index``.
    """
    t = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape distinto: true {t.shape} vs pred {p.shape}")
    valid = (t >= 0) & (t < num_classes) & (t != ignore_index)
    if ignore_index >= 0:
        valid &= p != ignore_index
    t, p = t[valid].astype(np.int64), p[valid].astype(np.int64)
    return np.bincount(num_classes * t + p, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def accumulate(
    total: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Acumula in-place un batch sobre la matriz total."""
    total += confusion(y_true, y_pred, num_classes, ignore_index)
    return total


@dataclass
class SegMetrics:
    miou: float
    pixel_acc: float
    iou: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    n_pixels: int
    present: np.ndarray
    soporte: np.ndarray          # píxeles de GT por clase (conf.sum(1))

    def as_dict(self, class_names: Sequence[str]) -> dict[str, object]:
        """Serializable a JSON (todo lo que no es escalar va como lista)."""
        return {
            "miou": round(float(self.miou), 6),
            "pixel_acc": round(float(self.pixel_acc), 6),
            "n_pixels": int(self.n_pixels),
            "por_clase": {
                name: {
                    "iou": round(float(self.iou[i]), 6),
                    "precision": round(float(self.precision[i]), 6),
                    "recall": round(float(self.recall[i]), 6),
                    "f1": round(float(self.f1[i]), 6),
                    # ⚠ Antes esto guardaba int(present[i]) = 0/1 (el booleano
                    # de "clase presente"), así que el informe mostraba
                    # "Soporte: 1 px" para todas las clases.
                    "soporte_px": int(self.soporte[i]),
                }
                for i, name in enumerate(class_names)
            },
        }


def from_confusion(conf: np.ndarray) -> SegMetrics:
    """IoU / precisión / recall / F1 / mIoU / pixel-accuracy desde la matriz."""
    conf = np.asarray(conf, dtype=np.float64)
    inter = np.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    iou = inter / np.maximum(union, 1.0)
    prec = inter / np.maximum(conf.sum(0), 1.0)
    rec = inter / np.maximum(conf.sum(1), 1.0)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    present = conf.sum(1) > 0
    total = conf.sum()
    return SegMetrics(
        miou=float(iou[present].mean()) if present.any() else 0.0,
        pixel_acc=float(np.trace(conf) / total) if total else 0.0,
        iou=iou,
        precision=prec,
        recall=rec,
        f1=f1,
        n_pixels=int(total),
        present=present,
        soporte=conf.sum(1),
    )


def agreement(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """
    Fracción de píxeles donde dos predicciones coinciden (0..1).

    Es la métrica correcta para comparar FP32 vs INT8 — NO la media de
    probabilidades, que era lo que usaba ``validate_int8_mission.py``.
    """
    a = np.asarray(pred_a).ravel()
    b = np.asarray(pred_b).ravel()
    return float((a == b).mean()) if a.size else 0.0


def print_table(m: SegMetrics, class_names: Sequence[str], title: str = "") -> None:
    if title:
        print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")
    print(f"  {'Clase':<14} {'IoU':>7} {'Precisión':>10} {'Recall':>8}")
    for i, name in enumerate(class_names):
        print(
            f"  {name:<14} {m.iou[i] * 100:6.1f}% "
            f"{m.precision[i] * 100:9.1f}% {m.recall[i] * 100:7.1f}%"
        )
    print("-" * 62)
    print(
        f"  mIoU: {m.miou * 100:.2f}%   |   Pixel accuracy: {m.pixel_acc * 100:.2f}%"
        f"   |   {m.n_pixels:,} px"
    )
