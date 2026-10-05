"""
MoE de daño — compuerta aprendida (integración del PoC V11 4.2).

La compuerta es una regresión logística sobre 7 features baratas de cada frame
(probabilidades de daño de los dos expertos, desacuerdo, brillo/contraste) que
elige QUÉ EXPERTO usar por frame:

  · experto A = daño principal (`cansat_damage3_mobilenetv2.onnx`)
  · experto B = two-stage UAV (`cansat_damage_v3_bal_ep6.onnx`)

Artefacto: ``outputs/moe_gate.json`` (generado por ``tools/moe_damage.py
--guardar-gate``). Sin artefacto, el llamador debe mantener el comportamiento
viejo (max de los expertos): la integración es opcional y no rompe nada.

PoC: docs/benchmarks/moe_damage.json (compuerta +4.5/+2.4 pts sobre el mejor
fijo en dos semillas).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Nombres de las features (mismo orden que el PoC/entrenamiento).
FEATURES: tuple[str, ...] = (
    "p_a_media", "p_a_max", "p_b_media", "p_b_max", "desacuerdo",
    "brillo", "contraste",
)


@dataclass
class Gate:
    """Regresión logística + estandarizado, serializada a JSON."""

    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    meta: dict

    def prob_b(self, feats: list[float]) -> float:
        """Probabilidad de elegir el experto B (two-stage UAV)."""
        z = self.intercept
        for x, m, s, w in zip(feats, self.mean, self.scale, self.coef,
                              strict=True):
            z += w * ((x - m) / (s if s else 1.0))
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def elige_b(self, feats: list[float], umbral: float = 0.5) -> bool:
        return self.prob_b(feats) >= umbral


def cargar_gate(path: str | Path) -> Gate | None:
    """Carga el gate; ``None`` si el archivo no existe o está corrupto."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        g = Gate(
            mean=[float(v) for v in d["scaler"]["mean"]],
            scale=[float(v) for v in d["scaler"]["scale"]],
            coef=[float(v) for v in d["coef"]],
            intercept=float(d["intercept"]),
            meta=dict(d.get("meta") or {}),
        )
        if len(g.mean) != len(FEATURES) or len(g.coef) != len(FEATURES):
            return None
        return g
    except (OSError, ValueError, KeyError, TypeError):
        return None


def features(prob_a: np.ndarray, prob_b: np.ndarray,
             tensor: np.ndarray) -> list[float]:
    """
    Features del frame, idénticas a las del PoC:

    ``prob_a``/``prob_b``: mapas de probabilidad (H, W) de la clase "dañado"
    (softmax del logit 2); ``tensor``: el tensor preprocesado (1, 3, H, W) del
    que se toman brillo/contraste (como en el entrenamiento del gate).
    """
    t = np.asarray(tensor)[0].transpose(1, 2, 0)
    return [
        float(prob_a.mean()), float(prob_a.max()),
        float(prob_b.mean()), float(prob_b.max()),
        float(np.abs(prob_a - prob_b).mean()),
        float(t.mean()), float(t.std()),
    ]


def elegir(gate: Gate | None, prob_a: np.ndarray, prob_b: np.ndarray,
           tensor: np.ndarray) -> tuple[str, float, list[float]]:
    """
    Devuelve ``("a"|"b", prob_b, features)``. Sin gate, elige "a" (el
    llamador decide el fallback real).
    """
    feats = features(prob_a, prob_b, tensor)
    if gate is None:
        return "a", 0.5, feats
    pb = gate.prob_b(feats)
    return ("b" if pb >= 0.5 else "a"), pb, feats
