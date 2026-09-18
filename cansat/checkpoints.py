"""
Formato único de checkpoint.

════════════════════════════════════════════════════════════════════════════
El problema anterior
════════════════════════════════════════════════════════════════════════════
``train.py`` guardaba un **dict con metadata**::

    torch.save({"model_state_dict": ..., "num_classes": ..., "img_size": ...,
                "miou": ..., "iou_per_class": ...}, "best_model.pth")

Los otros **15** scripts de entrenamiento guardaban el **state_dict crudo**::

    torch.save(model.state_dict(), "outputs/best_terrain_v2.pth")

Y los exportadores estaban atados a uno u otro: ``export_onnx.py`` hacía
``ckpt["model_state_dict"]`` mientras ``export_v2.py`` / ``export_damage_v3.py``
hacían ``load_state_dict(torch.load(...))``. **Ningún exportador era
intercambiable**, y ya había un ``try: M(5) except TypeError: M()`` parcheando
una confusión de firmas.

Además: **23 llamadas a ``torch.load()``, ninguna con ``weights_only=True``**
(una usaba explícitamente ``weights_only=False``). ``torch.load`` sin ese flag
ejecuta pickle arbitrario.

════════════════════════════════════════════════════════════════════════════
La solución
════════════════════════════════════════════════════════════════════════════
:func:`save_ckpt` escribe siempre el formato dict. :func:`load_model_state`
**acepta los dos formatos**, así que todos los checkpoints ya entrenados siguen
cargando sin migración.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STATE_KEY = "model_state_dict"


def _json_safe(v: Any) -> Any:
    """
    Convierte escalares/arrays de numpy a tipos nativos de Python.

    ⚠ Sin esto, metadata como ``miou=np.float64(0.52)`` o listas de
      ``np.float32`` quedan guardadas como globales de numpy y después
      ``torch.load(..., weights_only=True)`` las RECHAZA:
      "Unsupported global: numpy._core.multiarray.scalar". El checkpoint se
      escribía bien pero no se podía volver a leer (le pasó al primer
      entrenamiento del flood specialist).
    """
    import numpy as np

    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def save_ckpt(path: str | Path, model, **meta: Any) -> Path:
    """
    Guarda ``state_dict`` + metadata en un formato único.

    La metadata que importa para poder reproducir un modelo meses después::

        save_ckpt(out, model, num_classes=5, img_size=320,
                  class_names=CLASS_NAMES, miou=0.5317, iou_per_class=[...],
                  dataset="loveda_remapped", script="train_cbam.py",
                  epochs=40, seed=42)

    Toda la metadata se sanitiza a tipos nativos (ver :func:`_json_safe`) para
    que el checkpoint se pueda releer con ``weights_only=True``.
    """
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {STATE_KEY: model.state_dict()}
    payload.update({k: _json_safe(v) for k, v in meta.items() if v is not None})
    torch.save(payload, p)
    return p


def load_model_state(
    path: str | Path, strict: bool = True, min_loaded_frac: float = 1.0
) -> dict[str, Any]:
    """
    Devuelve el ``state_dict`` de un checkpoint, sea cual sea su formato.

    ``weights_only=True`` siempre: los checkpoints de este proyecto sólo
    contienen tensores y escalares, así que no hace falta el pickle completo
    (y de paso se cierra la ejecución de código arbitrario).

    ``strict=False`` permite carga parcial — pero entonces ``min_loaded_frac``
    exige que se haya cargado al menos esa fracción de las capas, porque
    ``train_damage_v3.py`` hacía ``load_state_dict(filtered, strict=False)`` y
    **seguía de largo en silencio** aunque no matcheara casi nada.
    """
    import torch

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {p}")

    obj = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and STATE_KEY in obj:
        return obj[STATE_KEY]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"{p}: formato de checkpoint no reconocido ({type(obj).__name__}).")


def load_into(
    model, path: str | Path, strict: bool = True, min_loaded_frac: float = 1.0, verbose: bool = True
):
    """
    Carga un checkpoint en ``model`` con verificación de cobertura real.

    Devuelve ``(model, fracción_cargada)``.
    """

    sd = load_model_state(path)
    target = model.state_dict()

    filtered = {k: v for k, v in sd.items() if k in target and target[k].shape == v.shape}
    frac = len(filtered) / max(1, len(target))

    skipped = len(sd) - len(filtered)
    if verbose and skipped:
        print(
            f"  [WARN] {skipped} tensor(es) del checkpoint no aplican "
            f"(shape o nombre distinto) — se ignoran."
        )
    if frac < min_loaded_frac:
        raise RuntimeError(
            f"{path}: sólo {frac:.0%} de las capas del modelo coinciden con el "
            f"checkpoint (mínimo exigido {min_loaded_frac:.0%}). "
            f"Es muy probable que la arquitectura no sea la misma."
        )

    model.load_state_dict(filtered, strict=strict and frac >= 1.0)
    return model, frac


def ckpt_meta(path: str | Path) -> dict[str, Any]:
    """Metadata guardada junto al state_dict (o ``{}`` si es un checkpoint viejo)."""
    import torch

    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if k != STATE_KEY}
    return {}
