"""Reproducibilidad. Antes sólo 6 de ~20 scripts de entrenamiento fijaban semilla."""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = False, workers: int = 0) -> None:
    """
    Fija todas las fuentes de aleatoriedad de una corrida de entrenamiento.

    ``deterministic=True`` además hace reproducibles las convoluciones de
    cuDNN (más lento). ``workers>0`` siembra los workers del DataLoader.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if workers > 0:
            torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def device(prefer: str = "cuda", allow_cpu: bool = True):
    """
    ``torch.device`` con fallback a CPU.

    Reemplaza al ``torch.device("cuda")`` **hardcodeado** de
    ``validate_int8_mission.py``, que hacía imposible correr ese script en una
    máquina sin NVIDIA (y por eso la validación INT8 nunca se pudo reproducir).
    """
    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if (
        prefer.startswith("mps")
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if not allow_cpu:
        raise RuntimeError(f"No hay device {prefer} disponible y allow_cpu=False.")
    return torch.device("cpu")
