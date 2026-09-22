"""
Calibración del clasificador de tipo de desastre — ECE, temperatura y sets APS.

Entrada: el CSV del LOEO con las 7 probabilidades por tile
(``train_disaster_type.py --loeo --probs-todas``; columnas ``evento,y,p_<clase>``).
Cada evento fue evaluado por un modelo que **no lo vio** (leave-one-event-out),
así que calibrar/evaluar acá no tiene fuga.

Qué mide/ajusta:
  · **ECE** (Expected Calibration Error, 15 bins) global, antes/después;
  · **temperature scaling**: ``q ∝ p^(1/T)`` (equivalente a ``softmax(z/T)``
    cuando se parte de logits). T se ajusta minimizando NLL en un split de
    **calibración por eventos** y se evalúa en los eventos restantes. Se ajusta
    una T global y una por **dominio** (prefijo del evento: xbd, crasar,
    kate_pd, rescuenet…), que es el dominio que este dataset sí declara;
  · **APS** (Adaptive Prediction Sets) para alpha 0.01/0.05/0.10: cuantil
    conformal del score acumulado y cobertura/tamaño MEDIDOS en evaluación.

Veredicto honesto: si la temperatura no baja el ECE en los eventos de
evaluación, se reporta que no mejoró (no se adopta).

Uso:
    python -m cansat.calibracion --probs outputs/loeo_probs_todas.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CLASES = ("huracan", "inundacion", "sismo", "incendio", "volcan", "tornado", "otro")


def _probs_cols(fieldnames: list[str]) -> list[str]:
    cols = [f"p_{c}" for c in CLASES]
    faltan = [c for c in cols if c not in fieldnames]
    if faltan:
        raise ValueError(f"faltan columnas de probabilidad: {faltan}")
    return cols


def leer_probs(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """CSV del LOEO → (eventos, probs (N,C), y (N,))."""
    eventos: list[str] = []
    filas: list[list[float]] = []
    etiquetas: list[int] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        cols = _probs_cols(list(rdr.fieldnames or []))
        for r in rdr:
            eventos.append(str(r["evento"]))
            etiquetas.append(int(r["y"]))
            filas.append([float(r[c]) for c in cols])
    return eventos, np.asarray(filas, dtype=np.float64), np.asarray(etiquetas, dtype=np.int64)


def dominio(evento: str) -> str:
    """Dominio declarado = prefijo del evento ('xbd:joplin' → 'xbd')."""
    return evento.split(":", 1)[0]


def _normalizar(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return p / p.sum(axis=-1, keepdims=True)


def aplicar_temperatura(probs: np.ndarray, T: float) -> np.ndarray:
    """``q ∝ p^(1/T)`` normalizado (T=1 no cambia nada; T>1 suaviza)."""
    if T <= 0:
        raise ValueError("la temperatura debe ser > 0")
    return _normalizar(np.power(np.clip(probs, 1e-12, 1.0), 1.0 / T))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = _normalizar(probs)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def ajustar_temperatura(probs: np.ndarray, y: np.ndarray,
                        t_min: float = 0.3, t_max: float = 20.0,
                        pasos: int = 160) -> float:
    """T que minimiza NLL por búsqueda en grilla log-espaciada (sin scipy)."""
    if len(y) == 0:
        return 1.0
    candidatas = np.exp(np.linspace(math.log(t_min), math.log(t_max), pasos))
    perdidas = [nll(aplicar_temperatura(probs, float(t)), y) for t in candidatas]
    return float(candidatas[int(np.argmin(perdidas))])


def aplicar_temperatura_por_clase(probs: np.ndarray,
                                  temps: np.ndarray) -> np.ndarray:
    """``q_i ∝ p_i^(1/T_i)`` normalizado (una T por clase)."""
    temps = np.asarray(temps, dtype=np.float64)
    if temps.ndim != 1 or temps.shape[0] != np.asarray(probs).shape[1]:
        raise ValueError("temps debe ser un vector (C,)")
    if (temps <= 0).any():
        raise ValueError("las temperaturas deben ser > 0")
    pot = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / temps[None, :])
    return _normalizar(pot)


def ajustar_temperatura_por_clase(probs: np.ndarray, y: np.ndarray,
                                  t_min: float = 0.3, t_max: float = 20.0,
                                  pasos: int = 80, min_n: int = 30) -> np.ndarray:
    """Una T por clase: cada T_c minimiza el NLL solo en las muestras con y==c.

    Clases con menos de ``min_n`` muestras en calibración quedan en T=1.0
    (no hay evidencia para ajustarlas y no se inventa).
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y)
    n_clases = probs.shape[1]
    temps = np.ones(n_clases)
    if len(y) == 0:
        return temps
    candidatas = np.exp(np.linspace(math.log(t_min), math.log(t_max), pasos))
    for c in range(n_clases):
        m = y == c
        if m.sum() < min_n:
            continue
        base = np.ones(n_clases)
        perdidas = []
        for t in candidatas:
            base[c] = float(t)
            perdidas.append(nll(aplicar_temperatura_por_clase(probs[m], base), y[m]))
        temps[c] = float(candidatas[int(np.argmin(perdidas))])
    return temps


def ranking_preservado(p_antes: np.ndarray, p_despues: np.ndarray,
                       clase: int) -> bool:
    """¿La transformación preserva el orden de p_clase en todas las parejas?

    La política ``fire_only_v1`` decide por umbral sobre p_incendio: si el
    orden se preserva, el ranking no cambia (solo habría que re-mapear τ).
    """
    a = np.asarray(p_antes)[:, clase]
    b = np.asarray(p_despues)[:, clase]
    # Inversión = pareja (i,j) con a[i] <= a[j] pero b[i] > b[j]: equivale a
    # que b no sea no-decreciente al ordenar por a (sort estable).
    orden = np.argsort(a, kind="stable")
    return bool((np.diff(b[orden]) >= 0).all())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error sobre la confianza top-1."""
    p = _normalizar(probs)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    acierto = (pred == y).astype(np.float64)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    if total == 0:
        return 0.0
    e = 0.0
    for i in range(n_bins):
        lo, hi = bordes[i], bordes[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        e += mask.sum() / total * abs(float(acierto[mask].mean() - conf[mask].mean()))
    return float(e)


def _scores_aps(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Score APS por muestra: prob acumulada (desc) hasta incluir la clase real."""
    p = _normalizar(probs)
    orden = np.argsort(-p, axis=1)
    acum = np.cumsum(np.take_along_axis(p, orden, axis=1), axis=1)
    pos = np.argmax(orden == y[:, None], axis=1)
    return acum[np.arange(len(y)), pos]


def aps_umbral(probs: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> float:
    """Cuantil conformal (1-alpha) del score APS de calibración."""
    s = np.sort(_scores_aps(probs, y))
    n = len(s)
    if n == 0 or not 0.0 < alpha < 1.0:
        return 1.0
    k = math.ceil((n + 1) * (1.0 - alpha))
    return float(s[min(k, n) - 1])


def aps_cobertura_tamano(probs: np.ndarray, y: np.ndarray,
                         tau: float) -> tuple[float, float]:
    """Cobertura (y ∈ set) y tamaño medio del set APS con umbral tau."""
    p = _normalizar(probs)
    orden = np.argsort(-p, axis=1)
    acum = np.cumsum(np.take_along_axis(p, orden, axis=1), axis=1)
    dentro = np.zeros_like(acum, dtype=bool)
    dentro[:, 0] = True
    for i in range(1, acum.shape[1]):
        dentro[:, i] = acum[:, i - 1] < tau          # incluye hasta alcanzar tau
    cubre = dentro[np.arange(len(y)), [int(np.where(orden[i] == y[i])[0][0])
                                      for i in range(len(y))]]
    return float(cubre.mean()), float(dentro.sum(axis=1).mean())


def _split_eventos(eventos: list[str]) -> tuple[list[str], list[str]]:
    """Mitad calibración / mitad evaluación, por evento y determinista."""
    unicos = sorted(set(eventos))
    calib = unicos[::2]
    evalu = unicos[1::2]
    return calib, evalu


def analizar(probs: np.ndarray, y: np.ndarray, eventos: list[str],
             alphas=(0.01, 0.05, 0.10)) -> dict:
    """ECE + temperaturas (global/por dominio) + APS, con split por eventos."""
    calib_ev, eval_ev = _split_eventos(eventos)
    es_calib = np.array([e in set(calib_ev) for e in eventos])
    es_eval = ~es_calib

    p_c, y_c = probs[es_calib], y[es_calib]
    p_e, y_e = probs[es_eval], y[es_eval]

    t_global = ajustar_temperatura(p_c, y_c)
    p_e_glob = aplicar_temperatura(p_e, t_global)

    por_dom: dict[str, dict] = {}
    dom_c = np.array([dominio(e) for e in np.asarray(eventos)[es_calib]])
    dom_e = np.array([dominio(e) for e in np.asarray(eventos)[es_eval]])
    for dom in sorted(set(dom_c.tolist())):
        mc = dom_c == dom
        me = dom_e == dom
        if mc.sum() < 30:
            continue
        t_dom = ajustar_temperatura(p_c[mc], y_c[mc])
        p_dom = aplicar_temperatura(p_c[mc], t_dom)
        entrada = {
            "n_calibracion": int(mc.sum()),
            "temperatura": round(t_dom, 3),
            "nll_antes": round(nll(p_c[mc], y_c[mc]), 4),
            "nll_despues": round(nll(p_dom, y_c[mc]), 4),
        }
        if me.any():
            entrada["n_evaluacion"] = int(me.sum())
            entrada["ece_antes"] = round(ece(p_c[mc], y_c[mc]), 4)
            entrada["ece_despues"] = round(ece(p_dom, y_c[mc]), 4)
        por_dom[dom] = entrada

    aps = {}
    for alpha in alphas:
        tau = aps_umbral(p_c, y_c, alpha)
        cob, tam = aps_cobertura_tamano(p_e, y_e, tau)
        aps[f"alpha_{alpha}"] = {
            "tau": round(tau, 4),
            "cobertura_evaluacion": round(cob, 4),
            "set_promedio": round(tam, 3),
            "n_evaluacion": len(y_e),
        }

    ece_antes = ece(p_e, y_e)
    ece_despues = ece(p_e_glob, y_e)
    return {
        "n_total": len(y),
        "n_eventos": len(set(eventos)),
        "eventos_calibracion": sorted(calib_ev),
        "eventos_evaluacion": sorted(eval_ev),
        "temperatura_global": round(t_global, 3),
        "ece_evaluacion_antes": round(ece_antes, 4),
        "ece_evaluacion_despues": round(ece_despues, 4),
        "nll_evaluacion_antes": round(nll(p_e, y_e), 4),
        "nll_evaluacion_despues": round(nll(p_e_glob, y_e), 4),
        "mejora_ece": round(ece_antes - ece_despues, 4),
        "por_dominio": por_dom,
        "aps": aps,
        "nota": ("T se ajusta sobre eventos de calibración y se evalúa en eventos "
                 "held-out (el CSV ya es LOEO). APS: cobertura y tamaño MEDIDOS. "
                 "Si mejora_ece ≤ 0 la temperatura no aporta y no se adopta."),
    }


def analizar_por_clase(probs: np.ndarray, y: np.ndarray,
                       eventos: list[str]) -> dict:
    """Compara crudo vs T global vs T por clase, con el mismo split por eventos.

    Veredicto: se adopta solo si la T por clase mejora el ECE de evaluación
    respecto de la global Y preserva el ranking de p_incendio (la política
    ``fire_only_v1`` decide por umbral sobre esa probabilidad).
    """
    calib_ev, eval_ev = _split_eventos(eventos)
    es_calib = np.array([e in set(calib_ev) for e in eventos])
    es_eval = ~es_calib

    p_c, y_c = probs[es_calib], y[es_calib]
    p_e, y_e = probs[es_eval], y[es_eval]

    t_global = ajustar_temperatura(p_c, y_c)
    p_e_glob = aplicar_temperatura(p_e, t_global)

    temps = ajustar_temperatura_por_clase(p_c, y_c)
    p_e_clase = aplicar_temperatura_por_clase(p_e, temps)

    idx_fuego = CLASES.index("incendio")
    monotona = ranking_preservado(p_e_glob, p_e_clase, idx_fuego)

    ece_crudo = ece(p_e, y_e)
    ece_glob = ece(p_e_glob, y_e)
    ece_clase = ece(p_e_clase, y_e)
    nll_crudo = nll(p_e, y_e)
    nll_glob = nll(p_e_glob, y_e)
    nll_clase = nll(p_e_clase, y_e)

    por_clase = {}
    for c, nombre in enumerate(CLASES):
        por_clase[nombre] = {
            "temperatura": round(float(temps[c]), 3),
            "n_calibracion": int((y_c == c).sum()),
            "n_evaluacion": int((y_e == c).sum()),
        }

    adoptada = bool(ece_clase < ece_glob and monotona)
    return {
        "n_total": len(y),
        "n_eventos": len(set(eventos)),
        "eventos_calibracion": sorted(calib_ev),
        "eventos_evaluacion": sorted(eval_ev),
        "ece_evaluacion_crudo": round(ece_crudo, 4),
        "ece_evaluacion_global": round(ece_glob, 4),
        "ece_evaluacion_por_clase": round(ece_clase, 4),
        "nll_evaluacion_crudo": round(nll_crudo, 4),
        "nll_evaluacion_global": round(nll_glob, 4),
        "nll_evaluacion_por_clase": round(nll_clase, 4),
        "temperatura_global": round(t_global, 3),
        "temperaturas_por_clase": por_clase,
        "ranking_incendio_preservado": monotona,
        "adoptada": adoptada,
        "nota": ("Se adopta solo si la T por clase baja el ECE en eventos "
                 "held-out respecto de la global y preserva el ranking de "
                 "p_incendio. Si no, se documenta como no adoptada."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibración del tipo de desastre")
    ap.add_argument("--probs", default="outputs/loeo_probs_todas.csv")
    ap.add_argument("--out", default="outputs/calibracion_tipo.json")
    args = ap.parse_args(argv)

    p = Path(args.probs)
    if not p.is_file():
        print(f"[ERROR] no existe {p}. Generá el CSV con:\n"
              f"  python train_disaster_type.py --loeo --probs-todas {p}")
        return 1
    eventos, probs, y = leer_probs(p)
    if len(y) == 0:
        print("[ERROR] CSV vacío")
        return 1
    print(f"  {len(y)} tiles · {len(set(eventos))} eventos · "
          f"{len(set(eventos)) // 2} para calibrar")
    res = analizar(probs, y, eventos)
    res["generado"] = datetime.now(timezone.utc).isoformat()
    res["script"] = "cansat/calibracion.py"
    res["fuente"] = str(p)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"  ECE evaluación: {res['ece_evaluacion_antes']:.4f} → "
          f"{res['ece_evaluacion_despues']:.4f} (T={res['temperatura_global']})")
    print(f"  NLL evaluación: {res['nll_evaluacion_antes']:.4f} → "
          f"{res['nll_evaluacion_despues']:.4f}")
    for k, v in res["aps"].items():
        print(f"  APS {k}: cobertura {v['cobertura_evaluacion'] * 100:.1f}% · "
              f"set medio {v['set_promedio']}")
    print(f"  → {out}")
    if res["mejora_ece"] <= 0:
        print("  [WARN] la temperatura NO mejora el ECE de evaluación: no se adopta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
