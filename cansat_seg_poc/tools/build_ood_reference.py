"""
Construye la referencia OOD desde LoveDA Val — `cansat/ood.py`.

Calcula, sobre imágenes de **Val** (nunca Train: la referencia se usa para
detectar frames que se parecen a *otra cosa*, no para repetir el entrenamiento):

  · distribución media de clases del terreno (máscaras remapeadas, ignora 255);
  · p95 de la divergencia Jensen-Shannon de cada imagen contra esa media
    (ese es el umbral declarado);
  · media/σ de `veg_exg_pct` (ExG) y `shadow_pct` medidos con `cansat/stress`.

Salida: `docs/benchmarks/ood_loveda_val.json` (versionado, chico).

Uso:
    python tools/build_ood_reference.py --max-images 300 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

from cansat import indices as IDX                             # noqa: E402
from cansat import ood as OOD                                 # noqa: E402
from cansat import stress as ST                               # noqa: E402

NOTA = (
    "Referencia OOD construida sobre LoveDA Val (no Train). El umbral JS es el "
    "p95 de la divergencia de cada imagen contra la media del propio Val; los "
    "índices de color usan z-score con 3σ. Es un proxy de alcance, no un "
    "detector formal de novedad."
)


def _mascara_proporciones(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    validos = m != 255
    if not validos.any():
        return None
    counts = np.bincount(m[validos].ravel(), minlength=IDX.NUM_CLASSES)[:IDX.NUM_CLASSES]
    return counts.astype(np.float64) / counts.sum()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Referencia OOD desde LoveDA Val")
    ap.add_argument("--dataset-root", default="dataset")
    ap.add_argument("--max-images", type=int, default=300,
                    help="tope por entorno para los índices de color (0 = todas)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/benchmarks/ood_loveda_val.json")
    args = ap.parse_args(argv)

    root = Path(args.dataset_root)
    rng = np.random.default_rng(args.seed)
    props: list[np.ndarray] = []
    exgs: list[float] = []
    sombras: list[float] = []
    detalle: dict[str, dict] = {}

    for env in ("Urban", "Rural"):
        mdir = root / "loveda_remapped" / "Val" / env / "masks_png"
        idir = root / "loveda_raw" / "Val" / env / "images_png"
        if not mdir.is_dir() or not idir.is_dir():
            print(f"  [WARN] falta {mdir} o {idir}; se saltea {env}")
            continue
        mascaras = sorted(mdir.glob("*.png"))
        n_env = 0
        for mp in mascaras:
            p = _mascara_proporciones(mp)
            if p is not None:
                props.append(p)
                n_env += 1
        imgs = sorted(idir.glob("*.png"))
        if args.max_images and len(imgs) > args.max_images:
            idx = sorted(rng.permutation(len(imgs))[: args.max_images].tolist())
            imgs = [imgs[i] for i in idx]
        for ip in imgs:
            bgr = cv2.imread(str(ip))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            esc = 320.0 / max(h, w)
            if esc < 1.0:
                bgr = cv2.resize(bgr, (max(1, int(w * esc)), max(1, int(h * esc))),
                                 interpolation=cv2.INTER_AREA)
            exgs.append(float(ST.exg_metrics(bgr)["veg_exg_pct"]))
            sombras.append(float(ST.shadow_pct(bgr)))
        detalle[env] = {"mascaras": n_env, "imagenes_indices": len(imgs)}
        print(f"  {env}: {n_env} máscaras · {len(imgs)} imágenes para índices")

    if not props:
        print("[ERROR] sin máscaras de Val; no se puede construir la referencia.")
        return 1

    clases_prom = np.mean(np.stack(props), axis=0)
    js_vals = np.array([OOD.js_divergence(p, clases_prom) for p in props])
    umbral_js = float(np.percentile(js_vals, 95))
    payload = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "script": "tools/build_ood_reference.py",
        "fuente": "LoveDA Val (Rural+Urban)",
        "clases": list(IDX.CLASS_NAMES),
        "clases_prom": [round(float(v), 5) for v in clases_prom],
        "n_mascaras": len(props),
        "js_media": round(float(js_vals.mean()), 5),
        "js_p95": round(umbral_js, 5),
        "umbral_js": round(umbral_js, 5),
        "veg_exg_pct_media": round(float(np.mean(exgs)), 4) if exgs else None,
        "veg_exg_pct_std": round(float(np.std(exgs)), 4) if exgs else None,
        "shadow_pct_media": round(float(np.mean(sombras)), 4) if sombras else None,
        "shadow_pct_std": round(float(np.std(sombras)), 4) if sombras else None,
        "umbral_z": OOD.UMBRAL_Z,
        "detalle": detalle,
        "nota": NOTA,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[OK] {out}")
    print(f"     clases_prom: {payload['clases_prom']}")
    print(f"     JS media {payload['js_media']} · umbral (p95) {payload['umbral_js']}")
    print(f"     exg {payload['veg_exg_pct_media']}±{payload['veg_exg_pct_std']} · "
          f"sombras {payload['shadow_pct_media']}±{payload['shadow_pct_std']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
