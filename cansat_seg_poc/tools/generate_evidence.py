"""
Genera las imágenes de evidencia para el informe de IA — CanSat LB135.

Produce comparativas lado a lado (imagen | ground truth | predicción) de cada
modelo, más una demo de bruma. Salida: `docs/evidencia/*.png`.

Uso:
    python tools/generate_evidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cansat                                                       # noqa: E402,F401
import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402

from cansat import onnxio                                           # noqa: E402
from cansat import preprocess as PP                                 # noqa: E402

OUT = ROOT / "docs" / "evidencia"

# Paletas BGR por tarea
PAL_TERRENO = {0: (80, 200, 80), 1: (60, 60, 220), 2: (220, 140, 60),
               3: (90, 160, 200), 4: (150, 150, 150)}
PAL_DANO = {0: (70, 70, 70), 1: (80, 200, 80), 2: (60, 60, 220)}
PAL_FLOOD = {0: (70, 70, 70), 1: (220, 120, 40), 2: (200, 180, 60)}
PAL_FUEGO = {0: (70, 70, 70), 1: (40, 140, 255), 2: (210, 210, 210)}
PAL_SEV = {0: (70, 70, 70), 1: (80, 200, 80), 2: (60, 220, 220),
           3: (40, 150, 250), 4: (50, 50, 220)}


def colorizar(mask: np.ndarray, paleta: dict) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), np.uint8)
    for k, c in paleta.items():
        out[mask == k] = c
    return out


def predecir(onnx: str, bgr: np.ndarray) -> np.ndarray:
    sess = onnxio.load_required(str(ROOT / onnx), "evidencia")
    size = sess.size_px or 320
    x = PP.preprocess_bgr(bgr, size)
    pred = np.argmax(sess.run({"input": x})[0], axis=0)
    return cv2.resize(pred.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def fila(titulos_imgs: list[tuple[str, np.ndarray]], path: Path, escala=1.0):
    """Concatena imágenes (con su título dibujado) en una fila."""
    altos = [im.shape[0] for _t, im in titulos_imgs]
    h = max(altos)
    partes = []
    for titulo, im in titulos_imgs:
        if im.shape[0] != h:
            esc = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * esc), h))
        barra = np.full((34, im.shape[1], 3), 28, np.uint8)
        cv2.putText(barra, titulo, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 1, cv2.LINE_AA)
        partes.append(np.vstack([barra, im]))
    out = np.hstack(partes)
    if escala != 1.0:
        out = cv2.resize(out, None, fx=escala, fy=escala)
    cv2.imwrite(str(path), out)
    print(f"  → {path.relative_to(ROOT)}  ({out.shape[1]}x{out.shape[0]})")


def ev_terreno():
    img_p = next((ROOT / "dataset/loveda_remapped/Val/Urban/images_png").glob("*.png"))
    msk_p = next((ROOT / "dataset/loveda_remapped/Val/Urban/masks_png").glob("*.png"))
    img = cv2.imread(str(img_p))
    gt = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
    pred = predecir("outputs/cansat_seg_terrain_v2.onnx", img)
    fila([("imagen", img), ("ground truth", colorizar(gt, PAL_TERRENO)),
          ("predicción (v2@320)", colorizar(pred, PAL_TERRENO))],
         OUT / "01_terreno_gt_vs_pred.png")


def ev_dano():
    man = ROOT / "dataset/rescuenet_tiles/manifest_val.csv"
    import csv
    row = next(csv.DictReader(man.open(encoding="utf-8")))
    img = cv2.imread(str(ROOT / row["image"]))
    gt = cv2.imread(str(ROOT / row["mask"]), cv2.IMREAD_GRAYSCALE)
    pred = predecir("outputs/cansat_damage_v3_bal.onnx", img)
    fila([("imagen (UAV RescueNet)", img), ("GT daño", colorizar(gt, PAL_DANO)),
          ("predicción (two-stage UAV)", colorizar(pred, PAL_DANO))],
         OUT / "02_dano_gt_vs_pred.png", escala=0.8)


def ev_flood():
    img_p = sorted((ROOT / "dataset/floodnet_remapped/val/images").glob("*.png"))[0]
    msk_p = ROOT / "dataset/floodnet_remapped/val/masks" / img_p.name
    img = cv2.imread(str(img_p))
    gt = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
    esc = 720.0 / max(img.shape[:2])
    if esc < 1.0:
        img = cv2.resize(img, None, fx=esc, fy=esc)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    pred = predecir("outputs/cansat_flood_specialist_224.onnx", img)
    fila([("imagen (FloodNet UAV)", img), ("GT", colorizar(gt, PAL_FLOOD)),
          ("predicción (flood 224)", colorizar(pred, PAL_FLOOD))],
         OUT / "03_flood_gt_vs_pred.png")


def ev_fuego():
    man = ROOT / "dataset/fire_smoke/manifest_test.csv"
    import csv
    row = next(csv.DictReader(man.open(encoding="utf-8")))
    img = cv2.imread(str(ROOT / row["image"]))
    gt = cv2.imread(str(ROOT / row["mask"]), cv2.IMREAD_GRAYSCALE)
    pred = predecir("outputs/cansat_fire_smoke.onnx", img)
    fila([("imagen (FLAME UAV)", img), ("GT fuego/humo", colorizar(gt, PAL_FUEGO)),
          ("predicción", colorizar(pred, PAL_FUEGO))],
         OUT / "04_fuego_humo_gt_vs_pred.png")


def ev_bruma():
    from cansat import stress as ST
    img_p = next((ROOT / "dataset/loveda_remapped/Val/Rural/images_png").glob("*.png"))
    img = cv2.imread(str(img_p))
    img = cv2.resize(img, (512, 512))
    bruma = np.clip(img.astype(np.float32) * 0.30 + 255 * 0.70 * 0.9,
                    0, 255).astype(np.uint8)
    m1, m2 = ST.haze_metrics(img), ST.haze_metrics(bruma)
    t1 = f"clara: haze {m1['haze_pct']:.1f}% · vis {m1['visibility']:.2f}"
    t2 = f"bruma sintética: haze {m2['haze_pct']:.1f}% · vis {m2['visibility']:.2f}"
    fila([(t1, img), (t2, bruma)], OUT / "05_bruma_metrica.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Evidencia en {OUT}")
    for fn in (ev_terreno, ev_dano, ev_flood, ev_fuego, ev_bruma):
        try:
            fn()
        except Exception as e:
            print(f"  [WARN] {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
