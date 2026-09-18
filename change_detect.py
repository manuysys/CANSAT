"""
CanSat La Base — Change-detection REAL pre/post (xBD).
ANTES: modelo de terreno marca edificios. DESPUES: modelo de daños marca
intacto/dañado. Compara contra la máscara ground-truth de xBD.

Uso:
    python change_detect.py --list        # top 10 pares con más daño
    python change_detect.py               # mejor par automático
    python change_detect.py --name palu-tsunami_00000118_post_disaster
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TERRAIN = "outputs/cansat_seg_terrain_v2.onnx"   # el modelo de vuelo
DAMAGE = "outputs/cansat_damage3_mobilenetv2.onnx"
XBD = Path("dataset/xbd")
MASKS = Path("dataset/xbd_masks")


def prep(img):
    small = cv2.resize(img, (320, 320))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis, ...]


def run(sess, img):
    return np.argmax(sess.run(["logits"], {"input": prep(img)})[0][0], axis=0)


def main():
    ap = argparse.ArgumentParser(description="Change-detection real xBD")
    ap.add_argument("--name", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    manifest = list(csv.DictReader(open(MASKS / "manifest.csv", encoding="utf-8")))
    manifest.sort(key=lambda r: -(int(r["destruido"]) + int(r["mayor"])))
    if args.list:
        for r in manifest[:10]:
            print(f"{r['name']:<45} destr={int(r['destruido']):>7} "
                  f"mayor={int(r['mayor']):>7}")
        return

    if args.name:
        row = next(r for r in manifest if r["name"] == args.name)
    else:
        row = manifest[0]
    stem = row["name"].replace("_post_disaster", "")

    pre_p = next(XBD.rglob(f"{stem}_pre_disaster.png"), None)
    post_p = next(XBD.rglob(f"{stem}_post_disaster.png"), None)
    if pre_p is None or post_p is None:
        print("[ERROR] No encontré el par pre/post.")
        raise SystemExit(1)

    pre = cv2.imread(str(pre_p))
    post = cv2.imread(str(post_p))
    h, w = post.shape[:2]

    sess_t = ort.InferenceSession(TERRAIN)
    sess_d = ort.InferenceSession(DAMAGE)
    tpre = run(sess_t, pre)
    dpost = run(sess_d, post)

    bui_b = float((tpre == 1).mean()) * 100
    int_a = float((dpost == 1).mean()) * 100
    dan_a = float((dpost == 2).mean()) * 100

    # IoU de "dañado" contra el ground truth de xBD
    gt = cv2.imread(str(MASKS / f"{row['name']}.png"), cv2.IMREAD_GRAYSCALE)
    gt_d = (gt >= 2).astype(np.uint8)
    pr_d = cv2.resize((dpost == 2).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    inter = int((pr_d & gt_d).sum())
    union = int((pr_d | gt_d).sum())
    iou = inter / union if union else 0.0

    # Visual: antes | después | overlay
    overlay = post.copy()
    m1 = cv2.resize((dpost == 1).astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    m2 = pr_d.astype(bool)
    overlay[m1] = (overlay[m1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    overlay[m2] = (overlay[m2] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

    W = 480
    panels = []
    for img, tag in [(pre, "ANTES"), (post, "DESPUES"), (overlay, "DETECTADO")]:
        t = cv2.resize(img, (W, int(h * W / w)))
        cv2.putText(t, tag, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        panels.append(t)
    body = cv2.hconcat(panels)
    bar = np.zeros((70, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{stem}  |  ANTES bui {bui_b:.1f}%  ->  "
                     f"DESPUES dan {dan_a:.1f}%  int {int_a:.1f}%",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bar, f"IoU danado vs GT xBD: {iou:.2f}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out = Path(f"outputs/change_{stem}.jpg")
    cv2.imwrite(str(out), cv2.vconcat([bar, body]))

    print(f"Par: {row['name']}")
    print(f"  ANTES   : edificios {bui_b:.1f}%")
    print(f"  DESPUES : intacto {int_a:.1f}% | dañado {dan_a:.1f}%")
    print(f"  IoU dañado vs GT xBD: {iou:.2f}")
    print(f"  Guardado: {out}")


if __name__ == "__main__":
    main()
