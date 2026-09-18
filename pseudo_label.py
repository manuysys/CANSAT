"""
CanSat La Base — Pseudo-labels con teacher ensemble (B5+destilado+flood)
sobre imágenes sin label, con filtro de confianza por píxel y por frame.

Uso:
    python pseudo_label.py --max 4000
Genera: dataset/pseudo/masks/*.png + dataset/pseudo/manifest.json
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

B5 = "outputs/cansat_seg_terrain_segformer_b5_512.onnx"
T2 = "outputs/cansat_seg_terrain_v2.onnx"
FL = "outputs/cansat_flood_specialist.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep(img, size):
    small = cv2.resize(img, (size, size))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis, ...]


def softmax_np(x):
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def pool_images():
    files = []
    roots = ["dataset/loveda_raw/Test/Rural/images_png",
             "dataset/loveda_raw/Test/Urban/images_png",
             "dataset/floodnet"]
    for r in roots:
        p = Path(r)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if (f.suffix.lower() in (".jpg", ".png", ".jpeg")
                    and not any("mask" in q.lower() for q in f.parts)):
                files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=4000)
    ap.add_argument("--pix-conf", type=float, default=0.90)
    ap.add_argument("--frame-keep", type=float, default=0.70)
    ap.add_argument("--out", default="dataset/pseudo")
    args = ap.parse_args()

    sess_b5 = ort.InferenceSession(B5)
    sess_t = ort.InferenceSession(T2)
    sess_f = ort.InferenceSession(FL)

    files = pool_images()
    random.seed(7)
    random.shuffle(files)
    files = files[:args.max]
    print(f"[OK] pool: {len(files)} imágenes sin label")

    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        S = 512
        P = 0.55 * softmax_np(sess_b5.run(["logits"], {"input": prep(img, S)})[0][0])
        pt = softmax_np(sess_t.run(["logits"], {"input": prep(img, 320)})[0][0])
        P += 0.25 * np.stack([cv2.resize(m, (S, S)) for m in pt])
        pf = softmax_np(sess_f.run(["logits"], {"input": prep(img, 320)})[0][0])
        pf = np.stack([cv2.resize(m, (S, S)) for m in pf])
        P[2] += 0.20 * (pf[1] + pf[2])
        Pn = P / P.sum(axis=0, keepdims=True)
        conf = Pn.max(axis=0)
        seg = Pn.argmax(axis=0).astype(np.uint8)
        keep = conf >= args.pix_conf
        if keep.mean() < args.frame_keep:
            continue
        mask = np.where(keep, seg, 255).astype(np.uint8)
        mp = out / "masks" / f"{f.stem}_{i:05d}.png"
        cv2.imwrite(str(mp), mask)
        manifest.append({"image": str(f).replace("\\", "/"),
                         "mask": str(mp).replace("\\", "/"),
                         "keep": round(float(keep.mean()), 3)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(files)} → {len(manifest)} frames útiles")

    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(f"[OK] {len(manifest)} pseudo-frames en {out}")


if __name__ == "__main__":
    main()
