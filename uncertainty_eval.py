"""
CanSat La Base — Incertidumbre de predicción (varianza TTA).
Frames con alta incertidumbre → "BAJA CONFIANZA" (no alertar).

Uso:
    python uncertainty_eval.py --folder dataset/loveda_raw/Test/Urban/images_png --frames 12
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax_np(x):
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def prep(bgr, size=320):
    small = cv2.resize(bgr, (size, size))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD) \
        .transpose(2, 0, 1)[np.newaxis, ...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--thresh", type=float, default=0.03,
                    help="Varianza media sobre la cual = BAJA CONFIANZA")
    ap.add_argument("--onnx", default="outputs/cansat_seg_terrain_v2.onnx")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx)
    imgs = sorted(Path(args.folder).glob("*.png"))
    random.Random(42).shuffle(imgs)

    print(f"{'frame':>8s} | {'incertidumbre':>13s} | confianza")
    print("-" * 45)
    n_low = 0
    for p in imgs[:args.frames]:
        bgr = cv2.imread(str(p))
        probs = []
        for im, flip in [(bgr, None), (cv2.flip(bgr, 1), "h"),
                         (cv2.flip(bgr, 0), "v"), (cv2.flip(bgr, -1), "hv")]:
            lg = sess.run(["logits"], {"input": prep(im)})[0]
            if flip == "h":
                lg = lg[:, :, :, ::-1]
            elif flip == "v":
                lg = lg[:, :, ::-1, :]
            elif flip == "hv":
                lg = lg[:, :, ::-1, ::-1]
            probs.append(softmax_np(lg[0]))
        var = np.var(np.stack(probs), axis=0).mean()
        low = var > args.thresh
        n_low += low
        print(f"{p.stem:>8s} | {var:13.4f} | "
              f"{'BAJA' if low else 'OK'}")
    print("-" * 45)
    print(f"Frames con BAJA CONFIANZA: {n_low}/{min(args.frames, len(imgs))}")
    print("(en post-vuelo, descartar o marcar estos frames del summary)")


if __name__ == "__main__":
    main()
