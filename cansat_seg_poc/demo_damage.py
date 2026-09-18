"""
CanSat La Base — Demo del modelo de daños sobre un desastre REAL (tsunami de Palu).
Pinta verde=intacto, rojo=dañado y imprime porcentajes.
Uso:  python demo_damage.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ONNX = "outputs/cansat_damage3_mobilenetv2.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="")
    args = ap.parse_args()

    img_path = args.image
    if not img_path or not Path(img_path).exists():
        cands = sorted(Path("dataset/xbd").rglob("*palu-tsunami*_post_disaster.png"))
        if not cands:
            print("[ERROR] No encontré imagen de Palu; pasá --image")
            raise SystemExit(1)
        img_path = str(cands[len(cands) // 2])   # una del medio de la lista

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    sess = ort.InferenceSession(ONNX)

    small = cv2.resize(img, (320, 320))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD)
    x = x.transpose(2, 0, 1)[np.newaxis, ...]
    pred = np.argmax(sess.run(["logits"], {"input": x})[0][0], axis=0)
    pred = cv2.resize(pred.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)

    overlay = img.copy()
    m1, m2 = pred == 1, pred == 2
    overlay[m1] = (overlay[m1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    overlay[m2] = (overlay[m2] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

    p1, p2 = m1.mean() * 100, m2.mean() * 100
    cv2.rectangle(overlay, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(overlay, f"INTACTO {p1:.1f}%   DANADO {p2:.1f}%",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite("outputs/demo_damage.jpg", overlay)

    print(f"Imagen: {img_path}")
    print(f"  intacto : {p1:.1f}%")
    print(f"  DAÑADO  : {p2:.1f}%")
    print("  Guardado: outputs/demo_damage.jpg")


if __name__ == "__main__":
    main()
