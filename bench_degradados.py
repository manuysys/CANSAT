"""
CanSat La Base — Bench v2: matriz degradaciones x mejoradores.
Regímenes: leve (defocus+ruido), movimiento (motion blur), severa.
Mejoras: nada, unsharp suave, denoise+unsharp, espcn, fsrcnn.
Métrica: acuerdo con la segmentación de la imagen nítida + nitidez (Laplaciano).

Uso:  python bench_degradados.py
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

from enhance_image import ensure_model

ONNX = "outputs/cansat_seg_terrain_v2.onnx"   # el modelo de vuelo
IMG = "dataset/loveda_raw/Test/Urban/images_png/5417.png"
if not Path(IMG).exists():
    IMG = "dataset/pruebas/PERSONAS_AUTOS.png"


def motion_kernel(size, angle):
    kern = np.zeros((size, size), np.float32)
    kern[size // 2, :] = 1.0 / size
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    return cv2.warpAffine(kern, M, (size, size))


def degrade_mild(img, rng):
    h, w = img.shape[:2]
    low = cv2.resize(cv2.resize(img, (w // 2, h // 2)), (w, h))
    blur = cv2.GaussianBlur(low, (5, 5), 0)
    return np.clip(blur.astype(np.float32) + rng.normal(0, 3, blur.shape),
                   0, 255).astype(np.uint8)


def degrade_motion(img, rng, k=15, sigma=4):
    blur = cv2.filter2D(img, -1, motion_kernel(k, 35))
    return np.clip(blur.astype(np.float32) + rng.normal(0, sigma, blur.shape),
                   0, 255).astype(np.uint8)


def degrade_severe(img, rng):
    h, w = img.shape[:2]
    low = cv2.resize(cv2.resize(img, (w // 3, h // 3)), (w, h))
    blur = cv2.filter2D(low, -1, motion_kernel(25, 20))
    return np.clip(blur.astype(np.float32) + rng.normal(0, 8, blur.shape),
                   0, 255).astype(np.uint8)


def unsharp(img, s):
    g = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + s, g, -s, 0)


def denoise_unsharp(img):
    return unsharp(cv2.bilateralFilter(img, 7, 50, 50), 0.8)


def sr_back(img, key):
    algo, path = ensure_model(key)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path)
    sr.setModel(algo.lower(), 2)
    return cv2.resize(sr.upsample(img), (img.shape[1], img.shape[0]))


def segment(sess, img):
    small = cv2.resize(img, (320, 320))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], np.float32))
         / np.array([0.229, 0.224, 0.225], np.float32))
    x = x.transpose(2, 0, 1)[np.newaxis, ...]
    return np.argmax(sess.run(["logits"], {"input": x})[0][0], axis=0)


def blur_score(img):
    """Varianza del Laplaciano: más alto = más nítido (métrica de gate)."""
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def main():
    sharp = cv2.imread(IMG)
    rng = np.random.default_rng(42)
    sess = ort.InferenceSession(ONNX)
    ref = segment(sess, sharp)

    degradaciones = {
        "leve": degrade_mild(sharp, rng),
        "movimiento": degrade_motion(sharp, rng),
        "severa": degrade_severe(sharp, rng),
    }

    print(f"{'degradación':<12} {'mejora':<16} {'acuerdo':>8} {'nitidez':>10}")
    for dname, bad in degradaciones.items():
        candidatos = {
            "nada": bad,
            "unsharp_suave": unsharp(bad, 0.5),
            "denoise+unsharp": denoise_unsharp(bad),
            "espcn": sr_back(bad, "espcn_x2"),
            "fsrcnn": sr_back(bad, "fsrcnn_x2"),
        }
        for cname, img in candidatos.items():
            agree = (segment(sess, img) == ref).mean() * 100
            print(f"{dname:<12} {cname:<16} {agree:7.2f}% {blur_score(img):10.1f}")
        print("-" * 52)


if __name__ == "__main__":
    main()
