"""
CanSat La Base — Altura de edificios por longitud de sombra.
Edificios colapsados proyectan sombras cortas/anómalas.

Uso:
    python shadow_height.py --image IMG --alt 250 --sun-elev 45 --sun-azim 135
"""
import argparse
import math

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--alt", type=float, default=250.0)
    ap.add_argument("--sun-elev", type=float, default=45.0,
                    help="Elevación solar en grados (ver hora del vuelo)")
    ap.add_argument("--sun-azim", type=float, default=135.0,
                    help="Azimut solar (0=N, 90=E, 180=S)")
    ap.add_argument("--onnx", default="outputs/cansat_seg_terrain_v2.onnx")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx)
    img = cv2.imread(args.image)
    h, w = img.shape[:2]

    small = cv2.resize(img, (320, 320))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    seg = np.argmax(sess.run(["logits"], {"input": x})[0][0], axis=0)
    seg = cv2.resize(seg.astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_NEAREST)
    bmask = (seg == 1).astype(np.uint8)
    bui_pct = float((seg == 1).mean()) * 100.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shadow = (gray < otsu * 0.6).astype(np.uint8)

    # Metros por píxel de ESTA imagen: 2·alt·tan(FOV/2) / ancho_en_píxeles.
    # ⚠ FIX: antes era 2·alt·tan(FOV) (FOV completo, no el semiángulo) y además
    #   dividía por 1024 fijo aunque la imagen no midiera 1024: doble error.
    mpp = 2 * args.alt * math.tan(math.radians(33.0 / 2)) / max(1, w)
    az = math.radians(args.sun_azim + 180.0)   # sombra apunta opuesto al sol
    dx, dy = math.sin(az), -math.cos(az)

    contours, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    vis = img.copy()
    heights = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue
        M = cv2.moments(cnt)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        # salir del edificio antes de medir sombra
        step = 0
        while step < 100:
            xi, yi = int(cx + dx * step), int(cy + dy * step)
            if not (0 <= xi < w and 0 <= yi < h):
                break
            if not bmask[yi, xi]:
                break
            step += 1
        L = 0
        gap = 0
        for s2 in range(step, step + 200):
            xi, yi = int(cx + dx * s2), int(cy + dy * s2)
            if not (0 <= xi < w and 0 <= yi < h):
                break
            if shadow[yi, xi]:
                L = s2 - step
                gap = 0
            else:
                gap += 1
                if gap > 6:
                    break
        hgt = L * mpp * math.tan(math.radians(args.sun_elev))
        heights.append(hgt)
        color = (0, 0, 255) if hgt < 3.0 else (0, 255, 0)
        cv2.drawContours(vis, [cnt], -1, color, 2)

    if heights:
        bajas = sum(1 for hh in heights if hh < 3.0)
        print(f"Edificios analizados: {len(heights)}")
        print(f"Altura media: {np.mean(heights):.1f} m")
        print(f"Bajos/colapsados (<3 m): {bajas} "
              f"({100 * bajas / len(heights):.0f}%)")
        if bajas / len(heights) > 0.4 and bui_pct > 15:
            print("→ POSIBLES EDIFICIOS COLAPSADOS EN ZONA URBANA")
        elif bajas / len(heights) > 0.4:
            print("→ Zona rural/cobertizos: alturas bajas NORMALES (no colapso)")
    else:
        print("Sin edificios detectados en el frame.")
    cv2.imwrite("outputs/shadow_height_vis.jpg", vis)
    print("[OK] outputs/shadow_height_vis.jpg (rojo=bajo, verde=normal)")


if __name__ == "__main__":
    main()
