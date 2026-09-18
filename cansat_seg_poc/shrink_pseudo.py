# shrink_pseudo.py
import json
from pathlib import Path
import cv2

man = json.load(open("dataset/pseudo/manifest.json", encoding="utf-8"))
out = Path("dataset/pseudo/img320")
out.mkdir(exist_ok=True)
n = 0
for it in man:
    dst = out / (Path(it["mask"]).stem + ".jpg")
    if dst.exists():
        continue
    img = cv2.imread(it["image"])
    if img is None:
        continue
    cv2.imwrite(str(dst), cv2.resize(img, (320, 320)),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    n += 1
print(f"[OK] {n} imágenes reducidas en {out}")
