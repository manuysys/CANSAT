"""
CanSat La Base — Prepara FloodNet (estructura Kaggle Track 1 auto-detectada).
Maneja image/mask, color-masks vs label-masks, y empareja por nombre.
"""
import random
from pathlib import Path

import cv2
import numpy as np

SRC = Path("dataset/floodnet")
DST = Path("dataset/floodnet_remapped")
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
STRIP = ["_lab", "_seg", "_mask", "_masks", "_label", "_labels", "_segm", "_gt"]


def base_stem(s):
    for suf in STRIP:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def nunique(p):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return len(np.unique(g)) if g is not None else 999


def main():
    by_dir = {}
    for p in SRC.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            by_dir.setdefault(p.parent, []).append(p)

    if not by_dir:
        print("[ERROR] No hay imágenes. ¿ZIP sin extraer? Contenido:")
        for p in sorted(SRC.iterdir()):
            print("  ", p.name)
        raise SystemExit(1)

    imgs, masks = {}, {}
    for d, fs in by_dir.items():
        n = d.name.lower()
        if any(k in n for k in ("mask", "label", "seg")):
            kind = "mask"
        elif any(k in n for k in ("image", "img")):
            kind = "image"
        else:
            kind = "mask" if nunique(fs[0]) <= 12 else "image"

        if kind == "mask":
            for f in fs:
                s = base_stem(f.stem.lower())
                old = masks.get(s)
                if old is None or nunique(f) < nunique(old):
                    masks[s] = f          # prefiere label mask indexada
        else:
            for f in fs:
                s = f.stem.lower()
                old = imgs.get(s)
                if old is None or (f.suffix == ".jpg" and old.suffix != ".jpg"):
                    imgs[s] = f           # prefiere el .jpg real

    pairs = [(imgs[s], m) for s, m in masks.items() if s in imgs]
    print(f"Carpetas: {len(by_dir)} | imgs: {len(imgs)} | "
          f"masks: {len(masks)} | pares: {len(pairs)}")
    if not pairs:
        print("[ERROR] Sin pares. Pegame: Get-ChildItem dataset\\floodnet -Recurse -Directory")
        raise SystemExit(1)

    # ══════════════════════════════════════════════════════════════════════
    #  Remapeo según el class_mapping.csv OFICIAL del dataset
    # ══════════════════════════════════════════════════════════════════════
    #  0 Background · 1 Building-flooded · 2 Building-non-flooded
    #  3 Road-flooded · 4 Road-non-flooded · 5 Water · 6 Tree · 7 Vehicle
    #  8 Pool · 9 Grass
    #
    #  ⚠ ESTABA MAL DE DOS FORMAS y el modelo entrenado es inservible:
    #    1) `shift = 1 if gmin == 0 else 0` desplazaba TODAS las clases un
    #       lugar cuando el mask traía el fondo 0 (que es lo normal en
    #       FloodNet). Con el shift, "flood" pasaba a ser la clase 2
    #       (Building-NON-flooded = 2.98 % del dataset) y "agua normal" la
    #       clase 6 (Tree = 16.55 %): el especialista aprendió árboles como
    #       agua y edificios sin inundar como inundación.
    #    2) Sólo se mapeaba Road-flooded (3) y "pool" era en realidad Vehicle
    #       (7). Faltaba Building-flooded (1), que es la mitad del evento.
    #
    #  El histograma medido de las 398 máscaras crudas confirma la
    #  indexación oficial: 1.33 % fondo, 1.60 % building-flooded,
    #  1.90 % road-flooded, 12.36 % water, 16.55 % tree, 57.65 % grass.
    # ══════════════════════════════════════════════════════════════════════
    FLOOD_SRC = (1, 3)     # Building-flooded + Road-flooded
    WATER_SRC = (5,)       # Water (lago/río: agua "normal", no inundación)

    def remap(m):
        out = np.zeros_like(m, dtype=np.uint8)      # 0 = other
        for src in FLOOD_SRC:
            out[m == src] = 1                       # 1 = inundación
        for src in WATER_SRC:
            out[m == src] = 2                       # 2 = agua normal
        return out

    # Chequeo de sanidad del dataset: que los valores observados existan en el
    # mapa oficial. Si aparece algo raro, avisar en vez de descartarlo mudo.
    vistos = set()
    for _, mp in pairs[:80]:
        g = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            vistos.update(v for v in np.unique(g).tolist())
    desconocidos = sorted(v for v in vistos if v > 9)
    if desconocidos:
        print(f"[WARN] valores de máscara fuera del mapa 0-9: {desconocidos}. "
              f"Se ignoran (quedan como 'other'). ¿Variante de FloodNet distinta?")
    faltan = sorted(set(FLOOD_SRC + WATER_SRC) - vistos)
    if faltan:
        print(f"[WARN] clases objetivo sin píxeles en la muestra: {faltan}")

    random.Random(42).shuffle(pairs)
    cut = int(len(pairs) * 0.8)
    for split, sub in [("train", pairs[:cut]), ("val", pairs[cut:])]:
        ib, mb = DST / split / "images", DST / split / "masks"
        ib.mkdir(parents=True, exist_ok=True)
        mb.mkdir(parents=True, exist_ok=True)
        n = 0
        for i, (ip, mp) in enumerate(sub):
            img = cv2.imread(str(ip))
            msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            cv2.imwrite(str(ib / f"fl_{i:05d}.png"), img)
            cv2.imwrite(str(mb / f"fl_{i:05d}.png"), remap(msk))
            n += 1
        print(f"  {split}: {n}")
    print(f"[OK] {DST}")


if __name__ == "__main__":
    main()
