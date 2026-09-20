"""
CanSat La Base — Prepara el dataset del especialista de VÍAS (consulta terrestre).

Fuentes:
  · **FloodNet Track 1 crudo** (clases oficiales): road-flooded (3) y
    road-non-flooded (4) → 1; el resto → 0. Es el dominio UAV/inundación.
  · **LoveDA raw**: road (3) → 1; no-data (0/255) → 255 (ignore). Es el dominio
    satelital 0.3 m, el mismo del modelo de terreno de vuelo.

Salida: ``dataset/vias/manifest_{train,val}.csv`` (name,image,mask,via,dominio)
+ máscaras binarias en ``dataset/vias/masks/``. Las imágenes se REFERENCIAN en
su ubicación original (no se copian).

⚠ Split de FloodNet: el MISMO ``shuffle(42)`` 80/20 que ``prepare_floodnet.py``,
para que el val del especialista de vías sea sobre las mismas 80 imágenes que
el val del especialista de inundación (comparaciones manzana con manzana).
LoveDA usa su split oficial Train/Val y NO participa del gate (dominio
distinto, se reporta aparte).

Uso:
    python prepare_vias.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np

from prepare_floodnet import base_stem, nunique

SRC = Path("dataset/floodnet")
Loveda = Path("dataset/loveda_raw")
DST = Path("dataset/vias")
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}

VIA_FLOODNET = (3, 4)   # Road-flooded + Road-non-flooded (índices oficiales)
VIA_LOVEDA = (3,)       # LoveDA: 3 = road
IGNORE = 255


def descubrir_floodnet() -> list[tuple[Path, Path, str]]:
    """Pares (imagen, máscara, stem) de FloodNet con la heurística de prepare_floodnet."""
    by_dir: dict[Path, list[Path]] = {}
    for p in SRC.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            by_dir.setdefault(p.parent, []).append(p)

    imgs: dict[str, Path] = {}
    masks: dict[str, Path] = {}
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
                    masks[s] = f
        else:
            for f in fs:
                s = f.stem.lower()
                old = imgs.get(s)
                if old is None or (f.suffix == ".jpg" and old.suffix != ".jpg"):
                    imgs[s] = f

    # ⚠ Sin sorted(): el orden de inserción de estos dicts es el que usó
    # prepare_floodnet.py, y el shuffle(42) posterior depende de ese orden.
    # Ordenar acá daría OTRO split que floodnet_remapped.
    return [(imgs[s], m, s) for s, m in masks.items() if s in imgs]


def remap_floodnet(m: np.ndarray) -> np.ndarray:
    """Road-flooded/non-flooded → 1; valores fuera del mapa oficial → 255."""
    out = np.zeros_like(m, dtype=np.uint8)
    out[np.isin(m, VIA_FLOODNET)] = 1
    out[m > 9] = IGNORE
    return out


def remap_loveda(m: np.ndarray) -> np.ndarray:
    """LoveDA raw: road (3) → 1; no-data (0/255) → 255; el resto → 0."""
    out = np.zeros_like(m, dtype=np.uint8)
    out[np.isin(m, VIA_LOVEDA)] = 1
    out[(m == 0) | (m == 255)] = IGNORE
    return out


def _fila(nombre: str, img: Path, mask: Path, n_via: int, dominio: str) -> dict:
    return {
        "name": nombre,
        "image": str(img),
        "mask": str(mask),
        "via": n_via,
        "dominio": dominio,
    }


def _escribir_mascara(dst: Path, remap: np.ndarray) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), remap)
    return int((remap == 1).sum())


def main() -> int:
    if not SRC.is_dir() and not Loveda.is_dir():
        print("[ERROR] No hay dataset/floodnet ni dataset/loveda_raw.")
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    masks_dir = DST / "masks"

    # ── FloodNet (mismo shuffle 42 que prepare_floodnet) ─────────────── #
    filas_train: list[dict] = []
    filas_val: list[dict] = []
    if SRC.is_dir():
        pares = descubrir_floodnet()
        random.Random(42).shuffle(pares)
        cut = int(len(pares) * 0.8)
        print(f"FloodNet: {len(pares)} pares (train {cut} / val {len(pares) - cut})")
        for split, sub in (("train", pares[:cut]), ("val", pares[cut:])):
            destino = filas_train if split == "train" else filas_val
            px_via = px_tot = 0
            for ip, mp, stem in sub:
                img = cv2.imread(str(ip))
                msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if img is None or msk is None:
                    continue
                remap = remap_floodnet(msk)
                nombre = f"floodnet_{stem}"
                n_via = _escribir_mascara(masks_dir / f"{nombre}.png", remap)
                destino.append(_fila(nombre, ip, masks_dir / f"{nombre}.png",
                                     n_via, "floodnet"))
                px_via += n_via
                px_tot += int((remap != IGNORE).sum())
            frac = 100.0 * px_via / max(px_tot, 1)
            print(f"  floodnet/{split}: {len(destino)} imgs · vía {frac:.2f}% de píxeles")

    # ── LoveDA raw (split oficial) ───────────────────────────────────── #
    if Loveda.is_dir():
        for split, destino in (("Train", filas_train), ("Val", filas_val)):
            px_via = px_tot = 0
            n_antes = len(destino)
            for env in ("Urban", "Rural"):
                idir = Loveda / split / env / "images_png"
                mdir = Loveda / split / env / "masks_png"
                if not mdir.is_dir():
                    print(f"  [WARN] falta {mdir}, salteo.")
                    continue
                for ip in sorted(idir.glob("*.png")):
                    mp = mdir / ip.name
                    if not mp.is_file():
                        continue
                    msk = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
                    if msk is None:
                        continue
                    if msk.ndim == 3:
                        msk = msk[:, :, 0]
                    remap = remap_loveda(msk)
                    nombre = f"loveda_{split.lower()}_{env.lower()}_{ip.stem}"
                    n_via = _escribir_mascara(masks_dir / f"{nombre}.png", remap)
                    destino.append(_fila(nombre, ip, masks_dir / f"{nombre}.png",
                                         n_via, "loveda"))
                    px_via += n_via
                    px_tot += int((remap != IGNORE).sum())
            frac = 100.0 * px_via / max(px_tot, 1)
            print(f"  loveda/{split.lower()}: {len(destino) - n_antes} imgs · "
                  f"vía {frac:.2f}% de píxeles")

    for nombre, filas in (("train", filas_train), ("val", filas_val)):
        out = DST / f"manifest_{nombre}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["name", "image", "mask", "via", "dominio"])
            wr.writeheader()
            wr.writerows(filas)
        print(f"[OK] {out} — {len(filas)} filas")
    print(f"[OK] máscaras en {masks_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
