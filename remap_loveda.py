"""
Remapea las máscaras de LoveDA a nuestras 5 clases.

Etiquetas oficiales de LoveDA:
  0 = no-data       → 255 (ignore)
  1 = background    → 4   (other)
  2 = building      → 1   (building)
  3 = road          → 3   (bare_ground)
  4 = water         → 2   (water)
  5 = barren        → 3   (bare_ground)
  6 = forest        → 0   (vegetation)
  7 = agriculture   → 0   (vegetation)
  255 = sin dato    → 255 (ignore)

Nuestras clases (índices 0-4, 255=ignore):
  0 = vegetation
  1 = building
  2 = water
  3 = bare_ground
  4 = other
  255 = ignore
"""

import os
import sys
import numpy as np
from PIL import Image

RAW_DIR = os.path.join("dataset", "loveda_raw")
OUT_DIR = os.path.join("dataset", "loveda_remapped")

# Mapeo LoveDA → nuestras clases
# Índices: 0=vegetation, 1=building, 2=water, 3=bare_ground, 4=other, 255=ignore
LOVEDA_TO_OURS = np.array([
    255,  # 0: no-data → ignore
    4,    # 1: background → other
    1,    # 2: building → building
    3,    # 3: road → bare_ground
    2,    # 4: water → water
    3,    # 5: barren → bare_ground
    0,    # 6: forest → vegetation
    0,    # 7: agriculture → vegetation
], dtype=np.uint8)

# Valores que van a 255 (ignore)
IGNORE_VALUES = {0, 255}


def find_masks(raw_dir):
    """Busca todas las carpetas masks_png en el dataset descargado."""
    results = []
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        if os.path.basename(dirpath) == "masks_png":
            # Determinar el path relativo para reproducir estructura
            rel = os.path.relpath(dirpath, raw_dir)
            # Verificar que exista la carpeta images_png correspondiente
            images_dir = os.path.join(os.path.dirname(dirpath), "images_png")
            if os.path.isdir(images_dir):
                results.append({
                    "masks_dir": dirpath,
                    "images_dir": images_dir,
                    "rel_path": rel,
                })
    return results


def remap_mask(mask):
    """
    Remapea una máscara de LoveDA a nuestras 5 clases.
    Valores fuera de rango (0-7) se mapean a 255 (ignore).
    """
    remapped = np.full(mask.shape, 255, dtype=np.uint8)

    # Mapear valores válidos (0-7)
    valid = (mask >= 0) & (mask <= 7)
    remapped[valid] = LOVEDA_TO_OURS[mask[valid]]

    # Valores fuera de rango o 255 original → 255 (ignore)
    # (ya están en 255 por defecto)

    return remapped


def main():
    print("[INFO] Remapeando máscaras de LoveDA a nuestras 5 clases...")
    print(f"  Fuente: {os.path.abspath(RAW_DIR)}")
    print(f"  Destino: {os.path.abspath(OUT_DIR)}")
    print()
    print("  Mapeo aplicado:")
    print("    LoveDA 0 (no-data)     → 255 (ignore)")
    print("    LoveDA 1 (background)  → 4   (other)")
    print("    LoveDA 2 (building)    → 1   (building)")
    print("    LoveDA 3 (road)        → 3   (bare_ground)")
    print("    LoveDA 4 (water)       → 2   (water)")
    print("    LoveDA 5 (barren)      → 3   (bare_ground)")
    print("    LoveDA 6 (forest)      → 0   (vegetation)")
    print("    LoveDA 7 (agriculture) → 0   (vegetation)")
    print("    LoveDA 255 (sin dato)  → 255 (ignore)")
    print()

    if not os.path.isdir(RAW_DIR):
        print(f"[ERROR] No existe '{RAW_DIR}'. Ejecutá primero download_loveda.py")
        sys.exit(1)

    mask_dirs = find_masks(RAW_DIR)
    if not mask_dirs:
        print("[ERROR] No se encontraron carpetas masks_png en el dataset.")
        sys.exit(1)

    total_processed = 0
    total_ignore = 0
    total_pixels = 0

    for entry in mask_dirs:
        masks_dir = entry["masks_dir"]
        images_dir = entry["images_dir"]
        rel_path = entry["rel_path"]

        # Crear estructura de salida
        out_masks_dir = os.path.join(OUT_DIR, rel_path)
        os.makedirs(out_masks_dir, exist_ok=True)

        # También copiamos/symlinks de imágenes (no las remapeamos)
        out_images_dir = os.path.join(OUT_DIR, os.path.dirname(rel_path), "images_png")
        os.makedirs(out_images_dir, exist_ok=True)

        mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])

        for i, fname in enumerate(mask_files):
            mask_path = os.path.join(masks_dir, fname)

            # Cargar máscara
            mask_pil = Image.open(mask_path)
            mask = np.array(mask_pil)

            # Si tiene 3 canales, tomar el primero
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            # Contar píxeles de ignore antes del remapeo
            total_pixels += mask.size

            # Remapear
            remapped = remap_mask(mask)

            # Contar píxeles de ignore después del remapeo
            total_ignore += np.sum(remapped == 255)

            # Guardar máscara remapeada
            out_path = os.path.join(out_masks_dir, fname)
            Image.fromarray(remapped, mode='L').save(out_path)

            # Copiar imagen correspondiente (si existe)
            img_path = os.path.join(images_dir, fname)
            out_img_path = os.path.join(out_images_dir, fname)
            if os.path.exists(img_path) and not os.path.exists(out_img_path):
                # Copiar la imagen (no la remapeamos)
                img = Image.open(img_path)
                img.save(out_img_path)

            total_processed += 1

        print(f"  [OK] {rel_path}: {len(mask_files)} máscaras remapeadas")

    print(f"\n[OK] Total: {total_processed} máscaras remapeadas")
    print(f"[OK] Píxeles totales: {total_pixels:,}")
    print(f"[OK] Píxeles ignore (255): {total_ignore:,} ({100*total_ignore/total_pixels:.1f}%)")
    print(f"[OK] Píxeles con clase válida: {total_pixels - total_ignore:,} ({100*(total_pixels-total_ignore)/total_pixels:.1f}%)")
    print(f"\n[OK] Dataset remapeado en: {os.path.abspath(OUT_DIR)}")
    print("[OK] Ejecutá verify_dataset.py para verificar la integridad.")


if __name__ == "__main__":
    main()
