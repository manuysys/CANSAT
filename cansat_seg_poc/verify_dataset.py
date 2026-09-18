"""
Verifica la integridad del dataset remapeado:
  - Todas las imágenes tienen su máscara correspondiente
  - Todas las máscaras tienen su imagen correspondiente
  - Los valores únicos en las máscaras son correctos (0-4 y 255)
  - No hay máscaras vacías (todo 255)
  - Distribución de clases
"""

import os
import sys
import numpy as np
from PIL import Image
from collections import defaultdict

REMAP_DIR = os.path.join("dataset", "loveda_remapped")

CLASS_NAMES = {
    0: "vegetation",
    1: "building",
    2: "water",
    3: "bare_ground",
    4: "other",
    255: "ignore",
}

# Valores válidos en máscaras remapeadas
VALID_VALUES = {0, 1, 2, 3, 4, 255}


def find_pairs(root):
    """Encuentra pares imagen-máscara en el dataset remapeado."""
    pairs = []
    issues = []

    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "images_png":
            # Buscar la carpeta masks_png correspondiente
            masks_dir = os.path.join(os.path.dirname(dirpath), "masks_png")

            img_files = {f for f in os.listdir(dirpath) if f.lower().endswith('.png')}

            if not os.path.isdir(masks_dir):
                issues.append(f"[WARN] No existe masks_png para: {dirpath}")
                continue

            mask_files = {f for f in os.listdir(masks_dir) if f.lower().endswith('.png')}

            # Imágenes sin máscara
            missing_masks = img_files - mask_files
            if missing_masks:
                issues.append(f"[WARN] {len(missing_masks)} imágenes sin máscara en {os.path.dirname(dirpath)}")
                for f in sorted(missing_masks)[:5]:
                    issues.append(f"    - {f}")

            # Máscaras sin imagen
            missing_images = mask_files - img_files
            if missing_images:
                issues.append(f"[WARN] {len(missing_images)} máscaras sin imagen en {os.path.dirname(dirpath)}")
                for f in sorted(missing_images)[:5]:
                    issues.append(f"    - {f}")

            # Pares válidos
            common = img_files & mask_files
            for f in sorted(common):
                pairs.append((os.path.join(dirpath, f), os.path.join(masks_dir, f)))

    return pairs, issues


def verify_masks(pairs, max_check=None):
    """Verifica valores únicos y distribución de clases en las máscaras."""
    all_unique_values = set()
    class_pixel_counts = defaultdict(int)
    total_pixels = 0
    empty_masks = []
    invalid_value_masks = []
    checked = 0

    limit = max_check if max_check else len(pairs)

    for i, (img_path, mask_path) in enumerate(pairs[:limit]):
        mask_pil = Image.open(mask_path)
        mask = np.array(mask_pil)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Valores únicos en esta máscara
        unique_vals = set(np.unique(mask).tolist())
        all_unique_values.update(unique_vals)

        # Verificar valores inválidos
        invalid_vals = unique_vals - VALID_VALUES
        if invalid_vals:
            invalid_value_masks.append((mask_path, invalid_vals))

        # Contar píxeles por clase
        for val in unique_vals:
            if val in VALID_VALUES:
                class_pixel_counts[val] += int(np.sum(mask == val))
                total_pixels += int(np.sum(mask == val))

        # Verificar máscaras vacías (todo 255)
        if np.all(mask == 255):
            empty_masks.append(mask_path)

        checked += 1

    return all_unique_values, class_pixel_counts, total_pixels, empty_masks, invalid_value_masks, checked


def main():
    print("=" * 60)
    print("VERIFICACIÓN DEL DATASET REMAPEADO")
    print("=" * 60)
    print(f"\n[INFO] Directorio: {os.path.abspath(REMAP_DIR)}")

    if not os.path.isdir(REMAP_DIR):
        print(f"[ERROR] No existe '{REMAP_DIR}'. Ejecutá primero remap_loveda.py")
        sys.exit(1)

    # Paso 1: Encontrar pares
    print("\n[INFO] Buscando pares imagen-máscara...")
    pairs, issues = find_pairs(REMAP_DIR)

    print(f"[OK] {len(pairs)} pares imagen-máscara encontrados")
    if issues:
        print(f"\n[WARN] {len(issues)} advertencias:")
        for issue in issues:
            print(f"  {issue}")

    if not pairs:
        print("[ERROR] No se encontraron pares imagen-máscara.")
        sys.exit(1)

    # Paso 2: Verificar máscaras
    print(f"\n[INFO] Verificando {len(pairs)} máscaras...")
    all_unique, class_counts, total_pixels, empty_masks, invalid_masks, _checked = verify_masks(pairs)

    # Paso 3: Reporte de valores únicos
    print(f"\n[INFO] Valores únicos encontrados en todas las máscaras: {sorted(all_unique)}")
    invalid_values_found = all_unique - VALID_VALUES
    if invalid_values_found:
        print(f"[ERROR] Valores inválidos encontrados: {sorted(invalid_values_found)}")
        print(f"  Valores válidos esperados: {sorted(VALID_VALUES)}")
        for path, vals in invalid_masks[:5]:
            print(f"  Archivo con valores inválidos: {path} → valores: {sorted(vals)}")
    else:
        print(f"[OK] Solo se encontraron valores válidos: {sorted(all_unique)}")

    # Paso 4: Distribución de clases
    print(f"\n[INFO] Distribución de clases ({total_pixels:,} píxeles totales):")
    print(f"  {'Clase':<15} {'Índice':<8} {'Píxeles':<15} {'Porcentaje'}")
    print(f"  {'-'*50}")
    for idx in sorted(class_counts.keys()):
        name = CLASS_NAMES.get(idx, f"desconocido({idx})")
        count = class_counts[idx]
        pct = 100 * count / total_pixels if total_pixels > 0 else 0
        print(f"  {name:<15} {idx:<8} {count:<15,} {pct:>6.2f}%")

    # Paso 5: Máscaras vacías
    if empty_masks:
        print(f"\n[WARN] {len(empty_masks)} máscaras completamente vacías (todo 255):")
        for path in empty_masks[:5]:
            print(f"    - {path}")
    else:
        print("\n[OK] No hay máscaras completamente vacías")

    # Paso 6: Verificar que 255 es realmente ignore/no-data
    ignore_count = class_counts.get(255, 0)
    ignore_pct = 100 * ignore_count / total_pixels if total_pixels > 0 else 0
    print(f"\n[INFO] Píxeles ignore (255): {ignore_count:,} ({ignore_pct:.2f}%)")
    if ignore_pct < 5:
        print("  [OK] Bajo porcentaje de ignore, esperable para no-data/bordes")
    elif ignore_pct > 30:
        print("  [WARN] Alto porcentaje de ignore, verificar que el mapeo sea correcto")
    else:
        print("  [OK] Porcentaje de ignore dentro de rango esperable")

    # Paso 7: Verificar que las clases están balanceadas
    class_percentages = {idx: class_counts[idx] / total_pixels for idx in class_counts if idx != 255}
    if class_percentages:
        max_pct = max(class_percentages.values())
        min_pct = min(class_percentages.values())
        print("\n[INFO] Balance de clases (excluyendo ignore):")
        print(f"  Clase más frecuente: {max_pct*100:.1f}%")
        print(f"  Clase menos frecuente: {min_pct*100:.1f}%")
        if max_pct > 0.7:
            print("  [WARN] Clase dominante muy frecuente, considerar técnicas de balanceo en entrenamiento")

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    all_ok = True
    if invalid_values_found:
        print("  ✗ Valores inválidos encontrados en máscaras")
        all_ok = False
    else:
        print("  ✓ Solo valores válidos (0-4 y 255)")

    if len(pairs) > 0:
        print(f"  ✓ {len(pairs)} pares imagen-máscara encontrados")
    else:
        print("  ✗ No se encontraron pares imagen-máscara")
        all_ok = False

    if empty_masks:
        print(f"  ⚠ {len(empty_masks)} máscaras vacías (todo 255)")
    else:
        print("  ✓ Sin máscaras vacías")

    if 255 in class_counts:
        print(f"  ✓ 255 funciona como ignore/no-data ({ignore_pct:.1f}% de píxeles)")
    else:
        print("  ⚠ No se encontraron píxeles con valor 255 (puede ser correcto si no hay no-data)")

    if all_ok:
        print("\n[OK] Dataset verificado correctamente.")
        print("[OK] Ejecutá visualize_masks.py para ver visualizaciones.")
    else:
        print("\n[ERROR] Hay problemas que resolver antes de continuar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
