"""
Genera visualizaciones del dataset remapeado:
  - Imagen original
  - Máscara coloreada
  - Overlay (imagen + máscara con transparencia)

Genera al menos 10 visualizaciones de cada split.
Guarda en: dataset/visualizations/
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # No mostrar ventanas, solo guardar archivos
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

REMAP_DIR = os.path.join("dataset", "loveda_remapped")
VIS_DIR = os.path.join("dataset", "visualizations")

# Definición de clases
CLASS_NAMES = {
    0: "vegetation",
    1: "building",
    2: "water",
    3: "bare_ground",
    4: "other",
    255: "ignore",
}

# Colores para cada clase (RGB, 0-255)
CLASS_COLORS = {
    0: (34, 139, 34),     # vegetation - verde
    1: (220, 20, 60),     # building - rojo
    2: (30, 144, 255),    # water - azul
    3: (160, 82, 45),     # bare_ground - marrón
    4: (128, 128, 128),   # other - gris
    255: (0, 0, 0),       # ignore - negro (transparente en overlay)
}

# Colormap para matplotlib
CMAP_COLORS = [
    (34/255, 139/255, 34/255, 1.0),    # vegetation
    (220/255, 20/255, 60/255, 1.0),    # building
    (30/255, 144/255, 255/255, 1.0),   # water
    (160/255, 82/255, 45/255, 1.0),    # bare_ground
    (128/255, 128/255, 128/255, 1.0),  # other
    (0, 0, 0, 0.0),                     # ignore - transparente
]
CMAP = ListedColormap(CMAP_COLORS)


def find_pairs(root):
    """Encuentra pares imagen-máscara."""
    pairs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "images_png":
            masks_dir = os.path.join(os.path.dirname(dirpath), "masks_png")
            if not os.path.isdir(masks_dir):
                continue
            img_files = {f for f in os.listdir(dirpath) if f.lower().endswith('.png')}
            mask_files = {f for f in os.listdir(masks_dir) if f.lower().endswith('.png')}
            common = sorted(img_files & mask_files)
            for f in common:
                pairs.append((os.path.join(dirpath, f), os.path.join(masks_dir, f)))
    return pairs


def colorize_mask(mask):
    """Convierte una máscara de índices a imagen coloreada."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in CLASS_COLORS.items():
        colored[mask == idx] = color
    return colored


def create_overlay(image, mask, alpha=0.5):
    """Crea overlay de imagen + máscara coloreada."""
    colored_mask = colorize_mask(mask)

    # Crear máscara de validez (no ignorar)
    valid_mask = (mask != 255).astype(np.float32)

    # Aplicar overlay solo donde hay clase válida
    overlay = image.copy().astype(np.float32)
    for c in range(3):
        overlay[:, :, c] = (1 - alpha * valid_mask) * overlay[:, :, c] + alpha * valid_mask * colored_mask[:, :, c]

    return overlay.astype(np.uint8)


def create_visualization(img_path, mask_path, output_path, title=""):
    """Genera una figura con imagen original + máscara + overlay."""
    # Cargar imagen
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Cargar máscara
    mask_pil = Image.open(mask_path)
    mask = np.array(mask_pil)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Redimensionar imagen al tamaño de la máscara si es necesario
    if img_np.shape[:2] != mask.shape:
        img = img.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
        img_np = np.array(img)

    # Generar visualizaciones
    colored_mask = colorize_mask(mask)
    overlay = create_overlay(img_np, mask, alpha=0.5)

    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Imagen original
    axes[0].imshow(img_np)
    axes[0].set_title("Imagen Original", fontsize=12)
    axes[0].axis("off")

    # Máscara coloreada
    axes[1].imshow(colored_mask)
    axes[1].set_title("Máscara (5 clases)", fontsize=12)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (α=0.5)", fontsize=12)
    axes[2].axis("off")

    # Leyenda
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=(c[0]/255, c[1]/255, c[2]/255), label=CLASS_NAMES[i])
        for i, c in CLASS_COLORS.items() if i != 255
    ]
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 0), alpha=0.3, label="ignore (255)"))
    axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8,
                   bbox_to_anchor=(1.02, -0.15), ncol=3)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def select_diverse_pairs(pairs, n=10):
    """
    Selecciona pares diversos para visualización.
    Intenta incluir ejemplos de diferentes subdivisiones (Rural/Urban, Train/Val).
    """
    # Agrupar por carpeta
    from collections import defaultdict
    by_folder = defaultdict(list)
    for img_path, mask_path in pairs:
        folder = os.path.dirname(os.path.dirname(img_path))  # carpeta padre
        by_folder[folder].append((img_path, mask_path))

    # Distribuir proporcionalmente
    selected = []
    n_folders = len(by_folder)
    per_folder = max(1, n // n_folders)

    for folder, folder_pairs in sorted(by_folder.items()):
        # Seleccionar espaciados para obtener diversidad
        step = max(1, len(folder_pairs) // per_folder)
        selected.extend(folder_pairs[::step][:per_folder])

    # Si no hay suficientes, completar con más pares
    if len(selected) < n:
        remaining = [p for p in pairs if p not in selected]
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]


def main():
    print("[INFO] Generando visualizaciones del dataset remapeado...")
    print(f"  Fuente: {os.path.abspath(REMAP_DIR)}")
    print(f"  Salida: {os.path.abspath(VIS_DIR)}")

    if not os.path.isdir(REMAP_DIR):
        print(f"[ERROR] No existe '{REMAP_DIR}'. Ejecutá primero remap_loveda.py")
        sys.exit(1)

    # Crear directorio de salida
    os.makedirs(VIS_DIR, exist_ok=True)

    # Encontrar pares
    print("\n[INFO] Buscando pares imagen-máscara...")
    pairs = find_pairs(REMAP_DIR)
    print(f"[OK] {len(pairs)} pares encontrados")

    if not pairs:
        print("[ERROR] No se encontraron pares imagen-máscara.")
        sys.exit(1)

    # Seleccionar pares diversos
    n_vis = min(12, len(pairs))
    selected = select_diverse_pairs(pairs, n=n_vis)
    print(f"[INFO] Generando {len(selected)} visualizaciones...")

    # Generar visualizaciones
    for i, (img_path, mask_path) in enumerate(selected):
        # Nombre de archivo descriptivo
        rel_img = os.path.relpath(img_path, REMAP_DIR)
        # Extraer información del path para el nombre
        parts = rel_img.replace(os.sep, "/").split("/")
        split_name = parts[0] if len(parts) > 0 else "unknown"
        env_name = parts[1] if len(parts) > 1 else "unknown"
        img_name = os.path.splitext(parts[-1])[0]

        output_name = f"{i+1:02d}_{split_name}_{env_name}_{img_name}.png"
        output_path = os.path.join(VIS_DIR, output_name)

        title = f"{split_name}/{env_name} - {img_name}"

        create_visualization(img_path, mask_path, output_path, title=title)
        print(f"  [OK] {output_name}")

    print(f"\n[OK] {len(selected)} visualizaciones generadas en: {os.path.abspath(VIS_DIR)}")
    print("\n[OK] Revisá visualmente que las máscaras se correspondan con las imágenes.")
    print("[OK] Clases esperadas:")
    print("  Verde   = vegetation (forest + agriculture)")
    print("  Rojo    = building")
    print("  Azul    = water")
    print("  Marrón  = bare_ground (road + barren)")
    print("  Gris    = other (background)")
    print("  Negro   = ignore (no-data)")


if __name__ == "__main__":
    main()
