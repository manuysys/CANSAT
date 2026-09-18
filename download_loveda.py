"""
Descarga el dataset LoveDA desde Zenodo (registro oficial).
Usa aria2c para descargas rápidas multi-hilo, con fallback a requests.

Registro: https://zenodo.org/records/5706578

Archivos descargados:
  - Train.zip (4.0 GB)
  - Val.zip (2.4 GB)
  - Test.zip (3.1 GB)  ← AGREGADO

Estructura resultante:
  dataset/loveda_raw/Train/Rural/images_png/
  dataset/loveda_raw/Train/Rural/masks_png/
  dataset/loveda_raw/Train/Urban/images_png/
  dataset/loveda_raw/Train/Urban/masks_png/
  dataset/loveda_raw/Val/Rural/images_png/
  dataset/loveda_raw/Val/Rural/masks_png/
  dataset/loveda_raw/Val/Urban/images_png/
  dataset/loveda_raw/Val/Urban/masks_png/
  dataset/loveda_raw/Test/Rural/images_png/     ← AGREGADO (sin máscaras)
  dataset/loveda_raw/Test/Urban/images_png/     ← AGREGADO (sin máscaras)
"""

import os
import sys
import hashlib
import subprocess
import shutil
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("dataset/loveda_raw")
DOWNLOADS_DIR = Path("dataset/downloads")

# URLs oficiales de Zenodo (Registro: 5706578)
ZENO_DO_FILES = {
    "Train.zip": {
        "url": "https://zenodo.org/records/5706578/files/Train.zip?download=1",
        "size_bytes": 4_294_967_296,  # ~4.0 GB
        "md5": "de2b196043ed9b4af1690b3f9a7d558f",
    },
    "Val.zip": {
        "url": "https://zenodo.org/records/5706578/files/Val.zip?download=1",
        "size_bytes": 2_576_980_378,  # ~2.4 GB
        "md5": "84cae2577468ff0b5386758bb386d31d",
    },
    "Test.zip": {  # ← AGREGADO
        "url": "https://zenodo.org/records/5706578/files/Test.zip?download=1",
        "size_bytes": 3_328_599_654,  # ~3.1 GB
        "md5": "a489be0090465e01fb067795d24e6b47",
    },
}


def check_aria2c() -> bool:
    """Verifica si aria2c está instalado."""
    return shutil.which("aria2c") is not None


def compute_md5(filepath: Path, chunk_size: int = 8192 * 16) -> str:
    """Calcula el MD5 de un archivo."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def download_with_aria2c(url: str, dest_path: Path) -> bool:
    """
    Descarga usando aria2c con múltiples conexiones.
    Mucho más rápido y robusto que requests para archivos grandes.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Si ya existe y está completo, saltar
    if dest_path.exists():
        print(f"  [SKIP] Ya existe: {dest_path.name}")
        return True

    # Comando aria2c con opciones optimizadas
    cmd = [
        "aria2c",
        "--continue=true",           # Reanudar descargas
        "--max-connection-per-server=16",  # 16 conexiones paralelas
        "--min-split-size=10M",      # Dividir en chunks de 10MB
        "--split=16",                # 16 splits
        "--file-allocation=none",    # No pre-allocar espacio
        "--console-log-level=warn",  # Menos verbose
        "--summary-interval=10",     # Resumen cada 10s
        "--dir", str(dest_path.parent),
        "--out", dest_path.name,
        url
    ]

    print("  [ARIA2C] Descargando con multi-hilo...")
    print(f"  [CMD] {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] aria2c falló con código {e.returncode}")
        return False
    except Exception as e:
        print(f"\n  [ERROR] Error ejecutando aria2c: {e}")
        return False


def download_with_requests(url: str, dest_path: Path, expected_size: int | None = None, chunk_size: int = 8192 * 16) -> bool:
    """
    Fallback: descarga con requests + tqdm (single-threaded).
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    mode = "wb"
    resume_byte_pos = 0

    # Verificar si el archivo ya existe y está incompleto
    if dest_path.exists():
        current_size = dest_path.stat().st_size
        if expected_size and current_size == expected_size:
            print(f"  [SKIP] Ya existe y está completo: {dest_path.name}")
            return True
        if current_size > 0:
            print(f"  [RESUME] Reanudando desde byte {current_size:,}...")
            headers["Range"] = f"bytes={current_size}-"
            resume_byte_pos = current_size
            mode = "ab"

    try:
        print("  [REQUESTS] Descargando (single-thread)...")
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        if response.status_code == 200 and resume_byte_pos > 0:
            print("  [WARN] Servidor no soporta reanudación, descargando desde el inicio...")
            mode = "wb"
            resume_byte_pos = 0

        total_size = int(response.headers.get("content-length", 0)) + resume_byte_pos
        initial_pos = resume_byte_pos

        with open(dest_path, mode) as f, tqdm(
            total=total_size,
            initial=initial_pos,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"  {dest_path.name}",
            leave=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n  [ERROR] Falló la descarga: {e}")
        return False


def download_file(url: str, dest_path: Path, expected_size: int | None = None) -> bool:
    """
    Descarga usando aria2c si está disponible, sino fallback a requests.
    """
    if check_aria2c():
        print("  [INFO] Usando aria2c (multi-hilo, más rápido)")
        success = download_with_aria2c(url, dest_path)

        # Si aria2c falla, intentar con requests como último recurso
        if not success:
            print("  [WARN] aria2c falló, intentando con requests...")
            # Eliminar archivo parcial si existe
            if dest_path.exists():
                dest_path.unlink()
            return download_with_requests(url, dest_path, expected_size)
        return success
    print("  [WARN] aria2c no instalado, usando requests (más lento)")
    print("  [TIP] Instalá aria2c para descargas más rápidas:")
    print("        Ubuntu/Debian: sudo apt install aria2")
    print("        macOS: brew install aria2")
    print("        Windows: choco install aria2")
    return download_with_requests(url, dest_path, expected_size)


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extrae un archivo ZIP."""
    try:
        print(f"  [EXTRACT] Extrayendo {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  [OK] Extraído en: {extract_to}")
        return True
    except zipfile.BadZipFile:
        print(f"  [ERROR] {zip_path.name} no es un ZIP válido o está corrupto.")
        return False
    except Exception as e:
        print(f"  [ERROR] No se pudo extraer: {e}")
        return False


def find_structure(root: Path) -> dict:
    """
    Busca la estructura del dataset en varias configuraciones posibles.
    Devuelve un dict con las rutas encontradas.
    """
    splits = ["Train", "Val", "Test"]  # ← AGREGADO "Test"
    envs = ["Rural", "Urban"]
    found = {}

    # Estructura 1: root/Train/Rural/images_png/
    for split in splits:
        for env in envs:
            img_dir = root / split / env / "images_png"
            mask_dir = root / split / env / "masks_png"
            if img_dir.is_dir():
                key = f"{split}/{env}"
                found[key] = {
                    "images": img_dir,
                    "masks": mask_dir if mask_dir.is_dir() else None,
                }

    # Estructura 2: root/data/Train/Rural/images_png/
    if not found:
        data_root = root / "data"
        if data_root.is_dir():
            for split in splits:
                for env in envs:
                    img_dir = data_root / split / env / "images_png"
                    mask_dir = data_root / split / env / "masks_png"
                    if img_dir.is_dir():
                        key = f"{split}/{env}"
                        found[key] = {
                            "images": img_dir,
                            "masks": mask_dir if mask_dir.is_dir() else None,
                        }

    # Estructura 3: buscar recursivamente carpetas images_png
    if not found:
        for dirpath, dirnames, filenames in os.walk(root):
            if "images_png" in dirnames:
                img_dir = Path(dirpath) / "images_png"
                mask_dir = Path(dirpath) / "masks_png"
                rel = Path(dirpath).relative_to(root).as_posix()
                key = rel.replace("/", os.sep).replace(os.sep, "/")
                found[key] = {
                    "images": img_dir,
                    "masks": mask_dir if mask_dir.is_dir() else None,
                }

    return found


def main():
    print("=" * 60)
    print("  LoveDA Dataset Downloader (Zenodo)")
    print("=" * 60)
    print("  Fuente: https://zenodo.org/records/5706578")
    print("  Archivos: Train.zip, Val.zip, Test.zip")  # ← ACTUALIZADO
    print(f"  Destino: {OUTPUT_DIR.resolve()}")

    if check_aria2c():
        print("  Descarga: aria2c (multi-hilo, rápido)")
    else:
        print("  Descarga: requests (single-thread, lento)")
        print("  [TIP] Instalá aria2c para mayor velocidad")

    print("=" * 60)
    print()

    # Crear directorios
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Descargar archivos
    for filename, info in ZENO_DO_FILES.items():
        zip_path = DOWNLOADS_DIR / filename

        print(f"\n[{filename}]")
        print(f"  Tamaño: ~{info['size_bytes'] / (1024**3):.1f} GB")
        print(f"  MD5: {info['md5']}")

        # Descargar
        success = download_file(
            url=info["url"],
            dest_path=zip_path,
            expected_size=info["size_bytes"],
        )

        if not success:
            print(f"  [ERROR] No se pudo descargar {filename}.")
            print("  Verificá tu conexión a internet e intentá de nuevo.")
            sys.exit(1)

        # Verificar MD5
        print("  [CHECK] Verificando integridad MD5...")
        actual_md5 = compute_md5(zip_path)
        if actual_md5 != info["md5"]:
            print("  [ERROR] MD5 no coincide!")
            print(f"    Esperado: {info['md5']}")
            print(f"    Actual:   {actual_md5}")
            print(f"  [WARN] Eliminando archivo corrupto: {zip_path}")
            zip_path.unlink()
            sys.exit(1)
        print(f"  [OK] MD5 verificado: {actual_md5}")

        # Extraer
        if not extract_zip(zip_path, OUTPUT_DIR):
            print(f"  [ERROR] No se pudo extraer {filename}.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  Verificación de estructura")
    print("=" * 60)

    # Verificar estructura
    found = find_structure(OUTPUT_DIR)

    if not found:
        print("[ERROR] No se encontró la estructura del dataset.")
        print(f"  Verificá manualmente: {OUTPUT_DIR.resolve()}")
        sys.exit(1)

    print(f"[OK] Estructura encontrada con {len(found)} subdivisiones:\n")
    total_images = 0
    for key, paths in sorted(found.items()):
        n_imgs = len(list(paths["images"].glob("*.png"))) if paths["images"].is_dir() else 0
        has_masks = paths["masks"] is not None and paths["masks"].is_dir()
        n_masks = len(list(paths["masks"].glob("*.png"))) if has_masks else 0
        total_images += n_imgs

        # Indicar si es Test (sin máscaras esperadas)
        if key.startswith("Test"):
            status = "✓ (Test - sin máscaras, solo inferencia)"
        else:
            status = "✓" if has_masks else "✗ SIN MÁSCARAS"

        print(f"  {key}: {n_imgs} imágenes, {n_masks} máscaras {status}")

    print(f"\n[OK] Total: {total_images} imágenes encontradas.")
    print("[OK] Dataset completo (Train + Val + Test) listo para usar.")
    print("[OK] Para remapear clases: python remap_loveda.py")
    print()


if __name__ == "__main__":
    main()
