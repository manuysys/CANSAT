"""
CanSat La Base — Descarga FloodNet (aéreo post-inundación, dominio drone).
Intenta con Kaggle; si no, instrucciones manuales.
"""
import subprocess
import sys
from pathlib import Path

DST = Path("dataset/floodnet")
DST.mkdir(parents=True, exist_ok=True)

try:
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                    "-d", "aletbm/aerial-imagery-dataset-floodnet-challenge",
                    "-p", str(DST), "--unzip"], check=True)
    print("[OK] FloodNet descargado con Kaggle")
except Exception:
    print("[!] Kaggle no configurado. Descarga manual:")
    print("    1. https://www.kaggle.com/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge")
    print("    2. Download → extraer el ZIP en dataset/floodnet/")
    print("    (o desde https://roc-hci.github.io/NADBenchmarks/FloodNet.html)")
