"""
CanSat La Base — Personas/vehículos v2: fine-tune YOLO11n en VisDrone (largo).

Mejora sobre `visdrone_yolo11.py` (10 épocas): 40 épocas con paciencia 15,
batch 16, semilla fija, y export ONNX + intento de export IMX500 (Ultralytics
soporta YOLO11n → IMX oficialmente: packerOut.zip para `imx500-package`).

Uso:
    python visdrone_yolo11_v2.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "runs/visdrone11/train40/weights/best.pt"
OUT_PT = ROOT / "models/yolo11n_visdrone_v2.pt"


def main() -> int:
    print("[1/4] Fine-tune YOLO11n en VisDrone (40 épocas)...")
    model = YOLO(str(ROOT / "models/yolo11n.pt"))
    model.train(data=str(ROOT / "VisDrone.yaml"), epochs=40, imgsz=640,
                batch=16, device=0, project=str(ROOT / "runs/visdrone11"),
                name="train40", workers=8, patience=15, seed=42,
                plots=True)

    print("[2/4] Copiando el mejor modelo...")
    shutil.copy(BEST, OUT_PT)

    print("[3/4] Validación en el split val (mAP)...")
    m = YOLO(str(OUT_PT))
    metrics = m.val(data=str(ROOT / "VisDrone.yaml"), imgsz=640, device=0)
    print(f"      mAP50-95: {metrics.box.map:.4f} · mAP50: {metrics.box.map50:.4f}")

    print("[4/4] Export ONNX + intento IMX500...")
    m.export(format="onnx", imgsz=640)
    try:
        m.export(format="imx", data=str(ROOT / "VisDrone.yaml"), imgsz=640)
        print("[OK] export IMX500: revisar runs/…/yolo11n_imx_model/packerOut.zip")
    except Exception as e:
        print(f"[WARN] export IMX no disponible acá ({type(e).__name__}: {e}).")
        print("       El ONNX queda listo para MCT+imxconv en la PC Linux.")
    print(f"[OK] {OUT_PT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
