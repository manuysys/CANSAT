"""
CanSat La Base — Fine-tune YOLOv11n con VisDrone (más nuevo que v8n, +6% mAP).
El dataset VisDrone ya está descargado en datasets/.

Uso:
    python visdrone_yolo11.py
"""
import shutil
from pathlib import Path

# pyrefly: ignore [missing-import]
from ultralytics import YOLO


def main():
    print("[1/3] Fine-tune YOLOv11n con VisDrone (10 epochs)...")
    model = YOLO("models/yolo11n.pt")   # ruta local: sin red no descarga
    model.train(data="VisDrone.yaml", epochs=10, imgsz=640, batch=8,
                device=0, project="runs/visdrone11", name="train", workers=4)

    print("[2/3] Guardando mejor modelo...")
    best = Path("runs/visdrone11/train/weights/best.pt")
    shutil.copy(best, "models/yolo11n_visdrone.pt")

    print("[3/3] Exportando ONNX...")
    m = YOLO("models/yolo11n_visdrone.pt")
    m.export(format="onnx", imgsz=640)
    print("[OK] models/yolo11n_visdrone.pt (+ ONNX)")
    print("    Uso: --det-model models/yolo11n_visdrone.pt "
          "--det-person 0,1 --det-vehicles 2,3,4,5,6,7,8")


if __name__ == "__main__":
    main()
