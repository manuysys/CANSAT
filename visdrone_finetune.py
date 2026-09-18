"""
CanSat La Base — Fine-tune YOLOv8n con VisDrone COMPLETO (detección aérea).
Ultralytics descarga y convierte el dataset automáticamente (~4 GB, 1 vez).

Uso:
    python visdrone_finetune.py
"""
import shutil
from pathlib import Path

# pyrefly: ignore [missing-import]
from ultralytics import YOLO


def main():
    print("[1/3] Fine-tune YOLOv8n con VisDrone (10 epochs, ~30-50 min)...")
    model = YOLO("models/yolov8n.pt")   # ruta local: sin red no descarga
    model.train(data="VisDrone.yaml", epochs=10, imgsz=640, batch=8,
                device=0, project="runs/visdrone", name="train", workers=4)

    print("[2/3] Guardando mejor modelo...")
    best = Path("runs/visdrone/train/weights/best.pt")
    shutil.copy(best, "models/yolov8n_visdrone.pt")

    print("[3/3] Exportando ONNX...")
    m = YOLO("models/yolov8n_visdrone.pt")
    m.export(format="onnx", imgsz=640)
    print("[OK] models/yolov8n_visdrone.pt (+ ONNX en models/)")
    print("    Clases VisDrone: 0=pedestrian 1=people 2=bicycle 3=car "
          "4=van 5=truck 6=tricycle 7=awning 8=motor 9=others")


if __name__ == "__main__":
    main()
