"""
CanSat La Base — Detección de personas y vehículos con YOLO.

Paso 2 de la misión secundaria: contar personas para estimar población expuesta.

ARREGLADO respecto de la versión anterior:
  · ``--model`` default era ``"yolov8n.pt"``, un nombre suelto que ultralytics
    **descarga al CWD** la primera vez. En el día del vuelo, sin red, eso falla.
    Ahora el default es ``models/yolov8n_visdrone.pt`` (que está en el repo, y es
    el que usa ``mission_pipeline.py``) y se valida que exista antes de cargar.
  · ⚠ CAMBIO 2026-09-17: el default pasó de ``yolov8s.pt`` (COCO) al fine-tune
    de **VisDrone** porque en vista aérea la diferencia es enorme. Medido sobre
    ``dataset/pruebas/PERSONAS_AUTOS.png`` a 1280 px: el VisDrone detectó
    **254 personas** y el COCO **42** (los modelos COCO pierden personas
    sub-píxel). Los IDs de clase se derivan del nombre del modelo.
  · Usaba ``imgsz=640`` mientras el pipeline usa ``1280``, y ``yolov8n`` mientras
    el pipeline usa ``yolov8s``: dos configuraciones distintas para la misma
    tarea, así que los conteos de este script no eran comparables con los del
    vuelo. Los defaults ahora coinciden con el pipeline.
  · Los IDs de clase vivían duplicados (constante del módulo + string del CLI).
    Ahora se derivan del ``names`` del modelo cargado, con presets COCO/VisDrone.
  · Variable ``conf`` del loop sombreaba ``args.conf``.
  · Sobrescribía la imagen de entrada en memoria sin copia.

Uso:
    python detect_people.py --num-samples 3
    python detect_people.py --image ruta/a/foto_con_gente.jpg
    python detect_people.py --folder alguna/carpeta --imgsz 1280
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from cansat import paths as PROJ

# Presets de clases. Se validan contra ``model.names`` en runtime.
COCO_PERSON = ("person",)
COCO_VEHICLE = ("bicycle", "car", "motorcycle", "bus", "truck", "van")
VISDRONE_PERSON = ("pedestrian", "people")
VISDRONE_VEHICLE = ("bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus")

DEFAULT_MODEL = "models/yolov8n_visdrone.pt"   # el mismo que usa mission_pipeline.py
DEFAULT_IMGSZ = 1280                           # ídem


def ids_for(names: dict, wanted: tuple[str, ...]) -> set[int]:
    """Resuelve nombres de clase → IDs usando el ``names`` real del modelo."""
    if not names:
        return set()
    lower = {str(v).lower(): k for k, v in names.items()}
    return {lower[w] for w in wanted if w in lower}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Detección de personas y vehículos (YOLO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", default=None, help="una imagen puntual")
    ap.add_argument("--folder", default=None, help="carpeta con imágenes")
    ap.add_argument("--num-samples", type=int, default=3,
                    help="cantidad de imágenes si se muestrea una carpeta")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="pesos YOLO locales (no se descarga nada en runtime)")
    ap.add_argument("--conf", type=float, default=0.25, help="confianza mínima")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help="tamaño interno de proceso (más alto = ve cosas más chicas)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(PROJ.OUTPUTS / "people_detection"))
    ap.add_argument("--save", action="store_true", default=True,
                    help="guardar las imágenes anotadas (default)")
    ap.add_argument("--no-save", dest="save", action="store_false")
    args = ap.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.stderr.write(
            "[ERROR] Falta ultralytics.\n"
            "  PC        : pip install ultralytics\n"
            "  Raspberry : pip install ultralytics   ← pi/guia_pi.md NO lo\n"
            "              instalaba, así que mission_pipeline.py no arrancaba.\n"
            "              Está en requirements-flight.txt.\n")
        return 1

    mp = Path(args.model)
    if not mp.is_file():
        print(f"[ERROR] No existe el modelo: {mp}")
        print("        En el repo hay models/yolov8n_visdrone.pt (default), ")
        print("        models/yolo11n_visdrone.pt. Ultralytics descargaría al CWD")
        print("        si le pasás un nombre suelto, y eso en vuelo (sin red) falla.")
        return 1

    print("=" * 62)
    print("  CanSat La Base — Detección de personas y vehículos")
    print(f"  Modelo : {mp.name}   conf {args.conf}   imgsz {args.imgsz}")
    print("=" * 62)

    model = YOLO(str(mp))
    names = getattr(model, "names", {}) or {}
    visdrone = "visdrone" in mp.name.lower()
    person_ids = ids_for(names, VISDRONE_PERSON if visdrone else COCO_PERSON)
    veh_ids = ids_for(names, VISDRONE_VEHICLE if visdrone else COCO_VEHICLE)
    if not person_ids:
        print(f"  [WARN] El modelo no tiene ninguna clase de persona en {list(names.values())[:12]}")
    print(f"  Personas : {sorted(person_ids)} "
          f"({[names.get(i) for i in sorted(person_ids)]})")
    print(f"  Vehículos: {sorted(veh_ids)} "
          f"({[names.get(i) for i in sorted(veh_ids)]})")

    # ── Lista de imágenes ───────────────────────────────────────────────
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if args.image:
        paths = [Path(args.image)]
        if not paths[0].is_file():
            print(f"[ERROR] No existe la imagen: {paths[0]}")
            return 1
    elif args.folder:
        d = Path(args.folder)
        if not d.is_dir():
            print(f"[ERROR] No existe la carpeta: {d}")
            return 1
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
        if args.num_samples and len(paths) > args.num_samples:
            rng = random.Random(args.seed)
            paths = sorted(rng.sample(paths, args.num_samples))
    else:
        print("[ERROR] Especificá --image o --folder")
        return 1

    if not paths:
        print("[ERROR] No se encontraron imágenes.")
        return 1

    out = Path(args.out_dir)
    if args.save:
        out.mkdir(parents=True, exist_ok=True)

    total_people = total_veh = 0
    por_img: list[tuple[str, int, int]] = []

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [WARN] No se pudo leer {p}")
            continue
        canvas = img.copy()          # antes se dibujaba sobre la entrada

        results = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)
        people = veh = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            cf = float(box.conf[0])
            if cls in person_ids:
                people += 1
            elif cls in veh_ids:
                veh += 1
            else:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if cls in person_ids else (0, 200, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{names.get(cls, cls)} {cf:.2f}"
            cv2.putText(canvas, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if args.save:
            cv2.imwrite(str(out / f"{p.stem}_det.jpg"), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        total_people += people
        total_veh += veh
        por_img.append((p.name, people, veh))
        print(f"  → {p.name}: {people} personas, {veh} vehículos")

    print("=" * 62)
    print(f"  TOTAL: {total_people} personas, {total_veh} vehículos "
          f"en {len(por_img)} imágenes")
    if args.save:
        print(f"  Imágenes con recuadros: {out.resolve()}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
