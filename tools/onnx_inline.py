#!/usr/bin/env python3
"""
Empotra los pesos externos (``.onnx.data``) dentro del ``.onnx``.

¿Por qué? Los exportadores de este proyecto escribieron varios modelos con
pesos EXTERNOS: el ``.onnx`` queda de ~0.3 MB y al lado hay un
``<modelo>.onnx.data`` de ~48 MB. ONNX Runtime resuelve ese archivo hermano sin
problema, pero **OpenCV DNN no lo lee**: al cargar el modelo falla con
"kernel_size not specified" (el Conv ve el tensor vacío). Y ``cv2.dnn`` es el
backend de respaldo de ``cansat.onnxio`` para las placas sin wheels de
onnxruntime (Raspberry Pi Zero v1, ARMv6).

Uso:
    python tools/onnx_inline.py outputs/cansat_damage3_mobilenetv2.onnx
    python tools/onnx_inline.py outputs/*.onnx --out-dir models-pi
    python tools/onnx_inline.py outputs/cansat_siamese_damage.onnx --check

``--check`` sólo verifica si el modelo necesita pesos externos, sin escribir.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import onnx
except ImportError:  # pragma: no cover
    sys.exit("[ERROR] Falta el paquete 'onnx':  pip install onnx")


def needs_external(model_path: Path) -> bool:
    """True si el grafo referencia tensores en un archivo externo."""
    m = onnx.load_model(str(model_path), load_external_data=False)
    return any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer)


def inline_one(model_path: Path, out_dir: Path | None = None) -> Path:
    """Carga con datos externos y re-guarda autocontenido. Devuelve la ruta."""
    out_dir = out_dir or model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / model_path.name
    tmp_path = out_dir / (model_path.stem + ".inline.tmp.onnx")

    m = onnx.load(str(model_path))          # resuelve el .data hermano
    onnx.save_model(m, str(tmp_path), save_as_external_data=False)
    # También limpia el modelo en memoria (si quedó algún data_location).
    tmp_path.replace(out_path)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Empotrar pesos externos en un ONNX")
    ap.add_argument("modelos", nargs="+", help="rutas de .onnx")
    ap.add_argument("--out-dir", default=None,
                    help="carpeta de salida (default: junto al original)")
    ap.add_argument("--check", action="store_true",
                    help="sólo informar si necesitan pesos externos")
    ap.add_argument("--overwrite", action="store_true",
                    help="sobrescribir el original (backup .bak)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else None
    errores = 0
    for raw in args.modelos:
        p = Path(raw)
        if not p.is_file():
            print(f"[ERROR] no existe: {p}")
            errores += 1
            continue
        try:
            ext = needs_external(p)
        except Exception as e:
            print(f"[ERROR] {p.name}: no se pudo leer ({type(e).__name__}: {e})")
            errores += 1
            continue

        if args.check:
            data = p.with_suffix(p.suffix + ".data")
            estado = "EXTERNO" if ext else "autocontenido"
            extra = f" (+{data.name})" if ext and data.is_file() else ""
            print(f"{estado:13s} {p.name}{extra}")
            continue

        if not ext:
            # Ya es autocontenido: si pidieron otra carpeta, copiarlo igual
            # (si no, el deploy queda sin el modelo).
            if out_dir is not None and out_dir.resolve() != p.parent.resolve():
                out_dir.mkdir(parents=True, exist_ok=True)
                destino = out_dir / p.name
                shutil.copy2(p, destino)
                print(f"[=] {p.name}: ya era autocontenido -> copiado a {destino}")
            else:
                print(f"[=] {p.name}: ya es autocontenido")
            continue

        try:
            if args.overwrite:
                bak = p.with_suffix(p.suffix + ".bak")
                if not bak.exists():
                    p.replace(bak)
                destino = inline_one(bak, p.parent)
                if destino != p:
                    destino.replace(p)
                resultado = p
            else:
                resultado = inline_one(p, out_dir)
        except Exception as e:
            print(f"[ERROR] {p.name}: {type(e).__name__}: {e}")
            errores += 1
            continue
        print(f"[OK] {p.name}: {p.stat().st_size / 1e6:.1f} MB -> "
              f"{resultado} ({resultado.stat().st_size / 1e6:.1f} MB)")

    if not args.check:
        print("\nVerificá que onnxruntime (PC) y cv2.dnn (Pi) carguen los modelos "
              "resultantes. En la Pi copiá SOLO estos .onnx: no necesitan .data.")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
