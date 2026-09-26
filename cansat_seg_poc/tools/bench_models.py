"""
Benchmark de los modelos de vuelo — ms por inferencia en la placa real.

Pensado para correr EN la Raspberry Pi (ARMv6/cv2.dnn) o en la PC (ONNX
Runtime), con la MISMA imagen y midiendo sólo la inferencia (sin pre/post
del pipeline). El JSON de salida es la evidencia que faltaba: hasta que no
se mida en la placa, todos los tiempos son estimaciones.

Uso:
    python tools/bench_models.py                      # 3 corridas por modelo
    python tools/bench_models.py --runs 5 --image tiles/cap.png
    python tools/bench_models.py --out docs/benchmarks/pi_zero_w_sframe.json

El tamaño de entrada sale del nombre (``*_224`` → 224) con el mapa
SIZE_POR_MODELO para el resto (fuente: MODELS.yaml). Modelos con 2 entradas
(siamés) se saltean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

from cansat import preprocess as PP                           # noqa: E402
from cansat.onnxio import OnnxModel                           # noqa: E402

#: Tamaño de entrada por modelo (ver MODELS.yaml). El default es 320.
SIZE_POR_MODELO: dict[str, int] = {
    "cansat_seg_terrain_v2_224.onnx": 224,
    "cansat_seg_terrain_tiny_224.onnx": 224,
    "cansat_flood_specialist_224.onnx": 224,
    "cansat_fire_smoke.onnx": 256,
}

#: Modelos de vuelo (sin el siamés: tiene 2 entradas), ordenados por
#: importancia de vuelo: si la placa corta a mitad, lo medido primero es lo
#: que más importa.
MODELOS_DEFAULT: tuple[str, ...] = (
    "cansat_seg_terrain_v2_224.onnx",
    "cansat_damage_v3_bal.onnx",
    "cansat_flood_specialist_224.onnx",
    "cansat_fire_smoke.onnx",
    "cansat_seg_terrain_v2.onnx",
    "cansat_seg_terrain_tiny_224.onnx",
    "cansat_damage3_mobilenetv2.onnx",
    "cansat_severity.onnx",
)


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def imagen_de_prueba(imagen: str | None, carpeta: str | None) -> Path:
    """Una imagen para el bench: --image, o la primera de la carpeta."""
    if imagen:
        p = Path(imagen)
        if not p.is_file():
            raise FileNotFoundError(f"no existe la imagen: {p}")
        return p
    for d in ([carpeta] if carpeta else []) + ["tiles", "dataset/pruebas"]:
        if not d:
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            found = sorted((ROOT / d).glob(ext))
            if found:
                return found[0]
    raise FileNotFoundError(
        "no encontré imagen de prueba; pasá --image o --carpeta (hay tiles en dist_pi/tiles)")


def medir(modelo: OnnxModel, size: int, bgr: np.ndarray, runs: int) -> dict:
    tensor = PP.preprocess_bgr(bgr, size)
    times: list[float] = []
    for i in range(runs + 1):                      # 1 de warmup
        t0 = time.perf_counter()
        salida = modelo.run({"input": tensor})
        ms = (time.perf_counter() - t0) * 1000.0
        if i > 0:
            times.append(ms)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "ms_min": round(float(arr.min()), 1),
        "ms_media": round(float(arr.mean()), 1),
        "ms_p95": round(float(np.percentile(arr, 95)), 1),
        "n_runs": runs,
        "salida_shape": list(np.asarray(salida).shape),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark de modelos por inferencia")
    ap.add_argument("--models-dir", default="outputs")
    ap.add_argument("--models", nargs="*", default=list(MODELOS_DEFAULT))
    ap.add_argument("--image", default=None)
    ap.add_argument("--carpeta", default=None, help="carpeta con imágenes de prueba")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--backend", default=None,
                    choices=("auto", "onnxruntime", "cv2"),
                    help="forzar backend (en la Pi no hay ORT: cv2)")
    ap.add_argument("--out", default="outputs/bench_models.json")
    ap.add_argument("--nota", default="")
    args = ap.parse_args(argv)

    img = imagen_de_prueba(args.image, args.carpeta)
    bgr = cv2.imread(str(img))
    if bgr is None:
        print(f"[ERROR] no pude leer {img}")
        return 1
    print(f"imagen: {img} {bgr.shape[1]}x{bgr.shape[0]} · {args.runs} corridas/modelo")

    try:
        import onnxruntime as _ort
        ort_version = _ort.__version__
    except ImportError:
        ort_version = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    resultados: dict[str, dict] = {}
    payload = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "script": "tools/bench_models.py",
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cv2": cv2.__version__,
        "onnxruntime": ort_version,
        "imagen": str(img),
        "runs": args.runs,
        "completo": False,
        "resultados": resultados,
        "nota": (args.nota or "Solo inferencia (sin pre/post del pipeline). "
                 "En ARMv6 el backend cae a cv2.dnn por subproceso."),
    }

    def guardar(completo: bool) -> None:
        """Escribe el JSON tras CADA modelo: un corte no pierde lo medido."""
        payload["completo"] = completo
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    for nombre in args.models:
        path = Path(args.models_dir) / nombre
        if not path.is_file():
            print(f"  [skip] no existe {path}", flush=True)
            continue
        # La CARGA también se mide: en ARMv6 parsear 50 MB de ONNX no es gratis.
        t_load = time.perf_counter()
        try:
            modelo = OnnxModel(path, required=True, label=nombre,
                               backend=args.backend or "auto")
        except Exception as exc:
            print(f"  [ERROR] {nombre}: {exc}", flush=True)
            resultados[nombre] = {"error": str(exc)[:200]}
            guardar(False)
            continue
        ms_carga = (time.perf_counter() - t_load) * 1000.0
        size = SIZE_POR_MODELO.get(nombre, 320)
        try:
            m = medir(modelo, size, bgr, args.runs)
        except Exception as exc:
            print(f"  [ERROR] inferencia {nombre}: {exc}", flush=True)
            resultados[nombre] = {"error": str(exc)[:200], "ms_carga": round(ms_carga, 1)}
            guardar(False)
            continue
        m.update({"size": size, "sha256_16": sha256_16(path),
                  "mb": round(path.stat().st_size / 1e6, 1),
                  "backend": modelo.backend, "ms_carga": round(ms_carga, 1)})
        resultados[nombre] = m
        guardar(False)                       # guardado incremental
        print(f"  {nombre:<38} {m['ms_media']:8.1f} ms  "
              f"(p95 {m['ms_p95']:7.1f}) [{m['backend']}] {m['mb']} MB · "
              f"carga {ms_carga:.0f} ms", flush=True)

    guardar(True)
    print(f"  → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
