"""
Agrega la matriz de tamaños de entrada de terreno (F1.1).

Lee los JSON de ``outputs/metrics/inputsize_*.json`` (producidos por
``evaluate.py --checkpoint outputs/best_terrain_v2.pth --img-size N``) y
escribe ``outputs/metrics/input_matrix.json`` + imprime la tabla.

Uso:
    python tools/bench_input_sizes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cansat  # noqa: E402,F401  (activa UTF-8 en stdout/stderr de Windows)


def main() -> int:
    files = sorted(ROOT.glob("outputs/metrics/inputsize_*.json"))
    if not files:
        print("[ERROR] No hay JSON inputsize_*.json en outputs/metrics/")
        print("        Corré: python evaluate.py --checkpoint outputs/best_terrain_v2.pth "
              "--arch deeplabv3plus --img-size N --out outputs/metrics/inputsize_N.json")
        return 1

    rows = []
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d["metrics"]
        rows.append({
            "img_size": int(d["img_size"]),
            "miou": float(m["miou"]),
            "pixel_acc": float(m.get("pixel_acc", 0.0)),
            "segundos": float(d["segundos"]),
            "img_s": round(d["n_images"] / max(d["segundos"], 1e-9), 1),
            "n_images": int(d["n_images"]),
            "modelo_sha256_16": d.get("modelo_sha256_16", "?"),
            "iou_por_clase": {
                k: round(float(v["iou"]), 4)
                for k, v in (m.get("por_clase") or {}).items()
            } or None,
        })
    rows.sort(key=lambda r: r["img_size"])

    base = next((r for r in rows if r["img_size"] == 320), rows[-1])
    print("=" * 78)
    print("  MATRIZ DE TAMAÑOS DE ENTRADA — terreno v2 (Val completo)")
    print("=" * 78)
    print(f"  {'size':>4} | {'mIoU':>6} | {'Δ vs 320':>8} | {'pixAcc':>7} | "
          f"{'s':>6} | {'img/s':>6} | n")
    for r in rows:
        delta = (r["miou"] - base["miou"]) * 100
        print(f"  {r['img_size']:>4} | {r['miou'] * 100:>5.2f}% | {delta:>+7.2f}  | "
              f"{r['pixel_acc'] * 100:>6.2f}% | {r['segundos']:>6.1f} | "
              f"{r['img_s']:>6.1f} | {r['n_images']}")

    out = ROOT / "outputs/metrics/input_matrix.json"
    payload = {
        "script": "tools/bench_input_sizes.py",
        "modelo": "outputs/best_terrain_v2.pth",
        "modelo_sha256_16": base["modelo_sha256_16"],
        "nota": ("mIoU medida con evaluate.py sobre LoveDA Val completo; "
                 "los segundos son de la RTX 5060, NO de la Pi Zero W."),
        "filas": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n  → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
