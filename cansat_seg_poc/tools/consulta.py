"""
CLI de la Consulta Terrestre (``cansat/consultas.py``).

La estación lo invoca por subproceso (así no necesita numpy/cv2) y también
sirve para depurar en la raíz del proyecto.

Uso:
    python tools/consulta.py --q "área de edificios inundados"
    python tools/consulta.py --q "¿qué fracción de las vías está inundada?"
    python tools/consulta.py --q "..." --region "[[lon,lat],...]" --json

Exit code: 0 consulta soportada, 2 consulta no soportada, 1 error de datos.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cansat.consultas import DatosMision, responder       # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Consulta Terrestre (simbólica)")
    ap.add_argument("--q", required=True, help="consulta en español")
    ap.add_argument("--masks", default="entrega/masks",
                    help="carpeta con las máscaras <src>_<fuente>.png")
    ap.add_argument("--telemetry", default="outputs/mission/telemetry.csv")
    ap.add_argument("--jsonl", default="outputs/mission/telemetry.jsonl",
                    help="JSONL con detecciones (posiciones de personas)")
    ap.add_argument("--region", default=None,
                    help="polígono lon/lat como JSON [[lon,lat],...] o @archivo.json")
    ap.add_argument("--json", action="store_true", help="JSON compacto (sin indentar)")
    args = ap.parse_args(argv)

    region = None
    if args.region:
        raw = (Path(args.region[1:]).read_text(encoding="utf-8")
               if args.region.startswith("@") else args.region)
        try:
            region = json.loads(raw)
        except json.JSONDecodeError as e:
            print(json.dumps({"soportada": False,
                              "motivo": f"--region no es JSON válido: {e}"},
                             ensure_ascii=False))
            return 1

    datos = DatosMision.cargar(args.masks, args.telemetry, args.jsonl)
    res = responder(args.q, datos, region)
    print(json.dumps(res, ensure_ascii=False,
                     indent=(None if args.json else 2)))
    return 0 if res.get("soportada") else 2


if __name__ == "__main__":
    raise SystemExit(main())
