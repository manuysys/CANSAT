"""
Descarga los datasets multi-desastre (F2 v3) — CanSat LB135.

1. KATE-PD (terremoto Türkiye 2023, HuggingFace `cscrs/kate-pd`, 630 MB):
   imagen post-desastre 0.3-0.5 m con polígonos/máscaras.
2. CRASAR-U-DROIDs (HuggingFace `CRASAR/CRASAR-U-DROIDs`, CC BY 4.0): se baja
   TODAS las anotaciones (JSON chicos) + un subset de ortomosaicos sUAS
   (los de menos gigapíxeles por evento, priorizando los eventos de test:
   tornado Mayfield, Kilauea, Idalia, incendio Mussett; y algunos de train).

Uso:
    python tools/descargar_datasets_multi.py            # KATE-PD + CRASAR subset
    python tools/descargar_datasets_multi.py --solo-crasar
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

STATS_URL = ("https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs/"
             "raw/main/statistics.csv")
CRASAR = "CRASAR/CRASAR-U-DROIDs"
KATE = "cscrs/kate-pd"
# Eventos de test de CRASAR primero (el modelo nunca los vio) + un par de train.
PRIORIDAD = ["Mayfield Tornado", "Kilauea Eruption", "Hurricane Idalia",
             "Mussett Bayou Fire", "Champlain Towers Collapse",
             "Hurricane Laura", "Hurricane Ida"]
MAX_GPX_POR_ARCHIVO = 0.6
MAX_ARCHIVOS_POR_EVENTO = 3


def descargar_crasar(destino: Path) -> None:
    print("[CRASAR] anotaciones (todas)...")
    snapshot_download(repo_id=CRASAR, repo_type="dataset",
                      local_dir=str(destino),
                      allow_patterns=["*/annotations/*", "format/*",
                                      "statistics.csv", "README.md"])
    data = urllib.request.urlopen(STATS_URL, timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(data)))
    uas = [r for r in rows if "UAS" in (r["Source"] or "").upper()]
    elegidos: dict[str, int] = {}
    total_gpx = 0.0
    for ev in PRIORIDAD:
        for r in sorted((x for x in uas if x["Event"] == ev),
                        key=lambda x: float(x["Gigapixels Counted"] or 1e9)):
            if elegidos.get(ev, 0) >= MAX_ARCHIVOS_POR_EVENTO:
                break
            gpx = float(r["Gigapixels Counted"] or 0)
            if gpx > MAX_GPX_POR_ARCHIVO:
                continue
            split = r["Train/Test"].strip().lower()
            rel = f"{split}/imagery/UAS/{r['Orthomosaic']}"
            try:
                hf_hub_download(repo_id=CRASAR, repo_type="dataset",
                                filename=rel, local_dir=str(destino))
                elegidos[ev] = elegidos.get(ev, 0) + 1
                total_gpx += gpx
                print(f"  ✓ {ev}: {r['Orthomosaic']} ({gpx:.3f} Gpx)")
            except Exception as e:
                print(f"  [WARN] {rel}: {type(e).__name__}: {e}")
    print(f"[CRASAR] {sum(elegidos.values())} ortomosaicos sUAS "
          f"({total_gpx:.1f} Gpx) + anotaciones en {destino}")


def descargar_kate(destino: Path) -> None:
    print("[KATE-PD] descargando (630 MB)...")
    snapshot_download(repo_id=KATE, repo_type="dataset", local_dir=str(destino))
    print(f"[KATE-PD] listo en {destino}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-crasar", action="store_true")
    args = ap.parse_args()
    if not args.solo_crasar:
        descargar_kate(ROOT / "dataset/kate_pd")
    descargar_crasar(ROOT / "dataset/crasar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
