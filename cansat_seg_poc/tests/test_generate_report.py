"""Tests de generate_report.py — selección del JSON de métricas (regresión CI).

El informe se regenera en CI (`generate_report.py --check`) y ahí el checkout de
git deja todos los archivos con el mismo mtime: si la selección del JSON "más
reciente" depende del mtime, el reporte cambia entre máquinas y el check falla.
Este test fija el criterio: `generado` del JSON primero, mtime como desempate.
"""

import importlib.util
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "generate_report", ROOT / "generate_report.py")
GR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GR)

VUELO = "outputs/cansat_seg_terrain_v2.onnx"


def _escribir(p: Path, generado: str, modelo: str = VUELO) -> Path:
    p.write_text(json.dumps({"generado": generado, "modelo": modelo}),
                 encoding="utf-8")
    return p


def test_latest_json_usa_generado_y_no_mtime(tmp_path):
    viejo = _escribir(tmp_path / "val_viejo.json", "2026-09-17T00:00:00+00:00")
    nuevo = _escribir(tmp_path / "val_nuevo.json", "2026-09-20T00:00:00+00:00")
    # mtime invertido a propósito: el viejo queda más "nuevo" en disco.
    futuro = time.time() + 100
    os.utime(viejo, (futuro, futuro))
    elegido = GR._latest_json(tmp_path, "val_", preferir=VUELO)
    assert elegido is not None and elegido.name == nuevo.name


def test_latest_json_respeta_preferir(tmp_path):
    _escribir(tmp_path / "val_vuelo.json", "2026-09-17T00:00:00+00:00", VUELO)
    # Un modelo de prueba evaluado DESPUÉS no debe pisar al de vuelo.
    _escribir(tmp_path / "val_cbam.json", "2026-09-21T00:00:00+00:00",
              "outputs/cansat_seg_terrain_v2_cbam.onnx")
    elegido = GR._latest_json(tmp_path, "val_", preferir=VUELO)
    assert elegido is not None and elegido.name == "val_vuelo.json"


def test_latest_json_sin_candidatos(tmp_path):
    assert GR._latest_json(tmp_path, "val_", preferir=VUELO) is None
