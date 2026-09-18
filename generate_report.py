"""
CanSat La Base — Generador del reporte de pruebas (bitácora).

Arma ``docs/reporte_pruebas.md`` combinando:
  · lo que se **mide** (métricas de Val, conteos de evidencia, resultado de la
    validación INT8) — calculado en runtime desde los artefactos en disco;
  · lo que se **declara** (decisiones de diseño, limitaciones conocidas,
    próximos pasos) — leído de ``docs/decisiones.yaml``.

════════════════════════════════════════════════════════════════════════════
POR QUÉ SE REESCRIBIÓ
════════════════════════════════════════════════════════════════════════════
El script anterior se presentaba como *"documento vivo: re-generar tras cada
tanda de pruebas"*, pero las secciones 5 (Decisiones de diseño) y 6
(Limitaciones) eran **listas de strings hardcodeadas adentro del propio
script**. O sea que el informe se regeneraba y esas dos secciones decían
siempre exactamente lo mismo, aunque la lógica hubiera cambiado.

Peor: los números que aparecían en esos strings no salían de ninguna medición.
``"Cuantización INT8 descartada para vuelo: coincidencia 87.6% < umbral 95%"``
era texto literal, y el 87.6% provenía de una validación que medía otra cosa
(ver ``validate_int8_mission.py``). Lo mismo con
``"Veredictos: USI>3 ALTO ESTRÉS, USI>1 MODERADO"``: describía las reglas de
``analyze_stress.py`` mientras ``validate_int8_mission.py`` usaba USI>30.

Ahora lo declarado vive en ``docs/decisiones.yaml`` (editable sin tocar código,
con ``id`` y ``estado`` por ítem) y lo medido se calcula. Si falta un artefacto,
el informe dice **que falta** en vez de inventar un número.

Uso:
    python generate_report.py
    python generate_report.py --check      # no escribe; avisa si el .md está viejo
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cansat import indices as IDX
from cansat import paths as PROJ

OUT_MD = Path("docs/reporte_pruebas.md")
DECISIONES_YAML = Path("docs/decisiones.yaml")


# ══════════════════════════════════════════════════════════════════════ #
#  Carga de lo declarado
# ══════════════════════════════════════════════════════════════════════ #
def load_declarado() -> dict:
    """Lee ``docs/decisiones.yaml``. Si no está, devuelve un esqueleto honesto."""
    if not DECISIONES_YAML.is_file():
        return {"modelo_vuelo": {}, "decisiones": [], "limitaciones": [],
                "proximos_pasos": [f"FALTA {DECISIONES_YAML}"]}
    try:
        import yaml
    except ImportError:
        # Sin pyyaml no se puede parsear; mejor decirlo que inventar.
        return {"modelo_vuelo": {}, "decisiones": [], "limitaciones": [],
                "proximos_pasos": [
                    "No se pudo leer docs/decisiones.yaml porque falta el "
                    "paquete 'pyyaml' (pip install pyyaml). Las secciones 5, 6 "
                    "y 7 de este informe quedaron vacías: NO están hardcodeadas "
                    "como antes."]}
    with DECISIONES_YAML.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _items(seccion) -> list[str]:
    """Acepta tanto listas de strings como listas de {id, texto, estado}."""
    out: list[str] = []
    for it in seccion or []:
        if isinstance(it, str):
            out.append(it.strip())
        elif isinstance(it, dict):
            txt = str(it.get("texto", "")).strip()
            estado = str(it.get("estado", "")).strip().lower()
            if estado and estado != "vigente":
                txt = f"**[{estado.upper()}]** {txt}"
            if txt:
                out.append(txt)
    return out


# ══════════════════════════════════════════════════════════════════════ #
#  Lo medido
# ══════════════════════════════════════════════════════════════════════ #
def section_modelo(L: list[str], dec: dict) -> None:
    mv = dec.get("modelo_vuelo") or {}
    L.append("\n## 1. Modelo de segmentación")
    if not mv:
        L.append("- ⚠ Sin datos en `docs/decisiones.yaml` → `modelo_vuelo`.")
        return
    if mv.get("arquitectura"):
        L.append(f"- {mv['arquitectura']}, {mv.get('clases', '?')} clases.")
    if mv.get("nombres_clases"):
        L.append(f"- Clases: {', '.join(mv['nombres_clases'])}.")
    if mv.get("entrada"):
        L.append(f"- Entrada: `{mv['entrada']}`.")
    if mv.get("onnx"):
        p = Path(str(mv["onnx"]))
        if p.is_file():
            L.append(f"- ONNX: `{p}` ({p.stat().st_size / 1e6:.1f} MB).")
        else:
            L.append(f"- ONNX declarado: `{p}` — ⚠ **NO EXISTE en el repo**.")
    if mv.get("precision_vuelo"):
        L.append(f"- Precisión de vuelo: **{mv['precision_vuelo']}**.")
    if mv.get("nota_precision"):
        L.append(f"\n> {str(mv['nota_precision']).strip()}")


def section_val(L: list[str], dec: dict | None = None) -> None:
    """Métricas de Val, leídas del JSON que escribe ``evaluate.py``."""
    L.append("\n## 2. Métricas cuantitativas (Val, modelo ONNX)")

    latest = _latest_json(PROJ.METRICS_DIR, "val_",
                          preferir=(dec.get("modelo_vuelo") or {}).get("onnx"))
    if latest is None:
        # Fallback: la matriz de confusión cruda que guardaba evaluate_val.py.
        conf_p = PROJ.OUTPUTS / "val_confusion_onnx.npy"
        if not conf_p.is_file():
            L.append("- ⚠ **Aún no generadas.** Correr:")
            L.append("  ```bash")
            L.append("  python evaluate.py            # Val completo, escribe outputs/metrics/")
            L.append("  ```")
            L.append("- El informe anterior citaba 46.36% / 52.44% / 0.5317 según")
            L.append("  el archivo: **cuatro mIoU distintos para el mismo modelo**,")
            L.append("  tres de ellos hardcodeados en un `print()`. Ninguno se")
            L.append("  reproduce acá hasta que exista una medición real.")
            return
        conf = np.load(conf_p)
        from cansat.metrics import from_confusion
        m = from_confusion(conf)
        L.append(f"- Fuente: `{conf_p}` (matriz de confusión cruda, sin metadata")
        L.append("  de checkpoint ni de tamaño de muestra — no se sabe de qué corrida es).")
        L.append("")
        L.append("| Clase | IoU | Precisión | Recall |")
        L.append("|---|---|---|---|")
        for i, name in enumerate(IDX.CLASS_NAMES):
            L.append(f"| {name} | {m.iou[i] * 100:.1f}% | "
                     f"{m.precision[i] * 100:.1f}% | {m.recall[i] * 100:.1f}% |")
        L.append(f"\nmIoU: {m.miou * 100:.2f}% | Pixel accuracy: "
                 f"{m.pixel_acc * 100:.2f}% | {m.n_pixels:,} píxeles")
        return

    data = json.loads(latest.read_text(encoding="utf-8"))
    met = data.get("metrics") or data
    L.append(f"- Fuente: `{latest}` — generada {data.get('generado', '?')}")
    L.append(f"- Modelo: `{data.get('modelo', data.get('onnx', '?'))}`")
    L.append(f"- Muestra: {data.get('n_images', '?')} imágenes de Val "
             f"(semilla {data.get('seed', '?')}).")
    if data.get("nota"):
        L.append(f"- {data['nota']}")
    L.append("")
    L.append("| Clase | IoU | Precisión | Recall | Soporte (px) |")
    L.append("|---|---|---|---|---|")
    for name, v in (met.get("por_clase") or {}).items():
        L.append(f"| {name} | {v['iou'] * 100:.1f}% | {v['precision'] * 100:.1f}% "
                 f"| {v['recall'] * 100:.1f}% | {v.get('soporte_px', '?'):,} |")
    L.append(f"\n**mIoU: {met.get('miou', 0) * 100:.2f}%** | Pixel accuracy: "
             f"{met.get('pixel_acc', 0) * 100:.2f}% | {met.get('n_pixels', 0):,} píxeles")


def section_int8(L: list[str]) -> None:
    """Resultado de la validación INT8, leído del JSON de validate_int8_mission."""
    L.append("\n## 3. Validación de cuantización INT8")
    latest = _latest_json(PROJ.METRICS_DIR, "int8_")
    if latest is None:
        L.append("- ⚠ **Aún no generada.** Correr:")
        L.append("  ```bash")
        L.append("  python validate_int8_mission.py")
        L.append("  ```")
        L.append("- El 87.6% que figuraba en el informe anterior **no es válido**:")
        L.append("  se midió sobre las evidencias anotadas (overlay + cajas +")
        L.append("  texto quemado), con fórmulas de USI/NDVI distintas a las del")
        L.append("  vuelo, y comparando modelos distintos. Ver el docstring de")
        L.append("  `validate_int8_mission.py`.")
        return
    d = json.loads(latest.read_text(encoding="utf-8"))
    icon = {"APTO": "✅", "MARGINAL": "⚠️", "NO APTO": "❌"}.get(
        d.get("veredicto_validacion", ""), "•")
    L.append(f"- Fuente: `{latest}` — generada {d.get('generado', '?')}")
    L.append(f"- FP32 `{d.get('fp32')}` vs INT8 `{d.get('int8')}` "
             f"sobre **{d.get('n_frames')} frames crudos**.")
    L.append("")
    L.append("| Métrica | Valor | Umbral |")
    L.append("|---|---|---|")
    umb = d.get("umbrales", {})
    L.append(f"| Acuerdo píxel a píxel | {d.get('acuerdo_pixel', 0) * 100:.2f}% "
             f"| {umb.get('pixel', 0) * 100:.0f}% |")
    L.append(f"| Acuerdo de veredicto | {d.get('acuerdo_veredicto', 0) * 100:.2f}% "
             f"| {umb.get('veredicto', 0) * 100:.0f}% |")
    L.append(f"| Acuerdo de alerta | {d.get('acuerdo_alerta', 0) * 100:.2f}% "
             f"| {umb.get('alerta', 0) * 100:.0f}% |")
    L.append(f"| MAE de % de terreno | {d.get('mae_pct_terreno', 0):.2f} pts | — |")
    L.append(f"\n{icon} **{d.get('veredicto_validacion', '?')}**")


def section_simulacro(L: list[str]) -> None:
    L.append("\n## 4. Simulacro de descenso (prueba de integridad N.º 6 del DPD)")

    cands = sorted(PROJ.MISSION_DIR.glob("telemetry*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        L.append("- ⚠ Simulacro **aún no corrido**:")
        L.append("  ```bash")
        L.append("  python mission_pipeline.py --folder <tiles> --frames 28 \\")
        L.append("      --interval 3 --shuffle --sampler")
        L.append("  ```")
        return

    src = cands[0]
    with src.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        L.append(f"- ⚠ `{src}` existe pero no tiene filas.")
        return

    alts = [float(r["alt_m"]) for r in rows if r.get("alt_m")]
    ver = Counter(r["verdict"] for r in rows if r.get("verdict"))
    diag = Counter(r["diag"] for r in rows if r.get("diag"))
    n_alert = sum(1 for r in rows if str(r.get("alert")) == "1")
    pri = Counter(r.get("sample_pri") for r in rows if r.get("sample_pri"))

    L.append(f"- Fuente: `{src}` ({len(rows)} paquetes).")
    if alts:
        L.append(f"- Altitud: {max(alts):.0f} m → {min(alts):.0f} m.")
    if ver:
        L.append("- Veredictos: " + ", ".join(f"{k}: {v}" for k, v in ver.most_common()) + ".")
    if diag:
        L.append("- Diagnósticos: " + ", ".join(f"{k}: {v}" for k, v in diag.most_common()) + ".")
    L.append(f"- Alertas: {n_alert} de {len(rows)} frames.")
    if pri:
        L.append("- Sampler: " + ", ".join(f"{k}: {v}" for k, v in pri.most_common()) + ".")

    # Nitidez: el gate que antes era sólo log.
    sharp = [float(r["sharp"]) for r in rows if r.get("sharp")]
    if sharp:
        bajos = sum(1 for s in sharp if s < 50.0)
        L.append(f"- Nitidez (Laplaciano): min {min(sharp):.0f} · "
                 f"mediana {sorted(sharp)[len(sharp) // 2]:.0f} · max {max(sharp):.0f}"
                 + (f" · {bajos} frame(s) bajo 50" if bajos else "") + ".")


def section_evidencias(L: list[str]) -> None:
    L.append("\n## 5. Evidencias generadas")
    conteos = {
        "Tiles Test segmentados": _count(PROJ.OUTPUTS / "test_inference", "*_segmented.jpg"),
        "Frames de evidencia de misión": _count(PROJ.MISSION_DIR / "vis", "*_evid.jpg"),
        "Detecciones YOLO anotadas": _count(PROJ.OUTPUTS / "people_detection", "*_det.jpg"),
        "Frames alta prioridad (sampler)": _count(PROJ.MISSION_DIR / "high_res", "*"),
        "Frames prioridad media": _count(PROJ.MISSION_DIR / "full_res", "*"),
        "Thumbnails": _count(PROJ.MISSION_DIR / "thumb", "*"),
        "Overlays ensemble (post-vuelo)": _count(PROJ.ENTREGA / "ens_seg", "*_b5.png"),
        "Frames mejorados con EDSR": _count(PROJ.ENTREGA / "enhanced", "*_edsr.jpg"),
        "Simulaciones de desastre": _count(PROJ.OUTPUTS / "disaster_sim", "*"),
    }
    for k, v in conteos.items():
        L.append(f"- {k}: {v}")


def _count(folder: Path, pattern: str) -> int:
    return len(list(folder.rglob(pattern))) if folder.is_dir() else 0


def _latest_json(folder: Path, prefix: str, preferir: str | None = None) -> Path | None:
    """
    El JSON de métricas más reciente… con preferencia por el modelo de vuelo.

    ⚠ Antes tomaba SIEMPRE el más nuevo, así que alcanzaba con evaluar un modelo
      de prueba después para que el informe citara SUS métricas como si fueran
      las del vuelo (pasó: el informe mostró el 45.76 % del best_model porque
      fue la última evaluación). Si ``preferir`` es un nombre de archivo, se
      busca un JSON cuyo campo ``modelo`` termine con ese nombre.
    """
    if not folder.is_dir():
        return None
    cands = sorted(folder.glob(f"{prefix}*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    if preferir:
        nombre = Path(preferir).name
        for p in cands:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if str(data.get("modelo", "")).replace("\\", "/").endswith(nombre):
                return p
    return cands[0]


def section_licencias(L: list[str]) -> None:
    L.append("\n## 6. Datasets y licencias")
    L.append("")
    L.append("| Dataset | Uso en el proyecto | Licencia | Restricción |")
    L.append("|---|---|---|---|")
    L.append("| LoveDA (Zenodo 5706578) | Segmentación de terreno (5 clases) | "
             "CC BY 4.0 | Atribución |")
    L.append("| xBD / xView2 | Modelos de daño estructural | "
             "**CC BY-NC-SA 4.0** | **No comercial + ShareAlike** |")
    L.append("| FloodNet (Kaggle) | Especialista de inundación | "
             "sin declarar en la ficha | Verificar antes de distribuir |")
    L.append("| VisDrone | Fine-tune del detector | "
             "ver términos del challenge | Verificar |")
    L.append("| Esri World Imagery | `baseline.png` del siamés | "
             "términos de ArcGIS Online | Verificar para un entregable |")
    L.append("")
    L.append("⚠ **ShareAlike de xBD:** los pesos entrenados con xBD")
    L.append("  (`best_damage*.pth`, `best_siamese_damage.pth` y sus ONNX) son")
    L.append("  obra derivada y heredan la licencia no comercial + compartir-igual.")
    L.append("  Atribuir explícitamente en el DPD y consultar a CONAE antes de")
    L.append("  distribuirlos. Los modelos entrenados **sólo con LoveDA** no tienen")
    L.append("  esa restricción: conviene tener claro cuál es cuál (ver `MODELS.yaml`).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generador del reporte de pruebas")
    ap.add_argument("--out", default=str(OUT_MD))
    ap.add_argument("--check", action="store_true",
                    help="no escribe; exit 1 si el .md no existe")
    args = ap.parse_args(argv)

    dec = load_declarado()

    L: list[str] = [
        "# CanSat La Base (CONAE 135) — Reporte de pruebas de misión secundaria",
        f"\nGenerado: {datetime.date.today().isoformat()} por `generate_report.py`.",
        "",
        "> Las secciones 1–5 salen de **artefactos medidos** en disco. Las 7–9",
        "> salen de `docs/decisiones.yaml`, que es editable sin tocar código.",
        "> Si falta un artefacto, el informe **dice que falta** en vez de citar",
        "> un número hardcodeado (que era lo que pasaba antes).",
    ]

    section_modelo(L, dec)
    section_val(L, dec)
    section_int8(L)
    section_simulacro(L)
    section_evidencias(L)
    section_licencias(L)

    L.append("\n## 7. Decisiones de diseño")
    items = _items(dec.get("decisiones"))
    if items:
        for t in items:
            L.append(f"- {_oneline(t)}")
    else:
        L.append("- ⚠ Vacío: falta `docs/decisiones.yaml` o el paquete `pyyaml`.")

    L.append("\n## 8. Limitaciones conocidas del modelo")
    items = _items(dec.get("limitaciones"))
    if items:
        for t in items:
            L.append(f"- {_oneline(t)}")
    else:
        L.append("- ⚠ Vacío: falta `docs/decisiones.yaml` o el paquete `pyyaml`.")

    L.append("\n## 9. Próximos pasos")
    for t in _items(dec.get("proximos_pasos")) or ["- ⚠ Vacío."]:
        L.append(f"- {_oneline(t)}")

    text = "\n".join(L).replace("\n\n\n", "\n\n") + "\n"

    if args.check:
        out = Path(args.out)
        if not out.is_file():
            print(f"[CHECK] {out} no existe.")
            return 1
        actual = out.read_text(encoding="utf-8")
        # Comparar ignorando la fecha de generación.

        def strip(s: str) -> str:
            return "\n".join(linea for linea in s.splitlines()
                             if not linea.startswith("Generado:"))

        if strip(actual) != strip(text):
            print(f"[CHECK] {out} está DESACTUALIZADO. Correr: python generate_report.py")
            return 1
        print(f"[CHECK] {out} al día.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[OK] Reporte generado: {out.resolve()}")
    return 0


def _oneline(t: str) -> str:
    """Colapsa un bloque multilínea YAML en una viñeta legible."""
    return " ".join(x.strip() for x in t.splitlines() if x.strip())


if __name__ == "__main__":
    raise SystemExit(main())
