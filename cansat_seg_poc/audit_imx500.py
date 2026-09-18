"""
Audita que un ONNX cumpla los requisitos del conversor Sony IMX500.

════════════════════════════════════════════════════════════════════════════
POR QUÉ SE REESCRIBIÓ
════════════════════════════════════════════════════════════════════════════
La versión anterior se llamaba "Revisa que el ONNX cumpla los requisitos
típicos del conversor IMX500" y **sólo imprimía** opset, dimensiones y recuento
de operadores. No comparaba contra ningún requisito, no devolvía código de
salida, y tenía la ruta del modelo hardcodeada. Era un ``print``, no una
auditoría.

Ésta sí valida, contra los requisitos que el propio ``docs/reporte_pruebas.md``
declara como decisiones de diseño:

    "Exportación ONNX con dynamo=False y sin dynamic_axes (requisito del
     conversor Sony IMX500). Opset 17, entrada fija [1,3,320,320],
     salida 'logits'."

Devuelve exit code != 0 si algo no cumple, así se puede poner en CI o en un
pre-commit y falla de verdad.

⚠ Los requisitos concretos del conversor están en la documentación de Sony
  (``imx500-converter`` / Raspberry Pi AI HAT+). Los de acá son los que el
  proyecto declaró; verificar contra la versión instalada antes del vuelo con
  ``pi/convertir_modelos.sh --help``.

Uso:
    python audit_imx500.py
    python audit_imx500.py --onnx outputs/cansat_seg_terrain_v2.onnx
    python audit_imx500.py --all          # audita todos los ONNX de outputs/
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cansat import paths as PROJ

# ── Requisitos declarados en docs/reporte_pruebas.md ──────────────────── #
REQ_OPSET = 17
REQ_BATCH = 1
REQ_INPUT_NAME = "input"
REQ_OUTPUT_NAME = "logits"
REQ_CHANNELS = 3
# Tamaño de entrada: el conversor exige cuadrado estático, no un valor puntual.
# El modelo de vuelo pasó de 320 (v2) a 224 (F1 tiny): con --size N se exige N,
# con 0 (default) se acepta cualquier cuadrado estático y se reporta cuál es.
REQ_SIZE = 0
# Operadores que el conversor IMX500 no suele soportar (lista conservadora;
# ampliar según la salida real de imx500_converter).
OPS_RIESGOSAS = {
    "NonMaxSuppression", "Multinomial", "RandomUniform", "RandomNormal",
    "Loop", "If", "Scan", "RoiAlign", "GridSample", "ScatterND",
    "Einsum", "ReverseSequence", "NonZero", "DynamicSlice",
}


def audit(path: Path, verbose: bool = True,
          req_size: int = REQ_SIZE) -> tuple[bool, list[str]]:
    """
    Audita un ONNX. Devuelve ``(ok, lista_de_problemas)``.

    Nunca lanza: un archivo corrupto se reporta como problema, no como traceback.
    """
    probs: list[str] = []
    try:
        import onnx
    except ImportError:
        return False, ["falta el paquete 'onnx' (pip install onnx)"]

    if not path.is_file():
        return False, [f"no existe el archivo: {path}"]

    try:
        m = onnx.load(str(path))
    except Exception as e:
        return False, [f"no se pudo cargar ({type(e).__name__}: {e})"]

    try:
        onnx.checker.check_model(m)
    except Exception as e:
        probs.append(f"onnx.checker falló: {e}")

    # ── opset ───────────────────────────────────────────────────────────
    opsets = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    ver = opsets.get("ai.onnx")
    if ver is None:
        probs.append(f"no declara el dominio ai.onnx (tiene {opsets})")
    elif ver != REQ_OPSET:
        probs.append(f"opset {ver} ≠ {REQ_OPSET} (requisito declarado en "
                     f"docs/reporte_pruebas.md)")

    # ── entrada ─────────────────────────────────────────────────────────
    if len(m.graph.input) != 1:
        probs.append(f"tiene {len(m.graph.input)} entradas; el conversor espera 1")
    else:
        inp = m.graph.input[0]
        if inp.name != REQ_INPUT_NAME:
            probs.append(f"entrada se llama '{inp.name}', se esperaba '{REQ_INPUT_NAME}'")
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in inp.type.tensor_type.shape.dim]
        if len(dims) != 4:
            probs.append(f"entrada de rango {len(dims)}, se esperaba 4 (NCHW)")
        else:
            n, c, h, w = dims
            if n != REQ_BATCH:
                probs.append(f"batch = {n!r}; debe ser fijo = {REQ_BATCH} "
                             f"(sin dynamic_axes)")
            if c != REQ_CHANNELS:
                probs.append(f"canales = {c!r}, se esperaba {REQ_CHANNELS}")
            if isinstance(h, int) and isinstance(w, int) and h != w:
                probs.append(f"entrada no cuadrada {h}×{w}: el conversor espera "
                             f"alto = ancho")
            if req_size and (h != req_size or w != req_size):
                probs.append(f"entrada {h!r}×{w!r}; se exigía "
                             f"{req_size}×{req_size} (--size)")
            if any(isinstance(x, str) for x in dims):
                probs.append(f"hay dimensiones simbólicas {dims}: el conversor "
                             f"necesita shapes estáticos")

    # ── salida ──────────────────────────────────────────────────────────
    outs = [o.name for o in m.graph.output]
    if REQ_OUTPUT_NAME not in outs:
        probs.append(f"salida {outs}; se esperaba '{REQ_OUTPUT_NAME}'")

    # ── operadores ──────────────────────────────────────────────────────
    ops = Counter(n.op_type for n in m.graph.node)
    malas = sorted(set(ops) & OPS_RIESGOSAS)
    if malas:
        probs.append(f"operadores probablemente no soportados por el conversor: "
                     f"{malas}")

    size_mb = os.path.getsize(path) / 1e6
    if size_mb > 200:
        probs.append(f"{size_mb:.0f} MB: demasiado grande para la NPU/SD de la Pi")

    # ── pesos externos ──────────────────────────────────────────────────
    # La fuente de verdad es el GRAFO: si ningún initializer declara
    # data_location=EXTERNAL, el modelo es autocontenido aunque haya un
    # .onnx.data suelto al lado (residuo de un export anterior o de otro
    # modelo: daba un falso positivo que bloqueaba la auditoría).
    sidecar = path.with_name(path.name + ".data")
    external_locs = any(
        t.data_location == onnx.TensorProto.EXTERNAL
        for t in m.graph.initializer)
    if external_locs:
        probs.append("usa pesos externos (.onnx.data): el conversor necesita el "
                     "modelo en un solo archivo; re-exportá sin external data")
    elif sidecar.is_file():
        print(f"  [i] {sidecar.name} presente pero NO referenciado por el grafo: "
              f"es un residuo, se puede borrar (no afecta la auditoría).")

    if verbose:
        print(f"\n{'─' * 70}\n  {path.name}   ({size_mb:.1f} MB)")
        print(f"  opset   : {opsets}")
        try:
            dims = [d.dim_value or d.dim_param
                    for d in m.graph.input[0].type.tensor_type.shape.dim]
            print(f"  entrada : {m.graph.input[0].name} {dims}")
        except Exception:
            pass
        print(f"  salida  : {outs}")
        print(f"  nodos   : {sum(ops.values())} · {len(ops)} op_types distintos")
        for op, n in ops.most_common(12):
            print(f"            {op:<22} ×{n}")
        if len(ops) > 12:
            print(f"            … y {len(ops) - 12} op_types más")
        if probs:
            print("  ✗ PROBLEMAS:")
            for p in probs:
                print(f"      · {p}")
        else:
            print("  ✓ cumple los requisitos declarados")

    return (not probs), probs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Audita ONNX contra los requisitos del conversor IMX500")
    ap.add_argument("--onnx", default="outputs/cansat_seg_deeplabv3plus_mobilenetv2.onnx",
                    help="modelo a auditar")
    ap.add_argument("--all", action="store_true",
                    help="auditar todos los .onnx de outputs/ (sin external data)")
    ap.add_argument("--size", type=int, default=REQ_SIZE,
                    help="exigir un tamaño de entrada puntual (0 = cualquier "
                         "cuadrado estático)")
    args = ap.parse_args(argv)

    if args.all:
        targets = sorted(p for p in (PROJ.OUTPUTS).glob("*.onnx"))
        if not targets:
            print(f"[ERROR] No hay .onnx en {PROJ.OUTPUTS}")
            return 1
    else:
        targets = [Path(args.onnx)]

    print("=" * 70)
    print("  AUDITORÍA IMX500")
    size_txt = (f"{args.size}×{args.size}" if args.size else "cuadrado estático")
    print(f"  Requisitos: opset {REQ_OPSET} · batch fijo {REQ_BATCH} · "
          f"entrada '{REQ_INPUT_NAME}' "
          f"[{REQ_BATCH},{REQ_CHANNELS},H,W] ({size_txt}) · "
          f"salida '{REQ_OUTPUT_NAME}'")
    print("=" * 70)

    total_ok = 0
    for t in targets:
        ok, _probs = audit(t, req_size=args.size)
        total_ok += int(ok)

    print(f"\n{'=' * 70}")
    print(f"  {total_ok}/{len(targets)} modelos cumplen")
    if total_ok < len(targets):
        print("  ✗ Hay modelos que NO pasan la auditoría. No convertirlos al")
        print("    formato IMX500 hasta arreglarlos (re-exportar con export_onnx.py).")
    print("=" * 70)
    return 0 if total_ok == len(targets) else 2


if __name__ == "__main__":
    raise SystemExit(main())
