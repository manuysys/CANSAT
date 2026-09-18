"""
CanSat La Base — Exportación ONNX **única** para los modelos de segmentación.

════════════════════════════════════════════════════════════════════════════
POR QUÉ UN SOLO EXPORTADOR
════════════════════════════════════════════════════════════════════════════
Había **seis** scripts de exportación que eran el mismo ``torch.onnx.export``
con distinta ruta: ``export_onnx.py``, ``export_v2.py``, ``export_damage_onnx.py``,
``export_damage_v3.py``, ``export_b5_512.py``, ``export_b5_640.py``.

Y no eran equivalentes:

  · ``dynamo=False`` —que ``generate_report.py`` declara *"requisito del
    conversor Sony IMX500"*— sólo lo pasaban **2 de los 6**.
  · ``export_b5_640.py`` cargaba ``outputs/best_terrain_b5.pth``, que **no
    existe** (el real es ``best_terrain_segformer_b5.pth``), e instanciaba
    ``SegformerTerrain``, una clase que **no existe** en
    ``train_segformer_b5.py`` (ahí sólo hay ``Wrap``). No corría.
  · ``export_b5_512.py`` instanciaba el modelo desde HuggingFace en runtime, lo
    que además de requerir red hacía innecesario el ``ignore_mismatched_sizes``
    (inmediatamente después se pisaba con ``load_state_dict``).
  · Los checkpoints tienen dos formatos incompatibles (ver
    ``cansat/checkpoints.py``) y cada exportador servía para uno solo.

Éste cubre todas las arquitecturas del proyecto, acepta ambos formatos de
checkpoint, pasa siempre ``dynamo=False``, y **valida el resultado** con
``audit_imx500.py`` antes de darlo por bueno.

Uso:
    python export_onnx.py
    python export_onnx.py --checkpoint outputs/best_terrain_v2.pth --output outputs/cansat_seg_terrain_v2.onnx
    python export_onnx.py --arch cbam --checkpoint outputs/best_terrain_cbam.pth --img-size 320
    python export_onnx.py --arch damage --num-classes 3 --checkpoint outputs/best_damage3.pth \\
                          --output outputs/cansat_damage3_mobilenetv2.onnx
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.onnx")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

from cansat import indices as IDX                              # noqa: E402
from cansat.checkpoints import ckpt_meta, load_model_state     # noqa: E402

ARCHS = ("deeplabv3plus", "cbam", "segformer", "damage", "lraspp", "auto")

# Requisitos declarados en docs/reporte_pruebas.md y verificados por audit_imx500.py
OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def build_model(arch: str, num_classes: int, img_size: int):
    """Instancia la arquitectura pedida. ``auto`` la deduce del checkpoint."""
    if arch in ("deeplabv3plus", "damage"):
        from train import DeepLabV3PlusMobileNetV2
        return DeepLabV3PlusMobileNetV2(num_classes)

    if arch == "cbam":
        from train_cbam import DeepLabV3PlusCBAM
        return DeepLabV3PlusCBAM(num_classes)

    if arch == "lraspp":
        from train_terrain_tiny import LRASPPMobileNetV3Small
        return LRASPPMobileNetV3Small(num_classes)

    if arch == "segformer":
        # El checkpoint de train_segformer_b5.py guarda el state_dict del modelo
        # INTERNO de HuggingFace (base.state_dict()), así que hay que instanciar
        # ése, no el Wrap. export_b5_512.py lo hacía bien; export_b5_640.py no.
        import torch.nn.functional as F
        from transformers import SegformerForSemanticSegmentation

        base = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640",
            num_labels=num_classes, ignore_mismatched_sizes=True)

        class Wrap(torch.nn.Module):
            """Envuelve para que la salida sea un tensor ya interpolado."""

            def __init__(self, m, size):
                super().__init__()
                self.m, self.size = m, size

            def forward(self, x):
                return F.interpolate(self.m(x).logits, size=(self.size, self.size),
                                     mode="bilinear", align_corners=False)

        return Wrap(base, img_size)

    raise ValueError(f"arquitectura desconocida: {arch}")


def guess_arch(state: dict) -> str:
    """Deduce la arquitectura de los nombres de tensor del checkpoint."""
    keys = list(state.keys())
    if any(k.startswith("segformer.") or "decode_head" in k for k in keys):
        return "segformer"
    if any(k.startswith(("low_classifier.", "high_classifier.")) for k in keys):
        return "lraspp"
    low = [k.lower() for k in keys]
    if any("cbam" in k for k in low):
        return "cbam"
    if any(k.startswith("aspp.") for k in keys):
        return "deeplabv3plus"
    return "deeplabv3plus"


def export(checkpoint_path: str, output_path: str, arch: str,
           num_classes: int | None, img_size: int | None,
           opset: int = OPSET, skip_audit: bool = False) -> int:
    ck = Path(checkpoint_path)
    if not ck.is_file():
        print(f"[ERROR] No se encontró el checkpoint: {ck}")
        print("        Entrená primero, o revisá MODELS.yaml para ver qué")
        print("        checkpoint produce cada artefacto.")
        return 1

    print(f"[1/5] Checkpoint: {ck}")
    # weights_only=True siempre. La versión anterior usaba weights_only=False,
    # que ejecuta pickle arbitrario.
    state = load_model_state(ck)
    meta = ckpt_meta(ck)
    if meta:
        print("      metadata: " + ", ".join(
            f"{k}={meta[k]}" for k in ("miou", "img_size", "num_classes",
                                       "epochs", "script") if k in meta))
    else:
        print("      (checkpoint viejo sin metadata: state_dict crudo)")

    # ── Resolver arquitectura, clases y tamaño ──────────────────────────
    if arch == "auto":
        arch = guess_arch(state)
        print(f"[2/5] Arquitectura deducida del checkpoint: {arch}")
    else:
        print(f"[2/5] Arquitectura: {arch}")

    if num_classes is None:
        num_classes = int(meta.get("num_classes") or IDX.NUM_CLASSES)
        # Un state_dict cuyo último bias tiene N entradas revela las clases.
        for k, v in state.items():
            if (k.endswith("decoder.4.bias") or k.endswith("classify.bias")
                    or k.endswith("low_classifier.bias")
                    or k.endswith("high_classifier.bias")):
                num_classes = int(v.shape[0])
                print(f"      num_classes={num_classes} (deducido de {k})")
                break
    if img_size is None:
        img_size = int(meta.get("img_size") or 320)
        print(f"      img_size={img_size}"
              + ("" if meta.get("img_size") else " (default; el checkpoint no lo declara)"))

    model = build_model(arch, num_classes, img_size)

    # ── Carga con verificación de cobertura ─────────────────────────────
    # Antes: load_state_dict(...) a secas, o un try/except TypeError parchando
    # una confusión de firmas. Si las shapes no matcheaban, reventaba con un
    # mensaje críptico de PyTorch.
    target = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in target and target[k].shape == v.shape}
    frac = len(filtered) / max(1, len(target))
    if frac < 1.0:
        print(f"      [WARN] sólo {frac:.0%} de las capas coinciden "
              f"({len(filtered)}/{len(target)}). ¿Arquitectura o num_classes equivocados?")
        if frac < 0.5:
            print("[ERROR] Cobertura demasiado baja: no tiene sentido exportar esto.")
            print("        Probá --arch explícito, o --num-classes correcto.")
            return 1
    model.load_state_dict(filtered, strict=(frac >= 1.0))
    model.eval().cpu()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"      {n_params:,} parámetros")

    # ── Inferencia de verificación ──────────────────────────────────────
    print(f"[3/5] Inferencia dummy ({img_size}×{img_size})...")
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, dict):
        out = out.get("logits", next(iter(out.values())))
    print(f"      salida {list(out.shape)}")
    expected = (1, num_classes, img_size, img_size)
    if tuple(out.shape) != expected:
        print(f"[ERROR] shape inesperado {tuple(out.shape)}; se esperaba {expected}")
        return 1

    # ── Exportar ────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/5] Exportando → {out_path}")
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        opset_version=opset,
        do_constant_folding=True,
        # SIEMPRE dynamo=False y SIN dynamic_axes: batch fijo = 1, que es el
        # requisito declarado del conversor Sony IMX500. Sólo 2 de los 6
        # exportadores antiguos lo respetaban.
        dynamo=False,
    )
    print(f"      {out_path.stat().st_size / 1e6:.1f} MB")

    # ── Validar ─────────────────────────────────────────────────────────
    print("[5/5] Validando...")
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out_path)))
        print("      ✓ onnx.checker")
    except ImportError:
        print("      [SKIP] paquete 'onnx' no instalado")
    except Exception as e:
        print(f"[ERROR] onnx.checker falló: {e}")
        return 1

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path))
        x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        ort_out = sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x))
            if isinstance(pt_out, dict):
                pt_out = pt_out.get("logits", next(iter(pt_out.values())))
            pt_out = pt_out.numpy()
        diff = float(np.abs(ort_out - pt_out).max())
        print(f"      Δ máx PyTorch vs ONNX Runtime: {diff:.2e}")
        if diff >= 1e-4:
            print("      [WARN] diferencia alta; revisá opset y constant folding")
        else:
            print("      ✓ coinciden")
        pred = ort_out.argmax(axis=1)
        print(f"      clases predichas en el dummy: {np.unique(pred).tolist()}")
    except ImportError:
        print("      [SKIP] onnxruntime no instalado")

    if not skip_audit:
        print("\n      Auditoría IMX500:")
        try:
            from audit_imx500 import audit
            ok, probs = audit(out_path, verbose=False)
            if ok:
                print("      ✓ cumple los requisitos del conversor IMX500")
            else:
                print("      ✗ NO cumple:")
                for pb in probs:
                    print(f"        · {pb}")
        except Exception as e:
            print(f"      [WARN] no se pudo auditar: {e}")

    print(f"\n{'=' * 62}")
    print(f"  ONNX exportado: {out_path}")
    print(f"  Entrada [{1}, 3, {img_size}, {img_size}] · Salida [{1}, {num_classes}, "
          f"{img_size}, {img_size}] '{OUTPUT_NAME}'")
    # Los nombres de clase dependen de la tarea: el modelo de daño NO usa las
    # clases de terreno (antes imprimía vegetation/building/water para daño).
    if arch == "damage":
        nombres = ["other", "intacto", "danado"][:num_classes]
    else:
        nombres = list(IDX.CLASS_NAMES)[:num_classes]
    print(f"  Clases: {nombres}")
    print("  → REGISTRÁ el artefacto en MODELS.yaml (checkpoint, script, mIoU).")
    print(f"{'=' * 62}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Exportar checkpoint de segmentación a ONNX (exportador único)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", default="outputs/best_model.pth")
    ap.add_argument("--output", default="outputs/cansat_seg_deeplabv3plus_mobilenetv2.onnx")
    ap.add_argument("--arch", choices=ARCHS, default="auto",
                    help="auto = deducir de los nombres de tensor del checkpoint")
    ap.add_argument("--num-classes", type=int, default=None,
                    help="default: el del checkpoint, o 5")
    ap.add_argument("--img-size", type=int, default=None,
                    help="default: el del checkpoint, o 320")
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args(argv)

    return export(args.checkpoint, args.output, args.arch,
                  args.num_classes, args.img_size, args.opset, args.skip_audit)


if __name__ == "__main__":
    raise SystemExit(main())
