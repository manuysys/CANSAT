"""
Conversión de NUESTRO tiny de terreno a `.rpk` (IMX500) — cadena verificada.

Corre DENTRO de un contenedor Linux con `edge-mdt[pt]` (Sony, PyPI público) +
`torch` CPU (2.7.x: con 2.14 el export de MCT falla por el exporter dynamo) +
`onnxscript` + un JRE (el compilador DSP de Sony usa java). Cadena completa:

    # 1) contenedor (una vez)
    docker run -d --name cansat-mdt -v "<repo>:/work" -w /work python:3.11-slim \
        sleep infinity
    docker exec cansat-mdt bash -c "pip install 'edge-mdt[pt]' \
        'torch==2.7.1' 'torchvision==0.22.1' onnxscript \
        --index-url https://download.pytorch.org/whl/cpu"   # torch aparte
    docker exec cansat-mdt bash -c "apt-get update && \
        apt-get install -y default-jre-headless"

    # 2) PTQ + export (este script)
    docker exec cansat-mdt python tools/convertir_rpk.py \
        --checkpoint outputs/best_terrain_tiny.pth \
        --calib-dir dataset/loveda_remapped/Train \
        --out outputs/tiny_mct.onnx

    # 3) compilar (Sony)
    docker exec cansat-mdt imxconv-pt -i outputs/tiny_mct.onnx \
        -o outputs/rpk_out --no-input-persistency --overwrite-output

    # 4) empaquetar (EN LA PI: apt imx500-tools)
    imx500-package -i packerOut.zip -o /home/pi/modelos/   # → network.rpk

Medido 2026-10-05: memoria 4.28/8 MB (54 %, entra), KPI del NPU 4.5 ms,
end-to-end ~0.4-1.2 fps (limitado por la transferencia de la salida de
5×224×224; ver docs/benchmarks/rpk_propio_tiny.json).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

import model_compression_toolkit as mct
from edgemdt_tpc import get_target_platform_capabilities

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 224


class LRASPPMobileNetV3Small(nn.Module):
    """Copia autónoma de train_terrain_tiny.LRASPPMobileNetV3Small."""

    def __init__(self, num_classes: int = 5, mid_ch: int = 128):
        super().__init__()
        from torchvision.models import mobilenet_v3_small

        feats = mobilenet_v3_small(weights=None).features
        blocks = list(feats.children())
        self.low = nn.Sequential(*blocks[:4])
        self.high = nn.Sequential(*blocks[4:13])
        self.cbr = nn.Sequential(
            nn.Conv2d(576, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, mid_ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(24, num_classes, 1)
        self.high_classifier = nn.Conv2d(mid_ch, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        high = self.high(low)
        h = self.cbr(high)
        h = h * self.scale(high)
        h = self.high_classifier(h)
        h = F.interpolate(h, size=low.shape[-2:], mode="bilinear",
                          align_corners=False)
        out = self.low_classifier(low) + h
        return F.interpolate(out, size=size, mode="bilinear",
                             align_corners=False)


def _calib_files(root: Path, n: int) -> list[Path]:
    files = sorted(root.glob("*/images_png/*.png"))
    if not files:
        files = sorted(root.glob("**/*.png"))
    if not files:
        raise RuntimeError(f"sin imágenes de calibración en {root}")
    print(f"calibración: {len(files)} disponibles, uso {min(n, len(files))}")
    return files[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Tiny de terreno → ONNX INT8 (MCT)")
    ap.add_argument("--checkpoint", default="outputs/best_terrain_tiny.pth")
    ap.add_argument("--calib-dir", default="dataset/loveda_remapped/Train")
    ap.add_argument("--calib-imgs", type=int, default=128)
    ap.add_argument("--tpc-version", default="5.0")
    ap.add_argument("--out", default="outputs/tiny_mct.onnx")
    args = ap.parse_args()

    files = _calib_files(Path(args.calib_dir), args.calib_imgs)

    def gen():
        for p in files:
            img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
            rgb = np.asarray(img, np.float32) / 255.0
            x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
            yield [torch.from_numpy(x).float()]

    model = LRASPPMobileNetV3Small(5)
    state = torch.load(args.checkpoint, map_location="cpu",
                       weights_only=False)
    sd = (state.get("model_state_dict", state)
          if isinstance(state, dict) else state)
    model.load_state_dict(sd, strict=False)
    model.eval()
    with torch.no_grad():
        y = model(torch.zeros(1, 3, SIZE, SIZE))
    print("modelo cargado, salida:", tuple(y.shape))

    tpc = get_target_platform_capabilities(args.tpc_version)
    print(f"TPC {args.tpc_version} cargado; PTQ...", flush=True)
    quantized, _info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=gen,
        target_platform_capabilities=tpc,
    )
    mct.exporter.pytorch_export_model(
        model=quantized,
        save_model_path=args.out,
        repr_dataset=gen,
        serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX,
    )
    print(f"[OK] {args.out} — ahora: imxconv-pt -i {args.out} -o outputs/rpk_out "
          f"--no-input-persistency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
