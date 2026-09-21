"""
Grad-CAM para los modelos de segmentación — explicabilidad puntual.

Genera un mapa de calor de las regiones que más contribuyeron a la clase
predicha (en un punto o en la clase dominante del frame) y lo superpone a la
imagen. Versión mínima sin dependencias extra (hooks de torch), pensada para
auditar frames raros en post-vuelo; el PNG lo puede consumir la estación.

Limitación declarada: la capa objetivo se elige como la última convolución que
NO sea el clasificador final. Es un heurístico estable para las arquitecturas
DeepLabV3+ del repo, no un análisis de atribución exhaustivo.

Uso:
    python tools/gradcam.py --image dataset/pruebas/PERSONAS_AUTOS.png \
        --checkpoint outputs/best_damage3.pth --arch deeplabv3plus --num-classes 3 \
        --out outputs/gradcam_dano.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from torch import nn                                         # noqa: E402

from cansat import preprocess as PP                           # noqa: E402


def _capa_objetivo(model) -> nn.Module:
    """Última convolución que no sea el clasificador final (heurístico)."""
    candidatas = [(n, m) for n, m in model.named_modules()
                  if isinstance(m, nn.Conv2d) and not n.startswith("classifier")]
    if not candidatas:
        raise ValueError("no encontré ninguna Conv2d fuera del classifier")
    return candidatas[-1][1]


def gradcam(model, tensor: np.ndarray, capa: nn.Module | None = None,
            clase: int | None = None, punto: tuple[int, int] | None = None,
            img_size: int | None = None) -> np.ndarray:
    """
    Grad-CAM sobre ``tensor`` (1,C,H,W) numpy. Devuelve el mapa (H', W') en [0,1].

    ``clase``: clase objetivo; por default la predicha en ``punto`` (o la
    dominante del frame). El score es la media de los logits de esa clase
    restringida a los píxeles donde el modelo predice esa clase.
    """
    capa = capa or _capa_objetivo(model)
    activaciones: list[torch.Tensor] = []
    gradientes: list[torch.Tensor] = []

    h1 = capa.register_forward_hook(lambda _m, _i, out: activaciones.append(out))

    def _hook_grad(_m, _gin, gout):
        gradientes.append(gout[0].detach())

    h2 = capa.register_full_backward_hook(_hook_grad)

    try:
        x = torch.from_numpy(np.ascontiguousarray(tensor)).float()
        x.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        logits = model(x)                        # (1, C, h, w)
        if logits.shape[-1] != tensor.shape[-1] or logits.shape[-2] != tensor.shape[-2]:
            logits = nn.functional.interpolate(
                logits, size=(tensor.shape[-2], tensor.shape[-1]),
                mode="bilinear", align_corners=False)
        mapa = logits[0]
        if clase is None:
            if punto is not None:
                y, xp = punto
                clase = int(mapa[:, y, xp].argmax())
            else:
                clase = int(mapa.mean(dim=(1, 2)).argmax())
        mask = mapa.argmax(0) == clase
        if mask.sum() == 0:
            score = mapa[clase].mean()
        else:
            score = mapa[clase][mask].mean()
        model.zero_grad(set_to_none=True)
        score.backward()

        if not activaciones or not gradientes:
            raise RuntimeError("los hooks no capturaron activaciones/gradientes")
        act = activaciones[0][0]                 # (K, h, w)
        grad = gradientes[0][0]                  # (K, h, w)
        pesos = grad.mean(dim=(1, 2))            # (K,)
        cam = torch.relu((pesos[:, None, None] * act).sum(0))
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= max(float(cam.max()), 1e-9)
        return cam.astype(np.float32)
    finally:
        h1.remove()
        h2.remove()


def superponer(bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Heatmap JET redimensionado al frame, mezclado sobre la imagen."""
    h, w = bgr.shape[:2]
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1.0 - alpha, heat, alpha, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grad-CAM (explicabilidad)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", default="deeplabv3plus",
                    choices=("deeplabv3plus", "cbam", "lraspp"))
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--clase", type=int, default=None,
                    help="clase objetivo (default: dominante del frame)")
    ap.add_argument("--punto", default=None, metavar="Y,X",
                    help="píxel de interés: la clase objetivo es la predicha ahí")
    ap.add_argument("--out", default="outputs/gradcam.jpg")
    args = ap.parse_args(argv)

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"[ERROR] no se pudo leer {args.image}")
        return 1

    from cansat.checkpoints import load_into
    from evaluate import _build_model

    model = _build_model(args.checkpoint, {}, args.num_classes, args.arch)
    if model is None:
        print(f"[ERROR] no pude construir la arquitectura '{args.arch}'")
        return 1
    load_into(model, args.checkpoint, strict=False, min_loaded_frac=0.5)
    model.eval()

    tensor = PP.preprocess_bgr(bgr, args.img_size)
    punto = None
    if args.punto:
        try:
            y, x = (int(v) for v in args.punto.split(","))
            # A la resolución del tensor de entrada.
            punto = (int(y * args.img_size / bgr.shape[0]),
                     int(x * args.img_size / bgr.shape[1]))
        except ValueError:
            print("[ERROR] --punto espera 'Y,X'")
            return 1

    cam = gradcam(model, tensor, clase=args.clase, punto=punto)
    vis = superponer(bgr, cam)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    clase_final = args.clase if args.clase is not None else "dominante"
    print(f"[OK] Grad-CAM clase {clase_final} → {out} (cam {cam.shape[1]}x{cam.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
