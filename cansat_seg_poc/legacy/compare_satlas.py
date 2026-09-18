"""
ComparaciÃ³n exploratoria contra SatlasPretrain (Swin-B).

â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
POR QUÃ‰ ESTO YA NO SE LLAMA test_satlas.py
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
El archivo original se llamaba ``test_satlas.py``, pero **no era un test**: no
hay pytest ni unittest en el proyecto, no tenÃ­a aserciones, y su nombre hacÃ­a
que cualquier runner lo tomara como prueba y fallara. Los tests reales estÃ¡n en
``tests/``.

AdemÃ¡s tenÃ­a tres defectos concretos:
  Â· ``torch.hub.load('allenai/satlas-pretrain', 'SwimB_MultiTask', ...)`` â€”
    **typo**: es ``SwinB_MultiTask`` (Swin, no Swim). Lanzaba al ejecutar.
  Â· AsumÃ­a que la salida del modelo es un ``dict`` con clave ``'land_cover'``.
    SegÃºn la API de satlas-pretrain, ``SwinB_MultiTask`` devuelve un **tensor**
    por tarea en el orden declarado, no un dict.
  Â· LeÃ­a ``dataset/loveda_raw/Test/Urban/images_png/5861.png``, que no existe en
    el repo (el dataset no viaja en el ZIP).

Sigue siendo un script exploratorio que **descarga pesos en runtime** y por eso
vive en ``tools/``, no en la raÃ­z: no forma parte del pipeline de vuelo ni del
de entrega.

Uso (con red y con torch):
    python tools/compare_satlas.py --image alguna/tile.png
    python tools/compare_satlas.py --folder dataset/loveda_raw/Test/Urban/images_png --n 5
"""
from __future__ import annotations

# ── legacy/: script fuera del pipeline activo. Para correrlo desde la raíz:
#     python legacy/compare_satlas.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

HUB_REPO = "allenai/satlas-pretrain"
HUB_MODEL = "SwinB_MultiTask"        # â† era "SwimB_MultiTask" (typo) en el original
LAND_COVER_CLASSES = 7               # satlas land_cover tiene 7 clases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="ComparaciÃ³n exploratoria con SatlasPretrain")
    ap.add_argument("--image", default=None)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args(argv)

    try:
        import torch
        from PIL import Image
        import torchvision.transforms as T
    except ImportError as e:
        print(f"[ERROR] Este script necesita torch/torchvision: {e}")
        print("        Es exploratorio y no forma parte del pipeline de vuelo.")
        return 1

    # â”€â”€ ImÃ¡genes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if args.image:
        paths = [Path(args.image)]
    elif args.folder:
        d = Path(args.folder)
        if not d.is_dir():
            print(f"[ERROR] No existe la carpeta: {d}")
            return 1
        paths = sorted(d.glob("*.png"))[:args.n]
    else:
        print("[ERROR] PasÃ¡ --image o --folder.")
        print("        (El original tenÃ­a una ruta a dataset/ hardcodeada que no")
        print("         existe en el repo: el dataset no viaja en el ZIP.)")
        return 1

    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("[ERROR] Ninguna de las imÃ¡genes existe.")
        return 1

    # â”€â”€ Modelo (descarga en runtime: sÃ³lo con red) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"[i] Descargando {HUB_REPO} / {HUB_MODEL} (requiere red y ~350 MB)...")
    try:
        model = torch.hub.load(HUB_REPO, HUB_MODEL, pretrained=True)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar desde torch.hub: {type(e).__name__}: {e}")
        print("        VerificÃ¡ el nombre del entrypoint en el repo de satlas-pretrain;")
        print(f"        el original usaba '{HUB_MODEL.replace('Swin','Swim')}', que no existe.")
        return 1
    model.eval()

    transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"\n{'=' * 66}")
    print(f"  SatlasPretrain land_cover sobre {len(paths)} tile(s)")
    print("  âš  Las clases de Satlas NO son las nuestras. Esto es exploratorio:")
    print("    sirve para ver quÃ© encuentra un modelo preentrenado en satÃ©lite,")
    print("    no para comparar mIoU contra el nuestro.")
    print(f"{'=' * 66}")

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  [WARN] no se pudo abrir {p.name}: {e}")
            continue
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(x)

        # âš  El original hacÃ­a outputs['land_cover'], asumiendo un dict.
        #   SwinB_MultiTask devuelve tensores; se acepta cualquiera de las dos
        #   formas para no volver a romper.
        if isinstance(out, dict):
            seg_t = out.get("land_cover")
            if seg_t is None:
                print(f"  [WARN] la salida es un dict sin 'land_cover'. "
                      f"Claves: {list(out)}")
                continue
        elif isinstance(out, (list, tuple)):
            seg_t = out[0]
            print("  [i] la salida es una lista/tupla: se toma el primer tensor.")
        else:
            seg_t = out

        seg = seg_t.argmax(1).squeeze().cpu().numpy()
        uniq, counts = np.unique(seg, return_counts=True)
        dist = ", ".join(f"c{int(u)}:{100 * c / seg.size:.0f}%"
                         for u, c in zip(uniq, counts, strict=False))
        print(f"  {p.name:<24} shape {seg.shape} Â· clases {dist}")

    print(f"{'=' * 66}")
    print("  Para una comparaciÃ³n real de mIoU usÃ¡ evaluate.py contra LoveDA Val")
    print("  con las etiquetas remapeadas, no este script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
