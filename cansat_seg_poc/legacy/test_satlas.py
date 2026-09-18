"""
test_satlas.py - Compara SatlasPretrain vs nuestro modelo actual
"""

# ── legacy/: script fuera del pipeline activo. Para correrlo desde la raíz:
#     python legacy/test_satlas.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Descargar el modelo SatlasPretrain (Swin-Base, mejor para segmentación)
# https://huggingface.co/allenai/satlas-pretrain
model = torch.hub.load('allenai/satlas-pretrain', 'SwimB_MultiTask', pretrained=True)
model.eval()

# Probar en una tile de LoveDA
img = Image.open('dataset/loveda_raw/Test/Urban/images_png/5861.png')
transform = T.Compose([T.Resize(512), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
x = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(x)
    # SatlasPretrain tiene 137 tareas, extraemos "land_cover" (segmentación)
    seg = outputs['land_cover'].argmax(1).squeeze().numpy()

print(f"Clases detectadas: {np.unique(seg)}")
print(f"Forma: {seg.shape}")
