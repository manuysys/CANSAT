# legacy/ — scripts fuera del pipeline activo

Acá viven los scripts que **no sirven para el vuelo ni para la entrega** y que
la auditoría del 2026-09-16/17 marcó como rotos, abandonados o reemplazados.
Se conservan por trazabilidad histórica (y porque alguno puede rescatarse),
pero **no forman parte del sistema**.

Para correr cualquiera desde la raíz del proyecto (los scripts asumen CWD=raíz):

```bash
python legacy/<script>.py
```

Cada archivo lleva un bootstrap de `sys.path` para que sus `from train import …`
sigan funcionando desde acá.

| Archivo | Por qué está acá |
|---|---|
| `test_satlas.py` | **No es un test** (no hay pytest para él), tiene `'SwimB_MultiTask'` (typo por `SwinB`), asume que la salida es un dict, lee una imagen que no existe y descarga pesos de `torch.hub` en runtime. |
| `compare_satlas.py` | Comparador del anterior; mismo destino. |
| `train_qat.py` | QAT con `torch.quantization` eager (deprecado y roto en torch 2.x para módulos con `F.interpolate`), `qconfig='fbgemm'` (x86: inútil para la Pi ARM), pesos hardcodeados y un mIoU `0.5317` impreso como constante. La decisión es volar FP32. |
| `mae_pretrain.py` | Preentrenamiento MAE: entrenaba sobre `loveda_raw` completo (incluido Val), guardaba el encoder en cada época en la raíz y usaba una geometría (stride 32) que después se cargaba en un backbone dilatado (OS8). |
| `mae_finetune_terrain.py` | Fine-tune con una arquitectura distinta al resto (sin decoder, stride 32) y evaluación sesgada. |
| `pack_mae_data.py` | Empaquetador de datos del camino MAE. |
| `mae_mobilenetv2_encoder.pth` | Pesos del camino MAE (abandonado). |
| `train_v3.py` | Fork semi-supervisado (CBAM+MAE+pseudo) con BN congelado, sin evidencia de mejora y que exportaba con nombre equivocado (`_v4.onnx`). |
| `train_v4.py` | Fork con el bug de `AugDataset` que entrenaba con etiquetas desalineadas en ~1/3 de las muestras; `BoundaryLoss` no diferenciable (corregidos en el código, pero el checkpoint arrastra el defecto). |
| `yolo26n.pt` | Pesos YOLO26n sin ninguna referencia en el repo (el detector de vuelo es `models/yolov8s.pt`). |

> Los pesos de `train_v3/v4/mae` que quedaron en `outputs/` están marcados
> **EXPERIMENTAL/ABANDONADO** en `MODELS.yaml`. El de `train_v4` no debe usarse.
