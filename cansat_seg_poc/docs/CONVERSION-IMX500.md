# Conversión a IMX500 (.rpk) — guía para la PC Linux (F4)

> **Estado**: guía escrita y verificada en cuanto a *requisitos del conversor*
> (los valida `audit_imx500.py`), pero **NO ejecutada** todavía: requiere la PC
> Linux con el tooling de Sony (Edge-MDT). Los comandos exactos pueden variar
> con la versión del converter — verificar con `imxconv-pt --help`.

## 0. Qué se convierte y en qué orden

| Prioridad | Modelo ONNX | Tamaño FP32 | Por qué |
|---|---|---|---|
| 1 | `cansat_seg_terrain_tiny_224.onnx` | 4.3 MB | El candidato NPU: entra holgado en el límite de <8 MB |
| 2 | `cansat_flood_specialist_224.onnx` | 51 MB | FP32 no entra; **requiere cuantización INT8** (~12.8 MB de pesos → no entra igual; ver §5) |
| 3 | `cansat_fire_smoke.onnx` | 51 MB | Ídem flood |
| 4 | `cansat_damage_v3_bal.onnx` / `cansat_severity.onnx` | 51 MB | Ídem; el daño/seguimiento puede quedar en `cv2.dnn` |

> ⚠ El límite del IMX500 es **<8 MB de memoria total** (pesos + tensores).
> Solo el `tiny` (1.08 M params) lo cumple con margen. Los modelos
> DeepLabV3+MV2 (5.5 M params) necesitan cuantización + compresión agresiva y
> aun así quedan al borde: medir antes de prometerlos.

## 1. Requisitos (PC Linux)

- Ubuntu 22.04/24.04 (x86_64), 16 GB de disco libre, Docker.
- Tooling **Sony Edge-MDT** (IMX500 Converter): imagen Docker oficial de Sony
  + `model_compression_toolkit` (MCT) en su versión compatible.
- Python 3.11 para los scripts de empaquetado.
- Los ONNX de `cansat_seg_poc/outputs/` (o `dist_pi/outputs/`, ya
  autocontenidos).

## 2. Verificación previa (en cualquier PC)

```bash
python audit_imx500.py --onnx outputs/cansat_seg_terrain_tiny_224.onnx --size 224
# Requisitos: opset 17 · batch fijo 1 · entrada 'input' [1,3,H,W] · salida 'logits'
#             · sin ops prohibidas · sin pesos externos · <200 MB
```

Los 6 modelos de vuelo ya pasan esta auditoría (`audit_imx500.py --all`).

## 3. Flujo de conversión (esquema)

```
ONNX FP32
   │  MCT (quantize + compress INT8, target IMX500)
   ▼
model_mct.onnx
   │  imxconv-pt -i model_mct.onnx -o out/
   ▼
out/  (grafo compilado + pesos)
   │  packager (imx500-package)
   ▼
packerOut.zip  →  network.rpk
```

Pasos típicos (ajustar a la versión):

```bash
# 1) Cuantizar con MCT (target Sony IMX500)
python -m model_compression_toolkit.ptq \
    --model model.onnx --target imx500 --output model_mct.onnx

# 2) Compilar con el converter de Sony (imagen Docker Edge-MDT)
imxconv-pt -i model_mct.onnx -o out/
#    (o dentro del contenedor: docker run --rm -v $PWD:/work sony/imx500-converter ...)

# 3) Empaquetar a .rpk
imx500-package -i out/ -o packerOut.zip
#    → el .rpk queda dentro del zip (network.rpk)
```

## 4. Probar en la Pi

```bash
# copiar el .rpk a la Pi
scp network.rpk pi@cansat.local:/home/pi/modelos/

# validar que el NPU lo corre (sin CPU)
python -m cansat.imx500 --model /home/pi/modelos/network.rpk --seconds 10
```

Si imprime detecciones/segmentaciones, el pipeline lo usa con:

```bash
python mission_pipeline.py --camera --det-backend imx500 --frames 5 \
    --onnx /home/pi/modelos/network.rpk
```

## 5. Si un modelo no entra en 8 MB

Opciones, en orden:

1. **Reducir la entrada** (224 → 192 → 160): los tensores intermedios son el
   costo dominante.
2. **Cuantización INT8 con MCT** (no INT8 dinámico: ese ya se descartó por
   23–26× más lento y no cargar en `cv2.dnn`).
3. **Pruning estructurado** de canales en ASPP/decoder + fine-tune corto.
4. **Cambiar de backbone** al diseño tiny (MobileNetV3-S + LR-ASPP ya está en
   el repo: 1.08 M params) o a un depthwise U-Net estilo PicoSAM2 (1.3 M).
5. Dejarlo en `cv2.dnn` (CPU) y usar el NPU solo para los que entren.

## 6. Qué NO esperar

- **SegFormer-B5** (338 MB, LayerNorm/Softmax/Erf/GELU): no pasa el converter.
- **Siamés** (2 entradas): inconvertible directo.
- **Ops prohibidas**: `Loop/If/Scan/GridSample/RoiAlign/ScatterND/Einsum/
  NonZero` (el audit las detecta).
- **INT8 dinámico** (`DynamicQuantizeLinear/ConvInteger`): no soportado.

## 7. Referencias

- Requisitos y auditoría: `audit_imx500.py`, `docs/reporte_pruebas.md`.
- Modelos y métricas: `MODELS.yaml`.
- Estado del arte y plan: `PLAN_MEJORA_IA_PI_ZERO_W_IMX500.md` (§F4, Apéndice).
- Sony: documentación de Edge-MDT / IMX500 Converter (developer.sony.com) y el
  tutorial de ArduCAM para modelos custom (2025-11).
