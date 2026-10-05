# Conversión a IMX500 (.rpk) — guía de la cadena verificada (F4)

> **Estado: EJECUTADA el 2026-10-05.** Nuestro tiny de terreno corre en el NPU
> del IMX500: `network.rpk` (1.53 MB) en `/home/pi/modelos/`, memoria **4.28 MB
> de 8 MB (54 %)**, KPI del NPU **4.5 ms**, end-to-end ~0.4-1.2 fps (limitado
> por la transferencia de la salida de 5×224×224; la CPU queda libre).
> Evidencia: `docs/benchmarks/rpk_propio_tiny.json` +
> `docs/evidencia/14_pi_rpk_propio.png`. La cadena corre en **Docker Desktop
> (WSL2)**: el tooling de Sony es público en PyPI (`edge-mdt[pt]`,
> `imx500-converter`) — no hace falta registro ni PC Linux nativa.

## 0. Qué se convierte y en qué orden

| Prioridad | Modelo ONNX | Tamaño FP32 | Por qué |
|---|---|---|---|
| 1 | `cansat_seg_terrain_tiny_224.onnx` | 4.3 MB | ✅ **CONVERTIDO (2026-10-05) → `network.rpk`** (memoria 4.28/8 MB; KPI 4.5 ms) |
| 2 | `cansat_flood_specialist_224.onnx` | 51 MB | FP32 no entra; **requiere cuantización INT8** (~12.8 MB de pesos → no entra igual; ver §5) |
| 3 | `cansat_fire_smoke.onnx` | 51 MB | Ídem flood |
| 4 | `cansat_damage_v3_bal_ep6.onnx` / `cansat_severity.onnx` | 51 MB | Ídem; el daño/seguimiento puede quedar en `cv2.dnn` |

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

### 3b. Comandos EXACTOS que funcionaron (2026-10-05, Docker/WSL2)

```bash
# 0) contenedor con el repo montado
docker run -d --name cansat-mdt -v "<repo>:/work" -w /work python:3.11-slim sleep infinity

# 1) tooling: edge-mdt[pt] de PyPI + torch CPU 2.7 (con 2.14 el export de MCT
#    falla por el exporter dynamo) + onnxscript (lo pide torch 2.7+) + java
#    (el compilador DSP de Sony usa java; sin él: "sdspconv exited 127")
docker exec cansat-mdt bash -c "pip install --no-cache-dir 'edge-mdt[pt]'"
docker exec cansat-mdt bash -c "pip install --no-cache-dir 'torch==2.7.1' \
    'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cpu"
docker exec cansat-mdt bash -c "pip install --no-cache-dir onnxscript"
docker exec cansat-mdt bash -c "apt-get update && apt-get install -y default-jre-headless"

# 2) PTQ con MCT (TPC IMX500 5.0) desde el checkpoint PyTorch + export ONNX
docker exec cansat-mdt python tools/convertir_rpk.py \
    --checkpoint outputs/best_terrain_tiny.pth \
    --calib-dir dataset/loveda_remapped/Train --out outputs/tiny_mct.onnx

# 3) compilar con el converter de Sony (~25 s)
docker exec cansat-mdt imxconv-pt -i outputs/tiny_mct.onnx \
    -o outputs/rpk_out --no-input-persistency --overwrite-output

# 4) empaquetar EN LA PI (apt imx500-tools, ya instalado con imx500-all)
imx500-package -i packerOut.zip -o /home/pi/modelos/   # → network.rpk
```

El contenedor queda creado con todo instalado: se reusa con
`docker start cansat-mdt` (y se apaga con `docker stop cansat-mdt`).

**Variante compacta (recomendada)**: `tools/convertir_rpk.py --compacta`
exporta el tiny truncado a /8 (sin el upsample final): la salida pasa de
`(1, 5, 224, 224)` (1 MB/frame) a `(1, 5, 28, 28)` (~16 KB/frame) y el
end-to-end deja de estar limitado por la transferencia:

| Variante | Memoria | KPI dnn | fps (lazo crudo) | fps (módulo) |
|---|---|---|---|---|
| full (224×224) | 4.28/8 MB (54 %) | 4.5 ms | 1.2 | 0.7 |
| **compacta (/8)** | **1.84/8 MB (23 %)** | **2.6 ms** | **16.6** | **7.4** |

Notas medidas: `imxconv-pt` reporta el `MemoryReport`; MCT avisa de tensores con
rango dinámico subóptimo en dos FullyConnected del backbone (warnings, no
bloquean). `cansat/imx500_seg.py` convierte los logits a máscara con
`mask_desde_salida()` y `--clases love` resume las 5 clases de terreno.
TRADEOFF: el IMX500 corre UNA red por vez; el vuelo mantiene YOLO11n (NPU) +
terreno CPU y el rpk compacto queda como opción (ver `MODELS.yaml`).

## 4. Probar en la Pi (verificado 2026-10-05)

```bash
scp network_compacto.rpk pi@cansat.local:/home/pi/modelos/

# segmentación con NUESTRO modelo (logits 5 clases → máscara LoveDA)
python -m cansat.imx500_seg --model /home/pi/modelos/network_compacto.rpk \
    --clases love --seconds 12 --guardar salida.png
# → warm-up ~5 s; imprime la cobertura por clase (vegetacion/edificio/agua/…)
```

Medido con la variante **compacta**: KPI del NPU 2.6 ms; **7.4 fps con el
módulo completo / 16.6 fps en el lazo crudo** (la CPU queda 100 % libre). La
variante full (224×224) queda en 0.4-1.2 fps por la transferencia de 1 MB/frame.

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
