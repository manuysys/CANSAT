# LoveDA Segmentation Pipeline - CanSat

> ## 📌 Estado del proyecto (2026-09-17)
> Este README describe **sólo la fase 1** (dataset LoveDA). El proyecto completo
> es un pipeline de misión con 60+ scripts; la documentación viva está en:
>
> | Documento | Qué contiene |
> |---|---|
> | [`docs/ALINEACION-DPD.md`](docs/ALINEACION-DPD.md) | Cada requisito del DPD → implementación → estado |
> | [`docs/DATASETS-Y-TECNICAS.md`](docs/DATASETS-Y-TECNICAS.md) | Datasets (BRIGHT, RescueNet…) y técnicas de mejora |
> | [`MODELS.yaml`](MODELS.yaml) | **Qué modelo vuela**, con métricas medidas y hash |
> | [`docs/decisiones.yaml`](docs/decisiones.yaml) | Decisiones de diseño (fuente del informe) |
> | [`docs/reporte_pruebas.md`](docs/reporte_pruebas.md) | Informe generado con números medidos |
> | [`pi/guia_pi.md`](pi/guia_pi.md) | Puesta a punto de la Raspberry + AI Camera |
>
> **Modelo de vuelo**: `outputs/cansat_seg_terrain_v2.onnx` (DeepLabV3+
> MobileNetV2, 5 clases, **mIoU 52.19 %** en el Val completo).
> **Entrada**: `python mission_pipeline.py --camera --no-damage --det-backend imx500`.
>
> ### Datasets y licencias
> | Dataset | Uso | Licencia | Restricción |
> |---|---|---|---|
> | LoveDA (Zenodo 5706578) | Segmentación de terreno (5 clases) | **CC BY 4.0** | Atribución |
> | xBD / xView2 | Modelos de daño estructural | **CC BY-NC-SA 4.0** | **No comercial + ShareAlike**: los pesos de daño son obra derivada |
> | FloodNet (Kaggle) | Especialista de inundación | sin declarar en la ficha | Verificar antes de distribuir |
> | VisDrone | Fine-tune del detector de personas | términos del challenge | Verificar |
> | Esri World Imagery | `baseline.png` del siamés | términos de ArcGIS Online | Verificar |
> | EDSR (`models/EDSR_x2.pb`) | Super-resolución post-vuelo | ver repo de origen (Saafke/EDSR_Tensorflow) | Verificar |
>
> Los modelos entrenados **solo con LoveDA** (terreno) no tienen la
> restricción no-comercial; los de **daño** (xBD) sí. Ver `docs/reporte_pruebas.md` §6.

Pipeline de procesamiento del dataset LoveDA para segmentación semántica en el proyecto CanSat.

## Flujo de Trabajo

```
1. download_loveda.py    → Descarga Train.zip y Val.zip desde Zenodo
2. remap_loveda.py       → Remapea máscaras a 5 clases personalizadas
3. verify_dataset.py     → Verifica integridad del dataset
4. visualize_masks.py    → Genera visualizaciones para revisión
```

## Ejecución Rápida

```bash
# Opción 1: Ejecutar todo el pipeline de una vez
run_pipeline.bat

# Opción 2: Ejecutar paso a paso
venv\Scripts\activate
python download_loveda.py
python remap_loveda.py
python verify_dataset.py
python visualize_masks.py
```

## Estructura de Directorios

```
cansat_seg_poc/
├── dataset/
│   ├── loveda_raw/          # Datos originales sin modificar
│   │   ├── Train/
│   │   │   ├── Rural/
│   │   │   │   ├── images_png/
│   │   │   │   └── masks_png/
│   │   │   └── Urban/
│   │   │       ├── images_png/
│   │   │       └── masks_png/
│   │   └── Val/
│   │       ├── Rural/
│   │       └── Urban/
│   ├── loveda_remapped/     # Máscaras transformadas a 5 clases
│   │   ├── Train/
│   │   │   ├── Rural/
│   │   │   │   ├── images_png/
│   │   │   │   └── masks_png/
│   │   │   └── Urban/
│   │   └── Val/
│   ├── visualizations/      # Visualizaciones para revisión
│   └── downloads/           # Archivos ZIP descargados
├── download_loveda.py       # Descarga desde Zenodo
├── remap_loveda.py          # Remapea clases
├── verify_dataset.py        # Verifica integridad
├── visualize_masks.py       # Genera visualizaciones
├── requirements.txt         # Dependencias
└── run_pipeline.bat         # Script de ejecución
```

## Detalles de Cada Script

### 1. download_loveda.py

Descarga el dataset LoveDA desde el registro oficial de Zenodo.

**Fuente:** https://zenodo.org/records/5706578

**Archivos descargados:**
- `Train.zip` (~4.0 GB, MD5: de2b196043ed9b4af1690b3f9a7d558f)
- `Val.zip` (~2.4 GB, MD5: 84cae2577468ff0b5386758bb386d31d)

**Características:**
- ✅ Soporte para reanudación de descargas (HTTP Range)
- ✅ Detección de archivos ya descargados
- ✅ Verificación de integridad MD5
- ✅ NO descarga Test.zip (solo Train y Val)

**Salida:** `dataset/loveda_raw/`

---

### 2. remap_loveda.py

Remapea las máscaras originales de LoveDA (8 clases) a nuestras 5 clases personalizadas.

**Mapeo aplicado:**

| LoveDA | Nombre Original | → | Nuestra Clase | Índice |
|--------|----------------|---|---------------|--------|
| 0 | no-data | → | ignore | 255 |
| 1 | background | → | other | 4 |
| 2 | building | → | building | 1 |
| 3 | road | → | bare_ground | 3 |
| 4 | water | → | water | 2 |
| 5 | barren | → | bare_ground | 3 |
| 6 | forest | → | vegetation | 0 |
| 7 | agriculture | → | vegetation | 0 |
| 255 | sin dato | → | ignore | 255 |

**Nuestras 5 clases:**
- 0: vegetation (forest + agriculture)
- 1: building
- 2: water
- 3: bare_ground (road + barren)
- 4: other (background)
- 255: ignore (no-data)

**Salida:** `dataset/loveda_remapped/`

---

### 3. verify_dataset.py

Verifica la integridad del dataset remapeado:

- ✓ Todas las imágenes tienen su máscara correspondiente
- ✓ Todas las máscaras tienen su imagen correspondiente
- ✓ Los valores únicos son correctos (0-4 y 255)
- ✓ No hay máscaras vacías (todo 255)
- ✓ Distribución de clases balanceada
- ✓ Porcentaje de píxeles ignore

**Salida:** Reporte de verificación en consola

---

### 4. visualize_masks.py

Genera visualizaciones del dataset remapeado para revisión visual.

**Para cada imagen genera:**
1. Imagen original
2. Máscara coloreada (5 clases)
3. Overlay (imagen + máscara con transparencia)

**Colores de visualización:**
- Verde: vegetation
- Rojo: building
- Azul: water
- Marrón: bare_ground
- Gris: other
- Negro: ignore

**Salida:** `dataset/visualizations/` (mínimo 10 visualizaciones)

---

## Dependencias

Todas las dependencias están en `requirements.txt`:

```
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
requests>=2.31.0
tqdm>=4.65.0
```

El entorno virtual (`venv/`) ya tiene todas las dependencias instaladas.

## Notas Importantes

### Sobre Test.zip

**NO se descarga Test.zip** porque:
1. El dataset de test no tiene máscaras públicas (solo imágenes)
2. Se usa para evaluación en competencias oficiales
3. Para entrenamiento solo necesitamos Train y Val

### Sobre EarthVQA

**NO se descarga EarthVQA** en esta etapa. Es un dataset diferente y se integrará posteriormente si es necesario.

### Verificación Visual Antes de Entrenar

Es **crítico** revisar las visualizaciones generadas por `visualize_masks.py` antes de proceder con el entrenamiento:

1. Abrir `dataset/visualizations/`
2. Revisar que las máscaras coloreadas correspondan a las imágenes
3. Verificar que las clases estén correctamente asignadas
4. Confirmar que no hay errores en el remapeo

Si las visualizaciones son correctas, el dataset está listo para entrenamiento.

## Solución de Problemas

### Error de descarga
```
[ERROR] Falló la descarga
```
- Verifica tu conexión a internet
- El script soporta reanudación, ejecútalo de nuevo
- Si persiste, verifica que Zenodo esté accesible

### MD5 no coincide
```
[ERROR] MD5 no coincide!
```
- El archivo se corrompió durante la descarga
- El script lo elimina automáticamente
- Ejecuta de nuevo para re-descargar

### No se encontraron pares imagen-máscara
```
[ERROR] No se encontraron pares
```
- Verifica que `dataset/loveda_raw/` tenga la estructura correcta
- Asegúrate de haber ejecutado `download_loveda.py` primero

### Máscaras vacías
```
[WARN] X máscaras completamente vacías
```
- Revisa las visualizaciones de esas máscaras
- Puede ser normal si hay imágenes sin datos anotados

## Próximos Pasos

Después de completar este pipeline:

1. ✅ Revisar visualizaciones en `dataset/visualizations/`
2. ⏭️ Preparar dataloaders para PyTorch
3. ⏭️ Entrenar modelo de segmentación
4. ⏭️ Evaluar en Val set
5. ⏭️ (Opcional) Integrar EarthVQA
6. ⏭️ (Opcional) Descargar Test.zip para evaluación final
