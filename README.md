# CANSAT — CanSat "La Base" (Equipo 135, CONAE × ACEMA 2026)

Repositorio del proyecto CanSat: software de vuelo con IA a bordo y estación
terrena. La misión secundaria genera un mapa del terreno con imágenes mejoradas
por IA, porcentajes de cobertura (vegetación, edificios, agua, suelo), estrés
ambiental, detección de daños y estimación de pérdidas humanas.

## Estructura

| Carpeta / archivo | Qué es |
|---|---|
| `cansat_seg_poc/` | **Software de vuelo**: pipeline de misión (Raspberry Pi Zero W v1 + AI Camera IMX500), modelos de IA (segmentación de terreno, daño, inundación), entrenamiento/evaluación, deploy para la Pi. Ver `cansat_seg_poc/README.md` y `cansat_seg_poc/MODELS.yaml`. |
| `EstacionTerrena_MuestreoDeDatos/` | **Estación terrena**: servidor web (stdlib, SSE) + app React (Vuelo / Post / Informe / Presentación), simulacro, checklist de integridad. Ver `EstacionTerrena_MuestreoDeDatos/README-web.md`. |
| `La Base - DPD CANSAT 2026.pdf` | Documento Preliminar de Diseño presentado. |
| `PLAN_MEJORA_IA_PI_ZERO_W_IMX500.md` | Plan de mejora de IA para la Pi Zero W v1 + IMX500 (tracks F0-F7). |

## Puesta en marcha rápida

```bash
# Software de vuelo (PC)
cd cansat_seg_poc
pip install -r requirements.txt
python -m pytest tests -q                     # 190 tests
python mission_pipeline.py --folder tiles --frames 3 --no-detect --no-damage

# Estación terrena
cd EstacionTerrena_MuestreoDeDatos
python web_server.py                          # servidor en localhost
cd web-app && npm install && npm run dev      # UI de desarrollo
npm run smoke                                 # verificación E2E
```

## Estado y evidencias

- Métricas de modelos: `cansat_seg_poc/MODELS.yaml` y `cansat_seg_poc/outputs/metrics/*.json`
  (todo número citado sale de un JSON medido, no de memoria).
- Reporte de pruebas: `cansat_seg_poc/docs/reporte_pruebas.md`.
- Alineación con el DPD: `cansat_seg_poc/docs/ALINEACION-DPD.md`.
- Sustitución de hardware (Zero 2 W → Zero W v1, Camera Module 3 → AI Camera):
  `cansat_seg_poc/docs/SUSTITUCION-HW.md`.

## Hardware de vuelo

Raspberry Pi Zero W v1 (ARMv6, 32 bits) + AI Camera (IMX500) + Heltec WiFi LoRa 32 V3
(computadora de vuelo ESP32-S3, dos unidades: vuelo y estación), BME280, MPU6050,
GPS ATGM336H, batería 18650 con step-up de 5V.
