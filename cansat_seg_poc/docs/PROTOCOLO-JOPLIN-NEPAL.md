# Protocolo Joplin / Nepal (daño cross-evento)

Fecha: 2026-09-20. Modelo: `outputs/best_damage3.pth` (xBD, 10 eventos originales
— Joplin y Nepal NO estuvieron en su entrenamiento, son held-out genuinos).
Herramienta: `tools/eval_damage_manifests.py --splits all --cpu`.

## Resultados medidos

| Evento (held-out) | IoU binaria de daño | DAÑADO* sobre edificios | tiles |
|---|---|---|---|
| **Joplin (tornado)** | **0.2332** | **0.4953** | 148 |
| **Nepal (inundación)** | **0.1266** | **0.2777** | 576 |
| Midwest flooding (referencia) | 0.1671 | 0.2941 | 244 |

## Lectura honesta

- El **patrón coincide con el paper de xBD** (ICDM 2022): al dejar fuera
  **Joplin** la performance cae poco, y al dejar fuera **Nepal** se derrumba
  (ellos reportan F1 armónico 0.60→0.56 en Joplin y **0.60→0.06** en Nepal;
  nosotros medimos IoU, no F1, así que no son comparables 1:1 — lo comparable
  es la dirección y magnitud relativa).
- **Nepal es el peor caso** también para nosotros (0.127 vs 0.233 de Joplin):
  la inundación de Nepal tiene firma visual distinta al resto del dataset
  (geografía, vegetación, escala) — mismo diagnóstico que el paper.
- Joplin (tornado, viento) transfiere mejor que Nepal, consistente con que el
  daño por viento comparte textura con huracanes ya vistos.
- Estos números son de **daño en satélite 0.5 m**; el modelo de vuelo es otro
  (`cansat_damage_v3_bal.onnx`, UAV/RescueNet) y no se reemplaza por esto.

## Reproducir

```
python tools/eval_damage_manifests.py --checkpoint outputs/best_damage3.pth \
    --manifest dataset/xbd_masks/manifest_joplin-tornado.csv --splits all --cpu
python tools/eval_damage_manifests.py --checkpoint outputs/best_damage3.pth \
    --manifest dataset/xbd_masks/manifest_nepal-flooding.csv --splits all --cpu
```

Manifiestos por evento generados desde `dataset/xbd_masks/manifest.csv` (xBD
Tier 3 incluido). No se tocó ningún modelo: es solo medición.
