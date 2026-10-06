# Guion de presentación — CanSat LB135 (miércoles 7 de octubre)

Duración objetivo: **8–10 min** + preguntas. Recorrido completo:
Vuelo → Detalle → Post-vuelo → Informe → Jurado, con la Pi en vivo como cierre.

## Roles

| Rol | Persona | Qué hace |
|---|---|---|
| Presentador | — | Habla, señala la pantalla |
| Operador | — | Maneja la estación (clicks/teclas) |
| Técnico Pi | — | Enciende/muestra la Pi y la Heltec (opcional) |

## Antes de entrar (5 min antes)

1. `cd EstacionTerrena_MuestreoDeDatos && python web_server.py` → abrir
   **http://localhost:8000** en el navegador (zoom 100 %, F11 pantalla completa).
2. Pi encendida, Heltec por USB y cableada a GPIO15 (si se muestra en vivo).
3. Backup a mano: `EstacionTerrena_MuestreoDeDatos/docs/poster_LB135.png`
   abierto y el video `EstacionTerrena_MuestreoDeDatos/docs/demo_respaldo.webm`.

## Guion (con teclas)

| # | Tiempo | Qué se muestra | Qué se dice (idea) |
|---|---|---|---|
| 1 | 0:00–0:30 | **Vuelo** (vista inicial) | "El CanSat LB135 segmenta terreno, detecta daño, flood y fuego; la estación es 100 % offline." |
| 2 | 0:30–1:30 | Corredor + Briefing | Señalar el corredor, el momento crítico y la barra de terreno. `Enter` para abrir el frame crítico. |
| 3 | 1:30–2:30 | **Detalle** del frame | Mostrar el HUD (personas, daño), las pestañas de imagen (vis/ens_seg/EDSR) y el botón **Grad-CAM** (explicabilidad). |
| 4 | 2:30–3:30 | **Post-vuelo** | Resumen del contrato, veredictos, pérdidas estimadas con supuestos declarados (banda ×0.5–×2). |
| 5 | 3:30–4:30 | **Informe** | `Imprimir/PDF` (sale en tinta). "Todo lo que se ve sale de la telemetría medida." |
| 6 | 4:30–5:30 | **Consulta Terrestre** (en Vuelo/Sampling) | Probar un chip: "¿cuántas personas hay a menos de 200 m de una vía?" → motor simbólico, sin inventar. |
| 7 | 5:30–6:30 | **Jurado** | Portada + frames críticos. "Honestidad: LOEO 0.33, el tipo es pista contextual; la señal fuerte es daño/severidad." |
| 8 | 6:30–8:30 | **Pi en vivo** (opcional) | Mostrar la Pi capturando (terminal o vista en vivo) y la telemetría con BME280 real; correr `demo_pi.ps1 -Frames 6` y mostrar el resultado en la estación. Alternativa **pull en vivo**: con `pi/servidor_vivo.py` corriendo en la Pi, `python tools/vivo_pi.py` sincroniza el vuelo cada 2 s y la web se actualiza sola. **Beat estrella (30 s)**: `python -m cansat.imx500_seg --model /home/pi/modelos/network_compacto.rpk --clases love` → NUESTRO modelo segmentando en el NPU en vivo (imprime la cobertura por clase a ~7 fps, sin CPU). |
| 9 | 8:30–9:30 | Cierre | "Métricas medidas, límites declarados: hardware validado en placa (tiny ~3 s/frame, YOLO11n 6.8 fps y **nuestro `.rpk` 16.6 fps** en el NPU, UART con soak 597/597)." |

## Teclas útiles (atajos de la estación)

`1/2/3` vistas Vuelo/Post/Informe · `A` solo alertas · `/` buscar · `E` exportar
CSV · `P` presentación automática · `Esc` cerrar. (Ver pie de la app.)

## Plan B por si falla algo

| Falla | Qué hacer |
|---|---|
| La estación no carga | Abrir el **póster PNG** y el **video de respaldo**; seguir el guion con ellos. |
| La Pi no responde | Mostrar la captura ya hecha (misión en la estación) y explicar el hardware con las fotos de evidencia. |
| La Heltec no emite | `demo_vivo.sh` avisa y usa atmósfera simulada; decir que el UART ya fue validado (40/40). |
| No hay red en el aula | Todo es offline salvo la Pi; la estación y el póster funcionan igual. |
| Poco tiempo | Saltar pasos 6 y 8 (Consulta y Pi en vivo) y cerrar con el 7 + 9. |

## Datos para memorizar (respuestas a preguntas típicas)

- Terreno de vuelo: tiny@224, **~3 s/frame medido** (v2@224 tarda ~4 min: no vuela).
- Detección NPU: **YOLO11n mAP 0.374** (~7 fps sin CPU) vs SSD 0.218; segmentación
  DeepLabV3+ a ~1 fps (no vuela, queda de herramienta); pose **descartada** con
  evidencia (mAP 0.188, sin esqueleto confiable).
- **`.rpk` PROPIO HECHO (2026-10-05)**: nuestro tiny de terreno corre en el NPU
  (conversión MCT + Sony en Docker; **memoria 4.28→1.84 MB de 8**, variante
  compacta **7-17 fps**, CPU 100 % libre). El NPU corre una red por vez: el
  vuelo mantiene YOLO (detección) + terreno en CPU.
- **LoRa implementado en el firmware** (TX vuelo / RX estación, 915 MHz) y
  **GPS real** (ATGM336H, NMEA con fallback simulado declarado).
- Modelo de daño **ep6**: 0.780 UAV / 0.414 xBD (antes 0.735 / 0.096);
  **MoE** de daño integrado al post-vuelo (gate 0.513 vs 0.490 del mejor fijo).
- **gray-world** en el preprocesamiento: **+2.05 pts** de mIoU del modelo de
  vuelo (val completo) con costo nulo.
- UART: **40/40 por GPIO15** y **soak 10 min: 597/597 paquetes, 0 pérdidas**;
  contrato v2 con checksum; eventos de vuelo por MPU6050 (despegue/aterrizaje)
  validados en hardware.
- Pi: salud en `pi_health.json` (40–42 °C, sin throttling); autostart del vuelo
  por **systemd** validado (`cansat-vuelo.service`); pull en vivo por HTTP.
- Tipo de desastre: LOEO **0.33** (varianza ±0.04); no se adopta one-vs-rest (0.26).
- Calibración: ECE 0.47 → 0.12, pero monótona sobre p_incendio: la política no cambia.
- Experimentos descartados CON evidencia: destilado B5→tiny (0.433 vs 0.440),
  aug UAV, tipo balanceado, calibración por clase, tiny flood/fuego.
- Todo el detalle en `INFORME-IA-Y-ESTACION.md` y `RESUMEN-PROYECTO.md`.
