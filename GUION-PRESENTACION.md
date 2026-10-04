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
| 8 | 6:30–8:30 | **Pi en vivo** (opcional) | Mostrar la Pi capturando (terminal o vista en vivo) y la telemetría con BME280 real; correr `demo_pi.ps1 -Frames 6` y mostrar el resultado en la estación. |
| 9 | 8:30–9:30 | Cierre | "Métricas medidas, límites declarados: hardware validado en placa (tiny ~3 s/frame, NPU ~3 fps, UART 40/40)." |

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
- NPU IMX500: persona detectada, **~3 fps sin CPU**; `.rpk` propio pendiente (Edge-MDT/Linux).
- UART: **40/40 paquetes** por GPIO15 con la Heltec; contrato v2 con checksum.
- Tipo de desastre: LOEO **0.33** (varianza ±0.04); no se adopta one-vs-rest (0.26).
- Calibración: ECE 0.47 → 0.12, pero monótona sobre p_incendio: la política no cambia.
- Todo el detalle en `INFORME-IA-Y-ESTACION.md` y `RESUMEN-PROYECTO.md`.
