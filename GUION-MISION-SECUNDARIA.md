# Guión — Misión secundaria (2–3 min) · CanSat LB135

> Objetivo: contar QUÉ ES, LO MÁS IMPORTANTE (hecho) y EN QUÉ SE ESTÁ TRABAJANDO,
> con algo en pantalla en todo momento. Todos los números son medidos
> (`docs/benchmarks/`, `INFORME-IA-Y-ESTACION.md`).

## Antes de empezar (2 min)

1. `cd EstacionTerrena_MuestreoDeDatos && python web_server.py` → abrir
   **http://localhost:8000** (F11, zoom 100 %).
2. Tener abierto el **esquema** `EstacionTerrena_MuestreoDeDatos/docs/
   esquema_mision_secundaria.png` (para el arranque y el cierre).
3. Plan B a mano: `docs/demo_respaldo.webm` (video) y `docs/poster_LB135.png`.
4. Si se muestra el rpk en vivo: Pi encendida y
   `ssh pi@192.168.68.240` con el comando listo (ver tramo 4).

## Guión

| T | Qué decir (idea) | Qué mostrar (pantalla) |
|---|---|---|
| 0:00 | "La misión secundaria son **dos sistemas**: IA a bordo en la Raspberry Pi con la AI Camera, y la estación terrena que reconstruye el vuelo. En vuelo se segmenta el terreno y se detectan personas sin gastar CPU; en tierra se calcula daño, inundación, fuego y severidad." | **Esquema** (`esquema_mision_secundaria.png`) |
| 0:25 | "Lo que se ve acá es una misión completa: el corredor de frames del descenso, la cobertura de terreno por frame y las alertas. Todo sale de la telemetría del contrato de 34 columnas, con posición GPS por imagen." | **Estación → Vuelo**: corredor + barra de terreno + KPIs. `Enter` abre el frame crítico. |
| 0:50 | "Estos son los números que sostienen la misión, todos medidos: terreno **0.44 mIoU** en el modelo que vuela a ~3 s/frame; daño **0.780** en dominio UAV; fuego **0.79**; inundación **0.489**; severidad de colapso **0.633**; y el detector de personas **0.374 mAP** corriendo en el NPU a ~7 fps." | **Detalle del frame**: HUD (personas, daño) + pestañas de imagen + botón **Grad-CAM** (explicabilidad). |
| 1:20 | "La estación es 100 % offline para el jurado: 5 vistas, auto-refresh, Consulta Terrestre **simbólica** (no inventa: responde con las máscaras), e Informe imprimible en tinta. El post-vuelo completo corre en la PC sobre los frames crudos." | **Post-vuelo** (tecla `2`) → **Informe** (`3`) → probar un chip de la **Consulta** ("¿personas a menos de 200 m de una vía?"). |
| 1:50 | "Y esto no es simulación de escritorio: la Pi Zero + AI Camera está **validada en hardware** — detección 6.8 fps sin CPU, UART con soak de 10 minutos **597/597 sin pérdidas**, LoRa y GPS implementados en el firmware. Y **nuestro modelo de terreno convertido a `.rpk` corre en el NPU** a 16.6 fps." | Foto de la Pi (`docs/evidencia/11b_pi_imx500_foto.png`) o **demo en vivo**: `python -m cansat.imx500_seg --model /home/pi/modelos/network_compacto.rpk --clases love --seconds 10` (imprime cobertura por clase). |
| 2:20 | "Ahora mismo seguimos mejorando con evidencia: adoptamos el modelo de daño **ep6** (0.780 vs 0.735) y la corrección de color **gray-world** (+2 pts de mIoU); integramos un **MoE** de daño en el post-vuelo. Y lo que no funcionó también se declara: UDA, augmentación y severidad multi-desastre se probaron y se rechazaron con números. Lo próximo: más modelos al NPU y el vuelo de campo." | Volver al **esquema** (franja **AHORA**) o a la vista **Jurado** para el cierre. |
| 2:45 | Cierre: "Cada número que mostramos sale de una medición, y los límites también se declaran." | Póster o portada. |

## Claves y atajos

`1/2/3` Vuelo/Post/Informe · `G` Grad-CAM en el detalle · `A` solo alertas ·
`P` presentación automática · `Esc` cerrar.

## Qué tener listo (checklist rápido)

- [ ] Servidor de la estación corriendo y navegador en pantalla completa.
- [ ] Esquema PNG abierto en el visor de imágenes (Alt+Tab listo).
- [ ] Un frame con **Grad-CAM** ya cacheado (abrir una vez antes).
- [ ] Pi encendida; si se muestra el rpk, terminal con el comando escrito.
- [ ] Video de respaldo y póster a mano (plan B sin red/luz).
