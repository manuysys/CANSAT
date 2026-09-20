# 🛰️ Estación Terrena Web — CanSat LB135

Visualización de la misión recuperada post-vuelo. Dos frontends para el mismo
backend y el mismo contrato de datos:

| Frontend | Ruta | Stack | Estado |
|---|---|---|---|
| **v4 (principal)** | `web-app/` | Vite + React + TypeScript + Tailwind v4 + shadcn/ui + Motion + Recharts + driver.js + zustand + **three.js/@react-three/fiber + GSAP** | activo |
| v2/v3 (legacy) | `web/` | HTML + CSS + JS vanilla, sin build | fallback |

El backend **no se tocó** para el rebuild: `web_server.py` sirve la API y los
estáticos, y ahora elige raíz automáticamente: **`web-app/dist/` si existe,
si no `web/`** (único cambio, más el flag `--root` que ya existía).

---

## 1. Arranque rápido

### Producción (PC del laboratorio, sin node)
```bash
cd web-app && npm install && npm run build   # genera web-app/dist/ (solo la primera vez)
cd .. && python web_server.py                # → http://localhost:8000
```
Con `dist/` presente, Python sirve el build estático y todo funciona offline
(fuente Inter empaquetada, sin CDNs).

### Desarrollo
```bash
python web_server.py                 # terminal 1, puerto 8000 (API + /img)
cd web-app && npm run dev            # terminal 2, puerto 5173 con proxy /api y /img → 8000
```

---

## 2. Stack y mapeo de features → librería

| Feature | Implementación |
|---|---|
| Tabs Vuelo / Post-vuelo / Informe | shadcn `Tabs` + indicador `layoutId="tab-pill"` (Motion) + crossfade 200 ms con `AnimatePresence mode="wait"` |
| KPIs con count-up | shadcn `Card` + `useMotionValue` + `useSpring` + `useTransform`; entrada staggered (`staggerChildren: 0.07`) |
| Curva daño vs altitud (X invertido 250→0) | Recharts `LineChart` + dots propios clickeables con anillo pulsante (`alert-ping`) en alertas |
| Donut de veredictos | Recharts `PieChart` animado |
| Corredor de descenso (eje Y = altitud) | Cards absolutas posicionadas por altitud con anti-solape; `whileInView` (scroll reveal) + `layout` al filtrar; selección con `layoutId="frame-glow"` |
| Detalle de frame (protagonista) | shadcn `Tabs` internas (Evidencia/Ensemble/EDSR/High-res, **ocultas si el archivo no existe**) + `AnimatePresence mode="wait"` con fade+scale 1.02→1 |
| Tabla timeline | shadcn `Table` + filas `motion.tr` con hover lift y `layout`; colapsable |
| Alertas | shadcn `Card` con borde semántico, pulso Motion en el icono y slide-in de las nuevas durante el auto-refresh; panel ausente si no hay alertas |
| Tour de primera vez | driver.js, 5 pasos, `localStorage['lb135-tour-v4']`; el botón “?” del header lo reabre |
| Tooltips de glosario | shadcn `Tooltip` en USI, NDVI, sampler, pri, daño, nitidez y cada diagnóstico (una línea) |
| Skeletons / estados de carga | shadcn `Skeleton` disponible; estados vacíos amigables con icono lucide + frase |
| Estado global | zustand (`src/store/mission.ts`): frames, selección, filtros, orden, vista, presentación |
| Iconos | lucide-react (monocromos) |
| Fuente | `@fontsource-variable/inter` (local, offline-safe) |
| Animaciones | paquete `motion`; `prefers-reduced-motion` respetado nativamente |

### Amigabilidad (v3 → v4)
- **Resumen narrativo automático** arriba de todo, recalculado en cada sync, con
  botón **“Ver momento crítico”** que selecciona el frame y scrollea al detalle.
- Microcopy en español claro; títulos de sección en mayúsculas suaves solo como
  etiqueta de panel; leyenda de terreno **persistente** junto a las barras.
- Empty states con icono + frase (“Sin alertas: la misión no vio desastres 🎉”).
- Footer con atajos: `↑↓` navegar · `Enter` detalle · `A` alertas · `/` buscar ·
  `1-2-3` vistas · `E` exportar · `P` auto · `Esc` cerrar.
- **Modo presentación** (▶ Presentar): fullscreen, avance cada 4 s con imagen
  grande + veredicto + diag; `Esc` sale, `←→` manual, dots de progreso clickeables.

---

## 3. Contrato de datos (solo lectura, idéntico al pipeline)

```
outputs/mission/telemetry.csv      1 fila por frame (23 columnas)
outputs/mission/vis/<src>_evid.jpg frame anotado (overlay + cajas + HUD)
outputs/mission/{high_res,full_res,thumb}/<src>.*
outputs/corridor_map.jpg           corredor apilado (lo usa el Informe)
entrega/summary.json               contrato de post-vuelo
entrega/ens_seg/<src>_b5.png       overlays ensemble (opcional)
entrega/enhanced/<src>_edsr.jpg    frames EDSR (opcional)
```

Columnas: `t_s, alt_m, p_hpa, temp_c, veg, bui, wat, bare, oth, dom, usi, ndvi,
verdict, people, vehicles, danado_pct, aff_m2, diag, alert, sharp, src,
sample_pri, sample_score`.

Degradación elegante: si un archivo no existe, su tab/galería/sección se oculta
sin error; sin `telemetry.csv` la app queda en estados vacíos con el dot ámbar.

---

## 4. API (web_server.py, contrato v3 con consulta terrestre)

| Endpoint | Descripción |
|---|---|
| `GET /api/mission` | CSV parseado + `summary.json` + mapa de archivos por frame |
| `GET /api/frame/<src>` | Una fila + rutas de imágenes disponibles |
| `GET /api/samples` | Campos por frame del JSONL que **no** están en el CSV (incertidumbre, `ms_seg/ms_dmg/ms_total`, `nodata_pct`) + agregados del panel de Muestreo |
| `GET /api/consulta?q=…&poly=…` | **Consulta Terrestre**: consultas simbólicas sobre las máscaras de clase (área, conteo con buffer, fracción de longitud, distancia, personas). `poly` es un polígono `[[lon,lat],…]` opcional |
| `GET /api/events` | **SSE**: push al instante cuando cambian `telemetry.csv` o `summary.json` (el polling de 3 s queda de red) |
| `GET /api/summary` · `GET /api/health` | contrato crudo · estado del servidor |
| `GET /img/<relpath>` | Proxy de imágenes sin caché (vuelos live) |
| `GET /` | `web-app/dist/index.html` si existe, si no `web/index.html` |

Notas: caché interna invalidada por `mtime` (el polling de 3 s no re-escanea el
árbol); `Cache-Control: no-store` en `/api/*` e `/img/*`; anti path-traversal en
ambas raíces estáticas; `--root` permite servir otra copia del proyecto.

La consulta no corre en este proceso: el servidor invoca por subproceso
`tools/consulta.py` del repo de vuelo (así la estación sigue siendo stdlib-only)
con el venv de `../cansat_seg_poc` o el intérprete actual. `--flight-root` y
`--consulta-python` permiten apuntar a otra copia. Si no hay
`entrega/masks/*.png` (post-vuelo sin `--no-masks`), las consultas que dependen
de máscaras responden "consulta no soportada" con sugerencias; el resto de la
estación funciona igual.

---

## 5. Auto-refresh sin parpadeo

`src/hooks/useFeed.ts` hace polling de 3 s; `applyPayload()` firma el payload
(hash djb2) y **si no cambió no toca el estado**: React no re-renderiza, no hay
re-animaciones ni parpadeo. Los frames nuevos entran por diff (slide-in en
alertas, layout en corredor y tabla, toast informativo).

---

## 6. Pruebas

```bash
# demo sintética coherente (12 frames) si no hay vuelo real
python tools/make_demo_mission.py            # --sin-postvuelo / --sin-corredor / --clean

# smoke E2E del frontend v4 (build production servido por Python)
npm i playwright && npx playwright install --with-deps chromium   # una vez
node tools/smoke-v4.mjs

# smoke E2E del frontend legacy (web/)
node tools/smoke.mjs
```

`smoke-v4.mjs` verifica (**54 aserciones**, cero errores de consola): tour una sola
vez y reabrible, resumen narrativo, mismo detalle desde corredor/tabla/alertas/
curva/galería, tabs de imagen degradadas, teclado (`↑↓ A / Enter Esc 1-2-3`),
export CSV del filtro, vistas Post-vuelo e Informe con firmas, modo presentación
con avance automático, auto-refresh sin recarga sobre un segundo servidor
(`--root` en `/tmp`) y caída/recuperación sin CSV. Capturas en
`os.tmpdir()/lb135-shots/` (el smoke las escribe ahí; `tools/shots/` guarda las
capturas históricas de las rondas de upgrade).

---

## 7. Resolución de problemas

| Síntoma | Causa / solución |
|---|---|
| Python sirve la web vanilla vieja | Falta `web-app/dist/`: `cd web-app && npm run build` |
| `npm run dev` sin datos | Levantá `web_server.py` antes: el proxy apunta a `:8000` |
| Tour no aparece | Ya está marcado en `localStorage`; botón “?” lo reabre |
| Puerto ocupado | `python web_server.py --port 8080` (y ajustá el proxy en `vite.config.ts`) |

Navegadores objetivo: Chrome y Firefox, 1366×768 y 1920×1080 (probado con
Chromium headless en ambos tamaños).

---

## 8. Upgrade UI/UX cinematográfico (v4.1)

Capa de motion/3D encima del contrato de datos existente (nada del pipeline
ni de la API cambió). Resumen de lo que se agregó y dónde vive:

| Pieza | Dónde | Detalle |
|---|---|---|
| **Cutscene de arranque** | `src/components/Cutscene.tsx` | Boot cinematográfico con GSAP + TextPlugin: log de enlace tipeado, logo con reveal (blur+spring), anillo expansivo, flash y cortinas que abren al dashboard. Una vez por sesión (`sessionStorage['lb135-cutscene-v1']`); se salta con click/Enter/Esc (fast-forward ×5); el botón  del TopBar la repite (`lb135-cutscene-replay`). El tour de driver.js espera a que termine (`lb135-cutscene-done`). |
| **Escena 3D del descenso** | `src/components/scene/DescentScene.tsx` | three.js + @react-three/fiber: CanSat con paracaídas sobre terreno low-poly wireframe, estrellas, niebla, haz del corredor, sombra proyectada y parallax de cámara. La altitud sigue al frame seleccionado; botón **▶ Descenso** reproduce la misión completa (selecciona frames en orden). Chunk async (code-splitting), `frameloop='never'` fuera de viewport y durante Presentar, y fallback SVG si no hay WebGL. |
| **Fondo ambiental** | `src/components/Backdrop.tsx` + `index.css` | Aurora verde/azul en deriva lenta, grilla HUD con máscara radial, scanline que barre, ruido SVG y viñeta. Fijo detrás de todo, `pointer-events: none`. |
| **Sistema de paneles** | `index.css` (`[data-slot="card"]`) | Borde en degradado verde→azul (mask-composite), profundidad interna, barrido de luz al hover y halo suave. Extras: `.corners` (miras de esquina), `.tilt-glare` + `hooks/useTilt.ts` (inclinación con puntero y glare que sigue al cursor), `.kpi-num` (glow de cifras). |
| **Títulos que se decodifican** | `src/components/Scramble.tsx` | Efecto scramble/decode al entrar en viewport (paneles de Vuelo, Post-vuelo y galerías). |
| **Gráfico de daño** | `DamageChart.tsx` | ComposedChart con área en degradado + trazo verde→azul con glow (`drop-shadow`). |
| **Presentación cine** | `Presentation.tsx` | Letterbox animado, Ken Burns lento sobre el frame, viñeta, scanlines, barra de progreso de 4 s por slide y dots clickeables. |
| **Dock de atajos** | `ShortcutsFooter.tsx` | Pill flotante glass fija abajo (antes era un footer estático). |
| **Tipografía** | `index.css` | JetBrains Mono Variable empaquetada para datos/consola (Inter sigue para UI). |
| **Filtro de consola** | `src/lib/consoleFilter.ts` | Silencia sólo el deprecation interno `THREE.Clock` (lo emite three.js vía R3F) y mensajes de performance del driver GL headless. Todo lo demás sigue visible. |

### Rendimiento y accesibilidad
- three.js vive en un chunk async: el dashboard pinta antes que el 3D
  (`React.lazy` + fallback "cargando motor 3d…"). La escena se monta **después**
  de la cutscene para no competir por CPU durante el boot.
- `prefers-reduced-motion`: sin cutscene, sin aurora/scanline/estrellas,
  cámara 3D quieta y sin barridos de luz.
- Impresión: el informe sigue saliendo solo en tinta; los reveals on-screen
  se neutralizan con `#informe-print [style] { opacity/transform }` en
  `@media print`.

### Pruebas
`tools/smoke-v4.mjs` sigue siendo la regresión completa (41 aserciones,
TODO VERDE). Se adaptó mínimamente al upgrade:
- `skipCutscene(page)`: saltea la cutscene con Esc y espera su desmontaje.
- Allowlist `GL_NOISE` para warnings del driver SwiftShader (sólo headless).
- Waits un poco mayores en dos pasos sensibles a CPU headless.

Capturas de verificación del upgrade: `tools/shots/up/`
(`node tools/shots-up.mjs` y `node tools/print-check.mjs` las regeneran).

### 8.1 Ronda 2 (feedback del operador)

| Issue | Fix |
|---|---|
| El tour de ayuda salía **vacío** | `lib/tour.ts`: driver.js v1 espera `popover: { title, description }` por step; se mapean los textos al formato correcto. |
| Fotos **pixeladas** en Presentar | La presentación usa la mejor fuente disponible (`enhanced` EDSR 1280×960 → `full_res` → `high_res`) con **PiP** de la evidencia anotada YOLO y toggle `Ver anotada / Ver nítida`. En Detail, chip "✦ hay versión nítida (EDSR) · ver" cuando la vista es `vis` 640×480. |
| El modelo 3D: cuerdas sueltas fuera de la tela | `scene/DescentScene.tsx` reescrito: paracaídas de **domo** con gores verdes, anillo de borde y ventral; cuerdas ancladas al borde exacto de la tela; cuerpo con rieles, banda emisiva, **baliza que titila**, antena con punta emisiva, aletas y **gimbal de cámara**; cámara más cerca. |
| ▶ Descenso movía la pantalla y tapaba el 3D | `lib/fly.ts`: durante el flythrough se silencia el auto-scroll del corredor y al iniciar se lleva la vista al panel 3D (`scrollIntoView` del propio panel). |
| Panel Detail con scroll interno minúsculo | Detail crece con la página: sin `overflow-y-auto` anidado, imagen hasta 64 vh y métricas completas a scroll de página. |
| "Que el 3D sea **en vivo** con GPS/IMU" | Botón **Live** en la escena: el CanSat sigue el **último frame que arriva del pipeline** (el server polea `telemetry.csv` cada 3 s; durante un vuelo real el pipeline escribe filas nuevas y el descenso se anima solo), con chip "● EN VIVO" y flash del HUD en cada frame nuevo. |

Capturas de la ronda 2: `tools/shots/up2/` (`node tools/shots-up2.mjs`).

### 8.2 Ronda 3 (sobre el rediseño editorial de la otra IA)

Trabajado directamente sobre el zip que subió el operador (secciones 01–04,
hero "Descenso recuperado.", subtítulo del TopBar). Fixes:

| Issue | Fix |
|---|---|
| Panel de evidencia enorme con aire alrededor | `Detail` re-compuesto: imagen a la izquierda (EDSR por defecto, sin caja negra full-width) y métricas 2×2 a la derecha en `xl`; sin scroll anidado y sin stretch que dejara hueco bajo la card. Explorador reorganizado: corredor+alertas en una fila, detalle full-width debajo. |
| Imagen pixelada | `store.detailImg` default `'enhanced'` (EDSR 1280×960); degrada a `vis` si no existe. Chip "✦ hay versión nítida" sigue apareciendo si elegís Evidencia a mano. |
| Esquinas con L muy pegadas | `.corners` con inset 12 px y opacidad .5; fuera de las KPI cards (solo paneles grandes). |
| KPI huérfano en 2 columnas | Último KPI (`daño promedio`) con `xl:col-span-2`. |
| Glare/glow apagados por el rediseño | Re-habilitados en `.mission-page` (glare de tilt y text-shadow de KPIs). |
| Párrafo duplicado del briefing | `Narrative`: el 2.º párrafo ya no repite la frase del momento crítico. |
| Huecos bajo corredor/alertas | Filas del explorador con alturas propias; cards estiran solo donde aporta. |

Smoke: waits fijos → pollers (`waitRows`, `waitDetailSrc`) porque las filas y el
swap de imagen ahora animan entrada/salida; click del punto de la curva vía
`dispatchEvent` (el anillo pulsante vuelve inestable el hit-test). 35 aserciones,
TODO VERDE. Capturas: `tools/shots/up/` y `tools/shots/up2/`.

### 8.3 Semana 1 · Validación y simulacro (roadmap ronda 4)

- **`tools/simulacro.py`**: ensayo end-to-end sin hardware. Copia imágenes y
  contrato a una carpeta viva y hace stream de `telemetry.csv` a cadencia real
  o acelerada (`--velocidad`), con anomalías: `--dropout` (corte TLM),
  `--rafaga N` (alertas inyectadas), `--trunco-en FILA` (caída y recuperación
  del CSV) y `--sin-postvuelo`. Guion operativo completo en
  **`CHECKLIST-SIMULACRO.md`** (12 verificaciones + variantes obligatorias).
- **Validación del contrato** (`src/lib/validate.ts`): chequea monotonicidad de
  `t_s`, rangos de `alt_m`/`danado_pct`, src duplicados y coherencia con
  `summary.n_frames`. Badge `datos: ok / N avisos` en el TopBar con tooltip.
  No bloquea: avisa. (En el simulacro con ráfaga marca 2 avisos = correcto.)
- **Assets cinemáticos locales** (generados en el sandbox, sin servicios
  externos): `public/assets/earth_limb.jpg` (fondo de la cutscene con máscara)
  y `public/assets/keyart_mision.jpg` (banda del Informe en pantalla,
  `no-print` para que el papel siga en tinta).

### 8.4 Semanas 2–3 · Ops, jurado y remates (roadmap ronda 4)

**Ops (Sprint 1)**
- **Máquina de estados** (`lib/missionState.ts`): sin-telemetría → ascenso →
  descenso → aterrizado → post-vuelo, inferida de altitud/tendencia/summary.
  Chip de fase en el TopBar y **paracaídas consciente de la fase** (plegado en
  ascenso, apoyado/rotado al aterrizar).
- **Tape de altitud + variómetro** en la escena: altitud live, m/s con flecha,
  marca de apogeo y alerta "tasa anómala" si el descenso se sale de rango.
- **Scrubber de replay** (`Scrubber.tsx`): línea de tiempo arrastrable +
  reproducción 0.5×/1×/2×/4× que sincroniza 3D, detalle, curva y corredor
  (usa `lib/fly` para no robar el scroll).
- **Bitácora de eventos** (`Bitacora.tsx`): registro automático (fases, apogeo,
  alertas, aterrizaje, enlace, post) con export JSON. Se llena desde el diff
  del store en `applyPayload`.
- **Modo sol de campo**: toggle en TopBar, persistido; tema claro de alto
  contraste, escena 3D adaptada (sin estrellas, terreno claro) y overrides de
  tintas hardcodeadas.

**Jurado (Sprint 2)**
- **Modo jurado manual** (`Jurado.tsx`): deck de 6 slides (portada con keyart,
  briefing, descenso 3D embebido, momentos críticos, veredictos, cierre),
  ←→ / dots / auto 12 s / Esc. Sin narrativa automática compleja.
- **Póster PNG 1920×1080** (`lib/poster.ts`): canvas offscreen con KPIs, hero
  del momento crítico y tira de frames; botón en Post-vuelo.
- **Perfil alt(t)/v(t)** (`AltProfile.tsx`) con marca de apogeo, gemelo de la
  curva de daño en la sección 02.

**Remates**
- **Push SSE** `GET /api/events` en `web_server.py` (ThreadingHTTPServer, poll
  de mtime por cliente); el frontend sincroniza al instante con `EventSource`
  y el polling de 3 s queda de red.
- **Toasts** de feedback (CSV, póster, frames nuevos en vivo, jurado).
- Post-vuelo e Informe con **code-splitting** (lazy).
- A11y: anillo `:focus-visible`, `aria-live` en alertas, contraste de ticks.
- Smoke: flag `--js-flags=--jitless` para sandboxes con poca RAM (V8 no puede
  reservar CodeRange); capturas de esta ronda en `tools/shots/r4/`.
