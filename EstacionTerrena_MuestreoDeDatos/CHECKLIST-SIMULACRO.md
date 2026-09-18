# ✅ Checklist de simulacro end-to-end (sin hardware)

Ensayo operativo completo del sistema: pipeline → `telemetry.csv` →
`web_server.py` → dashboard. Objetivo: que el día D nadie improvise.
Frecuencia sugerida: **una vez por semana hasta el vuelo, y después de cada
cambio grande del frontend**.

## Preparación

```bash
cd web-app && npm run build && cd ..          # solo si cambió el frontend
# Terminal 1: misión "viva" acelerada 4× con anomalías
python tools/simulacro.py --dst /tmp/sim-live --velocidad 4 --dropout 6 --rafaga 2 --trunco-en 9
# Terminal 2: servidor apuntando a la carpeta viva
cp -r web-app/dist /tmp/sim-live/web-app/   # el server con --root sirve estáticos de la copia
python web_server.py --root /tmp/sim-live --port 8004
```

Abrir `http://localhost:8004` en el navegador de ops (y, si existe, en la
tablet de campo). Activar **Live** en la escena 3D. Roles: *Operador* maneja
la consola, *Observador* tilda esta lista y anota tiempos.

## Guion y verificaciones

| # | Momento | Esperado | OK |
|---|---|---|---|
| 1 | T+0 stream | Primer frame pintado < 4 s; dot verde "sync"; badge `datos: ok` en el TopBar | ☐ |
| 2 | Ascenso | Regla de altitud sube; CanSat 3D gana altura; HUD muestra el último src | ☐ |
| 3 | Apogeo (log del simulacro) | La altitud deja de crecer; curva de daño empieza a moverse | ☐ |
| 4 | Primer `alert=1` | Panel Alertas suma card con pulso rojo + ping en la curva; toast/entrada en bitácora¹ | ☐ |
| 5 | Ráfaga inyectada | 2 alertas nuevas entran sin parpadeo del resto (diff, no re-render global) | ☐ |
| 6 | Dropout TLM (6 s) | Dot pasa a ámbar/"sin sync"; al volver, retoma sin duplicar frames | ☐ |
| 7 | Trunco (fila 9) | Corredor queda vacío con empty state amable; **cero errores de consola**; al restaurar, vuelven los frames | ☐ |
| 8 | Aterrizaje | Altitud → ~1 m; al publicarse `summary.json`, Post-vuelo se llena solo | ☐ |
| 9 | Replay | Scrubber² o ▶ Descenso recorre la misión sin robar el scroll | ☐ |
| 10 | Informe | Vista Informe refleja lo simulado; Imprimir/PDF sale en tinta | ☐ |
| 11 | Caída del server | Matar Terminal 2: banner offline; revivir: recupera sin recargar | ☐ |
| 12 | Cierre | Observador anota tiempos reales vs esperados y abre issue si hay Δ | ☐ |

¹ La bitácora de eventos (`Bitacora.tsx`) y ² el scrubber (`Scrubber.tsx`) YA
están implementados y montados en `App.tsx`: no marcar N/A. El chip de fase de
misión (`missionState.ts` + `PhaseChip` en el TopBar) también está activo y es
lo que verifican las filas 2, 3 y 8. El badge `datos: ok` es la validación de
contrato (`lib/validate.ts`).

> ⚠ Nota de entorno: los comandos de arriba son POSIX (`/tmp`, `cp -r`). En
> Windows usar `C:\Windows\TEMP\...` y `Copy-Item -Recurse`. `--disable-gpu`
> es innecesario: si no hay WebGL la escena degrada sola (`lib/webgl.ts`).

## Variantes obligatorias del guion

- **Sin post-vuelo**: `--sin-postvuelo` → Post-vuelo debe mostrar empty state, no rotura.
- **Sin WebGL** (PC vieja del lab): abrir con `--disable-gpu` en Chrome → la escena degrada a perfil SVG.
- **Resolución chica**: 1366×768 y zoom 100 % → dock, explorador y detalle siguen legibles.
- **Misión sintética nueva**: `python tools/make_demo_mission.py --clean` y repetir (cambia el seed de la historia).

## Registro de corridas

| Fecha | Operador | Variantes | Resultado | Notas |
|---|---|---|---|---|
|  |  |  |  |  |
