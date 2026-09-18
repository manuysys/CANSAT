# Estación Terrena LB135

Frontend activo de la estación terrena CanSat. Es una experiencia de briefing
de misión que combina narrativa, telemetría, evidencia visual y un informe
imprimible sin cambiar el contrato de datos del pipeline.

## Desarrollo

Desde la raíz del proyecto, levanta primero el servidor de datos:

```bash
python web_server.py
```

En otra terminal:

```bash
cd web-app
npm install
npm run dev
```

La app de Vite queda en `http://localhost:5173` y redirige `/api` e `/img` al
servidor Python en el puerto 8000.

## Producción local

```bash
cd web-app
npm run build
cd ..
python web_server.py
```

Python sirve `web-app/dist/` automáticamente cuando existe. Esto permite usar la
estación en el PC del laboratorio sin Node en tiempo de ejecución.

## Comandos

```bash
npm run lint
npm run build
```

El smoke E2E vive en `tools/smoke-v4.mjs` y se ejecuta con el servidor Python
activo desde la raíz:

```bash
node tools/smoke-v4.mjs
```

La prueba cubre tour, selección de frames, filtros, exportación, post-vuelo,
informe, presentación, auto-refresh, caída del CSV y layout móvil de 375 px.

## Estructura

- `src/App.tsx`: composición de las vistas Vuelo, Post-vuelo e Informe.
- `src/components/`: hero de misión, 3D, gráficos, evidencia y controles.
- `src/store/mission.ts`: estado y selectores de la misión.
- `src/hooks/useFeed.ts`: polling, sincronización manual y recuperación offline.
- `src/lib/types.ts`: tipos del contrato consumido por el frontend.
- `../web_server.py`: API y servidor de estáticos sin dependencias externas.
