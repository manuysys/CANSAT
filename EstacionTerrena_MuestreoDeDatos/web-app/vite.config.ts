import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Estación Terrena LB135 — frontend Vite + React + TS.
// En desarrollo, el backend sigue siendo web_server.py (puerto 8000):
// le proxeamos la API y el proxy de imágenes para no tocar el contrato.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // changeOrigin:true reescribe el Host al del backend. Antes estaba en
      // false y funcionaba solo porque web_server.py no valida Host; detras
      // de un reverse proxy o con validacion de Host el modo dev se rompia.
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/img': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    // Nota: Vite hashea los assets por defecto (index-<hash>.js/css). El
    // comentario anterior decia "mejor sin hashes" pero no habia ninguna
    // configuracion que los quitara: la intencion no estaba implementada. Se
    // deja el hash (cache-busting correcto) y se limita el warning de tamano.
    chunkSizeWarningLimit: 900,
  },
})
