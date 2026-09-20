/* Basemap OFFLINE de la trayectoria GPS (DPD: "asociar posición a cada imagen").
 *
 * La estación corre sin internet: en lugar de Leaflet + OSM online, se dibuja
 * un mapa Web Mercator con los tiles locales de ``public/tiles/`` (z13-18
 * alrededor del área de vuelo, ver tools/bajar_tiles.py). La trayectoria se
 * superpone como SVG en coordenadas de píxel de tile, con las alertas en rojo.
 *
 * Degradación elegante: si el tile central no existe (vuelo en otra zona, o
 * PC sin la carpeta de tiles), cae al scatter sin basemap (GpsTrack), que
 * funciona siempre. Click en un punto → selecciona ese frame. */
import { useEffect, useMemo, useState, type PointerEvent } from 'react'
import { motion } from 'motion/react'
import { MapPin } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { EmptyState } from '@/components/EmptyState'
import { GpsTrack } from '@/components/GpsTrack'
import { Scramble } from '@/components/Scramble'
import { useMission } from '@/store/mission'
import { gpsDe } from '@/components/Sampling'

const TILE = 256
const MAX_TILES = 64 /* 8x8: techo de tiles por vista */
const MARGEN_TILES = 2 /* tiles extra por lado para llenar el contenedor */
const Z_MAX = 18
const Z_MIN = 13

function lonApx(lon: number, z: number): number {
  return ((lon + 180) / 360) * (1 << z) * TILE
}

function latApx(lat: number, z: number): number {
  const r = (lat * Math.PI) / 180
  return ((1 - Math.asinh(Math.tan(r)) / Math.PI) / 2) * (1 << z) * TILE
}

export function MapOffline() {
  const frames = useMission(s => s.frames)
  const select = useMission(s => s.select)
  const selected = useMission(s => s.selected)
  const consulta = useMission(s => s.consulta)
  const zona = useMission(s => s.consultaZona)
  const dibujando = useMission(s => s.dibujandoZona)
  const setZona = useMission(s => s.setConsultaZona)
  const puntos = useMemo(() => gpsDe(frames), [frames])
  const [sinTiles, setSinTiles] = useState(false)

  /* Polígonos del resultado de consulta para el frame seleccionado. */
  const poligonosResultado = useMemo(() => {
    if (!consulta?.soportada || !selected) return []
    const fr = consulta.por_frame?.find(f => f.src === selected)
    return fr?.poligonos ?? []
  }, [consulta, selected])

  const geo = useMemo(() => {
    if (puntos.length < 2) return null
    const latMin = Math.min(...puntos.map(p => p.lat))
    const latMax = Math.max(...puntos.map(p => p.lat))
    const lonMin = Math.min(...puntos.map(p => p.lon))
    const lonMax = Math.max(...puntos.map(p => p.lon))
    const pad = Math.max(1e-4, (latMax - latMin) * 0.15, (lonMax - lonMin) * 0.15)
    const b = { latMin: latMin - pad, latMax: latMax + pad,
                lonMin: lonMin - pad, lonMax: lonMax + pad }
    let z = Z_MIN
    let grid = { x0: 0, x1: 0, y0: 0, y1: 0 }
    for (let zi = Z_MAX; zi >= Z_MIN; zi--) {
      const x0 = Math.floor(lonApx(b.lonMin, zi) / TILE) - MARGEN_TILES
      const x1 = Math.floor(lonApx(b.lonMax, zi) / TILE) + MARGEN_TILES
      const y0 = Math.floor(latApx(b.latMax, zi) / TILE) - MARGEN_TILES
      const y1 = Math.floor(latApx(b.latMin, zi) / TILE) + MARGEN_TILES
      if ((x1 - x0 + 1) * (y1 - y0 + 1) <= MAX_TILES) {
        z = zi
        grid = { x0, x1, y0, y1 }
        break
      }
    }
    const ox = grid.x0 * TILE
    const oy = grid.y0 * TILE
    const px = (lon: number) => lonApx(lon, z) - ox
    const py = (lat: number) => latApx(lat, z) - oy
    const w = (grid.x1 - grid.x0 + 1) * TILE
    const h = (grid.y1 - grid.y0 + 1) * TILE
    const track = puntos.map(p => ({ ...p, x: px(p.lon), y: py(p.lat) }))
    const cx = track.reduce((a, p) => a + p.x, 0) / track.length
    const cy = track.reduce((a, p) => a + p.y, 0) / track.length
    return { z, grid, w, h, track, cx, cy, px, py }
  }, [puntos])

  /* Chequeo del tile central: si no existe, no hay basemap para esta zona. */
  useEffect(() => {
    if (!geo) return
    const x = geo.grid.x0 + Math.floor((geo.grid.x1 - geo.grid.x0) / 2)
    const y = geo.grid.y0 + Math.floor((geo.grid.y1 - geo.grid.y0) / 2)
    const img = new Image()
    img.onload = () => setSinTiles(false)
    img.onerror = () => setSinTiles(true)
    img.src = `/tiles/${geo.z}/${x}/${y}.png`
  }, [geo])

  if (puntos.length < 2) {
    return (
      <EmptyState
        icon={MapPin}
        title="Sin trayectoria GPS."
        hint="El pipeline asocia la posición que recibe por UART (--uart-state) a cada frame. En la demo y en el simulacro también se genera una trayectoria sintética."
        className="mx-auto max-w-xl py-10"
      />
    )
  }

  if (sinTiles || !geo) return <GpsTrack />

  /* Clic en el mapa dibujando zona: píxel → lon/lat (inversa Web Mercator). */
  const agregarVertice = (e: PointerEvent<HTMLDivElement>) => {
    if (!dibujando) return
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left + geo.grid.x0 * TILE
    const y = e.clientY - rect.top + geo.grid.y0 * TILE
    const escala = (1 << geo.z) * TILE
    const lon = (x / escala) * 360 - 180
    const lat = (180 / Math.PI) * Math.atan(
      Math.sinh(Math.PI - (2 * Math.PI * y) / escala))
    setZona([...zona, [Number(lon.toFixed(7)), Number(lat.toFixed(7))]])
  }

  const conAlerta = geo.track.filter(p => p.alert).length
  const poly = geo.track.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <Card className="border-border/60 bg-card/60">
        <div className="flex items-center justify-between border-b border-border/50 px-5 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            <Scramble text="Trayectoria GPS del descenso" />
          </h2>
          <span className="font-mono text-[10.5px] text-muted-foreground/70">
            z{geo.z} · {puntos.length} posiciones · {conAlerta} con alerta
          </span>
        </div>
        <div className="relative h-[300px] w-full overflow-hidden bg-[#0b1017]">
          <div
            className="absolute"
            style={{ left: `calc(50% - ${geo.cx.toFixed(1)}px)`,
                     top: `calc(50% - ${geo.cy.toFixed(1)}px)`,
                     width: geo.w, height: geo.h,
                     cursor: dibujando ? 'crosshair' : undefined }}
            onPointerDown={agregarVertice}
          >
            {Array.from({ length: geo.grid.x1 - geo.grid.x0 + 1 }, (_, ix) =>
              Array.from({ length: geo.grid.y1 - geo.grid.y0 + 1 }, (_, iy) => {
                const x = geo.grid.x0 + ix
                const y = geo.grid.y0 + iy
                return (
                  <img
                    key={`${x}-${y}`}
                    src={`/tiles/${geo.z}/${x}/${y}.png`}
                    alt=""
                    draggable={false}
                    onError={e => { (e.target as HTMLImageElement).style.visibility = 'hidden' }}
                    style={{ position: 'absolute', left: ix * TILE, top: iy * TILE,
                             width: TILE, height: TILE }}
                  />
                )
              }),
            )}
            <svg
              className="absolute inset-0"
              width={geo.w}
              height={geo.h}
              viewBox={`0 0 ${geo.w} ${geo.h}`}
            >
              <polyline
                points={poly}
                fill="none"
                stroke="#0b1017"
                strokeWidth={5}
                strokeOpacity={0.55}
                strokeLinejoin="round"
              />
              <polyline
                points={poly}
                fill="none"
                stroke="#3ddc84"
                strokeWidth={2}
                strokeLinejoin="round"
              />
              {geo.track.map(p => (
                <circle
                  key={p.src}
                  cx={p.x}
                  cy={p.y}
                  r={p.alert ? 5 : 3.5}
                  fill={p.alert ? '#ff4d5e' : '#3ddc84'}
                  stroke="#0b1017"
                  strokeWidth={1}
                  style={{ cursor: 'pointer' }}
                  onClick={() => { if (!dibujando) select(p.src) }}
                >
                  <title>
                    {`${p.lat.toFixed(5)}, ${p.lon.toFixed(5)} · ${p.alt.toFixed(0)} m${p.alert ? ' · ALERTA' : ''}`}
                  </title>
                </circle>
              ))}
              {zona.length > 0 && (
                <polygon
                  points={zona.map(([lon, lat]) =>
                    `${geo.px(lon).toFixed(1)},${geo.py(lat).toFixed(1)}`).join(' ')}
                  fill="rgba(61,220,132,0.10)"
                  stroke="#3ddc84"
                  strokeWidth={1.5}
                  strokeDasharray="5 4"
                  pointerEvents="none"
                />
              )}
              {zona.map(([lon, lat], i) => (
                <circle key={`z${i}`} cx={geo.px(lon)} cy={geo.py(lat)} r={3}
                        fill="#3ddc84" stroke="#0b1017" strokeWidth={1}
                        pointerEvents="none" />
              ))}
              {poligonosResultado.map((poly, i) => (
                <polygon
                  key={`r${i}`}
                  points={poly.map(([lon, lat]) =>
                    `${geo.px(lon).toFixed(1)},${geo.py(lat).toFixed(1)}`).join(' ')}
                  fill="rgba(255,159,67,0.22)"
                  stroke="#ff9f43"
                  strokeWidth={1.2}
                  pointerEvents="none"
                />
              ))}
            </svg>
          </div>
        </div>
        <p className="border-t border-border/50 px-5 py-2 text-[10.5px] text-muted-foreground/70">
          Basemap offline (tiles locales z{Z_MIN}-{Z_MAX} · © OpenStreetMap) ·
          arriba = norte · el punto es la posición de cada frame, en rojo las
          alertas · click abre ese frame · con «Dibujar zona» el click agrega
          vértices y los polígonos ámbar son el resultado de la consulta del
          frame seleccionado.
        </p>
      </Card>
    </motion.div>
  )
}
