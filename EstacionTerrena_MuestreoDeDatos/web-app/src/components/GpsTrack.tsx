/* Trayectoria GPS del descenso (DPD: "asociar posición a cada imagen").
 *
 * Sin basemap (la estación corre offline en la PC del laboratorio): la traza
 * se dibuja en un scatter con coordenadas LOCALES EN METROS para que la forma
 * no salga estirada (antes se graficaba lat/lon con escalas independientes y
 * una traza de ~200 m se veía deformada). El dominio es cuadrado (mismo rango
 * en x e y), así que el norte siempre queda arriba y el este a la derecha.
 * Altitud en el tamaño del punto y tooltip con src/alt/lat/lon. Click en un
 * punto → selecciona ese frame. */
import { useMemo } from 'react'
import { motion } from 'motion/react'
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart,
         Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { MapPin } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { EmptyState } from '@/components/EmptyState'
import { Scramble } from '@/components/Scramble'
import { useMission } from '@/store/mission'
import { gpsDe } from '@/components/Sampling'
import { num } from '@/lib/format'

const M_POR_GRADO_LAT = 111320

interface PuntoLocal {
  src: string
  lat: number
  lon: number
  alt: number
  alert: boolean
  dens: number | null   // U4: hab/km² (WorldPop por GPS o supuesto)
  fuente: string | null
  x: number          // este local (m)
  y: number          // norte local (m)
}

function TooltipPunto({ active, payload }: {
  active?: boolean
  payload?: Array<{ payload: PuntoLocal }>
}) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="rounded-lg border border-[#263241] bg-[#121a24] px-2.5 py-1.5 font-mono text-[10.5px] leading-relaxed shadow-lg">
      <div className="text-[#dbe4ee]">{p.src}</div>
      <div className="text-[#8b9aab]">alt {num(p.alt, 1)} m</div>
      <div className="text-[#8b9aab]">lat {p.lat.toFixed(5)} · lon {p.lon.toFixed(5)}</div>
      {p.dens != null && (
        <div className="text-[#ffb020]">
          población {num(p.dens, 0)} hab/km²
          {p.fuente ? ` · ${p.fuente}` : ''}
        </div>
      )}
    </div>
  )
}

/* U4: umbral de densidad "alta" (hab/km²). WorldPop da valores continuos;
 * 1000 separa rural de urbano-periurbano denso (declarado, no normativo). */
const DENS_ALTA = 1000

export function GpsTrack() {
  const frames = useMission(s => s.frames)
  const samples = useMission(s => s.samples)
  const select = useMission(s => s.select)

  const puntos = useMemo(() => gpsDe(frames).map(p => ({
    ...p,
    dens: (samples[p.src]?.pop_density as number | undefined) ?? null,
    fuente: (samples[p.src]?.pop_fuente as string | undefined) ?? null,
  })), [frames, samples])
  const conAlerta = puntos.filter(p => p.alert)
  const densos = puntos.filter(p => !p.alert && (p.dens ?? 0) >= DENS_ALTA)
  const normales = puntos.filter(p => !p.alert && (p.dens ?? 0) < DENS_ALTA)

  /* Proyección local: metros al este/norte respecto del centroide. */
  const geo = useMemo(() => {
    if (puntos.length < 2) return null
    const lat0 = puntos.reduce((a, p) => a + p.lat, 0) / puntos.length
    const lon0 = puntos.reduce((a, p) => a + p.lon, 0) / puntos.length
    const mPorLon = M_POR_GRADO_LAT * Math.cos((lat0 * Math.PI) / 180)
    const local: PuntoLocal[] = puntos.map(p => ({
      ...p,
      x: (p.lon - lon0) * mPorLon,
      y: (p.lat - lat0) * M_POR_GRADO_LAT,
    }))
    const xs = local.map(p => p.x)
    const ys = local.map(p => p.y)
    const xMin = Math.min(...xs), xMax = Math.max(...xs)
    const yMin = Math.min(...ys), yMax = Math.max(...ys)
    const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2
    // Dominio CUADRADO: mismo rango en ambos ejes → sin deformación.
    const half = Math.max(xMax - xMin, yMax - yMin) / 2 * 1.25 + 5
    return {
      local, lat0, lon0, mPorLon, cx, cy, half,
      dominioX: [cx - half, cx + half] as [number, number],
      dominioY: [cy - half, cy + half] as [number, number],
      // Ticks en grados (mismos valores que antes) aunque la escala sea métrica.
      tickLon: (x: number) => (lon0 + x / mPorLon).toFixed(4),
      tickLat: (y: number) => (lat0 + y / M_POR_GRADO_LAT).toFixed(4),
    }
  }, [puntos])

  if (!geo) {
    return (
      <EmptyState
        icon={MapPin}
        title="Sin trayectoria GPS."
        hint="El pipeline asocia la posición que recibe por UART (--uart-state) a cada frame. En la demo y en el simulacro también se genera una trayectoria sintética."
        className="mx-auto max-w-xl py-10"
      />
    )
  }

  const click = (data: unknown) => {
    const p = data as { src?: string } | undefined
    if (p?.src) select(p.src)
  }

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
            {puntos.length} posiciones · {conAlerta.length} con alerta
          </span>
        </div>
        <div className="h-[240px] w-full p-3">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="#263241" strokeDasharray="3 3" />
              <XAxis
                type="number" dataKey="x" name="Este"
                domain={geo.dominioX} reversed={false}
                tick={{ fill: '#8b9aab', fontSize: 10 }}
                tickFormatter={geo.tickLon}
                stroke="#263241"
              />
              <YAxis
                type="number" dataKey="y" name="Norte"
                domain={geo.dominioY}
                tick={{ fill: '#8b9aab', fontSize: 10 }}
                tickFormatter={geo.tickLat}
                stroke="#263241" width={54}
              />
              <ZAxis type="number" dataKey="alt" range={[40, 140]} name="Altitud (m)" />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={<TooltipPunto />}
              />
              <Scatter name="frames" data={normales} fill="#3ddc84" onClick={click} cursor="pointer" />
              <Scatter name="densidad alta" data={densos} fill="#ffb020" onClick={click} cursor="pointer" />
              <Scatter name="alertas" data={conAlerta} fill="#ff4d5e" onClick={click} cursor="pointer" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="border-t border-border/50 px-5 py-2 text-[10.5px] text-muted-foreground/70">
          Norte arriba · escala real · punto = altitud · naranja ≥ {DENS_ALTA} hab/km²
          (WorldPop) · rojo = alerta · click abre el frame · offline
        </p>
      </Card>
    </motion.div>
  )
}
