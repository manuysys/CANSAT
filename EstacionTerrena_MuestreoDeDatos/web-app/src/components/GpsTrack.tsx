/* Trayectoria GPS del descenso (DPD: "asociar posición a cada imagen").
 *
 * Sin basemap (la estación corre offline en la PC del laboratorio): se dibuja
 * lat/lon en un scatter con el eje X/Y invertidos para que "arriba" sea norte
 * y "derecha" sea este, con la altitud en el tooltip y las alertas en rojo.
 * Click en un punto → selecciona ese frame. */
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

export function GpsTrack() {
  const frames = useMission(s => s.frames)
  const select = useMission(s => s.select)

  const puntos = useMemo(() => gpsDe(frames), [frames])
  const conAlerta = puntos.filter(p => p.alert)
  const normales = puntos.filter(p => !p.alert)

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

  const latMin = Math.min(...puntos.map(p => p.lat))
  const latMax = Math.max(...puntos.map(p => p.lat))
  const lonMin = Math.min(...puntos.map(p => p.lon))
  const lonMax = Math.max(...puntos.map(p => p.lon))
  const pad = Math.max(1e-4, (latMax - latMin) * 0.15, (lonMax - lonMin) * 0.15)

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
                type="number" dataKey="lon" name="Longitud"
                domain={[lonMin - pad, lonMax + pad]} reversed={false}
                tick={{ fill: '#8b9aab', fontSize: 10 }}
                tickFormatter={(v: number) => v.toFixed(4)}
                stroke="#263241"
              />
              <YAxis
                type="number" dataKey="lat" name="Latitud"
                domain={[latMin - pad, latMax + pad]}
                tick={{ fill: '#8b9aab', fontSize: 10 }}
                tickFormatter={(v: number) => v.toFixed(4)}
                stroke="#263241" width={54}
              />
              <ZAxis type="number" dataKey="alt" range={[40, 140]} name="Altitud (m)" />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ background: '#121a24', border: '1px solid #263241', borderRadius: 10, fontSize: 11 }}
                labelStyle={{ color: '#dbe4ee' }} itemStyle={{ color: '#8b9aab' }}
                formatter={(value, name) => {
                  const v = Number(value)
                  return name === 'Altitud (m)'
                    ? [`${num(v, 1)} m`, String(name)]
                    : [v.toFixed(5), String(name)]
                }}
              />
              <Scatter name="frames" data={normales} fill="#3ddc84" onClick={click} cursor="pointer" />
              <Scatter name="alertas" data={conAlerta} fill="#ff4d5e" onClick={click} cursor="pointer" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="border-t border-border/50 px-5 py-2 text-[10.5px] text-muted-foreground/70">
          Arriba = norte · derecha = este · el tamaño del punto es la altitud ·
          click en un punto abre ese frame. Sin basemap: la estación funciona offline.
        </p>
      </Card>
    </motion.div>
  )
}
