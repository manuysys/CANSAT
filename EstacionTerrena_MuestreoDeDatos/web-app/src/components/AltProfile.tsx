/* Perfil físico del descenso: altitud(t) y velocidad vertical(t) derivada,
   con marca de apogeo. Gemelo científico de la curva de daño. */
import { useMemo } from 'react'
import {
  CartesianGrid, ComposedChart, Line, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Card } from '@/components/ui/card'
import { Scramble } from '@/components/Scramble'
import { apogeo } from '@/lib/missionState'
import { num } from '@/lib/format'
import { useMission } from '@/store/mission'

function Tip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { t: number; alt: number; v: number; src: string } }> }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="rounded-lg border border-border/70 bg-popover/95 px-3 py-2 font-mono text-[11px] leading-relaxed text-popover-foreground shadow-lg">
      <div className="font-semibold">{d.src}</div>
      <div className="text-muted-foreground">t {num(d.t, 1)} s · alt {num(d.alt, 1)} m</div>
      <div className="text-muted-foreground">v vertical {num(d.v, 1)} m/s</div>
    </div>
  )
}

export function AltProfile() {
  const frames = useMission(s => s.frames)

  const data = useMemo(() => frames.map((f, i) => {
    const t = Number(f.t_s) || 0
    const alt = Number(f.alt_m) || 0
    let v = 0
    if (i > 0) {
      const dt = t - (Number(frames[i - 1].t_s) || 0)
      if (dt > 0) v = (alt - (Number(frames[i - 1].alt_m) || 0)) / dt
    }
    return { t, alt, v, src: f.src }
  }), [frames])

  const apo = apogeo(frames)

  return (
    <Card className="border-border/60 bg-card/60">
      <div className="flex items-center justify-between gap-4 border-b border-border/50 px-5 py-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          <Scramble text="Altitud y velocidad vertical" />
        </h2>
        <span className="font-mono text-[10.5px] text-muted-foreground/70">
          {apo ? `apogeo ${num(apo.alt, 0)} m · ${apo.src}` : 'física del descenso'}
        </span>
      </div>
      <div className="h-[148px] px-2 py-2">
        {data.length >= 2 ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 8, right: 6, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#161e29" vertical={false} />
              <XAxis
                dataKey="t" type="number" domain={['dataMin', 'dataMax']}
                tickFormatter={v => `${Math.round(Number(v))} s`}
                tick={{ fill: '#7d8c9d', fontSize: 10 }} axisLine={{ stroke: '#1c2530' }} tickLine={false}
              />
              <YAxis
                yAxisId="alt" domain={[0, 'dataMax + 20']} width={44}
                tickFormatter={v => `${Math.round(Number(v))} m`}
                tick={{ fill: '#7d8c9d', fontSize: 10 }} axisLine={false} tickLine={false}
              />
              <YAxis
                yAxisId="v" orientation="right" width={40}
                tickFormatter={v => `${Math.round(Number(v))}`}
                tick={{ fill: '#7d8c9d', fontSize: 10 }} axisLine={false} tickLine={false}
              />
              <Tooltip content={<Tip />} cursor={{ stroke: '#263241', strokeDasharray: '3 3' }} />
              <Line yAxisId="alt" type="monotone" dataKey="alt" stroke="#3f8fd1" strokeWidth={1.8} dot={false} isAnimationActive animationDuration={700} />
              <Line yAxisId="v" type="monotone" dataKey="v" stroke="#ffb020" strokeWidth={1.3} strokeDasharray="5 4" dot={false} isAnimationActive animationDuration={700} />
              {apo && (
                <ReferenceDot yAxisId="alt" x={Number(frames[apo.idx].t_s) || 0} y={apo.alt} r={4} fill="#7ee8b0" stroke="#0b0e14" strokeWidth={1.5} />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            El perfil físico aparece con el segundo frame de la misión.
          </div>
        )}
      </div>
    </Card>
  )
}
