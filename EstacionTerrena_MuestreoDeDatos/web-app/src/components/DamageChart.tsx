/* La historia de la misión en una curva: daño % vs altitud (X invertido 250→0).
   Recharts LineChart con puntos clickeables propios; los puntos con alert=1
   llevan anillo pulsante en rojo semántico. */
import { Card } from '@/components/ui/card'
import { Scramble } from '@/components/Scramble'
import { Play } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { useMission } from '@/store/mission'
import { intOr, num } from '@/lib/format'
import { isAlertFrame } from '@/lib/vocab'
import type { Frame } from '@/lib/types'

interface DotProps {
  cx?: number
  cy?: number
  payload?: Frame & { dano: number; alt: number }
  index?: number
}

/* Punto clickeable: selecciona su frame (el mismo detalle que corredor/tabla). */
function ClickDot({ cx = 0, cy = 0, payload }: DotProps) {
  const select = useMission(s => s.select)
  const selected = useMission(s => s.selected)
  if (!payload) return null
  const isAlert = isAlertFrame(payload)
  const isSel = payload.src === selected
  const activate = () => select(payload.src)
  return (
    <g
      className="cursor-pointer outline-none"
      data-dot-src={payload.src}
      onClick={activate}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate() } }}
      tabIndex={0}
      role="button"
      aria-label={`${payload.src}, ${num(payload.alt_m, 0)} metros, daño ${num(payload.danado_pct, 1)} por ciento`}
    >
      <title>{`${payload.src} · ${num(payload.alt_m, 0)} m · daño ${num(payload.danado_pct, 1)}%`}</title>
      {isAlert && (
        <circle cx={cx} cy={cy} r={7} fill="none" stroke="#ff4d5e" strokeWidth={1.4} className="alert-ping" />
      )}
      <circle
        cx={cx} cy={cy} r={isAlert ? 3.8 : 3}
        fill={isAlert ? '#ff4d5e' : '#0b0e14'}
        stroke={isSel ? '#3ddc84' : isAlert ? '#0b0e14' : '#6b7a8c'}
        strokeWidth={isSel ? 2.2 : 1.4}
      />
      {/* área de click cómoda, invisible */}
      <circle cx={cx} cy={cy} r={9} fill="transparent" />
    </g>
  )
}

function ChartTip({ active, payload }: { active?: boolean; payload?: Array<{ payload: Frame & { dano: number } }> }) {
  if (!active || !payload?.length) return null
  const f = payload[0].payload
  return (
    <div className="rounded-lg border border-border/70 bg-popover/95 px-3 py-2 font-mono text-[11px] leading-relaxed text-popover-foreground shadow-lg">
      <div className="font-semibold">{f.src}</div>
      <div className="text-muted-foreground">{num(f.alt_m, 0)} m · daño {num(f.danado_pct, 1)}%</div>
      {intOr(f.alert) === 1 && <div className="text-[#ff8f9b]">alerta activa</div>}
    </div>
  )
}

export function DamageChart() {
  const frames = useMission(s => s.frames)
  const startPresent = useMission(s => s.startPresent)

  const data = frames.map(f => ({ ...f, dano: Number(f.danado_pct) || 0, alt: Number(f.alt_m) || 0 }))
  const altMax = Math.max(...data.map(d => d.alt), 1)

  return (
    <Card className="border-border/60 bg-card/60">
      <div className="chart-heading flex items-center justify-between gap-4 border-b border-border/50 px-5 py-3">
        <div>
          <div className="mission-kicker"><span className="size-1.5 rounded-full bg-[#3ddc84]" /> perfil de riesgo</div>
          <h2 className="mt-1 text-sm font-semibold tracking-tight text-foreground">
            <Scramble text="Daño durante el descenso" />
          </h2>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10.5px] text-muted-foreground/70">altitud {num(altMax, 0)} m → 0 m</span>
          <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" onClick={startPresent} title="Recorrer la misión en pantalla completa">
            <Play className="size-3" />
            Presentar
          </Button>
        </div>
      </div>
      <div className="h-[220px] px-2 py-3 md:h-[250px]">
        {data.length >= 2 ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 8, right: 18, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="dmgFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3ddc84" stopOpacity={0.32} />
                  <stop offset="60%" stopColor="#3ddc84" stopOpacity={0.07} />
                  <stop offset="100%" stopColor="#3ddc84" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="dmgStroke" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#7ee8b0" />
                  <stop offset="55%" stopColor="#3ddc84" />
                  <stop offset="100%" stopColor="#3f8fd1" />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#161e29" vertical={false} />
              <XAxis
                dataKey="alt" type="number" domain={[0, altMax]} reversed
                tickFormatter={v => `${Math.round(Number(v))} m`}
                tick={{ fill: '#5d6c7d', fontSize: 10 }} axisLine={{ stroke: '#1c2530' }} tickLine={false}
              />
              <YAxis
                domain={[0, 100]} width={40}
                tickFormatter={v => `${v}%`}
                tick={{ fill: '#5d6c7d', fontSize: 10 }} axisLine={false} tickLine={false}
              />
              <Tooltip content={<ChartTip />} cursor={{ stroke: '#263241', strokeDasharray: '3 3' }} />
              <Area
                type="monotone" dataKey="dano" stroke="none" fill="url(#dmgFill)"
                isAnimationActive animationDuration={900} animationEasing="ease-out"
              />
              <Line
                className="dmg-line"
                type="monotone" dataKey="dano" stroke="url(#dmgStroke)" strokeWidth={1.9}
                dot={(dotProps: unknown) => {
                  const pp = dotProps as DotProps
                  return <ClickDot key={pp.payload?.src ?? pp.index} cx={pp.cx} cy={pp.cy} payload={pp.payload} />
                }}
                activeDot={{ r: 4.5, fill: '#3ddc84', stroke: '#0b0e14' }}
                isAnimationActive animationDuration={700} animationEasing="ease-out"
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            La curva aparece con el segundo frame de la misión.
          </div>
        )}
      </div>
    </Card>
  )
}
