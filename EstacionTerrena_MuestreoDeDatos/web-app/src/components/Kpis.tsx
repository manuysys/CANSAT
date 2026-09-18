/* KPIs en Cards con count-up, entrada staggered (staggerChildren 0.07),
   tilt de puntero, glare y chip de icono con glow semántico. */
import type { ReactNode } from 'react'
import { motion } from 'motion/react'
import { Layers, TriangleAlert, Users, TrendingDown, Zap } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { CountUp } from '@/components/CountUp'
import { useTilt } from '@/hooks/useTilt'
import { useMission } from '@/store/mission'
import { intOr, mean, num } from '@/lib/format'
import { cn } from '@/lib/utils'
import { isAlertFrame } from '@/lib/vocab'

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
}
const item = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { type: 'spring' as const, stiffness: 300, damping: 26 } },
}

function KpiCard({ icon: Icon, label, val, tone, chip, span2 }: {
  icon: typeof Layers
  label: string
  val: ReactNode
  tone?: string
  chip?: string
  span2?: boolean
}) {
  const tilt = useTilt(5)
  return (
    <motion.div variants={item} className={cn('h-full', span2 && 'xl:col-span-2')}>
      <Card
        {...tilt}
        className="tilt-glare flex h-full flex-row items-center gap-3 border-border/60 bg-card/60 px-4 py-4 text-left will-change-transform"
      >
        <span className="glare" />
        <span
          className={cn(
            'grid size-9 shrink-0 place-items-center rounded-lg ring-1',
            chip || 'bg-[#3ddc84]/10 text-[#3ddc84] ring-[#3ddc84]/25',
          )}
        >
          <Icon className="size-4" strokeWidth={1.8} />
        </span>
        <span className="min-w-0">
          <span className={cn('block font-mono text-[21px] font-semibold leading-tight tracking-tight', tone || 'kpi-num')}>
            {val}
          </span>
          <span className="block truncate text-[11px] text-muted-foreground">{label}</span>
        </span>
      </Card>
    </motion.div>
  )
}

export function Kpis({ className }: { className?: string }) {
  const frames = useMission(s => s.frames)
  const summary = useMission(s => s.summary)

  const alerts = frames.filter(isAlertFrame).length
  const people = frames.reduce((a, f) => a + intOr(f.people), 0)
  const veh = frames.reduce((a, f) => a + intOr(f.vehicles), 0)
  const altMax = summary?.alt_max_m ?? (frames.length ? Math.max(...frames.map(f => Number(f.alt_m) || 0)) : null)
  const altMin = summary?.alt_min_m ?? (frames.length ? Math.min(...frames.map(f => (f.alt_m === null ? Infinity : Number(f.alt_m)))) : null)
  const dmg = summary?.danado_pct_prom ?? mean(frames.map(f => f.danado_pct))
  const has = frames.length > 0

  return (
    <motion.div variants={container} initial="hidden" animate="show" className={cn('grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5', className)}>
      <KpiCard icon={Layers} label="frames recibidos" val={<CountUp value={has ? frames.length : null} />} />
      <KpiCard
        icon={TriangleAlert} label="alertas"
        val={<CountUp value={has ? alerts : null} />}
        tone={alerts > 0 ? 'text-[#ff4d5e] kpi-num-bad' : 'kpi-num'}
        chip={alerts > 0 ? 'bg-[#ff4d5e]/10 text-[#ff4d5e] ring-[#ff4d5e]/30' : undefined}
      />
      <KpiCard icon={Users} label="personas · vehículos" val={<span className="tabular-nums">{has ? `${people} / ${veh}` : '—'}</span>} />
      <KpiCard
        icon={TrendingDown} label="altitud max → min"
        val={<span className="tabular-nums">{altMax === null || altMax === undefined ? '—' : `${num(altMax, 0)} → ${num(altMin === Infinity ? null : altMin, 0)} m`}</span>}
        chip="bg-[#3f8fd1]/10 text-[#7db8e8] ring-[#3f8fd1]/25"
      />
      <KpiCard
        span2
        icon={Zap} label="daño promedio"
        val={<CountUp value={dmg ?? null} decimals={1} suffix="%" />}
        chip="bg-[#ffb020]/10 text-[#ffb020] ring-[#ffb020]/25"
      />
    </motion.div>
  )
}
