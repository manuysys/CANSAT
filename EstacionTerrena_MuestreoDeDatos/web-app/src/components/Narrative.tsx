/* Resumen narrativo automático: la misión contada en lenguaje humano,
   recalculada de los datos en cada sync. */
import { motion } from 'motion/react'
import { ArrowDownRight, NotebookPen, Radio, Target } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { criticalFrame, useMission } from '@/store/mission'
import { mean, num } from '@/lib/format'
import { isAlertFrame, TERRAIN } from '@/lib/vocab'

export function Narrative() {
  const frames = useMission(s => s.frames)
  const summary = useMission(s => s.summary)
  const select = useMission(s => s.select)

  if (!frames.length) {
    return (
      <Card className="mission-brief mission-brief-empty border-border/60 bg-card/60 p-5 text-sm text-muted-foreground">
        <div className="mission-kicker"><Radio className="size-3.5" /> enlace esperando datos</div>
        <h2>La misión empieza con el primer frame.</h2>
        <p><NotebookPen className="mr-2 inline size-4" />Cuando llegue telemetría, esta vista resumirá el descenso, sus momentos críticos y la evidencia asociada.</p>
      </Card>
    )
  }

  const crit = criticalFrame(frames)
  const alerts = frames.filter(isAlertFrame).length
  const high = frames.filter(f => f.sample_pri === 'HIGH').length
  const altMax = summary?.alt_max_m ?? Math.max(...frames.map(f => Number(f.alt_m) || 0))
  const altMin = summary?.alt_min_m ?? Math.min(...frames.map(f => Number(f.alt_m) || 0))
  const means = TERRAIN.map(t => ({ t, v: mean(frames.map(f => f[t.key])) ?? 0 }))
  const dom = means.slice().sort((a, b) => b.v - a.v)[0]

  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
      <Card className="mission-brief border-border/60 bg-card/60 px-5 py-5 md:px-6 md:py-6">
        <span className="corners" />
        <div className="mission-brief-top">
          <div className="mission-kicker"><Radio className="size-3.5" /> misión recuperada · datos de solo lectura</div>
          <span className="mission-index">01 / briefing</span>
        </div>
        <div className="mission-brief-main">
          <div className="min-w-0 flex-1">
            <h2>Descenso <span>recuperado.</span></h2>
            <p className="mission-lead">
              La misión <strong>{summary?.mision || 'LB135'}</strong> recorrió {num(altMax, 0)} m → {num(altMin, 0)} m y dejó una secuencia completa de evidencia aérea.
              {crit && Number(crit.danado_pct) > 0 ? <> El momento más crítico fue <strong className="font-mono">{crit.src}</strong>, con {num(crit.danado_pct, 1)}% de daño a {num(crit.alt_m, 0)} m.</> : null}
            </p>
            <p className="mission-copy">
              {!(crit && Number(crit.danado_pct) > 0) && <>No se detectó daño estructural en todo el descenso. </>}
              Se registraron {alerts} evento{alerts === 1 ? '' : 's'} de atención y {high} frames en prioridad HIGH.
            </p>
          </div>
          <div className="mission-critical">
            <span className="mission-critical-label">Terreno dominante</span>
            <strong>{dom.t.label}</strong>
            <span>{num(dom.v, 1)}% promedio · {frames.length} frames</span>
          </div>
        </div>
        {crit && (
          <Button
            variant="outline" size="sm"
            className="mission-cta border-[#ff4d5e]/40 text-[#ff8f9b] hover:bg-[#ff4d5e]/10 hover:text-[#ffc2c8]"
            onClick={() => {
              select(crit.src)
              window.setTimeout(() => document.getElementById('tour-detalle')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 60)
            }}
          >
            <Target className="size-3.5" />
            Ver momento crítico
            <ArrowDownRight className="size-3.5" />
          </Button>
        )}
      </Card>
    </motion.div>
  )
}
