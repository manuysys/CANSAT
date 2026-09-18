/* Panel de Muestreo: qué decidió el sampler a bordo y con qué señal.
 *
 * Usa dos fuentes:
 *   · el CSV (prioridad, score por frame, terreno) → cobertura y distribución;
 *   · /api/samples (JSONL) → incertidumbre, tiempos por etapa y frames borrosos,
 *     que el pipeline registra a bordo pero no van al CSV.
 *
 * El objetivo operativo es doble: verificar que el muestreo adaptativo cubrió
 * todas las clases (y no se quedó mirando sólo el desastre) y detectar frames
 * que conviene revisar a mano (incertidumbre alta). */
import { useMemo } from 'react'
import { motion } from 'motion/react'
import { Activity, Clock, Gauge, ScanEye } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { EmptyState } from '@/components/EmptyState'
import { Scramble } from '@/components/Scramble'
import { useMission } from '@/store/mission'
import { num } from '@/lib/format'
import { TERRAIN, priColor } from '@/lib/vocab'
import type { Frame } from '@/lib/types'

function Bar({ label, value, max, color, hint }: {
  label: string; value: number; max: number; color: string; hint?: string
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  return (
    <div className="flex items-center gap-2" title={hint}>
      <span className="w-24 shrink-0 truncate text-[11px] text-muted-foreground">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted/30">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="w-14 shrink-0 text-right font-mono text-[11px] tabular-nums">{num(value, 1)}</span>
    </div>
  )
}

export function Sampling() {
  const frames = useMission(s => s.frames)
  const samples = useMission(s => s.samples)
  const resumen = useMission(s => s.samplesResumen)
  const select = useMission(s => s.select)

  const pri = useMemo(() => {
    const out: Record<string, number> = { HIGH: 0, MEDIUM: 0, LOW: 0 }
    frames.forEach(f => {
      const k = String(f.sample_pri || 'LOW').toUpperCase()
      if (k in out) out[k] += 1
    })
    return out
  }, [frames])

  const cobertura = useMemo(() => {
    if (!frames.length) return [] as Array<{ k: string; label: string; hex: string; mean: number }>
    return TERRAIN.map(t => {
      const mean = frames.reduce((a, f) => a + (Number(f[t.key]) || 0), 0) / frames.length
      return { k: t.key, label: t.label, hex: t.hex, mean }
    })
  }, [frames])

  const review = useMemo(() => {
    return frames
      .map(f => ({ f, u: Number(samples[f.src]?.uncert) }))
      .filter(x => Number.isFinite(x.u))
      .sort((a, b) => b.u - a.u)
      .slice(0, 6)
  }, [frames, samples])

  const maxPri = Math.max(1, ...Object.values(pri))
  const maxCob = Math.max(1, ...cobertura.map(c => c.mean))

  if (!frames.length) {
    return (
      <EmptyState
        icon={ScanEye}
        title="Sin datos de muestreo todavía."
        hint="El panel se llena cuando el pipeline escribe telemetría (y su JSONL con la incertidumbre y los tiempos por frame)."
        className="mx-auto max-w-xl py-10"
      />
    )
  }

  const hayJsonl = Object.keys(samples).length > 0

  return (
    <motion.div
      className="grid gap-4 lg:grid-cols-3"
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <Card className="border-border/60 bg-card/60">
        <div className="flex items-center justify-between border-b border-border/50 px-5 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            <Scramble text="Prioridad del sampler" />
          </h2>
          <Gauge className="size-3.5 text-muted-foreground/60" />
        </div>
        <div className="flex flex-col gap-2 p-4">
          {(['HIGH', 'MEDIUM', 'LOW'] as const).map(k => (
            <Bar key={k} label={`${k} → ${k === 'HIGH' ? 'high_res' : k === 'MEDIUM' ? 'full_res' : 'thumb'}`}
                 value={pri[k]} max={maxPri} color={priColor(k)}
                 hint={`${pri[k]} frame(s) guardados con prioridad ${k}`} />
          ))}
          <p className="mt-1 text-[11px] text-muted-foreground/80">
            {num(frames.length, 0)} frames · {num(pri.HIGH, 0)} a resolución completa
          </p>
        </div>
      </Card>

      <Card className="border-border/60 bg-card/60">
        <div className="flex items-center justify-between border-b border-border/50 px-5 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            <Scramble text="Cobertura por clase" />
          </h2>
          <ScanEye className="size-3.5 text-muted-foreground/60" />
        </div>
        <div className="flex flex-col gap-2 p-4">
          {cobertura.map(c => (
            <Bar key={c.k} label={c.label} value={c.mean} max={maxCob} color={c.hex}
                 hint={`Promedio ${c.mean.toFixed(1)} % por frame (uniforme = 20 %)`} />
          ))}
          <p className="mt-1 text-[11px] text-muted-foreground/80">
            Promedio por frame. El muestreo es bueno si ninguna clase queda en 0.
          </p>
        </div>
      </Card>

      <Card className="border-border/60 bg-card/60">
        <div className="flex items-center justify-between border-b border-border/50 px-5 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            <Scramble text="Señal y costo a bordo" />
          </h2>
          <Activity className="size-3.5 text-muted-foreground/60" />
        </div>
        <div className="p-4">
          {hayJsonl ? (
            <ul className="grid grid-cols-2 gap-x-3 gap-y-2 text-[11px]">
              <li className="text-muted-foreground">Incertidumbre p50</li>
              <li className="text-right font-mono tabular-nums">{num(resumen?.uncert_p50, 3)}</li>
              <li className="text-muted-foreground">Incertidumbre p95</li>
              <li className="text-right font-mono tabular-nums">{num(resumen?.uncert_p95, 3)}</li>
              <li className="text-muted-foreground">Tiempo/frame (mediana)</li>
              <li className="text-right font-mono tabular-nums">
                {resumen?.ms_total_mediana != null ? `${num(resumen.ms_total_mediana, 0)} ms` : '—'}
              </li>
              <li className="text-muted-foreground">Tiempo/frame (p95)</li>
              <li className="text-right font-mono tabular-nums">
                {resumen?.ms_total_p95 != null ? `${num(resumen.ms_total_p95, 0)} ms` : '—'}
              </li>
              <li className="text-muted-foreground">Sin datos p95</li>
              <li className="text-right font-mono tabular-nums">{num(resumen?.nodata_p95, 1)} %</li>
              <li className="text-muted-foreground">Frames borrosos</li>
              <li className="text-right font-mono tabular-nums">{num(resumen?.n_borrosos, 0)}</li>
            </ul>
          ) : (
            <p className="text-[11px] text-muted-foreground">
              Sin <code>telemetry.jsonl</code> en el servidor: la incertidumbre y los
              tiempos por frame se registran a bordo pero no viajan en el CSV.
            </p>
          )}
          {review.length > 0 && (
            <div className="mt-3 border-t border-border/50 pt-3">
              <p className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.08em] text-muted-foreground">
                <Clock className="size-3" /> Mayor incertidumbre (revisar)
              </p>
              <div className="flex flex-wrap gap-1.5">
                {review.map(({ f, u }) => (
                  <button
                    key={f.src}
                    type="button"
                    onClick={() => select(f.src)}
                    className="rounded-md border border-border/50 bg-muted/10 px-2 py-0.5 font-mono text-[10px] transition-colors hover:border-[#3ddc84]/60"
                    title={`uncert ${u.toFixed(3)} — click para abrir el frame`}
                  >
                    {f.src} · {u.toFixed(2)}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  )
}

export function gpsDe(frames: Frame[]): Array<{ src: string; lat: number; lon: number; alt: number; alert: boolean }> {
  return frames
    .map(f => ({
      src: f.src,
      lat: Number(f.lat),
      lon: Number(f.lon),
      alt: Number(f.alt_m) || 0,
      alert: Number(f.alert) === 1,
    }))
    .filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lon) && (p.lat !== 0 || p.lon !== 0))
}
