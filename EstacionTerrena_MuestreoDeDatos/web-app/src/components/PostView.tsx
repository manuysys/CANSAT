/* Vista Post-vuelo: donut de veredictos (Recharts), resumen del contrato y
   galerías EDSR / ensemble como thumbs clickeables que abren el detalle. */
import { useMemo } from 'react'
import { motion } from 'motion/react'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { PackageOpen, Images, ImageDown } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/EmptyState'
import { Scramble } from '@/components/Scramble'
import { useTilt } from '@/hooks/useTilt'
import { descargarPoster } from '@/lib/poster'
import { useMission } from '@/store/mission'
import { imgURL } from '@/lib/api'
import { mean, num } from '@/lib/format'
import { verdictClass, verdictColor } from '@/lib/vocab'
import type { Frame, ImgKey } from '@/lib/types'

function StatTile({ k, v, sub }: { k: string; v: string; sub: string }) {
  const tilt = useTilt(4)
  return (
    <div
      {...tilt}
      className="tilt-glare relative overflow-hidden rounded-xl border border-border/60 bg-muted/20 px-3 py-2.5 will-change-transform"
    >
      <span className="glare" />
      <span className="block text-[10px] uppercase tracking-[0.08em] text-muted-foreground">{k}</span>
      <span className="kpi-num block font-mono text-lg font-semibold tabular-nums">{v}</span>
      <span className="block truncate font-mono text-[10px] text-muted-foreground/80">{sub}</span>
    </div>
  )
}

function Gallery({ title, hint, items, bucket }: {
  title: string; hint: string; items: Frame[]; bucket: ImgKey
}) {
  const select = useMission(s => s.select)
  const setView = useMission(s => s.setView)
  const selected = useMission(s => s.selected)
  if (!items.length) return null
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <Card className="border-border/60 bg-card/60">
        <div className="flex items-center justify-between border-b border-border/50 px-5 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground"><Scramble text={title} /></h2>
          <span className="font-mono text-[10.5px] text-muted-foreground/70">{hint}</span>
        </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-3 p-4">
        {items.map(f => (
          <motion.figure
            key={f.src}
            whileHover={{ y: -3 }}
            whileTap={{ scale: 0.98 }}
            role="button"
            tabIndex={0}
            aria-label={`${title} de ${f.src}. Abrir detalle`}
            className={`cursor-pointer overflow-hidden rounded-xl border bg-secondary/30 transition-shadow ${f.src === selected ? 'border-[#3ddc84] shadow-[0_0_26px_-6px_rgba(61,220,132,.5)]' : 'border-border/60 hover:shadow-[0_0_22px_-8px_rgba(61,220,132,.35)]'}`}
            onClick={() => { setView('vuelo'); select(f.src, bucket) }}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setView('vuelo'); select(f.src, bucket) } }}
            title={`${f.src} · abre el detalle con esta variante`}
          >
            <img src={imgURL(f.files?.[bucket]) ?? ''} alt={`${title} de ${f.src}`} loading="lazy" className="h-[100px] w-full object-cover" />
            <figcaption className="truncate border-t border-border/50 bg-[#121a24] px-2 py-1 font-mono text-[10px] text-muted-foreground">
              {f.src}
            </figcaption>
          </motion.figure>
        ))}
      </div>
      </Card>
    </motion.div>
  )
}

export function PostView() {
  const frames = useMission(s => s.frames)
  const summary = useMission(s => s.summary)
  const select = useMission(s => s.select)

  const verd = useMemo(() => {
    const src = summary?.veredictos ?? {}
    if (Object.keys(src).length) return src
    const out: Record<string, number> = {}
    frames.forEach(f => { const v = String(f.verdict || 'SIN DATOS'); out[v] = (out[v] || 0) + 1 })
    return out
  }, [summary, frames])
  const entries = Object.entries(verd).filter(([, v]) => Number(v) > 0)
  const total = entries.reduce((a, [, v]) => a + Number(v), 0) || 1

  const galEnh = frames.filter(f => f.files?.enhanced)
  const galEns = frames.filter(f => f.files?.ens_seg)

  if (!summary && !galEnh.length && !galEns.length) {
    return (
      <EmptyState
        icon={PackageOpen}
        title="Análisis post-vuelo pendiente."
        hint="Todavía no existe entrega/summary.json ni galerías generadas. Esta vista se llena sola en cuanto aparezcan."
        className="mx-auto max-w-xl py-16"
      />
    )
  }

  const alerts = summary?.alertas ?? []
  const perd = summary?.perdidas
  const stats: Array<[string, string, string]> = summary ? [
    ['Frames', num(summary.n_frames, 0), 'procesados a bordo'],
    ['Altitud máx.', num(summary.alt_max_m, 1, ' m'), 'liberación / apogeo'],
    ['Altitud mín.', num(summary.alt_min_m, 1, ' m'), 'fin del descenso'],
    ['Personas', num(summary.personas_total, 0), 'detecciones a bordo'],
    ['Vehículos', num(summary.vehiculos_total, 0), 'detecciones a bordo'],
    ['Daño prom.', num(summary.danado_pct_prom, 1, ' %'), 'consenso de modelos'],
    ['Alertas', num(alerts.length, 0), alerts.length ? 'ver detalle abajo' : 'sin eventos'],
    ['USI prom.', num(mean(frames.map(f => f.usi)), 3), 'estrés urbano'],
    ['Afectados (est.)', num(perd?.personas_afectadas, 1), 'personas en el área dañada'],
    ['Pérdidas (est.)', num(perd?.perdidas_estimadas, 2),
      perd?.perdidas_min != null
        ? `banda ${num(perd.perdidas_min, 2)}–${num(perd.perdidas_max, 2)}`
        : 'exposición, no predicción'],
  ] : []

  return (
    <div className="flex flex-col gap-4">
      <motion.div
        className="grid gap-4 lg:grid-cols-[minmax(280px,380px)_minmax(0,1fr)]"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        <Card className="border-border/60 bg-card/60">
          <span className="corners" />
          <div className="border-b border-border/50 px-5 py-2.5">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground"><Scramble text="Veredictos ambientales" /></h2>
          </div>
          <div className="flex flex-col items-center gap-4 p-5">
            <div className="relative h-[180px] w-[180px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={entries.map(([name, value]) => ({ name, value, fill: verdictColor(name) }))}
                    dataKey="value" nameKey="name"
                    innerRadius="62%" outerRadius="92%" paddingAngle={2}
                    stroke="#0b0e14" strokeWidth={2}
                    isAnimationActive animationDuration={800}
                  >
                    {entries.map(([name]) => <Cell key={name} fill={verdictColor(name)} />)}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: '#121a24', border: '1px solid #263241', borderRadius: 10, fontSize: 11 }}
                    labelStyle={{ color: '#dbe4ee' }} itemStyle={{ color: '#8b9aab' }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
                <span className="font-mono text-[26px] font-semibold tabular-nums">{total}</span>
                <span className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">frames</span>
              </div>
            </div>
            <ul className="w-full flex flex-col gap-1.5">
              {entries.map(([name, v]) => (
                <li key={name} className="grid grid-cols-[10px_1fr_auto] items-center gap-2 text-xs">
                  <span className="size-2.5 rounded-[3px]" style={{ background: verdictColor(name) }} />
                  <span className={`truncate font-medium ${verdictClass(name) === 'v-ok' ? 'text-[#3ddc84]' : verdictClass(name) === 'v-mod' ? 'text-[#ffb020]' : verdictClass(name) === 'v-high' ? 'text-[#ff8f9b]' : verdictClass(name) === 'v-bare' ? 'text-[#d8b48a]' : 'text-muted-foreground'}`}>
                    {name}
                  </span>
                  <span className="font-mono tabular-nums text-muted-foreground">{v} · {((Number(v) / total) * 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </Card>

        <Card className="border-border/60 bg-card/60">
          <div className="flex items-center justify-between border-b border-border/50 px-5 py-2.5">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground"><Scramble text="Resumen del contrato" /></h2>
            <div className="flex items-center gap-2">
              <Button
                variant="outline" size="sm" className="h-7 gap-1.5 text-xs"
                title="Generar y descargar un póster infográfico 1920×1080"
                onClick={() => { void descargarPoster(frames, summary) }}
              >
                <ImageDown className="size-3" /> Póster PNG
              </Button>
              <span className="font-mono text-[10.5px] text-muted-foreground/70">{summary?.mision ? `misión ${summary.mision}` : 'sin summary.json'}</span>
            </div>
          </div>
          <div className="p-4">
            {summary ? (
              <>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  {stats.map(([k, v, sub]) => <StatTile key={k} k={k} v={v} sub={sub} />)}
                </div>
                {perd && perd.perdidas_estimadas != null && (
                  <p className="mt-3 rounded-lg border border-border/40 bg-muted/10 px-3 py-2 text-[10.5px] leading-relaxed text-muted-foreground">
                    <strong className="text-foreground/80">Estimación de pérdidas humanas:</strong>{' '}
                    modelo de exposición con supuestos declarados
                    {perd.supuestos?.pop_density != null
                      ? ` (densidad ${num(perd.supuestos.pop_density, 0)} hab/km², ocupación ${num(perd.supuestos.occupancy, 2)}, colapso ${num(perd.supuestos.collapse_frac, 2)}, letalidad ${num(perd.supuestos.fatality, 2)})`
                      : ''}. No es una predicción de víctimas: la banda refleja la
                    incertidumbre de los factores de vulnerabilidad.
                  </p>
                )}
                {alerts.length > 0 && (
                  <div className="mt-4 border-t border-border/50 pt-3">
                    <p className="mb-2 text-[10px] uppercase tracking-[0.08em] text-muted-foreground">
                      Alertas del consenso · {alerts.length}
                    </p>
                    <ul className="flex max-h-56 flex-col gap-1 overflow-y-auto pr-1">
                      {alerts.map(a => (
                        <li key={a.src}>
                          <button
                            type="button"
                            onClick={() => select(a.src)}
                            title="Seleccionar este frame en la vista Vuelo"
                            className="grid w-full grid-cols-[auto_auto_1fr] items-center gap-2 rounded-lg border border-border/40 bg-muted/10 px-2 py-1.5 text-left text-[11px] transition-colors hover:border-[#3ddc84]/50"
                          >
                            <span className="font-mono text-muted-foreground">{a.src}</span>
                            <span className="rounded bg-[#ff4d5e]/15 px-1.5 py-0.5 font-mono text-[10px] text-[#ff8f9b]">
                              {a.diag ?? 'ALERTA'}
                            </span>
                            <span className="truncate text-muted-foreground">
                              {[
                                a.motivo,
                                a.danado_pct !== null && a.danado_pct !== undefined
                                  ? `daño ${num(Number(a.danado_pct), 1, ' %')}` : null,
                                a.sample_pri ? `pri ${a.sample_pri}` : null,
                              ].filter(Boolean).join(' · ')}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <p className="p-4 text-xs text-muted-foreground">
                Sin <code>entrega/summary.json</code>: el contrato de post-vuelo aún no existe, pero las galerías de abajo ya están disponibles.
              </p>
            )}
          </div>
        </Card>
      </motion.div>

      <Gallery title="Galería mejorada" hint="EDSR · click abre el detalle" items={galEnh} bucket="enhanced" />
      <Gallery title="Overlays ensemble" hint="segmentación post-vuelo · click abre el detalle" items={galEns} bucket="ens_seg" />
      {!galEnh.length && !galEns.length && (
        <EmptyState icon={Images} title="Sin galerías post-vuelo para esta misión." hint="Cuando el pipeline genere EDSR o ens_seg, aparecen como miniaturas clickeables." />
      )}
      <p className="text-center font-mono text-[10.5px] text-muted-foreground/60">
        {frames.length} frames en memoria · veredicto dominante: {entries.slice().sort((a, b) => Number(b[1]) - Number(a[1]))[0]?.[0] ?? '—'}
      </p>
    </div>
  )
}
