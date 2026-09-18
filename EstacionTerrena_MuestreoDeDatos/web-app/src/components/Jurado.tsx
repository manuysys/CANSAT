/* Modo jurado: deck MANUAL de 6 slides para la defensa (sin narrativa
   automática compleja): portada → briefing → descenso 3D → momentos
   críticos → veredictos → cierre. Navegación ←→, dots, auto 12 s, Esc. */
import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ChevronLeft, ChevronRight, GraduationCap, Play, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { criticalFrame, useMission } from '@/store/mission'
import { imgURL } from '@/lib/api'
import { mean, num } from '@/lib/format'
import { TERRAIN, verdictClass } from '@/lib/vocab'
import { isAlertFrame } from '@/lib/vocab'

const DescentScene = lazy(() =>
  import('@/components/scene/DescentScene').then(m => ({ default: m.DescentScene })),
)

const DONUT: Record<string, string> = {
  'v-ok': '#2fbf74', 'v-mod': '#ffb020', 'v-high': '#ff4d5e', 'v-bare': '#b98a5e',
}

export function Jurado() {
  const on = useMission(s => s.jurado)
  const setJurado = useMission(s => s.setJurado)
  const frames = useMission(s => s.frames)
  const summary = useMission(s => s.summary)
  const [slide, setSlide] = useState(0)
  const [auto, setAuto] = useState(false)

  const N = 6
  const ir = (i: number) => setSlide(((i % N) + N) % N)

  useEffect(() => {
    if (!on) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setJurado(false)
      if (e.key === 'ArrowRight') ir(slide + 1)
      if (e.key === 'ArrowLeft') ir(slide - 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [on, slide, setJurado])

  useEffect(() => {
    if (!on || !auto) return
    const id = window.setInterval(() => setSlide(s => (s + 1) % N), 12000)
    return () => window.clearInterval(id)
  }, [on, auto])

  const crit = useMemo(() => criticalFrame(frames), [frames])
  const alerts = useMemo(() => frames.filter(isAlertFrame), [frames])
  const top3 = useMemo(() => alerts.slice(0, 3), [alerts])
  const verd = useMemo(() => {
    const out: Record<string, number> = {}
    frames.forEach(f => { const v = String(f.verdict || 'SIN DATOS'); out[v] = (out[v] || 0) + 1 })
    return Object.entries(out).sort((a, b) => b[1] - a[1])
  }, [frames])
  const means = useMemo(() => TERRAIN.map(t => ({ t, v: mean(frames.map(f => Number(f[t.key]) || 0)) ?? 0 })).sort((a, b) => b.v - a.v), [frames])

  if (!on) return null

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex flex-col bg-[#05070a]"
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[5vh] bg-black" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-[5vh] bg-black" />
      <div className="present-vignette pointer-events-none absolute inset-0" />

      {/* header */}
      <div className="relative flex items-center gap-4 px-8 py-4">
        <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-[#3ddc84]">
          <GraduationCap className="size-4" /> modo jurado · deck manual
        </span>
        <span className="font-mono text-xs text-muted-foreground">{slide + 1} / {N}</span>
        <Button variant="outline" size="sm" className={`gap-2 ${auto ? 'border-[#3ddc84]/50 text-[#7ee8b0]' : ''}`} onClick={() => setAuto(a => !a)} title="Avance automático cada 12 s">
          <Play className="size-3" /> auto
        </Button>
        <Button variant="ghost" size="sm" className="ml-auto gap-2" onClick={() => setJurado(false)}>
          <X className="size-4" /> Salir (Esc)
        </Button>
      </div>

      {/* slide */}
      <div className="relative min-h-0 flex-1 px-8 pb-2">
        <AnimatePresence mode="wait">
          <motion.div
            key={slide}
            initial={{ opacity: 0, x: 46 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -46 }}
            transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
            className="h-full w-full"
          >
            {slide === 0 && (
              <div className="relative h-full overflow-hidden rounded-2xl border border-border/50">
                <img src="/assets/keyart_mision.jpg" alt="" className="absolute inset-0 h-full w-full object-cover opacity-60" />
                <div className="absolute inset-0 bg-gradient-to-t from-[#05070a] via-transparent to-[#05070a]/60" />
                <div className="relative flex h-full flex-col items-center justify-center gap-3 text-center">
                  <span className="font-mono text-[11px] uppercase tracking-[0.3em] text-[#7ee8b0]">cansat estudiantil · defensa técnica</span>
                  <h1 className="text-5xl font-bold tracking-tight text-[#eafff3] md:text-7xl">Estación Terrena</h1>
                  <span className="font-mono text-sm text-muted-foreground">misión {summary?.mision || 'LB135'} · {new Date().toLocaleDateString('es-AR')}</span>
                </div>
              </div>
            )}

            {slide === 1 && (
              <div className="flex h-full flex-col justify-center gap-6 md:px-10">
                <h2 className="text-4xl font-bold tracking-tight text-[#eafff3]">Descenso <span className="text-[#3ddc84]">recuperado.</span></h2>
                <p className="max-w-3xl text-[15px] leading-relaxed text-foreground/85">
                  La misión <strong>{summary?.mision || 'LB135'}</strong> procesó <strong>{frames.length} frames</strong> de evidencia
                  aérea entre {num(summary?.alt_max_m, 0)} m y {num(summary?.alt_min_m, 0)} m.
                  {crit && <> El momento más crítico fue <strong className="font-mono">{crit.src}</strong> ({crit.diag}, daño {num(crit.danado_pct, 1)}%).</>}
                </p>
                <div className="flex flex-wrap gap-3 font-mono text-[13px]">
                  {[
                    [`${frames.length}`, 'frames'],
                    [`${alerts.length}`, 'alertas'],
                    [`${num(summary?.danado_pct_prom ?? 0, 1)}%`, 'daño prom.'],
                    [`${num(mean(frames.map(f => f.usi)), 3)}`, 'USI prom.'],
                  ].map(([v, k]) => (
                    <span key={k} className="rounded-xl border border-border/60 bg-card/60 px-4 py-2">
                      <span className="text-[18px] font-semibold text-[#eafff3]">{v}</span>{' '}
                      <span className="text-muted-foreground">{k}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {slide === 2 && (
              <div className="flex h-full flex-col gap-2">
                <Suspense fallback={<div className="h-full animate-pulse rounded-2xl bg-card/40" />}>
                  <DescentScene />
                </Suspense>
                <span className="text-center font-mono text-[10.5px] text-muted-foreground">
                  el CanSat 3D reproduce el descenso real: pulsá ▶ Descenso o arrastrá el scrubber de la vista Vuelo
                </span>
              </div>
            )}

            {slide === 3 && (
              <div className="flex h-full flex-col justify-center gap-5">
                <h2 className="text-3xl font-bold tracking-tight text-[#eafff3]">Momentos críticos</h2>
                <div className="grid gap-4 md:grid-cols-3">
                  {top3.map(f => (
                    <figure key={f.src} className="overflow-hidden rounded-xl border border-[#ff4d5e]/30 bg-card/60">
                      <img src={imgURL(f.files?.vis) ?? ''} alt={`Evidencia de ${f.src}`} className="h-52 w-full object-cover" />
                      <figcaption className="space-y-0.5 px-3 py-2 font-mono text-[10.5px]">
                        <div className="flex justify-between"><span className="font-semibold text-[#eafff3]">{f.src}</span><span className="text-[#ff8f9b]">daño {num(f.danado_pct, 1)}%</span></div>
                        <div className="text-muted-foreground">{f.diag} · {num(f.alt_m, 0)} m</div>
                      </figcaption>
                    </figure>
                  ))}
                </div>
              </div>
            )}

            {slide === 4 && (
              <div className="flex h-full flex-col justify-center gap-5 md:px-10">
                <h2 className="text-3xl font-bold tracking-tight text-[#eafff3]">Veredictos ambientales</h2>
                <ul className="flex max-w-xl flex-col gap-2.5">
                  {verd.map(([name, v]) => (
                    <li key={name} className="grid grid-cols-[12px_1fr_auto] items-center gap-3 text-[14px]">
                      <span className="size-3 rounded-[4px]" style={{ background: DONUT[verdictClass(name)] || '#7d8a99' }} />
                      <span className="text-foreground/90">{name}</span>
                      <span className="font-mono tabular-nums text-muted-foreground">{v} · {((v / frames.length) * 100).toFixed(0)}%</span>
                    </li>
                  ))}
                </ul>
                <p className="font-mono text-[12px] text-muted-foreground">
                  terreno dominante: {means[0]?.t.label.toLowerCase()} ({num(means[0]?.v, 1)}% promedio)
                </p>
              </div>
            )}

            {slide === 5 && (
              <div className="flex h-full flex-col items-center justify-center gap-5 text-center">
                <h2 className="text-4xl font-bold tracking-tight text-[#eafff3]">Conclusiones</h2>
                <ul className="flex max-w-2xl flex-col gap-2 text-[14px] leading-relaxed text-foreground/85">
                  <li>· {frames.length} frames procesados a bordo con sampler adaptativo ({frames.filter(f => f.sample_pri === 'HIGH').length} en prioridad HIGH).</li>
                  <li>· {alerts.length} eventos de atención detectados por consenso de modelos; el más severo {crit?.src} con {num(crit?.danado_pct, 1)}% de daño.</li>
                  <li>· Descenso completo registrado de {num(summary?.alt_max_m, 0)} m a {num(summary?.alt_min_m, 0)} m con evidencia georreferenciable por frame.</li>
                </ul>
                <span className="mt-4 font-mono text-[12px] uppercase tracking-[0.3em] text-[#7ee8b0]">gracias · preguntas bienvenidas</span>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* footer nav */}
      <div className="relative flex items-center justify-center gap-4 px-8 py-4">
        <Button variant="outline" size="icon" className="size-8" onClick={() => ir(slide - 1)} title="Slide anterior (←)"><ChevronLeft className="size-4" /></Button>
        <div className="flex gap-1.5">
          {Array.from({ length: N }, (_, i) => (
            <button
              key={i} type="button" onClick={() => ir(i)} aria-label={`Ir al slide ${i + 1}`}
              className={`h-[3px] w-7 rounded-full transition-colors ${i === slide ? 'bg-[#3ddc84]' : 'bg-[#222d3a]'}`}
            />
          ))}
        </div>
        <Button variant="outline" size="icon" className="size-8" onClick={() => ir(slide + 1)} title="Slide siguiente (→)"><ChevronRight className="size-4" /></Button>
      </div>
    </motion.div>
  )
}
