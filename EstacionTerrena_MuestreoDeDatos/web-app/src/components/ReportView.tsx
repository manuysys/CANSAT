/* Vista Informe: documento imprimible (KPIs, alertas, top-5 HIGH, corredor y
   firmas). El CSS de impresión muestra SOLO #informe-print, en tinta. */
import type { ReactNode } from 'react'
import { motion } from 'motion/react'
import { Printer } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { alertFrames, useMission } from '@/store/mission'
import { imgURL } from '@/lib/api'
import { intOr, mean, num } from '@/lib/format'
import { domLabel, TERRAIN } from '@/lib/vocab'

/* Reveal on-screen: en impresión el CSS fuerza opacity/transform visibles. */
function Reveal({ children }: { children: ReactNode }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-30px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.section>
  )
}

export function ReportView() {
  const frames = useMission(s => s.frames)
  const summary = useMission(s => s.summary)
  const corridor = useMission(s => s.assets.corridor_map)

  if (!frames.length) {
    return (
      <Card className="mx-auto max-w-xl p-8 text-center text-sm text-muted-foreground">
        Sin telemetría: el informe se genera automáticamente cuando existan frames.
      </Card>
    )
  }

  const alerts = alertFrames(frames)
  const top5 = frames.filter(f => f.sample_pri === 'HIGH')
    .sort((a, b) => (Number(b.sample_score) || 0) - (Number(a.sample_score) || 0)).slice(0, 5)
  const verd: Record<string, number> = summary?.veredictos && Object.keys(summary.veredictos).length
    ? summary.veredictos
    : frames.reduce<Record<string, number>>((acc, f) => {
        const v = String(f.verdict || 'SIN DATOS'); acc[v] = (acc[v] || 0) + 1; return acc
      }, {})
  const people = summary?.personas_total ?? frames.reduce((a, f) => a + intOr(f.people), 0)
  const veh = summary?.vehiculos_total ?? frames.reduce((a, f) => a + intOr(f.vehicles), 0)
  const dmg = summary?.danado_pct_prom ?? mean(frames.map(f => f.danado_pct))
  const altMax = summary?.alt_max_m ?? Math.max(...frames.map(f => Number(f.alt_m) || 0))
  const altMin = summary?.alt_min_m ?? Math.min(...frames.map(f => Number(f.alt_m) || 0))
  const kpis: Array<[string, string]> = [
    ['Frames', String(frames.length)], ['Alertas', String(alerts.length)],
    ['Personas', String(people)], ['Vehículos', String(veh)],
    ['Alt. máx (m)', num(altMax, 1)], ['Alt. mín (m)', num(altMin, 1)],
    ['Daño prom (%)', num(dmg, 1)], ['USI prom.', num(mean(frames.map(f => f.usi)), 3)],
    ['NDVI prom.', num(mean(frames.map(f => f.ndvi)), 3)], ['Score prom.', num(mean(frames.map(f => f.sample_score)), 3)],
  ]

  return (
      <div id="tour-informe" className="mx-auto w-full max-w-[940px]">
      <div className="no-print mb-4 flex items-center gap-3">
        <Button onClick={() => window.print()} className="gap-2">
          <Printer className="size-4" /> Imprimir / PDF
        </Button>
        <span className="text-xs text-muted-foreground">El informe se imprime en blanco y negro, sin cromatismo de consola.</span>
      </div>

      <article id="informe-print" className="flex flex-col gap-6 rounded-2xl border border-border/60 bg-card/70 p-8 shadow-[0_40px_90px_-50px_rgba(0,0,0,.95)]">
        <div className="no-print -mx-8 -mt-8 h-28 overflow-hidden rounded-t-2xl">
          <img src="/assets/keyart_mision.jpg" alt="" className="h-full w-full object-cover opacity-90" />
        </div>
        <header className="flex items-start justify-between gap-4 border-b-2 border-[#3ddc84] pb-4">
          <div>
            <h2 className="text-xl font-bold uppercase tracking-[0.04em]">Informe de misión · {summary?.mision || 'LB135'}</h2>
            <p className="mt-1 font-mono text-[11.5px] leading-relaxed text-muted-foreground">
              CanSat estudiantil · Estación Terrena Web<br />
              Segmentación 5 clases · Detección YOLO · Consenso de daño · USI/NDVI · Sampler adaptativo
            </p>
          </div>
          <div className="text-right font-mono text-[11px] leading-relaxed text-muted-foreground">
            Generado: {new Date().toLocaleString('es-AR')}<br />
            Fuente: outputs/mission/telemetry.csv<br />
            {summary ? 'Post-vuelo: entrega/summary.json' : 'Post-vuelo: pendiente'}
          </div>
        </header>

        <Reveal>
          <h3 className="mb-2 border-b border-border/60 pb-1 text-[11px] font-bold uppercase tracking-[0.14em] text-[#3ddc84]">1 · Indicadores clave</h3>
          <div className="grid grid-cols-3 gap-2 md:grid-cols-5">
            {kpis.map(([k, v]) => (
              <div key={k} className="rounded-lg border border-border/60 bg-muted/20 px-3 py-2">
                <span className="block text-[10px] uppercase tracking-[0.08em] text-muted-foreground">{k}</span>
                <span className="block font-mono text-lg font-semibold tabular-nums">{v}</span>
              </div>
            ))}
          </div>
        </Reveal>

        <Reveal>
          <h3 className="mb-2 border-b border-border/60 pb-1 text-[11px] font-bold uppercase tracking-[0.14em] text-[#3ddc84]">
            2 · Alertas y diagnósticos de desastre ({alerts.length})
          </h3>
          {alerts.length ? (
            <table className="w-full border-collapse font-mono text-[11px]">
              <thead>
                <tr className="border-b border-border/60 text-left text-[9.5px] uppercase tracking-[0.1em] text-muted-foreground">
                  <th className="py-1.5 pr-2">#</th><th className="py-1.5 pr-2">src</th>
                  <th className="py-1.5 pr-2 text-right">t s</th><th className="py-1.5 pr-2 text-right">alt m</th>
                  <th className="py-1.5 pr-2">diag</th><th className="py-1.5 pr-2 text-right">daño %</th>
                  <th className="py-1.5 pr-2 text-right">pers.</th><th className="py-1.5 pr-2 text-right">veh.</th>
                  <th className="py-1.5 pr-2">pri</th><th className="py-1.5 text-right">alert</th>
                </tr>
              </thead>
              <tbody>
                {alerts.map((f, i) => (
                  <tr key={f.src} className="border-b border-border/40 text-muted-foreground">
                    <td className="py-1.5 pr-2">{i + 1}</td><td className="py-1.5 pr-2 text-foreground/90">{f.src}</td>
                    <td className="py-1.5 pr-2 text-right">{num(f.t_s, 1)}</td>
                    <td className="py-1.5 pr-2 text-right">{num(f.alt_m, 1)}</td>
                    <td className="py-1.5 pr-2">{f.diag || '—'}</td>
                    <td className="py-1.5 pr-2 text-right">{num(f.danado_pct, 1)}</td>
                    <td className="py-1.5 pr-2 text-right">{intOr(f.people)}</td>
                    <td className="py-1.5 pr-2 text-right">{intOr(f.vehicles)}</td>
                    <td className="py-1.5 pr-2">{f.sample_pri || '—'}</td>
                    <td className="py-1.5 text-right">{intOr(f.alert)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="font-mono text-[11px] text-muted-foreground">La misión no registró alertas ni diagnósticos de desastre.</p>
          )}
        </Reveal>

        <Reveal>
          <h3 className="mb-2 border-b border-border/60 pb-1 text-[11px] font-bold uppercase tracking-[0.14em] text-[#3ddc84]">3 · Top 5 frames prioridad HIGH</h3>
          {top5.length ? (
            <table className="w-full border-collapse font-mono text-[11px]">
              <thead>
                <tr className="border-b border-border/60 text-left text-[9.5px] uppercase tracking-[0.1em] text-muted-foreground">
                  <th className="py-1.5 pr-2">#</th><th className="py-1.5 pr-2">src</th>
                  <th className="py-1.5 pr-2 text-right">score</th><th className="py-1.5 pr-2 text-right">t s</th>
                  <th className="py-1.5 pr-2 text-right">alt m</th><th className="py-1.5 pr-2">veredicto</th>
                  <th className="py-1.5 pr-2">diag</th><th className="py-1.5 pr-2 text-right">daño %</th>
                  <th className="py-1.5">clase dom.</th>
                </tr>
              </thead>
              <tbody>
                {top5.map((f, i) => (
                  <tr key={f.src} className="border-b border-border/40 text-muted-foreground">
                    <td className="py-1.5 pr-2">{i + 1}</td><td className="py-1.5 pr-2 text-foreground/90">{f.src}</td>
                    <td className="py-1.5 pr-2 text-right">{num(f.sample_score, 3)}</td>
                    <td className="py-1.5 pr-2 text-right">{num(f.t_s, 1)}</td>
                    <td className="py-1.5 pr-2 text-right">{num(f.alt_m, 1)}</td>
                    <td className="py-1.5 pr-2">{f.verdict || '—'}</td>
                    <td className="py-1.5 pr-2">{f.diag || '—'}</td>
                    <td className="py-1.5 pr-2 text-right">{num(f.danado_pct, 1)}</td>
                    <td className="py-1.5">{domLabel(f)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="font-mono text-[11px] text-muted-foreground">Ningún frame alcanzó la prioridad HIGH.</p>
          )}
        </Reveal>

        <div className="grid gap-6 md:grid-cols-2">
          <Reveal>
            <h3 className="mb-2 border-b border-border/60 pb-1 text-[11px] font-bold uppercase tracking-[0.14em] text-[#3ddc84]">4 · Veredictos ambientales</h3>
            <table className="w-full border-collapse font-mono text-[11px]">
              <tbody>
                {Object.entries(verd).map(([k, v]) => (
                  <tr key={k} className="border-b border-border/40 text-muted-foreground">
                    <td className="py-1.5">{k}</td>
                    <td className="py-1.5 text-right">{v}</td>
                    <td className="py-1.5 text-right">{((Number(v) / (frames.length || 1)) * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Reveal>
          <Reveal>
            <h3 className="mb-2 border-b border-border/60 pb-1 text-[11px] font-bold uppercase tracking-[0.14em] text-[#3ddc84]">5 · Terreno promedio</h3>
            <table className="w-full border-collapse font-mono text-[11px]">
              <tbody>
                {TERRAIN.map(t => (
                  <tr key={t.key} className="border-b border-border/40 text-muted-foreground">
                    <td className="py-1.5">{t.label}</td>
                    <td className="py-1.5 text-right">{num(mean(frames.map(f => Number(f[t.key]) || 0)), 1)}%</td>
                  </tr>
                ))}
                <tr className="border-b border-border/40 text-muted-foreground">
                  <td className="py-1.5">USI promedio</td><td className="py-1.5 text-right">{num(mean(frames.map(f => f.usi)), 3)}</td>
                </tr>
                <tr className="border-b border-border/40 text-muted-foreground">
                  <td className="py-1.5">NDVI promedio</td><td className="py-1.5 text-right">{num(mean(frames.map(f => f.ndvi)), 3)}</td>
                </tr>
              </tbody>
            </table>
          </Reveal>
        </div>

        {corridor && (
          <Reveal>
            <h3 className="mb-2 border-b border-border/60 pb-1 text-[11px] font-bold uppercase tracking-[0.14em] text-[#3ddc84]">6 · Corredor de vuelo</h3>
            <img src={imgURL(corridor) ?? ''} alt="Corredor de vuelo apilado" className="w-full rounded-lg border border-border/60" />
          </Reveal>
        )}

        <div className="mt-6 grid grid-cols-3 gap-8">
          {['Operador de estación', 'Responsable de misión', 'Docente asesor'].map(role => (
            <div key={role} className="border-t border-foreground/40 pt-1.5 text-center text-[11px] text-muted-foreground">
              {role}
              <small className="block text-[10px] text-muted-foreground/70">nombre y aclaración</small>
            </div>
          ))}
        </div>

        <footer className="flex justify-between border-t border-border/60 pt-2 font-mono text-[10px] text-muted-foreground">
          <span>Estación Terrena · CanSat {summary?.mision || 'LB135'} · documento generado automáticamente (solo lectura)</span>
          <span>{frames.length} frames · {alerts.length} alertas</span>
        </footer>
      </article>
    </div>
  )
}
