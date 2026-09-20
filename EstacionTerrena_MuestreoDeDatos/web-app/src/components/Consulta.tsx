/* Consulta Terrestre: panel de consultas simbólicas sobre las máscaras de clase
 * por frame (entrega/masks, contrato v3). El servidor delega en
 * cansat/consultas.py del repo de vuelo; acá no hay LLM ni cálculo, solo UI:
 * chips de plantillas, dibujo de zona en el mapa y overlay de resultados. */
import { useState } from 'react'
import { motion } from 'motion/react'
import { MapPinned, Search, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Scramble } from '@/components/Scramble'
import { fetchConsulta } from '@/lib/api'
import { useMission } from '@/store/mission'
import type { ConsultaResult, ConsultaTotal } from '@/lib/types'

const SUGERENCIAS = [
  'área de edificios inundados',
  '¿cuántos edificios con daño hay a menos de 50 m de una vía?',
  '¿qué fracción de las vías está inundada?',
  'distancia entre edificios y agua',
  '¿cuántas personas hay en la zona?',
  '¿hay agua en la zona?',
]

function totalTexto(c: ConsultaResult | null): string {
  const t = c?.total as ConsultaTotal | undefined
  if (t === null || t === undefined) return '—'
  if (typeof t === 'number') {
    const n = t.toLocaleString('es-AR', { maximumFractionDigits: 2 })
    return `${n} ${c?.unidades ?? ''}`.trim()
  }
  if ('fraccion' in t) {
    const f = ((t.fraccion ?? 0) * 100).toFixed(1)
    return `${f} % · ${t.longitud_afectada_m ?? '—'} / ${t.longitud_a_m ?? '—'} m`
  }
  if ('min_m' in t) return `mín ${t.min_m ?? '—'} m · media ${t.media_m ?? '—'} m`
  if ('frames_con_presencia' in t) {
    const n = t.frames_con_presencia ?? 0
    const de = t.de ?? 0
    return `${n > 0 ? 'sí' : 'no'} · ${n}/${de} frames con presencia`
  }
  return '—'
}

export function Consulta() {
  const consulta = useMission(s => s.consulta)
  const cargando = useMission(s => s.consultaCargando)
  const zona = useMission(s => s.consultaZona)
  const dibujando = useMission(s => s.dibujandoZona)
  const setConsulta = useMission(s => s.setConsulta)
  const setCargando = useMission(s => s.setConsultaCargando)
  const setZona = useMission(s => s.setConsultaZona)
  const setDibujando = useMission(s => s.setDibujandoZona)
  const select = useMission(s => s.select)
  const [texto, setTexto] = useState(SUGERENCIAS[0])

  const correr = async (q: string) => {
    const limpio = q.trim()
    if (!limpio) return
    setTexto(limpio)
    setCargando(true)
    try {
      setConsulta(await fetchConsulta(limpio, zona.length >= 3 ? zona : null))
    } catch (e) {
      setConsulta({ consulta: limpio, soportada: false, motivo: `error de red: ${e}` })
    } finally {
      setCargando(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <Card className="border-border/60 bg-card/60" data-testid="consulta-panel">
        <div className="flex items-center justify-between border-b border-border/50 px-5 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            <Scramble text="Consulta terrestre" />
          </h2>
          <div className="flex items-center gap-2">
            {consulta?.soportada && (
              <Badge
                variant="secondary"
                data-testid="consulta-badge"
                className="border-[#3ddc84]/40 bg-[#3ddc84]/10 text-[#3ddc84]"
              >
                consulta_espacial
              </Badge>
            )}
            <MapPinned className="size-3.5 text-muted-foreground/60" />
          </div>
        </div>

        <div className="p-4">
          <form
            className="flex gap-2"
            onSubmit={e => { e.preventDefault(); void correr(texto) }}
          >
            <input
              value={texto}
              onChange={e => setTexto(e.target.value)}
              placeholder="p. ej. ¿cuántos edificios con daño hay a menos de 50 m de una vía?"
              data-testid="consulta-input"
              className="h-9 min-w-0 flex-1 rounded-md border border-border/60 bg-muted/10 px-3 text-[12px] outline-none focus:border-[#3ddc84]/60"
            />
            <Button type="submit" size="sm" disabled={cargando}>
              <Search data-icon="inline-start" />
              {cargando ? 'Consultando…' : 'Consultar'}
            </Button>
          </form>

          <div className="mt-2 flex flex-wrap gap-1.5">
            {SUGERENCIAS.map((s, i) => (
              <button
                key={s}
                type="button"
                onClick={() => void correr(s)}
                data-testid={i === 0 ? 'consulta-chip' : undefined}
                className="rounded-md border border-border/50 bg-muted/10 px-2 py-0.5 text-left text-[10.5px] transition-colors hover:border-[#3ddc84]/60"
              >
                {s}
              </button>
            ))}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border/50 pt-3 text-[11px]">
            <Button
              type="button"
              variant={dibujando ? 'secondary' : 'outline'}
              size="xs"
              onClick={() => setDibujando(!dibujando)}
            >
              <MapPinned data-icon="inline-start" />
              {dibujando ? 'Clic en el mapa…' : 'Dibujar zona'}
            </Button>
            {zona.length > 0 && (
              <>
                <span className="font-mono text-[10.5px] text-muted-foreground">
                  {zona.length} vértices
                </span>
                <Button type="button" variant="ghost" size="xs"
                        onClick={() => setDibujando(false)}>
                  Cerrar zona
                </Button>
                <Button
                  type="button" variant="ghost" size="xs"
                  onClick={() => { setZona([]); setDibujando(false) }}
                >
                  <Trash2 data-icon="inline-start" /> Limpiar
                </Button>
              </>
            )}
            {zona.length < 3 && zona.length > 0 && (
              <span className="text-[10.5px] text-muted-foreground">
                (hacen falta ≥ 3 vértices)
              </span>
            )}
          </div>

          {consulta && (
            <div className="mt-3 border-t border-border/50 pt-3" data-testid="consulta-result">
              {consulta.soportada ? (
                <>
                  <p className="text-[11px] text-muted-foreground">
                    {consulta.plantilla} · {consulta.n_frames} frames · {consulta.region_declarada ? 'con zona' : 'misión completa'}
                  </p>
                  <p className="mt-1 font-mono text-xl tabular-nums text-[#3ddc84]"
                     data-testid="consulta-total">
                    {totalTexto(consulta)}
                  </p>
                  <ul className="mt-2 grid gap-1">
                    {(consulta.por_frame ?? []).slice(0, 5).map(f => (
                      <li key={f.src} className="flex items-center justify-between gap-2 text-[11px]">
                        <button
                          type="button"
                          onClick={() => select(f.src)}
                          className="font-mono text-muted-foreground transition-colors hover:text-foreground"
                        >
                          {f.src}
                        </button>
                        <span className="font-mono tabular-nums">
                          {f.valor} {f.unidad}
                        </span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-[10px] text-muted-foreground/80">
                    {consulta.georref
                      ? 'Zona georreferenciada de forma aproximada (FOV declarado, sin heading).'
                      : consulta.limitaciones?.[0]}
                  </p>
                </>
              ) : (
                <div>
                  <p className="text-[11px] text-destructive">Consulta no soportada</p>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    {consulta.motivo ?? 'no mapea a ninguna plantilla'}
                  </p>
                  {(consulta.sugerencias ?? []).length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {(consulta.sugerencias ?? []).map(s => (
                        <button
                          key={s}
                          type="button"
                          onClick={() => void correr(s)}
                          className="rounded-md border border-border/50 px-2 py-0.5 text-[10.5px]"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <p className="mt-3 text-[10.5px] text-muted-foreground/70">
            Motor simbólico (sin LLM): plantillas en español → operaciones sobre
            las máscaras de clase por frame. Lo que no mapea, no se responde.
          </p>
        </div>
      </Card>
    </motion.div>
  )
}
