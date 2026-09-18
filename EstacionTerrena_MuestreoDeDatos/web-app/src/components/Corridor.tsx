/* Corredor de descenso vertical: el eje Y es la altitud (250 → 0 m).
   Cada frame es una card con thumb ≥140 px, t, altitud, borde por prioridad y
   anillo rojo si alert=1. Scroll-reveal con whileInView y layout al filtrar. */
import { useEffect, useMemo, useRef } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { MountainSnow } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { EmptyState } from '@/components/EmptyState'
import { filteredFrames, useMission } from '@/store/mission'
import { imgURL } from '@/lib/api'
import { fly } from '@/lib/fly'
import { intOr, num } from '@/lib/format'
import { priColor } from '@/lib/vocab'

const CARD_H = 121
const CARD_GAP = 12
const PAD = 16

export function Corridor() {
  const frames = useMission(s => s.frames)
  const filters = useMission(s => s.filters)
  const sort = useMission(s => s.sort)
  const selected = useMission(s => s.selected)
  const select = useMission(s => s.select)
  const scrollRef = useRef<HTMLDivElement>(null)

  const list = useMemo(() => filteredFrames({ frames, filters, sort }), [frames, filters, sort])

  // Geometría: altura del lienzo tal que el espaciado respete la altitud
  // sin solapar tarjetas (con relajación anti-colapso y tope de seguridad).
  const geo = useMemo(() => {
    if (!list.length) return null
    const alts = list.map(f => Number(f.alt_m)).filter(v => !Number.isNaN(v))
    const altMax = Math.max(...alts, 1)
    const sorted = alts.slice().sort((a, b) => b - a)
    let minFrac = 1
    for (let i = 0; i < sorted.length - 1; i++) {
      const fr = (sorted[i] - sorted[i + 1]) / altMax
      if (fr > 0) minFrac = Math.min(minFrac, fr)
    }
    const base = list.length * (CARD_H + CARD_GAP) + PAD * 2
    const need = minFrac > 0.004 ? Math.max(base, (CARD_H + CARD_GAP) / minFrac + CARD_H + PAD * 2) : base
    const canvasH = Math.min(6000, Math.round(need))
    const usable = canvasH - CARD_H - PAD * 2
    const yOf = (a: number) => PAD + (1 - Math.max(0, Math.min(altMax, a)) / altMax) * usable
    const ticks: number[] = []
    for (let a = 0; a <= altMax; a += 50) ticks.push(a)
    if (ticks[ticks.length - 1] !== Math.round(altMax)) ticks.push(Math.round(altMax))
    return { altMax, canvasH, yOf, ticks }
  }, [list])

  // Posiciones con relajación (nunca solapar) en orden de altitud.
  const placed = useMemo(() => {
    if (!geo) return []
    const order = list.map(f => ({ f, y: geo.yOf(Number(f.alt_m) || 0) })).sort((a, b) => a.y - b.y)
    let prev = -Infinity
    for (const o of order) { o.y = Math.max(o.y, prev + CARD_H + CARD_GAP); prev = o.y }
    return order
  }, [list, geo])

  // Lleva la selección al centro del scroll del corredor
  // (silenciado durante el flythrough 3D para no robar la pantalla).
  useEffect(() => {
    if (!selected || !scrollRef.current || fly.on) return
    const el = scrollRef.current.querySelector<HTMLElement>(`[data-card-src="${CSS.escape(selected)}"]`)
    if (el) el.scrollIntoView({ block: 'nearest' })
  }, [selected])

  return (
    <Card id="tour-corredor" className="flex h-full min-h-0 flex-col overflow-hidden border-border/60 bg-card/60">
      <div className="flex items-center justify-between border-b border-border/50 px-4 py-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Descenso</h2>
        <span className="font-mono text-[10.5px] text-muted-foreground/70">altitud (m) ↓</span>
      </div>

      <div ref={scrollRef} className="relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
        {!list.length ? (
          <div className="p-4">
            <EmptyState
              icon={MountainSnow}
              title={frames.length ? 'Ningún frame pasa el filtro actual.' : 'Sin frames todavía.'}
              hint={frames.length ? 'Probá limpiar los filtros de la timeline.' : 'El corredor se dibuja solo cuando llegue la primera telemetría.'}
            />
          </div>
        ) : (
          <div className="relative" style={{ height: geo?.canvasH }}>
            {geo?.ticks.map(t => (
              <div key={t} className="absolute left-0 right-0 border-t border-dashed border-border/40" style={{ top: geo.yOf(t) }}>
                <span className="absolute left-2 -top-2 rounded bg-card px-1 font-mono text-[10px] text-muted-foreground/70">{t} m</span>
              </div>
            ))}
            <AnimatePresence initial={false}>
              {placed.map(({ f, y }) => {
                const pri = String(f.sample_pri ?? 'LOW')
                const thumb = imgURL(f.files?.thumb || f.files?.vis)
                const isSel = f.src === selected
                return (
                  <motion.button
                    key={f.src}
                    type="button"
                    data-card-src={f.src}
                    layout
                    initial={{ opacity: 0, y: 14 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.97 }}
                    viewport={{ root: scrollRef, margin: '-8% 0px -8% 0px', amount: 0.2 }}
                    transition={{ type: 'spring', stiffness: 320, damping: 30 }}
                    whileHover={{ scale: 1.015 }}
                    onClick={() => select(f.src)}
                    className="absolute left-14 right-3 grid grid-cols-[140px_minmax(0,1fr)] overflow-hidden rounded-xl border border-border/70 bg-secondary/40 text-left"
                    style={{ top: y, height: CARD_H, borderLeft: `3px solid ${priColor(pri)}` }}
                    title={`${f.src} · ${num(f.alt_m, 1)} m`}
                  >
                    {isSel && (
                      <motion.span
                        layoutId="frame-glow"
                        className="pointer-events-none absolute inset-0 rounded-xl ring-2 ring-[#3ddc84]"
                        transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                      />
                    )}
                    {thumb ? (
                      <img src={thumb} alt={`Miniatura de ${f.src}`} loading="lazy" className="h-full w-[140px] object-cover" />
                    ) : (
                      <span className="h-full w-[140px] bg-muted/30" />
                    )}
                    <span className="flex min-w-0 flex-col gap-0.5 px-3 py-2">
                      <span className="truncate font-mono text-xs font-semibold">{f.src}</span>
                      <span className="truncate font-mono text-[11px] text-muted-foreground">t {num(f.t_s, 1)} s</span>
                      <span className="truncate font-mono text-[11px] text-muted-foreground/70">{num(f.alt_m, 1)} m</span>
                    </span>
                    {intOr(f.alert) === 1 && (
                      <span className="absolute left-2 top-2 size-3 rounded-full border-2 border-[#ff4d5e] bg-[#ff4d5e]/25" title="alert=1" />
                    )}
                  </motion.button>
                )
              })}
            </AnimatePresence>
          </div>
        )}
      </div>
    </Card>
  )
}
