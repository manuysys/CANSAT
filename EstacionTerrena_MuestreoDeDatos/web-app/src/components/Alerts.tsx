/* Panel de alertas: solo existe si hay algo que contar. Borde semántico por
   severidad, pulso Motion en el icono y slide-in de las alertas nuevas que
   llegan con el auto-refresh. */
import { AnimatePresence, motion } from 'motion/react'
import { ChevronDown, PartyPopper, TriangleAlert } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/EmptyState'
import { alertFrames, useMission } from '@/store/mission'
import { intOr, num } from '@/lib/format'
import { diagSev } from '@/lib/vocab'

export function Alerts() {
  const frames = useMission(s => s.frames)
  const selected = useMission(s => s.selected)
  const select = useMission(s => s.select)
  const collapsed = useMission(s => s.alertsCollapsed)
  const toggle = useMission(s => s.toggleAlertsCollapsed)

  const list = alertFrames(frames)
  if (!list.length) {
    return (
      <Card id="tour-alertas" className="flex h-full min-h-0 flex-col overflow-hidden border-border/60 bg-card/60">
        <div className="flex items-center justify-between border-b border-border/50 px-4 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Alertas</h2>
          <span className="font-mono text-[10.5px] text-muted-foreground/70">0</span>
        </div>
        <div className="flex flex-1 items-center justify-center p-4">
          <EmptyState icon={PartyPopper} title="Sin alertas: la misión no vio desastres 🎉" hint="Si un frame nuevo llega con alert=1 o diagnóstico de desastre, aparece acá al instante." />
        </div>
      </Card>
    )
  }

  return (
    <Card id="tour-alertas" className="flex h-full min-h-0 flex-col overflow-hidden border-border/60 bg-card/60">
      <div className="flex items-center justify-between border-b border-border/50 px-4 py-2">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Alertas</h2>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10.5px] text-[#ff8f9b]">{list.length}</span>
          <Button variant="ghost" size="icon" className="size-7" onClick={toggle} aria-expanded={!collapsed} title="Colapsar / expandir">
            <ChevronDown className={`size-4 transition-transform ${collapsed ? '-rotate-90' : ''}`} />
          </Button>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: 'easeInOut' }}
            className="min-h-0 flex-1 overflow-y-auto"
          >
            <ul className="flex flex-col gap-2 p-3" aria-live="polite">
              <AnimatePresence initial={false}>
                {list.map(f => {
                  const sev = diagSev(f.diag) + intOr(f.alert)
                  const bad = sev >= 4
                  return (
                    <motion.li
                      key={f.src}
                      layout
                      initial={{ x: 28, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      exit={{ x: 28, opacity: 0 }}
                      transition={{ type: 'spring', stiffness: 340, damping: 30 }}
                    >
                      <button
                        type="button"
                        onClick={() => select(f.src)}
                        className={`w-full rounded-xl border bg-secondary/40 px-3 py-2.5 text-left transition-colors hover:bg-secondary/70 ${
                          f.src === selected ? 'ring-2 ring-[#3ddc84]' : ''
                        } ${bad ? 'border-l-4 border-l-[#ff4d5e] border-border/60' : 'border-l-4 border-l-[#ffb020] border-border/60'}`}
                      >
                        <span className="flex items-start gap-2.5">
                          <motion.span
                            animate={{ scale: [1, 1.14, 1] }}
                            transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut' }}
                            className={`mt-0.5 flex ${bad ? 'text-[#ff4d5e]' : 'text-[#ffb020]'}`}
                          >
                            <TriangleAlert className="size-4" strokeWidth={1.9} />
                          </motion.span>
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-xs font-semibold text-foreground/95">
                              {f.diag || 'Alerta'}{intOr(f.alert) === 1 ? ' · alert=1' : ''}
                            </span>
                            <span className="mt-0.5 block truncate font-mono text-[10.5px] text-muted-foreground">
                              {f.src} · {num(f.t_s, 1)} s · {num(f.alt_m, 0)} m · {num(f.danado_pct, 1)}%
                            </span>
                          </span>
                        </span>
                      </button>
                    </motion.li>
                  )
                })}
              </AnimatePresence>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  )
}
