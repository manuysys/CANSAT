/* Pila de toasts discretos arriba a la derecha (feedback de acciones). */
import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle2 } from 'lucide-react'
import { useMission } from '@/store/mission'

export function Toasts() {
  const toasts = useMission(s => s.toasts)
  const drop = useMission(s => s.dropToast)
  return (
    <div className="pointer-events-none fixed right-4 top-16 z-[60] flex w-72 flex-col gap-2">
      <AnimatePresence>
        {toasts.map(t => (
          <motion.button
            key={t.id}
            type="button"
            onClick={() => drop(t.id)}
            initial={{ opacity: 0, x: 24, scale: .96 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 16, scale: .96 }}
            transition={{ duration: .22, ease: 'easeOut' }}
            className="glass pointer-events-auto flex items-start gap-2 rounded-lg border border-[#3ddc84]/30 bg-[#10161f]/90 px-3 py-2 text-left text-[11px] text-foreground/90 shadow-[0_12px_32px_-16px_rgba(0,0,0,.9)]"
          >
            <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-[#3ddc84]" />
            <span>{t.msg}</span>
          </motion.button>
        ))}
      </AnimatePresence>
    </div>
  )
}
