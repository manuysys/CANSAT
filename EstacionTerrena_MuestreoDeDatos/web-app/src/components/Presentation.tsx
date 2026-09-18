/* Modo presentación para el jurado: pantalla completa cinematográfica con
   letterbox, Ken Burns lento sobre la imagen, viñeta, scanlines y avance
   automático cada 4 s con barra de progreso. Esc o el botón salen. */
import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ScanEye, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { filteredFrames, useMission } from '@/store/mission'
import { imgURL } from '@/lib/api'
import { intOr, num } from '@/lib/format'
import { verdictClass, verdictColor } from '@/lib/vocab'

export function Presentation() {
  const on = useMission(s => s.present.on)
  const idx = useMission(s => s.present.idx)
  const stop = useMission(s => s.stopPresent)
  const setIdx = useMission(s => s.setPresentIdx)
  const frames = useMission(s => s.frames)
  const filters = useMission(s => s.filters)
  const sort = useMission(s => s.sort)

  const list = filteredFrames({ frames, filters, sort })
  const [annotated, setAnnotated] = useState(false)

  // Avance automático + fullscreen.
  useEffect(() => {
    if (!on) return
    const id = window.setInterval(() => {
      const st = useMission.getState()
      const n = filteredFrames(st).length
      if (n) st.setPresentIdx((st.present.idx + 1) % n)
    }, 4000)
    try { void document.documentElement.requestFullscreen?.() } catch { /* opcional */ }
    return () => {
      window.clearInterval(id)
      try { if (document.fullscreenElement) void document.exitFullscreen() } catch { /* opcional */ }
    }
  }, [on])

  const f = list.length ? list[idx % list.length] : null

  return (
    <AnimatePresence>
      {on && (
        <motion.div
          key="present"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          exit={{ opacity: 0, transition: { duration: 0.18, ease: 'easeOut' } }}
          className="fixed inset-0 z-50 flex flex-col gap-4 bg-[#05070a] px-8 py-4"
          role="dialog"
          aria-modal="true"
          aria-label="Presentación de la misión"
        >
          {/* letterbox cinematográfico */}
          <motion.div
            initial={{ scaleY: 0 }} animate={{ scaleY: 1 }}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            className="pointer-events-none absolute inset-x-0 top-0 h-[5vh] origin-top bg-black"
          />
          <motion.div
            initial={{ scaleY: 0 }} animate={{ scaleY: 1 }}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            className="pointer-events-none absolute inset-x-0 bottom-0 h-[5vh] origin-bottom bg-black"
          />
          <div className="cut-scanlines pointer-events-none absolute inset-0 opacity-60" />
          <div className="present-vignette pointer-events-none absolute inset-0" />

          <div className="relative flex items-center gap-4">
            <span className="font-mono text-[15px] font-semibold text-[#eafff3]">{f?.src ?? '—'}</span>
            <span className="font-mono text-xs text-muted-foreground">{list.length ? (idx % list.length) + 1 : 0} / {list.length}</span>
            <span className="hidden font-mono text-[10px] uppercase tracking-[0.2em] text-[#3ddc84]/80 md:inline">
              ● presentación en vivo
            </span>
            <Button
              variant="outline" size="sm" className="gap-2"
              title="Alternar entre la imagen nítida (EDSR/full-res) y la evidencia anotada por el pipeline"
              onClick={() => setAnnotated(a => !a)}
            >
              <ScanEye className="size-3.5" />
              {annotated ? 'Ver nítida' : 'Ver anotada'}
            </Button>
            <Button variant="ghost" size="sm" className="ml-auto gap-2" onClick={stop}>
              <X className="size-4" /> Salir (Esc)
            </Button>
          </div>

          <figure className="relative flex min-h-0 flex-1 items-center justify-center">
            <AnimatePresence mode="wait">
              {f && (
                <motion.img
                  key={f.src + (annotated ? ':ann' : ':sharp')}
                  src={imgURL(
                    annotated
                      ? (f.files?.vis || f.files?.enhanced || f.files?.high_res || f.files?.thumb)
                      : (f.files?.enhanced || f.files?.full_res || f.files?.high_res || f.files?.vis || f.files?.thumb),
                  ) ?? ''}
                  alt={`Frame ${f.src} en presentación`}
                  initial={{ opacity: 0, scale: 1.0 }}
                  animate={{ opacity: 1, scale: 1.06 }}
                  exit={{ opacity: 0, transition: { duration: 0.22, ease: 'easeOut' } }}
                  transition={{ opacity: { duration: 0.35, ease: 'easeOut' }, scale: { duration: 4.2, ease: 'linear' } }}
                  className="max-h-full max-w-full rounded-xl border border-border/50 object-contain shadow-[0_0_80px_-20px_rgba(61,220,132,.25)]"
                />
              )}
            </AnimatePresence>

            {/* PiP: evidencia anotada cuando el principal es la copia nítida */}
            {f && !annotated && f.files?.vis && (f.files?.enhanced || f.files?.full_res) && (
              <div className="pointer-events-none absolute bottom-2 right-2 w-[200px] overflow-hidden rounded-lg border border-[#3ddc84]/40 shadow-[0_0_30px_-8px_rgba(61,220,132,.5)] md:w-[260px]">
                <img src={imgURL(f.files.vis) ?? ''} alt={`Evidencia anotada de ${f.src}`} className="block w-full" />
                <span className="block bg-[#0b0e14]/90 px-2 py-1 font-mono text-[9px] uppercase tracking-[0.12em] text-[#7ee8b0]">
                  evidencia anotada · yolo
                </span>
              </div>
            )}
          </figure>

          <div className="relative flex flex-wrap items-center justify-center gap-x-8 gap-y-2 pb-1">
            {f && (
              <>
                <span className="text-[15px] font-bold uppercase tracking-[0.12em]" style={{ color: verdictColor(f.verdict) }}>
                  {f.verdict || '—'}
                  <span className="sr-only">{verdictClass(f.verdict)}</span>
                </span>
                <span className="font-mono text-[13px] text-foreground/85">{f.diag || '—'}</span>
                <span className="font-mono text-xs text-muted-foreground">
                  t {num(f.t_s, 1)} s · {num(f.alt_m, 1)} m · daño {num(f.danado_pct, 1)}% · {String(f.sample_pri ?? '—')} {num(f.sample_score, 2)}
                  {intOr(f.alert) === 1 ? ' · ALERTA' : ''}
                </span>
              </>
            )}
          </div>

          <div className="relative flex justify-center gap-1.5 pb-2">
            {list.map((x, i) => (
              <button
                key={x.src} type="button" title={x.src} onClick={() => setIdx(i)}
                className={`h-[3px] w-6 overflow-hidden rounded-full transition-colors ${i === idx % list.length ? 'bg-[#222d3a]' : i < idx ? 'bg-[#6b7a8c]' : 'bg-[#222d3a]'}`}
              >
                {i === idx % list.length && (
                  <motion.i
                    key={`p-${idx}`}
                    className="block h-full w-full bg-[#3ddc84]"
                    initial={{ x: '-100%' }} animate={{ x: '0%' }}
                    transition={{ duration: 4, ease: 'linear' }}
                  />
                )}
              </button>
            ))}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
