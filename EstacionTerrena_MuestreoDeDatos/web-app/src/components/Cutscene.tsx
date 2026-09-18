/* Cutscene de arranque: secuencia cinematográfica de "boot" de la estación
   (log de enlace tipeado, logo con reveal y cortina que abre al dashboard).
   - Se reproduce una sola vez por sesión (sessionStorage).
   - Se salta con click / Enter / Esc (fast-forward suave).
   - El botón "clapperboard" del TopBar la repite (evento lb135-cutscene-replay).
   - Al terminar despacha lb135-cutscene-done (el tour de driver.js espera esto).
   - prefers-reduced-motion: no se reproduce. */
import { useEffect, useRef, useState } from 'react'
import gsap from 'gsap'
import { TextPlugin } from 'gsap/TextPlugin'
import { Satellite } from 'lucide-react'

gsap.registerPlugin(TextPlugin)

const KEY = 'lb135-cutscene-v1'

const LOG: Array<[string, string]> = [
  ['› enlace tlm 915 mhz', 'ok'],
  ['› pipeline yolo · edsr', 'ok'],
  ['› sampler adaptativo', 'ok'],
  ['› telemetría recuperada', '12 frames'],
]

/* Flag síncrono para que TourBoot sepa si debe esperar la cutscene. */
let pending = false
export function cutscenePending(): boolean {
  return pending
}

function reduced(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

export function Cutscene() {
  const [phase, setPhase] = useState<'play' | 'done'>(() => {
    const play = !sessionStorage.getItem(KEY) && !reduced()
    pending = play
    return play ? 'play' : 'done'
  })
  const root = useRef<HTMLDivElement>(null)
  const tl = useRef<gsap.core.Timeline | null>(null)

  const finish = () => {
    pending = false
    sessionStorage.setItem(KEY, '1')
    setPhase('done')
    window.dispatchEvent(new Event('lb135-cutscene-done'))
  }

  // Replay desde el TopBar.
  useEffect(() => {
    const replay = () => {
      sessionStorage.removeItem(KEY)
      pending = true
      setPhase('play')
    }
    window.addEventListener('lb135-cutscene-replay', replay)
    return () => window.removeEventListener('lb135-cutscene-replay', replay)
  }, [])

  useEffect(() => {
    if (phase !== 'play') return
    const el = root.current
    if (!el) { finish(); return }
    const q = gsap.utils.selector(el)
    const tlx = gsap.timeline({ onComplete: finish, defaults: { ease: 'power2.out' } })
    tl.current = tlx

    tlx.to(q('.cut-bar i'), { width: '100%', duration: 2.7, ease: 'power1.inOut' }, 0.15)
    q('.cut-line').forEach((line, i) => {
      const t = 0.25 + i * 0.3
      tlx.to(line, { text: { value: LOG[i][0] }, duration: 0.32, ease: 'none' }, t)
      tlx.fromTo([q('.cut-dots')[i], q('.cut-ok')[i]], { opacity: 0 }, { opacity: 1, duration: 0.18 }, t + 0.32)
    })
    tlx
      .fromTo(q('.cut-icon'), { scale: 0, rotate: -140, opacity: 0 }, { scale: 1, rotate: 0, opacity: 1, duration: 0.7, ease: 'back.out(1.9)' }, 1.7)
      .fromTo(q('.cut-ring'), { scale: 0.4, opacity: 0.8 }, { scale: 1.7, opacity: 0, duration: 1.1, ease: 'power2.out' }, 1.85)
      .fromTo(q('.cut-title'), { y: 22, opacity: 0, filter: 'blur(10px)' }, { y: 0, opacity: 1, filter: 'blur(0px)', duration: 0.65 }, 1.95)
      .fromTo(q('.cut-sub'), { opacity: 0, letterSpacing: '0.6em' }, { opacity: 1, letterSpacing: '0.34em', duration: 0.6 }, 2.15)
      .fromTo(q('.cut-rule'), { scaleX: 0 }, { scaleX: 1, duration: 0.5 }, 2.2)
      .to(q('.cut-flash'), { opacity: 0.45, duration: 0.12, ease: 'power2.in' }, 3.05)
      .to(q('.cut-flash'), { opacity: 0, duration: 0.55 }, 3.17)
      // wipe final: se apaga el contenido y las cortinas abren hacia afuera
      .to(el, { pointerEvents: 'none', duration: 0.01 }, 3.05)
      .to(q('.cut-fade'), { opacity: 0, duration: 0.35 }, 3.05)
      .to(q('.cut-curtain-top'), { yPercent: -103, duration: 0.9, ease: 'power3.inOut' }, 3.15)
      .to(q('.cut-curtain-bottom'), { yPercent: 103, duration: 0.9, ease: 'power3.inOut' }, 3.15)

    const skip = () => { if (tl.current) tl.current.timeScale(5) }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') skip()
    }
    el.addEventListener('click', skip)
    window.addEventListener('keydown', onKey)
    return () => {
      el.removeEventListener('click', skip)
      window.removeEventListener('keydown', onKey)
      tlx.kill()
    }
  }, [phase])

  if (phase === 'done') return null

  return (
    <div
      id="cutscene"
      ref={root}
      className="fixed inset-0 z-[70] cursor-pointer select-none overflow-hidden"
      role="presentation"
    >
      {/* cortinas: tapan el dashboard hasta el wipe final */}
      <div className="cut-curtain-top absolute inset-x-0 top-0 h-1/2 bg-[#04060a]" />
      <div className="cut-curtain-bottom absolute inset-x-0 bottom-0 h-1/2 bg-[#04060a]" />

      <img
        src="/assets/earth_limb.jpg" alt=""
        className="cut-fade cut-limb pointer-events-none absolute inset-x-0 bottom-0 h-[68%] w-full object-cover"
      />
      <div className="cut-stars cut-fade absolute inset-0" />
      <div className="cut-scanlines cut-fade absolute inset-0" />

      {/* log de boot */}
      <div className="cut-fade absolute left-6 top-6 space-y-1 font-mono text-[11px] text-[#7ee8b0]/90 md:left-10 md:top-10">
        {LOG.map(([txt, ok]) => (
          <div key={txt} className="flex items-center gap-2">
            <span className="cut-line min-h-[1em] whitespace-pre">{''}</span>
            <span className="cut-dots text-[#39424e]">·········</span>
            <span className="cut-ok font-semibold text-[#3ddc84]">{ok}</span>
          </div>
        ))}
      </div>

      {/* marca central */}
      <div className="cut-fade absolute inset-0 flex flex-col items-center justify-center gap-4">
        <div className="relative">
          <span className="cut-ring absolute inset-0 rounded-full border border-[#3ddc84]/50" />
          <span className="cut-icon grid size-20 place-items-center rounded-full border border-[#3ddc84]/40 bg-[#3ddc84]/10 shadow-[0_0_60px_-10px_rgba(61,220,132,.65)]">
            <Satellite className="size-9 text-[#3ddc84]" strokeWidth={1.6} />
          </span>
        </div>
        <h1 className="cut-title text-3xl font-bold tracking-[0.08em] text-[#eafff3] md:text-5xl">
          ESTACIÓN TERRENA
        </h1>
        <div className="cut-sub font-mono text-[11px] uppercase text-[#8b9aab] md:text-xs">
          cansat · misión lb135 · descenso recuperado
        </div>
        <div className="cut-rule h-px w-56 bg-gradient-to-r from-transparent via-[#3ddc84] to-transparent md:w-80" />
      </div>

      {/* barra de progreso + hint */}
      <div className="cut-fade absolute inset-x-10 bottom-10 md:inset-x-16">
        <div className="cut-bar h-[2px] w-full overflow-hidden rounded-full bg-[#161e29]">
          <i className="block h-full w-0 bg-gradient-to-r from-[#3ddc84] to-[#7ee8b0]" />
        </div>
        <div className="mt-2 flex justify-between font-mono text-[10px] text-[#5d6c7d]">
          <span>sincronizando consola…</span>
          <span>click / esc para saltar</span>
        </div>
      </div>

      <div className="cut-flash pointer-events-none absolute inset-0 bg-[#3ddc84] opacity-0" />
    </div>
  )
}
