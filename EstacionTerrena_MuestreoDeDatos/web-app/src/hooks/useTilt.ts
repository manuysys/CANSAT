/* Tilt de puntero con retorno suave: devuelve handlers para spreadear en un
   elemento. Escribe transform y las vars --glare-x/--glare-y (usadas por
   .tilt-glare) directo en el DOM, sin re-renders. */
import type { MouseEvent } from 'react'

export function useTilt(max = 5) {
  const onMouseMove = (e: MouseEvent<HTMLElement>) => {
    const el = e.currentTarget
    const r = el.getBoundingClientRect()
    const px = (e.clientX - r.left) / r.width - 0.5
    const py = (e.clientY - r.top) / r.height - 0.5
    el.style.transform =
      `perspective(760px) rotateX(${(-py * max).toFixed(2)}deg) rotateY(${(px * max).toFixed(2)}deg) translateY(-2px)`
    el.style.setProperty('--glare-x', `${((px + 0.5) * 100).toFixed(1)}%`)
    el.style.setProperty('--glare-y', `${((py + 0.5) * 100).toFixed(1)}%`)
  }
  const onMouseLeave = (e: MouseEvent<HTMLElement>) => {
    const el = e.currentTarget
    el.style.transition = 'transform .45s cubic-bezier(.22,1,.36,1)'
    el.style.transform = ''
    window.setTimeout(() => { el.style.transition = '' }, 460)
  }
  return { onMouseMove, onMouseLeave }
}
