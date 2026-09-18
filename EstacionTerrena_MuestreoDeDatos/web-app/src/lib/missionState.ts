/* Fase de la misión inferida de la telemetría (sin hardware extra):
   sin-telemetría → ascenso → descenso → aterrizado → post-vuelo.
   Más apogeo y velocidad vertical para el tape/variómetro. */
import type { Frame, Summary } from './types'

export type PhaseId = 'sin-tlm' | 'ascenso' | 'descenso' | 'aterrizado' | 'post-vuelo'

export interface Phase {
  id: PhaseId
  label: string
  color: string
}

export function missionPhase(frames: Frame[], summary: Summary | null): Phase {
  if (!frames.length) return { id: 'sin-tlm', label: 'sin telemetría', color: '#6b7a8c' }
  const alts = frames.map(f => Number(f.alt_m) || 0)
  const last = alts[alts.length - 1]
  if (last <= 3) {
    return summary
      ? { id: 'post-vuelo', label: 'post-vuelo', color: '#3f8fd1' }
      : { id: 'aterrizado', label: 'aterrizado', color: '#ffb020' }
  }
  if (alts.length >= 2 && last > alts[alts.length - 2]) {
    return { id: 'ascenso', label: 'ascenso', color: '#7ee8b0' }
  }
  if (alts.length < 3) return { id: 'ascenso', label: 'ascenso', color: '#7ee8b0' }
  return { id: 'descenso', label: 'descenso', color: '#3ddc84' }
}

export function apogeo(frames: Frame[]): { src: string; alt: number; idx: number } | null {
  if (!frames.length) return null
  let bi = 0
  frames.forEach((f, i) => {
    if ((Number(f.alt_m) || 0) > (Number(frames[bi].alt_m) || 0)) bi = i
  })
  return { src: frames[bi].src, alt: Number(frames[bi].alt_m) || 0, idx: bi }
}

/** m/s entre los dos últimos frames (negativo = bajando). */
export function verticalSpeed(frames: Frame[]): number | null {
  const n = frames.length
  if (n < 2) return null
  const a = frames[n - 2]
  const b = frames[n - 1]
  const dt = (Number(b.t_s) || 0) - (Number(a.t_s) || 0)
  if (dt <= 0) return null
  return ((Number(b.alt_m) || 0) - (Number(a.alt_m) || 0)) / dt
}

/** Tacha la tasa vertical como anómala para un descenso con paracaídas. */
export function tasaAnomala(phase: PhaseId, vs: number | null): boolean {
  if (vs === null || phase !== 'descenso') return false
  return vs < -12 || vs > -1.0
}
