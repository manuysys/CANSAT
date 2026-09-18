/* Validación ligera del contrato de datos al ingerir el payload.
   No bloquea nada: genera avisos humanos para el badge del TopBar
   ("datos: ok" / "datos: N avisos") y evita confiar a ciegas en un CSV
   escrito por el pipeline en caliente. */
import type { Frame, Summary } from './types'

export function validarPayload(frames: Frame[], summary: Summary | null): string[] {
  const counts = new Map<string, number>()
  const add = (k: string) => counts.set(k, (counts.get(k) || 0) + 1)
  if (frames.length) {
    let tPrev = -Infinity
    frames.forEach(f => {
      const t = Number(f.t_s)
      const alt = Number(f.alt_m)
      const d = Number(f.danado_pct)
      if (!Number.isFinite(t)) add('t_s no numérico')
      else if (t < tPrev) add('t_s no monótono')
      if (Number.isFinite(t)) tPrev = t
      if (Number.isFinite(alt) && (alt < -50 || alt > 40000)) add('alt_m fuera de rango')
      if (Number.isFinite(d) && (d < 0 || d > 100)) add('danado_pct fuera de 0–100')
      if (!f.src) add('fila sin src')
    })
    if (new Set(frames.map(f => f.src)).size !== frames.length) add('src duplicados')
  }
  if (summary && Number.isFinite(Number(summary.n_frames)) && Number(summary.n_frames) !== frames.length) {
    add(`summary.n_frames (${summary.n_frames}) ≠ frames recibidos (${frames.length})`)
  }
  return [...counts.entries()].map(([k, n]) => (n > 1 ? `${k} ×${n}` : k)).slice(0, 6)
}
