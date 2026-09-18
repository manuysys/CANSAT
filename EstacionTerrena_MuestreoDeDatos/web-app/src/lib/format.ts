/* Utilidades de formato tolerantes a nulos (el CSV puede traer huecos). */

export function num(v: number | null | undefined, d = 1, suffix = ''): string {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  const n = Number(v)
  const s = Math.abs(n) >= 1000 ? n.toLocaleString('es-AR', { maximumFractionDigits: d }) : n.toFixed(d)
  return s + suffix
}

export const intOr = (v: number | null | undefined, dflt = 0): number =>
  v === null || v === undefined || Number.isNaN(Number(v)) ? dflt : Math.round(Number(v))

export function mean(vals: Array<number | null | undefined>): number | null {
  const ok = vals.filter(v => v !== null && v !== undefined && !Number.isNaN(Number(v))).map(Number)
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null
}

/** Normaliza sin acentos/mayúsculas; normKey quita además todo signo. */
export const norm = (s: unknown): string =>
  String(s ?? '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toUpperCase().replace(/\s+/g, ' ').trim()

export const normKey = (s: unknown): string => norm(s).replace(/[^A-Z0-9]/g, '')

export const clampPct = (v: number | null | undefined): number => Math.max(0, Math.min(100, Number(v) || 0))

/** Hash djb2: firma barata del payload para detectar cambios sin re-renderizar de más. */
export function hashStr(str: string): string {
  let h = 5381
  for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) >>> 0
  return h.toString(36)
}
