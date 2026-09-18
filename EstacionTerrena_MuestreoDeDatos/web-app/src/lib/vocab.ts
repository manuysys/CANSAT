/* Vocabulario del contrato + glosario de una línea para tooltips. */
import { normKey } from './format'
import type { Frame, ImgKey } from './types'

export const TERRAIN = [
  { key: 'veg',  label: 'Vegetación',     en: 'vegetation',  hex: '#2fbf74' },
  { key: 'bui',  label: 'Edificios',      en: 'building',    hex: '#e08c3a' },
  { key: 'wat',  label: 'Agua',           en: 'water',       hex: '#3f8fd1' },
  { key: 'bare', label: 'Suelo expuesto', en: 'bare_ground', hex: '#b98a5e' },
  { key: 'oth',  label: 'Otro',           en: 'other',       hex: '#7d8a99' },
] as const

export type TerrainKey = (typeof TERRAIN)[number]['key']

export const PRI_ORDER = ['HIGH', 'MEDIUM', 'LOW'] as const

/** Color semántico por prioridad del sampler. */
export const priColor = (p: string | null | undefined): string =>
  p === 'HIGH' ? '#ff4d5e' : p === 'MEDIUM' ? '#ffb020' : '#6b7a8c'

const VERDICT_CLASS: Record<string, string> = {
  ZONASALUDABLE: 'v-ok',
  ESTRESMODERADO: 'v-mod',
  ALTOESTRESURBANO: 'v-high',
  SUELOEXPUESTO: 'v-bare',
}
export const verdictClass = (v: string | null | undefined): string => VERDICT_CLASS[normKey(v)] ?? ''

export const verdictColor = (v: string | null | undefined): string => {
  const c = verdictClass(v)
  return c === 'v-ok' ? '#3ddc84' : c === 'v-mod' ? '#ffb020' : c === 'v-high' ? '#ff4d5e' : c === 'v-bare' ? '#b98a5e' : '#8b9aab'
}

const DIAG_SEV: Record<string, number> = {
  SINDESASTRE: 0,
  AGUAEXTENSALAGORIO: 1,
  POSIBLEINCENDIOEROSION: 2,
  POSIBLESISMOVIENTO: 3,
  INUNDACIONURBANA: 3,
  INUNDACIONSEVERA: 4,
}
export const diagSev = (d: string | null | undefined): number => {
  const k = normKey(d)
  return k in DIAG_SEV ? DIAG_SEV[k] : d ? 1 : 0
}
export const diagTone = (d: string | null | undefined): 'none' | 'warn' | 'bad' => {
  const s = diagSev(d)
  return s === 0 ? 'none' : s >= 3 ? 'bad' : 'warn'
}

/** Un frame requiere atención si el pipeline lo marcó o diagnosticó un evento. */
export const isAlertFrame = (f: Pick<Frame, 'alert' | 'diag'>): boolean =>
  (Number(f.alert) || 0) === 1 || diagSev(f.diag) >= 2

/** Tabs de imagen del detalle, en orden de presentación. */
export const IMG_TABS: Array<{ k: ImgKey; label: string }> = [
  { k: 'vis', label: 'Evidencia' },
  { k: 'ens_seg', label: 'Ensemble' },
  { k: 'enhanced', label: 'EDSR' },
  { k: 'high_res', label: 'High-res' },
]

/** Glosario de una línea para los tooltips. */
export const GLOSSARY: Record<string, string> = {
  usi: 'Índice de estrés urbano: carga antrópica ponderada de la superficie (0 = natural, 1 = todo construido).',
  ndvi: 'Verdor relativo: diferencia normalizada entre vegetación y suelo expuesto (-1 a 1).',
  sampler: 'Sampler adaptativo a bordo: puntúa cada frame (0 a 1) y decide en qué prioridad guardarlo.',
  pri: 'Prioridad de guardado del frame: HIGH (rojo), MEDIUM (ámbar) o LOW (gris).',
  danado: 'Porcentaje de tejido construido con daño estimado por consenso de 3 modelos.',
  sharp: 'Nitidez del frame (respuesta de bordes): baja con nubosidad o movimiento.',
  SINDESASTRE: 'Sin daño estructural detectado por el consenso de modelos.',
  AGUAEXTENSALAGORIO: 'Agua extensa de origen natural: lago o río, no una inundación.',
  INUNDACIONURBANA: 'Agua sobre tejido urbano: calles y manzanas anegadas.',
  INUNDACIONSEVERA: 'Inundación mayor con daño estructural alto en zona urbana.',
  POSIBLESISMOVIENTO: 'Patrón de colapso compatible con sismo o viento extremo.',
  POSIBLEINCENDIOEROSION: 'Superficie quemada o erosionada detectada en el frame.',
}

/** Etiqueta legible de la clase dominante de un frame. */
export function domLabel(f: Frame): string {
  const d = f.dom
  if (d !== null && d !== undefined && d !== '') {
    if (typeof d === 'number' || /^\d+$/.test(String(d).trim())) {
      const t = TERRAIN[Number(d)]
      if (t) return t.label
    } else {
      const k = normKey(d)
      const t = TERRAIN.find(x => normKey(x.en) === k || normKey(x.label) === k || normKey(x.key) === k)
      if (t) return t.label
      return String(d)
    }
  }
  const best = [...TERRAIN].sort((a, b) => (Number(f[b.key]) || 0) - (Number(f[a.key]) || 0))[0]
  return best ? best.label : '—'
}
