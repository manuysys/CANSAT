/* Contrato de datos de la misión (solo lectura), tipado para el frontend. */

import type { Summary } from './summarySchema'

export type { Alerta, Summary, SummaryArchivos } from './summarySchema'

export type SamplePri = 'HIGH' | 'MEDIUM' | 'LOW'
/* Buckets que la UI ofrece como TABS del detalle (subconjunto de FrameFiles).
 * ⚠ No es lo mismo que FrameFiles: `full_res` y `thumb` existen en el contrato
 *   y en el servidor pero no se muestran como tabs (serían 6 tabs). Si algún
 *   día se quieren ofrecer, ampliar acá Y en vocab.ts IMG_TABS, y actualizar
 *   smoke-v4.mjs (hoy espera 4 tabs en cap_0000 y 2 en cap_0001). */
export type ImgKey = 'vis' | 'ens_seg' | 'enhanced' | 'high_res'
export type ViewId = 'vuelo' | 'post' | 'informe'
export type SortKey = 't_s' | 'alt_m' | 'danado_pct' | 'sample_score'

export interface FrameFiles {
  vis: string | null
  high_res: string | null
  full_res: string | null
  thumb: string | null
  ens_seg: string | null
  enhanced: string | null
}

export interface Frame {
  t_s: number | null
  alt_m: number | null
  p_hpa: number | null
  temp_c: number | null
  veg: number | null
  bui: number | null
  wat: number | null
  bare: number | null
  oth: number | null
  dom: number | string | null
  usi: number | null
  ndvi: number | null
  verdict: string | null
  people: number | null
  vehicles: number | null
  danado_pct: number | null
  aff_m2: number | null
  diag: string | null
  alert: number | null
  sharp: number | null
  src: string
  sample_pri: SamplePri | string | null
  sample_score: number | null
  /* Extensiones DPD */
  lat?: number | null
  lon?: number | null
  hum_pct?: number | null
  area_m2?: number | null
  personas_afectadas?: number | null
  perdidas_est?: number | null
  /* F3: fuego/humo */
  fire_pct?: number | null
  smoke_pct?: number | null
  /* Estrés ambiental: bruma por imagen + calor por sensores */
  haze_pct?: number | null
  humidex?: number | null
  stress_idx?: number | null
  /* Severidad: fracción de colapso medida (null = supuesto) */
  colapso_pct?: number | null
  _idx: number
  files?: FrameFiles
  [extra: string]: unknown
}

/** Un frame del JSONL: campos que no están en el CSV. */
export interface SampleExtra {
  uncert?: number | null
  ms_seg?: number | null
  ms_dmg?: number | null
  ms_total?: number | null
  nodata_pct?: number | null
  danado2_edif_pct?: number | null
  sharp_ok?: boolean | null
  area_m2?: number | null
  supuestos?: Record<string, number> | null
  /* Contrato v3: trazabilidad de fuentes y modelos por frame */
  colapso_fuente?: string | null
  pop_fuente?: string | null
  ocupacion_fuente?: string | null
  tipo_desastre?: string | null
  tipo_conf?: number | null
  model_ids?: Record<string, string> | null
  quant?: string | null
}

export interface SamplesResumen {
  n_frames?: number
  ms_total_mediana?: number | null
  ms_total_p95?: number | null
  uncert_p50?: number | null
  uncert_p95?: number | null
  nodata_p95?: number | null
  prioridad?: Record<string, number>
  n_borrosos?: number
}

export interface SamplesPayload {
  ok: boolean
  samples: Record<string, SampleExtra>
  resumen: SamplesResumen
}

export interface MissionPayload {
  ok: boolean
  n_frames: number
  frames: Frame[]
  summary: Summary | null
  assets: { corridor_map: string | null }
  buckets: Record<string, boolean>
}
