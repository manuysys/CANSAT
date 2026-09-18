/* Espejo TypeScript del schema canónico de `entrega/summary.json`.
 *
 * Fuente de verdad: `cansat_seg_poc/cansat/summary.py` (SCHEMA_VERSION=2).
 * Antes el campo `alertas` estaba tipado como `unknown[]` porque los dos
 * productores (post_flight.py y tools/make_demo_mission.py) escribían schemas
 * INCOMPATIBLES: `list[str]` vs `list[dict]`, y `archivos` con rutas vs con
 * conteos. El backend ahora normaliza (`web_server.normalize_summary`) y acá
 * se tipa el formato normalizado: alertas siempre `Alerta[]`.
 *
 * Si cambia `build_summary`/`Alerta` en `cansat/summary.py`, actualizar este
 * archivo en el mismo commit.
 */

export const SCHEMA_VERSION = 2

export interface Alerta {
  src: string
  t_s?: number | null
  alt_m?: number | null
  diag?: string | null
  danado_pct?: number | null
  sample_pri?: string | null
  motivo?: string | null
}

export interface SummaryArchivos {
  rutas: Record<string, string | null>
  conteos: Record<string, number | null>
}

/** Estimación de pérdidas humanas (DPD). Ver cansat/casualties.py del pipeline. */
export interface Perdidas {
  area_relevada_m2?: number
  area_danada_m2?: number
  personas_expuestas?: number
  personas_afectadas?: number
  perdidas_estimadas?: number
  perdidas_min?: number
  perdidas_max?: number
  supuestos?: Record<string, number>
  n_frames_con_dano?: number
  nota?: string
}

export interface Summary {
  schema_version?: number
  mision?: string
  n_frames?: number
  alt_max_m?: number | null
  alt_min_m?: number | null
  veredictos?: Record<string, number>
  personas_total?: number
  vehiculos_total?: number
  alertas?: Alerta[]
  danado_pct_por_frame?: Record<string, number>
  danado_pct_prom?: number | null
  terrain_b5_por_frame?: Record<string, Record<string, number>>
  perdidas?: Perdidas
  archivos?: SummaryArchivos
  generado?: string
  nota?: string
  [extra: string]: unknown
}
