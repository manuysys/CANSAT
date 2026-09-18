/* Capa de red: misma API de web_server.py, sin cambios. */
import type { MissionPayload, SamplesPayload } from './types'

export async function fetchMission(signal?: AbortSignal): Promise<MissionPayload> {
  const res = await fetch('/api/mission', { cache: 'no-store', signal })
  if (!res.ok) throw new Error('HTTP ' + res.status)
  return (await res.json()) as MissionPayload
}

/** Campos por frame que sólo viven en el JSONL (incertidumbre, tiempos, nodata)
 *  + agregados del panel de Muestreo. */
export async function fetchSamples(signal?: AbortSignal): Promise<SamplesPayload> {
  const res = await fetch('/api/samples', { cache: 'no-store', signal })
  if (!res.ok) throw new Error('HTTP ' + res.status)
  return (await res.json()) as SamplesPayload
}

/** Ruta relativa del proyecto -> URL del proxy de imágenes del servidor. */
export const imgURL = (rel: string | null | undefined): string | null =>
  rel ? '/img/' + String(rel).split('/').map(encodeURIComponent).join('/') : null
