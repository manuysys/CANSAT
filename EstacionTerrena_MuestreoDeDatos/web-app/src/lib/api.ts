/* Capa de red: misma API de web_server.py, sin cambios. */
import type { ConsultaResult, GradcamModelo, GradcamResult, MissionPayload, SamplesPayload } from './types'

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

/** Consulta Terrestre simbólica. `zona` es un polígono [[lon,lat],…] opcional. */
export async function fetchConsulta(
  q: string,
  zona?: [number, number][] | null,
  signal?: AbortSignal,
): Promise<ConsultaResult> {
  const params = new URLSearchParams({ q })
  if (zona && zona.length >= 3) params.set('poly', JSON.stringify(zona))
  const res = await fetch('/api/consulta?' + params.toString(), { cache: 'no-store', signal })
  if (!res.ok) throw new Error('HTTP ' + res.status)
  return (await res.json()) as ConsultaResult
}

/** Grad-CAM: qué regiones del frame justifican el daño predicho.
 *  Devuelve el JSON incluso en 404 (error claro si falta torch/checkpoint). */
export async function fetchGradcam(
  src: string,
  modelo: GradcamModelo,
  signal?: AbortSignal,
): Promise<GradcamResult> {
  const params = new URLSearchParams({ src, modelo })
  const res = await fetch('/api/gradcam?' + params.toString(), { cache: 'no-store', signal })
  return (await res.json()) as GradcamResult
}

/** Ruta relativa del proyecto -> URL del proxy de imágenes del servidor. */
export const imgURL = (rel: string | null | undefined): string | null =>
  rel ? '/img/' + String(rel).split('/').map(encodeURIComponent).join('/') : null
