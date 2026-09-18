/* Auto-refresh de 3 s contra /api/mission. El diff lo hace el store (sig):
   si no cambió nada, React no re-renderiza y no hay parpadeo ni re-animaciones. */
import { useEffect } from 'react'
import { fetchMission, fetchSamples } from '@/lib/api'
import { useMission } from '@/store/mission'

export function useFeed(): void {
  useEffect(() => {
    let stopped = false
    let controller: AbortController | null = null
    let requestId = 0
    const load = async () => {
      if (stopped || controller) return
      const currentId = ++requestId
      controller = new AbortController()
      useMission.getState().setSyncing(true)
      try {
        // Misión y muestreo en paralelo: el panel de Muestreo usa el JSONL
        // (uncert, tiempos por etapa, nodata) que no viaja en el CSV.
        const [data, samples] = await Promise.all([
          fetchMission(controller.signal),
          fetchSamples(controller.signal).catch(() => null),
        ])
        if (!stopped && currentId === requestId) {
          useMission.getState().applyPayload(data)
          if (samples?.ok) useMission.getState().setSamples(samples.samples, samples.resumen)
        }
      } catch (error) {
        if (!stopped && currentId === requestId && (error as Error).name !== 'AbortError') {
          useMission.getState().setOffline()
        }
      } finally {
        if (currentId === requestId) {
          controller = null
          useMission.getState().setSyncing(false)
        }
      }
    }
    void load()
    const id = window.setInterval(() => {
      if (useMission.getState().auto && !document.hidden) void load()
    }, 3000)
    // Push SSE: sincroniza al instante cuando el pipeline escribe; el
    // interval de 3 s queda como red de seguridad (y para navegadores sin SSE).
    let es: EventSource | null = null
    if ('EventSource' in window) {
      try {
        es = new EventSource('/api/events')
        es.onmessage = () => { if (useMission.getState().auto && !document.hidden) void load() }
      } catch { /* degradación silenciosa a polling */ }
    }
    const onManual = () => void load()
    const onVis = () => { if (!document.hidden && useMission.getState().auto) void load() }
    document.addEventListener('visibilitychange', onVis)
    window.addEventListener('lb135-sync', onManual)
    return () => {
      stopped = true
      es?.close()
      window.clearInterval(id)
      document.removeEventListener('visibilitychange', onVis)
      window.removeEventListener('lb135-sync', onManual)
      controller?.abort()
    }
  }, [])
}
