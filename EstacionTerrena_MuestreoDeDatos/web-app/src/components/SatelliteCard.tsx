/* Capa satelital OPCIONAL (V11 4.5): contexto ESRI World Imagery del centro de
 * la trayectoria. La estación es offline-first: se pide SOLO al hacer click y
 * si no hay red muestra un aviso (no rompe nada). La imagen queda cacheada en
 * el server (outputs/satellite) y a partir de ahí también sirve offline. */
import { useMemo, useState } from 'react'
import { Satellite } from 'lucide-react'
import { useMission } from '@/store/mission'
import { gpsDe } from '@/components/Sampling'

type Estado = 'idle' | 'cargando' | 'ok' | 'sin-red'

export function SatelliteCard() {
  const frames = useMission(s => s.frames)
  const [img, setImg] = useState<string | null>(null)
  const [estado, setEstado] = useState<Estado>('idle')
  const [fuente, setFuente] = useState('')

  const centro = useMemo(() => {
    const pts = gpsDe(frames)
    if (!pts.length) return null
    return {
      lat: pts.reduce((a, p) => a + p.lat, 0) / pts.length,
      lon: pts.reduce((a, p) => a + p.lon, 0) / pts.length,
    }
  }, [frames])

  if (!centro) return null

  const cargar = async () => {
    setEstado('cargando')
    try {
      const r = await fetch(`/api/satellite?lat=${centro.lat}&lon=${centro.lon}`)
      const j = await r.json()
      if (j.ok) {
        setImg(`/img/${j.img}`)
        setFuente(`${j.fuente}${j.cache ? ' · caché local' : ''}`)
        setEstado('ok')
      } else {
        setEstado('sin-red')
      }
    } catch {
      setEstado('sin-red')
    }
  }

  return (
    <div className="border-t border-border/50 px-5 py-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10.5px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Contexto satelital (opcional)
        </span>
        {estado !== 'ok' && (
          <button
            onClick={cargar}
            disabled={estado === 'cargando'}
            className="rounded border border-border/60 px-2 py-1 text-[10.5px] text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
          >
            <Satellite className="mr-1 inline size-3" />
            {estado === 'cargando' ? 'Cargando…' : 'Pedir imagen (requiere red)'}
          </button>
        )}
      </div>
      {estado === 'ok' && img && (
        <figure className="mt-2">
          <img
            src={img}
            alt="Contexto satelital del predio"
            className="w-full rounded-lg border border-border/50"
          />
          <figcaption className="mt-1 font-mono text-[10px] text-muted-foreground/70">
            {fuente} · centro {centro.lat.toFixed(5)}, {centro.lon.toFixed(5)} ·
            el resto de la estación sigue offline
          </figcaption>
        </figure>
      )}
      {estado === 'sin-red' && (
        <p className="mt-2 text-[10.5px] text-muted-foreground/70">
          Sin red o tiles no disponibles: la capa satelital es opcional; el
          resto de la estación funciona offline.
        </p>
      )}
    </div>
  )
}
