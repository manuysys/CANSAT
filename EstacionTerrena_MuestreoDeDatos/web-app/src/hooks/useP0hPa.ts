/* Hook de calibración altimétrica: presión a nivel del mar (hPa).
   Carga p0 desde el servidor en mount (si hay red); si no, queda el
   valor de localStorage / 1013.25 del store. */
import { useEffect } from 'react'
import { fetchP0hPa, updateP0hPa } from '@/lib/api'
import { useMission } from '@/store/mission'

export function useP0hPa() {
  const p0 = useMission(st => st.p0_hPa)
  const apply = useMission(st => st.updateP0)

  useEffect(() => {
    let ignore = false
    fetchP0hPa()
      .then(val => { if (!ignore) apply(val) })
      .catch(() => { /* conserva el valor local */ })
    return () => { ignore = true }
  }, [apply])

  /** Actualiza p0: escribe store + localStorage + server. */
  async function push(val: number) {
    apply(val)
    await updateP0hPa(val).catch(() => { /* servidor no disponible */ })
  }

  return { p0, push }
}
