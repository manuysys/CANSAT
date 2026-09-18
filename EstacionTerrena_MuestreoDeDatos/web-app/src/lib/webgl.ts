/* ¿Hay WebGL disponible? Si no (impresión, VM sin GPU), la escena 3D degrada
   a un perfil SVG del descenso sin romper nada. */
let cached: boolean | null = null

export function webglOK(): boolean {
  if (cached !== null) return cached
  try {
    const c = document.createElement('canvas')
    cached = !!(c.getContext('webgl2') || c.getContext('webgl'))
  } catch {
    cached = false
  }
  return cached
}
