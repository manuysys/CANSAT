/* Fondo ambiental fijo: aurora de misión, grilla HUD, scanline, ruido y viñeta.
   Vive detrás de todo (-z-10) y no intercepta punteros. */
export function Backdrop() {
  return (
    <div aria-hidden data-backdrop className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="bg-aurora absolute -inset-[18%]" />
      <div className="bg-grid absolute inset-0" />
      <div className="bg-scan absolute inset-0" />
      <div className="bg-noise absolute inset-0" />
      <div className="bg-vignette absolute inset-0" />
    </div>
  )
}
