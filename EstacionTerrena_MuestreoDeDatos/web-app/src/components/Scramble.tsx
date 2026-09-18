/* Título estable de sección. Se conserva la API anterior para no acoplar las
   vistas a una animación decorativa que podía dejar texto ilegible al capturar. */
export function Scramble({ text, className = '', duration = 560 }: {
  text: string
  className?: string
  duration?: number
}) {
  void duration
  return <span className={className}>{text}</span>
}
