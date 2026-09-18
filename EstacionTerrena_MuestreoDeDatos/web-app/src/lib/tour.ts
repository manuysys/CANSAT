/* Tour de primera vez con driver.js: 5 pasos, una sola vez (localStorage).
   El botón "?" del header lo reabre cuando el operador lo pida.
   Nota: driver.js v1 espera el contenido dentro de `popover: {…}`. */
import { driver, type Driver } from 'driver.js'
import 'driver.js/dist/driver.css'

const KEY = 'lb135-tour-v4'
let instance: Driver | null = null

const STEPS = [
  { element: '#tour-corredor', title: 'Corredor de descenso', text: 'Cada tarjeta es un frame ordenado por altitud (250 → 0 m). El borde indica la prioridad y el anillo rojo, una alerta.' },
  { element: '#tour-detalle', title: 'Detalle del frame', text: 'El protagonista: imagen grande con tabs (Evidencia, Ensemble, EDSR, High-res) y todas sus métricas debajo.' },
  { element: '#tour-alertas', title: 'Alertas', text: 'Solo aparecen si la misión vio algo: frames con alert=1 o diagnóstico de desastre, ordenados por severidad.' },
  { element: '#tour-tabla', title: 'Timeline', text: 'Tabla colapsable con filtros, orden y búsqueda. Exporta el filtro activo a CSV con E o el botón.' },
  { element: '#tour-informe', title: 'Informe', text: 'La vista imprimible para el jurado: KPIs, alertas, top-5 HIGH, corredor y firmas.' },
]

export function tourDone(): boolean {
  return localStorage.getItem(KEY) === '1'
}

export function startTour(): void {
  const steps = STEPS
    .map(s => ({ ...s }))
    .filter(s => document.querySelector(s.element))
    .map(s => ({
      element: s.element,
      popover: { title: s.title, description: s.text },
    }))
  if (!steps.length) return
  instance?.destroy()
  instance = driver({
    showProgress: true,
    animate: true,
    smoothScroll: true,
    allowClose: true,
    nextBtnText: 'Siguiente',
    prevBtnText: 'Anterior',
    doneBtnText: 'Listo',
    popoverClass: 'lb135-popover',
    steps,
    onDestroyStarted: () => {
      localStorage.setItem(KEY, '1')
      instance?.destroy()
    },
  })
  window.setTimeout(() => instance?.drive(), 250)
}
