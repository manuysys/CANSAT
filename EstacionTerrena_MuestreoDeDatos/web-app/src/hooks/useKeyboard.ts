/* Atajos globales: ↑↓ navegar · ←→ frame ant./sig. · Enter detalle · A alertas ·
   H solo HIGH · / buscar · E exportar · R sync · P auto · 1-2-3-i vistas · Esc cerrar.
   Con Ctrl/Alt/Meta NO se dispara nada (antes Ctrl+R emitía el sync y además
   recargaba el navegador; Ctrl+P/E secuestraban imprimir y exportar). */
import { useEffect } from 'react'
import { filteredFrames, useMission } from '@/store/mission'
import { exportCSV } from '@/lib/exportCsv'

export function useKeyboard(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const st = useMission.getState()
      const target = e.target as HTMLElement | null
      const typedInput = !!target && (/^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName) || target.isContentEditable)

      // Modo presentación: navegación propia.
      if (st.present.on) {
        const n = filteredFrames(st).length || st.frames.length
        if (e.key === 'Escape') { st.stopPresent(); e.preventDefault() }
        if (e.key === 'ArrowRight' && n) st.setPresentIdx((st.present.idx + 1) % n)
        if (e.key === 'ArrowLeft' && n) st.setPresentIdx((st.present.idx - 1 + n) % n)
        return
      }
      if (e.key === 'Escape') {
        if (typedInput) {
          const input = target as HTMLInputElement
          if (input.id === 'busca-src') st.setQ('')
          input.blur()
        }
        return
      }
      // Los atajos globales no deben pisar combinaciones del navegador ni
      // dispararse mientras el operador escribe.
      if (e.ctrlKey || e.metaKey || e.altKey || typedInput) return
      const list = filteredFrames(st)
      const move = (delta: number) => {
        if (!list.length) return
        let i = list.findIndex(f => f.src === st.selected)
        i = i < 0 ? (delta > 0 ? 0 : list.length - 1) : Math.min(list.length - 1, Math.max(0, i + delta))
        st.select(list[i].src)
        scrollSelectionIntoView(list[i].src)
      }
      if (e.key === 'ArrowDown') { e.preventDefault(); move(1); return }
      if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); return }
      if (e.key === 'ArrowRight') { move(1); return }
      if (e.key === 'ArrowLeft') { move(-1); return }
      if (e.key === 'Enter' && st.selected) {
        e.preventDefault()
        document.getElementById('tour-detalle')?.scrollIntoView({ block: 'nearest' })
        return
      }
      const k = e.key.toLowerCase()
      if (k === 'a') { st.setAlertOnly(!st.filters.alertOnly); return }
      if (k === 'h') {
        const on = st.filters.pri.has('HIGH') && st.filters.pri.size === 1
        useMission.setState({ filters: { ...st.filters, pri: new Set(on ? [] : ['HIGH']) } })
        return
      }
      if (e.key === '/') { e.preventDefault(); document.getElementById('busca-src')?.focus(); return }
      if (k === 'e') { exportCSV(); return }
      if (k === 'r') { window.dispatchEvent(new CustomEvent('lb135-sync')); return }
      if (k === 'p') { st.setAuto(!st.auto); return }
      if (k === '1') st.setView('vuelo')
      if (k === '2') st.setView('post')
      if (k === '3' || k === 'i') st.setView('informe')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}

/** Lleva la selección a la vista en corredor y tabla. */
export function scrollSelectionIntoView(src: string): void {
  const card = document.querySelector(`[data-card-src="${CSS.escape(src)}"]`)
  card?.scrollIntoView({ block: 'nearest' })
  const row = document.querySelector(`tr[data-row-src="${CSS.escape(src)}"]`)
  row?.scrollIntoView({ block: 'nearest' })
}
