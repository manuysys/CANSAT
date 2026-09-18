/* Exporta el filtro activo a CSV (BOM UTF-8 para Excel). */
import { filteredFrames, useMission } from '@/store/mission'

export function exportCSV(): void {
  const rows = filteredFrames(useMission.getState())
  if (!rows.length) return
  const cols = Object.keys(rows[0]).filter(c => !c.startsWith('_') && c !== 'files')
  const q = (v: unknown): string => {
    if (v === null || v === undefined) return ''
    const s = String(v)
    return /[",\n;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s
  }
  const csv = [cols.join(',')].concat(rows.map(r => cols.map(c => q(r[c])).join(','))).join('\r\n')
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `LB135_filtro_${rows.length}frames_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.csv`
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1500)
  useMission.getState().pushToast(`CSV exportado (${rows.length} filas del filtro activo)`)
}
