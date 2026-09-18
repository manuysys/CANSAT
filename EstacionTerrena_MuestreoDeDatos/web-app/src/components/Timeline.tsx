/* Timeline colapsable: filtros animados, orden por columna y filas con hover
   lift + layout. Mantiene el teclado y el export CSV de la v2. */
import { useMemo } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowDownUp, ChevronDown, Search, Table2, X } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { filteredFrames, useMission } from '@/store/mission'
import { exportCSV } from '@/lib/exportCsv'
import { clampPct, intOr, num } from '@/lib/format'
import { diagTone, priColor } from '@/lib/vocab'
import type { SamplePri, SortKey } from '@/lib/types'

const PRI_CHIPS: SamplePri[] = ['HIGH', 'MEDIUM', 'LOW']

function SortHead({ k, label, num: isNum }: { k: SortKey; label: string; num?: boolean }) {
  const sort = useMission(s => s.sort)
  const toggleSort = useMission(s => s.toggleSort)
  const active = sort.key === k
  return (
    <TableHead className={isNum ? 'text-right' : ''}>
      <button
        type="button"
        onClick={() => toggleSort(k)}
        className={`inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-[0.12em] transition-colors ${active ? 'text-[#3ddc84]' : 'text-muted-foreground hover:text-foreground'}`}
        title={`Ordenar por ${label}`}
      >
        {label}
        <ArrowDownUp className={`size-3 ${active ? (sort.dir === -1 ? 'rotate-180' : '') : 'opacity-40'}`} />
      </button>
    </TableHead>
  )
}

export function Timeline() {
  const frames = useMission(s => s.frames)
  const filters = useMission(s => s.filters)
  const sort = useMission(s => s.sort)
  const selected = useMission(s => s.selected)
  const select = useMission(s => s.select)
  const collapsed = useMission(s => s.tableCollapsed)
  const toggleCollapsed = useMission(s => s.toggleTableCollapsed)
  const { togglePri, setAlertOnly, setDiag, setVerdict, setQ, clearFilters } = useMission.getState()

  const rows = useMemo(() => filteredFrames({ frames, filters, sort }), [frames, filters, sort])
  const diagOpts = useMemo(() => [...new Set(frames.map(f => f.diag).filter((v): v is string => !!v))].sort(), [frames])
  const verdictOpts = useMemo(() => [...new Set(frames.map(f => f.verdict).filter((v): v is string => !!v))].sort(), [frames])
  const filtering = filters.pri.size > 0 || filters.alertOnly || !!filters.diag || !!filters.verdict || !!filters.q.trim()

  return (
    <Card id="tour-tabla" className="border-border/60 bg-card/60">
      <div className="flex items-center justify-between gap-3 px-4 py-2">
        <button type="button" onClick={toggleCollapsed} aria-expanded={!collapsed} className="flex items-center gap-2">
          <ChevronDown className={`size-4 text-muted-foreground transition-transform ${collapsed ? '-rotate-90' : ''}`} />
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Timeline de frames</h2>
          {filtering && <span className="font-mono text-[10.5px] text-[#3ddc84]">{rows.length} de {frames.length}</span>}
        </button>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={() => exportCSV()} title="Exportar el filtro activo (E)">
            Exportar CSV
          </Button>
          <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={() => clearFilters()} disabled={!filtering}>
            <X className="size-3.5" /> Limpiar
          </Button>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.24, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            {/* Filtros */}
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-y border-border/50 bg-muted/15 px-4 py-2.5">
              <div className="flex items-center gap-2">
                <span className="text-[10px] uppercase tracking-[0.1em] text-muted-foreground">Prioridad</span>
                <div className="flex gap-1.5">
                  {PRI_CHIPS.map(p => {
                    const on = filters.pri.has(p)
                    return (
                      <motion.button
                        key={p} type="button" whileTap={{ scale: 0.94 }}
                        onClick={() => togglePri(p)}
                        aria-pressed={on}
                        className={`rounded-full border px-3 py-1 text-[11px] font-semibold transition-colors ${on ? 'border-transparent text-[#0b0e14]' : 'border-border/70 text-muted-foreground hover:text-foreground'}`}
                        style={on ? { background: priColor(p) } : undefined}
                      >
                        {p.charAt(0) + p.slice(1).toLowerCase()}
                      </motion.button>
                    )
                  })}
                </div>
              </div>

              <label className="flex items-center gap-2 text-xs text-muted-foreground" title="Solo frames con alert=1 o diagnóstico de desastre (A)">
                <Switch checked={filters.alertOnly} onCheckedChange={setAlertOnly} className="scale-90" />
                solo alertas
              </label>

              <Select value={filters.diag || '__all'} onValueChange={v => setDiag(v === '__all' ? '' : v)}>
                <SelectTrigger className="h-8 w-[190px] text-xs"><SelectValue placeholder="diag: todos" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all">diag: todos</SelectItem>
                  {diagOpts.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                </SelectContent>
              </Select>

              <Select value={filters.verdict || '__all'} onValueChange={v => setVerdict(v === '__all' ? '' : v)}>
                <SelectTrigger className="h-8 w-[190px] text-xs"><SelectValue placeholder="veredicto: todos" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all">veredicto: todos</SelectItem>
                  {verdictOpts.map(v => <SelectItem key={v} value={v}>{v}</SelectItem>)}
                </SelectContent>
              </Select>

              <div className="relative min-w-[180px] flex-1">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="busca-src"
                  value={filters.q}
                  onChange={e => setQ(e.target.value)}
                  placeholder="buscar src…  ( / )"
                  className="h-8 pl-8 text-xs"
                />
              </div>
            </div>

            {/* Tabla */}
            <div className="max-h-[300px] overflow-auto">
              <Table className="text-xs">
                <TableHeader className="sticky top-0 z-10 bg-[#121a24]">
                  <TableRow className="hover:bg-transparent">
                    <SortHead k="t_s" label="t s" num />
                    <TableHead>src</TableHead>
                    <SortHead k="alt_m" label="alt m" num />
                    <TableHead>diag</TableHead>
                    <SortHead k="danado_pct" label="daño" num />
                    <TableHead className="text-center" >pri</TableHead>
                    <SortHead k="sample_score" label="score" num />
                    <TableHead className="text-center">!</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <AnimatePresence initial={false}>
                    {rows.map(f => {
                      const pri = String(f.sample_pri ?? 'LOW')
                      const d = Number(f.danado_pct) || 0
                      const tone = diagTone(f.diag)
                      return (
                        <motion.tr
                          key={f.src}
                          data-row-src={f.src}
                          layout="position"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.18 }}
                          whileHover={{ y: -1, backgroundColor: 'rgba(255,255,255,0.03)' }}
                          onClick={() => select(f.src)}
                          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(f.src) } }}
                          tabIndex={0}
                          aria-selected={f.src === selected}
                          className={`cursor-pointer border-b border-border/40 ${f.src === selected ? 'bg-[#12241b]' : ''}`}
                        >
                          <TableCell className="py-1.5 text-right font-mono tabular-nums text-muted-foreground">{num(f.t_s, 1)}</TableCell>
                          <TableCell className="py-1.5 font-mono font-medium text-foreground/95">{f.src}</TableCell>
                          <TableCell className="py-1.5 text-right font-mono tabular-nums text-muted-foreground">{num(f.alt_m, 1)}</TableCell>
                          <TableCell className="py-1.5">
                            <span className={`inline-block max-w-[230px] truncate rounded-full border px-2 py-0.5 text-[10.5px] font-medium ${tone === 'bad' ? 'border-[#ff4d5e]/35 bg-[#ff4d5e]/8 text-[#ffc2c8]' : tone === 'warn' ? 'border-[#ffb020]/35 bg-[#ffb020]/8 text-[#ffe3b0]' : 'border-border/60 bg-muted/30 text-muted-foreground'}`}>
                              {f.diag || '—'}
                            </span>
                          </TableCell>
                          <TableCell className="py-1.5 text-right">
                            <span className="inline-flex items-center gap-2 font-mono tabular-nums">
                              <span className="h-1.5 w-14 overflow-hidden rounded-full bg-muted/40">
                                <span className="block h-full" style={{ width: `${clampPct(d)}%`, background: d >= 40 ? '#ff4d5e' : d >= 15 ? '#ffb020' : '#6b7a8c' }} />
                              </span>
                              {num(f.danado_pct, 1)}
                            </span>
                          </TableCell>
                          <TableCell className="py-1.5 text-center">
                            <span className="inline-block size-2.5 rounded-full" style={{ background: priColor(pri) }} title={pri} />
                          </TableCell>
                          <TableCell className="py-1.5 text-right font-mono tabular-nums text-muted-foreground">{num(f.sample_score, 2)}</TableCell>
                          <TableCell className="py-1.5 text-center">
                            {intOr(f.alert) === 1 && <span className="inline-block size-2 rounded-full bg-[#ff4d5e] shadow-[0_0_0_3px_rgba(255,77,94,0.18)]" title="alert=1" />}
                          </TableCell>
                        </motion.tr>
                      )
                    })}
                  </AnimatePresence>
                  {!rows.length && (
                    <TableRow>
                      <TableCell colSpan={8} className="py-8 text-center">
                        <Table2 className="mx-auto mb-2 size-5 text-muted-foreground/60" />
                        <p className="text-xs text-muted-foreground">
                          {frames.length ? 'Ningún frame pasa el filtro actual.' : 'Sin telemetría: esperando outputs/mission/telemetry.csv.'}
                        </p>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  )
}
