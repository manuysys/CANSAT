/* Bitácora de eventos: registro automático de la misión (fases, apogeo,
   alertas, aterrizaje, enlace, post-vuelo). Se llena sola desde el diff del
   store y se exporta a JSON para el informe post-vuelo. */
import { useState } from 'react'
import { ChevronDown, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Scramble } from '@/components/Scramble'
import { useMission, type Evento } from '@/store/mission'

const TONE: Record<Evento['tipo'], string> = {
  frame: 'text-[#7db8e8]',
  alerta: 'text-[#ff8f9b]',
  fase: 'text-[#7ee8b0]',
  enlace: 'text-[#ffb020]',
  post: 'text-[#d8b48a]',
}

export function Bitacora() {
  const eventos = useMission(s => s.eventos)
  const mision = useMission(s => s.summary?.mision)
  const pushToast = useMission(s => s.pushToast)
  const [open, setOpen] = useState(true)

  const exportar = () => {
    const blob = new Blob([JSON.stringify({ mision: mision || 'LB135', exportado: new Date().toISOString(), eventos }, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `bitacora_${mision || 'LB135'}.json`
    a.click()
    URL.revokeObjectURL(a.href)
    pushToast(`Bitácora exportada (${eventos.length} eventos)`)
  }

  return (
    <Card className="border-border/60 bg-card/60">
      <div className="flex items-center justify-between gap-3 border-b border-border/50 px-5 py-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          <Scramble text="Bitácora de eventos" />
        </h2>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10.5px] text-muted-foreground/70">{eventos.length} eventos</span>
          <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" onClick={exportar} disabled={!eventos.length} title="Descargar el registro en JSON">
            <Download className="size-3" /> Exportar
          </Button>
          <Button variant="ghost" size="icon" className="size-7" title={open ? 'Colapsar' : 'Expandir'} onClick={() => setOpen(o => !o)}>
            <ChevronDown className={`size-4 transition-transform ${open ? '' : '-rotate-90'}`} />
          </Button>
        </div>
      </div>
      {open && (
        <div className="max-h-44 overflow-y-auto px-5 py-3 font-mono text-[10.5px] leading-relaxed">
          {eventos.length ? (
            eventos.slice().reverse().map(e => (
              <div key={e.id} className="flex gap-3">
                <span className="shrink-0 text-muted-foreground/60">{e.wall}</span>
                <span className="shrink-0 text-muted-foreground/60">t {e.t}</span>
                <span className={`w-14 shrink-0 uppercase tracking-[0.08em] ${TONE[e.tipo]}`}>{e.tipo}</span>
                <span className="text-foreground/85">{e.msg}</span>
              </div>
            ))
          ) : (
            <span className="text-muted-foreground">
              sin eventos todavía — el registro se llena solo con la misión (fases, apogeo, alertas, aterrizaje, enlace).
            </span>
          )}
        </div>
      )}
    </Card>
  )
}
