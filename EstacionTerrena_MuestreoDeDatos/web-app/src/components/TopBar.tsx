/* Header sticky con glass: identidad, tabs de vista con indicador layoutId,
   estado de enlace, reloj y acciones mínimas. */
import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { Clapperboard, GraduationCap, HelpCircle, RefreshCw, Satellite, SunMedium, Moon } from 'lucide-react'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { useMission } from '@/store/mission'
import { startTour } from '@/lib/tour'
import { missionPhase } from '@/lib/missionState'
import type { ViewId } from '@/lib/types'

const VIEWS: Array<{ id: ViewId; label: string }> = [
  { id: 'vuelo', label: 'Vuelo' },
  { id: 'post', label: 'Post-vuelo' },
  { id: 'informe', label: 'Informe' },
]

function Clock() {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000)
    return () => window.clearInterval(id)
  }, [])
  return (
    <span className="font-mono text-xs tabular-nums text-muted-foreground">
      {now.toLocaleTimeString('es-AR', { hour12: false })}
    </span>
  )
}

function SyncDot() {
  const status = useMission(s => s.status)
  const syncing = useMission(s => s.syncing)
  const lastSync = useMission(s => s.lastSync)
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])
  const secs = lastSync ? Math.max(0, Math.round((now - lastSync) / 1000)) : null
  const color = status === 'live' ? 'bg-[#3ddc84]' : status === 'offline' ? 'bg-[#ff4d5e]' : 'bg-[#ffb020]'
  const title = status === 'live' ? 'Enlace activo con el servidor'
    : status === 'offline' ? 'Sin contacto con el servidor' : 'Servidor accesible, sin telemetría'
  return (
    <span className="sync-status flex items-center gap-2 text-[11px] text-muted-foreground" title={title} aria-label={title}>
      <span className="relative flex size-2">
        {syncing && <span className={`absolute inline-flex size-full animate-ping rounded-full opacity-60 ${color}`} />}
        <span className={`relative inline-flex size-2 rounded-full ${color}`} />
      </span>
      <span className="hidden sm:inline">{secs === null ? 'sin sync' : `sync ${secs}s`}</span>
    </span>
  )
}

function PhaseChip() {
  const frames = useMission(s => s.frames)
  const summary = useMission(s => s.summary)
  const ph = missionPhase(frames, summary)
  return (
    <span
      className="flex items-center gap-1.5 rounded-full border border-border/50 bg-muted/30 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em]"
      style={{ color: ph.color }}
      title="Fase de misión inferida de la telemetría (altitud y tendencia)"
    >
      <span className="size-1.5 rounded-full" style={{ background: ph.color }} />
      {ph.label}
    </span>
  )
}

function SolToggle() {
  const sol = useMission(s => s.sol)
  const setSol = useMission(s => s.setSol)
  const toggle = () => setSol(!sol)
  return (
    <Button variant="ghost" size="icon" className="size-8" title={sol ? 'Volver al tema oscuro de consola' : 'Modo sol de campo (alto contraste)'} onClick={toggle}>
      {sol ? <Moon className="size-4" /> : <SunMedium className="size-4" />}
    </Button>
  )
}

export function TopBar() {
  const view = useMission(s => s.view)
  const setView = useMission(s => s.setView)
  const auto = useMission(s => s.auto)
  const setAuto = useMission(s => s.setAuto)
  const mision = useMission(s => s.summary?.mision)
  const avisos = useMission(s => s.avisosDatos)
  const syncing = useMission(s => s.syncing)

  return (
    <header className="glass sticky top-0 z-40 border-b border-border/60">
      <div className="topbar-inner mx-auto flex h-14 max-w-[1720px] items-center gap-4 px-6">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="relative grid size-8 place-items-center rounded-lg border border-[#3ddc84]/30 bg-[#3ddc84]/10 shadow-[0_0_18px_-4px_rgba(61,220,132,.55)]">
            <Satellite className="size-4.5 shrink-0 text-[#3ddc84]" strokeWidth={1.8} />
          </span>
          <div className="min-w-0">
            <h1 className="truncate text-[15px] font-semibold tracking-tight">Estación Terrena</h1>
            <span className="hidden text-[9px] uppercase tracking-[0.18em] text-muted-foreground/70 sm:block">mission control / LB135</span>
          </div>
          <Badge variant="secondary" className="border-[#3ddc84]/30 bg-[#3ddc84]/10 font-mono text-[11px] text-[#3ddc84]">
            {mision || 'LB135'}
          </Badge>
          <PhaseChip />
          <Badge
            variant="secondary"
            title={avisos.length ? avisos.join('\n') : 'Contrato de datos válido (columnas, rangos y monotonicidad)'}
            className={`font-mono text-[10px] ${avisos.length ? 'border-[#ffb020]/40 bg-[#ffb020]/10 text-[#ffb020]' : 'border-border/50 bg-muted/30 text-muted-foreground'}`}
          >
            datos: {avisos.length ? `${avisos.length} aviso${avisos.length === 1 ? '' : 's'}` : 'ok'}
          </Badge>
        </div>

        <Tabs value={view} onValueChange={v => setView(v as ViewId)} className="topbar-tabs mx-auto">
          <TabsList className="relative h-9 bg-muted/40">
            {VIEWS.map(v => (
              <TabsTrigger key={v.id} value={v.id} className="relative z-10 gap-1.5 data-[state=active]:bg-transparent data-[state=active]:shadow-none">
                {v.id === view && (
                  <motion.span
                    layoutId="tab-pill"
                    className="absolute inset-0 rounded-md border border-border/60 bg-[#1a2432]"
                    transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                  />
                )}
                <span className="relative z-10">{v.label}</span>
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>

        <div className="flex items-center gap-3">
          <SyncDot />
          <Clock />
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground" title="Auto-refresh cada 3 s (P)">
            <Switch checked={auto} onCheckedChange={setAuto} className="scale-90" />
            Auto
          </label>
          <Button
            variant="ghost" size="icon" className="size-8" aria-label="Sincronizar ahora"
            title="Sincronizar ahora (R)"
            onClick={() => window.dispatchEvent(new CustomEvent('lb135-sync'))}
          >
            <RefreshCw className={`size-4 ${syncing ? 'animate-spin' : ''}`} />
          </Button>
          <Button
            variant="ghost" size="icon" className="size-8" aria-label="Reproducir introducción de misión"
            title="Reproducir la intro de la misión"
            onClick={() => window.dispatchEvent(new CustomEvent('lb135-cutscene-replay'))}
          >
            <Clapperboard className="size-4" />
          </Button>
          <Button
            variant="ghost" size="icon" className="size-8" aria-label="Modo jurado"
            title="Modo jurado: deck manual para la defensa"
            onClick={() => {
              useMission.getState().setJurado(true)
              useMission.getState().pushToast('Modo jurado: ←→ navegan, Esc sale')
            }}
          >
            <GraduationCap className="size-4" />
          </Button>
          <SolToggle />
          <Button variant="ghost" size="icon" className="size-8" aria-label="Reabrir ayuda" title="Reabrir el tour de introducción" onClick={startTour}>
            <HelpCircle className="size-4" />
          </Button>
        </div>
      </div>
    </header>
  )
}
