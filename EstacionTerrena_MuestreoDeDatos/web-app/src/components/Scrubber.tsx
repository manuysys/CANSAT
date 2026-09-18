/* Scrubber de replay tipo video: arrastrás la línea de tiempo de la misión
   (o reproducís a 0.5×/1×/2×/4×) y TODOS los paneles siguen el cursor:
   3D, detalle, curva, corredor y alertas. Usa lib/fly para no robar scroll. */
import { useEffect, useRef, useState } from 'react'
import { Pause, Play, SkipBack } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useMission } from '@/store/mission'
import { num } from '@/lib/format'
import { fly } from '@/lib/fly'

const SPEEDS = [0.5, 1, 2, 4]

export function Scrubber() {
  const frames = useMission(s => s.frames)
  const selected = useMission(s => s.selected)
  const select = useMission(s => s.select)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const mt = useRef<number | null>(null)

  const idx = Math.max(0, frames.findIndex(f => f.src === selected))
  const cur = frames[idx]

  useEffect(() => {
    if (!playing) { fly.on = false; return }
    fly.on = true
    if (mt.current === null) mt.current = Number(frames[idx]?.t_s) || 0
    const id = window.setInterval(() => {
      const st = useMission.getState()
      const list = st.frames
      if (!list.length) { setPlaying(false); return }
      mt.current = (mt.current ?? 0) + 0.12 * speed
      const t = mt.current
      let i = 0
      while (i < list.length - 1 && (Number(list[i + 1].t_s) || 0) <= t) i++
      if (list[i].src !== st.selected) st.select(list[i].src)
      if (i >= list.length - 1) { setPlaying(false); mt.current = null }
    }, 120)
    return () => window.clearInterval(id)
  }, [playing, speed, frames, idx])

  useEffect(() => () => { fly.on = false }, [])

  if (!frames.length) return null

  return (
    <Card className="border-border/60 bg-card/60">
      <div className="flex flex-wrap items-center gap-3 px-4 py-2.5">
        <div className="flex items-center gap-1.5">
          <Button
            variant="outline" size="icon" className="size-7"
            title="Reproducir/pausar el replay de la misión"
            onClick={() => setPlaying(p => !p)}
          >
            {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          </Button>
          <Button
            variant="outline" size="icon" className="size-7"
            title="Volver al inicio de la misión"
            onClick={() => { setPlaying(false); mt.current = null; select(frames[0].src) }}
          >
            <SkipBack className="size-3.5" />
          </Button>
          <Button
            variant="outline" size="sm" className="h-7 w-12 font-mono text-[11px]"
            title="Velocidad de reproducción"
            onClick={() => setSpeed(s => SPEEDS[(SPEEDS.indexOf(s) + 1) % SPEEDS.length])}
          >
            {speed}×
          </Button>
        </div>

        <input
          type="range" min={0} max={frames.length - 1} step={1} value={idx}
          aria-label="Línea de tiempo de la misión"
          className="scrub min-w-0 flex-1"
          onChange={e => {
            setPlaying(false)
            mt.current = null
            select(frames[Number(e.target.value)].src)
          }}
        />

        <span className="shrink-0 font-mono text-[10.5px] text-muted-foreground">
          <span className="text-foreground/90">{cur?.src ?? '—'}</span>
          {' · t '}{num(cur?.t_s, 1)} s · {num(cur?.alt_m, 1)} m
        </span>
      </div>
    </Card>
  )
}
