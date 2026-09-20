/* Detalle del frame: el protagonista de la vista Vuelo.
   Imagen grande con tabs internas (solo las variantes que existen) y, debajo,
   terreno + leyenda persistente, ambiente, daño y sampler con tooltips de glosario. */
import { useEffect, useMemo } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ChevronLeft, ChevronRight, ImageOff, TriangleAlert } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { EmptyState } from '@/components/EmptyState'
import { TerrainLegend } from '@/components/TerrainLegend'
import { Gloss } from '@/components/Gloss'
import { filteredFrames, useMission } from '@/store/mission'
import { imgURL } from '@/lib/api'
import { clampPct, intOr, num } from '@/lib/format'
import { diagTone, domLabel, IMG_TABS, priColor, TERRAIN, verdictClass } from '@/lib/vocab'
import type { Frame, ImgKey } from '@/lib/types'

function Kv({ k, v, u, tone, color, gloss }: { k: string; v: string; u?: string; tone?: string; color?: string; gloss?: string }) {
  return (
    <div className="rounded-lg border border-border/60 bg-muted/25 px-2.5 py-1.5">
      <span className="block text-[9.5px] uppercase tracking-[0.08em] text-muted-foreground">
        {gloss ? <Gloss term={gloss}>{k}</Gloss> : k}
      </span>
      <span className={`block font-mono text-[14px] font-semibold leading-snug ${tone || ''}`} style={color ? { color } : undefined}>
        {v}{u ? <span className="ml-1 text-[10px] font-normal text-muted-foreground">{u}</span> : null}
      </span>
    </div>
  )
}

export function Detail() {
  const frames = useMission(s => s.frames)
  const bySrc = useMission(s => s.bySrc)
  const filters = useMission(s => s.filters)
  const sort = useMission(s => s.sort)
  const selected = useMission(s => s.selected)
  const detailImg = useMission(s => s.detailImg)
  const select = useMission(s => s.select)
  const setDetailImg = useMission(s => s.setDetailImg)
  const samples = useMission(s => s.samples)

  const f: Frame | null = selected ? bySrc.get(selected) ?? null : null
  const ex = f ? samples[f.src] : undefined
  const list = useMemo(() => filteredFrames({ frames, filters, sort }), [frames, filters, sort])

  const avail = useMemo(() => (f ? IMG_TABS.filter(t => f.files?.[t.k]) : []), [f])
  useEffect(() => {
    if (f && avail.length && !avail.some(t => t.k === detailImg)) setDetailImg(avail[0].k)
  }, [f, avail, detailImg, setDetailImg])

  const step = (delta: number) => {
    if (!list.length || !f) return
    const i = list.findIndex(x => x.src === f.src)
    const j = i < 0 ? 0 : (i + delta + list.length) % list.length
    select(list[j].src)
  }

  if (!f) {
    return (
      <Card id="tour-detalle" className="flex flex-col border-border/60 bg-card/60">
        <div className="border-b border-border/50 px-4 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Detalle del frame</h2>
        </div>
        <div className="flex flex-1 items-center justify-center p-6">
          <EmptyState
            icon={ImageOff}
            title={frames.length ? 'Ningún frame seleccionado.' : 'Sin telemetría todavía.'}
            hint={frames.length ? 'Elegí una tarjeta del corredor, un punto de la curva o una fila de la timeline.' : 'Cuando lleguen frames, este panel se llena solo.'}
          />
        </div>
      </Card>
    )
  }

  const pri = String(f.sample_pri ?? 'LOW')
  const isAlert = intOr(f.alert) === 1
  const rel = f.files?.[detailImg as ImgKey] || f.files?.vis || null
  const url = imgURL(rel)
  const tabLabel = IMG_TABS.find(t => t.k === detailImg)?.label ?? 'Evidencia'
  const idx = intOr(f._idx, frames.indexOf(f))
  const tone = diagTone(f.diag)

  return (
    <Card id="tour-detalle" className="flex flex-col border-border/60 bg-card/60">
      {/* Cabecera */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border/50 px-4 py-2.5">
        <h2 className="font-mono text-[13px] font-semibold">{f.src}</h2>
        <Badge variant="secondary" className="font-mono text-[10px]" style={{ color: priColor(pri), borderColor: priColor(pri) + '55', background: priColor(pri) + '14' }}>
          {pri}
        </Badge>
        {isAlert && (
          <Badge variant="secondary" className="gap-1 border-[#ff4d5e]/40 bg-[#ff4d5e]/10 text-[10px] text-[#ff8f9b]">
            <motion.span animate={{ scale: [1, 1.18, 1] }} transition={{ repeat: Infinity, duration: 1.6, ease: 'easeInOut' }} className="flex">
              <TriangleAlert className="size-3" />
            </motion.span>
            alerta
          </Badge>
        )}
        {/* Política fire_only_v1: badge SOLO si el modelo lo confirmó
            (tipo_estado=confirmado_por_modelo). El top-1 crudo no se muestra:
            el clasificador no generaliza a eventos nuevos (LOEO 0.248). */}
        {ex?.tipo_estado === 'confirmado_por_modelo' && ex?.tipo_desastre === 'incendio' && (
          <Badge
            variant="secondary"
            className="border-[#5aa9e6]/40 bg-[#5aa9e6]/10 text-[10px] text-[#a8d4f5]"
            title={`Confirmado por el clasificador de tipo · política ${ex.tipo_politica} · umbral ${num(ex.tipo_umbral_incendio, 2)} · modelo ${ex.tipo_modelo_hash ?? 's/h'}`}
          >
            evento: {ex.tipo_desastre}{Number(ex.tipo_conf) > 0 ? ` · ${num(ex.tipo_conf, 2)}` : ''}
          </Badge>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          <span className="font-mono text-[10.5px] text-muted-foreground">{idx + 1} / {frames.length}</span>
          <Button variant="ghost" size="icon" className="size-7" onClick={() => step(-1)} title="Frame anterior (←)"><ChevronLeft className="size-4" /></Button>
          <Button variant="ghost" size="icon" className="size-7" onClick={() => step(1)} title="Frame siguiente (→)"><ChevronRight className="size-4" /></Button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
      {/* Columna imagen */}
      <div className="flex min-h-0 flex-col">
      {/* Tabs de imagen: solo las que existen */}
      {avail.length > 1 && (
        <Tabs value={detailImg} onValueChange={v => setDetailImg(v as ImgKey)} className="px-4 pt-3">
          <TabsList className="h-8 bg-muted/40">
            {avail.map(t => (
              <TabsTrigger key={t.k} value={t.k} className="h-6 px-3 text-xs">{t.label}</TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      )}

      {/* Atajo a la copia nítida cuando la vista actual es de baja resolución */}
      {detailImg === 'vis' && f.files?.enhanced && (
        <div className="px-4 pt-2">
          <button
            type="button"
            onClick={() => setDetailImg('enhanced')}
            className="rounded-full border border-[#3ddc84]/35 bg-[#3ddc84]/10 px-3 py-1 font-mono text-[10px] text-[#7ee8b0] transition-colors hover:bg-[#3ddc84]/20"
          >
            ✦ hay versión nítida (EDSR) · ver
          </button>
        </div>
      )}

      {/* Imagen protagonista con fade + scale 1.02 → 1 */}
      <div className="relative flex flex-1 items-center justify-center px-4 py-4">
        <AnimatePresence mode="wait">
          {url ? (
            <motion.figure
              key={detailImg + ':' + f.src}
              initial={{ opacity: 0, scale: 1.02 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18, ease: 'easeOut' }}
              className="flex w-fit flex-col items-center justify-center"
            >
              <img src={url} alt={`${tabLabel} de ${f.src}`} className="max-h-[46vh] w-auto max-w-full rounded-lg border border-border/40 bg-[#070a0f] object-contain shadow-[0_24px_60px_-28px_rgba(0,0,0,.95)] xl:max-h-[520px]" />
              <figcaption className="mt-2 w-full truncate text-center font-mono text-[10.5px] text-muted-foreground/80">
                {tabLabel} · {rel}
              </figcaption>
            </motion.figure>
          ) : (
            <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full">
              <EmptyState icon={ImageOff} title="Este frame no tiene imágenes disponibles." />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      </div>

      {/* Métricas */}
      <div className="grid content-start gap-2.5 border-t border-border/50 p-3.5 md:grid-cols-2 xl:border-l xl:border-t-0">
        <section className="rounded-xl border border-border/60 bg-muted/20 p-2.5">
          <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Terreno</h3>
          <div className="flex flex-col gap-1.5">
            {TERRAIN.map(t => {
              const v = clampPct(Number(f[t.key]))
              return (
                <div key={t.key} className="grid grid-cols-[92px_1fr_44px] items-center gap-2">
                  <span className="truncate text-[11px] text-muted-foreground">{t.label}</span>
                  <span className="h-2 overflow-hidden rounded-full border border-border/60 bg-muted/30">
                    <motion.span className="block h-full rounded-full" style={{ background: t.hex }} initial={false} animate={{ width: `${v}%` }} transition={{ type: 'spring', stiffness: 160, damping: 24 }} />
                  </span>
                  <span className="text-right font-mono text-[11px] tabular-nums text-foreground/90">{num(f[t.key], 1)}%</span>
                </div>
              )
            })}
          </div>
          <div className="mt-2.5 border-t border-border/40 pt-2">
            <TerrainLegend compact />
          </div>
          <p className="mt-1.5 font-mono text-[10.5px] text-muted-foreground">
            dominante: <span className="text-foreground/90">{domLabel(f)}</span> · nitidez{' '}
            <Gloss term="sharp"><span className="text-foreground/90">{num(f.sharp, 1)}</span></Gloss>
          </p>
        </section>

        <section className="rounded-xl border border-border/60 bg-muted/20 p-2.5">
          <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Ambiente</h3>
          <div className="grid grid-cols-3 gap-1.5">
            <Kv k="Altitud" v={num(f.alt_m, 1)} u="m" />
            <Kv k="Presión" v={num(f.p_hpa, 1)} u="hPa" />
            <Kv k="Temp." v={num(f.temp_c, 1)} u="°C" />
            {/* Humedad del BME280 (DPD): 0/ausente = sin sensor */}
            {Number(f.hum_pct) > 0 && <Kv k="Humedad" v={num(f.hum_pct, 0)} u="%" />}
            {/* Estrés ambiental (2026-09-18): humidex de sensores + bruma */}
            {Number(f.humidex) > 0 && <Kv k="Humidex" v={num(f.humidex, 1)} />}
            {Number(f.haze_pct) > 0 && <Kv k="Bruma" v={num(f.haze_pct, 1)} u="%"
              tone={Number(f.haze_pct) >= 45 ? 'text-[#ff4d5e]' : Number(f.haze_pct) >= 20 ? 'text-[#ffb020]' : ''} />}
            {Number(f.stress_idx) > 0 && <Kv k="Estrés" v={num(f.stress_idx, 0)} />}
            <Kv k="USI" v={num(f.usi, 3)} gloss="usi" />
            <Kv k="NDVI" v={num(f.ndvi, 3)} gloss="ndvi" />
            <Kv k="t misión" v={num(f.t_s, 1)} u="s" />
          </div>
          <div
            className={`mt-2 rounded-lg border px-3 py-1.5 text-center text-[11px] font-semibold uppercase tracking-[0.1em] ${verdictClass(f.verdict) === 'v-ok' ? 'border-[#3ddc84]/35 bg-[#3ddc84]/8 text-[#3ddc84]' : verdictClass(f.verdict) === 'v-mod' ? 'border-[#ffb020]/35 bg-[#ffb020]/8 text-[#ffb020]' : verdictClass(f.verdict) === 'v-high' ? 'border-[#ff4d5e]/35 bg-[#ff4d5e]/8 text-[#ff8f9b]' : 'border-[#b98a5e]/35 bg-[#b98a5e]/8 text-[#d8b48a]'}`}
          >
            {f.verdict || 'sin veredicto'}
          </div>
        </section>

        <section className="rounded-xl border border-border/60 bg-muted/20 p-2.5">
          <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Daño</h3>
          <div className="grid grid-cols-2 gap-1.5">
            <Kv k="Daño" v={num(f.danado_pct, 1)} u="%" gloss="danado"
              tone={Number(f.danado_pct) >= 40 ? 'text-[#ff4d5e]' : Number(f.danado_pct) >= 15 ? 'text-[#ffb020]' : 'text-[#3ddc84]'} />
            {/* F3/F2b: fuego/humo y colapso medido (solo si el modelo corrió) */}
            {Number(f.fire_pct) > 0 && <Kv k="Fuego" v={num(f.fire_pct, 1)} u="%" tone="text-[#ff4d5e]" />}
            {Number(f.smoke_pct) > 0 && <Kv k="Humo" v={num(f.smoke_pct, 1)} u="%" tone="text-[#ffb020]" />}
            {/* Severidad textual: colapso MEDIDO (modelo) vs supuesto 0.3 */}
            {Number(f.colapso_pct) > 0 && (
              <Kv k="Colapso" v={`${num(f.colapso_pct, 1)}${ex?.colapso_fuente ? ` ${ex.colapso_fuente}` : ''}`} u="%"
                gloss="colapso" />
            )}
            <Kv k="Afectado" v={num(f.aff_m2, 0)} u="m²" />
            <Kv k="Personas" v={String(intOr(f.people))} tone={intOr(f.people) ? 'text-[#ffb020]' : ''} />
            <Kv k="Vehículos" v={String(intOr(f.vehicles))} tone={intOr(f.vehicles) ? 'text-[#ffb020]' : ''} />
          </div>
          <div className={`mt-2 rounded-lg border px-3 py-1.5 text-center font-mono text-[11.5px] ${tone === 'bad' ? 'border-[#ff4d5e]/35 bg-[#ff4d5e]/8 text-[#ffc2c8]' : tone === 'warn' ? 'border-[#ffb020]/35 bg-[#ffb020]/8 text-[#ffe3b0]' : 'border-[#3ddc84]/30 bg-[#3ddc84]/6 text-[#b6f0cd]'}`}>
            <Gloss term={f.diag ?? 'SINDESASTRE'}>{f.diag || 'sin diagnóstico'}</Gloss>
            {isAlert ? ' · alerta activa' : ''}
          </div>
          {/* Contrato v3: hash corto del ONNX que produjo cada campo (trazabilidad) */}
          {ex?.model_ids && Object.keys(ex.model_ids).length > 0 && (
            <p
              className="mt-2 font-mono text-[9.5px] leading-relaxed text-muted-foreground/70"
              title={`Trazabilidad: hash corto (8 hex) del ONNX que produjo cada campo · quant ${ex.quant ?? 'fp32'}`}
            >
              modelos: {Object.entries(ex.model_ids).map(([k, v]) => `${k} ${v}`).join(' · ')}
            </p>
          )}
        </section>

        <section className="rounded-xl border border-border/60 bg-muted/20 p-2.5">
          <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Sampler</h3>
          <div className="grid grid-cols-2 gap-1.5">
            <Kv k="Prioridad" v={pri} gloss="pri" color={priColor(pri)} />
            <Kv k="Score" v={num(f.sample_score, 3)} gloss="sampler" />
          </div>
          <span className="mt-2 block h-2 overflow-hidden rounded-full border border-border/60 bg-muted/30">
            <motion.span className="block h-full" style={{ background: priColor(pri) }} initial={false} animate={{ width: `${clampPct((Number(f.sample_score) || 0) * 100)}%` }} />
          </span>
          <p className="mt-2 font-mono text-[10.5px] text-muted-foreground">
            guardado en prioridad {pri.toLowerCase()} por el sampler a bordo
          </p>
        </section>
      </div>
      </div>
    </Card>
  )
}
