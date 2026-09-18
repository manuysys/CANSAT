/* Póster infográfico 1920×1080 generado en canvas offscreen (sin servicios
   externos): fondo de misión, KPIs, hero del momento crítico y tira de
   frames. Se descarga como PNG desde Post-vuelo. */
import { criticalFrame, useMission } from '@/store/mission'
import { imgURL } from '@/lib/api'
import { mean, num } from '@/lib/format'
import { isAlertFrame } from '@/lib/vocab'
import type { Frame, Summary } from '@/lib/types'

function load(src: string | null): Promise<HTMLImageElement | null> {
  return new Promise(res => {
    if (!src) return res(null)
    const im = new Image()
    im.onload = () => res(im)
    im.onerror = () => res(null)
    im.src = src
  })
}

export async function descargarPoster(frames: Frame[], summary: Summary | null): Promise<void> {
  const mision = summary?.mision || 'LB135'
  const W = 1920
  const H = 1080
  const cv = document.createElement('canvas')
  cv.width = W
  cv.height = H
  const ctx = cv.getContext('2d')
  if (!ctx) return

  // fondo
  const bg = ctx.createLinearGradient(0, 0, 0, H)
  bg.addColorStop(0, '#0b0e14')
  bg.addColorStop(1, '#0e1723')
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, W, H)
  ctx.strokeStyle = 'rgba(61,220,132,.25)'
  ctx.lineWidth = 2
  ctx.strokeRect(48, 48, W - 96, H - 96)

  // tipografía
  const mono = (px: number, bold = false) => `${bold ? '700' : '400'} ${px}px "JetBrains Mono", ui-monospace, monospace`
  const sans = (px: number, bold = false) => `${bold ? '700' : '500'} ${px}px Inter, system-ui, sans-serif`

  ctx.fillStyle = '#7ee8b0'
  ctx.font = mono(22, true)
  ctx.fillText('ESTACIÓN TERRENA · CANSAT', 96, 140)
  ctx.fillStyle = '#eafff3'
  ctx.font = sans(84, true)
  ctx.fillText(`MISIÓN ${mision}`, 96, 232)
  ctx.fillStyle = '#8b9aab'
  ctx.font = mono(24)
  ctx.fillText(`descenso recuperado · ${new Date().toLocaleDateString('es-AR')}`, 96, 280)

  // KPIs
  const alerts = frames.filter(isAlertFrame).length
  const crit = criticalFrame(frames)
  const kpis: Array<[string, string]> = [
    ['FRAMES', String(frames.length)],
    ['ALERTAS', String(alerts)],
    ['ALT MÁX', `${num(summary?.alt_max_m, 0)} m`],
    ['DAÑO PROM', `${num(summary?.danado_pct_prom ?? mean(frames.map(f => f.danado_pct)), 1)}%`],
    ['USI PROM', num(mean(frames.map(f => f.usi)), 3)],
  ]
  kpis.forEach(([k, v], i) => {
    const x = 96 + i * 250
    ctx.fillStyle = '#10161f'
    ctx.fillRect(x, 340, 220, 120)
    ctx.strokeStyle = 'rgba(61,220,132,.35)'
    ctx.lineWidth = 1.5
    ctx.strokeRect(x, 340, 220, 120)
    ctx.fillStyle = '#8b9aab'
    ctx.font = mono(17)
    ctx.fillText(k, x + 18, 378)
    ctx.fillStyle = '#eafff3'
    ctx.font = mono(40, true)
    ctx.fillText(v, x + 18, 432)
  })

  // hero crítico
  const hero = await load(imgURL(crit?.files?.enhanced || crit?.files?.vis || null))
  if (hero) {
    const hw = 880
    const hh = 560
    const hx = W - 96 - hw
    const hy = 340
    ctx.save()
    ctx.beginPath()
    ctx.rect(hx, hy, hw, hh)
    ctx.clip()
    const scale = Math.max(hw / hero.width, hh / hero.height)
    const dw = hero.width * scale
    const dh = hero.height * scale
    ctx.drawImage(hero, hx + (hw - dw) / 2, hy + (hh - dh) / 2, dw, dh)
    ctx.restore()
    ctx.strokeStyle = 'rgba(255,77,94,.6)'
    ctx.lineWidth = 3
    ctx.strokeRect(hx, hy, hw, hh)
    ctx.fillStyle = '#ff8f9b'
    ctx.font = mono(20, true)
    ctx.fillText(`MOMENTO CRÍTICO · ${crit?.src ?? ''} · daño ${num(crit?.danado_pct, 1)}%`, hx, hy + hh + 34)
  }

  // tira de frames
  const thumbs = await Promise.all(frames.slice(0, 12).map(f => load(imgURL(f.files?.thumb || f.files?.vis || null))))
  const tw = 128
  const gap = 12
  thumbs.forEach((t, i) => {
    if (!t) return
    const x = 96 + i * (tw + gap)
    const y = H - 260
    ctx.drawImage(t, x, y, tw, 96)
    ctx.strokeStyle = 'rgba(139,154,171,.35)'
    ctx.lineWidth = 1
    ctx.strokeRect(x, y, tw, 96)
    ctx.fillStyle = '#5d6c7d'
    ctx.font = mono(13)
    ctx.fillText(frames[i].src.replace('cap_', '#'), x, y + 118)
  })

  ctx.fillStyle = '#5d6c7d'
  ctx.font = mono(18)
  ctx.fillText('generado por Estación Terrena Web · datos de solo lectura · lb135', 96, H - 96)

  await new Promise<void>(res => {
    cv.toBlob(b => {
      if (!b) return res()
      const a = document.createElement('a')
      a.href = URL.createObjectURL(b)
      a.download = `poster_${mision}.png`
      a.click()
      URL.revokeObjectURL(a.href)
      res()
    }, 'image/png')
  })
  useMission.getState().pushToast('Póster 1920×1080 descargado')
}
