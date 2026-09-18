/* Estado global ligero (zustand): frames, selección, filtros, vista, presentación.
   El auto-refresh vive en hooks/useFeed.ts y solo llama a applyPayload(). */
import { create } from 'zustand'
import { hashStr, intOr } from '@/lib/format'
import { validarPayload } from '@/lib/validate'
import { apogeo, missionPhase } from '@/lib/missionState'
import { diagSev, isAlertFrame } from '@/lib/vocab'
import type { Frame, ImgKey, MissionPayload, SampleExtra, SamplePri, SamplesResumen,
              SortKey, Summary, ViewId } from '@/lib/types'

export interface Filters {
  pri: Set<SamplePri>
  alertOnly: boolean
  diag: string
  verdict: string
  q: string
}

export type Status = 'loading' | 'live' | 'nodata' | 'offline'

export interface Evento {
  id: number
  wall: string
  t: string
  tipo: 'frame' | 'alerta' | 'fase' | 'enlace' | 'post'
  msg: string
}

export interface Toast { id: number; msg: string }

let seq = 1

interface MissionState {
  status: Status
  syncing: boolean
  frames: Frame[]
  bySrc: Map<string, Frame>
  summary: Summary | null
  assets: { corridor_map: string | null }
  buckets: Record<string, boolean>
  /* Panel de Muestreo: campos del JSONL que no están en el CSV */
  samples: Record<string, SampleExtra>
  samplesResumen: SamplesResumen | null
  sig: string
  freshSrcs: string[]
  avisosDatos: string[]
  eventos: Evento[]
  toasts: Toast[]
  jurado: boolean
  sol: boolean
  lastSync: number | null

  view: ViewId
  selected: string | null
  detailImg: ImgKey
  filters: Filters
  sort: { key: SortKey; dir: 1 | -1 }
  auto: boolean
  alertsCollapsed: boolean
  tableCollapsed: boolean
  present: { on: boolean; idx: number }

  applyPayload: (p: MissionPayload) => void
  setSamples: (s: Record<string, SampleExtra>, r: SamplesResumen | null) => void
  setOffline: () => void
  setSyncing: (b: boolean) => void
  setView: (v: ViewId) => void
  select: (src: string | null, img?: ImgKey) => void
  setDetailImg: (k: ImgKey) => void
  togglePri: (p: SamplePri) => void
  setAlertOnly: (b: boolean) => void
  setDiag: (s: string) => void
  setVerdict: (s: string) => void
  setQ: (s: string) => void
  clearFilters: () => void
  toggleSort: (k: SortKey) => void
  pushEvento: (tipo: Evento['tipo'], msg: string, t?: string) => void
  pushToast: (msg: string) => void
  dropToast: (id: number) => void
  setJurado: (b: boolean) => void
  setSol: (b: boolean) => void
  setAuto: (b: boolean) => void
  toggleAlertsCollapsed: () => void
  toggleTableCollapsed: () => void
  startPresent: () => void
  stopPresent: () => void
  setPresentIdx: (i: number) => void
}

const emptyFilters = (): Filters => ({ pri: new Set(), alertOnly: false, diag: '', verdict: '', q: '' })

export const useMission = create<MissionState>((set, get) => ({
  status: 'loading',
  syncing: false,
  frames: [],
  bySrc: new Map(),
  summary: null,
  assets: { corridor_map: null },
  buckets: {},
  sig: '',
  freshSrcs: [],
  avisosDatos: [],
  samples: {},
  samplesResumen: null,
  eventos: [],
  toasts: [],
  jurado: false,
  sol: typeof document !== 'undefined' && document.documentElement.dataset.theme === 'sol',
  lastSync: null,

  view: 'vuelo',
  selected: null,
  detailImg: 'enhanced',
  filters: emptyFilters(),
  sort: { key: 't_s', dir: 1 },
  auto: true,
  alertsCollapsed: false,
  tableCollapsed: false,
  present: { on: false, idx: 0 },

  /** Fusiona el payload del servidor. Si la firma no cambia, no toca nada:
      así el polling de 3 s no re-anima ni parpadea (solo diff real). */
  applyPayload: p => {
    const s = get()
    const frames = Array.isArray(p.frames) ? p.frames : []
    const sig = hashStr(JSON.stringify({ f: frames, s: p.summary, a: p.assets }))
    const fresh = frames.filter(f => f?.src && !s.bySrc.has(f.src)).map(f => f.src)
    const bySrc = new Map<string, Frame>()
    frames.forEach((f, i) => {
      if (f._idx === null || f._idx === undefined) f._idx = i
      bySrc.set(f.src, f)
    })
    const status: Status = frames.length ? 'live' : 'nodata'

    if (sig === s.sig) {
      set({ status, lastSync: Date.now(), syncing: false })
      return
    }

    // ── bitácora: diff de fase, alertas nuevas, apogeo, aterrizaje, post ──
    const freshFrames = s.sig === '' ? [] : frames.filter(f => f?.src && !s.bySrc.has(f.src))
    const faseAntes = s.frames.length ? missionPhase(s.frames, s.summary).id : null
    const faseNueva = missionPhase(frames, p.summary ?? null).id
    const ev: Array<[Evento['tipo'], string, string?]> = []
    if (s.sig !== '') {
      freshFrames.filter(f => intOr(f.alert) === 1).forEach(f =>
        ev.push(['alerta', `alerta nueva: ${f.src} · ${f.diag} · daño ${Number(f.danado_pct) || 0}%`, String(f.t_s)]))
      if (freshFrames.length && freshFrames.every(f => intOr(f.alert) !== 1)) {
        ev.push(['frame', `${freshFrames.length} frame(s) nuevo(s): ${freshFrames.map(f => f.src).join(', ')}`, String(frames[frames.length - 1]?.t_s)])
      }
    }
    if (faseAntes && faseAntes !== faseNueva) {
      if (faseAntes === 'ascenso' && faseNueva === 'descenso') {
        const apo = apogeo(frames)
        ev.push(['fase', `apogeo: ${apo?.alt ?? '—'} m en ${apo?.src ?? '—'} → inicia descenso`, apo ? String(frames[apo.idx].t_s) : undefined])
      } else if (faseNueva === 'aterrizado' || faseNueva === 'post-vuelo') {
        ev.push(['fase', `aterrizaje: altitud final ${Number(frames[frames.length - 1]?.alt_m) || 0} m`, String(frames[frames.length - 1]?.t_s)])
      } else {
        ev.push(['fase', `fase de misión: ${faseNueva}`, String(frames[frames.length - 1]?.t_s)])
      }
    }
    if (!s.summary && p.summary) ev.push(['post', 'summary.json publicado → post-vuelo disponible'])
    if (s.status === 'offline' && status === 'live') ev.push(['enlace', 'enlace restablecido'])

    // Primera carga: el protagonista es el frame con más daño (la historia).
    let selected = s.selected
    if (s.sig === '' && frames.length) {
      const worst = [...frames].sort((a, b) => (Number(b.danado_pct) || 0) - (Number(a.danado_pct) || 0))[0]
      selected = worst.src
    }
    if (selected && !bySrc.has(selected)) selected = null

    set({
      status,
      frames,
      bySrc,
      summary: p.summary,
      assets: p.assets,
      buckets: p.buckets,
      sig,
      freshSrcs: s.sig === '' ? [] : fresh,
      avisosDatos: validarPayload(frames, p.summary ?? null),
      lastSync: Date.now(),
      syncing: false,
      selected,
    })
    ev.forEach(([tipo, msg, t]) => get().pushEvento(tipo, msg, t))
    if (s.sig !== '' && freshFrames.length) {
      get().pushToast(`${freshFrames.length} frame${freshFrames.length === 1 ? '' : 's'} nuevo${freshFrames.length === 1 ? '' : 's'} en vivo`)
    }
  },

  pushEvento: (tipo, msg, t) => set(st => ({
    eventos: [...st.eventos.slice(-199), {
      id: seq++,
      wall: new Date().toLocaleTimeString('es-AR', { hour12: false }),
      t: t ?? '—',
      tipo,
      msg,
    }],
  })),
  pushToast: msg => {
    const id = seq++
    set(st => ({ toasts: [...st.toasts.slice(-3), { id, msg }] }))
    window.setTimeout(() => get().dropToast(id), 3600)
  },
  dropToast: id => set(st => ({ toasts: st.toasts.filter(t => t.id !== id) })),
  setJurado: b => set({ jurado: b }),
  setSol: b => {
    if (b) document.documentElement.dataset.theme = 'sol'
    else delete document.documentElement.dataset.theme
    localStorage.setItem('lb135-theme', b ? 'sol' : 'oscuro')
    set({ sol: b })
  },

  setOffline: () => {
    if (get().status !== 'offline') get().pushEvento('enlace', 'enlace caído con el servidor')
    set({ status: 'offline', syncing: false })
  },
  setSamples: (samples, resumen) => set({ samples, samplesResumen: resumen }),
  setSyncing: b => set({ syncing: b }),
  setView: v => set({ view: v }),
  select: (src, img) => set(st => ({
    selected: src,
    detailImg: img ?? st.detailImg,
  })),
  setDetailImg: k => set({ detailImg: k }),

  togglePri: p => set(st => {
    const pri = new Set(st.filters.pri)
    if (pri.has(p)) pri.delete(p); else pri.add(p)
    return { filters: { ...st.filters, pri } }
  }),
  setAlertOnly: b => set(st => ({ filters: { ...st.filters, alertOnly: b } })),
  setDiag: v => set(st => ({ filters: { ...st.filters, diag: v } })),
  setVerdict: v => set(st => ({ filters: { ...st.filters, verdict: v } })),
  setQ: v => set(st => ({ filters: { ...st.filters, q: v } })),
  clearFilters: () => set({ filters: emptyFilters() }),
  toggleSort: k => set(st => ({
    sort: st.sort.key === k ? { key: k, dir: (st.sort.dir * -1) as 1 | -1 } : { key: k, dir: k === 't_s' ? 1 : -1 },
  })),
  setAuto: b => set({ auto: b }),
  toggleAlertsCollapsed: () => set(st => ({ alertsCollapsed: !st.alertsCollapsed })),
  toggleTableCollapsed: () => set(st => ({ tableCollapsed: !st.tableCollapsed })),

  startPresent: () => {
    const st = get()
    const list = filteredFrames(st)
    const i = list.findIndex(f => f.src === st.selected)
    set({ present: { on: true, idx: i >= 0 ? i : 0 } })
  },
  stopPresent: () => set({ present: { on: false, idx: 0 } }),
  setPresentIdx: i => set({ present: { on: true, idx: i } }),
}))

/* ── Selectores puros (se usan con useMission(selector) o fuera de React) ── */

export function filteredFrames(st: Pick<MissionState, 'frames' | 'filters' | 'sort'>): Frame[] {
  const { pri, alertOnly, diag, verdict, q } = st.filters
  const needle = q.trim().toLowerCase()
  const out = st.frames.filter(f => {
    if (pri.size && !pri.has(String(f.sample_pri || '') as SamplePri)) return false
    if (alertOnly && !(intOr(f.alert) === 1 || diagSev(f.diag) >= 2)) return false
    if (diag && String(f.diag || '') !== diag) return false
    if (verdict && String(f.verdict || '') !== verdict) return false
    if (needle && !String(f.src || '').toLowerCase().includes(needle)) return false
    return true
  })
  const { key, dir } = st.sort
  const val = (f: Frame): number | string => {
    const v = f[key]
    if (typeof v === 'number' && !Number.isNaN(v)) return v
    if (v === null || v === undefined) return -Infinity
    return String(v)
  }
  out.sort((a, b) => {
    const x = val(a), y = val(b)
    if (x === y) return intOr(a._idx) - intOr(b._idx)
    return (x < y ? -1 : 1) * dir
  })
  return out
}

export function alertFrames(frames: Frame[]): Frame[] {
  return frames
    .filter(isAlertFrame)
    .sort((a, b) => {
      const s = (diagSev(b.diag) + intOr(b.alert)) - (diagSev(a.diag) + intOr(a.alert))
      return s !== 0 ? s : intOr(a._idx) - intOr(b._idx)
    })
}

/** Frame más crítico de la misión (para el resumen narrativo). */
export function criticalFrame(frames: Frame[]): Frame | null {
  if (!frames.length) return null
  return [...frames].sort((a, b) => (Number(b.danado_pct) || 0) - (Number(a.danado_pct) || 0))[0]
}
