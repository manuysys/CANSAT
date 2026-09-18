/* ═══════════════════════════════════════════════════════════════════════════
   ESTACIÓN TERRENA · CanSat LB135 — app.js  (v2)
   ---------------------------------------------------------------------------
   Refactor visual sobre la misma maquinaria: contrato de datos, auto-refresh
   de 3 s sin parpadeo, filtros/orden/búsqueda y export CSV intactos.

   Principios de esta versión:
     · Cada dato vive en UN solo lugar por vista.
     · El detalle del frame es el protagonista (columna central).
     · El corredor cuenta el descenso: eje Y = altitud.
     · Tres vistas: VUELO · POST-VUELO · INFORME.
     · Degradar siempre con elegancia: si un archivo no existe, se oculta.
   ═══════════════════════════════════════════════════════════════════════════ */
(() => {
'use strict';

/* ═════════════════════════════════════════════════════════════════════════
   1. UTILIDADES
   ═════════════════════════════════════════════════════════════════════════ */
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const REFRESH_MS = 3000;          // polling del auto-refresh (feature G)
const PRESENT_MS = 4000;          // avance del modo presentación

const imgURL = rel => (rel ? '/img/' + String(rel).split('/').map(encodeURIComponent).join('/') : null);

const norm = s => String(s ?? '')
  .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
  .toUpperCase().replace(/\s+/g, ' ').trim();
/** Clave dura sin acentos ni signos para buscar en diccionarios. */
const normKey = s => norm(s).replace(/[^A-Z0-9]/g, '');

function num(v, d = 1, suffix = '') {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  const n = Number(v);
  const s = Math.abs(n) >= 1000 ? n.toLocaleString('es-AR', { maximumFractionDigits: d }) : n.toFixed(d);
  return s + suffix;
}
const intOr = (v, dflt = 0) =>
  (v === null || v === undefined || Number.isNaN(Number(v)) ? dflt : Math.round(Number(v)));

/** Escapa HTML: todo dato del CSV entra como texto, nunca como markup. */
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function mean(vals) {
  const ok = vals.filter(v => v !== null && v !== undefined && !Number.isNaN(Number(v))).map(Number);
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
}
/** Hash djb2: detecta cambios entre polls para no re-renderizar de más. */
function hashStr(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) >>> 0;
  return h.toString(36);
}
const clampPct = v => Math.max(0, Math.min(100, Number(v) || 0));
function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/* ═════════════════════════════════════════════════════════════════════════
   2. VOCABULARIO DEL CONTRATO
   ═════════════════════════════════════════════════════════════════════════ */
const TERRAIN = [
  { key: 'veg',  label: 'Vegetación',     en: 'vegetation',  color: 'var(--d-veg)',  hex: '#2fbf74' },
  { key: 'bui',  label: 'Edificios',      en: 'building',    color: 'var(--d-bui)',  hex: '#e08c3a' },
  { key: 'wat',  label: 'Agua',           en: 'water',       color: 'var(--d-wat)',  hex: '#3f8fd1' },
  { key: 'bare', label: 'Suelo expuesto', en: 'bare_ground', color: 'var(--d-bare)', hex: '#b98a5e' },
  { key: 'oth',  label: 'Otro',           en: 'other',       color: 'var(--d-oth)',  hex: '#7d8a99' },
];
const PRI_ORDER = ['HIGH', 'MEDIUM', 'LOW'];

const VERDICT_CLASS = { ZONASALUDABLE: 'v-ok', ESTRESMODERADO: 'v-mod',
                        ALTOESTRESURBANO: 'v-high', SUELOEXPUESTO: 'v-bare' };
const verdictClass = v => VERDICT_CLASS[normKey(v)] || '';

const DIAG_SEV = { SINDESASTRE: 0, AGUAEXTENSALAGORIO: 1, POSIBLEINCENDIOEROSION: 2,
                   POSIBLESISMOVIENTO: 3, INUNDACIONURBANA: 3, INUNDACIONSEVERA: 4 };
const diagSev = d => (normKey(d) in DIAG_SEV ? DIAG_SEV[normKey(d)] : (d ? 1 : 0));
const diagClass = d => { const s = diagSev(d); return s === 0 ? 'd-none' : (s >= 3 ? 'd-bad' : 'd-warn'); };

/** Tabs de imagen del detalle, en orden de presentación. */
const IMG_TABS = [
  { k: 'vis',      label: 'Evidencia' },
  { k: 'ens_seg',  label: 'Ensemble'  },
  { k: 'enhanced', label: 'EDSR'      },
  { k: 'high_res', label: 'High-res'  },
];

/* ═════════════════════════════════════════════════════════════════════════
   3. ESTADO
   ═════════════════════════════════════════════════════════════════════════ */
const state = {
  frames: [], bySrc: new Map(), summary: null, assets: {}, buckets: {}, sig: '',
  view: 'vuelo',
  sort: { key: 't_s', dir: 1 },
  filters: { pri: new Set(), alertOnly: false, diag: '', verdict: '', q: '' },
  filtered: [],
  selected: null,
  detailImg: 'vis',
  collapsed: { alerts: false, table: false },
  present: { on: false, idx: 0, timer: null },
  auto: true, timer: null, lastSync: null, fails: 0,
  knownSrcs: new Set(), firstLoad: true,
};

/* ═════════════════════════════════════════════════════════════════════════
   4. RED
   ═════════════════════════════════════════════════════════════════════════ */
async function apiJSON(url, timeoutMs = 8000) {
  const ctrl = new AbortController();
  const to = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { cache: 'no-store', signal: ctrl.signal });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return await res.json();
  } finally { clearTimeout(to); }
}

async function fetchMission() {
  try {
    const data = await apiJSON('/api/mission');
    state.fails = 0;
    applyMission(data);
  } catch {
    state.fails++;
    setConn('offline');
    if (state.fails === 1) toast('Sin respuesta del servidor. Reintentando cada 3 s.', 'warn');
  }
}

/** Fusiona el payload; si nada cambió no toca el DOM (cero parpadeo). */
function applyMission(data) {
  if (!data || typeof data !== 'object') return;
  const frames = Array.isArray(data.frames) ? data.frames : [];
  const sig = hashStr(JSON.stringify({ f: frames, s: data.summary, a: data.assets }));

  const fresh = [];
  frames.forEach(f => {
    const src = String(f?.src ?? '');
    if (src && !state.knownSrcs.has(src)) fresh.push(src);
  });

  const changed = sig !== state.sig;
  state.sig = sig;
  state.frames = frames;
  state.summary = data.summary || null;
  state.assets = data.assets || {};
  state.buckets = data.buckets || {};

  state.bySrc = new Map();
  state.knownSrcs = new Set();
  frames.forEach((f, i) => {
    const src = String(f.src ?? ('frame_' + i));
    f.src = src;
    if (f._idx === undefined || f._idx === null) f._idx = i;
    state.bySrc.set(src, f);
    state.knownSrcs.add(src);
  });

  state.lastSync = new Date();
  setConn(frames.length ? 'live' : 'nodata');
  updateSyncLabel();

  if (!changed) return;

  // Si el frame seleccionado desapareció, elegimos otro ancla.
  if (state.selected && !state.bySrc.has(state.selected)) state.selected = null;

  const scroll = preserveScroll();
  renderAll(fresh);
  scroll.restore();

  const wasFirst = state.firstLoad;
  if (state.firstLoad) {
    state.firstLoad = false;
    // Protagonista inicial: el frame con más daño (la historia de la misión).
    if (!state.selected && frames.length) {
      const worst = frames.slice().sort((a, b) => (Number(b.danado_pct) || 0) - (Number(a.danado_pct) || 0))[0];
      state.selected = worst.src;
      renderDetail();
      markSelection();
    }
    if (!frames.length) toast('Sin telemetría todavía: se espera outputs/mission/telemetry.csv', 'warn');
  }
  if (fresh.length && !wasFirst) {
    toast(`+${fresh.length} frame${fresh.length > 1 ? 's' : ''} nuevo${fresh.length > 1 ? 's' : ''}: ${fresh[fresh.length - 1]}`, 'ok');
  }
  if (state.present.on) renderPresent();
}

function preserveScroll() {
  const nodes = ['#descentScroll', '#tableScroll', '#alertsBody', '#dtMetrics'].map(s => $(s));
  const vals = nodes.map(n => (n ? n.scrollTop : 0));
  return { restore() { nodes.forEach((n, i) => { if (n) n.scrollTop = vals[i]; }); } };
}

/* ═════════════════════════════════════════════════════════════════════════
   5. ESTADO DE ENLACE, RELOJ, TOAST
   ═════════════════════════════════════════════════════════════════════════ */
function setConn(kind) {
  const dot = $('#connDot');
  dot.classList.remove('is-live', 'is-warn', 'is-bad');
  dot.classList.add(kind === 'live' ? 'is-live' : kind === 'nodata' ? 'is-warn' : 'is-bad');
  dot.title = kind === 'live' ? 'Enlace activo con el servidor'
    : kind === 'nodata' ? 'Servidor accesible, sin telemetría' : 'Sin contacto con el servidor';
}

let syncTicker = null;
function updateSyncLabel() {
  const paint = () => {
    const el = $('#footInfo');
    if (!el) return;
    const s = state.lastSync ? Math.max(0, Math.round((Date.now() - state.lastSync.getTime()) / 1000)) : null;
    el.textContent = `Estación Terrena · datos de solo lectura · ${state.auto ? 'auto 3 s' : 'manual'} · ` +
      (s === null ? 'sin sincronizar' : `sync hace ${s} s`);
  };
  paint();
  if (!syncTicker) syncTicker = setInterval(paint, 1000);
}

function startClock() {
  const el = $('#clock');
  const paint = () => { el.textContent = new Date().toLocaleTimeString('es-AR', { hour12: false }); };
  paint(); setInterval(paint, 1000);
}

let toastTimer = null;
function toast(msg, kind = '') {
  const el = $('#toast');
  if (!el) return;
  el.textContent = msg;
  el.className = 'toast' + (kind ? ' ' + kind : '');
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4200);
}

/* ═════════════════════════════════════════════════════════════════════════
   6. RENDER MAESTRO
   ═════════════════════════════════════════════════════════════════════════ */
function renderAll(fresh = []) {
  computeFiltered();
  renderKPIs();
  if (state.view === 'vuelo') renderSpark();
  renderDescent();
  renderDetail();
  renderAlerts();
  renderTable(fresh);
  renderFilterOptions();
  renderPost();
  if (state.view === 'informe') renderInforme();
  $('#missionId').textContent = state.summary?.mision || 'LB135';
}

/* ── 6.1 KPIs: cinco números ─────────────────────────────────────────── */
function renderKPIs() {
  const F = state.frames, has = F.length > 0;
  $('#kpiFrames').textContent = has ? F.length : '—';

  const alerts = F.filter(f => intOr(f.alert) === 1).length;
  $('#kpiAlerts').textContent = has ? alerts : '—';
  $('#kpiAlertCard').classList.toggle('is-alert', alerts > 0);

  const people = F.reduce((a, f) => a + intOr(f.people), 0);
  const veh = F.reduce((a, f) => a + intOr(f.vehicles), 0);
  $('#kpiDetect').textContent = has ? `${people} / ${veh}` : '—';

  const s = state.summary || {};
  const altMax = s.alt_max_m ?? (has ? Math.max(...F.map(f => Number(f.alt_m) || 0)) : null);
  const altMin = s.alt_min_m ?? (has ? Math.min(...F.map(f => (f.alt_m === null ? Infinity : Number(f.alt_m)))) : null);
  $('#kpiAlt').textContent = (altMax === null || altMax === undefined) ? '—'
    : `${num(altMax, 0)} → ${num(altMin === Infinity ? null : altMin, 0)} m`;

  const dmg = s.danado_pct_prom ?? mean(F.map(f => f.danado_pct));
  $('#kpiDmg').textContent = (dmg === null || dmg === undefined) ? '—' : num(dmg, 1) + '%';
}

/* ── 6.2 Sparkline: daño vs altitud (la historia en una curva) ───────── */
function renderSpark() {
  const wrap = $('#sparkWrap'), svg = $('#spark'), F = state.frames;
  const empty = F.length < 2;
  $('#sparkEmpty').hidden = !empty;
  svg.innerHTML = '';
  if (empty) return;

  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (!W || !H) return;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const padL = 44, padR = 20, padT = 14, padB = 22;
  const alts = F.map(f => Number(f.alt_m)).filter(v => !Number.isNaN(v));
  const altMax = Math.max(...alts, 1);
  // X = altitud descendente (250 m a la izquierda, 0 m a la derecha)
  const xOf = a => padL + (1 - Math.max(0, Math.min(altMax, a)) / altMax) * (W - padL - padR);
  const yOf = d => (H - padB) - (clampPct(d) / 100) * (H - padT - padB);

  let out = '';
  // Rejilla mínima: 0 / 50 / 100 % de daño
  [0, 50, 100].forEach(d => {
    const y = yOf(d);
    out += `<line class="grid-line" x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"/>`;
    out += `<text class="axis-lbl" x="${padL - 8}" y="${y + 3}" text-anchor="end">${d}%</text>`;
  });
  out += `<text class="axis-lbl" x="${padL}" y="${H - 6}">${num(altMax, 0)} m</text>`;
  out += `<text class="axis-lbl" x="${W - padR}" y="${H - 6}" text-anchor="end">0 m</text>`;

  const pts = F.map(f => ({ f, x: xOf(Number(f.alt_m) || 0), y: yOf(Number(f.danado_pct) || 0) }));
  out += `<polyline class="curve" points="${pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}"/>`;
  pts.forEach(p => {
    const cls = 'pt' + (intOr(p.f.alert) === 1 ? ' is-alert' : '') + (p.f.src === state.selected ? ' is-sel' : '');
    out += `<circle class="${cls}" data-src="${esc(p.f.src)}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.2"/>`;
  });
  svg.innerHTML = out;
}

function bindSpark() {
  const svg = $('#spark'), tip = $('#sparkTip'), wrap = $('#sparkWrap');
  svg.addEventListener('mouseover', e => {
    const c = e.target.closest('.pt');
    if (!c) return;
    const f = state.bySrc.get(c.dataset.src);
    if (!f) return;
    tip.innerHTML = `${esc(f.src)} · ${num(f.alt_m, 0)} m · daño ${num(f.danado_pct, 1)}%`;
    tip.hidden = false;
    tip.style.left = c.getAttribute('cx') + 'px';
    tip.style.top = c.getAttribute('cy') + 'px';
  });
  svg.addEventListener('mouseout', e => { if (e.target.closest('.pt')) tip.hidden = true; });
  svg.addEventListener('click', e => {
    const c = e.target.closest('.pt');
    if (c) { tip.hidden = true; selectFrame(c.dataset.src); }
  });
  window.addEventListener('resize', debounce(() => { if (state.view === 'vuelo') renderSpark(); }, 160));
}

/* ── 6.3 Corredor de descenso: eje Y = altitud ───────────────────────── */
const CARD_H = 121, CARD_GAP = 12, CANVAS_PAD = 16;

function renderDescent() {
  const canvas = $('#descentCanvas'), F = state.frames;
  canvas.innerHTML = '';
  $('#descEmpty').hidden = F.length > 0;
  $('#descCount').textContent = 'altitud (m) ↓';
  if (!F.length) { canvas.style.height = ''; return; }

  const alts = F.map(f => Number(f.alt_m)).filter(v => !Number.isNaN(v));
  const altMax = Math.max(...alts, 1);

  // Altura del lienzo: la justa para que el espaciado respete la altitud
  // sin solapar tarjetas (con tope por seguridad).
  const sorted = alts.slice().sort((a, b) => b - a);
  let minFrac = 1;
  for (let i = 0; i < sorted.length - 1; i++) {
    const fr = (sorted[i] - sorted[i + 1]) / altMax;
    if (fr > 0) minFrac = Math.min(minFrac, fr);
  }
  const base = F.length * (CARD_H + CARD_GAP) + CANVAS_PAD * 2;
  const need = minFrac > 0.004 ? Math.max(base, (CARD_H + CARD_GAP) / minFrac + CARD_H + CANVAS_PAD * 2) : base;
  const canvasH = Math.min(6000, Math.round(need));
  canvas.style.height = canvasH + 'px';

  const usable = canvasH - CARD_H - CANVAS_PAD * 2;
  const yOf = a => CANVAS_PAD + (1 - Math.max(0, Math.min(altMax, a)) / altMax) * usable;

  // Eje de altitud: marcas cada 50 m + el tope.
  const ticks = new Set([Math.round(altMax)]);
  for (let a = 0; a <= altMax; a += 50) ticks.add(a);
  ticks.forEach(a => {
    const y = yOf(a);
    const t = document.createElement('div');
    t.className = 'tick';
    t.style.top = y.toFixed(1) + 'px';
    t.innerHTML = `<span>${a} m</span>`;
    canvas.appendChild(t);
  });

  // Tarjetas en orden de altitud, con relajación anti-solape.
  const order = F.map(f => ({ f, y: yOf(Number(f.alt_m) || 0) })).sort((a, b) => a.y - b.y);
  let prev = -Infinity;
  order.forEach(o => { o.y = Math.max(o.y, prev + CARD_H + CARD_GAP); prev = o.y; });

  const frag = document.createDocumentFragment();
  order.forEach(o => {
    const f = o.f;
    const pri = PRI_ORDER.includes(String(f.sample_pri)) ? String(f.sample_pri) : 'LOW';
    const files = f.files || {};
    const thumb = imgURL(files.thumb || files.vis);
    const b = document.createElement('button');
    b.type = 'button';
    b.className = `dcard pri-${pri}` + (f.src === state.selected ? ' is-sel' : '');
    b.dataset.src = f.src;
    b.style.top = o.y.toFixed(1) + 'px';
    b.title = `${f.src} · ${num(f.alt_m, 1)} m`;
    b.innerHTML =
      (thumb ? `<img class="dthumb" loading="lazy" src="${thumb}" alt="Miniatura de ${esc(f.src)}">`
             : '<span class="dthumb"></span>') +
      `<span class="dmeta">
         <span class="dsrc">${esc(f.src)}</span>
         <span class="dline">t ${num(f.t_s, 1)} s</span>
         <span class="dline dim">${num(f.alt_m, 1)} m</span>
       </span>` +
      (intOr(f.alert) === 1 ? '<span class="dring" title="alert=1"></span>' : '');
    frag.appendChild(b);
  });
  canvas.appendChild(frag);
}

/* ── 6.4 Detalle protagonista ────────────────────────────────────────── */
function renderDetail() {
  const f = state.selected ? state.bySrc.get(state.selected) : null;
  const img = $('#dtImg'), emptyBox = $('#dtEmpty'), cap = $('#dtCaption');

  if (!f) {
    $('#dtTitle').textContent = 'Ningún frame seleccionado';
    $('#dtPri').hidden = true; $('#dtAlert').hidden = true;
    $('#dtPos').textContent = '—';
    $('#imgTabs').hidden = true; $('#imgTabs').innerHTML = '';
    img.hidden = true; img.removeAttribute('src'); img.dataset.src = '';
    cap.hidden = true; emptyBox.hidden = false;
    $('#dtEmptyTxt').textContent = state.frames.length
      ? 'Seleccioná un frame del corredor, la tabla o las alertas.'
      : 'Sin telemetría: cuando lleguen frames, el detalle se llena solo.';
    $('#dtMetrics').hidden = true;
    return;
  }

  // Cabecera
  $('#dtTitle').textContent = f.src;
  const pri = PRI_ORDER.includes(String(f.sample_pri)) ? String(f.sample_pri) : 'LOW';
  const badge = $('#dtPri');
  badge.hidden = false; badge.textContent = pri; badge.className = 'badge ' + pri;
  const isAlert = intOr(f.alert) === 1;
  $('#dtAlert').hidden = !isAlert;
  const idx = intOr(f._idx, state.frames.indexOf(f));
  $('#dtPos').textContent = `${idx + 1} / ${state.frames.length}`;

  // Tabs de imagen: solo las variantes que existen
  const files = f.files || {};
  const avail = IMG_TABS.filter(t => files[t.k]);
  if (!avail.some(t => t.k === state.detailImg)) state.detailImg = avail[0]?.k || 'vis';
  const tabs = $('#imgTabs');
  if (avail.length <= 1) { tabs.hidden = true; tabs.innerHTML = ''; }
  else {
    tabs.hidden = false;
    tabs.innerHTML = avail.map(t =>
      `<button type="button" role="tab" data-key="${t.k}"
        class="${t.k === state.detailImg ? 'is-active' : ''}"
        aria-selected="${t.k === state.detailImg}">${t.label}</button>`).join('');
  }

  // Imagen principal
  const rel = files[state.detailImg] || files.vis || null;
  if (rel) {
    const url = imgURL(rel);
    emptyBox.hidden = true; img.hidden = false; cap.hidden = false;
    if (img.dataset.src !== url) { img.dataset.src = url; img.src = url; }
    const lbl = IMG_TABS.find(t => t.k === state.detailImg)?.label || 'Evidencia';
    cap.textContent = `${lbl} · ${rel}`;
    img.onerror = () => { img.hidden = true; emptyBox.hidden = false; cap.hidden = true; };
  } else {
    img.hidden = true; img.removeAttribute('src'); img.dataset.src = '';
    cap.hidden = true; emptyBox.hidden = false;
    $('#dtEmptyTxt').textContent = 'Este frame no tiene imágenes disponibles.';
  }

  // Métricas
  $('#dtMetrics').hidden = false;

  $('#dtTerrain').innerHTML = TERRAIN.map(t => {
    const v = clampPct(f[t.key]);
    return `<div class="bar-row">
      <span class="bar-lbl"><i style="background:${t.hex}"></i>${t.label}</span>
      <span class="bar-track"><i class="bar-fill" style="width:${v}%;background:${t.hex}"></i></span>
      <span class="bar-val">${num(f[t.key], 1)}%</span></div>`;
  }).join('');

  $('#dtEnv').innerHTML = [
    ['Altitud', num(f.alt_m, 1), 'm'], ['Presión', num(f.p_hpa, 1), 'hPa'],
    ['Temp.', num(f.temp_c, 1), '°C'], ['USI', num(f.usi, 3), ''],
    ['NDVI', num(f.ndvi, 3), ''], ['t misión', num(f.t_s, 1), 's'],
  ].map(([k, v, u]) => `<div class="cell"><span class="k">${k}</span>
      <span class="v">${v}${u ? ` <small>${u}</small>` : ''}</span></div>`).join('');
  const vb = $('#dtVerdict');
  vb.textContent = f.verdict || 'sin veredicto';
  vb.className = 'verdict ' + verdictClass(f.verdict);

  const dmg = Number(f.danado_pct);
  const dmgCls = Number.isNaN(dmg) ? '' : (dmg >= 40 ? 'is-bad' : dmg >= 15 ? 'is-warn' : 'is-ok');
  $('#dtDamage').innerHTML = [
    ['Daño', num(f.danado_pct, 1), '%', dmgCls], ['Afectado', num(f.aff_m2, 0), 'm²', ''],
    ['Personas', intOr(f.people), '', intOr(f.people) ? 'is-warn' : ''],
    ['Vehículos', intOr(f.vehicles), '', intOr(f.vehicles) ? 'is-warn' : ''],
  ].map(([k, v, u, c]) => `<div class="cell ${c}"><span class="k">${k}</span>
      <span class="v">${v}${u ? ` <small>${u}</small>` : ''}</span></div>`).join('');
  const db = $('#dtDiag');
  db.textContent = `${f.diag || 'sin diagnóstico'}${isAlert ? ' · alerta activa' : ''}`;
  db.className = 'diag ' + diagClass(f.diag) + (isAlert ? ' d-bad' : '');

  const score = clampPct((Number(f.sample_score) || 0) * 100);
  $('#dtSampler').innerHTML = `
    <div class="cell"><span class="k">Prioridad</span>
      <span class="v" style="color:var(--${pri === 'HIGH' ? 'red' : pri === 'MEDIUM' ? 'amber' : 'gray'})">${pri}</span></div>
    <div class="cell"><span class="k">Score</span><span class="v">${num(f.sample_score, 3)}</span></div>
    <div class="cell" style="grid-column:1/-1"><span class="k">score del sampler</span>
      <span class="bar-track" style="display:block;margin-top:6px">
        <i class="bar-fill" style="width:${score}%;background:var(--${pri === 'HIGH' ? 'red' : pri === 'MEDIUM' ? 'amber' : 'gray'})"></i>
      </span></div>
    <div class="cell"><span class="k">Nitidez</span><span class="v">${num(f.sharp, 1)}</span></div>`;
}

/* ── 6.5 Alertas (columna derecha, solo si existen) ──────────────────── */
function alertFrames() {
  return state.frames
    .filter(f => intOr(f.alert) === 1 || diagSev(f.diag) >= 2)
    .sort((a, b) => {
      const s = (diagSev(b.diag) + intOr(b.alert)) - (diagSev(a.diag) + intOr(a.alert));
      return s !== 0 ? s : intOr(a._idx) - intOr(b._idx);
    });
}

function renderAlerts() {
  const list = alertFrames();
  const panel = $('#alertsPanel');
  if (!list.length) { panel.hidden = true; return; }   // degradación: sin alertas, sin panel
  panel.hidden = false;
  $('#alertCount').textContent = list.length;

  $('#alertsBody').innerHTML = list.map(f => {
    const sev = diagSev(f.diag) + intOr(f.alert);
    return `<button type="button" class="alert-card ${sev >= 4 ? 'sev-high' : ''}${f.src === state.selected ? ' is-sel' : ''}"
      data-src="${esc(f.src)}">
      <svg class="alert-ico" aria-hidden="true"><use href="#i-warn"/></svg>
      <span class="alert-title">${esc(f.diag || 'Alerta')}${intOr(f.alert) === 1 ? ' · alert=1' : ''}</span>
      <span class="alert-meta">${esc(f.src)} · ${num(f.t_s, 1)} s · ${num(f.alt_m, 0)} m · ${num(f.danado_pct, 1)}%</span>
    </button>`;
  }).join('');
}

/* ── 6.6 Tabla colapsable con celdas simplificadas ───────────────────── */
function computeFiltered() {
  const { pri, alertOnly, diag, verdict, q } = state.filters;
  const needle = q.trim().toLowerCase();
  let out = state.frames.filter(f => {
    if (pri.size && !pri.has(String(f.sample_pri || ''))) return false;
    if (alertOnly && !(intOr(f.alert) === 1 || diagSev(f.diag) >= 2)) return false;
    if (diag && String(f.diag || '') !== diag) return false;
    if (verdict && String(f.verdict || '') !== verdict) return false;
    if (needle && !String(f.src || '').toLowerCase().includes(needle)) return false;
    return true;
  });
  const { key, dir } = state.sort;
  const val = f => {
    const v = f[key];
    if (typeof v === 'number' && !Number.isNaN(v)) return v;
    if (v === null || v === undefined) return -Infinity;
    return String(v);
  };
  out.sort((a, b) => {
    const x = val(a), y = val(b);
    if (x === y) return intOr(a._idx) - intOr(b._idx);
    return (x < y ? -1 : 1) * dir;
  });
  state.filtered = out;
  return out;
}

function renderTable(fresh = []) {
  const rows = computeFiltered();
  const tbody = $('#tbody');
  const freshSet = new Set(fresh);

  if (!rows.length) {
    tbody.innerHTML = '';
    $('#tableEmpty').hidden = false;
    $('#tableEmpty').textContent = state.frames.length
      ? 'Sin frames que coincidan con el filtro.'
      : 'Sin telemetría cargada: esperando outputs/mission/telemetry.csv.';
  } else {
    $('#tableEmpty').hidden = true;
    tbody.innerHTML = rows.map(f => rowHTML(f, freshSet.has(f.src))).join('');
  }

  // Contador solo cuando el filtro recorta (evita repetir el KPI de frames).
  $('#rowCount').textContent = rows.length === state.frames.length ? '' : `${rows.length} de ${state.frames.length}`;

  const act = [];
  if (state.filters.pri.size) act.push('pri ' + [...state.filters.pri].join('/').toLowerCase());
  if (state.filters.alertOnly) act.push('solo alertas');
  if (state.filters.diag) act.push('diag');
  if (state.filters.verdict) act.push('veredicto');
  if (state.filters.q.trim()) act.push('«' + state.filters.q.trim() + '»');
  $('#tableInfo').textContent = act.length
    ? `Filtros: ${act.join(' · ')} · orden ${state.sort.key} ${state.sort.dir > 0 ? 'asc' : 'desc'}`
    : (state.frames.length ? `Orden: ${state.sort.key} ${state.sort.dir > 0 ? 'asc' : 'desc'}` : '');

  $$('#tbl thead th.sortable').forEach(th => {
    const on = th.dataset.sort === state.sort.key;
    th.classList.toggle('sorted', on);
    th.classList.toggle('desc', on && state.sort.dir < 0);
  });
  markSelection();
}

function rowHTML(f, isNew) {
  const pri = PRI_ORDER.includes(String(f.sample_pri)) ? String(f.sample_pri) : 'LOW';
  const d = Number(f.danado_pct) || 0;
  const barCls = d >= 40 ? 'hot' : d >= 15 ? 'warm' : '';
  const cls = (intOr(f.alert) === 1 ? 'is-alert ' : '') + (isNew ? 'row-new ' : '') +
              (f.src === state.selected ? 'is-sel' : '');
  return `<tr class="${cls.trim()}" data-src="${esc(f.src)}">
    <td class="num">${num(f.t_s, 1)}</td>
    <td class="c-src">${esc(f.src)}</td>
    <td class="num">${num(f.alt_m, 1)}</td>
    <td><span class="chip-diag ${diagClass(f.diag)}" title="${esc(f.diag || '')}">${esc(f.diag || '—')}</span></td>
    <td class="num"><span class="dmg-cell">
        <span class="dmg-bar"><i class="${barCls}" style="width:${clampPct(d)}%"></span></span>${num(f.danado_pct, 1)}</span></td>
    <td class="c-pri"><span class="pri-dot ${pri}" title="${pri}"></span></td>
    <td class="num">${num(f.sample_score, 2)}</td>
    <td class="c-alert">${intOr(f.alert) === 1 ? '<span class="alert-dot" title="alert=1"></span>' : ''}</td>
  </tr>`;
}

/** Resalta la selección en corredor, tabla, alertas y galerías. */
function markSelection() {
  $$('#descentCanvas .dcard').forEach(c => c.classList.toggle('is-sel', c.dataset.src === state.selected));
  $$('#tbody tr').forEach(tr => tr.classList.toggle('is-sel', tr.dataset.src === state.selected));
  $$('#alertsBody .alert-card').forEach(c => c.classList.toggle('is-sel', c.dataset.src === state.selected));
  $$('.gal figure').forEach(g => g.classList.toggle('is-sel', g.dataset.src === state.selected));
  $$('#spark .pt').forEach(p => p.classList.toggle('is-sel', p.dataset.src === state.selected));
}

/** Selecciona un frame y lo lleva al detalle protagonista. */
function selectFrame(src, opts = {}) {
  if (!src || !state.bySrc.has(src)) return;
  state.selected = src;
  if (opts.img) state.detailImg = opts.img;
  renderDetail();
  markSelection();
  if (opts.scroll !== false) {
    const card = $(`#descentCanvas .dcard[data-src="${CSS.escape(src)}"]`);
    if (card) {
      const box = $('#descentScroll');
      const y = card.offsetTop - box.clientHeight / 2 + CARD_H / 2;
      box.scrollTo({ top: Math.max(0, y), behavior: 'smooth' });
    }
    const row = $(`#tbody tr[data-src="${CSS.escape(src)}"]`);
    if (row) row.scrollIntoView({ block: 'nearest' });
  }
}

/* ── 6.7 Opciones de filtros (solo si cambia el conjunto) ────────────── */
const selectCache = { diag: '', verdict: '' };
function renderFilterOptions() {
  fillSelect($('#selDiag'), state.frames.map(f => f.diag), state.filters.diag, 'diag');
  fillSelect($('#selVerdict'), state.frames.map(f => f.verdict), state.filters.verdict, 'verdict');
}
function fillSelect(sel, values, current, cacheKey) {
  if (!sel) return;
  const uniq = [...new Set(values.filter(v => v !== null && v !== undefined && v !== ''))].sort();
  const key = uniq.join('\u0001');
  if (selectCache[cacheKey] === key) { if (current) sel.value = current; return; }
  selectCache[cacheKey] = key;
  const first = sel.querySelector('option');
  sel.innerHTML = '';
  if (first) sel.appendChild(first);
  uniq.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = v;
    sel.appendChild(o);
  });
  sel.value = (current && uniq.includes(current)) ? current : '';
}

/* ── 6.8 Post-vuelo: donut, contrato y galerías que abren el detalle ─── */
const DONUT_COLORS = ['#2fbf74', '#ffb020', '#ff4d5e', '#b98a5e', '#3f8fd1', '#7d8a99'];

function renderPost() {
  const s = state.summary;
  const galEnh = state.frames.filter(f => f.files && f.files.enhanced);
  const galEns = state.frames.filter(f => f.files && f.files.ens_seg);

  $('#galEnhPanel').hidden = galEnh.length === 0;
  $('#galEnsPanel').hidden = galEns.length === 0;
  $('#postEmpty').hidden = Boolean(s) || galEnh.length || galEns.length;

  // Donut de veredictos (del contrato, o calculado si falta)
  const verd = (s && s.veredictos && typeof s.veredictos === 'object') ? s.veredictos : countVerdicts();
  const entries = Object.entries(verd).filter(([, v]) => Number(v) > 0);
  const total = entries.reduce((a, [, v]) => a + Number(v), 0) || 1;
  $('#donutTotal').textContent = total;
  let acc = 0;
  $('#donut').innerHTML =
    `<circle class="bg" r="15.9155" cx="21" cy="21"></circle>` +
    entries.map(([k, v], i) => {
      const len = (Number(v) / total) * 100;
      const c = `<circle r="15.9155" cx="21" cy="21" pathLength="100"
          stroke="${DONUT_COLORS[i % DONUT_COLORS.length]}"
          stroke-dasharray="${len.toFixed(3)} ${(100 - len).toFixed(3)}"
          stroke-dashoffset="${(-acc).toFixed(3)}"><title>${esc(k)}: ${v}</title></circle>`;
      acc += len;
      return c;
    }).join('');
  $('#donutLegend').innerHTML = entries.map(([k, v], i) => `
    <li><i class="tsw" style="background:${DONUT_COLORS[i % DONUT_COLORS.length]}"></i>
        <span class="verdict ${verdictClass(k)}" style="margin:0;padding:0;border:0;background:none;
                text-align:left;text-transform:none;letter-spacing:0;font-weight:600;font-size:12px">${esc(k)}</span>
        <span class="n">${v} · ${((Number(v) / total) * 100).toFixed(0)}%</span></li>`).join('') ||
    '<li class="hint">Sin veredictos.</li>';

  // Resumen del contrato
  $('#postStamp').textContent = s ? (s.mision ? `misión ${s.mision}` : '') : 'sin summary.json';
  if (s) {
    const alerts = Array.isArray(s.alertas) ? s.alertas : [];
    const stats = [
      ['Frames', num(s.n_frames, 0), 'procesados a bordo'],
      ['Altitud máx.', num(s.alt_max_m, 1, ' m'), 'liberación / apogeo'],
      ['Altitud mín.', num(s.alt_min_m, 1, ' m'), 'fin del descenso'],
      ['Personas', num(s.personas_total, 0), 'detecciones YOLO'],
      ['Vehículos', num(s.vehiculos_total, 0), 'detecciones YOLO'],
      ['Daño prom.', num(s.danado_pct_prom, 1, ' %'), 'consenso de 3 modelos'],
      ['Alertas', num(alerts.length, 0), alerts.length ? 'ver vista Vuelo' : 'sin eventos'],
      ['Archivos', num(Object.keys(s.archivos || {}).length, 0), 'entregados'],
    ];
    $('#postStats').innerHTML = stats.map(([k, v, sub]) =>
      `<div class="stat"><span class="s-lbl">${esc(k)}</span><span class="s-val">${esc(v)}</span>
       <span class="s-sub">${esc(sub)}</span></div>`).join('');
  } else {
    $('#postStats').innerHTML =
      '<p class="empty-line">Sin <code>entrega/summary.json</code>: el contrato de post-vuelo aún no existe.</p>';
  }

  // Galerías clickeables -> abren el detalle con esa variante
  paintGallery($('#galEnhanced'), galEnh, 'enhanced', 'EDSR');
  paintGallery($('#galEns'), galEns, 'ens_seg', 'ensemble');
}

function paintGallery(container, items, bucket, tag) {
  if (!container) return;
  container.innerHTML = items.map(f => `
    <figure data-src="${esc(f.src)}" data-bucket="${bucket}" title="${tag} · ${esc(f.src)}">
      <img loading="lazy" src="${imgURL(f.files[bucket])}" alt="${tag} de ${esc(f.src)}">
      <figcaption>${esc(f.src)}</figcaption>
    </figure>`).join('');
}

function countVerdicts() {
  const out = {};
  state.frames.forEach(f => { const v = String(f.verdict || 'SIN DATOS'); out[v] = (out[v] || 0) + 1; });
  return out;
}

/* ── 6.9 Informe imprimible ──────────────────────────────────────────── */
function renderInforme() {
  const F = state.frames, s = state.summary || {};
  if (!F.length) {
    $('#docBody').innerHTML = '<p class="rp-note">Sin telemetría: el informe se genera ' +
      'automáticamente cuando existan frames.</p>';
    return;
  }
  const alerts = alertFrames();
  const top5 = F.filter(f => String(f.sample_pri) === 'HIGH')
    .sort((a, b) => (Number(b.sample_score) || 0) - (Number(a.sample_score) || 0)).slice(0, 5);
  const verd = s.veredictos || countVerdicts();
  const terrain = TERRAIN.map(t => [t.label, mean(F.map(f => f[t.key]))]);
  const people = s.personas_total ?? F.reduce((a, f) => a + intOr(f.people), 0);
  const veh = s.vehiculos_total ?? F.reduce((a, f) => a + intOr(f.vehicles), 0);
  const dmg = s.danado_pct_prom ?? mean(F.map(f => f.danado_pct));
  const altMax = s.alt_max_m ?? Math.max(...F.map(f => Number(f.alt_m) || 0));
  const altMin = s.alt_min_m ?? Math.min(...F.map(f => Number(f.alt_m) || 0));
  const corridor = state.assets?.corridor_map ? imgURL(state.assets.corridor_map) : null;

  const kpis = [
    ['Frames', F.length], ['Alertas', alerts.length], ['Personas', people], ['Vehículos', veh],
    ['Alt. máx (m)', num(altMax, 1)], ['Alt. mín (m)', num(altMin, 1)],
    ['Daño prom (%)', num(dmg, 1)], ['USI prom.', num(mean(F.map(f => f.usi)), 3)],
    ['NDVI prom.', num(mean(F.map(f => f.ndvi)), 3)], ['Score prom.', num(mean(F.map(f => f.sample_score)), 3)],
  ];

  $('#docBody').innerHTML = `
  <div class="rp-hd">
    <div>
      <h2>Informe de misión · ${esc(s.mision || 'LB135')}</h2>
      <p>CanSat estudiantil · Estación Terrena Web<br>
         Segmentación 5 clases · Detección YOLO · Consenso de daño · USI/NDVI · Sampler adaptativo</p>
    </div>
    <div class="rp-hd-right">
      <p>Generado: ${new Date().toLocaleString('es-AR')}<br>
         Fuente: <code>outputs/mission/telemetry.csv</code><br>
         ${state.summary ? 'Post-vuelo: <code>entrega/summary.json</code>' : 'Post-vuelo: pendiente'}</p>
    </div>
  </div>

  <section><div class="rp-h3">1 · Indicadores clave</div>
    <div class="rp-kpis">${kpis.map(([k, v]) =>
      `<div class="rp-kpi"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join('')}
    </div></section>

  <section><div class="rp-h3">2 · Alertas y diagnósticos de desastre (${alerts.length})</div>
    ${alerts.length ? `<table class="rp-tbl">
      <thead><tr><th>#</th><th>src</th><th class="num">t s</th><th class="num">alt m</th><th>diag</th>
        <th class="num">daño %</th><th class="num">personas</th><th class="num">vehículos</th><th>pri</th><th class="num">alert</th></tr></thead>
      <tbody>${alerts.map((f, i) => `<tr><td>${i + 1}</td><td>${esc(f.src)}</td>
        <td class="num">${num(f.t_s, 1)}</td><td class="num">${num(f.alt_m, 1)}</td>
        <td>${esc(f.diag || '—')}</td><td class="num">${num(f.danado_pct, 1)}</td>
        <td class="num">${intOr(f.people)}</td><td class="num">${intOr(f.vehicles)}</td>
        <td>${esc(f.sample_pri || '—')}</td><td class="num">${intOr(f.alert)}</td></tr>`).join('')}</tbody>
    </table>` : '<p class="rp-note">La misión no registró alertas ni diagnósticos de desastre.</p>'}
  </section>

  <section><div class="rp-h3">3 · Top 5 frames prioridad HIGH</div>
    ${top5.length ? `<table class="rp-tbl">
      <thead><tr><th>#</th><th>src</th><th class="num">score</th><th class="num">t s</th><th class="num">alt m</th>
        <th>veredicto</th><th>diag</th><th class="num">daño %</th><th>clase dom.</th></tr></thead>
      <tbody>${top5.map((f, i) => `<tr><td>${i + 1}</td><td>${esc(f.src)}</td>
        <td class="num">${num(f.sample_score, 3)}</td><td class="num">${num(f.t_s, 1)}</td>
        <td class="num">${num(f.alt_m, 1)}</td><td>${esc(f.verdict || '—')}</td>
        <td>${esc(f.diag || '—')}</td><td class="num">${num(f.danado_pct, 1)}</td>
        <td>${esc(domLabelOf(f))}</td></tr>`).join('')}</tbody>
    </table>` : '<p class="rp-note">Ningún frame alcanzó la prioridad HIGH.</p>'}
  </section>

  <div class="rp-2col">
    <section><div class="rp-h3">4 · Veredictos ambientales</div>
      <table class="rp-tbl"><tbody>${Object.entries(verd).map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td class="num">${v}</td>
         <td class="num">${((v / (F.length || 1)) * 100).toFixed(0)}%</td></tr>`).join('')}</tbody></table>
    </section>
    <section><div class="rp-h3">5 · Terreno promedio</div>
      <table class="rp-tbl"><tbody>${terrain.map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td class="num">${num(v, 1)}%</td></tr>`).join('')}
        <tr><td>USI promedio</td><td class="num">${num(mean(F.map(f => f.usi)), 3)}</td></tr>
        <tr><td>NDVI promedio</td><td class="num">${num(mean(F.map(f => f.ndvi)), 3)}</td></tr>
      </tbody></table>
    </section>
  </div>

  ${corridor ? `<section><div class="rp-h3">6 · Corredor de vuelo</div>
    <img class="rp-img" src="${corridor}" alt="Corredor de vuelo apilado"></section>` : ''}

  <div class="rp-sign">
    <div>Operador de estación<small>nombre y aclaración</small></div>
    <div>Responsable de misión<small>nombre y aclaración</small></div>
    <div>Docente asesor<small>nombre y aclaración</small></div>
  </div>

  <div class="rp-ft">
    <span>Estación Terrena · CanSat ${esc(s.mision || 'LB135')} · documento generado automáticamente (solo lectura)</span>
    <span>${F.length} frames · ${alerts.length} alertas</span>
  </div>`;
}

function domLabelOf(f) {
  const d = f.dom;
  if (d !== null && d !== undefined && d !== '') {
    if (typeof d === 'number' || /^\d+$/.test(String(d).trim())) {
      const t = TERRAIN[Number(d)];
      if (t) return t.label;
    } else {
      const k = normKey(d);
      const t = TERRAIN.find(x => normKey(x.en) === k || normKey(x.label) === k || normKey(x.key) === k);
      if (t) return t.label;
      return String(d);
    }
  }
  const best = TERRAIN.slice().sort((a, b) => clampPct(f[b.key]) - clampPct(f[a.key]))[0];
  return best ? best.label : '—';
}

/* ── 6.10 Modo presentación ──────────────────────────────────────────── */
function startPresent() {
  if (!state.frames.length) { toast('Nada para presentar todavía: no hay frames.', 'warn'); return; }
  const i = state.filtered.findIndex(f => f.src === state.selected);
  state.present.idx = i >= 0 ? i : 0;
  state.present.on = true;
  $('#present').hidden = false;
  document.body.style.overflow = 'hidden';
  try { document.documentElement.requestFullscreen?.().catch?.(() => {}); } catch { /* opcional */ }
  renderPresent();
  clearInterval(state.present.timer);
  state.present.timer = setInterval(() => {
    state.present.idx = (state.present.idx + 1) % state.frames.length;
    renderPresent();
  }, PRESENT_MS);
}

function stopPresent() {
  state.present.on = false;
  clearInterval(state.present.timer);
  $('#present').hidden = true;
  document.body.style.overflow = '';
  try { if (document.fullscreenElement) document.exitFullscreen?.().catch?.(() => {}); } catch { /* opcional */ }
}

function renderPresent() {
  if (!state.present.on) return;
  const list = state.filtered.length ? state.filtered : state.frames;
  if (!list.length) { stopPresent(); return; }
  if (state.present.idx >= list.length) state.present.idx = 0;
  const f = list[state.present.idx];
  const files = f.files || {};
  const rel = files.vis || files.enhanced || files.high_res || files.thumb;
  const url = imgURL(rel);
  const img = $('#pImg');
  if (img.dataset.src !== url) { img.dataset.src = url; img.src = url; }
  $('#pSrc').textContent = f.src;
  $('#pCount').textContent = `${state.present.idx + 1} / ${list.length}`;
  const vb = $('#pVerdict');
  vb.textContent = f.verdict || '—';
  vb.className = 'p-verdict ' + verdictClass(f.verdict);
  $('#pDiag').textContent = f.diag || '—';
  $('#pStats').textContent =
    `t ${num(f.t_s, 1)} s · ${num(f.alt_m, 1)} m · daño ${num(f.danado_pct, 1)}% · ` +
    `${f.sample_pri || '—'} ${num(f.sample_score, 2)}`;
  $('#pProgress').innerHTML = list.map((x, i) =>
    `<i class="${i === state.present.idx ? 'now' : i < state.present.idx ? 'done' : ''}"></i>`).join('');
}

/* ── 6.11 Vistas ─────────────────────────────────────────────────────── */
function switchView(v) {
  if (!['vuelo', 'post', 'informe'].includes(v)) return;
  state.view = v;
  $$('.tab').forEach(t => {
    const on = t.dataset.view === v;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', String(on));
  });
  $('#view-vuelo').hidden = v !== 'vuelo';
  $('#view-post').hidden = v !== 'post';
  $('#view-informe').hidden = v !== 'informe';
  if (v === 'vuelo') renderSpark();       // el ancho recién existe cuando es visible
  if (v === 'post') renderPost();
  if (v === 'informe') renderInforme();
}

/* ═════════════════════════════════════════════════════════════════════════
   7. EXPORT CSV
   ═════════════════════════════════════════════════════════════════════════ */
function exportCSV() {
  const rows = state.filtered;
  if (!rows.length) { toast('No hay frames en el filtro activo.', 'warn'); return; }
  const cols = Object.keys(rows[0]).filter(c => !c.startsWith('_') && c !== 'files');
  const q = v => {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return /[",\n;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const csv = [cols.join(',')].concat(rows.map(r => cols.map(c => q(r[c])).join(','))).join('\r\n');
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `LB135_filtro_${rows.length}frames_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
  toast(`CSV exportado: ${rows.length} frames.`, 'ok');
}

/* ═════════════════════════════════════════════════════════════════════════
   8. EVENTOS
   ═════════════════════════════════════════════════════════════════════════ */
function bindEvents() {
  // Vistas
  $$('.tab').forEach(t => t.addEventListener('click', () => switchView(t.dataset.view)));
  $('#btnInforme').addEventListener('click', () => switchView('informe'));
  $('#btnPrint').addEventListener('click', () => window.print());

  // Cabecera
  $('#btnRefresh').addEventListener('click', () => { fetchMission(); toast('Sincronizando con /api/mission…', 'ok'); });
  const btnAuto = $('#btnAuto');
  btnAuto.addEventListener('click', () => toggleAuto());

  // Todo lo que selecciona un frame abre el mismo detalle:
  $('#descentCanvas').addEventListener('click', e => {
    const c = e.target.closest('.dcard'); if (c) selectFrame(c.dataset.src);
  });
  $('#tbody').addEventListener('click', e => {
    const tr = e.target.closest('tr[data-src]'); if (tr) selectFrame(tr.dataset.src);
  });
  $('#alertsBody').addEventListener('click', e => {
    const c = e.target.closest('.alert-card'); if (c) selectFrame(c.dataset.src);
  });
  ['#galEnhanced', '#galEns'].forEach(sel => {
    $(sel).addEventListener('click', e => {
      const fig = e.target.closest('figure[data-src]');
      if (!fig) return;
      switchView('vuelo');
      selectFrame(fig.dataset.src, { img: fig.dataset.bucket });
    });
  });
  bindSpark();

  // Detalle: tabs de imagen y stepper
  $('#imgTabs').addEventListener('click', e => {
    const b = e.target.closest('button[data-key]');
    if (!b) return;
    state.detailImg = b.dataset.key;
    renderDetail();
  });
  $('#dtPrev').addEventListener('click', () => stepSelection(-1));
  $('#dtNext').addEventListener('click', () => stepSelection(1));

  // Colapsables
  $('#btnAlertCollapse').addEventListener('click', () => {
    state.collapsed.alerts = !state.collapsed.alerts;
    $('#alertsPanel').classList.toggle('is-collapsed', state.collapsed.alerts);
    $('#btnAlertCollapse').setAttribute('aria-expanded', String(!state.collapsed.alerts));
  });
  $('#btnTblToggle').addEventListener('click', () => {
    state.collapsed.table = !state.collapsed.table;
    $('#tblPanel').classList.toggle('is-collapsed', state.collapsed.table);
    $('#btnTblToggle').setAttribute('aria-expanded', String(!state.collapsed.table));
  });

  // Orden
  $$('#tbl thead th.sortable').forEach(th => th.addEventListener('click', () => {
    const key = th.dataset.sort;
    if (state.sort.key === key) state.sort.dir *= -1;
    else state.sort = { key, dir: key === 't_s' ? 1 : -1 };
    renderTable();
  }));

  // Filtros
  $('#priChips').addEventListener('click', e => {
    const btn = e.target.closest('.chip[data-pri]');
    if (!btn) return;
    const p = btn.dataset.pri;
    if (state.filters.pri.has(p)) state.filters.pri.delete(p); else state.filters.pri.add(p);
    btn.setAttribute('aria-pressed', String(state.filters.pri.has(p)));
    renderTable();
  });
  $('#chkAlert').addEventListener('change', e => { state.filters.alertOnly = e.target.checked; renderTable(); });
  $('#selDiag').addEventListener('change', e => { state.filters.diag = e.target.value; renderTable(); });
  $('#selVerdict').addEventListener('change', e => { state.filters.verdict = e.target.value; renderTable(); });
  let deb = null;
  $('#inpSearch').addEventListener('input', e => {
    state.filters.q = e.target.value;
    clearTimeout(deb); deb = setTimeout(() => renderTable(), 120);
  });
  $('#btnClearFilters').addEventListener('click', clearFilters);
  $('#btnExport').addEventListener('click', exportCSV);

  // Presentación
  $('#btnPresent').addEventListener('click', startPresent);
  $('#pExit').addEventListener('click', stopPresent);

  document.addEventListener('keydown', onKeydown);
}

function toggleAuto(force) {
  state.auto = force === undefined ? !state.auto : !!force;
  const b = $('#btnAuto');
  b.classList.toggle('is-on', state.auto);
  b.setAttribute('aria-pressed', String(state.auto));
  clearInterval(state.timer);
  if (state.auto) state.timer = setInterval(fetchMission, REFRESH_MS);
  updateSyncLabel();
  toast(state.auto ? 'Auto-refresh activado (3 s).' : 'Auto-refresh pausado.', state.auto ? 'ok' : 'warn');
}

function clearFilters() {
  state.filters = { pri: new Set(), alertOnly: false, diag: '', verdict: '', q: '' };
  $$('#priChips .chip').forEach(c => c.setAttribute('aria-pressed', 'false'));
  $('#chkAlert').checked = false;
  $('#selDiag').value = ''; $('#selVerdict').value = ''; $('#inpSearch').value = '';
  renderTable();
}

function stepSelection(delta) {
  const list = state.filtered.length ? state.filtered : state.frames;
  if (!list.length) return;
  let i = list.findIndex(f => f.src === state.selected);
  i = i < 0 ? 0 : (i + delta + list.length) % list.length;
  selectFrame(list[i].src);
}

function moveSelection(delta) {
  const list = state.filtered;
  if (!list.length) return;
  let i = list.findIndex(f => f.src === state.selected);
  i = i < 0 ? (delta > 0 ? 0 : list.length - 1) : Math.min(list.length - 1, Math.max(0, i + delta));
  selectFrame(list[i].src);
}

/** Atajos de teclado. */
function onKeydown(e) {
  // Modo presentación: solo navegación propia.
  if (state.present.on) {
    if (e.key === 'Escape') { stopPresent(); e.preventDefault(); }
    if (e.key === 'ArrowRight') { state.present.idx = (state.present.idx + 1) % state.frames.length; renderPresent(); }
    if (e.key === 'ArrowLeft') { state.present.idx = (state.present.idx - 1 + state.frames.length) % state.frames.length; renderPresent(); }
    return;
  }

  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || '');

  if (e.key === 'Escape') {
    if (typing && $('#inpSearch').value) { clearFilters(); $('#inpSearch').blur(); }
    else if (typing) document.activeElement.blur();
    return;
  }
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); moveSelection(e.key === 'ArrowDown' ? 1 : -1); return; }
  if (e.key === 'ArrowRight' && !typing) { stepSelection(1); return; }
  if (e.key === 'ArrowLeft' && !typing) { stepSelection(-1); return; }
  if (e.key === 'Enter' && state.selected && !typing) {
    e.preventDefault();
    selectFrame(state.selected);
    $('#detailPanel').scrollIntoView({ block: 'nearest' });
    return;
  }
  if (typing) return;

  const k = e.key.toLowerCase();
  if (k === 'a') {
    state.filters.alertOnly = !state.filters.alertOnly;
    $('#chkAlert').checked = state.filters.alertOnly;
    renderTable();
    toast(state.filters.alertOnly ? 'Filtro: solo alertas.' : 'Filtro de alertas desactivado.', 'warn');
  } else if (k === 'h') {
    const on = state.filters.pri.has('HIGH') && state.filters.pri.size === 1;
    state.filters.pri = new Set(on ? [] : ['HIGH']);
    $$('#priChips .chip').forEach(c => c.setAttribute('aria-pressed', String(state.filters.pri.has(c.dataset.pri))));
    renderTable();
  } else if (e.key === '/') { e.preventDefault(); $('#inpSearch').focus(); $('#inpSearch').select(); }
  else if (k === 'e') exportCSV();
  else if (k === 'r') { fetchMission(); toast('Sincronizando…', 'ok'); }
  else if (k === 'p') toggleAuto();
  else if (k === 'i') switchView('informe');
  else if (k === '1') switchView('vuelo');
  else if (k === '2') switchView('post');
  else if (k === '3') switchView('informe');
}

/* ═════════════════════════════════════════════════════════════════════════
   9. ARRANQUE
   ═════════════════════════════════════════════════════════════════════════ */
function init() {
  bindEvents();
  startClock();
  updateSyncLabel();
  renderAll();
  fetchMission();
  state.timer = setInterval(fetchMission, REFRESH_MS);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) fetchMission(); });
  if (location.protocol === 'file:') {
    toast('Abrí la estación con «python web_server.py» y entrá a http://localhost:8000', 'bad');
  }
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();

})();
