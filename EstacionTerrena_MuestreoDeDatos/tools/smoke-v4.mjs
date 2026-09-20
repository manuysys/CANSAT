// smoke-v4.mjs — prueba de humo E2E del frontend Vite+React (build production
// servido por web_server.py). Criterios: cero errores de consola, tour una sola
// vez, teclado, tabs de imagen degradadas, vistas, presentación, auto-refresh
// sin recarga y mismo detalle desde cualquier click.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const at = (...s) => path.join(ROOT, ...s);
const BASE = process.env.BASE || 'http://localhost:8000';
const SHOTS = path.join(os.tmpdir(), 'lb135-shots');
fs.mkdirSync(SHOTS, { recursive: true });

const problems = [];
const log = (...a) => console.log('[smoke4]', ...a);
const check = (cond, msg) => { if (cond) log('OK  ', msg); else { log('FAIL', msg); problems.push(msg); } };

const CSV = fs.readFileSync(at('outputs', 'mission', 'telemetry.csv'), 'utf8').trim().split(/\r?\n/);
const HEAD = CSV[0].split(',');
const ROWS = CSV.slice(1).map(l => Object.fromEntries(l.split(',').map((v, i) => [HEAD[i], v])));
const EXP = {
  n: ROWS.length,
  alerts: ROWS.filter(r => +r.alert === 1).length,
  cards: ROWS.filter(r => +r.alert === 1 || ['INUNDACION SEVERA', 'INUNDACION URBANA', 'POSIBLE SISMO/VIENTO', 'POSIBLE INCENDIO/EROSION'].includes(r.diag)).length,
  high: ROWS.filter(r => r.sample_pri === 'HIGH').length,
  enhanced: fs.readdirSync(at('entrega', 'enhanced')).length,
  ensSeg: fs.readdirSync(at('entrega', 'ens_seg')).length,
};

const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage', '--js-flags=--jitless'] });

/* Cutscene de arranque: si está activa, se saltea (Esc) y se espera su
   desmontaje antes de seguir con las aserciones. */
/* Espera estable la cantidad de filas de la timeline (las filas salen animadas). */
async function waitRows(pg, n, timeout = 6000) {
  await pg.waitForFunction(
    want => document.querySelectorAll('tr[data-row-src]').length === want,
    n, { timeout },
  ).catch(() => {});
}

/* Espera a que el detalle muestre la imagen del frame pedido (el swap tiene exit anim). */
async function waitDetailSrc(pg, src, timeout = 7000) {
  await pg.waitForFunction(
    want => (document.querySelector('#tour-detalle img')?.getAttribute('src') || '').includes(want),
    src, { timeout },
  ).catch(() => {});
}

async function skipCutscene(pg) {
  if (await pg.locator('#cutscene').count()) {
    await pg.keyboard.press('Escape');
    await pg.waitForSelector('#cutscene', { state: 'detached', timeout: 6000 }).catch(() => {});
    await pg.waitForTimeout(300);
  }
}

// ── Contexto 1: app completa ──
const ctx = await browser.newContext({ viewport: { width: 1366, height: 768 }, acceptDownloads: true });
const page = await ctx.newPage();
const GL_NOISE = /GL Driver Message|GPU stall due to ReadPixels/;  // artefacto headless/SwiftShader
page.on('console', m => { const t = m.type(); if ((t === 'error' || t === 'warning') && !GL_NOISE.test(m.text())) problems.push(`consola.${t}: ${m.text().slice(0, 160)}`); });
page.on('pageerror', e => problems.push('pageerror: ' + e.message.slice(0, 160)));
page.on('requestfailed', r => problems.push('reqfail: ' + r.url().slice(0, 120)));

await page.goto(BASE, { waitUntil: 'networkidle' });
await page.waitForSelector('[data-card-src]');
await skipCutscene(page);
await page.waitForSelector('.driver-popover', { timeout: 9000 });

// Tour de primera vez
check((await page.locator('.driver-popover').count()) === 1, 'tour visible en la primera visita');
await page.screenshot({ path: SHOTS + '/v4_tour.png' });
await page.locator('.driver-popover button', { hasText: 'Saltar' }).click().catch(() => page.keyboard.press('Escape'));
await page.waitForTimeout(500);
check((await page.locator('.driver-popover').count()) === 0, 'tour cerrado');
check(await page.evaluate(() => localStorage.getItem('lb135-tour-v4') === '1'), 'tour marcado como hecho en localStorage');
await page.reload({ waitUntil: 'networkidle' });
await page.waitForSelector('[data-card-src]');
await page.waitForTimeout(500);
check((await page.locator('.driver-popover').count()) === 0, 'tour NO reaparece tras reload');

// Resumen narrativo
const narr = await page.locator('main p').first().innerText();
check(/cap_0006/.test(narr) && /81\.0/.test(narr), 'resumen narrativo calculado (momento crítico cap_0006)');
await page.locator('button', { hasText: 'Ver momento crítico' }).click();
await page.waitForTimeout(700);
check((await page.locator('#tour-detalle h2').first().innerText()) === 'cap_0006', 'botón crítico selecciona y scrollea al detalle');

// KPIs
check((await page.locator('main').innerText()).includes('12'), 'KPI de frames presente');

// Mismo detalle desde corredor / tabla / alertas / curva
await page.click('[data-card-src="cap_0005"]');
await waitDetailSrc(page, 'cap_0005');
const imgCard = await page.locator('#tour-detalle img').first().getAttribute('src');
check(/cap_0005_(edsr|evid)/.test(imgCard || ''), 'card del corredor → detalle cap_0005 (mejor fuente disponible)');
await page.click('tr[data-row-src="cap_0005"]');
await waitDetailSrc(page, 'cap_0005');
check((await page.locator('#tour-detalle img').first().getAttribute('src')) === imgCard, 'fila de tabla → mismo detalle');
await page.click('#tour-alertas li button >> nth=0');
await page.waitForTimeout(300);
const alertTxt = await page.locator('#tour-alertas li button >> nth=0').innerText();
const alertSrc = alertTxt.split('\n')[1]?.split(' ')[0] ?? '';
check((await page.locator('#tour-detalle h2').first().innerText()) === alertSrc, `alerta selecciona su frame (${alertSrc})`);
const dots = page.locator('[data-dot-src]');
check((await dots.count()) === EXP.n, `curva con ${EXP.n} puntos`);
await dots.nth(9).dispatchEvent('click');  // click programático: el anillo pulsante vuelve inestable el hit-test
await page.waitForTimeout(300);
check((await page.locator('#tour-detalle h2').first().innerText()) === 'cap_0009', 'punto de la curva → detalle cap_0009');

// Extensiones DPD: panel de Muestreo y trayectoria GPS en la vista Vuelo
check(await page.getByText('Prioridad del sampler').count() >= 1, 'panel de muestreo presente');
check(await page.getByText('Cobertura por clase').count() >= 1, 'cobertura por clase presente');
check(await page.getByText('Trayectoria GPS del descenso').count() >= 1, 'trayectoria GPS presente');
// Basemap offline (MapOffline): tiles locales + track SVG. Si el tile central
// no existe, MapOffline cae al scatter sin basemap (GpsTrack) — ambos válidos.
const nMapa = await page.locator('img[src^="/tiles/"]').count();
const nScatter = await page.locator('.recharts-scatter').count();
check(nMapa >= 1 || nScatter >= 1,
  `trayectoria renderizada (${nMapa} tiles de basemap / ${nScatter} scatter)`);
if (nMapa >= 1) {
  check(await page.locator('svg polyline').count() >= 2, 'trazo SVG sobre el basemap');
}

// Política fire_only_v1: badge SOLO cuando el modelo confirmó incendio.
const samples = await page.evaluate(async () => (await fetch('/api/samples')).json());
const vals = Object.values(samples.samples || {});
const conf = Object.entries(samples.samples || {}).filter(([, s]) => s.tipo_estado === 'confirmado_por_modelo');
const malConfiable = vals.filter(s => s.tipo_es_confiable === true && !(s.tipo_desastre === 'incendio' && s.tipo_estado === 'confirmado_por_modelo'));
const malEstado = vals.filter(s => s.tipo_estado && s.tipo_estado !== 'confirmado_por_modelo' && s.tipo_es_confiable === true);
check(malConfiable.length === 0, `solo incendio confirmado es confiable (${malConfiable.length} violaciones)`);
check(malEstado.length === 0, `estados no confirmados nunca son confiables (${malEstado.length} violaciones)`);
if (conf.length > 0) {
  await page.click(`[data-card-src="${conf[0][0]}"]`);
  await page.waitForTimeout(500);
  const badges = await page.locator('text=/evento: /').allInnerTexts();
  check(badges.some(t => t.includes('incendio')), `badge de incendio confirmado visible (${conf[0][0]})`);
} else {
  check((await page.locator('text=/evento: /').count()) === 0, 'sin confirmados: ningún badge de tipo visible');
}
check(await page.getByText('modelos:', { exact: false }).count() >= 1, 'trazabilidad de modelos (contrato v3)');
check(await page.getByText('medido', { exact: false }).count() >= 1, 'colapso con fuente (medido/supuesto)');

// Tabs de imagen degradadas
await page.click('[data-card-src="cap_0000"]');
await waitDetailSrc(page, 'cap_0000');
check((await page.locator('#tour-detalle [role="tab"]').count()) === 4, 'cap_0000: 4 tabs de imagen');
await page.click('[data-card-src="cap_0001"]');
await page.waitForTimeout(300);
const tabs1 = await page.locator('#tour-detalle [role="tab"]').allInnerTexts();
check(!tabs1.includes('Ensemble') && !tabs1.includes('EDSR'), 'cap_0001: sin tabs Ensemble/EDSR (archivos inexistentes)');
check(tabs1.length === 2, `cap_0001: 2 tabs (${tabs1.join(', ')})`);
await page.click('[data-card-src="cap_0000"]');
await waitDetailSrc(page, 'cap_0000');
await page.locator('#tour-detalle [role="tab"]').filter({ hasText: 'EDSR' }).click();
await page.waitForTimeout(900);
check((await page.locator('#tour-detalle img').first().getAttribute('src') || '').includes('_edsr'), 'tab EDSR cambia la imagen');

// Teclado
await page.keyboard.press('ArrowDown');
await page.waitForTimeout(250);
const selAfterDown = await page.locator('#tour-detalle h2').first().innerText();
check(selAfterDown === 'cap_0001', `↑↓ navegan (cap_0000 → ${selAfterDown})`);
await page.keyboard.press('a');
await waitRows(page, EXP.cards);
check((await page.locator('tr[data-row-src]').count()) === EXP.cards, `A → solo alertas (${EXP.cards} filas)`);
await page.keyboard.press('a');
await waitRows(page, EXP.n);
await page.keyboard.press('/');
await page.type('#busca-src', '0009');
await waitRows(page, 1);
check((await page.locator('tr[data-row-src]').count()) === 1, 'búsqueda por src → 1 fila');
await page.keyboard.press('Escape');
await waitRows(page, EXP.n);
check((await page.locator('tr[data-row-src]').count()) === EXP.n, 'Esc limpia la búsqueda');

// Export CSV
await page.locator('button', { hasText: /^High$/ }).first().click();
await page.waitForTimeout(300);
const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('button', { hasText: 'Exportar CSV' }).click()]);
const dlPath = path.join(os.tmpdir(), 'exp_v4.csv');
await dl.saveAs(dlPath);
check(fs.readFileSync(dlPath, 'utf8').trim().split(/\r?\n/).length === EXP.high + 1, `CSV exportado con ${EXP.high} filas HIGH`);
const limpiar = page.locator('button', { hasText: 'Limpiar' });
if (await limpiar.isEnabled()) await limpiar.click();
await page.waitForTimeout(300);

// Vistas
await page.getByRole('tab', { name: 'Post-vuelo', exact: true }).click();
await page.waitForTimeout(700);
check(await page.locator('.recharts-pie').count() >= 1, 'donut de veredictos renderizado');
check((await page.locator('#tour-detalle').count()) === 0, 'vista post sin detalle duplicado');
// Extensiones DPD: KPI de estimación de pérdidas (summary.perdidas)
check(await page.getByText('Afectados (est.)').count() >= 1, 'KPI de afectados estimados presente');
check(await page.getByText('Pérdidas (est.)').count() >= 1, 'KPI de pérdidas estimadas presente');
const galFigures = await page.locator('figure').count();
check(galFigures === EXP.enhanced + EXP.ensSeg, `galerías con ${EXP.enhanced + EXP.ensSeg} thumbs`);
await page.screenshot({ path: SHOTS + '/v4_post.png' });
await page.locator('figure', { hasText: 'cap_0002' }).first().click();
await page.waitForTimeout(600);
check((await page.locator('#tour-detalle h2').first().innerText()) === 'cap_0002', 'thumb de galería abre el detalle');

await page.getByRole('tab', { name: 'Informe', exact: true }).click();
await page.waitForTimeout(600);
const doc = await page.locator('#informe-print').innerText();
check(/INFORME DE MISI/i.test(doc) && /firmas|asesor/i.test(doc), 'informe con contenido y firmas');
await page.screenshot({ path: SHOTS + '/v4_informe.png', fullPage: true });

// Presentación
await page.getByRole('tab', { name: 'Vuelo', exact: true }).click();
await page.waitForTimeout(500);
await page.locator('button', { hasText: 'Presentar' }).click();
await page.waitForTimeout(600);
check(await page.locator('text=Salir (Esc)').isVisible(), 'modo presentación activo');
const p1 = await page.locator('.fixed.z-50 .font-mono').first().innerText();
await page.waitForTimeout(4300);
const p2 = await page.locator('.fixed.z-50 .font-mono').first().innerText();
check(p1 !== p2, `presentación avanza sola (${p1} → ${p2})`);
await page.screenshot({ path: SHOTS + '/v4_present.png' });
await page.keyboard.press('Escape');
await page.waitForTimeout(1400);
check((await page.locator('text=Salir (Esc)').count()) === 0, 'Esc sale de la presentación');

// Capturas finales 1366 y 1920
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(400);
await page.screenshot({ path: SHOTS + '/v4_1366.png' });
await page.setViewportSize({ width: 1920, height: 1080 });
await page.waitForTimeout(700);
await page.screenshot({ path: SHOTS + '/v4_1920.png' });
await page.setViewportSize({ width: 375, height: 812 });
await page.waitForTimeout(700);
const mobileLayout = await page.evaluate(() => ({
  overflow: document.documentElement.scrollWidth > window.innerWidth + 2,
  header: Boolean(document.querySelector('header')),
  main: Boolean(document.querySelector('#main-content')),
}));
check(!mobileLayout.overflow, 'móvil 375px sin overflow horizontal');
check(mobileLayout.header && mobileLayout.main, 'móvil conserva navegación y contenido');
await page.screenshot({ path: SHOTS + '/v4_mobile.png', fullPage: true });
await ctx.close();

// ── Contexto 2: auto-refresh sin recarga sobre copia aislada ──
const LIVE = path.join(os.tmpdir(), 'lb135-v4');
fs.rmSync(LIVE, { recursive: true, force: true });
fs.mkdirSync(LIVE, { recursive: true });
for (const d of ['outputs', 'entrega', 'web', 'web-app']) fs.cpSync(at(d), path.join(LIVE, d), { recursive: true, filter: (src) => !src.includes('node_modules') });
const python = process.platform === 'win32' ? 'python' : 'python3';
const srv2 = spawn(python, [at('web_server.py'), '--port', '8004', '--root', LIVE], { stdio: 'ignore', detached: true });
await new Promise(r => setTimeout(r, 1800));
const ctx2 = await browser.newContext({ viewport: { width: 1366, height: 768 } });
const page2 = await ctx2.newPage();
page2.on('pageerror', e => problems.push('page2 pageerror: ' + e.message.slice(0, 120)));
page2.on('console', m => { if (m.type() === 'error') problems.push('page2 consola: ' + m.text().slice(0, 120)); });
await page2.goto('http://localhost:8004', { waitUntil: 'networkidle' });
await page2.waitForSelector('[data-card-src]');
await skipCutscene(page2);
await page2.waitForSelector('.driver-popover', { timeout: 9000 }).catch(() => {});
await page2.locator('.driver-popover button', { hasText: 'Saltar' }).click().catch(() => {});
const n0 = await page2.locator('tr[data-row-src]').count();
const liveCsv = path.join(LIVE, 'outputs', 'mission', 'telemetry.csv');
fs.writeFileSync(liveCsv, fs.readFileSync(liveCsv, 'utf8').trim() + '\r\n91.5,0.9,1013.1,26.4,12.0,3.0,0.0,85.0,0.0,3,0.19,-0.58,SUELO EXPUESTO,0,0,0.0,0.0,SIN DESASTRE,0,48.0,cap_9000,LOW,0.08');
await page2.waitForTimeout(4200);
check((await page2.locator('tr[data-row-src]').count()) === n0 + 1, `auto-refresh agrega frame nuevo (${n0} → ${n0 + 1}) sin recarga`);
check((await page2.locator('[data-card-src="cap_9000"]').count()) === 1, 'el corredor incorpora el frame nuevo');
fs.renameSync(liveCsv, liveCsv + '.hidden');
await page2.waitForTimeout(4200);
check((await page2.locator('[data-card-src]').count()) === 0, 'sin CSV → corredor vacío sin errores');
fs.renameSync(liveCsv + '.hidden', liveCsv);
await page2.waitForTimeout(4200);
check((await page2.locator('tr[data-row-src]').count()) === n0 + 1, 'recupera los frames al volver el CSV');
await ctx2.close();
try {
  if (process.platform === 'win32') spawnSync('taskkill', ['/pid', String(srv2.pid), '/t', '/f']);
  else process.kill(-srv2.pid);
} catch { /* ok */ }

await browser.close();
console.log('\n──────── RESUMEN V4 ────────');
if (problems.length) {
  console.log('PROBLEMAS (' + problems.length + '):');
  problems.forEach(p => console.log('  ✗ ' + p));
  process.exit(1);
} else {
  console.log('TODO VERDE ✔  (build production servido por Python, cero errores de consola)');
}
