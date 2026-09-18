// smoke.mjs — prueba de humo E2E de la Estación Terrena v2 (Chromium headless)
// Criterios: cero errores de consola · un click en cualquier thumb/fila/alerta/punto
// abre el MISMO detalle · teclado (↑↓, A, Enter, Esc) · tabs de imagen se ocultan si
// el archivo no existe · auto-refresh sin recarga · vistas · presentación · informe.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const at = (...seg) => path.join(ROOT, ...seg);
const BASE = process.env.BASE || 'http://localhost:8000';
const SHOTS = '/tmp/shots';
fs.mkdirSync(SHOTS, { recursive: true });

const problems = [];
const log = (...a) => console.log('[smoke]', ...a);
const check = (cond, msg) => {
  if (cond) log('OK  ', msg);
  else { log('FAIL', msg); problems.push(msg); }
};

// Expectativas derivadas del contrato real (no hardcodeadas).
const SUMMARY = JSON.parse(fs.readFileSync(at('entrega', 'summary.json'), 'utf8'));
const CSV = fs.readFileSync(at('outputs', 'mission', 'telemetry.csv'), 'utf8').trim().split(/\r?\n/);
const HEAD = CSV[0].split(',');
const ROWS = CSV.slice(1).map(l => Object.fromEntries(l.split(',').map((v, i) => [HEAD[i], v])));
const SEV = { 'INUNDACION SEVERA': 4, 'INUNDACION URBANA': 3, 'POSIBLE SISMO/VIENTO': 3, 'POSIBLE INCENDIO/EROSION': 2 };
const EXP = {
  n: ROWS.length,
  alerts: ROWS.filter(r => +r.alert === 1).length,
  cards: ROWS.filter(r => +r.alert === 1 || (SEV[r.diag] || 0) >= 2).length,
  high: ROWS.filter(r => r.sample_pri === 'HIGH').length,
  people: String(SUMMARY.personas_total),
  veh: String(SUMMARY.vehiculos_total),
  enhanced: fs.readdirSync(at('entrega', 'enhanced')).length,
  ensSeg: fs.readdirSync(at('entrega', 'ens_seg')).length,
  verdicts: new Set(ROWS.map(r => r.verdict)).size,
};

const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage'] });
const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, acceptDownloads: true });
page.on('console', m => {
  const t = m.type();
  if (t === 'error' || t === 'warning') problems.push(`consola.${t}: ${m.text()}`);
});
page.on('pageerror', e => problems.push('pageerror: ' + e.message));
page.on('requestfailed', r => problems.push('requestfailed: ' + r.url() + ' ' + (r.failure()?.errorText || '')));

await page.goto(BASE, { waitUntil: 'networkidle' });
await page.waitForSelector('#tbody tr');
await page.waitForTimeout(700);

// ── 1. KPIs (5) y header mínimo ──
check(await page.locator('#kpiFrames').innerText() === String(EXP.n), `KPI frames ${EXP.n}`);
check(await page.locator('#kpiAlerts').innerText() === String(EXP.alerts), `KPI alertas ${EXP.alerts}`);
check((await page.locator('#kpiDetect').innerText()).replace(/\s/g, '') === `${EXP.people}/${EXP.veh}`, 'KPI personas/vehículos juntos');
check((await page.locator('#kpiAlt').innerText()).includes('→'), 'KPI altitud max → min');
check((await page.locator('#kpiDmg').innerText()).includes('%'), 'KPI daño promedio');
check((await page.locator('.kpi').count()) === 5, 'exactamente 5 KPIs');
check((await page.locator('#connDot').getAttribute('class')).includes('is-live'), 'dot de estado vivo');
check(await page.locator('#toast').isHidden(), 'sin toast espurio en la primera carga');

// ── 2. Corredor vertical: eje de altitud + tarjetas ──
check((await page.locator('#descentCanvas .dcard').count()) === EXP.n, `corredor con ${EXP.n} tarjetas`);
check((await page.locator('#descentCanvas .tick').count()) >= 5, 'eje de altitud con marcas');
check((await page.locator('#descentCanvas .dring').count()) === EXP.alerts, `${EXP.alerts} anillos de alerta`);

// ── 3. Un click en cualquier lado abre el MISMO detalle ──
await page.click('#descentCanvas .dcard[data-src="cap_0005"]');
await page.waitForTimeout(350);
check(await page.locator('#dtTitle').innerText() === 'cap_0005', 'card del corredor → detalle cap_0005');
const imgFromCard = await page.locator('#dtImg').getAttribute('src');
check((imgFromCard || '').includes('cap_0005_evid'), 'imagen = evidencia de cap_0005');

await page.click('#tbody tr[data-src="cap_0005"]');
await page.waitForTimeout(250);
check((await page.locator('#dtImg').getAttribute('src')) === imgFromCard, 'fila de tabla → mismo detalle/imagen');

await page.click('#alertsBody .alert-card[data-src="cap_0009"]');
await page.waitForTimeout(250);
check(await page.locator('#dtTitle').innerText() === 'cap_0009', 'alerta → detalle cap_0009');

await page.click('#spark .pt[data-src="cap_0007"]');
await page.waitForTimeout(250);
check(await page.locator('#dtTitle').innerText() === 'cap_0007', 'punto del sparkline → detalle cap_0007');

// ── 4. Tabs de imagen: se ocultan si el archivo no existe ──
await page.click('#descentCanvas .dcard[data-src="cap_0000"]');
await page.waitForTimeout(250);
check((await page.locator('#imgTabs button').count()) === 4, 'cap_0000 muestra 4 tabs de imagen');
await page.click('#descentCanvas .dcard[data-src="cap_0001"]');
await page.waitForTimeout(250);
check((await page.locator('#imgTabs button[data-key="ens_seg"]').count()) === 0, 'cap_0001 sin tab Ensemble (no existe)');
check((await page.locator('#imgTabs button[data-key="enhanced"]').count()) === 0, 'cap_0001 sin tab EDSR (no existe)');
check((await page.locator('#imgTabs button').count()) === 2, 'cap_0001 con 2 tabs (evidencia + high-res)');
await page.click('#descentCanvas .dcard[data-src="cap_0000"]');
await page.waitForTimeout(200);
await page.click('#imgTabs button[data-key="enhanced"]');
await page.waitForTimeout(300);
check((await page.locator('#dtImg').getAttribute('src') || '').includes('_edsr'), 'tab EDSR cambia la imagen');
await page.click('#imgTabs button[data-key="vis"]');
await page.waitForTimeout(200);

// ── 5. Teclado: ↑↓ navega, A filtra, Enter selecciona, Esc limpia ──
const before = await page.locator('#dtTitle').innerText();
await page.keyboard.press('ArrowDown');
await page.waitForTimeout(250);
const after = await page.locator('#dtTitle').innerText();
check(after !== before, `↑/↓ navegan (${before} → ${after})`);
await page.keyboard.press('a');
await page.waitForTimeout(250);
check((await page.locator('#tbody tr').count()) === EXP.cards, `A → solo alertas (${EXP.cards} filas)`);
await page.keyboard.press('ArrowDown');
await page.waitForTimeout(200);
await page.keyboard.press('Enter');
await page.waitForTimeout(250);
const selRow = await page.locator('#tbody tr.is-sel').getAttribute('data-src');
check(selRow === (await page.locator('#dtTitle').innerText()), 'Enter confirma la selección en el detalle');
await page.keyboard.press('a');
await page.waitForTimeout(250);
check((await page.locator('#tbody tr').count()) === EXP.n, 'A de nuevo → todas las filas');
await page.keyboard.press('/');
await page.type('#inpSearch', '0009');
await page.waitForTimeout(300);
check((await page.locator('#tbody tr').count()) === 1, 'búsqueda por src → 1 fila');
await page.keyboard.press('Escape');
await page.waitForTimeout(250);
check((await page.locator('#tbody tr').count()) === EXP.n, 'Esc limpia la búsqueda');

// ── 6. Orden y filtros ──
await page.click('th[data-sort="danado_pct"]');
await page.waitForTimeout(200);
check((await page.locator('#tbody tr').first().getAttribute('data-src')) === 'cap_0006', 'orden por daño → cap_0006 primero');
await page.click('th[data-sort="t_s"]');
await page.waitForTimeout(150);
await page.click('.chip[data-pri="HIGH"]');
await page.waitForTimeout(200);
check((await page.locator('#tbody tr').count()) === EXP.high, `filtro HIGH → ${EXP.high} filas`);

// ── 7. Export CSV del filtro activo ──
const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#btnExport')]);
const dlPath = '/tmp/export_v2.csv';
await dl.saveAs(dlPath);
const lines = fs.readFileSync(dlPath, 'utf8').trim().split(/\r?\n/);
check(lines.length === EXP.high + 1, `CSV con ${EXP.high} filas + cabecera`);
await page.click('#btnClearFilters');
await page.waitForTimeout(200);

// ── 8. Colapsables ──
await page.click('#btnTblToggle');
await page.waitForTimeout(200);
check(await page.locator('#tblBody').isHidden(), 'tabla colapsada');
await page.click('#btnTblToggle');
await page.waitForTimeout(200);
check(await page.locator('#tblBody').isVisible(), 'tabla expandida');
await page.click('#btnAlertCollapse');
await page.waitForTimeout(200);
check(await page.locator('#alertsBody').isHidden(), 'alertas colapsadas');
await page.click('#btnAlertCollapse');
await page.waitForTimeout(200);

// ── 9. Vistas: post-vuelo y galerías que abren el detalle ──
await page.click('.tab[data-view="post"]');
await page.waitForTimeout(400);
check(await page.locator('#view-post').isVisible(), 'vista post-vuelo visible');
check((await page.locator('#donut circle').count()) === EXP.verdicts + 1, `donut con ${EXP.verdicts} veredictos`);
check((await page.locator('#galEnhanced figure').count()) === EXP.enhanced, `galería EDSR con ${EXP.enhanced}`);
check((await page.locator('#galEns figure').count()) === EXP.ensSeg, `galería ensemble con ${EXP.ensSeg}`);
check((await page.locator('#view-post .alert-card').count()) === 0, 'post-vuelo sin lista duplicada de alertas');
await page.screenshot({ path: SHOTS + '/v2_post.png' });

await page.click('#galEns figure[data-src="cap_0002"]');
await page.waitForTimeout(400);
check(await page.locator('#view-vuelo').isVisible(), 'click en galería vuelve a Vuelo');
check(await page.locator('#dtTitle').innerText() === 'cap_0002', 'galería abre el detalle de cap_0002');
check((await page.locator('#imgTabs button.is-active').innerText()) === 'Ensemble', 'con la tab Ensemble activa');

// ── 10. Informe ──
await page.click('.tab[data-view="informe"]');
await page.waitForTimeout(400);
const doc = await page.locator('#docBody').innerText();
check(/informe de misi.n/i.test(doc), 'informe con título');
check(/alertas y diagn.sticos/i.test(doc), 'informe con tabla de alertas');
check(/top 5 frames prioridad HIGH/i.test(doc), 'informe con top-5 HIGH');
check(/corredor de vuelo/i.test(doc), 'informe con corredor');
check((await page.locator('#docBody .rp-sign div').count()) === 3, 'informe con bloque de firmas');
await page.screenshot({ path: SHOTS + '/v2_informe.png', fullPage: true });

// ── 11. Modo presentación ──
await page.click('.tab[data-view="vuelo"]');
await page.waitForTimeout(300);
await page.click('#btnPresent');
await page.waitForTimeout(500);
check(await page.locator('#present').isVisible(), 'modo presentación activo');
const p1 = await page.locator('#pSrc').innerText();
check(p1.length > 0, 'presentación muestra un frame (' + p1 + ')');
await page.waitForTimeout(4300);
const p2 = await page.locator('#pSrc').innerText();
check(p2 !== p1, `presentación avanza sola (${p1} → ${p2})`);
await page.screenshot({ path: SHOTS + '/v2_present.png' });
await page.keyboard.press('Escape');
await page.waitForTimeout(300);
check(await page.locator('#present').isHidden(), 'Esc sale de la presentación');

// ── 12. Auto-refresh sin recarga (segundo servidor, copia aislada) ──
const { spawn } = await import('node:child_process');
const LIVE = '/tmp/lb135-live2';
fs.rmSync(LIVE, { recursive: true, force: true });
fs.mkdirSync(LIVE, { recursive: true });
for (const d of ['outputs', 'entrega', 'web']) fs.cpSync(at(d), path.join(LIVE, d), { recursive: true });
const srv2 = spawn('python3', [at('web_server.py'), '--port', '8003', '--root', LIVE], { stdio: 'ignore', detached: true });
await new Promise(r => setTimeout(r, 1800));
const page2 = await browser.newPage({ viewport: { width: 1366, height: 768 } });
page2.on('pageerror', e => problems.push('page2 pageerror: ' + e.message));
page2.on('console', m => { if (m.type() === 'error') problems.push('page2 consola.error: ' + m.text()); });
await page2.goto('http://localhost:8003', { waitUntil: 'networkidle' });
await page2.waitForSelector('#tbody tr');
const n0 = await page2.locator('#tbody tr').count();
const liveCsv = path.join(LIVE, 'outputs', 'mission', 'telemetry.csv');
fs.writeFileSync(liveCsv, fs.readFileSync(liveCsv, 'utf8').trim() +
  '\r\n91.5,0.9,1013.1,26.4,12.0,3.0,0.0,85.0,0.0,3,0.19,-0.58,SUELO EXPUESTO,0,0,0.0,0.0,SIN DESASTRE,0,48.0,cap_9000,LOW,0.08');
await page2.waitForTimeout(4200);
check((await page2.locator('#tbody tr').count()) === n0 + 1, `auto-refresh agrega el frame nuevo (${n0} → ${n0 + 1}) sin recarga`);
check((await page2.locator('#descentCanvas .dcard').count()) === n0 + 1, 'el corredor también lo incorpora');
check((await page2.locator('#toast').innerText()).includes('cap_9000'), 'toast de frame nuevo');
fs.renameSync(liveCsv, liveCsv + '.hidden');
await page2.waitForTimeout(4200);
check((await page2.locator('#kpiFrames').innerText()) === '—', 'sin CSV → KPI en —');
check((await page2.locator('#connDot').getAttribute('class')).includes('is-warn'), 'sin CSV → dot ámbar');
check(await page2.locator('#descEmpty').isVisible(), 'sin CSV → corredor con estado vacío');
fs.renameSync(liveCsv + '.hidden', liveCsv);
await page2.waitForTimeout(4200);
check((await page2.locator('#tbody tr').count()) === n0 + 1, 'recupera los frames al volver el CSV');
await page2.close();
try { process.kill(-srv2.pid); } catch { /* ya muerto */ }

// ── 13. Capturas finales ──
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(400);
await page.screenshot({ path: SHOTS + '/v2_vuelo.png' });
await page.setViewportSize({ width: 1600, height: 900 });
await page.waitForTimeout(600);
await page.screenshot({ path: SHOTS + '/v2_vuelo_1600.png' });

await browser.close();
console.log('\n──────── RESUMEN ────────');
if (problems.length) {
  console.log('PROBLEMAS (' + problems.length + '):');
  problems.forEach(p => console.log('  ✗ ' + p));
  process.exit(1);
} else {
  console.log('TODO VERDE ✔  (cero errores de consola, criterios v2 cumplidos)');
}
