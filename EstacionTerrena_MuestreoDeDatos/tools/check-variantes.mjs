// check-variantes.mjs — variantes automatizables del CHECKLIST-SIMULACRO.md.
//
// Verifica sin intervención humana:
//   1. Sin post-vuelo: Post-vuelo muestra empty state y no rompe (necesita un
//      root servido SIN entrega/summary.json, ver comando abajo).
//   2. Sin WebGL: la escena degrada al perfil SVG (no hay canvas ni errores).
//   3. Resolución chica: 1366×768 sin overflow horizontal.
//
// Uso:
//   # root sin post-vuelo (se prepara con simulacro --sin-postvuelo):
//   node tools/check-variantes.mjs --base http://127.0.0.1:8765 \
//        --base-sin-post http://127.0.0.1:8766
//
// Salida: líneas [variantes] OK/FAIL; exit 1 si algo falla. Capturas en
// tools/shots/checklist/ para adjuntar al registro del checklist.
import { chromium } from 'playwright';
import fs from 'node:fs';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};
const BASE = opt('--base', 'http://localhost:8000');
const BASE_SIN_POST = opt('--base-sin-post', null);
const OUT = 'tools/shots/checklist';
fs.mkdirSync(OUT, { recursive: true });

const problems = [];
const check = (cond, msg) => {
  console.log(`[variantes] ${cond ? 'OK  ' : 'FAIL'} ${msg}`);
  if (!cond) problems.push(msg);
};
const GL_NOISE = /GL Driver Message|GPU stall|Automatic fallback to software/i;

function vigilar(page) {
  const errs = [];
  page.on('console', m => {
    const t = m.type();
    if ((t === 'error' || t === 'warning') && !GL_NOISE.test(m.text())) {
      errs.push(`${t}: ${m.text().slice(0, 140)}`);
    }
  });
  page.on('pageerror', e => errs.push('pageerror: ' + e.message.slice(0, 140)));
  return errs;
}

async function saltarCutscene(page) {
  if (await page.locator('#cutscene').count()) {
    await page.keyboard.press('Escape');
    await page.waitForSelector('#cutscene', { state: 'detached', timeout: 6000 }).catch(() => {});
    await page.waitForTimeout(300);
  }
}

const ARGS = ['--no-sandbox', '--disable-dev-shm-usage', '--js-flags=--jitless'];

// ── Variante 2: sin WebGL + variante 3: 1366×768 ──────────────────────────
const browser = await chromium.launch({ args: ARGS });

// 3) Resolución chica (1366×768, el default del smoke): sin overflow horizontal.
{
  const ctx = await browser.newContext({ viewport: { width: 1366, height: 768 } });
  const page = await ctx.newPage();
  const errs = vigilar(page);
  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('[data-card-src]', { timeout: 15000 });
  await saltarCutscene(page);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth);
  check(overflow <= 1, `1366×768 sin overflow horizontal (Δ=${overflow}px)`);
  await page.screenshot({ path: `${OUT}/v3_1366.png` });
  check(errs.length === 0, `1366×768 sin errores de consola (${errs.join(' | ') || 'limpio'})`);
  await ctx.close();
}

// 2) Sin WebGL: la escena degrada al perfil SVG, la app sigue usable.
{
  const sinGl = await chromium.launch({ args: [...ARGS, '--disable-gpu', '--disable-webgl'] });
  const ctx = await sinGl.newContext({ viewport: { width: 1366, height: 768 } });
  const page = await ctx.newPage();
  const errs = vigilar(page);
  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('[data-card-src]', { timeout: 15000 });
  await saltarCutscene(page);
  const hayGl = await page.evaluate(() => {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  });
  const perfilSVG = await page.locator('svg[viewBox="0 0 100 100"]').count();
  check(!hayGl, 'sin WebGL: el navegador no expone WebGL');
  check(perfilSVG >= 1, `sin WebGL: perfil SVG presente (${perfilSVG})`);
  check(errs.length === 0, `sin WebGL sin errores de consola (${errs.join(' | ') || 'limpio'})`);
  await page.screenshot({ path: `${OUT}/v2_sin_webgl.png` });
  await sinGl.close();
}

await browser.close();

// ── Variante 1: sin post-vuelo ────────────────────────────────────────────
if (BASE_SIN_POST) {
  const b = await chromium.launch({ args: ARGS });
  const ctx = await b.newContext({ viewport: { width: 1366, height: 768 } });
  const page = await ctx.newPage();
  const errs = vigilar(page);
  const health = await page.evaluate(async url =>
    (await fetch(url + '/api/health')).json(), BASE_SIN_POST);
  check(health.summary_exists === false, 'sin post-vuelo: /api/health sin summary.json');
  await page.goto(BASE_SIN_POST, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('[data-card-src]', { timeout: 15000 });
  await saltarCutscene(page);
  await page.getByRole('tab', { name: 'Post-vuelo', exact: true }).click();
  await page.waitForTimeout(800);
  const vacio = await page.getByText('Análisis post-vuelo pendiente.', { exact: false }).count();
  const aviso = await page.getByText('sin summary.json', { exact: false }).count();
  check(vacio >= 1 || aviso >= 1,
    `sin post-vuelo: la vista Post degrada con aviso (${vacio ? 'empty state' : 'aviso + galerías'})`);
  check(errs.length === 0, `sin post-vuelo sin errores de consola (${errs.join(' | ') || 'limpio'})`);
  await page.screenshot({ path: `${OUT}/v1_sin_postvuelo.png` });
  await b.close();
} else {
  console.log('[variantes] SKIP sin post-vuelo (pasá --base-sin-post)');
}

if (problems.length) {
  console.log(`\n[variantes] ${problems.length} FALLA(S):`);
  problems.forEach(p => console.log('  -', p));
  process.exit(1);
}
console.log('\n[variantes] TODO VERDE');
