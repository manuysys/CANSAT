// Video de RESPALDO de la demo: recorre la estación completa (Vuelo → detalle
// con Grad-CAM → Post → Informe → Jurado → Presentación) y lo guarda como
// webm para usarlo si algo falla el día de la presentación.
//
// Uso:  python web_server.py   (en otra terminal)
//       node tools/demo_video.mjs
// Salida: tools/shots/demo_respaldo.webm
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const BASE = process.env.BASE || 'http://localhost:8000';
const OUT = path.join(ROOT, 'tools', 'shots', 'demo_respaldo.webm');
fs.mkdirSync(path.dirname(OUT), { recursive: true });

const browser = await chromium.launch({
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader'],
});
const ctx = await browser.newContext({
  viewport: { width: 1600, height: 900 },
  recordVideo: { dir: os.tmpdir(), size: { width: 1600, height: 900 } },
});
const page = await ctx.newPage();
const pausa = (ms) => page.waitForTimeout(ms);

await page.goto(BASE, { waitUntil: 'networkidle' });
await page.waitForSelector('[data-card-src]');
if (await page.locator('#cutscene').count()) {
  await page.keyboard.press('Escape');
  await page.waitForSelector('#cutscene', { state: 'detached' }).catch(() => {});
}
await page.waitForSelector('.driver-popover', { timeout: 9000 }).catch(() => {});
await page.locator('.driver-popover button', { hasText: 'Saltar' })
  .click().catch(() => page.keyboard.press('Escape'));
await pausa(1200);

// 1) Vuelo: corredor + briefing
await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
await pausa(3000);

// 2) Frame crítico en el detalle
await page.click('[data-card-src="cap_0006"]').catch(() => {});
await pausa(2500);

// 3) Grad-CAM (explicabilidad)
await page.locator('[data-testid="gradcam-pedir"]').click().catch(() => {});
await page.locator('[data-testid="gradcam-overlay"]')
  .waitFor({ timeout: 120000 }).catch(() => {});
await pausa(3000);

// 4) Post-vuelo
await page.getByRole('tab', { name: 'Post-vuelo', exact: true }).click().catch(() => {});
await pausa(3500);

// 5) Informe
await page.getByRole('tab', { name: 'Informe', exact: true }).click().catch(() => {});
await pausa(3500);

// 6) Modo Jurado (deck)
await page.locator('button[aria-label="Modo jurado"]').click().catch(() => {});
await pausa(4000);
await page.keyboard.press('Escape').catch(() => {});
await pausa(1200);

// 7) Presentación automática (avanza sola)
await page.getByRole('tab', { name: 'Vuelo', exact: true }).click().catch(() => {});
await pausa(800);
await page.locator('button', { hasText: 'Presentar' }).click().catch(() => {});
await pausa(5000);
await page.keyboard.press('Escape').catch(() => {});
await pausa(1000);

const video = page.video();
await page.close();               // finaliza la grabación
if (video) {
  await video.saveAs(OUT);        // válido con la página cerrada y el browser vivo
  const mb = (fs.statSync(OUT).size / 1e6).toFixed(1);
  console.log(`[OK] video de respaldo: ${OUT} (${mb} MB)`);
} else {
  console.log('[WARN] Playwright no devolvió video');
}
await ctx.close();
await browser.close();
