// Verificación focalizada de la ronda 2 de fixes.
import { chromium } from 'playwright';
import fs from 'node:fs';
const BASE = process.env.BASE || 'http://localhost:8000';
const OUT = 'tools/shots/up2';
fs.mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage', '--js-flags=--jitless', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errs = [];
page.on('console', m => { const t = m.type(); if ((t === 'error' || t === 'warning') && !/GL Driver Message|GPU stall/.test(m.text())) errs.push(`${t}: ${m.text().slice(0, 160)}`); });
page.on('pageerror', e => errs.push('pageerror: ' + e.message.slice(0, 160)));

await page.goto(BASE, { waitUntil: 'domcontentloaded' });
await page.waitForSelector('[data-card-src]', { timeout: 10000 });
await page.waitForSelector('.driver-popover', { timeout: 12000 }).catch(async () => {
  await page.locator('button[title="Reabrir el tour de introducción"]').click().catch(() => {});
  await page.waitForSelector('.driver-popover', { timeout: 8000 });
});
await page.waitForTimeout(600);
await page.screenshot({ path: `${OUT}/01_tour_con_texto.png` });
const popText = await page.locator('.driver-popover').innerText();
console.log('popover dice:', JSON.stringify(popText.slice(0, 120)));
await page.keyboard.press('Escape');
await page.waitForTimeout(700);

// modelo 3D nuevo
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(900);
await page.screenshot({ path: `${OUT}/02_modelo.png` });

// ▶ Descenso: la página NO debe irse del panel
const y0 = await page.evaluate(() => window.scrollY);
await page.locator('button', { hasText: 'Descenso' }).click();
await page.waitForTimeout(2500);
const y1 = await page.evaluate(() => window.scrollY);
console.log(`scroll antes=${y0} después=${y1} (debe quedar cerca del panel 3D)`);
await page.screenshot({ path: `${OUT}/03_flythrough.png` });
await page.locator('button', { hasText: 'Pausar' }).click().catch(() => {});

// live mode
await page.locator('button', { hasText: 'Live' }).click();
await page.waitForTimeout(800);
await page.screenshot({ path: `${OUT}/04_live.png` });
await page.locator('button', { hasText: 'En vivo' }).click().catch(() => {});

// detalle sin scroll anidado
await page.click('[data-card-src="cap_0006"]');
await page.waitForTimeout(600);
await page.locator('#tour-detalle').scrollIntoViewIfNeeded();
await page.waitForTimeout(500);
await page.screenshot({ path: `${OUT}/05_detalle.png` });

// presentación nítida + PiP + toggle anotada
await page.evaluate(() => window.scrollTo(0, 0));
await page.locator('button', { hasText: 'Presentar' }).click();
await page.waitForTimeout(1600);
await page.screenshot({ path: `${OUT}/06_present_nitida.png` });
await page.locator('button', { hasText: 'Ver anotada' }).click();
await page.waitForTimeout(900);
await page.screenshot({ path: `${OUT}/07_present_anotada.png` });
await page.keyboard.press('Escape');
await page.waitForTimeout(600);

console.log(errs.length ? 'CONSOLE ISSUES:\n' + errs.join('\n') : 'consola limpia ✔');
await browser.close();
