// Capturas de verificación del upgrade UI/UX (no es el smoke test).
import { chromium } from 'playwright';
import fs from 'node:fs';

const BASE = process.env.BASE || 'http://localhost:8000';
const OUT = 'tools/shots/up';
fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage', '--js-flags=--jitless', '--enable-unsafe-swiftshader'] });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
const errs = [];
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push(`${m.type()}: ${m.text().slice(0, 200)}`); });
page.on('pageerror', e => errs.push('pageerror: ' + e.message.slice(0, 200)));

await page.goto(BASE, { waitUntil: 'domcontentloaded' });
// cutscene en pleno boot
await page.waitForTimeout(1400);
await page.screenshot({ path: `${OUT}/00_cutscene.png` });
await page.waitForSelector('[data-card-src]', { timeout: 10000 });
// el tour se arma ~900 ms después de la cutscene
await page.waitForSelector('.driver-popover', { timeout: 12000 }).catch(async () => {
  await page.locator('button[title="Reabrir el tour de introducción"]').click().catch(() => {});
  await page.waitForSelector('.driver-popover', { timeout: 8000 });
});
await page.screenshot({ path: `${OUT}/01_tour.png` });
await page.keyboard.press('Escape');
await page.waitForTimeout(600);
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(600);
await page.screenshot({ path: `${OUT}/02_vuelo_top.png` });
await page.evaluate(() => window.scrollTo(0, 700));
await page.waitForTimeout(600);
await page.screenshot({ path: `${OUT}/03_vuelo_mid.png` });
await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
await page.waitForTimeout(700);
await page.screenshot({ path: `${OUT}/04_vuelo_bottom.png` });

await page.getByRole('tab', { name: 'Post-vuelo', exact: true }).click();
await page.waitForTimeout(1200);
await page.screenshot({ path: `${OUT}/05_post.png` });

await page.getByRole('tab', { name: 'Informe', exact: true }).click();
await page.waitForTimeout(900);
await page.screenshot({ path: `${OUT}/06_informe.png` });

await page.getByRole('tab', { name: 'Vuelo', exact: true }).click();
await page.waitForTimeout(700);
await page.locator('button', { hasText: 'Presentar' }).click();
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/07_present.png` });
await page.keyboard.press('Escape');
await page.waitForTimeout(500);

// replay de la cutscene desde el TopBar
await page.locator('button[title="Reproducir la intro de la misión"]').click();
await page.waitForTimeout(1600);
await page.screenshot({ path: `${OUT}/08_cutscene_replay.png` });
await page.keyboard.press('Escape');
await page.waitForTimeout(900);

console.log(errs.length ? 'CONSOLE ISSUES:\n' + errs.join('\n') : 'consola limpia ✔');
await browser.close();
