// Capturas de la ronda 4: ops (fase, tape, scrubber, bitácora), sol, jurado, póster.
import { chromium } from 'playwright';
import fs from 'node:fs';
const OUT = 'tools/shots/r4';
fs.mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage', '--js-flags=--jitless', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errs = [];
page.on('console', m => { const t = m.type(); if ((t === 'error' || t === 'warning') && !/GL Driver Message|GPU stall/.test(m.text())) errs.push(`${t}: ${m.text().slice(0, 140)}`); });
page.on('pageerror', e => errs.push('pageerror: ' + e.message.slice(0, 140)));
await page.goto('http://localhost:8000', { waitUntil: 'networkidle' });
await page.waitForSelector('[data-card-src]');
if (await page.locator('#cutscene').count()) { await page.keyboard.press('Escape'); await page.waitForSelector('#cutscene', { state: 'detached' }).catch(() => {}); }
await page.waitForSelector('.driver-popover', { timeout: 12000 }).catch(async () => {
  await page.locator('button[title="Reabrir el tour de introducción"]').click().catch(() => {});
  await page.waitForSelector('.driver-popover', { timeout: 8000 }).catch(() => {});
});
await page.keyboard.press('Escape');
await page.waitForTimeout(600);

// hero con fase + tape + scrubber + bitácora
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(800);
await page.screenshot({ path: `${OUT}/01_ops_hero.png` });
await page.evaluate(() => window.scrollTo(0, 620));
await page.waitForTimeout(600);
await page.screenshot({ path: `${OUT}/02_scrubber_perfil.png` });
await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
await page.waitForTimeout(700);
await page.screenshot({ path: `${OUT}/03_bitacora.png` });

// scrubber: play 2s
await page.evaluate(() => window.scrollTo(0, 620));
await page.locator('button[title="Reproducir/pausar el replay de la misión"]').click();
await page.waitForTimeout(2200);
await page.screenshot({ path: `${OUT}/04_replay.png` });
await page.locator('button[title="Reproducir/pausar el replay de la misión"]').click().catch(() => {});

// modo sol
await page.locator('button[title*="Modo sol"]').click();
await page.waitForTimeout(700);
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(500);
await page.screenshot({ path: `${OUT}/05_modo_sol.png` });
await page.locator('button[title*="tema oscuro"]').click();
await page.waitForTimeout(500);

// modo jurado: portada, slide 3 y 4
await page.locator('button[aria-label="Modo jurado"]').click();
await page.waitForTimeout(900);
await page.screenshot({ path: `${OUT}/06_jurado_portada.png` });
await page.keyboard.press('ArrowRight');
await page.keyboard.press('ArrowRight');
await page.keyboard.press('ArrowRight');
await page.waitForTimeout(1200);
await page.screenshot({ path: `${OUT}/07_jurado_criticos.png` });
await page.keyboard.press('Escape');
await page.waitForTimeout(600);

// mapa offline de la trayectoria (MapOffline: tiles locales Web Mercator)
await page.getByText('Trayectoria GPS del descenso').scrollIntoViewIfNeeded();
await page.waitForTimeout(900);
await page.screenshot({ path: `${OUT}/10_mapa_offline.png` });

// póster: descargar y verificar archivo
const [dl] = await Promise.all([
  page.waitForEvent('download', { timeout: 15000 }),
  (async () => {
    await page.getByRole('tab', { name: 'Post-vuelo', exact: true }).click();
    await page.waitForTimeout(1200);
    await page.locator('button', { hasText: 'Póster PNG' }).click();
  })(),
]);
const posterPath = '/tmp/poster_check.png';
await dl.saveAs(posterPath);
console.log('poster descargado:', dl.suggestedFilename(), fs.statSync(posterPath).size, 'bytes');
await page.screenshot({ path: `${OUT}/08_post_toast.png` });

console.log(errs.length ? 'CONSOLE ISSUES:\n' + errs.join('\n') : 'consola limpia ✔');
await browser.close();
