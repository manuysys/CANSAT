import { chromium } from 'playwright';
const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage', '--js-flags=--jitless', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
await page.goto('http://localhost:8005', { waitUntil: 'networkidle' });
await page.waitForSelector('[data-card-src]', { timeout: 15000 });
if (await page.locator('#cutscene').count()) { await page.keyboard.press('Escape'); await page.waitForSelector('#cutscene', { state: 'detached' }).catch(() => {}); }
await page.waitForSelector('.driver-popover', { timeout: 9000 }).catch(() => {});
await page.keyboard.press('Escape');
await page.waitForTimeout(800);
await page.screenshot({ path: 'tools/shots/up2/09_simulacro_live.png' });
const badge = await page.locator('header').innerText();
console.log('topbar incluye badge datos:', /datos: (ok|\d+ aviso)/.test(badge));
await browser.close();
