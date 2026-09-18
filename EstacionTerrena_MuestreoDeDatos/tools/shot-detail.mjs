import { chromium } from 'playwright';
const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage', '--js-flags=--jitless', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
await page.goto('http://localhost:8000', { waitUntil: 'networkidle' });
await page.waitForSelector('[data-card-src]');
if (await page.locator('#cutscene').count()) { await page.keyboard.press('Escape'); await page.waitForSelector('#cutscene', { state: 'detached' }).catch(() => {}); }
await page.waitForSelector('.driver-popover', { timeout: 9000 }).catch(() => {});
await page.keyboard.press('Escape');
await page.waitForTimeout(500);
await page.click('[data-card-src="cap_0006"]');
await page.waitForTimeout(900);
await page.locator('#tour-detalle').scrollIntoViewIfNeeded();
await page.waitForTimeout(600);
await page.screenshot({ path: 'tools/shots/up2/05_detalle.png' });
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(700);
await page.screenshot({ path: 'tools/shots/up2/08_hero.png' });
await browser.close();
console.log('listo');
