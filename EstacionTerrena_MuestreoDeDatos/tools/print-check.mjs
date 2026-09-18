import { chromium } from 'playwright';
const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage', '--js-flags=--jitless'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 1500 } });
await page.goto('http://localhost:8000', { waitUntil: 'networkidle' });
await page.waitForSelector('[data-card-src]');
if (await page.locator('#cutscene').count()) { await page.keyboard.press('Escape'); await page.waitForSelector('#cutscene', { state: 'detached' }).catch(() => {}); }
await page.waitForSelector('.driver-popover', { timeout: 9000 }).catch(() => {});
await page.keyboard.press('Escape');
await page.getByRole('tab', { name: 'Informe', exact: true }).click();
await page.waitForTimeout(900);
await page.emulateMedia({ media: 'print' });
await page.waitForTimeout(300);
await page.screenshot({ path: 'tools/shots/up/09_print.png', fullPage: false });
await browser.close();
console.log('print check listo');
