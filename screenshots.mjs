import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const BASE = 'http://localhost:5173';
const OUT  = './screenshots';
mkdirSync(OUT, { recursive: true });

const pages = [
  { name: '01-conversations',  path: '/' },
  { name: '02-session-trace',  path: '/sessions/dbf93e06-69eb-4cee-bfd4-bd5ea2c3c7c8' },
  { name: '03-human-review',   path: '/human-review' },
  { name: '04-system-health',  path: '/system-health' },
  { name: '05-audit-logs',     path: '/audit' },
  { name: '06-analytics',      path: '/analytics' },
  { name: '07-gateway',        path: '/gateway' },
  { name: '08-evaluation',     path: '/evaluation' },
  { name: '09-feedback',       path: '/feedback' },
  { name: '10-new-query',      path: '/new' },
  { name: '11-settings',       path: '/settings' },
];

const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

for (const p of pages) {
  await page.goto(BASE + p.path, { waitUntil: 'networkidle', timeout: 15000 }).catch(() => {
    return page.goto(BASE + p.path, { waitUntil: 'domcontentloaded', timeout: 10000 });
  });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT}/${p.name}.png`, fullPage: true });
  console.log(`✓ ${p.name}`);
}

await browser.close();
console.log('Done — screenshots saved to ./screenshots/');
