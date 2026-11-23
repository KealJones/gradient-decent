// Headless check of css/: runs window.selfTest() and a short generation. Needs playwright + Chrome.
// Usage (from the repo root): python3 -m http.server 8000 & node tools/check_css.cjs
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ channel: 'chrome' }), p = await b.newPage();
  p.on('pageerror', e => console.log('page error:', e.message));
  await p.goto('http://localhost:8000/css/'); await p.waitForTimeout(6000);
  console.log(await p.textContent('#stat')); console.log(JSON.stringify(await p.textContent('#out')));
  console.log(await p.evaluate(() => selfTest(40)));
  await p.screenshot({ path: '/tmp/css-check.png', fullPage: true }); await b.close();
})();
