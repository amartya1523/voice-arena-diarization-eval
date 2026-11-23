import { chromium } from 'playwright';
import { spawn } from 'child_process';
const mode = process.argv[2] || 'stills';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto('file://' + process.cwd() + '/render.html');
await page.evaluate(() => window.ready);
if (mode === 'stills') {
  const times = process.argv[3].split(',').map(Number);
  for (const t of times) {
    await page.evaluate(t => window.seek(t), t);
    await page.screenshot({ path: `frames/still_${t.toFixed(1)}.jpg`, type: 'jpeg', quality: 80 });
  }
} else {
  const fps = 30, dur = await page.evaluate(() => window.DUR);
  const ff = spawn('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', 'frames/video_noaudio.mp4']);
  const n = Math.round(dur * fps);
  for (let i = 0; i < n; i++) {
    await page.evaluate(t => window.seek(t), i / fps);
    const buf = await page.screenshot({ type: 'jpeg', quality: 92 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log('frame', i, '/', n);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
}
await browser.close();
console.log('done');
