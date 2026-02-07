// renderers/canvas.js — HTTP server + SSE → browser Canvas drawing

import { createServer } from 'node:http';

const HTML = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Face Expression</title>
<style>
  body { margin: 0; background: #111; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: monospace; color: #eee; }
  #wrap { text-align: center; }
  canvas { border: 1px solid #333; border-radius: 8px; }
  #emotion { font-size: 24px; margin: 16px 0; }
  #bars { text-align: left; display: inline-block; font-size: 14px; }
  .bar-row { margin: 4px 0; }
  .bar-label { display: inline-block; width: 80px; }
  .bar-bg { display: inline-block; width: 200px; height: 16px; background: #333; border-radius: 3px; vertical-align: middle; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.15s; }
</style></head><body>
<div id="wrap">
  <div id="emotion">Connecting...</div>
  <canvas id="c" width="400" height="480"></canvas>
  <div id="bars"></div>
</div>
<script>
const COLORS = { joy: '#f0c040', anger: '#e04040', sadness: '#4080d0', surprise: '#c060d0', fear: '#40b060', neutral: '#aaa' };
const GROUPS = {
  eyebrow_left: ['brow_L_inner','brow_L_mid','brow_L_peak','brow_L_outer'],
  eyebrow_right: ['brow_R_inner','brow_R_mid','brow_R_peak','brow_R_outer'],
  eye_left: ['eye_L_inner','eye_L_top','eye_L_outer','eye_L_bottom'],
  eye_right: ['eye_R_inner','eye_R_top','eye_R_outer','eye_R_bottom'],
  mouth: ['mouth_L','mouth_top_L','mouth_top','mouth_top_R','mouth_R','mouth_bot_R','mouth_bot','mouth_bot_L'],
};

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;

function draw(frame) {
  ctx.clearRect(0, 0, W, H);
  const color = COLORS[frame.dominant] || '#aaa';
  const pts = {};
  for (const p of frame.points) pts[p.name] = p;

  // Face oval
  ctx.beginPath();
  ctx.ellipse(W*0.5, H*0.5, W*0.32, H*0.40, 0, 0, Math.PI*2);
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw connected groups
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  for (const [group, names] of Object.entries(GROUPS)) {
    ctx.beginPath();
    const points = names.map(n => pts[n]).filter(Boolean);
    if (!points.length) continue;
    ctx.moveTo(points[0].x * W, points[0].y * H);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x * W, points[i].y * H);
    if (group.startsWith('eye') || group === 'mouth') ctx.closePath();
    ctx.stroke();
  }

  // Pupils
  for (const eye of ['eye_left', 'eye_right']) {
    const names = GROUPS[eye];
    const cx = (pts[names[0]].x + pts[names[2]].x) / 2 * W;
    const cy = (pts[names[1]].y + pts[names[3]].y) / 2 * H;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI*2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  // Nose
  if (pts.nose_bridge && pts.nose_tip) {
    ctx.beginPath();
    ctx.moveTo(pts.nose_bridge.x * W, pts.nose_bridge.y * H);
    ctx.lineTo(pts.nose_tip.x * W, pts.nose_tip.y * H);
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // All points as dots
  for (const p of frame.points) {
    ctx.beginPath();
    ctx.arc(p.x * W, p.y * H, 2, 0, Math.PI*2);
    ctx.fillStyle = '#666';
    ctx.fill();
  }

  // Emotion label
  document.getElementById('emotion').innerHTML =
    '<span style="color:' + color + '">' + frame.dominant.toUpperCase() + '</span>';

  // Bars
  const barsEl = document.getElementById('bars');
  barsEl.innerHTML = Object.entries(frame.sentiment).map(([e, v]) =>
    '<div class="bar-row"><span class="bar-label">' + e + '</span>' +
    '<span class="bar-bg"><span class="bar-fill" style="width:' + (v*100) + '%;background:' + (COLORS[e]||'#aaa') + '"></span></span>' +
    ' ' + (v*100).toFixed(0) + '%</div>'
  ).join('');
}

const es = new EventSource('/events');
es.onmessage = e => { try { draw(JSON.parse(e.data)); } catch {} };
es.onerror = () => document.getElementById('emotion').textContent = 'Disconnected';
</script></body></html>`;

let server = null;
let clients = [];

export default {
  async init(config) {
    this.port = config.port || 3000;

    await new Promise((resolve, reject) => {
      server = createServer((req, res) => {
        if (req.url === '/events') {
          res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
          });
          res.write(':\n\n');
          clients.push(res);
          req.on('close', () => {
            clients = clients.filter(c => c !== res);
          });
        } else {
          res.writeHead(200, { 'Content-Type': 'text/html' });
          res.end(HTML);
        }
      });
      server.listen(this.port, () => {
        console.error(`Canvas renderer: http://localhost:${this.port}`);
        resolve();
      });
      server.on('error', reject);
    });
  },

  render(frame) {
    const data = `data: ${JSON.stringify(frame)}\n\n`;
    for (const client of clients) {
      try { client.write(data); } catch {}
    }
  },

  async close() {
    for (const client of clients) {
      try { client.end(); } catch {}
    }
    clients = [];
    if (server) {
      await new Promise(resolve => server.close(resolve));
      server = null;
    }
  },
};
