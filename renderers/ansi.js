// renderers/ansi.js — Terminal face with Unicode art + emotion bars
// Draws a face on a character grid, with landmark-based feature placement

const WIDTH = 40;
const HEIGHT = 24;
const BAR_WIDTH = 20;

const EMOTION_COLORS = {
  joy:      '\x1b[33m', // yellow
  anger:    '\x1b[31m', // red
  sadness:  '\x1b[34m', // blue
  surprise: '\x1b[35m', // magenta
  fear:     '\x1b[32m', // green
  neutral:  '\x1b[37m', // white
};
const RESET = '\x1b[0m';
const DIM = '\x1b[2m';
const BOLD = '\x1b[1m';

const EMOTION_CHARS = {
  joy:      ':D',
  anger:    '>:(',
  sadness:  ':(',
  surprise: ':O',
  fear:     'D:',
  neutral:  ':|',
};

function createGrid() {
  return Array.from({ length: HEIGHT }, () => Array(WIDTH).fill(' '));
}

function plot(grid, nx, ny, ch) {
  const x = Math.round(nx * (WIDTH - 1));
  const y = Math.round(ny * (HEIGHT - 1));
  if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
    grid[y][x] = ch;
  }
}

function drawLine(grid, x1, y1, x2, y2, ch) {
  const steps = Math.max(Math.abs(Math.round(x2 * WIDTH) - Math.round(x1 * WIDTH)),
                         Math.abs(Math.round(y2 * HEIGHT) - Math.round(y1 * HEIGHT)), 1);
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    plot(grid, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, ch);
  }
}

function drawFace(grid, points) {
  const pt = {};
  for (const p of points) pt[p.name] = p;

  // Face outline (oval)
  for (let a = 0; a < Math.PI * 2; a += 0.08) {
    const x = 0.50 + Math.cos(a) * 0.30;
    const y = 0.50 + Math.sin(a) * 0.38;
    plot(grid, x, y, DIM + '.' + RESET);
  }

  // Eyebrows
  const browPairs = [
    ['brow_L_inner', 'brow_L_mid', 'brow_L_peak', 'brow_L_outer'],
    ['brow_R_inner', 'brow_R_mid', 'brow_R_peak', 'brow_R_outer'],
  ];
  for (const brow of browPairs) {
    for (let i = 0; i < brow.length - 1; i++) {
      drawLine(grid, pt[brow[i]].x, pt[brow[i]].y, pt[brow[i+1]].x, pt[brow[i+1]].y, '~');
    }
  }

  // Eyes
  const eyePairs = [
    ['eye_L_inner', 'eye_L_top', 'eye_L_outer', 'eye_L_bottom'],
    ['eye_R_inner', 'eye_R_top', 'eye_R_outer', 'eye_R_bottom'],
  ];
  for (const eye of eyePairs) {
    for (let i = 0; i < eye.length; i++) {
      const next = eye[(i + 1) % eye.length];
      drawLine(grid, pt[eye[i]].x, pt[eye[i]].y, pt[next].x, pt[next].y, 'o');
    }
    // Pupil at center
    const cx = (pt[eye[0]].x + pt[eye[2]].x) / 2;
    const cy = (pt[eye[1]].y + pt[eye[3]].y) / 2;
    plot(grid, cx, cy, '*');
  }

  // Nose
  plot(grid, pt.nose_bridge.x, pt.nose_bridge.y, '|');
  plot(grid, pt.nose_tip.x, pt.nose_tip.y, 'v');

  // Mouth
  const mouthOrder = ['mouth_L', 'mouth_top_L', 'mouth_top', 'mouth_top_R', 'mouth_R',
                       'mouth_bot_R', 'mouth_bot', 'mouth_bot_L'];
  for (let i = 0; i < mouthOrder.length; i++) {
    const next = mouthOrder[(i + 1) % mouthOrder.length];
    drawLine(grid, pt[mouthOrder[i]].x, pt[mouthOrder[i]].y, pt[next].x, pt[next].y,
      i < 4 ? '_' : '-');
  }
}

function emotionBar(name, value, dominant) {
  const filled = Math.round(value * BAR_WIDTH);
  const color = EMOTION_COLORS[name] || '';
  const bar = '\u2588'.repeat(filled) + DIM + '\u2591'.repeat(BAR_WIDTH - filled) + RESET;
  const label = name.padEnd(9);
  const pct = (value * 100).toFixed(0).padStart(3) + '%';
  const prefix = name === dominant ? BOLD + color + '> ' : '  ';
  return `${prefix}${color}${label}${RESET} ${bar} ${pct}`;
}

export default {
  init(config) {
    this.stream = config.stream || process.stdout;
    this.clear = config.clear !== false;
  },

  render(frame) {
    const grid = createGrid();
    drawFace(grid, frame.points);

    const lines = [];
    if (this.clear) lines.push('\x1b[2J\x1b[H'); // clear screen + home

    const color = EMOTION_COLORS[frame.dominant] || '';
    lines.push(`${BOLD}${color}  Emotion: ${frame.dominant.toUpperCase()} ${EMOTION_CHARS[frame.dominant] || ''}${RESET}`);
    lines.push('');

    // Draw grid
    const gridLines = grid.map(row => '  ' + row.join(''));

    // Overlay emotion bars on right side of grid
    const emotions = Object.keys(frame.sentiment);
    const barStart = Math.floor((HEIGHT - emotions.length * 2) / 2);

    for (let i = 0; i < HEIGHT; i++) {
      const barIdx = Math.floor((i - barStart) / 2);
      if (barIdx >= 0 && barIdx < emotions.length && (i - barStart) % 2 === 0) {
        lines.push(gridLines[i] + '  ' + emotionBar(emotions[barIdx], frame.sentiment[emotions[barIdx]], frame.dominant));
      } else {
        lines.push(gridLines[i]);
      }
    }

    lines.push('');
    lines.push(`${DIM}  t=${frame.timestamp}ms${RESET}`);

    this.stream.write(lines.join('\n') + '\n');
  },

  close() {},
};
