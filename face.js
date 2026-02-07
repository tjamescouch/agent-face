#!/usr/bin/env node
// face.js — CLI entry + pipeline orchestrator
// Commands: train, run, demo, landmarks, eval

import { createInterface } from 'node:readline';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SentimentAnalyzer } from './sentiment.js';
import { ExpressionMapper } from './expression.js';
import { RendererManager } from './renderer.js';
import { landmarks, GROUPS } from './landmarks.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEIGHTS_PATH = join(__dirname, 'weights.json');

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith('--')) {
        args[key] = next;
        i++;
      } else {
        args[key] = true;
      }
    } else {
      args._.push(argv[i]);
    }
  }
  return args;
}

const DEMO_LINES = [
  "Hello, this is a normal day at the office.",
  "The weather report says it will be partly cloudy.",
  "I just got the most amazing news!",
  "I am so happy and excited right now!",
  "This is the best day of my entire life!",
  "Wait, what just happened? I can't believe it!",
  "That was completely unexpected and shocking!",
  "I'm starting to feel worried about what comes next.",
  "Something doesn't feel right, I'm getting scared.",
  "I'm terrified, this is really frightening.",
  "This makes me absolutely furious!",
  "How could they do something so terrible and unfair!",
  "I am so angry I could scream right now.",
  "Now I just feel sad and empty inside.",
  "Everything feels hopeless and I miss how things were.",
  "I'm overwhelmed with sorrow and grief.",
  "But wait, maybe things will get better.",
  "Actually, I think everything is going to be okay.",
  "I'm feeling much better and more hopeful now!",
  "Life is beautiful and I am grateful for everything.",
];

async function cmdTrain(args) {
  const epochs = parseInt(args.epochs || '150', 10);
  const lr = parseFloat(args.lr || '0.05');
  console.log(`Training: epochs=${epochs}, lr=${lr}`);

  const sa = new SentimentAnalyzer();
  sa.train({
    epochs,
    lr,
    onEpoch({ epoch, loss, accuracy }) {
      if ((epoch + 1) % 10 === 0 || epoch === 0)
        console.log(`  epoch ${(epoch + 1).toString().padStart(4)} | loss: ${loss.toFixed(4)} | accuracy: ${(accuracy * 100).toFixed(1)}%`);
    },
  });

  const { accuracy, correct, total } = sa.evaluate();
  console.log(`\nFinal accuracy: ${(accuracy * 100).toFixed(1)}% (${correct}/${total})`);

  sa.save(WEIGHTS_PATH);
  console.log(`Weights saved to ${WEIGHTS_PATH}`);
}

async function cmdRun(args) {
  if (!existsSync(WEIGHTS_PATH)) {
    console.error('No weights found. Run `face train` first.');
    process.exit(1);
  }

  const rendererName = args.renderer || 'json';
  const fps = parseInt(args.fps || '10', 10);
  const smoothing = parseFloat(args.smoothing || '0.3');

  const sa = SentimentAnalyzer.load(WEIGHTS_PATH);
  const em = new ExpressionMapper({ smoothing });
  const rm = new RendererManager();
  await rm.use(rendererName);

  let latestFrame = em.frame();

  // Render loop
  const interval = setInterval(async () => {
    await rm.render(latestFrame);
  }, 1000 / fps);

  // Read stdin line by line
  const rl = createInterface({ input: process.stdin, terminal: false });

  rl.on('line', (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const scores = sa.push(trimmed);
    latestFrame = em.update(scores);
  });

  rl.on('close', async () => {
    // Final render
    await rm.render(latestFrame);
    clearInterval(interval);
    await rm.close();
  });

  // Handle Ctrl-C
  process.on('SIGINT', async () => {
    clearInterval(interval);
    await rm.close();
    process.exit(0);
  });
}

async function cmdDemo(args) {
  if (!existsSync(WEIGHTS_PATH)) {
    console.error('No weights found. Run `face train` first.');
    process.exit(1);
  }

  const rendererName = args.renderer || 'ansi';
  const delay = parseInt(args.delay || '1500', 10);

  const sa = SentimentAnalyzer.load(WEIGHTS_PATH);
  const em = new ExpressionMapper({ smoothing: 0.25 });
  const rm = new RendererManager();
  await rm.use(rendererName);

  for (const line of DEMO_LINES) {
    const scores = sa.push(line);
    const frame = em.update(scores);
    frame.text = line;
    await rm.render(frame);
    await new Promise(r => setTimeout(r, delay));
  }

  await rm.close();
}

async function cmdLandmarks() {
  console.log(JSON.stringify({ landmarks, groups: GROUPS }, null, 2));
}

async function cmdEval(args) {
  if (!existsSync(WEIGHTS_PATH)) {
    console.error('No weights found. Run `face train` first.');
    process.exit(1);
  }

  const sa = SentimentAnalyzer.load(WEIGHTS_PATH);
  const { accuracy, correct, total } = sa.evaluate();
  console.log(`Accuracy: ${(accuracy * 100).toFixed(1)}% (${correct}/${total})`);

  // Per-emotion breakdown
  const { trainingData } = await import('./training-data.js');
  const confusion = {};
  for (const e of sa.emotions) confusion[e] = { correct: 0, total: 0 };

  for (const d of trainingData) {
    const scores = sa.analyze(d.text);
    const predicted = sa.dominant(scores);
    confusion[d.emotion].total++;
    if (predicted === d.emotion) confusion[d.emotion].correct++;
  }

  console.log('\nPer-emotion:');
  for (const [e, c] of Object.entries(confusion)) {
    const pct = c.total > 0 ? (c.correct / c.total * 100).toFixed(1) : '0.0';
    console.log(`  ${e.padEnd(10)} ${pct}% (${c.correct}/${c.total})`);
  }
}

const USAGE = `Usage: face <command> [options]

Commands:
  train [--epochs N] [--lr F]          Train sentiment model, save weights
  run [--renderer ansi|json|canvas]    Stream stdin → face
  demo [--renderer ansi]               Built-in demo text
  landmarks                            Print landmark schema as JSON
  eval                                 Model accuracy on test split
`;

async function main() {
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);
  const cmd = args._[0];

  switch (cmd) {
    case 'train':     return cmdTrain(args);
    case 'run':       return cmdRun(args);
    case 'demo':      return cmdDemo(args);
    case 'landmarks': return cmdLandmarks(args);
    case 'eval':      return cmdEval(args);
    default:
      console.log(USAGE);
      if (cmd) {
        console.error(`Unknown command: ${cmd}`);
        process.exit(1);
      }
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
