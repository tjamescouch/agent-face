// sentiment.js — Tokenizer, Vocabulary, SentimentAnalyzer
// Bag-of-words + feed-forward NN for 6-class emotion classification

import { readFileSync, writeFileSync } from 'node:fs';
import { Matrix, Network } from './nn.js';
import { EMOTIONS, trainingData } from './training-data.js';

// Simple whitespace + punctuation tokenizer, lowercase, no stemming
export function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0);
}

export class Vocabulary {
  constructor(maxSize = 500) {
    this.maxSize = maxSize;
    this.word2idx = new Map();
    this.idx2word = [];
  }

  build(texts) {
    const freq = new Map();
    for (const text of texts) {
      for (const token of tokenize(text)) {
        freq.set(token, (freq.get(token) || 0) + 1);
      }
    }
    // Sort by frequency descending, take top N
    const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1]);
    this.idx2word = sorted.slice(0, this.maxSize).map(([w]) => w);
    this.word2idx = new Map(this.idx2word.map((w, i) => [w, i]));
    return this;
  }

  get size() { return this.idx2word.length; }

  // Bag-of-words encoding → column vector
  encode(text) {
    const tokens = tokenize(text);
    const vec = Matrix.zeros(this.size, 1);
    for (const t of tokens) {
      const idx = this.word2idx.get(t);
      if (idx !== undefined) vec.data[idx] = 1;
    }
    return vec;
  }

  toJSON() {
    return { maxSize: this.maxSize, words: this.idx2word };
  }

  static fromJSON(json) {
    const v = new Vocabulary(json.maxSize);
    v.idx2word = json.words;
    v.word2idx = new Map(json.words.map((w, i) => [w, i]));
    return v;
  }
}

export class SentimentAnalyzer {
  constructor({ vocabSize = 500, hiddenSize = 32, windowSize = 5 } = {}) {
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.windowSize = windowSize;
    this.vocab = null;
    this.network = null;
    this.window = [];
    this.emotions = EMOTIONS;
  }

  // Build vocabulary and prepare training samples
  prepare() {
    this.vocab = new Vocabulary(this.vocabSize);
    this.vocab.build(trainingData.map(d => d.text));

    const samples = trainingData.map(d => {
      const input = this.vocab.encode(d.text);
      const target = Matrix.zeros(this.emotions.length, 1);
      target.data[this.emotions.indexOf(d.emotion)] = 1;
      return { input, target };
    });

    this.network = new Network(
      [this.vocab.size, this.hiddenSize, this.emotions.length],
      'relu'
    );

    return samples;
  }

  train({ epochs = 100, lr = 0.05, batchSize = 16, onEpoch } = {}) {
    const samples = this.prepare();
    return this.network.train(samples, { epochs, lr, batchSize, onEpoch });
  }

  // Analyze a single text → emotion scores object
  analyze(text) {
    if (!this.network || !this.vocab) throw new Error('Model not trained or loaded');
    const input = this.vocab.encode(text);
    const output = this.network.predict(input);
    const scores = {};
    for (let i = 0; i < this.emotions.length; i++) {
      scores[this.emotions[i]] = output.data[i];
    }
    return scores;
  }

  // Push a line into the sliding window, return blended scores
  push(text) {
    this.window.push(text);
    if (this.window.length > this.windowSize) this.window.shift();

    // Analyze each line in window and average
    const combined = {};
    for (const e of this.emotions) combined[e] = 0;

    for (const line of this.window) {
      const scores = this.analyze(line);
      for (const e of this.emotions) combined[e] += scores[e];
    }
    for (const e of this.emotions) combined[e] /= this.window.length;
    return combined;
  }

  // Get dominant emotion from scores
  dominant(scores) {
    let best = null, bestVal = -1;
    for (const [k, v] of Object.entries(scores)) {
      if (v > bestVal) { best = k; bestVal = v; }
    }
    return best;
  }

  save(path) {
    const data = {
      vocabSize: this.vocabSize,
      hiddenSize: this.hiddenSize,
      windowSize: this.windowSize,
      vocab: this.vocab.toJSON(),
      network: JSON.parse(this.network.save()),
    };
    writeFileSync(path, JSON.stringify(data));
  }

  static load(path) {
    const data = JSON.parse(readFileSync(path, 'utf-8'));
    const sa = new SentimentAnalyzer({
      vocabSize: data.vocabSize,
      hiddenSize: data.hiddenSize,
      windowSize: data.windowSize,
    });
    sa.vocab = Vocabulary.fromJSON(data.vocab);
    sa.network = Network.load(data.network);
    return sa;
  }

  // Evaluate accuracy on training data (or provided test set)
  evaluate(testData) {
    const data = testData || trainingData;
    let correct = 0;
    for (const d of data) {
      const scores = this.analyze(d.text);
      if (this.dominant(scores) === d.emotion) correct++;
    }
    return { accuracy: correct / data.length, correct, total: data.length };
  }
}
