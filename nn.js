// nn.js — Minimal feed-forward neural network from scratch
// Matrix math, activations, forward/backprop, SGD

export class Matrix {
  constructor(rows, cols, data) {
    this.rows = rows;
    this.cols = cols;
    this.data = data instanceof Float64Array ? data : new Float64Array(rows * cols);
    if (Array.isArray(data)) {
      for (let i = 0; i < data.length; i++) this.data[i] = data[i];
    }
  }

  static zeros(rows, cols) {
    return new Matrix(rows, cols);
  }

  static fromArray(arr) {
    return new Matrix(arr.length, 1, arr);
  }

  static fromRows(rows) {
    const r = rows.length;
    const c = rows[0].length;
    const m = new Matrix(r, c);
    for (let i = 0; i < r; i++)
      for (let j = 0; j < c; j++)
        m.data[i * c + j] = rows[i][j];
    return m;
  }

  get(i, j) { return this.data[i * this.cols + j]; }
  set(i, j, v) { this.data[i * this.cols + j] = v; }

  add(other) {
    const out = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++)
      out.data[i] = this.data[i] + other.data[i];
    return out;
  }

  sub(other) {
    const out = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++)
      out.data[i] = this.data[i] - other.data[i];
    return out;
  }

  // Matrix multiplication: this (m×n) × other (n×p) → (m×p)
  mul(other) {
    const out = new Matrix(this.rows, other.cols);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < other.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++)
          sum += this.data[i * this.cols + k] * other.data[k * other.cols + j];
        out.data[i * other.cols + j] = sum;
      }
    return out;
  }

  // Element-wise multiplication (Hadamard)
  hadamard(other) {
    const out = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++)
      out.data[i] = this.data[i] * other.data[i];
    return out;
  }

  scale(s) {
    const out = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++)
      out.data[i] = this.data[i] * s;
    return out;
  }

  transpose() {
    const out = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++)
        out.data[j * this.rows + i] = this.data[i * this.cols + j];
    return out;
  }

  map(fn) {
    const out = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++)
      out.data[i] = fn(this.data[i], i);
    return out;
  }

  // Add a column vector (rows×1) to each column — broadcast bias
  addVec(vec) {
    const out = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++)
        out.data[i * this.cols + j] = this.data[i * this.cols + j] + vec.data[i];
    return out;
  }

  toArray() {
    return Array.from(this.data);
  }

  toJSON() {
    return { rows: this.rows, cols: this.cols, data: Array.from(this.data) };
  }

  static fromJSON(json) {
    return new Matrix(json.rows, json.cols, json.data);
  }

  clone() {
    return new Matrix(this.rows, this.cols, new Float64Array(this.data));
  }
}

// Xavier initialization
export function xavierInit(rows, cols) {
  const limit = Math.sqrt(6 / (rows + cols));
  const m = new Matrix(rows, cols);
  for (let i = 0; i < m.data.length; i++)
    m.data[i] = (Math.random() * 2 - 1) * limit;
  return m;
}

// Activation functions
export const activations = {
  relu: {
    fn: x => Math.max(0, x),
    dfn: x => x > 0 ? 1 : 0,
  },
  sigmoid: {
    fn: x => 1 / (1 + Math.exp(-Math.min(Math.max(x, -500), 500))),
    dfn: x => { const s = 1 / (1 + Math.exp(-Math.min(Math.max(x, -500), 500))); return s * (1 - s); },
  },
  tanh: {
    fn: x => Math.tanh(x),
    dfn: x => 1 - Math.tanh(x) ** 2,
  },
};

// Softmax on a column vector (rows×1)
export function softmax(vec) {
  const out = new Matrix(vec.rows, 1);
  let maxVal = -Infinity;
  for (let i = 0; i < vec.rows; i++)
    if (vec.data[i] > maxVal) maxVal = vec.data[i];
  let sum = 0;
  for (let i = 0; i < vec.rows; i++) {
    out.data[i] = Math.exp(vec.data[i] - maxVal);
    sum += out.data[i];
  }
  for (let i = 0; i < vec.rows; i++)
    out.data[i] /= sum;
  return out;
}

// Cross-entropy loss for softmax output vs one-hot target
export function crossEntropyLoss(predicted, target) {
  let loss = 0;
  for (let i = 0; i < target.rows; i++)
    if (target.data[i] > 0)
      loss -= target.data[i] * Math.log(Math.max(predicted.data[i], 1e-15));
  return loss;
}

export class Network {
  constructor(layerSizes, hiddenActivation = 'relu') {
    this.layerSizes = layerSizes;
    this.hiddenActivation = hiddenActivation;
    this.weights = [];
    this.biases = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      this.weights.push(xavierInit(layerSizes[i + 1], layerSizes[i]));
      this.biases.push(Matrix.zeros(layerSizes[i + 1], 1));
    }
  }

  forward(input) {
    const zs = [];   // pre-activation
    const as = [input]; // activations (input is a[0])
    let a = input;
    const act = activations[this.hiddenActivation];

    for (let i = 0; i < this.weights.length; i++) {
      const z = this.weights[i].mul(a).add(this.biases[i]);
      zs.push(z);
      if (i < this.weights.length - 1) {
        // Hidden layer
        a = z.map(act.fn);
      } else {
        // Output layer — softmax
        a = softmax(z);
      }
      as.push(a);
    }
    return { zs, as };
  }

  predict(input) {
    const { as } = this.forward(input);
    return as[as.length - 1];
  }

  // Backpropagation — returns weight and bias gradients
  backward(input, target) {
    const { zs, as } = this.forward(input);
    const act = activations[this.hiddenActivation];
    const L = this.weights.length;
    const dw = [];
    const db = [];

    // Output layer delta: softmax + cross-entropy simplifies to (predicted - target)
    let delta = as[L].sub(target);

    for (let i = L - 1; i >= 0; i--) {
      dw[i] = delta.mul(as[i].transpose());
      db[i] = delta.clone();

      if (i > 0) {
        // Propagate through hidden layer
        const dAct = zs[i - 1].map(act.dfn);
        delta = this.weights[i].transpose().mul(delta).hadamard(dAct);
      }
    }
    return { dw, db, loss: crossEntropyLoss(as[L], target) };
  }

  // Stochastic gradient descent on a batch
  train(samples, { lr = 0.01, epochs = 1, batchSize = 16, shuffle = true, onEpoch } = {}) {
    const history = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let indices = Array.from({ length: samples.length }, (_, i) => i);
      if (shuffle) {
        for (let i = indices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
        }
      }

      let totalLoss = 0;
      let correct = 0;

      for (let b = 0; b < indices.length; b += batchSize) {
        const batch = indices.slice(b, b + batchSize);
        const accDw = this.weights.map(w => Matrix.zeros(w.rows, w.cols));
        const accDb = this.biases.map(b => Matrix.zeros(b.rows, b.cols));

        for (const idx of batch) {
          const { input, target } = samples[idx];
          const { dw, db, loss } = this.backward(input, target);
          totalLoss += loss;

          // Check accuracy
          const pred = this.predict(input);
          let predMax = 0, targMax = 0;
          for (let i = 0; i < pred.rows; i++) {
            if (pred.data[i] > pred.data[predMax]) predMax = i;
            if (target.data[i] > target.data[targMax]) targMax = i;
          }
          if (predMax === targMax) correct++;

          for (let i = 0; i < dw.length; i++) {
            accDw[i] = accDw[i].add(dw[i]);
            accDb[i] = accDb[i].add(db[i]);
          }
        }

        // Apply gradients
        const scale = lr / batch.length;
        for (let i = 0; i < this.weights.length; i++) {
          this.weights[i] = this.weights[i].sub(accDw[i].scale(scale));
          this.biases[i] = this.biases[i].sub(accDb[i].scale(scale));
        }
      }

      const avgLoss = totalLoss / samples.length;
      const accuracy = correct / samples.length;
      history.push({ epoch, loss: avgLoss, accuracy });
      if (onEpoch) onEpoch({ epoch, loss: avgLoss, accuracy });
    }

    return history;
  }

  save() {
    return JSON.stringify({
      layerSizes: this.layerSizes,
      hiddenActivation: this.hiddenActivation,
      weights: this.weights.map(w => w.toJSON()),
      biases: this.biases.map(b => b.toJSON()),
    });
  }

  static load(json) {
    const data = typeof json === 'string' ? JSON.parse(json) : json;
    const net = new Network(data.layerSizes, data.hiddenActivation);
    net.weights = data.weights.map(w => Matrix.fromJSON(w));
    net.biases = data.biases.map(b => Matrix.fromJSON(b));
    return net;
  }
}
