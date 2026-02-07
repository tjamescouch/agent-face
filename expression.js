// expression.js — ExpressionMapper: sentiment scores → smoothed mocap frame
// Exponential moving average for fluid transitions between emotions

import { EMOTIONS } from './training-data.js';
import { deform } from './landmarks.js';

export class ExpressionMapper {
  constructor({ smoothing = 0.3 } = {}) {
    // smoothing: 0 = instant (no smoothing), 1 = frozen (never changes)
    // 0.3 = responsive but smooth
    this.smoothing = smoothing;
    this.current = {};
    for (const e of EMOTIONS) this.current[e] = e === 'neutral' ? 1 : 0;
    this.startTime = Date.now();
  }

  // Update with new raw sentiment scores, return a frame
  update(rawScores) {
    const alpha = this.smoothing;
    for (const e of EMOTIONS) {
      const raw = rawScores[e] || 0;
      this.current[e] = alpha * this.current[e] + (1 - alpha) * raw;
    }

    // Normalize to sum to 1
    let sum = 0;
    for (const e of EMOTIONS) sum += this.current[e];
    if (sum > 0) {
      for (const e of EMOTIONS) this.current[e] /= sum;
    }

    return this.frame();
  }

  // Generate a frame from current state
  frame() {
    const sentiment = { ...this.current };
    let best = null, bestVal = -1;
    for (const [k, v] of Object.entries(sentiment)) {
      if (v > bestVal) { best = k; bestVal = v; }
    }
    return {
      timestamp: Date.now() - this.startTime,
      sentiment,
      dominant: best,
      points: deform(sentiment),
    };
  }

  // Reset to neutral
  reset() {
    for (const e of EMOTIONS) this.current[e] = e === 'neutral' ? 1 : 0;
  }
}
