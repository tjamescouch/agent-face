// landmarks.js — 30 face landmarks, neutral positions, per-emotion deformation vectors
// Coordinates in normalized 0-1 space (origin top-left)

export const GROUPS = ['eyebrow_left', 'eyebrow_right', 'eye_left', 'eye_right', 'nose', 'mouth', 'jaw'];

// 30 landmarks: brows 8, eyes 8, nose 3, mouth 8, jaw 3
export const landmarks = [
  // Eyebrow left (4)
  { name: 'brow_L_inner',  group: 'eyebrow_left',  x: 0.35, y: 0.28 },
  { name: 'brow_L_mid',    group: 'eyebrow_left',  x: 0.30, y: 0.25 },
  { name: 'brow_L_peak',   group: 'eyebrow_left',  x: 0.25, y: 0.24 },
  { name: 'brow_L_outer',  group: 'eyebrow_left',  x: 0.20, y: 0.26 },

  // Eyebrow right (4)
  { name: 'brow_R_inner',  group: 'eyebrow_right', x: 0.65, y: 0.28 },
  { name: 'brow_R_mid',    group: 'eyebrow_right', x: 0.70, y: 0.25 },
  { name: 'brow_R_peak',   group: 'eyebrow_right', x: 0.75, y: 0.24 },
  { name: 'brow_R_outer',  group: 'eyebrow_right', x: 0.80, y: 0.26 },

  // Eye left (4)
  { name: 'eye_L_inner',   group: 'eye_left',      x: 0.34, y: 0.35 },
  { name: 'eye_L_top',     group: 'eye_left',      x: 0.29, y: 0.33 },
  { name: 'eye_L_outer',   group: 'eye_left',      x: 0.22, y: 0.35 },
  { name: 'eye_L_bottom',  group: 'eye_left',      x: 0.29, y: 0.37 },

  // Eye right (4)
  { name: 'eye_R_inner',   group: 'eye_right',     x: 0.66, y: 0.35 },
  { name: 'eye_R_top',     group: 'eye_right',     x: 0.71, y: 0.33 },
  { name: 'eye_R_outer',   group: 'eye_right',     x: 0.78, y: 0.35 },
  { name: 'eye_R_bottom',  group: 'eye_right',     x: 0.71, y: 0.37 },

  // Nose (3)
  { name: 'nose_bridge',   group: 'nose',          x: 0.50, y: 0.40 },
  { name: 'nose_tip',      group: 'nose',          x: 0.50, y: 0.48 },
  { name: 'nose_base',     group: 'nose',          x: 0.50, y: 0.52 },

  // Mouth (8)
  { name: 'mouth_L',       group: 'mouth',         x: 0.38, y: 0.65 },
  { name: 'mouth_top_L',   group: 'mouth',         x: 0.44, y: 0.63 },
  { name: 'mouth_top',     group: 'mouth',         x: 0.50, y: 0.62 },
  { name: 'mouth_top_R',   group: 'mouth',         x: 0.56, y: 0.63 },
  { name: 'mouth_R',       group: 'mouth',         x: 0.62, y: 0.65 },
  { name: 'mouth_bot_R',   group: 'mouth',         x: 0.56, y: 0.67 },
  { name: 'mouth_bot',     group: 'mouth',         x: 0.50, y: 0.68 },
  { name: 'mouth_bot_L',   group: 'mouth',         x: 0.44, y: 0.67 },

  // Jaw (3)
  { name: 'jaw_L',         group: 'jaw',           x: 0.18, y: 0.60 },
  { name: 'jaw_tip',       group: 'jaw',           x: 0.50, y: 0.82 },
  { name: 'jaw_R',         group: 'jaw',           x: 0.82, y: 0.60 },
];

// Per-emotion deformation vectors (dx, dy per landmark)
// Positive y = down, positive x = right
const deformations = {
  joy: [
    // Brows slightly raised
    { dx: 0, dy: -0.01 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: -0.01 },
    { dx: 0, dy: -0.01 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: -0.01 },
    // Eyes slightly squinted (happy eyes)
    { dx: 0, dy: 0 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0 }, { dx: 0, dy: -0.01 },
    { dx: 0, dy: 0 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0 }, { dx: 0, dy: -0.01 },
    // Nose unchanged
    { dx: 0, dy: 0 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0 },
    // Mouth wide smile — corners up and out
    { dx: -0.03, dy: -0.03 }, { dx: -0.01, dy: -0.02 }, { dx: 0, dy: -0.02 }, { dx: 0.01, dy: -0.02 },
    { dx: 0.03, dy: -0.03 }, { dx: 0.01, dy: 0 }, { dx: 0, dy: 0.01 }, { dx: -0.01, dy: 0 },
    // Jaw slightly open
    { dx: 0, dy: 0 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0 },
  ],
  anger: [
    // Brows furrowed — inner down, outer down
    { dx: 0.02, dy: 0.03 }, { dx: 0.01, dy: 0.02 }, { dx: 0, dy: 0.01 }, { dx: -0.01, dy: 0.02 },
    { dx: -0.02, dy: 0.03 }, { dx: -0.01, dy: 0.02 }, { dx: 0, dy: 0.01 }, { dx: 0.01, dy: 0.02 },
    // Eyes narrowed
    { dx: 0, dy: 0 }, { dx: 0, dy: 0.02 }, { dx: 0, dy: 0 }, { dx: 0, dy: -0.01 },
    { dx: 0, dy: 0 }, { dx: 0, dy: 0.02 }, { dx: 0, dy: 0 }, { dx: 0, dy: -0.01 },
    // Nose flared
    { dx: 0, dy: 0 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0.01 },
    // Mouth tense — corners down, pressed
    { dx: -0.01, dy: 0.02 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0.01 },
    { dx: 0.01, dy: 0.02 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: -0.01 },
    // Jaw clenched
    { dx: 0, dy: 0 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: 0 },
  ],
  sadness: [
    // Brows inner up (worry), outer down
    { dx: 0, dy: -0.03 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0.02 },
    { dx: 0, dy: -0.03 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0.02 },
    // Eyes droopy — top eyelid lower
    { dx: 0, dy: 0.01 }, { dx: 0, dy: 0.02 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0 },
    { dx: 0, dy: 0.01 }, { dx: 0, dy: 0.02 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0 },
    // Nose unchanged
    { dx: 0, dy: 0 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0 },
    // Mouth frown — corners down
    { dx: -0.01, dy: 0.03 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0.01 }, { dx: 0, dy: 0.01 },
    { dx: 0.01, dy: 0.03 }, { dx: 0, dy: 0 }, { dx: 0, dy: -0.01 }, { dx: 0, dy: 0 },
    // Jaw slack
    { dx: 0, dy: 0.01 }, { dx: 0, dy: 0.02 }, { dx: 0, dy: 0.01 },
  ],
  surprise: [
    // Brows raised high
    { dx: 0, dy: -0.04 }, { dx: 0, dy: -0.05 }, { dx: 0, dy: -0.05 }, { dx: 0, dy: -0.04 },
    { dx: 0, dy: -0.04 }, { dx: 0, dy: -0.05 }, { dx: 0, dy: -0.05 }, { dx: 0, dy: -0.04 },
    // Eyes wide open
    { dx: 0, dy: 0 }, { dx: 0, dy: -0.03 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0.02 },
    { dx: 0, dy: 0 }, { dx: 0, dy: -0.03 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0.02 },
    // Nose unchanged
    { dx: 0, dy: 0 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0 },
    // Mouth open oval
    { dx: -0.01, dy: 0 }, { dx: -0.01, dy: -0.02 }, { dx: 0, dy: -0.03 }, { dx: 0.01, dy: -0.02 },
    { dx: 0.01, dy: 0 }, { dx: 0.01, dy: 0.03 }, { dx: 0, dy: 0.04 }, { dx: -0.01, dy: 0.03 },
    // Jaw drops
    { dx: 0, dy: 0 }, { dx: 0, dy: 0.04 }, { dx: 0, dy: 0 },
  ],
  fear: [
    // Brows raised and tense — inner up
    { dx: 0.01, dy: -0.04 }, { dx: 0, dy: -0.04 }, { dx: 0, dy: -0.03 }, { dx: -0.01, dy: -0.02 },
    { dx: -0.01, dy: -0.04 }, { dx: 0, dy: -0.04 }, { dx: 0, dy: -0.03 }, { dx: 0.01, dy: -0.02 },
    // Eyes wide, tense
    { dx: 0, dy: 0 }, { dx: 0, dy: -0.03 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0.01 },
    { dx: 0, dy: 0 }, { dx: 0, dy: -0.03 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0.01 },
    // Nose slightly tense
    { dx: 0, dy: 0 }, { dx: 0, dy: 0 }, { dx: 0, dy: 0.01 },
    // Mouth tense open — horizontal stretch
    { dx: -0.02, dy: 0 }, { dx: -0.01, dy: -0.01 }, { dx: 0, dy: -0.01 }, { dx: 0.01, dy: -0.01 },
    { dx: 0.02, dy: 0 }, { dx: 0.01, dy: 0.02 }, { dx: 0, dy: 0.02 }, { dx: -0.01, dy: 0.02 },
    // Jaw slightly open
    { dx: 0, dy: 0 }, { dx: 0, dy: 0.02 }, { dx: 0, dy: 0 },
  ],
  neutral: landmarks.map(() => ({ dx: 0, dy: 0 })),
};

// Apply emotion deformations weighted by scores
// scores: { joy: 0.7, anger: 0.1, ... } — should sum to ~1
export function deform(scores) {
  return landmarks.map((lm, i) => {
    let dx = 0, dy = 0;
    for (const [emotion, weight] of Object.entries(scores)) {
      if (deformations[emotion]) {
        dx += deformations[emotion][i].dx * weight;
        dy += deformations[emotion][i].dy * weight;
      }
    }
    return {
      name: lm.name,
      group: lm.group,
      x: lm.x + dx,
      y: lm.y + dy,
    };
  });
}

// Get neutral positions
export function neutralPoints() {
  return landmarks.map(lm => ({ ...lm }));
}
