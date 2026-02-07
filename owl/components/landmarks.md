# landmarks

30-point face mesh with per-emotion deformation vectors.

## state

- 30 landmark definitions with name, group, and neutral (x, y) coordinates
- per-emotion deformation tables: dx/dy displacement vectors for each landmark
- 7 landmark groups: eyebrow_left, eyebrow_right, eye_left, eye_right, nose, mouth, jaw

## capabilities

- provide neutral face landmark positions
- deform landmarks by blending emotion-weighted displacement vectors
- return independent copies of landmark positions (no mutation of originals)

## interfaces

exposes:
- landmarks array (30 points with name, group, x, y)
- GROUPS list (7 group names)
- deform(scores) function: emotion weights to deformed point array
- neutralPoints() function: fresh copy of neutral positions

depends on:
- nothing (self-contained)

## invariants

- landmark coordinates are in normalized [0, 1] space
- deformation with pure neutral returns exact neutral positions
- deformation is linear: half-weight produces half-displacement
- all 30 landmarks are always present in deform output
- group distribution: brows 8, eyes 8, nose 3, mouth 8, jaw 3
