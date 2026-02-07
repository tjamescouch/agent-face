# constraints

## stack

- runtime: node.js
- module system: ESM (import/export)
- test framework: node:test (built-in)

## style

- zero external dependencies, node.js stdlib only
- all neural network math implemented from scratch (no ML libraries)
- functions and classes, no frameworks
- present tense in comments and documentation

## dependencies

- none. every capability is implemented in-project.

## portability

- renderers that use terminal features (ansi) target standard terminal emulators
- canvas renderer uses built-in node http server and browser EventSource API
- no platform-specific code; runs on any node.js environment

## data

- training data is embedded in source (no external datasets or downloads)
- model weights are saved as plain JSON
- all coordinates use normalized [0, 1] space
