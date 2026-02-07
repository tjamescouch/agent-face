# cli

command-line orchestrator that wires pipeline components together.

## state

- parsed command and flags from argv
- path to saved weights file
- built-in demo text lines

## capabilities

- parse CLI arguments into command and options
- train: build vocabulary, train network, save weights
- run: load weights, read stdin line-by-line, render at target fps
- demo: load weights, iterate built-in text with delay
- landmarks: output face mesh schema as JSON
- eval: load weights, evaluate per-emotion accuracy breakdown
- print usage on missing or unknown command

## interfaces

exposes:
- main entry point (direct execution via node or bin link)

depends on:
- sentiment (SentimentAnalyzer)
- expression (ExpressionMapper)
- renderer (RendererManager)
- landmarks (landmarks, GROUPS)

## invariants

- run and demo require weights file to exist
- eval requires weights file to exist
- unknown commands print usage and exit non-zero
