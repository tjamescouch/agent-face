# pipeline

stdin text to rendered face expression flow.

## flow

1. user starts `face run` with optional renderer and smoothing flags
2. system loads saved model weights from disk
3. system initializes expression mapper with smoothing factor
4. system loads and initializes the selected renderer
5. render loop starts at target fps, repeatedly rendering the current frame
6. stdin lines arrive and are trimmed
7. each non-empty line is pushed through the sliding window analyzer
8. analyzer tokenizes, encodes, classifies, and averages with window history
9. resulting emotion scores update the expression mapper
10. expression mapper applies EMA smoothing and normalizes scores
11. deformed landmark points are computed from smoothed scores
12. next render tick outputs the frame via the active renderer
13. on stdin close, a final frame is rendered and the renderer is closed

## failure modes

- missing weights file: exit with error message directing user to train first
- empty stdin lines: silently skipped
- renderer load failure: propagated as unhandled error
- SIGINT: render loop stopped, renderer closed, clean exit
