# expression

expression mapper that converts raw sentiment scores to smoothed animation frames.

## state

- current smoothed emotion scores (initialized to neutral)
- smoothing factor (0 = instant, 1 = frozen)
- start timestamp for relative frame timing

## capabilities

- apply exponential moving average to blend new scores with current state
- normalize scores to sum to 1 after each update
- determine dominant emotion from current state
- generate frame objects with timestamp, sentiment, dominant emotion, and deformed landmark points
- reset to neutral state

## interfaces

exposes:
- ExpressionMapper class with update, frame, reset

depends on:
- landmarks (deform)
- training data (EMOTIONS list)

## invariants

- smoothed scores always sum to 1 after normalization
- smoothing=0 produces instant transitions (no blending with previous state)
- initial state is always pure neutral
- frame always contains exactly 30 landmark points
