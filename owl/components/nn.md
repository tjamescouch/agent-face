# nn

matrix math and feed-forward neural network with backpropagation.

## state

- weight matrices and bias vectors for each layer
- layer topology (array of sizes)
- hidden activation function name

## capabilities

- matrix operations: add, subtract, multiply, hadamard, transpose, scale, map
- column vector broadcasting (addVec)
- xavier weight initialization
- forward pass through arbitrary layer depths
- softmax output normalization
- cross-entropy loss computation
- backpropagation with gradient accumulation
- mini-batch stochastic gradient descent
- training history tracking (loss, accuracy per epoch)
- network serialization and deserialization

## interfaces

exposes:
- Matrix class with arithmetic, serialization, and factory methods
- Network class with predict, train, forward, backward, save, load
- softmax, crossEntropyLoss, activations, xavierInit utilities

depends on:
- nothing (self-contained)

## invariants

- softmax output always sums to 1
- matrix dimensions are validated implicitly through typed array sizes
- xavier initialization bounds values within sqrt(6 / (fan_in + fan_out))
- training history length equals epoch count
