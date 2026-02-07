# agent-face

real-time text sentiment to animated face expression pipeline.

## components

- [nn](components/nn.md) - matrix math and feed-forward neural network engine
- [sentiment](components/sentiment.md) - tokenizer, vocabulary, and sentiment analyzer
- [landmarks](components/landmarks.md) - face mesh definition and emotion-driven deformation
- [expression](components/expression.md) - expression mapper with temporal smoothing
- [renderer](components/renderer.md) - pluggable renderer system
- [cli](components/cli.md) - command-line orchestrator

## behaviors

- [pipeline](behaviors/pipeline.md) - stdin text to rendered face expression flow
- [training](behaviors/training.md) - train, save, and load model lifecycle

## constraints

see [constraints.md](constraints.md)
