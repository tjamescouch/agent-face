# sentiment

tokenizer, vocabulary builder, and 6-class emotion classifier.

## state

- vocabulary: word-to-index mapping, index-to-word list, max size cap
- neural network: trained weight matrices for classification
- sliding window: recent text lines for temporal context blending

## capabilities

- tokenize text: lowercase, strip punctuation, split on whitespace
- build vocabulary from corpus sorted by word frequency
- encode text as bag-of-words column vector
- train classifier on labeled emotion data
- analyze single text line to produce emotion probability scores
- sliding window averaging across recent inputs
- determine dominant emotion from score distribution
- save and load trained model (vocabulary + network weights)
- evaluate accuracy on a dataset

## interfaces

exposes:
- tokenize function
- Vocabulary class with build, encode, serialize/deserialize
- SentimentAnalyzer class with train, analyze, push, dominant, save, load, evaluate

depends on:
- nn (Matrix, Network)
- training data (EMOTIONS list, labeled examples)

## invariants

- analyze throws if model is not trained or loaded
- emotion scores from analyze always sum to 1 (softmax output)
- vocabulary size never exceeds maxSize
- sliding window never exceeds windowSize entries
