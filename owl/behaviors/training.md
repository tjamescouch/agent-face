# training

train, save, and load model lifecycle.

## flow

1. user runs `face train` with optional epoch count and learning rate
2. system builds vocabulary from all training texts (top N words by frequency)
3. system encodes each training example as bag-of-words input with one-hot emotion target
4. system initializes neural network with xavier weights
5. system trains via mini-batch SGD for specified epochs
6. epoch progress is logged (loss and accuracy)
7. final accuracy is evaluated on the training set
8. vocabulary and network weights are serialized to a JSON file
9. on subsequent `face run` or `face eval`, the saved file is loaded
10. loaded model reconstructs vocabulary mapping and network weights identically

## failure modes

- training data missing or malformed: error at import time
- disk write failure on save: propagated as unhandled error
- corrupted weights file on load: JSON parse error
- model not trained before analyze call: throws descriptive error
