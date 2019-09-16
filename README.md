# Paraphrasing

Repository containing different paraphrasing related tools.

## Requirements
- 

## Sentence embeddings
The script for creating sentence embeddings. It is mainly a reimplementation of [Skip-Thoughts](https://github.com/ryankiros/skip-thoughts) in Keras. It's main goal is to provide a more simpler and updated implementationm in order to train and test more easily new models based on Skip-Thought vectors. For now the only new feature that was introduced is the size of the context window as a parameter. To see the usage of the script execute:
```
sent2vec.py -h
```

## Encoder
The encoder class creates an encoder that receives sentences as text and encodes it to an vector space. It also performs a vocabulary expansion with a pre-trained word-embeddings file provided.

## Sentence embeddings tests
The tests introduced are the Microsoft Reasearch Paraphrase Corpus and the SICK dataset. We use tha sem scripts provided in the Skip-Thoughts repository but with some library updates. It uses the encoder class to create the models in the test.
```
eval.py -h
```

## Paraphrase generation
These approach creates a seq2seq model with its encoder weights initilized with the skip encoder weights.
```
paraphrasing.py -h
```
For the tests the greedy embedding sentence similarity metric is used in the script from [this](https://github.com/julianser/hed-dlg-truncated) repository.
