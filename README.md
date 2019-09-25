# Paraphrasing

Repository containing different paraphrasing related tools.

## Requirements
The versions below are the ones that have been used, newer versions should work but not tested.

- Tensorflow 1.12
- Keras 2.2.4 (Keras> 2.2 needed in order to use keras tokenizer and data generator)
- Keras Preprocessing 1.0.6
- CUDA 9.0 (necessary for CuDNNGRU and CuDNNLSTM)
- H5py 2.8
- Gensim 3.8
- Numpy 1.15
- Sci-py 1.1
- Matplotlib 3.0.1
- Sacrebleu 1.3.7

## Sentence embeddings
The script for creating sentence embeddings. It is mainly a reimplementation of [Skip-Thoughts](https://github.com/ryankiros/skip-thoughts) in Keras. It's main goal is to provide a more simpler and updated implementationm in order to train and test more easily new models based on Skip-Thought vectors. For now the only new feature that was introduced is the size of the context window as a parameter. To see the usage of the script execute:
```
sent2vec.py -h
usage: Train sent2vec model [-h] [-g GPU] -c CORPUS --dev DEV [-t TOKENIZER]
                            -m MODEL [-s SIZE] [--cell {gru,lstm}]
                            [-v VOCAB_SIZE] [--embedding-dim EMBEDDING_DIM]
                            [-b BATCH_SIZE] [-e EPOCHS] [--max-len MAX_LEN]
                            [-sp STEPS] [-w WINDOW] [--no-filters]

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     GPU device to be used
  -c CORPUS, --corpus CORPUS
                        Corpus file for the training
  --dev DEV             Development set file
  -t TOKENIZER, --tokenizer TOKENIZER
                        File name to save the tokenizer
  -m MODEL, --model MODEL
                        Model name
  -s SIZE, --size SIZE  Size of encoder and decoder
  --cell {gru,lstm}     Cell type of the recurrent netowrk: GRU or LSTM
  -v VOCAB_SIZE, --vocab-size VOCAB_SIZE
                        Size of the vocabulary
  --embedding-dim EMBEDDING_DIM
                        Emmbedding vector dimensions
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  --max-len MAX_LEN     Max sequence length
  -sp STEPS, --steps STEPS
                        Number of steps (batches) per epoch
  -w WINDOW, --window WINDOW
                        Window of context. Number of sentences to use on
                        backward and forward
  --no-filters
```

## Encoder
The encoder class creates an encoder that receives sentences as text and encodes it to an vector space. It also performs a vocabulary expansion with a pre-trained word-embeddings file provided.

## Sentence embeddings tests
The tests introduced are the Microsoft Reasearch Paraphrase Corpus and the SICK dataset. We use tha sem scripts provided in the Skip-Thoughts repository but with some library updates. It uses the encoder class to create the models in the test.
```
eval.py -h
usage: eval.py [-h] [-d DATA] [-e EMBEDDINGS] [-v V] model tokenizer

positional arguments:
  model                 Model to evaluate
  tokenizer             Tokenizer object

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Path to test data
  -e EMBEDDINGS, --embeddings EMBEDDINGS
                        Embedding file
  -v V                  Verbose level
```

## Paraphrase generation
These approach creates a seq2seq model with its encoder weights initilized with the skip encoder weights.
```
paraphrasing.py -h
usage: Train paraphrase generator model [-h] -c CORPUS --dev DEV --test TEST
                                        -t TOKENIZER --encoder ENCODER
                                        [-b BATCH_SIZE] [-e EMBEDDING]
                                        [--epochs EPOCHS] [-sp STEPS]
                                        [--random]

optional arguments:
  -h, --help            show this help message and exit
  -c CORPUS, --corpus CORPUS
                        Corpus file for the training
  --dev DEV             Corpus file for the validation
  --test TEST           Corpus file for the test
  -t TOKENIZER, --tokenizer TOKENIZER
                        File name to save the tokenizer
  --encoder ENCODER     Encoder h5 model file
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EMBEDDING, --embedding EMBEDDING
                        Embedding file
  --epochs EPOCHS       Number of epochs
  -sp STEPS, --steps STEPS
                        Number of steps (batches) per epoch
  --random
```
For the tests the greedy embedding sentence similarity metric is used in the script from [this](https://github.com/julianser/hed-dlg-truncated) repository.

## Data used in the experiments
The data that was used to train sentence embeddings was a corpus of free available books crawled from smashwords with the [bookcorpus](https://github.com/soskek/bookcorpus) toolkit. For the paraphrase generation, a subsample of the XXXL PPDB lexical and phrasal databases with score higher than 3.8. All de data can be downloaded [here](https://mega.nz/#!lJFAQYgB!GxPiVXJZACtgwt_bZyR2otqadjIon27HXs_Hhk-f4pA) and the sentence embedding test data can be downloaded [here](https://mega.nz/#!dQNE0QYb!LcL2dBSTNaXj26b06XgtB-CgHAVff0r0qdKBgSYf45A)

## Reproducibility
For the reproducibility of the experiments, the seed that has been used in all the scripts is 333. Even so, the complete reproducibility is not guaranteed if the versions are different (specially CUDA and CuDNN versions), also `PYTHONHASHSEED` environtment variable must be set to 333.
