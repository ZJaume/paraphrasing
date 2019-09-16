from keras.models import Model
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from datagen import TripletsGenerator
from models import create_skip
import numpy as np
import tensorflow as tf
import keras
import datagen
import utils
import sys

# Set random seeds for reproducibility
import random
random.seed(333)
np.random.seed(333)
tf.set_random_seed(333)

# Default values
EMBEDDING_DIM   = 600
ENCODER_SIZE    = 2400
DECODER_SIZE    = 2400
CELL            = 'gru'
VOCAB_SIZE      = 20000
MAX_LEN         = 40
WINDOW          = 1

EPOCHS      = 100
BATCH_SIZE  = 128
STEPS_EPOCH = 1e3

if __name__ == "__main__":
    parser = ArgumentParser("Train sent2vec model")
    parser.add_argument('-g', '--gpu' ,type=int, default=0, help="GPU device to be used")
    parser.add_argument('-c','--corpus', required=True, help="Corpus file for the training")
    parser.add_argument('--dev', required=True, help="Development set file")
    parser.add_argument('-t','--tokenizer', default='tknzr', required=False, help="File name to save the tokenizer")
    parser.add_argument('-m','--model', required=True, help="Model name")
    parser.add_argument('-s','--size', type=int, default=ENCODER_SIZE, help="Size of encoder and decoder")
    parser.add_argument('--cell', type=str, default=CELL, choices=['gru', 'lstm'], help="Cell type of the recurrent netowrk: GRU or LSTM")
    parser.add_argument('-v','--vocab-size', type=int, default=VOCAB_SIZE, help="Size of the vocabulary")
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM, help="Emmbedding vector dimensions")
    parser.add_argument('-b','--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('-e','--epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--max-len', type=int, default=MAX_LEN, help="Max sequence length")
    parser.add_argument('-sp','--steps', type=int, default=STEPS_EPOCH, help='Number of steps (batches) per epoch')
    parser.add_argument('-w','--window', type=int, default=WINDOW, help='Window of context. Number of sentences to use on backward and forward')
    parser.add_argument('--no-filters', dest='filters', action='store_false')
    args = parser.parse_args(sys.argv[1:])
    print(args.__dict__)

    tripgen = TripletsGenerator(args.corpus,
                            batch_size=args.batch_size,
                            max_len=args.max_len,
                            vocab_size=args.vocab_size,
                            window=args.window,
                            filters=args.filters)
    devgen = TripletsGenerator(args.dev,
                            tokenizer=tripgen.tknzr,
                            batch_size=args.batch_size,
                            window=args.window,
                            max_len=MAX_LEN)
    utils.oov_pct(tripgen.data, tripgen.tknzr.word_index[datagen.OOV])

    if 'bpe' in args.corpus or 'BPE' in args.corpus:
        preprocess = '-bpe'
    else:
        preprocess = ''
    if args.filters:
        preprocess += '-filters'
    else:
        preprocess += '-nofilters'

    suffix = '-c' + utils.num_to_str(len(tripgen.data)) \
                            + '-' + args.cell + str(args.size) \
                            + '-w' + str(args.window) \
                            + '-e' + str(args.embedding_dim) \
                            + '-v' + utils.num_to_str(args.vocab_size) \
                            + preprocess
    modelname = 'models/' + args.model \
                            + suffix \
                            + '.best.h5'
    tokenname = 'models/' + args.tokenizer \
                            + suffix \
                            +'.pkl'
    datagen.save_tokenizer(tripgen.tknzr,tokenname)
    print('Vocab size:',args.vocab_size)
    print('Num samples:',len(tripgen.data))

with tf.device('/gpu:'+str(args.gpu)):
    model = create_skip(embedding_dim=args.embedding_dim,
                        encoder_size=args.size,
                        decoder_size=args.size,
                        cell=args.cell,
                        vocab_size=args.vocab_size,
                        max_len=args.max_len,
                        window=args.window,
                        )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()

    checkpoint = ModelCheckpoint(modelname,
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min',
                                period=1)
    earlystop = EarlyStopping(monitor='val_loss',
                            patience=10,
                            mode='min')
    history = model.fit_generator(generator=tripgen,
                use_multiprocessing=True,
                workers=8,
                shuffle=True,
                epochs=args.epochs,
                steps_per_epoch=args.steps,
                callbacks=[checkpoint,earlystop],
                validation_data=devgen,
                verbose=1,
                )
    utils.plot(history,args.model,modelname)
