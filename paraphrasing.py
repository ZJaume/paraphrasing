from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from argparse import ArgumentParser
from datagen import SentencesGenerator
from models import create_pp_generator
from encoder import Encoder
import keras.backend as K
import tensorflow as tf
import embedding_metrics
import datagen
import utils
import sacrebleu
import random
import sys

# Set random seeds for reproducibility
random.seed(333)
from numpy.random import seed
seed(333)
tf.set_random_seed(333)

# Default values
MAX_LEN=40
EPOCHS=150
BATCH_SIZE=64
STEPS_EPOCH=1e3

if __name__ == "__main__":
    parser = ArgumentParser("Train paraphrase generator model")
    parser.add_argument('-c','--corpus', required=True, help="Corpus file for the training")
    parser.add_argument('--dev', required=True, help="Corpus file for the validation")
    parser.add_argument('--test', required=True, help="Corpus file for the test")
    parser.add_argument('-t','--tokenizer', required=True, help="File name to save the tokenizer")
    parser.add_argument('--encoder', required=True, help="Encoder h5 model file")
    parser.add_argument('-b','--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('-e','--embedding', default=None, help="Embedding file")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('-sp','--steps', type=int, default=STEPS_EPOCH, help='Number of steps (batches) per epoch')
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    print(args.__dict__)

    # Create model with reandom weights if specified (baseline)
    if args.random:
        encodername = 'seq2seq'
    else:
        encodername = args.encoder.split('/')[-1].split('.')[0]
    modelname = 'models/ppgen-' + encodername + '.best.h5'

with tf.device('/gpu:0'):
    if 'mean' in args.encoder:
        encoder = Encoder(args.encoder, embedding_file=args.embedding)
    else:
        tokenizer = datagen.load_tokenizer(args.tokenizer)
        encoder = Encoder(args.encoder, embedding_file=args.embedding, tokenizer=tokenizer, drop_weights=args.random)
    tokenizer = encoder.tokenizer
    model = create_pp_generator(encoder,
                            max_len=MAX_LEN)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()

    sentgen = SentencesGenerator(args.corpus,
                            tokenizer,
                            batch_size=args.batch_size,
                            keep_original=True,
                            max_len=MAX_LEN)
    devgen = SentencesGenerator(args.dev,
                            tokenizer,
                            batch_size=args.batch_size,
                            max_len=MAX_LEN)
    testgen = SentencesGenerator(args.test,
                            tokenizer,
                            batch_size=2,
                            max_len=MAX_LEN,
                            keep_original=True)
    print(' Vocabulary size: ', tokenizer.num_words)

    print('Source:')
    utils.oov_pct(sentgen.source, sentgen.tknzr.word_index[datagen.OOV])
    print('Target:')
    utils.oov_pct(sentgen.target, sentgen.tknzr.word_index[datagen.OOV])

    checkpoint = ModelCheckpoint(modelname,
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min',
                                period=1)
    earlystop = EarlyStopping(monitor='val_loss',
                            patience=10,
                            mode='min')
    print('Train first phase')
    for layer in model.layers:
        if 'embeddings' in layer.name or 'encoder' in layer.name:
            layer.trainable = False
    model.fit_generator(generator=sentgen,
                use_multiprocessing=True,
                workers=8,
                shuffle=True,
                epochs=1,
                steps_per_epoch=args.steps,
                #callbacks=[checkpoint,earlystop],
                validation_data=devgen,
                verbose=1,
                )
    print('Train second phase')
    for layer in model.layers:
        if 'embeddings' in layer.name or 'encoder' in layer.name:
            layer.trainable = True
    model.fit_generator(generator=sentgen,
                use_multiprocessing=True,
                workers=8,
                shuffle=True,
                epochs=args.epochs,
                steps_per_epoch=args.steps,
                callbacks=[checkpoint,earlystop],
                validation_data=devgen,
                verbose=1,
                )

    model = load_model(modelname)
    # Compute test metrics
    print('Computing test')
    print('Decoding')
    import time
    start = time.time()
    hyps = utils.decode_sentences(testgen,model,tokenizer.index_word, k=1, cond=True, BOS=tokenizer.word_index[datagen.BOS])
    print('Decoding time:' + str(time.time()-start))
    print('Hypothesis set', len(hyps))
    for i in range(10):
        print('Source:',testgen.data[0][i])
        print('Hypothesis:',hyps[i])
        print('Target:',testgen.data[1][i])
        print('#############################')

    bleu = sacrebleu.raw_corpus_bleu(hyps, [testgen.data[1][:len(hyps)]])
    r = embedding_metrics.greedy_match(hyps,testgen.data[1][:len(hyps)], 'data/gnews-embeddings300.bin')
    greedy = "Greedy Matching Score: %f +/- %f ( %f )" %(r[0], r[1], r[2])
    print(bleu)
    print(greedy)

    with open(modelname[:-3] + '.score', 'w') as f:
        f.write(str(bleu))
        f.write('\n')
        f.write(greedy)
        f.write('\n')
