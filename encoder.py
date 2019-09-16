from keras.preprocessing.text import Tokenizer
from keras.models import load_model, Model
from keras.layers import Input, Embedding, Lambda
from gensim.models import KeyedVectors
from sklearn.linear_model import LinearRegression
import keras.backend as K
import numpy as np
import datagen
import pprint
import sys

def load_w2v(embedding_file):
    '''
    Load embedding file
    '''
    binary = embedding_file.endswith('.bin')
    return KeyedVectors.load_word2vec_format(embedding_file,binary=binary)

def vocab_expansion(rnn_vectors, embedding_file, tokenizer, batch_size=256):
    '''
    Create an embedding matrix adding pretrained embeddings
    applying a linear transformation
    '''
    print('Performing vocabulary expansion...')
    sys.stdout.flush()
    wv = load_w2v(embedding_file)

    # get shared words between embeddings from rnn and given embeddings
    shared = {}
    count = 0
    for word,i in tokenizer.word_index.items():
        if word in wv and i<tokenizer.num_words:
            shared[word] = count
            count += 1
    print('', count, 'shared words')

    w2v = np.zeros((len(shared), wv.vector_size), dtype='float32')
    rnn_emb = np.zeros((len(shared), rnn_vectors.shape[1]), dtype='float32')
    for word in shared:
        w2v[shared[word]] = wv[word]
        rnn_emb[shared[word]] = rnn_vectors[tokenizer.word_index[word]]

    print(' Training linear regression')
    clf = LinearRegression(n_jobs=-1)
    clf.fit(w2v, rnn_emb)

    # Obtain words that are not shared
    not_shared = set()
    for word in wv.vocab:
        if word not in shared:
            not_shared.add(word)

    print(' Applying regression for unseen', len(not_shared) ,'words')
    tokenizer.num_words += len(not_shared)
    print('  Tokenizer new vocab of size:', tokenizer.num_words)
    lower = tokenizer.lower
    tokenizer.lower = False # ensure not lowering case of new words
    filters = tokenizer.filters # fit without filters
    tokenizer.filters = ''
    tokenizer.fit_on_texts(list(not_shared)) # append the newer words to vocabulary

    # Compute embedding transformations
    embed_matrix = np.zeros((len(not_shared), rnn_vectors.shape[1]), dtype='float32')
    for word in not_shared:
        tok = tokenizer.texts_to_sequences([word])
        if len(tok[0])<1 or len(tok[0])>1:
            print(tok)
            continue
        embed_matrix[tok[0][0]-rnn_vectors.shape[0]] = clf.predict(np.array(wv[word],ndmin=2)).astype('float32')

    # Restore previous params
    tokenizer.filters = filters
    tokenizer.lower = lower

    return np.vstack((rnn_vectors, embed_matrix))

def mean_vectors(embedding_layer, maxlen=40):
    '''
    Create a model that encodes a sentences with the mean of its word embeddings
    '''
    encoder_input = Input((maxlen,), dtype="int32")
    embedding_layer.name = 'embeddings'
    encoder = embedding_layer(encoder_input)
    encoder = Lambda(lambda x: K.mean(x,axis=1), name='mean_embedding')(encoder)
    return Model(encoder_input, encoder)

class Encoder():
    '''
    Class that contains the encoder functionality
    '''

    def __init__(self, model, embedding_file=None, tokenizer=None, drop_weights=False):
        self.path = model
        self.tokenizer = tokenizer
        if 'mean' in model:
            self.create_mean(embedding_file)
        else:
            self.create_skip(embedding_file, tokenizer, drop_weights)

    def create_skip(self, embedding_file, tokenizer, drop_weights=False):
        '''
        Extract an encoder model
        '''
        print('Extract config from model')
        sys.stdout.flush()
        # Extract the config from the saved model
        skip = load_model(self.path)
        skip_config = skip.get_config()
        encoder_config = {'layers':[], 'name': 'skip-encoder'}
        encoder_embeddings = None
        for l in skip_config['layers']:
            if 'encoder' in l['name']:
                encoder_config['layers'].append(l)
            if 'embeddings' == l['name']:
                l['inbound_nodes'] = [l['inbound_nodes'][0]]
                encoder_config['layers'].append(l)
            # get max len of sequence
            if 'encoder_input' in l['name']:
                self.maxlen = l['config']['batch_input_shape'][1]

        encoder_config['input_layers'] = [skip_config['input_layers'][0]]
        encoder_config['output_layers'] = [[encoder_config['layers'][-1]['name'], 0, 0]]

        # Perform vocab expansion if embedding file is present
        if embedding_file is not None and tokenizer is not None:
            # Get embeddings from the network
            encoder_embeddings = skip.get_layer('embeddings').get_weights()[0]
            embed_matrix = vocab_expansion(encoder_embeddings, embedding_file, tokenizer)
            for l in encoder_config['layers']:
                if 'embeddings' == l['name']:
                    l['config']['input_dim'] = embed_matrix.shape[0]
                    break
            sys.stdout.flush()

        # Create the new model encoder and set weights
        print('Creating new model')
        self.encoder = Model.from_config(encoder_config)
        if not drop_weights:
            for l in self.encoder.layers:
                if embedding_file is not None and 'embeddings' == l.name:
                    l.set_weights([embed_matrix])
                    l.trainable = False
                    continue
                if 'input' not in l.name:
                    l.set_weights(skip.get_layer(l.name).get_weights())
        else:
            print('RANDOM WEIGTHS!')
            for l in self.encoder.layers:
                if 'embeddings' == l.name :
                    l.set_weights([np.random.rand(l.input_dim,l.output_dim)])
        self.encoder.summary()
        sys.stdout.flush()

    def create_mean(self, embedding_file):
        '''
        Create mean vectos model with pretrained embeddings
        '''
        wv = load_w2v(embedding_file)
        weights = np.zeros(wv.vector_size)
        wv.add(datagen.OOV,weights)
        wv.add(datagen.BOS,weights)
        wv.add(datagen.EOS,weights)
        self.tokenizer = Tokenizer(num_words=len(wv.vocab), lower=False, oov_token=datagen.OOV)
        # Populate tokenizer vocabulary
        for i in range(len(wv.index2word)):
            self.tokenizer.word_index[wv.index2word[i]] = i+1
            self.tokenizer.index_word[i+1] = wv.index2word[i]
        self.encoder = mean_vectors(wv.wv.get_keras_embedding(False))
        self.encoder.summary()

    def encode(self, X, batch_size=64, maxlen=40):
        '''
        Encode a batch of sentences
        '''
        X = datagen.tokenize(self.tokenizer, X, maxlen)
        return self.encoder.predict(X, batch_size=batch_size)

