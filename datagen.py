from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import keras
import pickle

OOV = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

def tokenize(tknzr, text, maxlen=40, pad=True):
    '''
    Tokenize a list of sentences
    '''
    tok = tknzr.texts_to_sequences(text)
    if pad:
        tok = pad_sequences(tok, maxlen=maxlen, padding='post', truncating='post')
    return tok

def load_tokenizer(path):
    '''
    Create a tokenizer object from pickle file
    '''
    if not path.endswith('.pkl'):
        raise Exception('File extension must be pkl')
    f = open(path, 'rb')
    tmp = pickle.load(f)
    f.close()
    tknzr = Tokenizer()
    tknzr.__dict__.update(tmp)

    return tknzr

def save_tokenizer(tknzr, path):
    '''
    Save the tokenizer object to a pickle file
    '''
    f = open(path, 'wb')
    pickle.dump(tknzr.__dict__, f)
    f.close()



class SentencesGenerator(keras.utils.Sequence):
    '''
    Generates batches of sentences
    '''
    def __init__(self, path, tknzr, batch_size=64,
                max_len=40, shuffle=False, keep_original=False):
        '''
        Instantiate parameters and load dataset
        '''
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.maxlen = max_len
        self.tknzr = tknzr
        self.tknzr.filters = ''
        self.tknzr.lower = False
        self.keep_original = keep_original
        if keep_original:
            self.data, self.source, self.source_t, self.target = SentencesGenerator.load_data(path, max_len, tknzr, keep_original)
        else:
            self.source, self.source_t, self.target = SentencesGenerator.load_data(path, max_len, tknzr)
        self.index = list(range(0,len(self.source),batch_size))

    def __len__(self):
        '''
        Lenght of epochs
        '''
        return int(np.floor(len(self.source) / self.batch_size))

    def __getitem__(self, index):
        '''
        Create a batch of source sentence and target sentence
        '''
        if len(self)-1 == index: # Avoid out of range
            idx = None
        else:
            idx = self.index[index+1]
        X = self.source[self.index[index]:idx]
        X_y = self.source_t[self.index[index]:idx]
        Y = []
        for y in self.target[self.index[index]:idx]:
            Y.append(np.reshape(y,y.shape+(1,)))

        return [X, X_y], np.array(Y)

    def load_data(path, maxlen, tknzr, keep=False):
        '''
        Read corpus file, tokenize words and encode to sequences
        keep=True to keep the data without tokenizing
        '''
        # Read file and append end anf begin of sentence tags
        print(' Reading data file')
        f = open(path,'r')
        X = []
        X_y = []
        Y = []
        if keep:
            data = ([],[])
        for line in f:
            xy = line[:-1].split('|||')
            X.append(BOS+' '+xy[0]+' ' + EOS)
            X_y.append(BOS+' '+xy[1])
            Y.append(xy[1]+' ' + EOS)
            if keep:
                data[0].append(xy[0])
                data[1].append(xy[1])
        f.close()

        print(' Word2idx len:',len(tknzr.word_index))

        # Create one_hot vectors
        print(' Creating one-hot vectors')
        X = tokenize(tknzr, X, maxlen)
        X_y = tokenize(tknzr, X_y, maxlen)
        Y = tokenize(tknzr, Y, maxlen)

        if keep:
            return data, X, X_y, Y
        else:
            return X, X_y, Y

class TripletsGenerator(keras.utils.Sequence):
    '''
    Generates triplets of source backward and forward sentences
    '''

    def __init__(self, path, vocab_size=20000, batch_size=64,
                max_len=30, shuffle=False, window=1, filters=True, tokenizer=None):
        '''
        Instantiate parameters and load dataset
        '''
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.maxlen = max_len
        self.vocab_size = vocab_size
        self.window = window
        self.data, self.tknzr = TripletsGenerator.load_data(path, vocab_size, max_len, filters, tknzr=tokenizer)
        # Create the indexes for the tirplets of data
        self.source_index = list(range(window,self.data.shape[0]-window,batch_size))
        self.backward_index = [None]*window
        self.forward_index = [None]*window
        for i in range(window):
            self.backward_index[i] = list(range(0,self.data.shape[0]-i+window,batch_size))
            self.forward_index[i] = list(range(i+window,self.data.shape[0],batch_size))

    def __len__(self):
        '''
        Lenght of epochs
        '''
        return int(np.floor((len(self.data)-self.window) / self.batch_size))

    def __getitem__(self,index):
        '''
        Create a batch of source, forward and backward sentences as input
        and forward and backward sentences as output
        '''
        source_idx = self.source_index[index+1]
        backward_idx, forward_idx = [], []
        for i in range(self.window):
            backward_idx.append(self.backward_index[i][index+1])
            forward_idx.append(self.forward_index[i][index+1])

        # Grab batches
        batch_source = self.data[self.source_index[index]:source_idx]
        batch_backward = [None]*self.window
        batch_forward = [None]*self.window
        for i in range(self.window):
            batch_backward[i] = self.data[self.backward_index[i][index]:backward_idx[i]]
            batch_forward[i] = self.data[self.forward_index[i][index]:forward_idx[i]]

        X = [batch_source]
        for y in batch_backward + batch_forward: # Make offset for the input of decoders
            X.append(np.where(y == self.tknzr.word_index[EOS], 0, y))
        Y = []
        for y in batch_backward + batch_forward: # Remove offset for the output
            shifted = pad_sequences(y[:,1:], maxlen=self.maxlen, padding='post', truncating='post')
            Y.append(np.reshape(shifted,shifted.shape+(1,)))

        return X,Y

    def on_epoch_end(self):
        pass

    def load_data(path, vocab_size, maxlen, filters, ids=False, tknzr=None):
        '''
        Read corpus file, tokenize words and encode to sequences
        '''
        # Read file and append end anf begin of sentence tags
        print(' Reading data file')
        f = open(path,'r')
        text = []
        if ids: # Open ids file
            # Change file extension to .ids
            name = ''.join(path.split('.')[:-1]) + 'ids'
            idfile = open(name)
            idname = ''

        for line in f:
            text.append(BOS +' '+line[:-1]+' ' +EOS)
            if ids: # Add context separator
                read = idfile.readline()
                if read != idname:
                    idname = read
                    text.append('<EOC>')
        f.close()

        # Create vocabulary
        if tknzr is None:
            print(' Generating vocabulary')
            tknzr = Tokenizer(num_words=vocab_size, lower=False, oov_token=OOV)
            if not filters:
                tknzr.filters = ''
            else:
                tknzr.filters = tknzr.filters.replace('<','') #need keep tags
                tknzr.filters = tknzr.filters.replace('>','')
            tknzr.fit_on_texts(text)
            print(' Word2idx len:',len(tknzr.word_index))

        # Create one_hot vectors
        print(' Creating one-hot vectors')
        data = tokenize(tknzr, text, maxlen)

        return data, tknzr
