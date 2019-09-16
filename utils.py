from keras.preprocessing.sequence import pad_sequences
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt
import numpy as np
import datagen
import math
'''
Utitlities file
Some auxiliar functions
'''

def num_to_str(num):
    '''
    Convert a number to magnitude
    '''
    str_num = str(num)
    digits = len(str_num)
    if digits > 6:
        return replace_dig(str_num,str_num[-6:],'M')
    elif digits > 3:
        return replace_dig(str_num,str_num[-3:],'K')

def replace_dig(s, z, rep):
    '''
    Replace digits for a letter in reverse order
    '''
    return s[::-1].replace(z[::-1],rep,1)[::-1]

def oov_pct(data, token):
    '''
    Count total, minimum and maximum unkown words in a dataset
    '''
    mean = 0
    minimum = 500
    maximum = 0
    total_unk = 0
    total = 0
    for i in range(len(data)):
        length = 0
        mean_sub = 0
        for j in range(len(data[i])):
            if data[i][j] == token:
                mean_sub += 1
            length += 1
            total += 1
            if data[i][j] == 0:
                break

        if minimum > mean_sub:
            minimum = mean_sub
        if maximum < mean_sub:
            maximum = mean_sub
        mean += mean_sub/length
        total_unk += mean_sub

    print(' Mean unkown words per sentence:', mean/len(data)*100)
    print(' Total pct of unkown words:', total_unk/total*100)
    print(' Max unkown words in a sentence:', maximum)
    print(' Min unkown words in a sentence:', minimum)

def decode_sentences(data, model, index2word, k=20, cond=False, BOS=2):
    '''
    Decode output predictions of the model to text
    '''
    np.seterr(divide = 'ignore')
    if cond:
        hyps = []
        for batch in data:
            for item in batch[0][0]:
                hyps.append(decode_sentences_sub(item,index2word,k,model=model))
    else:
        probs = []
        for d, _ in data:
            probs.append(model.predict_on_batch(d))
        probs = np.concatenate(probs,axis=0)
        pool = Pool(processes=12)
        hyps = pool.starmap(decode_sentences_sub, zip(probs,repeat(index2word),repeat(k)), 1)
    return hyps

def decode_sentences_sub(data, index2word, k, model=None, BOS=2):
    '''
    Decode reduced function to allow multiprocessing
    '''
    if model is None:
        hyp = beam_search_decoder(data,k)[0]
    else:
        hyp = beam_search_decoder_cond(data,model,k,BOS=BOS)[0]

    tokens = ''
    for index in hyp:
        if index == 0:
            break
        if index2word[index] == datagen.BOS or index2word[index] == datagen.EOS:
            continue
        # Decode merging BPE Segments
        if index2word[index].endswith('@@'):
            tokens += index2word[index][:-2]
            segment = True
        else:
            tokens += index2word[index] + ' '
    if tokens.endswith(' '):
        return tokens[:-1]
    else:
        return tokens

def beam_search_decoder(data, k):
    '''
    Beam search over already predicted data
    '''
    sequences = [[]]*k
    scores = np.zeros(k, dtype='float32')

    # walk over each step in sequence
    for step in range(len(data)):
        # Expand each node with the probabilities of all words at this time step
        if step != 0:
            probs = scores[:,None] - np.log(data[step])
            vocab_size = probs.shape[1]
        else:
            probs = -np.log(data[step])
            vocab_size = probs.shape[0]

        # Sort the new paths and get indexes
        probs_flat = probs.flatten()
        rank = np.argsort(probs_flat)[:k]
        rows = rank // vocab_size
        cols = rank % vocab_size

        # Append the new best paths to the list
        new_sequences = []
        for i, [row_idx, col_idx] in list(enumerate(zip(rows,cols))):
            scores[i] = probs_flat[rank[i]]
            new_sequences.append(sequences[row_idx] + [col_idx])
        sequences = new_sequences
    return sequences

def beam_search_decoder_cond(data, model, k, BOS=2):
    '''
    Beam search conditioning each step to the previous selected path
    '''
    sequences = [[BOS]]*k
    scores = np.zeros(k, dtype='float32')
    maxlen = len(data)
    batch = np.array([data]*k)

    # walk over each step in sequence
    for step in range(maxlen):
        # Expand each node with the probabilities of all words at this time step
        if step != 0:
            # Predict this time step conditioned on the previous path
            preds = model.predict_on_batch([batch,pad_sequences(sequences, maxlen=maxlen, padding='post')])[:,step]
            probs = scores[:,None] - np.log(preds)
            vocab_size = probs.shape[1]
        else:
            preds = model.predict_on_batch([data.reshape(1,maxlen),np.array([[BOS]*maxlen])])[0,step]
            probs = -np.log(preds)
            vocab_size = probs.shape[0]

        # Sort the new paths and get indexes
        probs_flat = probs.flatten()
        rank = np.argsort(probs_flat)[:k]
        rows = rank // vocab_size
        cols = rank % vocab_size

        # Append the new best paths to the list
        new_sequences = []
        for i, [row_idx, col_idx] in list(enumerate(zip(rows,cols))):
            scores[i] = probs_flat[rank[i]]
            new_sequences.append(sequences[row_idx] + [col_idx])
        sequences = new_sequences

    return sequences

def plot(history,modelname,filename):
    '''
    Plot the training history
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(modelname+' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Dev'], loc='upper left')
    plt.savefig(filename+'_plot.png')

