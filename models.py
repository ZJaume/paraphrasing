from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, TimeDistributed
from keras.layers import CuDNNGRU, CuDNNLSTM
import keras.backend as K
import keras

def decoder(decoder_input, encoder_state, vocab, name, decoder_size=1024, cell='gru', vocab_size=20000, embedding_dim=300, max_len=60):
    '''
    Create a decoder feeded by the las state of a decoder
    '''
    #decoder_input = Embedding(vocab_size, embedding_dim, input_length=max_len, name=name+'_embeddings')(decoder_input)
    if cell == 'gru':
        decoder_outputs, _ = CuDNNGRU(decoder_size, return_sequences=True,return_state=True, name='decoder_'+name+'_'+cell)(decoder_input, initial_state=encoder_state)
    elif cell == 'lstm':
        decoder_outputs, _, _ = CuDNNLSTM(decoder_size, return_sequences=True,return_state=True, name='decoder_'+name+'_'+cell)(decoder_input, initial_state=encoder_state)
    decoder_outputs = vocab(decoder_outputs)
    #decoder_outputs = TimeDistributed(Dense(vocab_size, activation='softmax', name='decoder'+name+'output'))(decoder_outputs)
    return decoder_outputs

def encoder(embedding, encoder_size=1024, cell='gru', max_len=20):
    '''
    Create an encoder and return the last state
    '''
    encoder_input = Input((max_len,), name='encoder_input', dtype='int32')
    encoder_embedding = embedding(encoder_input)
    if cell == 'gru':
        _, encoder_state = CuDNNGRU(encoder_size, return_state=True, name='encoder_'+cell)(encoder_embedding)
        return encoder_input, encoder_state
    elif cell == 'lstm':
        _, encoder_h, encoder_c = CuDNNLSTM(encoder_size, return_state=True, name='encoder_'+cell)(encoder_embedding)
        return encoder_input, [encoder_h, encoder_c]

def create_skip(embedding_dim=300, encoder_size=600, decoder_size=600, cell='gru', vocab_size=20000, max_len=20, window=1):
    '''
    Create Skip-Thoughts model with encoder feeding two decoders
    '''
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_len, name='embeddings')
    encoder_input, encoder_state = encoder(embedding,
                                        encoder_size=encoder_size,
                                        cell=cell,
                                        max_len=max_len)

    decoder_vocab = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decorder_vocab')

    decoderb_inputs = [None]*window
    decoderb_outputs = [None]*window
    for i in range(window):
        decoderb_inputs[i] = Input((max_len,), dtype='int32', name='decoderb_input'+str(i))
        decoderb_outputs[i] = decoder(embedding(decoderb_inputs[i]), encoder_state, decoder_vocab,'backward'+str(i),
                                    decoder_size=decoder_size,
                                    cell=cell,
                                    vocab_size=vocab_size,
                                    embedding_dim=embedding_dim,
                                    max_len=max_len)

    decoderf_inputs = [None]*window
    decoderf_outputs = [None]*window
    for i in range(window):
        decoderf_inputs[i] = Input((max_len,), dtype='int32', name='decoderf_input'+str(i))
        decoderf_outputs[i] = decoder(embedding(decoderf_inputs[i]), encoder_state, decoder_vocab, 'forward'+str(i),
                                    decoder_size=decoder_size,
                                    cell=cell,
                                    vocab_size=vocab_size,
                                    embedding_dim=embedding_dim,
                                    max_len=max_len)

    return Model(inputs=[encoder_input]+decoderb_inputs+decoderf_inputs,outputs=decoderb_outputs+decoderf_outputs)

def create_pp_generator(encoder, max_len=40):
    decoder_input = Input((max_len,))
    encoder_embedding = encoder.encoder.get_layer('embeddings')
    decoder_embedding = Embedding(encoder_embedding.input_dim, encoder_embedding.output_dim, input_length=max_len, name='decoder_embedding')(decoder_input)
    decoder = CuDNNGRU(int(encoder.encoder.output.shape[1]), return_sequences=True, name='decoder_gru')(decoder_embedding,initial_state=encoder.encoder.output)
    decoder = TimeDistributed(Dense(encoder_embedding.input_dim, activation='softmax'), name='decoder_output')(decoder)

    return Model(inputs=[encoder.encoder.input, decoder_input],outputs=[decoder])

