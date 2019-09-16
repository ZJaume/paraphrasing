from __future__ import print_function
from encoder import Encoder
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import eval_sick
import eval_msrp
import datagen
import sys

# Set random seeds for reproducibility
import random
random.seed(333)
np.random.seed(333)
tf.set_random_seed(333)

parser = ArgumentParser()
parser.add_argument('model',help='Model to evaluate')
parser.add_argument('tokenizer', help='Tokenizer object')
parser.add_argument('-d','--data',default='data/',help='Path to test data')
parser.add_argument('-e','--embeddings',help='Embedding file')
parser.add_argument('-v',type=int, default=0, help='Verbose level')
args = parser.parse_args(sys.argv[1:])
print(args)

tokenizer = datagen.load_tokenizer(args.tokenizer)
encoder= Encoder(args.model, args.embeddings, tokenizer)
print('Encoder created')

eval_sick.evaluate(encoder, evaltest=True, loc=args.data, verbose=args.v)
eval_msrp.evaluate(encoder, evalcv=True, evaltest=True, use_feats=True, loc=args.data)
