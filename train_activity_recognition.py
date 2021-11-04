import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from model.streamnet import StreamNet
from utils.data_processing import get_data, compute_time_statistics

### Argument and global variables
parser = argparse.ArgumentParser('StreamNet Activity Recognition Training')
parser.add_argument('-d', '--data', type=str, help='Dataset name',
                    default='milan')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
MEMORY_DIM = args.memory_dim

node_features, full_data, train_data, val_data, test_data = get_data(DATA)

