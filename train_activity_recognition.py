import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn

from model.streamnet import StreamNet
from utils.data_processing import get_data, compute_time_statistics

import subprocess
subprocess.Popen('caffeinate')

### Argument and global variables
parser = argparse.ArgumentParser('StreamNet Activity Recognition Training')
parser.add_argument('-d', '--data', type=str, help='Dataset name',
                    default='milan_graph_binary_normalized')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--embedding_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=2, help='Dimensions of the time embedding')
parser.add_argument('--memory_dim', type=int, default=16, help='Dimensions of the time embedding')
parser.add_argument('--house_state_dim', type=int, default=32, help='Dimensions of the time embedding')
parser.add_argument('--event_dim', type=int, default=3, help='Dimensions of summary of sensor event')
parser.add_argument('--num_classes', type=int, default=2, help='Number of activity types')
parser.add_argument('--backprop_every', type=int, default=10, help='Number of batches to process before each backprop')
parser.add_argument('--batch_duration', type=int, default=25, help='Duration of activity summarized by each batch')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

DATA = args.data
BATCH_SIZE = args.bs
PREFIX = args.prefix
NUM_HEADS = args.n_head
NUM_EPOCH = args.n_epoch
LEARNING_RATE = args.lr
DROP_OUT = args.drop_out
GPU = args.gpu
EMBEDDING_DIM = args.embedding_dim
TIME_DIM = args.time_dim
MEMORY_DIM = args.memory_dim
HOUSE_STATE_DIM = args.house_state_dim
EVENT_DIM = args.event_dim
NUM_CLASSES = args.num_classes

node_features, full_data, train_data, val_data, test_data = get_data(DATA)


# Compute time statistics
mean_time_shift_events, std_time_shift_events = \
  compute_time_statistics(full_data.sensors, full_data.timestamps)

# Prepare to store results
results_path = "results/{}.pkl".format(args.prefix)
Path("results/").mkdir(parents=True, exist_ok=True)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)


# Initialize Model
streamnet = StreamNet(memory_dim = MEMORY_DIM, embedding_dim = EMBEDDING_DIM, \
    house_state_dim = HOUSE_STATE_DIM, event_dim = EVENT_DIM, time_dim = TIME_DIM, \
    num_classes = NUM_CLASSES, num_heads = NUM_HEADS, dropout= DROP_OUT, \
    mean_time_shift_events = mean_time_shift_events, \
    std_time_shift_events = std_time_shift_events, 
    device = device, n_sensors = full_data.n_unique_sensors, house_time_dim=6)


optimizer = torch.optim.Adam(streamnet.parameters(), lr=LEARNING_RATE)

#print(streamnet.state_dict().keys())

streamnet = streamnet.to(device)
num_instance = len(train_data.sensors)
num_batch = math.ceil(num_instance / BATCH_SIZE)

print('Num of training instances: {}'.format(num_instance))
print('Num of batches per epoch: {}'.format(num_batch))

# Start training process
torch.autograd.set_detect_anomaly(True)


losses = []
soft = nn.Softmax(dim = 0)

dont_train = False
#streamnet.load_state_dict(torch.load("saved_models/streamnet_latest_act_detection"))

for epoch in range(args.n_epoch):

  train_f1s = []
  train_losses = []
  val_f1s = []
  val_losses = []

  tic = time.perf_counter()
  streamnet.train()
  streamnet.init_memory()
  loss = 0
  loss_accum = 0
  last_batch_time = 0
  most_recent_event = None
  current_batch = 0
  for event in range(len(train_data.event_idxs)):
    if dont_train:
      continue

    # Queue events for each batch
    if train_data.timestamps[event] - last_batch_time < args.batch_duration:
      sensor = train_data.sensors[event]
      streamnet.queue_event(sensor, event)
      most_recent_event = event
      continue
    else:
      # run forward pass on batch of recent events
      current_batch += 1
      most_recent_event = event
      last_batch_time = train_data.timestamps[most_recent_event]
      pred = streamnet.process_batch(train_data, node_features, most_recent_event)

      
      target = torch.tensor([train_data.labels[most_recent_event]]).float() # for focal
      #target = torch.tensor([train_data.labels[most_recent_event]]).long() # for cross entropy

      loss += streamnet.loss_fn(pred[np.newaxis,:], target)
      # for every backprop_every events, do backprop and then reset gradients and loss
      if current_batch % args.backprop_every == 0:
        loss_accum += loss
        loss /= args.backprop_every
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if current_batch % 10 == 0:
          loss_accum /= 10
          losses.append(loss_accum)
          loss_accum = 0

        loss = 0
        streamnet.detach_memory()

      streamnet.clear_recent_events()
      sensor = train_data.sensors[event]
      streamnet.queue_event(sensor, event)
      last_batch_time = train_data.timestamps[most_recent_event]
  # print(len(train_data.event_idxs))
  # print(current_batch)
  toc = time.perf_counter()

  if not dont_train:
    print("Epoch {}: Duration {}, Averaged Loss: {}".format(epoch, toc - tic, sum(losses)/len(losses)))
  
  losses = []

  #Compute training and validation accuracy
  print("start eval")
  last_batch_time = 0
  most_recent_event = None
  streamnet.clear_recent_events()
  streamnet.init_memory()
  streamnet.eval()


  # training eval
  preds = np.zeros(1)
  used_events = []
  
  for event in range(len(train_data.event_idxs)):

    # Queue events for each batch
    if train_data.timestamps[event] - last_batch_time < args.batch_duration:
      sensor = train_data.sensors[event]
      streamnet.queue_event(sensor, event)
      most_recent_event = event
      continue
    else:
      
      most_recent_event = event
      used_events.append(most_recent_event)
      pred = streamnet.process_batch(train_data, node_features, most_recent_event)

      # just for logging
      target = torch.tensor([train_data.labels[most_recent_event]]).float() 
      losses.append(streamnet.loss_fn(pred[np.newaxis,:], target))

      pred = np.array([torch.argmax(soft(pred.detach()))])
      preds = np.concatenate((preds, pred))

      streamnet.clear_recent_events()
      sensor = train_data.sensors[event]
      streamnet.queue_event(sensor, event)
      last_batch_time = train_data.timestamps[most_recent_event]

  print(np.unique(preds))
  targets = np.array(train_data.labels[used_events]) 
  preds = preds[1:]
  train_f1 = f1_score(targets, preds, average=None)

  # log
  train_f1s.append(train_f1)
  train_losses.append(sum(losses)/len(losses))


  print("Classes seen in Training: ")
  print(np.unique(targets))
  print("Train F1")
  print(train_f1)

  last_batch_time = 0
  most_recent_event = None

  # validation eval
  streamnet.clear_recent_events()
  preds = np.zeros(1)
  used_events = []
  losses = []
  
  for event in range(len(val_data.event_idxs)):

    # Queue events for each batch
    if val_data.timestamps[event] - last_batch_time < args.batch_duration:
      sensor = val_data.sensors[event]
      streamnet.queue_event(sensor, event)
      most_recent_event = event
      continue
    else:
      most_recent_event = event
      used_events.append(most_recent_event)
      pred = streamnet.process_batch(val_data, node_features, most_recent_event)

      # just for logging
      target = torch.tensor([train_data.labels[most_recent_event]]).float() 
      losses.append(streamnet.loss_fn(pred[np.newaxis,:], target))

      pred = np.array([torch.argmax(soft(pred.detach()))])
      preds = np.concatenate((preds, pred))

      streamnet.clear_recent_events()
      sensor = val_data.sensors[event]
      streamnet.queue_event(sensor, event)
      last_batch_time = val_data.timestamps[most_recent_event]
    
  targets = np.array(val_data.labels[used_events]) 
  preds = preds[1:]
  val_f1 = f1_score(targets, preds, average=None)
  prec, recall, f1_beta, support = precision_recall_fscore_support(targets, preds)
  print("Classes seen in validation: ")
  print(np.unique(targets))
  print("Validation F1")
  print(val_f1)
  print("Prec:")
  print(prec)
  print("recall:")
  print(recall)
  print("F1_beta")
  print(f1_beta)
  print("support")
  print(support)
  # log
  val_f1s.append(val_f1)
  val_losses.append(sum(losses)/len(losses))

  with open('train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
  with open('val_losses.pkl', 'wb') as f:
    pickle.dump(val_losses, f)
  with open('train_f1s.pkl', 'wb') as f:
    pickle.dump(train_f1s, f)
  with open('val_f1s.pkl', 'wb') as f:
    pickle.dump(val_f1s, f)
  
  torch.save(streamnet.state_dict(), "saved_models/streamnet_latest_act_detection_time_dim_2")

    



 
    


    
    




