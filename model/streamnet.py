import logging
import numpy as np
import torch
from StreamNet.utils.utils import Embedding_MLP, AR_MLP
from torch import nn

class StreamNet(torch.nn.Module):
    def __init__(self, memory_dim, embedding_dim, house_state_dim, event_dim, time_dim, \
     num_classes):
        super(StreamNet, self).__init__()
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.house_state_dim = house_state_dim
        self.event_dim = event_dim
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.n_unique_sensors_seen = 0 # will be used as batch size for input into GRU
        self.memory_funtion = get_memory_function(memory_dim, event_dim, time_dim)
        self.embedding_function = get_embedding_function(memory_dim, house_state_dim, \
            embedding_dim)
        self.sensor_attention_function = get_sensor_attention_function(embedding_dim)
        self.activity_recognition_MLP = AR_MLP(house_state_dim, num_classes)

def get_memory_function(memory_dim, event_dim, time_dim):
    # just a GRU that takes in old memory and concatenation of sensor event + encoding of
    # elapsed time since the last event happened
    return nn.GRUCell(input_size=memory_dim + event_dim + time_dim,
                                     hidden_size=memory_dim)

    
def get_embedding_function(memory_dim, house_state_dim, embedding_dim):
    # MLP (takes in sensor's current memory and current house memory)
    return Embedding_MLP(memory_dim, house_state_dim, embedding_dim)

def get_sensor_attention_function(memory_dim):
    # temporal graph attention over all sensor embeddings




