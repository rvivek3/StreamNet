import logging
import numpy as np
import torch
from utils.utils import Embedding_MLP, AR_MLP
from torch import nn
from utils.utils import TemporalAttentionLayer
from model.time_encoding import TimeEncode
from collections import defaultdict
from utils.focal_loss import WeightedFocalLoss, FocalLoss


class StreamNet(torch.nn.Module):
    def __init__(self, memory_dim, embedding_dim, house_state_dim, house_time_dim, event_dim, time_dim, \
     num_classes, device, n_sensors, num_heads, dropout, mean_time_shift_events, std_time_shift_events):
        super(StreamNet, self).__init__()
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.house_state_dim = house_state_dim
        self.event_dim = event_dim
        self.time_dim = time_dim
        self.time_encoder = TimeEncode(dimension=self.time_dim) # tgn uses n_node_features instead, might wanna check this
        self.num_classes = num_classes
        self.n_unique_sensors_seen = 0 # will be used as batch size for input into GRU
        self.n_sensors = n_sensors
        self.memory_function = get_memory_function(memory_dim, event_dim, time_dim)
        self.embedding_function = get_embedding_function(memory_dim, house_state_dim, \
            embedding_dim)
        self.sensor_attention_function = get_sensor_attention_function(memory_dim, \
            house_state_dim, time_dim, house_time_dim,num_heads)
        self.activity_recognition_MLP = AR_MLP(house_state_dim, num_classes)
        self.device = device
        self.num_heads = num_heads
        self.dropout = dropout
        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = WeightedFocalLoss(num_classes = self.num_classes)
        #self.loss_fn = FocalLoss()
        self.mean_time_shift_events = mean_time_shift_events
        self.std_time_shift_events = std_time_shift_events
        self.recent_events= {}
        self.init_memory()

    def init_memory(self):
        self.memory = nn.Parameter(torch.zeros((self.n_sensors, self.memory_dim)).to(self.device),
                               requires_grad=False) # doesnt require grad b/c these are just values that are outputted and
                               # stored, not optimized
        self.house_state = nn.Parameter(torch.zeros((self.house_state_dim)).to(self.device),
                               requires_grad=False)
        self.last_memory_update = nn.Parameter(torch.zeros(self.n_sensors).to(self.device),
                                    requires_grad=False)
        self.last_house_update = nn.Parameter(torch.zeros(1).to(self.device),
                                    requires_grad=False)

    def update_memory(self, event_data, sensors, timestamps):
        sensor_memory = self.get_memory(sensors)
        self.last_memory_update[sensors] = torch.tensor(timestamps).float()
        updated_sensor_memory = self.memory_function(event_data.float(), sensor_memory.float())
        self.set_memory(sensors, updated_sensor_memory)


    def clear_recent_events(self):
        self.recent_events = {}

    def queue_event(self, sensor, event):
        self.recent_events[sensor] = event 

    def get_memory(self, sensor):
        return self.memory[sensor, :]

    def set_memory(self, sensors, updated_sensor_memory):
        self.memory[sensors,:] = updated_sensor_memory

    def detach_memory(self):
        self.memory.detach_()
        self.house_state.detach_()

    def get_updated_memory(self, sensors, timestamps, event_data):
        updated_memory = self.memory.data
        updated_memory1 = self.memory.data.clone()
        updated_memory[sensors] = self.memory_function(event_data.float(), updated_memory1[sensors].float())
        return updated_memory


    def process_batch(self, data, node_features, most_recent_event):
        sensors = []
        events = []
        for sensor, event in self.recent_events.items():
            sensors.append(sensor)
            events.append(event)

        # get the elapsed time since the last update for each changed sensor to use in its memory update
        elapsed_times = torch.tensor(data.timestamps[events]).float() - torch.LongTensor(np.array(self.last_memory_update[sensors]))
        elapsed_times = (elapsed_times - self.mean_time_shift_events) / self.std_time_shift_events
        elapsed_time_features = self.time_encoder(elapsed_times[np.newaxis])

        # Get event data together
        event_data = torch.cat((torch.tensor(node_features[events]).to(self.device),
         torch.squeeze(elapsed_time_features, dim = 0)), dim = 1)

        # update the temporal encodings of all sensors, as they've all gotten older
        self.last_memory_update[sensors] = torch.tensor(data.timestamps[events]).float()
        time_diffs = torch.LongTensor(np.array(data.timestamps[most_recent_event])) - self.last_memory_update.data
        
        time_diffs = (time_diffs - self.mean_time_shift_events) / self.std_time_shift_events
        sensor_time_features = self.time_encoder(time_diffs[np.newaxis])

        # Get updated sensor memory for forward pass
        updated_memory = self.get_updated_memory(sensors, data.timestamps[events], event_data)

        # Update the house's sinusoidal temporal feature to the current time
        hour, minute, second = data.hours[most_recent_event], data.minutes[most_recent_event], data.seconds[most_recent_event]
        house_time_encoding = self.sinusoidal_time_encode(hour, minute, second)

        # Perform attention over all sensors to update the house state

        house_state, attn_weights = self.sensor_attention_function(self.house_state.data, house_time_encoding, updated_memory, \
        sensor_time_features)

        self.house_state[:] = house_state

        pred = self.activity_recognition_MLP(np.squeeze(house_state))

        # Now, actually update the streamnet's memory
        self.update_memory(event_data, sensors, data.timestamps[events])

        return pred

    def sinusoidal_time_encode(self, hour, minute, second):
        return torch.tensor(np.array([np.sin(hour/24), np.cos(hour/24),
         np.sin(minute/(60*24)), np.cos(minute/(60*24)),
          np.sin(second/(60*60*24)), np.cos(second/(60*60*24))]))

    def process_event(self, data, event, node_features):
        # update the last updated time of the sensor
        sensor = data.sensors[event]
        self.last_memory_update[sensor] = data.timestamps[event]

        # Update the temporal encodings of the sensor
        time_diff = torch.LongTensor([data.timestamps[event]]).to(self.device) - self.last_memory_update.long()
        time_diff = (time_diff - self.mean_time_shift_events) / self.std_time_shift_events
        sensor_time_features = self.time_encoder(time_diff[np.newaxis,:])
        
    
        # Get event data together
        event_data = torch.cat((torch.tensor(node_features[event]).to(self.device), np.squeeze(sensor_time_features[:,sensor, :])))
       
        # Get updated sensor memory for forward pass
        updated_memory, last_updates = self.get_updated_memory(sensor, data.timestamps[event], event_data)

        # Update the house's temporal feature (this should probably be changed to a representation that simply captures the current
        # time, not an elapsed time since the house state last changed)
        house_time_diff = torch.LongTensor([data.timestamps[event]]).to(self.device) - self.last_house_update.long()
        house_time_diff = (house_time_diff - self.mean_time_shift_events) / self.std_time_shift_events
        house_time_data = self.time_encoder(house_time_diff[np.newaxis,:])

        self.last_house_update[:] = data.timestamps[event]
        # Update the house state using temporal attention
        # self.house_state[:], attn_weights = self.sensor_attention_function(self.house_state, house_time_data, updated_memory, \
        # sensor_time_features)

        house_state, attn_weights = self.sensor_attention_function(self.house_state.data.clone(), house_time_data, updated_memory, \
        sensor_time_features)

        pred = self.activity_recognition_MLP(np.squeeze(house_state))

        # Now, actually update the streamnet's memory
        self.update_memory(event_data, sensor, data.timestamps[event])

        return pred
    
        

def get_memory_function(memory_dim, event_dim, time_dim):
    # just a GRU that takes in old memory and concatenation of sensor event + encoding of
    # elapsed time since the last event happened
    #return nn.Transformer(nhead=16, num_encoder_layers=12)
    return nn.GRUCell(input_size=time_dim + event_dim,
                                     hidden_size=memory_dim)


    
def get_embedding_function(memory_dim, house_state_dim, embedding_dim):
    # MLP (takes in sensor's current memory and current house memory)
    return Embedding_MLP(memory_dim, house_state_dim, embedding_dim)

def get_sensor_attention_function(memory_dim, house_state_dim, time_dim, house_time_dim, num_heads):
    # temporal graph attention over all sensor embeddings
    return TemporalAttentionLayer(memory_dim, house_state_dim, house_time_dim, time_dim, n_head = num_heads)




