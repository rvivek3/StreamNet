import numpy as np
import torch
from torch import nn


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)

    
class AR_MLP(torch.nn.Module):
  def __init__(self, house_state_dim, num_classes, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(house_state_dim, 60)
    self.fc_2 = torch.nn.Linear(60, 20)
    self.fc_3 = torch.nn.Linear(20, num_classes)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)

class Embedding_MLP(torch.nn.Module):
  def __init__(self, memory_dim, embedding_dim, house_state_dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(house_state_dim + memory_dim, 60)
    self.fc_2 = torch.nn.Linear(60, 20)
    self.fc_3 = torch.nn.Linear(20, embedding_dim)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x): # x should be house state and sensor memory concatenated
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)

class TemporalAttentionLayer(torch.nn.Module):
  """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
  """

  def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
               output_dimension, n_head=2,
               dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.n_head = n_head

    self.feat_dim = n_node_features
    self.time_dim = time_dim

    self.query_dim = n_node_features + time_dim
    self.key_dim = n_neighbors_features + time_dim + n_edge_features

    self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=n_head,
                                                   dropout=dropout)

  def forward(self, house_state, house_time_features, sensors_features,
            sensors_time_features):
    """
    "Temporal attention model
    # batch size always 1 b/c there's only 1 house
    :param house_state: float Tensor of shape [batch_size, house_state_dim] 

    # temporal encoding of current time
    :param house_time_features: float Tensor of shape [batch_size, 1, time_dim] 
    :param neighbors_features: float Tensor of shape [batch_size, n_sensors, n_node_features]
    :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors, time_dim]
    :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
    :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
    :return:
    attn_output: float Tensor of shape [1, batch_size, n_node_features]
    attn_output_weights: [batch_size, 1, n_neighbors]
    """

    src_node_features_unrolled = torch.unsqueeze(house_state, dim=1)

    query = torch.cat([src_node_features_unrolled, house_time_features], dim=2)
    key = torch.cat([sensors_features, sensors_time_features], dim=2)

    #Reshape tensors so to expected shape by multi head attention
    query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
    key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key)

    attn_output = attn_output.squeeze()
    attn_output_weights = attn_output_weights.squeeze()

    # Skip connection with temporal attention over neighborhood and the features of the node itself
    attn_output = self.merger(attn_output, house_state)

    return attn_output, attn_output_weights

