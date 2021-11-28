import numpy as np
import random
import pandas as pd


class Data:
  def __init__(self, sensors, timestamps, event_idxs, labels, hours, minutes, seconds):
    self.sensors = sensors
    self.timestamps = timestamps
    self.event_idxs = event_idxs
    self.labels = labels
    self.n_events = len(sensors)
    self.n_unique_sensors = len(set(sensors))
    self.hours = hours
    self.minutes = minutes
    self.seconds = seconds

def compute_time_statistics(sensors, timestamps):
  last_timestamp_sensors = dict()
  all_timediffs_sensors = []
  for k in range(len(sensors)):
    event_id = sensors[k]
    c_timestamp = timestamps[k]
    if event_id not in last_timestamp_sensors.keys():
      last_timestamp_sensors[event_id] = 0
    all_timediffs_sensors.append(c_timestamp - last_timestamp_sensors[event_id])
    last_timestamp_sensors[event_id] = c_timestamp
  assert len(all_timediffs_sensors) == len(sensors)
  mean_time_shift_events = np.mean(all_timediffs_sensors)
  std_time_shift_events = np.std(all_timediffs_sensors)

  return mean_time_shift_events, std_time_shift_events

def get_data(dataset_name):
  ### Load data and train val test split
  graph_df = pd.read_csv('data/processed_{}.csv'.format(dataset_name))
  node_features = np.load('data/{}_node.npy'.format(dataset_name)) 
  print(node_features.shape)
  val_time, test_time = list(np.quantile(graph_df.ts, [0.1, 0.2])) # should be .7, .85 for 70, 15, 15 split
  sensors = graph_df.sensor.values # sensor idx for each event in order
  event_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  hours = graph_df.hour.values
  minutes = graph_df.minute.values
  seconds = graph_df.second.values

  full_data = Data(sensors, timestamps, event_idxs, labels, hours, minutes, seconds)

  random.seed(2021)

  sensor_set = set(sensors)
  

  # Compute events which appear at test time
  test_event_set = set(sensors[timestamps > val_time])

  # For train we keep edges happening before the validation time
  train_mask = timestamps <= val_time

  train_data = Data(sensors[train_mask], timestamps[train_mask],
                    event_idxs[train_mask], labels[train_mask],
                    hours[train_mask], minutes[train_mask],
                    seconds[train_mask])


  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time


  # validation and test with all edges
  val_data = Data(sensors[val_mask], timestamps[val_mask],
                    event_idxs[val_mask], labels[val_mask],
                    hours[val_mask], minutes[val_mask],
                    seconds[val_mask])

  test_data = Data(sensors[test_mask], timestamps[test_mask],
                    event_idxs[test_mask], labels[test_mask],
                    hours[test_mask], minutes[test_mask],
                    seconds[test_mask])


  print("The dataset has {} events, involving {} different sensors".format(full_data.n_events,
                                                                      full_data.n_unique_sensors))
  print("The training dataset has {} events involving {} different sensors".format(
    train_data.n_events, train_data.n_unique_sensors))
  print("The validation dataset has {} events, involving {} different sensors".format(
    val_data.n_events, val_data.n_unique_sensors))
  print("The test dataset has {} events, involving {} different sensors".format(
    test_data.n_events, test_data.n_unique_sensors))

  return node_features, full_data, train_data, val_data, test_data
