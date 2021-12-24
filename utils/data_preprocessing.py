import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse



def preprocess(data_name):
  sensor_list, ts_list, hour_list, minute_list, second_list, label_list = [], [], [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',') # event
      sensor = int(e[0]) # which sensor 

      ts = float(e[2])
      hour = float(e[3])
      minute = float(e[4])
      sec = float(e[5])
      label = float(e[6])  # int(e[3])

      feat = np.array([float(x) for x in e[7:]])

      sensor_list.append(sensor)
      hour_list.append(hour)
      minute_list.append(minute)
      second_list.append(sec)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)
      feat_l.append(feat)
  return pd.DataFrame({'sensor': sensor_list,
                       'ts': ts_list,
                       'hour': hour_list,
                       'minute': minute_list,
                       'second': second_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)

def run(dataset):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = 'data/{}.csv'.format(dataset)
  OUT_NODE_FEAT = 'data/{}_node.npy'.format(dataset)
  OUT_DF = 'data/processed_{}.csv'.format(dataset)

  df, feat = preprocess(PATH)
  df.to_csv(OUT_DF)
  np.save(OUT_NODE_FEAT, feat)


parser = argparse.ArgumentParser('Interface for StreamNet data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name',
                    default='milan_graph_binary_normalized')

args = parser.parse_args()

run(args.data)