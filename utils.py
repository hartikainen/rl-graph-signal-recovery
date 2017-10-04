import os
import pathlib
import json
import pickle

import numpy as np
from networkx.readwrite import json_graph

class ObjectEncoder(json.JSONEncoder):
  def default(self, obj):
    if hasattr(obj, "to_json"):
      return self.default(obj.to_json())
    if isinstance(obj, set):
      return list(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
DEFAULT_JSON_ARGS = {
  "indent": 2,
  "separators": (',', ': '),
  "cls": ObjectEncoder
}

def dump_graph(nx_graph, filepath):
  dirpath = os.path.dirname(filepath)
  pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)

  data = json_graph.node_link_data(nx_graph)

  with open(filepath, "w") as f:
    json.dump(data, f, **DEFAULT_JSON_ARGS)

def load_graph(load_path):
  with open(load_path, "r") as f:
    data = json.load(f)
    nx_graph = json_graph.node_link_graph(data)
    return nx_graph

def dump_results(results, filepath):
  dirpath = os.path.dirname(filepath)
  pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)

  # TODO: might need to write these in pickle
  with open(filepath, "w") as f:
    json.dump(results, f, sort_keys=True, **DEFAULT_JSON_ARGS)

def load_samples(load_path):
  with open(load_path, 'r') as f:
    data = json.load(f)
    sampling_set = data['sampling_set']
  return sampling_set

def dump_pickle(data, filepath):
  dirpath = os.path.dirname(filepath)
  pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)

  with open(filepath, "wb") as f:
    pickle.dump(data, f)

def load_pickle(filepath):
  data = None
  with open(filepath, "rb") as f:
    data = pickle.load(f)
  return data

def draw_geometrically(low, high):
  return np.power(10, np.random.uniform(np.log10(low), np.log10(high)))
