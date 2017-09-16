import json

import networkx as nx

from utils import load_graph, load_samples

class GraphSamplingAlgorithm(object):
  def __init__(self, graph):
    super().__init__()

    if isinstance(graph, nx.Graph):
      self.graph = graph
    elif isinstance(graph, str):
      self.graph = load_graph(graph)
    else:
      raise ValueError("unexpected graph type: {}".format(type(graph)))

  def run(self):
    raise NotImplementedError("{} must implement run method"
                              "".format(self.__class__.__name__))

class GraphRecoveryAlgorithm(object):
  def __init__(self, graph, samples):
    super().__init__()

    if isinstance(graph, nx.Graph):
      self.graph = graph
    elif isinstance(graph, str):
      self.graph = load_graph(graph)
    else:
      raise ValueError("unexpected graph type: {}".format(type(graph)))

    if isinstance(samples, (list, tuple, set)):
      self.samples = samples
    elif isinstance(samples, str):
      self.samples = load_samples(samples)
    else:
      raise ValueError("unexpected samples type: {}".format(type(samples)))

  def run(self):
    raise NotImplementedError("{} must implement run method"
                              "".format(self.__class__.__name__))
