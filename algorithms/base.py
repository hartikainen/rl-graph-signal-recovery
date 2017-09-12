import json

from utils import load_graph
from utils import load_samples

class GraphSamplingAlgorithm(object):
  def __init__(self, graph_file):
    super().__init__()

    self.graph_file = graph_file
    self.import_graph()

  def import_graph(self):
    self.graph = load_graph(self.graph_file)

  def run(self):
    raise NotImplementedError("{} must implement run method"
                              "".format(self.__class__.__name__))

class GraphRecoveryAlgorithm(object):
  def __init__(self, graph_file, sample_file):
    super().__init__()

    self.graph_file = graph_file
    self.import_graph()
    self.sample_file = sample_file
    self.import_samples()

  def import_graph(self):
    self.graph = load_graph(self.graph_file)

  def run(self):
    raise NotImplementedError("{} must implement run method"
                              "".format(self.__class__.__name__))
