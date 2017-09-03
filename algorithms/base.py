from utils import load_graph

class GraphRecoveryAlgorithm(object):
  def __init__(self, graph_file):
    super().__init__()

    self.graph_file = graph_file
    self.import_graph()

  def import_graph(self):
    self.graph = load_graph(self.graph_file)

  def run(self):
    raise NotImplementedError("{} must implement run method"
                              "".format(self.__name__))
