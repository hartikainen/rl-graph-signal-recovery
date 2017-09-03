from algorithms.base import GraphRecoveryAlgorithm

class SparseLabelPropagation(GraphRecoveryAlgorithm):
  def __init__(self, graph_file, sample_file):
    super().__init__(graph_file, sample_file)

  def run(self):
    print("{}.run".format(self.__class__.__name__))
    results = {}
    return results
