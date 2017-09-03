from algorithms.base import GraphSamplingAlgorithm

class RandomWalkSampling(GraphSamplingAlgorithm):
  def __init__(self, graph_file):
    super().__init__(graph_file)

  def run(self):
    print("{}.run".format(self.__class__.__name__))
    results = {}
    return results
