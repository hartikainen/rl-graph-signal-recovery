from algorithms.base import GraphSamplingAlgorithm
from random import choice

L = 20
M = 10

class RandomWalkSampling(GraphSamplingAlgorithm):
  def __init__(self, graph_file):
    super().__init__(graph_file)

  def run(self):
    sampling_set = set()

    while len(sampling_set) < M:
      node = choice(self.graph.nodes())
      for l in range(L-1):
        neighbors = self.graph.neighbors(node)
        node = choice(neighbors)

      sampling_set.add(node)

    result = { "sampling_set": sampling_set }

    return result
