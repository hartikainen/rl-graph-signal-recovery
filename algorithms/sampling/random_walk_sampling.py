from algorithms.base import GraphSamplingAlgorithm
from random import choice

DEFAULT_SAMPLING_PARAMS = {
  "L": 20,
  "M": 10
}

class RandomWalkSampling(GraphSamplingAlgorithm):
  def __init__(self, graph_file, sampling_params):
    super().__init__(graph_file)

    self.sampling_params = DEFAULT_SAMPLING_PARAMS.copy()
    self.sampling_params.update(sampling_params)

  def run(self):
    L = self.sampling_params["L"]
    M = self.sampling_params["M"]
    sampling_set = set()

    while len(sampling_set) < M:
      node = choice(self.graph.nodes())
      for l in range(L-1):
        neighbors = self.graph.neighbors(node)
        node = choice(neighbors)

      sampling_set.add(node)

    result = { "sampling_set": sampling_set }

    return result
