from algorithms.base import GraphSamplingAlgorithm
from random import choice

DEFAULT_SAMPLING_PARAMS = {
  "M": 10
}

class RandomSampling(GraphSamplingAlgorithm):
  def __init__(self, graph, sampling_params=None):
    super().__init__(graph)

    sampling_params = sampling_params or {}

    self.sampling_params = DEFAULT_SAMPLING_PARAMS.copy()
    self.sampling_params.update(sampling_params)

  def run(self):
    M = self.sampling_params["M"]
    sampling_set = set()

    while len(sampling_set) < M:
      node = choice(self.graph.nodes())
      sampling_set.add(node)

    result = { "sampling_set": sampling_set }

    return result
