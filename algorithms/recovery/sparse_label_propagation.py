from algorithms.base import GraphRecoveryAlgorithm

DEFAULT_RECOVERY_PARAMS = {}

class SparseLabelPropagation(GraphRecoveryAlgorithm):
  def __init__(self, graph_file, sample_file, recovery_params):
    super().__init__(graph_file, sample_file)

    self.recovery_params = DEFAULT_RECOVERY_PARAMS.copy()
    self.recovery_params.update(recovery_params)

  def run(self):
    graph = self.graph
    samples = self.samples

    result = {}
    return result
