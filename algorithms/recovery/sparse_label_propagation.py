"""Sparse Label Propagation

See A. Jung "Sparse Label Propagation." for algorithm definition.
"""
from algorithms.base import GraphRecoveryAlgorithm
import numpy as np
import networkx as nx

DEFAULT_RECOVERY_PARAMS = {
  "number_of_iterations": 2000,
}

class SparseLabelPropagation(GraphRecoveryAlgorithm):
  def __init__(self, graph_file, sample_file, recovery_params):
    super().__init__(graph_file, sample_file)

    self.recovery_params = DEFAULT_RECOVERY_PARAMS.copy()
    self.recovery_params.update(recovery_params)

  def run(self):
    graph = self.graph
    # TODO: Rename samples & consider sample source: samples as used here is a
    # confusing name, as it refers to the sampling indices. The sample values
    # are stored in the graph. Should we support fetching samples from some
    # other source than the graph?
    samples = self.samples

    number_of_iterations = DEFAULT_RECOVERY_PARAMS['number_of_iterations']
    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()
    x_shape = (number_of_nodes, 1)
    y_shape = (number_of_edges, 1)

    # TODO: would it make sense to use sparse D?
    D = np.array(nx.incidence_matrix(graph).todense()).T
    z = np.zeros(x_shape)
    x0 = np.zeros(x_shape)
    x1 = np.zeros(x_shape)
    x_hat = np.zeros(x_shape)
    y = np.zeros(y_shape)

    sqrt_max_d = np.sqrt(np.max(list(graph.degree().values())))
    signal_samples = np.array([[graph.node[idx]['value']] for idx in samples])

    # TODO: check results, error to full signal doesn't look promising
    for i in range(number_of_iterations):
      edge_signal = y + 0.5 * sqrt_max_d * np.dot(D, z)
      y = edge_signal / np.max([1.0, np.linalg.norm(edge_signal)])
      x1 = x0 - 0.5 * sqrt_max_d * np.dot(D.T, y)
      x1[samples] = signal_samples
      z = 2 * x1 - x0
      x_hat += x1
      x0 = x1

    x_hat *= 1 / (i + 1)

    result = {'signal': x_hat}
    return result
