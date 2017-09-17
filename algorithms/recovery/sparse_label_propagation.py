"""Sparse Label Propagation

See A. Jung "Sparse Label Propagation." for algorithm definition.
"""
import numpy as np
import networkx as nx

from algorithms.base import GraphRecoveryAlgorithm

DEFAULT_RECOVERY_PARAMS = {
  "number_of_iterations": 200,
  "compute_error": True,
}

def sparse_label_propagation(graph, sample_idx, params=None):
  # TODO: Rename samples & consider sample source: samples as used here is a
  # confusing name, as the variable contains the sampling indices. The sample
  # values are stored in the graph. Should we support fetching samples from
  # some other source than the graph?
  params = dict(DEFAULT_RECOVERY_PARAMS, **params if params is not None else {})
  compute_error = params['compute_error']
  number_of_iterations = params['number_of_iterations']
  number_of_nodes = graph.number_of_nodes()
  number_of_edges = graph.number_of_edges()
  x_shape = (number_of_nodes, 1)
  y_shape = (number_of_edges, 1)

  # TODO: would it make sense to use sparse D?
  D = np.array(nx.incidence_matrix(graph, oriented=True).todense()).T
  z = np.zeros(x_shape)
  x0 = np.zeros(x_shape)
  x1 = np.zeros(x_shape)
  x_tilde = np.zeros(x_shape)
  y = np.zeros(y_shape)

  sqrt_max_d = np.sqrt(np.max(list(graph.degree().values())))
  signal_samples = np.array([[graph.node[idx]['value']]
                             for idx in sample_idx], dtype=np.float32)

  # TODO: fix nodes with degree == 0
  degrees = 1.0 / np.array(list(graph.degree().values()), dtype=np.float32)
  gamma = np.diag(degrees)

  # TODO: should the stopping criterion be dependent of the error?
  for _ in range(number_of_iterations):
    x1 = x0 - np.dot(gamma, np.dot(D.T, y))
    x1[sample_idx] = signal_samples
    x_tilde = 2 * x1 - x0
    edge_signal = y + 0.5 * np.dot(D, x_tilde)
    y = edge_signal / np.max([np.ones(y_shape), edge_signal], axis=0)
    x0 = x1

  return x_tilde

class SparseLabelPropagation(GraphRecoveryAlgorithm):
  def __init__(self, graph, samples, recovery_params):
    super().__init__(graph, samples)

    self.recovery_params = DEFAULT_RECOVERY_PARAMS.copy()
    self.recovery_params.update(recovery_params)

  def run(self):
    return sparse_label_propagation(self.graph,
                                    self.samples,
                                    self.recovery_params)
