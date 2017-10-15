"""Sparse Label Propagation

See A. Jung "Sparse Label Propagation." for algorithm definition.
"""
import numpy as np
import networkx as nx
from scipy import sparse

from algorithms.base import GraphRecoveryAlgorithm

DEFAULT_RECOVERY_PARAMS = {
  "number_of_iterations": 200,
  "lambda": 1.0,
  "alpha": 2.0,
}


def custom_incidence_matrix(graph):
  num_edges = graph.number_of_edges()
  num_nodes = graph.number_of_nodes()
  D = np.zeros((num_edges, num_nodes))
  for i, (start, end) in enumerate(graph.edges()):
    weight = graph.get_edge_data(start, end).get('weight', 1)
    D[i, start] = -1 * weight
    D[i, end] = 1 * weight
  return sparse.csr_matrix(D)


def sparse_label_propagation(graph, sampling_set_indices, params=None):
  params = dict(DEFAULT_RECOVERY_PARAMS, **params if params is not None else {})
  lambda_ = params["lambda"]
  alpha = params["alpha"]
  num_iterations = params['number_of_iterations']
  num_nodes = graph.number_of_nodes()
  num_edges = graph.number_of_edges()
  sampling_values = [graph.node[index]['value']
                     for index in sampling_set_indices]

  D = custom_incidence_matrix(graph)
  aux1 = np.squeeze(np.asarray(np.sum(np.abs(D).power(2.0 - alpha), 0)))
  aux2 = np.squeeze(np.asarray(np.sum(np.abs(D).power(alpha), 1)))

  node_scaling_matrix = sparse.spdiags(
      1.0 / (lambda_ * aux1), 0, num_nodes, num_nodes)
  edge_scaling_matrix = sparse.spdiags(
      lambda_ / aux2, 0, num_edges, num_edges)

  edge_scaled_D = sparse.csr_matrix(edge_scaling_matrix * D)
  node_scaled_D = sparse.csr_matrix(node_scaling_matrix * D.T)

  z = np.zeros((num_nodes, 1))
  xk = np.zeros((num_nodes, 1))
  hatx = np.zeros((num_nodes, 1))
  y = np.zeros((num_edges, 1))
  y_ones = np.ones_like(y)
  xk[sampling_set_indices, 0] = sampling_values
  for k in range(num_iterations):
    signal = y + (edge_scaled_D * z)
    y = (1.0 / np.max([y_ones, np.abs(signal)], axis=0)) * signal
    r = xk - (node_scaled_D * y)
    xk1 = r.copy()
    xk1[sampling_set_indices, 0] = sampling_values
    z = 2.0 * xk1 - xk
    hatx = hatx + xk1
    xk = xk1

  return hatx * (1 / (k + 1))

class SparseLabelPropagation(GraphRecoveryAlgorithm):
  def __init__(self, graph, samples, recovery_params):
    super().__init__(graph, samples)

    self.recovery_params = DEFAULT_RECOVERY_PARAMS.copy()
    self.recovery_params.update(recovery_params)

  def run(self):
    return sparse_label_propagation(self.graph,
                                     self.samples,
                                     self.recovery_params)
