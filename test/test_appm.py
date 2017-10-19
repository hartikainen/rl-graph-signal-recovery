import networkx as nx
import numpy as np
from nose.tools import assert_equal, assert_true

import generate_appm

def test_simple_adjacency():
  graph_args = {
    "generator_type": "uniform",
    "sizes": [2, 2],
    "p_in": 1.0,
    "p_out": 1.0,
    "out_path": None,
    "visualize": False,
    "cull_disconnected": False,
    "connect_disconnected": True,
    "generator_type": "uniform",
    "shuffle_labels": False
  }
  graph = generate_appm.main(graph_args)
  adjacency_matrix = nx.adjacency_matrix(graph).todense()
  correct_adjacency = np.ones((4,4))
  # no self-edges
  correct_adjacency[np.arange(4), np.arange(4)] = 0
  assert_true(np.allclose(adjacency_matrix, correct_adjacency))

def test_shuffle_adjacency():
  graph_args = {
    "generator_type": "uniform",
    "sizes": [2, 2],
    "p_in": 1.0,
    "p_out": .0,
    "out_path": None,
    "visualize": False,
    "cull_disconnected": False,
    "connect_disconnected": False,
    "generator_type": "uniform",
    "shuffle_labels": True
  }
  graph = generate_appm.main(graph_args)
  adjacency_matrix = nx.adjacency_matrix(
      graph, np.arange(graph.number_of_nodes())).todense()

  node_list = list(graph.nodes())
  first_node_index = node_list[0]
  second_node_index = node_list[1]
  third_node_index = node_list[2]

  # first and second should have an edge
  assert_equal(adjacency_matrix[first_node_index,
                                second_node_index], 1)
  # first and third should not have an edge
  assert_equal(adjacency_matrix[first_node_index,
                                third_node_index], 0)
