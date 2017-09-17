import glob
import os

import networkx as nx
from networkx.readwrite import json_graph
from nose.tools import assert_dict_equal, raises

from algorithms.base import (
  GraphSamplingAlgorithm,
  GraphRecoveryAlgorithm,
)
from utils import dump_graph, dump_results

FIXTURES_BASE = "./test/fixtures/algorithms/base"

def test_graph_sampling_algorithm():
  fixture_path = FIXTURES_BASE + "/test_graph_sampling_algorithm/*"
  for filename in glob.iglob(fixture_path):
    print(filename)


class TestGraphSamplingAlgorithm(object):
  def test_create_with_graph_file(self):
    graph = nx.Graph([(0,1), (1,2)])
    graph_path = "./tmp/graph1.json"
    dump_graph(graph, graph_path)
    graph_sampling_algorithm = GraphSamplingAlgorithm(graph_path)

    # TODO: use nx.is_isomorphic?
    expected = json_graph.node_link_data(graph)
    result = json_graph.node_link_data(graph_sampling_algorithm.graph)
    assert_dict_equal(result, expected)

    os.remove(graph_path)

  def test_create_with_graph_variable(self):
    graph = nx.Graph([(0,1), (1,2)])
    graph_sampling_algorithm = GraphSamplingAlgorithm(graph)

    # TODO: use nx.is_isomorphic?
    expected = json_graph.node_link_data(graph)
    result = json_graph.node_link_data(graph_sampling_algorithm.graph)
    assert_dict_equal(result, expected)

  @raises(ValueError)
  def test_fails_with_unexpected_graph_type(self):
    invalid_type_graph = 1
    GraphSamplingAlgorithm(invalid_type_graph)

class TestGraphRecoveryAlgorithm(object):
  def test_create_with_graph_file_sample_file(self):
    graph = nx.Graph([(0,1), (1,2)])
    samples = [0,1]
    graph_path = "./tmp/graph1.json"
    samples_path = "./tmp/samples1.json"
    dump_graph(graph, graph_path)
    dump_results({'sampling_set': samples}, samples_path)
    graph_recovery_algorithm = GraphRecoveryAlgorithm(graph_path, samples_path)

    # TODO: use nx.is_isomorphic?
    expected = {
      'graph': json_graph.node_link_data(graph),
      'samples': samples
    }
    result = {
      'graph': json_graph.node_link_data(graph_recovery_algorithm.graph),
      'samples': graph_recovery_algorithm.samples
    }
    assert_dict_equal(result, expected)

    os.remove(graph_path)
    os.remove(samples_path)

  def test_create_with_graph_variable_sample_variable(self):
    graph = nx.Graph([(0,1), (1,2)])
    samples = [0,1]
    graph_recovery_algorithm = GraphRecoveryAlgorithm(graph, samples)

    # TODO: use nx.is_isomorphic?
    expected = {
      'graph': json_graph.node_link_data(graph),
      'samples': samples
    }
    result = {
      'graph': json_graph.node_link_data(graph_recovery_algorithm.graph),
      'samples': graph_recovery_algorithm.samples
    }
    assert_dict_equal(result, expected)

  @raises(ValueError)
  def test_fails_with_unexpected_graph_type(self):
    invalid_type_graph = 1
    valid_samples = [1,2,3]
    GraphRecoveryAlgorithm(invalid_type_graph, valid_samples)

  @raises(ValueError)
  def test_fails_with_unexpected_samples_type(self):
    valid_graph = nx.Graph([(0,1)])
    invalid_type_samples = 1
    GraphRecoveryAlgorithm(valid_graph, invalid_type_samples)
