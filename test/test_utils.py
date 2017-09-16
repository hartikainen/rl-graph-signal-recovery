import os
import json

from nose.tools import assert_dict_equal, assert_equal
from networkx.readwrite import json_graph
import networkx as nx

import utils

def test_dump_load_graph():
  graph1 = nx.Graph([(1,2)])
  utils.dump_graph(graph1, "./tmp/test_graph.json")
  graph2 = utils.load_graph("./tmp/test_graph.json")

  expected = json_graph.node_link_data(graph1)
  result = json_graph.node_link_data(graph2)
  assert_dict_equal(result, expected)

  os.remove("./tmp/test_graph.json")

def test_dump_results():
  expected = {"a": 1, "b": 2}
  utils.dump_results(expected, "./tmp/test_results.json")

  with open('./tmp/test_results.json', 'r') as f:
    result = json.load(f)

  assert_dict_equal(result, expected)

  os.remove("./tmp/test_results.json")
