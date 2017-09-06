from networkx.readwrite import json_graph
import networkx as nx
import utils

def test_dump_load_graph():
  G = nx.Graph([(1,2)])
  data = json_graph.node_link_data(G)
  utils.dump_graph(G, "./test_graph.json")
  G2 = utils.load_graph("./test_graph.json")
  print(json_graph.node_link_data(G2))

def test_dump_results():
  utils.dump_results("./test_results.json", {"a": 1, "b": 2})
