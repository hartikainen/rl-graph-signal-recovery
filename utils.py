import json
from networkx.readwrite import json_graph

def dump_graph(nx_graph, dump_path):
  data = json_graph.node_link_data(nx_graph)

  with open(dump_path, "w") as f:
    json.dump(data, f, indent=2, separators=(',', ': '))

def load_graph(load_path):
  with open(load_path, "r") as f:
    data = json.load(f)
    nx_graph = json_graph.node_link_graph(data)
    return nx_graph
