import networkx as nx

from envs.graph_sampling_env import GraphSampling

def create_graph():
  g = nx.random_partition_graph([5,5], 1.0, 1.0)
  for node_index in g:
    g.node[node_index]['value'] = 1.0

  return g

def test_env():
  g = create_graph()
  env = GraphSampling(g)
  for i in range(10):
    env.step(i % 3)
