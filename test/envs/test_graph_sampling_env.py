import networkx as nx
import generate_appm

from envs import GraphSampling, GraphSampling2

def test_env():
  graph_args = {
    "seed": 1,
    "sizes": [10, 20, 30, 40],
    "p_in": 0.3,
    "p_out": 0.04,
    "num_graphs": 1,
    "out_path": None,
    "visualize": False,
    "cull_disconnected": True
  }
  graph = generate_appm.main(graph_args)
  env = GraphSampling(graph)
  for i in range(10):
    observation = env.step(i % 3)
    print(observation)
