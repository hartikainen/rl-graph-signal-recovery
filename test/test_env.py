from env import GraphSampling
from utils import load_graph

def test_env():
  g = load_graph("./data/graphs/out2.json")
  env = GraphSampling(g)

  for action in range(5):
    env.step(action)
