import networkx as nx
import generate_appm

from envs import GraphSamplingEnv

def test_env():
  env = GraphSamplingEnv(max_samples=5)
  for i in range(10):
    observation = env.step(i % 3)
    print(observation)
