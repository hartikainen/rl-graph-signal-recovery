import networkx as nx
import generate_appm
from nose.tools import assert_equal, assert_true

from envs import GraphSamplingEnv

def test_actions():
  env = GraphSamplingEnv(max_samples=3)
  env.reset()
  env.graph = nx.Graph()
  env.graph.add_edges_from([(0,1),(1,2),(2,3)])
  for node in env.graph:
    env.graph.node[node]['value'] = 1.0
  assert_equal(env._current_edge_idx, 0)
  env._current_node = 0
  # this node has only one edge
  env.step(1)
  assert_equal(env._current_edge_idx, 0)
  # move to next node
  env.step(2)
  assert_equal(env._current_node, 1)
  # sample this node
  env.step(0)
  # this shouldn't have effect
  env.step(0)
  env.step(1)
  assert_equal(env._current_edge_idx, 1)
  # it seems edge at index 1 leads to next node
  env.step(2)
  env.step(0)
  assert_equal(env._current_node, 2)
  env.step(1)
  env.step(2)
  obs, reward, done, info = env.step(0)
  assert_equal(env._current_node, 3)
  assert_true(reward > 0)
  assert_equal(set([1,2,3]), env.sampling_set)
  assert_true(done)
