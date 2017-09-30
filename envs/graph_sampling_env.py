"""Reinforcement learning environment presenting the graph sampling problem
"""

import logging
import random

from gym import Env
from gym.spaces import Discrete, Tuple, Box
from gym.utils import colorize, seeding
import numpy as np
import networkx as nx

from algorithms.recovery import sparse_label_propagation
from graph_functions import total_variance
from utils import draw_geometrically
import generate_appm


def generate_graph_args():
  graph_args = {
    "seed": 1,
    "sizes": [int(draw_geometrically(10, 50)) for _ in range(5)],
    "p_in": np.random.rand() * 0.30,
    "p_out": np.random.rand() * 0.05,
    "out_path": None,
    "visualize": False,
    "cull_disconnected": True
  }
  return graph_args


logger = logging.getLogger(__name__)

class GraphSamplingEnv(Env):

  metadata = {
    'render.modes': ('human',),
  }

  def __init__(self, max_samples=10):
    self._generate_new_graph()
    num_nodes = self.graph.number_of_nodes()
    self.sampling_set = set()

    # actions: 0: sample 1: next edge 2: move
    self.action_space = Discrete(3)

    observation_max_value = np.iinfo(np.int32)

    self.observation_space = Box(
        np.array([0] * 10),
        np.array([observation_max_value] * 10))
    self._current_node = 0
    self._current_edge_idx = 0
    self._clustering_coefficients = None
    self._max_samples = 10

    self._seed()
    self.reset()

  def _generate_new_graph(self):
    graph_args = generate_graph_args()
    graph = generate_appm.main(graph_args)
    self.graph = graph

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _randomize_position(self):
    self._current_node = random.sample(self.graph.nodes(), 1)[0]
    self._current_edge_idx = np.random.randint(
        len(self.graph.neighbors(self._current_node)))

  def _reset(self):
    self._generate_new_graph()
    self.sampling_set = set()
    self._randomize_position()
    self._clustering_coefficients = nx.clustering(self.graph)
    return self._get_observation()

  def _validate_action(self, action):
    if not self.action_space.contains(action):
      raise ValueError("{} ({}) invalid".format(action, type(action)))

  def _get_next_node(self):
    neighbors = self.graph.neighbors(self._current_node)
    return neighbors[self._current_edge_idx]

  def _get_observation(self):
    neighbors = self.graph.neighbors(self._current_node)
    neighborhood_coefficients = [
        self._clustering_coefficients[i] for i in neighbors]
    clustering_coefficients = [
        self._clustering_coefficients[self._current_node],
        self._clustering_coefficients[self._get_next_node()],
        np.mean(neighborhood_coefficients),
        np.max(neighborhood_coefficients),
        np.min(neighborhood_coefficients)
    ]
    neighborhood_degrees = [
        self.graph.degree(i) for i in neighbors]
    degrees = [
        self.graph.degree(self._current_node),
        self.graph.degree(self._get_next_node()),
        np.mean(neighborhood_degrees),
        np.max(neighborhood_degrees),
        np.min(neighborhood_degrees)
    ]
    return np.array((*clustering_coefficients,
                     *degrees))

  def _do_action(self, action):
    # actions: 0: sample 1: next edge 2: move
    if action == 0:
      self.sampling_set.add(self._current_node)
      self._randomize_position()
    elif action == 1:
      self._current_edge_idx = ((self._current_edge_idx + 1)
                                % self.graph.degree(self._current_node))
    elif action == 2:
      self._current_node = self._get_next_node()
      self._current_edge_idx = 0

  def _step(self, action):
    self._validate_action(action)
    self._do_action(action)
    observation = self._get_observation()

    reward = 0.0
    done = False

    if action == 0 and len(self.sampling_set) > 0 and not done:
      x_hat = sparse_label_propagation(self.graph, list(self.sampling_set))
      lambda_ = 1e-2 # TODO: make this parameter
      x = [self.graph.node[idx]['value']
           for idx in range(self.graph.number_of_nodes())]

      error = nmse(x, x_hat)
      tv = total_variance(self.graph.edges(), x_hat)
      reward = error + lambda_ * tv

    done = (len(self.sampling_set) >= self._max_samples)

    return observation, reward, done, {}
