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

logger = logging.getLogger(__name__)

class GraphSampling(Env):

  metadata = {
    'render.modes': ('human',),
  }

  def __init__(self, nx_graph, max_samples=10):
    self.graph = nx.convert_node_labels_to_integers(nx_graph)
    num_nodes = nx_graph.number_of_nodes()
    self.sampling_set = set()

    # actions: 0: sample 1: next edge 2: move
    self.action_space = Discrete(3)

    # observation: current node, number of edges from the current node, the
    # currently selected edge, number of edges from the next node, ...
    max_d = np.max(list(self.graph.degree().values()))
    self.observation_space = Box(
        np.array([0,0,0,0]),
        np.array([self.graph.number_of_nodes(), max_d, max_d, max_d]))
    self._current_node = 0
    self._current_edge_idx = 0

    self._seed()
    self.reset()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _reset(self):
    self.sampling_set = set()
    self._current_node = random.sample(self.graph.nodes(), 1)[0]
    self._current_edge_idx = np.random.randint(
        len(self.graph.neighbors(self._current_node)))
    return self._get_observation()

  def _validate_action(self, action):
    if not self.action_space.contains(action):
      raise ValueError("{} ({}) invalid".format(action, type(action)))

  def _get_next_node(self):
    neighbors = self.graph.neighbors(self._current_node)
    return neighbors[self._current_edge_idx]

  def _get_observation(self):
    num_edges = self.graph.degree(self._current_node)
    num_next_edges = self.graph.degree()[self._get_next_node()]
    return np.array((self._current_node,
                     num_edges,
                     self._current_edge_idx,
                     num_next_edges))

  def _do_action(self, action):
    # actions: 0: sample 1: next edge 2: move
    if action == 0:
      self.sampling_set.add(self._current_node)
    elif action == 1:
      self._current_edge_idx = ((self._current_edge_idx + 1)
                                % self.graph.degree(self._current_node))
    elif action == 2:
      self._current_node = self._get_next_node()

  def _step(self, action):
    self._validate_action(action)
    self._do_action(action)
    observation = self._get_observation()

    reward = 0.0
    done = False

    if action == 0 and len(self.sampling_set) > 0:
      x_hat = sparse_label_propagation(self.graph, list(self.sampling_set))
      lmbd = 1e-2 # TODO: make this parameter
      reward = np.sum([
        np.power(self.graph.node[i]['value'] - x_hat[i], 2.0)
        for i in self.sampling_set
      ]) + lmbd * total_variance(self.graph.edges(), x_hat)

    return observation, reward, done, {}

class GraphSampling2(Env):

  metadata = {
    'render.modes': ('human',),
  }

  def __init__(self, nx_graph, max_samples=10):
    self.graph = nx_graph
    # TODO: max_samples?
    self.max_samples = max_samples
    num_nodes = nx_graph.number_of_nodes()
    # TODO: action_space == graph nodes
    self.action_space = Discrete(num_nodes)
    # TODO: observation_space == graph nodes
    self.observation_space = Discrete(num_nodes)

    self._seed()
    self.reset()
    self.state = None

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _reset(self):
    self.sampling_set = []
    self.state = self.graph
    return self.state

  def _validate_action(self, action):
    if not self.action_space.contains(action):
      raise ValueError("{} ({}) invalid".format(action, type(action)))

  def _step(self, action):
    self._validate_action(action)

    observation = action

    self.sampling_set.append(action)
    num_samples = len(self.sampling_set)

    # TODO: sparse_label_propagation should probably just return signal
    x_hat = sparse_label_propagation(self.graph, self.sampling_set, params={})
    lmbd = 1e-2 # TODO: make this parameter
    reward = np.sum([
      np.power(self.graph.node[i]['value'] - x_hat[i], 2.0)
      for i in self.sampling_set
    ]) + lmbd * total_variance(self.graph.edges(), x_hat)

    done = (num_samples > self.max_samples)

    return observation, reward, done, {}
