"""Reinforcement learning environment presenting the graph sampling problem
"""

import logging

from gym import Env
from gym.spaces import Discrete, Tuple
from gym.utils import colorize, seeding
import numpy as np

from algorithms.recovery import sparse_label_propagation

logger = logging.getLogger(__name__)

class GraphSampling(Env):

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

  def total_variance(self, x_hat):
    graph = self.graph

    tv = np.sum([
      x_hat[i] - x_hat[j]
      for i,j in graph.edges_iter()
      if (i in x_hat) and (j in x_hat)
    ])

    return tv

  def _step(self, action):
    self._validate_action(action)

    observation = action

    self.sampling_set.append(action)
    num_samples = len(self.sampling_set)

    # TODO: sparse_label_propagation should probably just return signal
    x_hat = sparse_label_propagation(
      self.graph, self.sampling_set, params={})['signal']
    lmbd = 1e-2 # TODO: make this parameter
    reward = np.sum([
      np.power(self.graph.node[i]['value'] - x_hat[i], 2.0)
      for i in self.sampling_set
    ]) + lmbd * self.total_variance(x_hat)

    done = (num_samples > self.max_samples)

    return observation, reward, done, {}
