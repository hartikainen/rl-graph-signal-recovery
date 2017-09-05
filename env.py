"""Reinforcement learning environment presenting the graph sampling problem
"""

import logging

from gym import Env
from gym.spaces import Discrete, Tuple
from gym.utils import colorize, seeding
import numpy as np

logger = logging.getLogger(__name__)

class GraphSampling(Env):

  metadata = {
    'render.modes': ('human',),
  }

  def __init__(self, nx_graph):
    self.graph = nx_graph
    num_nodes = nx_graph.number_of_nodes()
    # TODO: action_space == graph nodes
    self.action_space = spaces.Discrete(num_nodes)
    # TODO: observation_space == graph nodes
    self.observation_space = spaces.Discrete(num_nodes)

    self._seed()
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
    # TODO: this should use x_hat, not self.graph

    tv = np.sum([
      self.graph[i]['value'] - self.graph[j]['value']
      for i,j in graph.edges_iter()
    ])

    return tv

  def _step(self, action):
    self._validate_action(action)

    observation = action

    self.sampling_set.append(action)
    num_samples = len(self.sampling_set)
    done = (num_samples > self.max_samples)

    reward = 0.0
    if done:
      lmbd = 1e-2 # TODO: make this parameter
      x_hat = 0.0 # TODO: add \hat{x}
      reward = np.sum([
        np.pow(self.graph.node[i]['value'] - x_hat, 2.0)
        for i in self.sampling_set
      ]) + lmbd * self.total_variance(x_hat))


    return observation, reward, done, {}
