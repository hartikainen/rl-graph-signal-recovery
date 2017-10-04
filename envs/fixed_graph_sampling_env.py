"""Reinforcement learning environment presenting the graph sampling problem
"""
import matplotlib
matplotlib.use('Agg')

import logging
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pygame
from gym import Env
from gym.spaces import Discrete, Tuple, Box, MultiBinary
from gym.utils import colorize, seeding

from algorithms.recovery import sparse_label_propagation
from envs import GraphSamplingEnv
from graph_functions import total_variance, nmse, slp_maximum_error
from utils import draw_geometrically
from visualization import draw_partitioned_graph
import generate_appm


logger = logging.getLogger(__name__)

class FixedGraphSamplingEnv(GraphSamplingEnv):

  metadata = {
    'render.modes': ('human',),
  }

  def __init__(self, max_samples=10, render_depth=2, fixed_graph=True):
    self.graph = None
    self._fixed_graph = fixed_graph
    super()._generate_new_graph()
    super().__init__(max_samples, render_depth)
    self.observation_space = Discrete(2)
    x = [self.graph.node[idx]['value']
         for idx in range(self.graph.number_of_nodes())]
    self.slp_maximum_error = slp_maximum_error(x)

  def _generate_new_graph(self):
    if self.graph is not None and self._fixed_graph:
      return
    super().__init__(self)

  def _get_observation(self):
    neighbors = self.graph.neighbors(self._current_node)
    observation = np.zeros(2)
    observation[-2] = self._current_node
    observation[-1] = neighbors[self._current_edge_idx]

    return observation

  def _step(self, action):
    self._validate_action(action)
    self._do_action(action)
    observation = self._get_observation()

    num_samples = len(self.sampling_set)
    reward = 0.0

    done = (num_samples >= self._max_samples
            or num_samples >= self.graph.number_of_nodes())

    if done:
      x_hat = sparse_label_propagation(self.graph, list(self.sampling_set))
      x = [self.graph.node[idx]['value']
           for idx in range(self.graph.number_of_nodes())]

      error = nmse(x, x_hat)
      reward = (self.slp_maximum_error - error) / self.slp_maximum_error

    return observation, reward, done, {}

  def get_current_nmse(self):
    graph = self.graph
    sampling_set = self.sampling_set

    x = [graph.node[i]['value'] for i in sorted(graph.nodes_iter())]
    x_hat = sparse_label_propagation(graph, list(sampling_set))

    return nmse(x, x_hat)
