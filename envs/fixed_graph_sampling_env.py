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


def generate_graph_args():
  graph_args = {
    "generator_type": "uniform",
    "seed": 1,
    "sizes": [int(draw_geometrically(10, 50)) for _ in range(5)],
    "p_in": np.random.rand() * 0.30,
    "p_out": np.random.rand() * 0.05,
    "out_path": None,
    "visualize": False,
    "cull_disconnected": True,
    "generator_type": "uniform"
  }
  return graph_args


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
    self.observation_space = MultiBinary(self.graph.number_of_nodes())

  def _generate_new_graph(self):
    if self.graph is not None and self._fixed_graph:
      return
    super().__init__(self)

  def _get_observation(self):
    neighbors = self.graph.neighbors(self._current_node)
    observation = np.zeros(self.graph.number_of_nodes(), dtype=np.bool_)
    observation[neighbors] = True
    return observation

  def get_current_nmse(self):
    graph = self.graph
    sampling_set = self.sampling_set

    x = [graph.node[i]['value'] for i in sorted(graph.nodes_iter())]
    x_hat = sparse_label_propagation(graph, list(sampling_set))

    return nmse(x, x_hat)
