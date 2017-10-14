"""Reinforcement learning environment presenting the graph sampling problem
"""
import logging
import random

import numpy as np
import networkx as nx
import pygame
from gym import Env
from gym.spaces import Discrete, Tuple, Box, MultiBinary
from gym.utils import colorize, seeding

from algorithms.recovery import sparse_label_propagation
from envs import GraphSamplingEnv
from graph_functions import total_variation, nmse, slp_maximum_error
from utils import draw_geometrically
from visualization import draw_partitioned_graph
import generate_appm


def generate_graph_args():
  graph_args = {
    "generator_type": "uniform",
    "sizes": [4, 4, 4],
    "p_in": 0.5,
    "p_out": 0.1,
    "out_path": None,
    "visualize": False,
    "cull_disconnected": False,
    "connect_disconnected": True,
    "generator_type": "uniform"
  }
  return graph_args


logger = logging.getLogger(__name__)

class FixedGraphSamplingEnv(GraphSamplingEnv):

  metadata = {
    'render.modes': ('human',),
  }

  def __init__(self, max_samples=3, render_depth=2, fixed_graph=True):
    self.graph = None
    self._fixed_graph = fixed_graph
    super().__init__(max_samples, render_depth)

  def _generate_new_graph(self):
    if self.graph is not None and self._fixed_graph:
      return
    graph_args = generate_graph_args()
    graph = generate_appm.main(graph_args)
    self.graph = graph
