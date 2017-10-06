"""Reinforcement learning environment presenting the graph sampling problem
"""
import matplotlib
matplotlib.use('Agg')

import logging
import random
from itertools import combinations

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
from graph_functions import total_variation, nmse, slp_maximum_error
from utils import draw_geometrically
from visualization import draw_partitioned_graph
import generate_appm


logger = logging.getLogger(__name__)

class SimpleThreeClusterEnv(GraphSamplingEnv):

  metadata = {
    'render.modes': ('human',),
  }

  def __init__(self, max_samples=10, render_depth=2, fixed_graph=True):
    self.graph = None
    self._fixed_graph = fixed_graph

    self._generate_new_graph()
    num_nodes = self.graph.number_of_nodes()
    self.sampling_set = set()

    # actions: { 0: sample, 1: move }
    self.action_space = Discrete(2)

    self.observation_space = Discrete(1)
    x = [self.graph.node[idx]['value'] for idx in range(num_nodes)]
    self.slp_maximum_error = slp_maximum_error(x)

    self._current_node = 0
    self._clustering_coefficients = None
    self._max_samples = max_samples
    self._render_depth = render_depth
    self._screen = None

    self.reset()

  def _generate_new_graph(self):
    if self.graph is not None and self._fixed_graph:
      return

    edge_list = [
      (0,1), (0,2), (1,2), # intra cluster 1
      (3,4), (3,5), (4,5), # intra cluster 1
      (6,7), (6,8), (7,8), # intra cluster 1
      (2,5), (2,6), (5,6), # inter cluster
    ]

    self.graph = nx.Graph(edge_list)

    clusters = [
      { "nodes": [0,1,2], "value": 0.1 },
      { "nodes": [3,4,5], "value": 0.7 },
      { "nodes": [6,7,8], "value": 0.9 },
    ]

    for cluster in clusters:
      cluster_value = cluster["value"]
      for node_idx in cluster["nodes"]:
        self.graph.node[node_idx]['value'] = cluster_value

    self.graph.graph['partition'] = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}]

  def _get_observation(self):
    # TODO: implement
    observation = None
    return observation

  def _get_next_node(self):
    # TODO: implement
    neighbors = self.graph.neighbors(self._current_node)
    return np.random.choice(neighbors)

  def _do_action(self, action):
    # actions: { 0: sample, 1: move }
    if action == 0:
      self.sampling_set.add(self._current_node)
      self._randomize_position()
    elif action == 1:
      self._current_node = self._get_next_node()

  def _step(self, action):
    self._validate_action(action)
    self._do_action(action)
    observation = self._get_observation()

    num_samples = len(self.sampling_set)
    reward = 0.0

    done = (num_samples >= self._max_samples
            or num_samples >= self.graph.number_of_nodes())

    if done:
      current_nmse = self.get_current_nmse()
      reward = 1.0/current_nmse

    return observation, reward, done, {}

  def get_current_nmse(self):
    graph = self.graph
    sampling_set = self.sampling_set

    x = [graph.node[i]['value'] for i in sorted(graph.nodes_iter())]
    x_hat = sparse_label_propagation(graph, list(sampling_set))

    return nmse(x, x_hat)
