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

    # flattened adj. matrix upper triangle + 4x vector of length num_nodes
    observation_length = sum(range(self.num_nodes + 1)) + 4 * self.num_nodes
    self.observation_space = Box(0, 1, observation_length)
    x = [self.graph.node[idx]['value']
         for idx in range(self.num_nodes)]
    self.slp_maximum_error = slp_maximum_error(x)

  def _generate_new_graph(self):
    if self.graph is not None and self._fixed_graph:
      return
    graph_args = generate_graph_args()
    graph = generate_appm.main(graph_args)
    self.graph = graph

  def _get_observation(self):
    state_descriptor = np.zeros((self.num_nodes, 3))
    # current node indicator
    state_descriptor[self._current_node, 0] = 1.
    # sampling set indicator
    for node in self.sampling_set:
      state_descriptor[node, 1] = 1.
    # next node indicator
    current_neighbor = self._get_next_node()
    state_descriptor[current_neighbor, 2] = 1.

    state_descriptor = np.reshape(state_descriptor, (-1))
    observation = np.hstack((self._adjacency_vector,
                             self._clustering_coefficient_vector,
                             state_descriptor))
    return observation

  def _get_next_node(self):
    neighbors = self.graph.neighbors(self._current_node)
    return neighbors[self._current_edge_idx]

  def _do_action(self, action):
    # actions: { 0: sample, 1: next edge 2: next node }
    if action == 0:
      self.sampling_set.add(self._current_node)
    elif action == 1:
      self._current_edge_idx = ((self._current_edge_idx + 1)
                                % self.graph.degree(self._current_node))
    elif action == 2:
      self._current_node = self._get_next_node()
      self._current_edge_idx = 0
    self._num_actions += 1

  def _reward(self):
    x_hat = sparse_label_propagation(self.graph, list(self.sampling_set))
    x = [self.graph.node[idx]['value']
         for idx in range(self.graph.number_of_nodes())]
    error = nmse(x, x_hat)
    self.error = error
    reward = (self.slp_maximum_error - error) / self.slp_maximum_error
    reward += 0.1 * 1.0 / self._num_actions
    return reward

  def _step(self, action):
    self._validate_action(action)
    observation = self._get_observation()

    num_samples = len(self.sampling_set)
    reward = 0.0

    done = (num_samples >= self._max_samples
            or num_samples >= self.graph.number_of_nodes())

    if done:
      reward = self._reward()

    self._do_action(action)

    return observation, reward, done, {}

  def _reset(self):
    self.num_nodes = self.graph.number_of_nodes()
    self.sampling_set = set()
    self._clustering_coefficients = nx.clustering(self.graph)
    self._current_node = np.random.randint(self.graph.number_of_nodes())
    self._current_edge_idx = 0
    self._num_actions = 0
    self.error = 0
    adjacency_matrix = nx.adjacency_matrix(self.graph).todense()
    self._adjacency_vector = np.squeeze(np.array(
      adjacency_matrix[np.triu_indices_from(adjacency_matrix)]))
    self._clustering_coefficient_vector = np.array(
        [self._clustering_coefficients[i] for i in range(self.num_nodes)])
    return self._get_observation()

  def get_current_nmse(self):
    graph = self.graph
    sampling_set = self.sampling_set

    x = [graph.node[i]['value'] for i in sorted(graph.nodes_iter())]
    x_hat = sparse_label_propagation(graph, list(sampling_set))

    return nmse(x, x_hat)
