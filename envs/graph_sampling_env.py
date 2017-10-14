"""Reinforcement learning environment presenting the graph sampling problem
"""
import matplotlib
matplotlib.use('Agg')

import logging
import random
import pygame

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from gym import Env
from gym.spaces import Discrete, Tuple, Box
from gym.utils import colorize

from algorithms.recovery import sparse_label_propagation
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

class GraphSamplingEnv(Env):

  metadata = {
    'render.modes': ('human',),
  }

  def __init__(self, max_samples=3, render_depth=3):
    self._generate_new_graph()
    self.sampling_set = set()

    # actions: 0: sample 1: next edge 2: move
    self.action_space = Discrete(3)

    self._current_node = 0
    self._current_edge_idx = 0
    self._clustering_coefficients = None
    self._max_samples = max_samples
    self._render_depth = render_depth
    self._screen = None

    self.reset()

    # flattened adj. matrix upper triangle + 4x vector of length num_nodes
    observation_length = sum(range(self.num_nodes + 1)) + 4 * self.num_nodes
    self.observation_space = Box(0, 1, observation_length)
    x = [self.graph.node[idx]['value']
         for idx in range(self.num_nodes)]
    self.slp_maximum_error = slp_maximum_error(x)

  def _generate_new_graph(self):
    graph_args = generate_graph_args()
    graph = generate_appm.main(graph_args)
    self.graph = graph

  def _randomize_position(self):
    self._current_node = random.sample(self.graph.nodes(), 1)[0]
    self._current_edge_idx = np.random.randint(
      len(self.graph.neighbors(self._current_node)))

  def _reset(self):
    self._generate_new_graph()
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

  def _validate_action(self, action):
    if not self.action_space.contains(action):
      raise ValueError("{} ({}) invalid".format(action, type(action)))

  def _get_next_node(self):
    neighbors = self.graph.neighbors(self._current_node)
    return neighbors[self._current_edge_idx]

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
    self._do_action(action)
    observation = self._get_observation()

    num_samples = len(self.sampling_set)
    reward = 0.0

    done = (num_samples >= self._max_samples
            or num_samples >= self.graph.number_of_nodes())

    if done:
      reward = self._reward()

    return observation, reward, done, {}

  def get_current_nmse(self):
    graph = self.graph
    sampling_set = self.sampling_set

    x = [graph.node[i]['value'] for i in sorted(graph.nodes_iter())]
    x_hat = sparse_label_propagation(graph, list(sampling_set))

    return nmse(x, x_hat)

  def _render(self, mode='human', close=False):
    if close:
      if self._screen is not None:
        pygame.display.quit()
        pygame.quit()
      return

    if self._screen is None:
      pygame.init()
      self._screen= pygame.display.set_mode(
          (800, 800))

    subgraph_paths = nx.single_source_shortest_path(
        self.graph, self._current_node,
        cutoff=self._render_depth)
    subgraph = self.graph.subgraph(subgraph_paths)

    nodelist = [[] for _ in range(self._render_depth+1)]
    nodelist_1d = []
    node_color = []
    for key, path in subgraph_paths.items():
      level = len(path) - 1
      nodelist[level].append(key)
      nodelist_1d.append(key)
      if key in self.sampling_set:
        node_color.append('r')
      else:
        node_color.append('g')

    for level in range(self._render_depth + 1):
      if len(nodelist[level]) == 0:
        del nodelist[level]

    local_labels = {key: str(self._clustering_coefficients[key])[:4]
                    for key in subgraph.node}

    edge_color = []
    edge_list = []
    edge_to = self.graph.neighbors(
        self._current_node)[self._current_edge_idx]
    for edge in subgraph.edges():
      edge_list.append(edge)
      if (edge_to in edge and self._current_node in edge):
        edge_color.append('r')
      elif (self._current_node in edge):
        edge_color.append('k')
      else:
        edge_color.append('b')

    fig = plt.figure(figsize=[4,4], dpi=200)
    layout = nx.shell_layout(subgraph, nodelist)
    nx.draw(subgraph,
            edgelist=edge_list,
            edge_color=edge_color,
            style='dashed',
            nodelist=nodelist_1d,
            node_color=node_color,
            width=0.5,
            with_labels=True,
            center=self._current_node,
            pos=layout,
            labels=local_labels,
            font_size=8)

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    size = canvas.get_width_height()
    raw_data = fig.canvas.tostring_rgb()
    plt.close(fig)
    surface = pygame.image.fromstring(raw_data, size, "RGB")
    self._screen.blit(surface, (0, 0))
    pygame.display.flip()

    # TODO: the event loop should be continuously pumped somewhere to avoid UI
    # freezing after every call to _render()
    pygame.event.get()
