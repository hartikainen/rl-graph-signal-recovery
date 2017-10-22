import numpy as np
from gym.spaces import Discrete, Box

from envs import GraphSamplingEnv
from algorithms.recovery import sparse_label_propagation
from graph_functions import (
  total_variation,
  nmse,
)

class SimpleActionsGraphSamplingEnv(GraphSamplingEnv):
  def __init__(self, max_samples=3, graph_args=None):
    super().__init__(max_samples, graph_args=graph_args)
    self.action_space = Discrete(self.num_nodes)
    observation_length = sum(range(self.num_nodes + 1)) + 3 * self.num_nodes
    self.observation_space = Box(0, 1, observation_length)

  def _reward(self):
    x_hat = sparse_label_propagation(self.graph, list(self.sampling_set))
    x = [self.graph.node[idx]['value']
         for idx in range(self.graph.number_of_nodes())]
    error = nmse(x, x_hat)
    self.error = error

    reward = -error

    return reward

  def _get_observation(self):
    state_descriptor = np.zeros((self.num_nodes, 1))
    # sampling set indicator
    for node in self.sampling_set:
      state_descriptor[node, 0] = 1.

    adjacency_vector = np.squeeze(self._adjacency_matrix[np.triu_indices_from(
      self._adjacency_matrix)])
    observation = np.reshape(
        np.concatenate((self._degree_vector,
                        self._clustering_coefficient_vector,
                        state_descriptor), 1), (-1))
    observation = np.concatenate((adjacency_vector, observation))
    return observation

  def _step(self, action):
    self._validate_action(action)
    self.sampling_set.add(action)
    observation = self._get_observation()

    num_samples = len(self.sampling_set)
    reward = 0.0

    done = (num_samples >= self._max_samples
            or num_samples >= self.graph.number_of_nodes())

    if done:
      reward = self._reward()

    return observation, reward, done, {}

  def _render(self, mode='human', close='False'):
    return
