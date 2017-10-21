from envs import GraphSamplingEnv
from gym.spaces import Discrete

class SimpleActionsGraphSamplingEnv(GraphSamplingEnv):
  def __init__(self, max_samples=3):
    super().__init__(max_samples)
    self.action_space = Discrete(self.num_nodes)

  def _reward(self):
    x_hat = sparse_label_propagation(self.graph, list(self.sampling_set))
    x = [self.graph.node[idx]['value']
         for idx in range(self.graph.number_of_nodes())]
    error = nmse(x, x_hat)
    self.error = error

    reward = -error

    return reward

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
