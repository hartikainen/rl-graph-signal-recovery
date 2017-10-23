import itertools
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from gym.spaces import Box, MultiBinary, Discrete
from baselines import deepq, logger
from baselines.common import tf_util
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from algorithms.recovery import sparse_label_propagation
from graph_functions import nmse, random_walk_error
from utils import TIMESTAMP_FORMAT, dump_pickle
from visualization import draw_partitioned_graph
from envs import GraphSamplingEnv, SimpleThreeClusterEnv

def model_fn(inpt, num_actions, scope, reuse=False):
  """This model takes as input an observation and returns values of all
  actions.
  """
  with tf.variable_scope(scope, reuse=reuse):
    out = inpt
    out = layers.fully_connected(out,
                                 num_outputs=50,
                                 activation_fn=tf.nn.relu)
    out = layers.fully_connected(out,
                                 num_outputs=num_actions,
                                 activation_fn=None)
    return out

class BaseAgent(object):
  def __init__(self,
               env,
               gamma=0.99,
               learning_rate=5e-4,
               replay_buffer_size=500000,
               exploration_schedule_steps=1000000,
               exploration_initial_prob=1.0,
               exploration_final_prob=0.02,
               random_walk_sampling_args=None):
    self.env = env
    self._gamma = gamma
    self._learning_rate = learning_rate
    self._replay_buffer_size = replay_buffer_size
    self._exploration_schedule_steps = exploration_schedule_steps
    self._exploration_final_prob = exploration_final_prob
    self._exploration_initial_prob = exploration_initial_prob
    self._random_walk_sampling_args = random_walk_sampling_args
    self._build_train()
    self.session = tf_util.make_session(1)

  def _observation_ph_generator(self, name):
    env = self.env

    if isinstance(env.observation_space, (MultiBinary, Discrete)):
      batch_shape = (env.observation_space.n,)
    elif isinstance(env.observation_space, Box):
      batch_shape = env.observation_space.shape
    else:
      raise ValueError("Unexpected observation space")

    return tf_util.BatchInput(batch_shape, name=name)

  def _build_train(self):
    env = self.env

    act, train, update_target, debug = deepq.build_train(
      make_obs_ph=self._observation_ph_generator,
      q_func=model_fn,
      num_actions=env.action_space.n,
      optimizer=tf.train.AdamOptimizer(learning_rate=self._learning_rate),
      gamma=self._gamma
    )

    self.act = act
    self.train = train
    self.update_target = update_target
    self.debug = debug

  def learn(self):
    act = self.act
    train = self.train
    update_target = self.update_target

    results = []

    env = self.env
    with self.session.as_default():
      replay_buffer = ReplayBuffer(self._replay_buffer_size)
      exploration = LinearSchedule(
          schedule_timesteps=self._exploration_schedule_steps,
          initial_p=self._exploration_initial_prob,
          final_p=self._exploration_final_prob)

      tf_util.initialize()
      update_target()

      episode_rewards = [0.0]
      episode_errors = []
      episode_rw_errors = []
      episode_error_diffs = []
      observation = env.reset()
      prev_steps = 0

      for t in itertools.count():
        # Take action and update exploration to the newest value
        action = act(observation[None], update_eps=exploration.value(t))[0]
        new_observation, reward, done, _ = env.step(action)
        # Store transition in the replay buffer.
        replay_buffer.add(observation, action, reward,
                          new_observation, float(done))
        observation = new_observation

        episode_rewards[-1] += reward

        if done:
          episode_errors.append(env.error)
          episode_rewards.append(0)
          if self._random_walk_sampling_args is not None:
            sampling_args = self._random_walk_sampling_args
            sampling_args.update({"graph": env.graph})
            rw_error = random_walk_error(sampling_args)
            episode_rw_errors.append(rw_error)
            episode_error_diffs.append(rw_error - env.error)

          nmse = env.get_current_nmse()
          if len(episode_rewards) % 10 == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", len(episode_rewards))
            logger.record_tabular("mean episode reward",
                                  round(np.mean(episode_rewards[-101:-1]), 3))
            logger.record_tabular("mean episode error",
                                  round(np.mean(episode_errors[-101:-1]), 3))
            logger.record_tabular("nmse", nmse)
            logger.record_tabular("sampling set",
                [int(v) for v in env.sampling_set])
            logger.record_tabular("% time spent exploring",
                                  int(100 * exploration.value(t)))
            if self._random_walk_sampling_args is not None:
              logger.record_tabular("mean random walk error", round(np.mean(
                episode_rw_errors[-101:-1]), 3))
              logger.record_tabular("mean error diff",
                  round(np.mean(episode_error_diffs[-101:-1]), 3))
            logger.dump_tabular()

          # TODO: Are steps and results used for something?
          steps = t - prev_steps
          results.append({
            "steps": steps,
            "reward": episode_rewards[-1],
            "nmse": nmse,
            "exploration_prop": int(100 * exploration.value(t)),
          })
          prev_steps = steps
          observation = env.reset()

        # Minimize the Bellman equation error on replay buffer sample batch
        if t > 1000:
          (observations_t, actions, rewards,
           observations_tp1, dones) = replay_buffer.sample(32)
          train(observations_t, actions, rewards,
                observations_tp1, dones, np.ones_like(rewards))
        if t % 1000 == 0:
          # Update target network periodically.
          update_target()

  def test(self):
    env = self.env
    act = self.act
    train = self.train
    update_target = self.update_target

    with self.session.as_default():
      observation, done = env.reset(), False
      while not done:
        action = act(observation[None], update_eps=0.9)[0]
        observation, reward, done, _ = env.step(action)

    nmse = env.get_current_nmse()
    print("nmse: ", nmse)
