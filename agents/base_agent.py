import numpy as np
import tensorflow as tf
from gym.spaces import Box, MultiBinary, Discrete
from baselines import deepq
from baselines.common import tf_util
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer
from datetime import datetime

from algorithms.recovery import sparse_label_propagation
from graph_functions import nmse
from utils import TIMESTAMP_FORMAT, dump_pickle

class BaseAgent(object):
  def __init__(self, env):
    self.env = env
    self._build_train()
    self.session = tf_util.make_session(1)

  def _build_train(self):
    env = self.env

    def observation_ph_generator(name):
      if isinstance(env.observation_space, (MultiBinary, Discrete)):
        batch_shape = (env.observation_space.n,)
      elif isinstance(env.observation_space, Box):
        batch_shape = env.observation_space.shape
      else:
        raise ValueError("Unexpected observation space")
      return tf_util.BatchInput(batch_shape, name=name)

    act, train, update_target, debug = deepq.build_train(
      make_obs_ph=observation_ph_generator,
      q_func=deepq.models.mlp([100]),
      num_actions=env.action_space.n,
      optimizer=tf.train.AdamOptimizer(learning_rate=1),
    )
    self.act = act
    self.train = train
    self.update_target = update_target
    self.debug = debug

  def learn(self, num_train_graphs=100):
    env = self.env

    act = self.act
    train = self.train
    update_target = self.update_target

    results = []

    with self.session.as_default():
      # Create the replay buffer
      replay_buffer = ReplayBuffer(32)
      # Create the schedule for exploration starting from 1 (every action is random) down to
      # 0.02 (98% of actions are selected according to values predicted by the model).
      exploration = LinearSchedule(schedule_timesteps=1000,
                                   initial_p=1.0,
                                   final_p=0.02)

      tf_util.initialize()
      update_target()

      episode_rewards = []
      observation = env.reset()
      prev_steps = 0

      for t in range(num_train_graphs):
        episode_rewards.append(0.0)
        done = False
        while not done:
          # Take action and update exploration to the newest value
          action = act(observation[None], update_eps=exploration.value(t))[0]
          new_observation, reward, done, _ = env.step(action)
          # Store transition in the replay buffer.
          replay_buffer.add(observation, action, reward,
                            new_observation, float(done))
          observation = new_observation

          episode_rewards[-1] += reward

          if done:
            nmse = env.get_current_nmse()
            if len(episode_rewards) % 10 == 0:
              print("steps", t)
              print("episodes", len(episode_rewards))
              print("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
              print("nmse: ", nmse)
              print("% time spent exploring", int(100 * exploration.value(t)))

            observation = env.reset()

            steps = t - prev_steps
            results.append({
              "steps": steps,
              "reward": episode_rewards[-1],
              "nmse": nmse,
              "exploration_prop": int(100 * exploration.value(t)),
            })
            prev_steps = steps

          is_solved = False
          if is_solved:
            # Show off the result
            env.render()
          else:
            # Minimize the Bellman equation error on replay buffer sample batch
            if t > 10:
              (observations_t, actions, rewards,
               observations_tp1, dones) = replay_buffer.sample(32)
              train(observations_t, actions, rewards,
                    observations_tp1, dones, np.ones_like(rewards))
            if t % 10 == 0:
              # Update target network periodically.
              update_target()

    now = datetime.now()
    results_path = f"./results/fixed_env/{now.strftime(TIMESTAMP_FORMAT)}.pk"
    dump_pickle(results, results_path)

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
