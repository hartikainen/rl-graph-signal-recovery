import numpy as np
import tensorflow as tf
from baselines import deepq
from baselines.common import tf_util
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from algorithms.recovery import sparse_label_propagation
from envs import GraphSamplingEnv

from graph_functions import nmse

def train_test_agent():
  M = 10
  num_train_graphs = 10

  env = GraphSamplingEnv(max_samples=M)
  with tf_util.make_session(1):
    def observation_ph_generator(name):
      return tf_util.BatchInput(env.observation_space.shape, name=name)

    act, train, update_target, debug = deepq.build_train(
      make_obs_ph=observation_ph_generator,
      q_func=deepq.models.mlp([50,50,50]),
      num_actions=env.action_space.n,
      optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
    )

    # Create the replay buffer
    replay_buffer = ReplayBuffer(10)
    # Create the schedule for exploration starting from 1 (every action is random) down to
    # 0.02 (98% of actions are selected according to values predicted by the model).
    exploration = LinearSchedule(schedule_timesteps=10000,
                                 initial_p=1.0,
                                 final_p=0.02)

    tf_util.initialize()
    update_target()

    episode_rewards = [0.0]
    observation = env.reset()

    for t in range(num_train_graphs):
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
          observation = env.reset()
          episode_rewards.append(0)

        is_solved = False
        if is_solved:
          # Show off the result
          env.render()
        else:
          # Minimize the Bellman equation error on replay buffer sample batch
          if t > 1000:
            (observations_t, actions, rewards,
             observations_tp1, dones) = replay_buffer.sample(32)
            train(observations_t, actions, rewards,
                  observations_tp1, dones, np.ones_like(rewards))
          if t % 1000 == 0:
            # Update target network periodically.
            update_target()

        if done and len(episode_rewards) % 10 == 0:
          print("steps", t)
          print("episodes", len(episode_rewards))
          print("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
          print("% time spent exploring", int(100 * exploration.value(t)))


    observation, done = env.reset(), False
    while not done:
      action = act(observation[None], update_eps=exploration.value(t))[0]
      observation, reward, done, _ = env.step(action)

    graph = env.graph
    x = [graph.node[i]['value'] for i in graph.nodes_iter()]
    sampling_set = env.sampling_set
    x_hat = sparse_label_propagation(graph, list(sampling_set))

    print("nmse: {}".format(nmse(x, x_hat)))

if __name__ == "__main__":
  train_test_agent()
