from collections import deque
import time

import numpy as np
import tensorflow as tf
from baselines.ppo1 import mlp_policy
from baselines.common import (tf_util,
                              Dataset,
                              explained_variance,
                              fmt_row,
                              zipsame)
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI

from envs import GraphSamplingEnv
from utils import TIMESTAMP_FORMAT
from graph_functions import random_walk_error

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def policy_fn(name, ob_space, ac_space):
  return mlp_policy.MlpPolicy(name=name,
                              hid_size=100,
                              num_hid_layers=3,
                              ob_space=ob_space,
                              ac_space=ac_space)

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with
    GAE(lambda)
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    new = np.append(seg["new"], 0)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def traj_segment_generator(pi, env, horizon, stochastic, rw_sampling_args):
  """From baselines/ppo1/pposdg_simply.py"""
  t = 0
  ac = env.action_space.sample() # not used, just so we have the datatype
  new = True # marks if we're on first timestep of an episode
  ob = env.reset()

  cur_ep_ret = 0 # return in current episode
  cur_ep_len = 0 # len of current episode
  ep_rets = [] # returns of completed episodes in this segment
  ep_lens = [] # lengths of ...
  ep_errors = []
  ep_rw_errors = [] # random walk errors
  ep_error_diffs = [] # differences between rl and rw error

  # Initialize history arrays
  obs = np.array([ob for _ in range(horizon)])
  rews = np.zeros(horizon, 'float32')
  vpreds = np.zeros(horizon, 'float32')
  news = np.zeros(horizon, 'int32')
  acs = np.array([ac for _ in range(horizon)])
  prevacs = acs.copy()

  while True:
    prevac = ac
    ac, vpred = pi.act(stochastic, ob)
    # Slight weirdness here because we need value function at time T
    # before returning segment [0, T-1] so we get the correct
    # terminal value
    if t > 0 and t % horizon == 0:
      yield {"ob": obs,
             "rew": rews,
             "vpred": vpreds,
             "new": news,
             "ac": acs,
             "prevac": prevacs,
             "nextvpred": vpred * (1 - new),
             "ep_rets": ep_rets,
             "ep_lens" : ep_lens,
             "ep_errors": ep_errors,
             "ep_rw_errors": ep_rw_errors,
             "ep_error_diffs": ep_error_diffs}
      # Be careful!!! if you change the downstream algorithm to aggregate
      # several of these batches, then be sure to do a deepcopy
      ep_rets = []
      ep_lens = []
      ep_errors = []
      ep_rw_errors = []
      ep_error_diffs = []
    i = t % horizon
    obs[i] = ob
    vpreds[i] = vpred
    news[i] = new
    acs[i] = ac
    prevacs[i] = prevac

    ob, rew, new, _ = env.step(ac)
    rews[i] = rew

    cur_ep_ret += rew
    cur_ep_len += 1
    if new or cur_ep_len > 5000:
      ep_rets.append(cur_ep_ret)
      ep_lens.append(cur_ep_len)

      cur_ep_ret = 0
      cur_ep_len = 0

      # store graph error statistics
      ep_errors.append(env.error)
      rw_sampling_args.update({"graph": env.graph})
      rw_error = random_walk_error(rw_sampling_args)
      ep_rw_errors.append(rw_error)
      ep_error_diffs.append(rw_error - env.error)

      ob = env.reset()
    t += 1


class PPOAgent(object):
  def __init__(self,
               env,
               max_timesteps=10000000,
               max_episodes=0,
               max_iters=0,
               max_seconds=0,
               callback=None,
               timesteps_per_batch=256,
               clip_param=0.2,
               entcoeff=0.01,
               optim_epochs=4,
               optim_stepsize=1e-3,
               optim_batchsize=64,
               adam_epsilon=1e-5,
               gamma=0.99,
               lambda_=0.95,
               schedule='linear',
               logdir='results/ppo_agent',
               random_walk_sampling_args=None):
    self._env = env
    self._max_timesteps = max_timesteps
    self._max_episodes = max_episodes
    self._max_iters = max_iters
    self._max_seconds =  max_seconds
    self._callback = callback
    self._timesteps_per_batch = timesteps_per_batch
    self._clip_param = clip_param
    self._entcoeff = entcoeff
    self._optim_epochs = optim_epochs
    self._optim_stepsize = optim_stepsize
    self._optim_batchsize = optim_batchsize
    self._adam_epsilon = adam_epsilon
    self._gamma = gamma
    self._lambda = lambda_
    self._schedule = schedule
    self._random_walk_sampling_args = random_walk_sampling_args

    self._session = tf_util.single_threaded_session()

  def learn(self):
    """Adapted from baselines/ppo1/pposgd_simple.py"""
    env = self._env
    with self._session as sess:
      ob_space = env.observation_space
      ac_space = env.action_space
      # Construct network for new policy
      pi = policy_fn("pi", ob_space, ac_space)
      # Network for old policy
      oldpi = policy_fn("oldpi", ob_space, ac_space)
      # Target advantage function (if applicable)
      atarg = tf.placeholder(dtype=tf.float32, shape=[None])
      # Empirical return
      ret = tf.placeholder(dtype=tf.float32, shape=[None])
      # learning rate multiplier, updated with schedule
      lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
      # Annealed cliping parameter epislon
      clip_param = self._clip_param * lrmult

      ob = tf_util.get_placeholder_cached(name="ob")
      ac = pi.pdtype.sample_placeholder([None])

      kloldnew = oldpi.pd.kl(pi.pd)
      ent = pi.pd.entropy()
      meankl = tf_util.mean(kloldnew)
      meanent = tf_util.mean(ent)
      pol_entpen = (-self._entcoeff) * meanent

      # pnew / pold
      ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
      # surrogate from conservative policy iteration
      surr1 = ratio * atarg
      surr2 = tf_util.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
      # PPO's pessimistic surrogate (L^CLIP)
      pol_surr = - tf_util.mean(tf.minimum(surr1, surr2))
      vf_loss = tf_util.mean(tf.square(pi.vpred - ret))
      total_loss = pol_surr + pol_entpen + vf_loss
      losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
      loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

      var_list = pi.get_trainable_variables()
      lossandgrad = tf_util.function(
          [ob, ac, atarg, ret, lrmult],
          losses + [tf_util.flatgrad(total_loss, var_list)])
      adam = MpiAdam(var_list, epsilon=self._adam_epsilon)

      assign_old_eq_new = tf_util.function([], [],
          updates = [tf.assign(oldv, newv) for (oldv, newv)
                     in zipsame(oldpi.get_variables(), pi.get_variables())])
      compute_losses = tf_util.function([ob, ac, atarg, ret, lrmult], losses)

      tf_util.initialize()
      adam.sync()

      # Prepare for rollouts
      # ----------------------------------------
      seg_gen = traj_segment_generator(
          pi, env, self._timesteps_per_batch, stochastic=True,
          rw_sampling_args=self._random_walk_sampling_args)

      episodes_so_far = 0
      timesteps_so_far = 0
      iters_so_far = 0
      tstart = time.time()

      lenbuffer = deque(maxlen=100)
      rewbuffer = deque(maxlen=100)
      errorbuffer = deque(maxlen=100)
      rwerrorbuffer = deque(maxlen=100)
      errordiffbuffer = deque(maxlen=100)

      assert (sum([self._max_iters>0,
                  self._max_timesteps>0,
                  self._max_episodes>0,
                  self._max_seconds>0]) == 1,
                  "Only one time constraint permitted")
      while True:
        if self._callback: self_callback(locals(), globals())
        if self._max_timesteps and timesteps_so_far >= self._max_timesteps:
          break
        elif self._max_episodes and episodes_so_far >= self._max_episodes:
          break
        elif self._max_iters and iters_so_far >= self._max_iters:
          break
        elif self._max_seconds and time.time() - tstart >= self._max_seconds:
          break

        if self._schedule == 'constant':
          cur_lrmult = 1.0
        elif self._schedule == 'linear':
          cur_lrmult =  max(1.0 - float(timesteps_so_far) / self._max_timesteps, 0)
        else:
          raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, self._gamma, self._lambda)

        ob, ac, atarg, tdlamret = (
            seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"])
        # predicted value function before udpate
        vpredbefore = seg["vpred"]
        # standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / atarg.std()
        d = Dataset({'ob': ob,
                     'ac': ac,
                     'atarg': atarg,
                     'vtarg': tdlamret}, shuffle=not pi.recurrent)
        optim_batchsize = self._optim_batchsize or ob.shape[0]
        # update running mean/std for policy
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)

        # set old parameter values to new parameter values
        assign_old_eq_new()
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(self._optim_epochs):
          # list of tuples, each of which gives the loss for a minibatch
          losses = []
          for batch in d.iterate_once(optim_batchsize):
            *newlosses, g = lossandgrad(
              batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
              cur_lrmult)
            adam.update(g, self._optim_stepsize * cur_lrmult)
            losses.append(newlosses)
          logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
          newlosses = compute_losses(
            batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
            cur_lrmult)
          losses.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)

        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
          logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before",
                              explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        # handle graph errors
        errors = seg["ep_errors"]
        rw_errors = seg["ep_rw_errors"]
        error_diffs = seg["ep_error_diffs"]
        listoferrors = MPI.COMM_WORLD.allgather(errors)
        listofrwerrors = MPI.COMM_WORLD.allgather(rw_errors)
        listoferrordiffs = MPI.COMM_WORLD.allgather(error_diffs)
        errors = flatten_lists(listoferrors)
        rwerrors = flatten_lists(listofrwerrors)
        errordiffs = flatten_lists(listoferrordiffs)
        errorbuffer.extend(errors)
        rwerrorbuffer.extend(rwerrors)
        errordiffbuffer.extend(errordiffs)

        logger.record_tabular("EpErrorMean", np.mean(errorbuffer))
        logger.record_tabular("EpRwErrorMean", np.mean(rwerrorbuffer))
        logger.record_tabular("EpErrorDiffMean", np.mean(errordiffbuffer))
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
          logger.dump_tabular()
