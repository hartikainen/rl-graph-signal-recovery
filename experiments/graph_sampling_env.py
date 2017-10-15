import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import argparse
from baselines import logger
from datetime import datetime

import generate_appm
from agents import BaseAgent
from envs import GraphSamplingEnv, SimpleThreeClusterEnv
from visualization import plot_agent_history
from utils import (
  load_pickle, load_baselines_tabular,
  TIMESTAMP_FORMAT, dump_pickle
)

SAMPLING_ARGS = {
  "sampling_params": {"L": 10, "M": 3},
  "sampling_method": "RandomWalkSampling"
}

LOGDIR = "./results/graph_sampling_env/"

def parse_args():
  parser = argparse.ArgumentParser("GraphSamplingEnv testbed")
  parser.add_argument("--step",
                      type=str,
                      default="run",
                      choices=["run", "visualize"],
                      help="Step to run")
  parser.add_argument("--replay_buffer_size",
                      type=int,
                      default=500000,
                      help="Size of the deepq replay buffer.")
  parser.add_argument("--exploration_schedule_steps",
                      type=int,
                      default=1000000,
                      help="Linearly decreases exploration over given steps.")
  parser.add_argument("--exploration_initial_prob",
                      type=float,
                      default=1.0,
                      help="Initial probability to take explorative actions.")
  parser.add_argument("--exploration_final_prob",
                      type=float,
                      default=0.05,
                      help="Final probability to take explorative actions.")
  parser.add_argument("--gamma",
                      type=float,
                      default=0.99,
                      help="Q-learning discount factor.")
  parser.add_argument("--learning_rate",
                      type=float,
                      default=5e-4,
                      help="Adam optimizer learning rate.")

  args, unknown = parser.parse_known_args()
  return vars(args)

def run(args):
  M = 3
  env = GraphSamplingEnv(max_samples=M)

  agent = BaseAgent(env=env,
                    gamma=args["gamma"],
                    learning_rate=args["learning_rate"],
                    replay_buffer_size=args["replay_buffer_size"],
                    exploration_schedule_steps=args["exploration_schedule_steps"],
                    exploration_initial_prob=args["exploration_initial_prob"],
                    exploration_final_prob=args["exploration_final_prob"],
                    random_walk_sampling_args=SAMPLING_ARGS)
  now = datetime.now()
  log_formats = ['stdout']
  logger.configure(dir=LOGDIR + f"{now.strftime(TIMESTAMP_FORMAT)}")
  agent.learn()
  agent.test()

def visualize(args):
  now = datetime.now()
  today_str = now.strftime(TIMESTAMP_FORMAT.split("-")[0])
  runs_today = glob.glob(LOGDIR + "/*")
  latest_run = sorted(runs_today)[-1]
  filepath = f"{latest_run}/progress.json"

  print(f"visualizing {filepath}")
  data = load_baselines_tabular(filepath)
  plot_agent_history(data)

def main(args):
  if args["step"] == "run":
    run(args)
  elif args["step"] == "visualize":
    visualize(args)

if __name__ == "__main__":
  args = parse_args()
  main(args)
