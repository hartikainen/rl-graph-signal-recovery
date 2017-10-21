import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import argparse
from baselines import logger
from datetime import datetime

import generate_appm
from agents import PPOAgent
from envs import SimpleActionsGraphSamplingEnv
from visualization import plot_agent_history
from utils import (
  load_pickle, load_baselines_tabular,
  TIMESTAMP_FORMAT, dump_pickle
)

SAMPLING_ARGS = {
  "sampling_params": {"L": 10, "M": 3},
  "sampling_method": "RandomWalkSampling"
}

LOGDIR = "./results/ppo_graph_sampling_env/"

def parse_args():
  parser = argparse.ArgumentParser("GraphSamplingEnv testbed")
  parser.add_argument("--step",
                      type=str,
                      default="run",
                      choices=["run", "visualize"],
                      help="Step to run")
  parser.add_argument("--gamma",
                      type=float,
                      default=0.99,
                      help="Q-learning discount factor.")

  args, unknown = parser.parse_known_args()
  return vars(args)

def run(args):
  M = 3
  env = SimpleActionsGraphSamplingEnv(max_samples=M)

  agent = PPOAgent(env=env,
                   random_walk_sampling_args=SAMPLING_ARGS)
  now = datetime.now()
  logger.configure(dir=LOGDIR + f"{now.strftime(TIMESTAMP_FORMAT)}")
  agent.learn()

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
