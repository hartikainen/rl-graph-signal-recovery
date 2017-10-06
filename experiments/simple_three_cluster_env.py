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



def parse_args():
  parser = argparse.ArgumentParser("SimpleThreeClusterEnv testbed")
  parser.add_argument('--step',
                      type=str,
                      default='run',
                      choices=['run', 'visualize'],
                      help="Step to run")

  args, unknown = parser.parse_known_args()
  return vars(args)

def run(args):
  M = 3
  env = SimpleThreeClusterEnv(max_samples=M)

  num_train_graphs = 10000

  agent = BaseAgent(env=env)
  now = datetime.now()
  results_path = f"./results/fixed_env/{now.strftime(TIMESTAMP_FORMAT)}"
  with logger.session(results_path):
    agent.learn(num_train_graphs)
  agent.test()

def visualize(args):
  now = datetime.now()
  today_str = now.strftime(TIMESTAMP_FORMAT.split("-")[0])
  runs_today = glob.glob(f"./results/fixed_env/{today_str}-*")
  latest_run = sorted(runs_today)[-1]
  filepath = f"{latest_run}/progress.json"

  print(f"visualizing {filepath}")
  data = load_baselines_tabular(filepath)
  plot_agent_history(data)

def main(args):
  if args['step'] == "run":
    run(args)
  elif args['step'] == "visualize":
    visualize(args)

if __name__ == "__main__":
  args = parse_args()
  main(args)
