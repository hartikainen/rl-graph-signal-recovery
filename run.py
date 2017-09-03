import argparse
from pprint import pprint
from distutils.util import strtobool
from datetime import datetime

import numpy as np

import algorithms
from utils import dump_results

def bool_type(x):
  return bool(strtobool(x))

DEFAULT_RECOVERY_METHOD = "SparseLabelPropagation"

def parse_args():
  parser = argparse.ArgumentParser(
    description=("An information-theoretic approach to"
                 " curiosity-driven reinforcement learning"))


  parser.add_argument("-v", "--verbose",
                      type=bool_type,
                      default=False,
                      help="Verbose")

  parser.add_argument("-g", "--graph-file",
                      type=str,
                      required=True,
                      help="Path to the graph file to be loaded")

  parser.add_argument("-r", "--recovery-method",
                      type=str,
                      default=DEFAULT_RECOVERY_METHOD,
                      help=("Path to the method used for the graph recovery."
                            "The recovery method must be importable from"
                            "algorithms module. Defaults to '{0}', i.e."
                            "algorithms.{0}."
                            "".format(DEFAULT_RECOVERY_METHOD)))

  parser.add_argument("--results-file", default=None, type=str,
                      help="File to write results to")

  args = vars(parser.parse_args())

  return args

def main(args):
  print(args)
  recovery_method_name = args["recovery_method"]
  graph_file = args["graph_file"]

  RecoveryMethodClass = getattr(algorithms, recovery_method_name)
  recovery_method = RecoveryMethodClass(graph_file)

  results = args.copy()

  run_results = recovery_method.run()

  results.update(run_results)

  results_file = args.get("results_file")
  if results_file is not None:
    dump_results(results, results_file)

if __name__ == "__main__":
  args = parse_args()
  main(args)
