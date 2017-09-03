import argparse
from pprint import pprint
from distutils.util import strtobool
from datetime import datetime

import numpy as np

from algorithms import recovery
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

  parser.add_argument("--graph-file",
                      type=str,
                      required=True,
                      help="Path to the graph file to be loaded for recovery")

  parser.add_argument("--sample-file",
                      type=str,
                      required=True,
                      help="Path to the sample file to be loaded for recovery")

  parser.add_argument("--recovery-method",
                      type=str,
                      default=DEFAULT_RECOVERY_METHOD,
                      help=("Name of the class used for graph recovery. The"
                            " recovery class must be importable from"
                            "algorithms.recovery module. Defaults to '{0}',"
                            " i.e. algorithms.recovery.{0}."
                            "".format(DEFAULT_RECOVERY_METHOD)))

  parser.add_argument("--results-file", default=None, type=str,
                      help="File to write results to")

  args = vars(parser.parse_args())

  return args

def main(args):
  print(args)
  recovery_method_name = args["recovery_method"]
  graph_file = args["graph_file"]
  sample_file = args["sample_file"]

  RecoveryMethodClass = getattr(recovery, recovery_method_name)
  recovery_method = RecoveryMethodClass(graph_file, sample_file)

  results = args.copy()

  run_results = recovery_method.run()

  results.update(run_results)

  results_file = args.get("results_file")
  if results_file is not None:
    dump_results(results, results_file)

if __name__ == "__main__":
  args = parse_args()
  main(args)
