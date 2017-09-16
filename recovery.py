import json
import argparse
from pprint import pprint
from distutils.util import strtobool
from datetime import datetime

import numpy as np

from algorithms import recovery
from utils import dump_results, load_graph, load_samples
from graph_functions import nmse

def bool_type(x):
  return bool(strtobool(x))

DEFAULT_RECOVERY_METHOD = "SparseLabelPropagation"

def parse_args():
  parser = argparse.ArgumentParser(
    description=("Graph Recovery: Script to run graph recovery based on graph"
                 " and given sample set of nodes"))


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
                            " algorithms.recovery module. Defaults to '{0}',"
                            " i.e. algorithms.recovery.{0}."
                            "".format(DEFAULT_RECOVERY_METHOD)))

  parser.add_argument("--recovery-params",
                      type=json.loads,
                      default={},
                      help=("Recovery algorithms parameters passed in json"
                            " format. These are passed directly to the"
                            " recovery method constructor"))

  parser.add_argument("--results-file", default=None, type=str,
                      help="File to write results to")

  args = vars(parser.parse_args())

  return args

def main(args):
  print(args)
  recovery_method_name = args["recovery_method"]
  recovery_params = args["recovery_params"]
  graph_file = args["graph_file"]
  sample_file = args["sample_file"]

  RecoveryMethodClass = getattr(recovery, recovery_method_name)
  graph = load_graph(graph_file)
  samples = load_samples(sample_file)
  recovery_method = RecoveryMethodClass(graph, samples, recovery_params)

  x_hat = recovery_method.run()

  results = args.copy()

  results.update({
    "x_hat": x_hat,
    "nmse": nmse(graph, x_hat)
  })

  results_file = args.get("results_file")

  if results_file is None:
    return results
  else:
    dump_results(results, results_file)

if __name__ == "__main__":
  args = parse_args()
  main(args)
