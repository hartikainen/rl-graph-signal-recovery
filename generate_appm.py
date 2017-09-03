import argparse
import matplotlib.pyplot as plt
import networkx as nx
from utils import dump_graph

def parse_args():
  parser = argparse.ArgumentParser(
      description="Creates an Assortative Planted Partition Model")

  parser.add_argument("--sizes",
                      type=int,
                      default=[4,4,4],
                      nargs='*',
                      help="Sizes of the groups to generate")

  parser.add_argument("--p_in",
                      type=float,
                      default=0.5,
                      help="Probability of connecting vertices within a group")

  parser.add_argument("--p_out",
                      type=float,
                      default=0.1,
                      help="Probability of connecting vertices between groups")

  parser.add_argument("--seed",
                      type=int,
                      default=None,
                      help="Random seed")

  parser.add_argument("--visualize",
                      action="store_true",
                      default=False,
                      help="Plot the generated graph")

  parser.add_argument("--out_path",
                      default=None,
                      help="Save graph output to out_path")

  args = vars(parser.parse_args())
  return args

def main():
  args = parse_args()
  sizes, p_in, p_out, seed = (args["sizes"], args["p_in"], args["p_out"],
                              args["seed"])
  visualize, out_path = args["visualize"], args["out_path"]

  appm = nx.random_partition_graph(sizes, p_in, p_out, seed=seed)

  if visualize:
    nx.draw(appm)
    plt.show()

  out_path = out_path if out_path is not None else "out.json"
  dump_graph(appm, out_path)

if __name__ == "__main__":
  main()
