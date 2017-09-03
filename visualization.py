"""Plot partitioned networkx graph.
Plotting logic taken from https://stackoverflow.com/a/43541777
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_partitioned_graph(g):
  partition_index = {
    node: idx
    for idx, partition in enumerate(g.graph["partition"])
    for node in partition
  }

  pos = community_layout(g, partition_index)
  nx.draw(g,
          pos,
          node_color=list(partition_index.values()),
          node_size=20,
          edge_color='#99ccff',
          width=0.5)
  plt.show()

def community_layout(g, partition):
  """Compute the layout for a modular graph.

  Args:
    g: networkx.Graph or networkx.DiGraph instance graph to plot
    partition: dict mapping int node -> int community graph partitions

  Returns:
    pos: dict mapping int node -> (float x, float y) node positions
  """

  pos_communities = _position_communities(g, partition, scale=3.)
  pos_nodes = _position_nodes(g, partition, scale=1.)

  # combine positions
  pos = dict()
  for node in g.nodes():
    pos[node] = pos_communities[node] + pos_nodes[node]

  return pos

def _position_communities(g, partition, **kwargs):
  # create a weighted graph, in which each node corresponds to a community,
  # and each edge weight to the number of edges between communities
  between_community_edges = _find_between_community_edges(g, partition)

  communities = set(partition.values())
  hypergraph = nx.DiGraph()
  hypergraph.add_nodes_from(communities)
  for (ci, cj), edges in between_community_edges.items():
    hypergraph.add_edge(ci, cj, weight=len(edges))

  # find layout for communities
  pos_communities = nx.spring_layout(hypergraph, **kwargs)

  # set node positions to position of community
  pos = dict()
  for node, community in partition.items():
    pos[node] = pos_communities[community]

  return pos

def _find_between_community_edges(g, partition):
  edges = defaultdict(list)

  for (ni, nj) in g.edges():
    ci = partition[ni]
    cj = partition[nj]
    if ci != cj:
      edges[(ci, cj)] += [(ni, nj)]

  return edges

def _position_nodes(g, partition, **kwargs):
  """Positions nodes within communities.
  """

  communities = defaultdict(list)
  for node, community in partition.items():
    communities[community] += [node]

  pos = dict()
  for ci, nodes in communities.items():
    subgraph = g.subgraph(nodes)
    pos_subgraph = nx.spring_layout(subgraph, **kwargs)
    pos.update(pos_subgraph)

  return pos
