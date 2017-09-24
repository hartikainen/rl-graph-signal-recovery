import networkx as nx
import numpy as np
from algorithms.recovery.sparse_label_propagation import (
    sparse_label_propagation2)
import matplotlib.pyplot as plt
from pylab import stem
from pylab import setp
from pylab import show

def test_recovery_matches_original():
  N = 100
  signal_template = [1,1,1,1,1,5,5,5,5,5]
  sampling_set = [1 + i * 5 for i in range(N // 5)]
  weights = [2 if i % 5 != 0 else 1 for i in range(1, N)]

  graph = nx.Graph()
  for index in range(N-1):
    weight = weights[index]
    graph.add_edge(index, index + 1, weight=weight)
    signal = signal_template[index % len(signal_template)]
    graph.node[index]['value'] = signal
  graph.node[N-1]['value'] = 2

  slp_hatx = sparse_label_propagation2(graph, sampling_set)

  # TODO: Make this a real testcase
  markerline, stemlines, baseline = stem(
      np.arange(0,100), slp_hatx[0:100], '-')
  setp(markerline, 'markerfacecolor', 'b')
  setp(baseline, 'color','b', 'linewidth', 2)
  setp(stemlines, 'color', 'b')
  show()

if __name__ == '__main__':
  test_recovery_matches_original()
