import numpy as np

def total_variance(graph, x_hat):
  tv = np.sum([
    x_hat[i] - x_hat[j]
    for i,j in graph.edges_iter()
    if (i in x_hat) and (j in x_hat)
  ])

  return tv

def normalized_mean_squared_error(graph, x_hat):
  full_signal = np.array([graph.node[node]['value']
                          for node in graph.nodes_iter()])

  x_diff = np.squeeze(x_hat) - full_signal
  x_diff_norm = np.linalg.norm(x_diff, 2)
  x_norm = np.linalg.norm(full_signal, 2)
  error = np.power(x_diff_norm, 2.0) / np.power(x_norm, 2.0)

  return error
