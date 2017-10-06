import numpy as np

def total_variation(edges, signal):
  tv = np.sum(np.absolute([
    signal[i] - signal[j] for i,j in edges
  ]))

  return tv

def normalized_mean_squared_error(x, x_hat):
  if len(x) == 0: return 0
  if len(x) != len(x_hat):
    raise ValueError("Expected equal shapes, got: {}, {}"
                     "".format(len(x), len(x_hat)))
  x = np.array(x)
  x_hat = np.squeeze(x_hat)

  x_diff = x_hat - x
  x_diff_norm = np.linalg.norm(x_diff, 2)
  x_norm = np.linalg.norm(x, 2)
  if x_norm == 0: return float('inf')
  error = np.power(x_diff_norm, 2.0) / np.power(x_norm, 2.0)

  return error

def slp_maximum_error(x):
  if len(x) == 0: return 0
  maximally_different_x = x.copy()
  for index, element in enumerate(x):
    element_vector = np.ones_like(x) * element
    maximally_different_x[index] = x[np.argmax(np.abs(x - element_vector))]
  error = normalized_mean_squared_error(x, maximally_different_x)
  return error

nmse = normalized_mean_squared_error
