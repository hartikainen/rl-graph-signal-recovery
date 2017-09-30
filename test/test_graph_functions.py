from nose.tools import assert_equal, assert_true
import numpy as np

from utils import load_graph
from algorithms.recovery import sparse_label_propagation
from algorithms.sampling import RandomWalkSampling
import generate_appm
from graph_functions import (
  normalized_mean_squared_error,
  total_variance,
  slp_maximum_error
)

def verify_nmse(test_case):
  x, x_hat, expected_nmse = test_case
  nmse = normalized_mean_squared_error(x, x_hat)
  assert_equal(nmse, expected_nmse)

def test_nmse():
  # TODO: add some real data test cases, preferably in fixture file form
  TEST_CASES = [
    ([1,2,3,4,5], [2,3,4,5,6], 0.090909090909090925),
    ([0,0], [0,0], 0),
    ([0,0], [1,1], 0),
    ([], [], 0),
  ]

  for test_case in TEST_CASES:
    yield verify_nmse, test_case

def verify_tv(test_case):
  edges, signal, expected_tv = test_case
  tv = total_variance(edges, signal)
  assert_equal(tv, expected_tv)

def test_total_variance():
  # TODO: add some real data test cases, preferably in fixture file form
  TEST_CASES = [
    ([(0,1), (0,2)], [0.1,0.1,0.1], 0),
    ([(0,1), (0,2)], [0.2,0.1,0.1], 0.2),
    ([(0,1), (0,2)], [0.1,0.1,0.2], 0.1),
    ([(0,1), (0,2)], [0.0,0.0,0.0], 0),
  ]

  for test_case in TEST_CASES:
    yield verify_tv, test_case

def test_slp_maximum_error():
  TEST_CASES = [
    ([1,0], 2),
    ([0,0,0,5], 4),
    ([-1,1], 4)
  ]

  for test_case in TEST_CASES:
    x, error = test_case
    assert_true(np.allclose(slp_maximum_error(x), error))
