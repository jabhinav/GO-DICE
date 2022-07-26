"""Utilities for caching and loading variables."""

from typing import Text

import numpy as np


def read_from_memmap(np_variable: np.ndarray, path: Text) -> np.ndarray:
  """Reads numpy variable from a memory-map.

  Args:
    np_variable: A numpy variable. This variable should be of the same size as
      the memory map. If not, incorrect variable will be loaded.
    path: path to the memory-map file.

  Returns:
    A numpy variable retrieved from the memory-map at path.
  """
  return np.memmap(path,
                   dtype=np_variable.dtype,
                   mode='r',
                   shape=np_variable.shape)


def save_to_memmap(np_variable: np.ndarray, path: Text):
  """Saves numpy variable to a memory-map.

  Args:
    np_variable: A numpy variable.
    path: path to the memory-map file.
  """
  fp = np.memmap(path,
                 dtype=np_variable.dtype,
                 mode='w+',
                 shape=np_variable.shape)
  fp[:] = np_variable[:]
  del fp
