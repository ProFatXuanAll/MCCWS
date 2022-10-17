"""Randomness utilites."""

import random

import numpy as np
import torch
import torch.backends
import torch.cuda


def set_seed(seed: int) -> None:
  """Do best effort to ensure reproducibility on the same machine.
  Set random seed on :py:mod:`random` module, :py:mod:`numpy.random`, :py:func:`torch.manual_seed` and
  :py:mod:`torch.cuda`.
  Parameters
  ----------
  seed: int
    Controlled random seed which does best effort to make experiment reproducible.
    Must be bigger than ``0``.
  See Also
  --------
  numpy.random.seed
    Initialize the random number generator provided by Numpy.
  random.seed
    Initialize the random number generator provided by Python.
  torch.backends.cudnn.benchmark
    Use deterministic convolution algorithms.
  torch.backends.cudnn.deterministic
    Use deterministic convolution algorithms.
  torch.cuda.manual_seed_all
    Initialize the random number generator over all CUDA devices.
  torch.manual_seed
    Initialize the random number generator provided by PyTorch.
  Notes
  -----
  Reproducibility is not guaranteed accross different python/numpy/pytorch release, different os platforms or different
  hardwares (including CPUs and GPUs).
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  if torch.cuda.is_available():
    # Disable cuDNN benchmark for deterministic selection on algorithm.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
