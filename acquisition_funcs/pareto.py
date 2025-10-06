#!/usr/bin/env python
import numpy as np


def is_non_dominated(Y, maximize=True, deduplicate=True):
    n = Y.shape[0]
    if n == 0:
        return np.zeros(n, dtype=bool)

    if maximize:

        def dominates(a, b):
            return np.all(a >= b) and np.any(a > b)

    else:

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(Y[j], Y[i]):
                is_efficient[i] = False
                break

    if deduplicate:
        _, unique_indices = np.unique(Y, axis=0, return_index=True)
        is_efficient &= np.isin(np.arange(n), unique_indices)

    return is_efficient


def _is_non_dominated_loop(Y, maximize=True, deduplicate=True):
    n = Y.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        i_is_efficient = is_efficient[i]
        if i_is_efficient:
            vals = Y[i]
            if maximize:
                update = np.any(Y > vals, axis=1)
            else:
                update = np.any(Y < vals, axis=1)
            update[i] = i_is_efficient
            is_efficient &= update | ~i_is_efficient

    if deduplicate:
        _, unique_indices = np.unique(Y, axis=0, return_index=True)
        is_efficient &= np.isin(np.arange(n), unique_indices)

    return is_efficient


def pareto_front(Y, maximize=True, deduplicate=True):
    """
    Computes the non-dominated front for a given set of points.

    Args:
        Y: A (n x m) NumPy array of outcomes.
        maximize: If True, assume maximization (default is True).
        deduplicate: If True, only return unique points on the Pareto frontier (default is True).

    Returns:
        A boolean array indicating whether each point is non-dominated.
    """
    n, m = Y.shape
    el_size = Y.itemsize * 8  # size of one element in bits

    if n > 1000 or n**2 * m * el_size / 8 > 5e6:
        return _is_non_dominated_loop(Y, maximize=maximize, deduplicate=deduplicate)
    else:
        return is_non_dominated(Y, maximize=maximize, deduplicate=deduplicate)