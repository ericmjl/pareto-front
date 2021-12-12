"""Pareto-front calculation."""
import numpy as np
from multipledispatch import dispatch
import pandas as pd


def pareto_indices(data: np.ndarray) -> np.ndarray:
    """
    Return the Pareto efficient row subset of a columnar dataset.

    Inspired from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    :param data: A numpy array of shape (n_samples, n_dims).
    :returns: All samples which lie on the pareto front considering all dimensions.
    """
    pareto_front_indices = data.sum(axis=1).argsort()[::-1]
    data = data[pareto_front_indices]
    undominated = np.ones(data.shape[0], dtype=bool)
    for i in range(data.shape[0]):
        n = data.shape[0]
        if i >= n:
            break
        undominated[i + 1 : n] = (data[i + 1 :] > data[i]).any(1)
        data = data[undominated[:n]]
        pareto_front_indices = pareto_front_indices[undominated[:n]]
    return pareto_front_indices


@dispatch(np.ndarray)
def pareto_front(data: np.ndarray) -> np.ndarray:
    """Return pareto front of a NumPy array.

    :param data: n-dimensional NumPy array of shape (n_samples, n_dimensions).
        n_dimensions should be >= 2.
    :returns: The subset of samples that correspond to the pareto front.
    """
    idxs = pareto_indices(data)
    return data[idxs]


@dispatch(pd.DataFrame)
def pareto_front(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate pareto front points from a pandas DataFrame."""
    idxs = pareto_indices(data.values)
    return data.iloc[idxs]
