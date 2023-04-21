"""
Utilities for self.organizing maps.
"""

import itertools
from typing import cast, Iterable, Iterator

import numpy as np
from scipy.spatial import distance
from scipy import stats

from awesom.typing import IntArray, FloatArray, Metric


def grid_iter(n_rows: int, n_cols: int) -> Iterator[tuple[int, int]]:
    """Compute grid indices of an two-dimensional array.

    Args:
        n_rows:  Number of array rows.
        n_cols:  Number of array columns.

    Returns:
        Multi-index iterator.
    """
    return itertools.product(range(n_rows), range(n_cols))


def grid(n_rows: int, n_cols: int) -> IntArray:
    """Compute grid indices of a two-dimensional array.

    Args:
        n_rows:  Number of array rows.
        n_cols:  Number of array columns.

    Returns:
        Two-dimensional array in which each row represents an multi-index.
    """
    return np.array(list(grid_iter(n_rows, n_cols)), dtype=int)


def decrease_linear(start: float, step: float, stop: float = 1.0
                    ) -> Iterator[float]:
    """Linearily decrease ``start``  in ``step`` steps to ``stop``."""
    if step < 1 or not isinstance(step, int):
        raise ValueError("Param `step` must be int >= 1.")
    if step == 1:
        yield start
    else:
        coef = (stop - start) / (step - 1)
        for stp in range(step):
            yield coef * stp + start


def decrease_expo(start: float, step: float, stop: float = 1.0
                  ) -> Iterator[float]:
    """Exponentially decrease ``start``  in ``step`` steps to ``stop``."""
    if step < 1 or not isinstance(step, int):
        raise ValueError("Param `step` must be int >= 1.")
    if step == 1:
        yield start
    else:
        coef = np.log(stop / start) / (step - 1)
        for stp in range(step):
            yield start * np.exp(coef*stp)


def best_match(weights: FloatArray, inp: FloatArray, metric: Metric,
               out: FloatArray | None = None) -> tuple[IntArray, FloatArray]:
    """Compute the best matching unit of ``weights`` for each
    element in ``inp``.

    If several elemets in ``weights`` have the same distance to the
    current element of ``inp``, the first element of ``weights`` is
    choosen to be the best matching unit.

    Args:
        weights:    Two-dimensional array of weights, in which each row
                    represents an unit.
        inp:        Array of test vectors. If two-dimensional, rows are
                    assumed to represent observations.
        metric:     Distance metric to use.

    Returns:
        Index and error of best matching units.
    """
    if weights.ndim != 2:
        msg = (f"Array ``weights`` has {weights.ndim} dimensions, it "
               "has to have exactly two dimensions.")
        raise ValueError(msg)

    if weights.shape[-1] != inp.shape[-1]:
        msg = (f"Feature dimension of ``weights`` has {weights.shape[0]} "
               "elemets, whereas ``inp`` has {inp.shape[-1]} elemets. "
               "However, both dimensions have to match exactly.")
        raise ValueError(msg)

    inp = np.atleast_2d(inp)
    if inp.ndim > 2:
        msg = (f"Array ``inp`` has {weights.ndim} dimensions, it "
               "has to have one or two dimensions.")
        raise ValueError(msg)
    dists = distance.cdist(weights, inp, metric, out=out)
    return dists.argmin(axis=0), dists.min(axis=0)


def sample_st_matrix(n_mat: int, size: int) -> FloatArray:
    """Sample stochastic matrices from Dirichlet distribution

    The distribution is configured to place five times more probability mass on
    the main diagonal than on the remaining elements.

    Args:
        n_mat:  Number of matrices
        size:   Number of matirx row/cols

    Returns:
        Two-dimensional array. Each row corresponds to a flattened matrix.
    """
    if n_mat < 1:
        raise ValueError("n_mat < 1")

    if size < 2:
        raise ValueError("size < 2")

    pfact = 5.0
    alpha = np.ones((size, size), dtype=np.float64)
    np.fill_diagonal(alpha, pfact)

    samples = [stats.dirichlet(a).rvs(n_mat) for a in alpha]
    return np.hstack(samples, dtype=np.float64)


def sample_st_vector(n_vectors: int, size: int) -> FloatArray:
    """Sample stochastic vectors

    Sample random stochastic vectors with uniformly distributed probability
    mass. The sum of each vector equals 1.0 and each element is a number
    between 0.0 and 1.0.

    Args:
        n_vectors: Number of vectors
        size:      Vector size

    Returns:
        Two-dimensional array, whose rows correspond to vectors
    """
    if n_vectors < 1:
        raise ValueError("``n_vectors < 0")

    if size < 1:
        raise ValueError("``size`` < 1")

    alpha = np.ones(size, dtype=np.float64)
    samples = stats.dirichlet(alpha).rvs(n_vectors)
    return np.asanyarray(samples, dtype=np.float64)


def distribute(bmu_idx: Iterable[int], n_units: int
               ) -> dict[int, list[int]]:
    """List training data matches per SOM unit.

    This method assumes that the ith element of ``bmu_idx`` corresponds to the
    ith vetor in a array of input data vectors.

    Empty units result in empty list.

    Args:
        bmu_idx:  Indices of best matching units.
        n_units:  Number of units on the SOM.

    Returns:
        Dictionary in which the keys represent the flat indices of SOM units.
        The corresponding value is a list of indices of those training data
        vectors that have been mapped to this unit.
    """
    unit_matches: dict[int, list[int]] = {i: [] for i in range(n_units)}
    for data_idx, bmu in enumerate(bmu_idx):
        unit_matches[bmu].append(data_idx)
    return unit_matches


def pca(data: FloatArray, n_comps: int = 2
        ) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Perfom principal component analysis

    Interanlly, ``data`` will be centered but not scaled.

    Args:
        data:     Data set
        n_comps:  Number of principal components

    Returns:
        ``n_comps`` largest singular values,
        ``n_comps`` largest eigen vectors,
        transformed input data.
    """
    data_centered = (data - data.mean(axis=0))
    _, vals, vects = np.linalg.svd(data_centered)

    ord_idx = np.flip(vals.argsort())[:n_comps]
    vals = vals[ord_idx]
    vects = vects[ord_idx]
    return vals, vects, data_centered @ vects.T


def scale(arr: FloatArray, new_min: int = 0, new_max: int = 1, axis: int = -1
          ) -> FloatArray:
    """Scale ``arr`` between ``new_min`` and ``new_max``

    Args:
        arr:        Array to be scaled.
        new_min:    Lower bound.
        new_max:    Upper bound.

    Return:
        One-dimensional array of transformed values.
    """
    xmax = arr.max(axis=axis, keepdims=True)
    xmin = arr.min(axis=axis, keepdims=True)

    fact = (arr-xmin) / (xmax - xmin)
    out = fact * (new_max - new_min) + new_min

    return cast(FloatArray, out)
