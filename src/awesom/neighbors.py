"""
Neighborhood computations
"""
from typing import cast
import numpy as np
import numpy.typing as npt

from scipy.spatial import distance

from awesom.typing import IntArray, FloatArray, Shape, Coord


def gaussian(grid: IntArray, center: npt.ArrayLike, radius: float,
             out: FloatArray | None = None) -> FloatArray:
    """Compute n-dimensional Gaussian neighbourhood.

    Gaussian neighborhood smoothes the array.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.
    """
    if radius <= 0:
        raise ValueError("Radius <= 0")

    if out is None:
        out = np.empty((grid.shape[0], 1), dtype=np.float64)

    distance.cdist(grid, center, metric="sqeuclidean", out=out)
    np.multiply(out, -1, out=out)
    np.divide(out, 2*radius**2, out)
    np.exp(out, out=out)
    return out


def mexican(grid: IntArray, center: npt.ArrayLike, radius: float
            ) -> FloatArray:
    """Compute n-dimensional Mexcican hat neighbourhood.

    Mexican hat neighborhood smoothes the array.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.
    """
    if radius <= 0:
        raise ValueError("Radius <= 0")

    rsq = radius**2
    dists = np.empty((grid.shape[0], 1), dtype=np.float64)
    gauss = np.empty((grid.shape[0], 1), dtype=np.float64)
    norm = np.empty((grid.shape[0], 1), dtype=np.float64)

    distance.cdist(grid, center, metric="sqeuclidean", out=dists)

    np.divide(-dists, rsq, out=norm)
    np.divide(norm, 2, out=gauss)
    np.add(1, norm, out=norm)
    np.exp(gauss, out=dists)
    np.multiply(norm, dists, out=dists)
    return dists


def star(grid: IntArray, center: npt.ArrayLike, radius: float) -> FloatArray:
    """Compute n-dimensional cityblock neighborhood.

    The cityblock neighborhood is a star-shaped area
    around ``center``.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.

    Returns:
    """
    dists = np.empty((grid.shape[0], 1), dtype=np.float64)
    distance.cdist(grid, center, metric="cityblock", out=dists)
    return np.less_equal(dists, radius, out=dists)


def neighborhood(grid: IntArray, metric: str = "sqeuclidean") -> FloatArray:
    """Compute n-dimensional cityblock neighborhood.

    The cityblock neighborhood is a star-shaped area
    around ``center``.

    Params:
        grid:      Array of n-dimensional indices.
        metric:    Distance metric.

    Returns:
        Pairwise distances of map units.
    """
    dists = distance.pdist(grid, metric)
    return cast(npt.NDArray[np.float64], distance.squareform(dists))


def rect(grid: IntArray, center: npt.ArrayLike, radius: float) -> FloatArray:
    """Compute n-dimensional Chebychev neighborhood.

    The Chebychev neighborhood is a square-shaped area
    around ``center``.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.

    Returns:
        Two-dimensional array of in
    """
    dists = np.empty((grid.shape[0], 1), dtype=np.float64)
    distance.cdist(grid, center, metric="chebychev", out=dists)
    return np.less_equal(dists, radius)


def check_bounds(shape: Shape, point: Coord) -> bool:
    """Return ``True`` if ``point`` is valid index in ``shape``.

    Args:
        shape:  Shape of two-dimensional array.
        point:  Two-dimensional coordinate.

    Return:
        True if ``point`` is within ``shape`` else ``False``.
    """
    return (0 <= point[0] < shape[0]) and (0 <= point[1] < shape[1])


def direct_rect_nb(shape: Shape, point: Coord) -> IntArray:
    """Return the set of direct neighbours of ``point`` given rectangular
    topology.

    Args:
        shape:  Shape of two-dimensional array.
        point:  Two-dimensional coordinate.

    Returns:
        Advanced index of points in neighbourhood set.
    """
    nhb = []
    for i in range(point[0]-1, point[0]+2):
        for j in range(point[1]-1, point[1]+2):
            if check_bounds(shape, (i, j)):
                nhb.append((i, j))
    return np.asarray(nhb)
