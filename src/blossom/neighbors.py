"""
Neighborhood computations
"""

import numpy as np

from scipy.spatial import distance

from . typealias import Array, Shape, Coord


def gaussian(grid, center, radius: float) -> Array:
    """Compute n-dimensional Gaussian neighbourhood.

    Gaussian neighborhood smoothes the array.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.
    """
    center = np.atleast_2d(center)
    dists = distance.cdist(center, grid, metric="sqeuclidean")
    return np.exp(-dists/(2*radius**2)).T


def mexican(grid, center, radius: float) -> Array:
    """Compute n-dimensional Mexcican hat neighbourhood.

    Mexican hat neighborhood smoothes the array.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.
    """
    center = np.atleast_2d(center)
    dists = distance.cdist(center, grid, metric="sqeuclidean")
    return ((1-(dists/radius**2)) * np.exp(-dists/(2*radius**2))).T


def star(grid, center, radius: float) -> Array:
    """Compute n-dimensional cityblock neighborhood.

    The cityblock neighborhood is a star-shaped area
    around ``center``.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.

    Returns:
    """
    center = np.atleast_2d(center)
    dists = distance.cdist(center, grid, "cityblock")
    return (dists <= radius).astype(int).T


def neighborhood(grid, metric: str = "sqeuclidean") -> Array:
    """Compute n-dimensional cityblock neighborhood.

    The cityblock neighborhood is a star-shaped area
    around ``center``.

    Params:
        grid:      Array of n-dimensional indices.
        metric:    Distance metric.

    Returns:
        Pairwise distances of map units.
    """
    return distance.squareform(distance.pdist(grid, metric))


def rect(grid, center, radius: float) -> Array:
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
    center = np.atleast_2d(center)
    dists = distance.cdist(center, grid, "chebychev")
    return (dists <= radius).astype(int).T


def check_bounds(shape: Shape, point: Coord) -> bool:
    """Return ``True`` if ``point`` is valid index in ``shape``.

    Args:
        shape:  Shape of two-dimensional array.
        point:  Two-dimensional coordinate.

    Return:
        True if ``point`` is within ``shape`` else ``False``.
    """
    return (0 <= point[0] < shape[0]) and (0 <= point[1] < shape[1])


def direct_rect_nb(shape: Shape, point: Coord) -> Array:
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
