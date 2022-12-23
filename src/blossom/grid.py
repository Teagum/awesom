"""
Grid for Self-organizing maps
"""
import numpy as np
from scipy.spatial import cKDTree

from . typealias import Array, Shape


class SomGrid:
    """Rectangular SOM grid
    """
    def __init__(self, shape: Shape) -> None:
        if not all(isinstance(val, int) and val >= 1 for val in shape):
            raise ValueError("Dimensions must be integer > 0.")
        self.shape = shape
        self.pos = np.asarray(list(np.ndindex(shape)), dtype=int)
        self.tree = cKDTree(self.pos)
        self.rows, self.cols = np.indices(shape)

    def nhb_idx(self, radius: float, points: Array | None = None) -> Array:
        """Compute the neighbourhood unit indices within ``radius``

        If ``points`` is given, return the neighbourhood around each unit in
        ``points``. Otherwise, return the neighbourhodd for each unit on the
        grid.

        Args:
            point:   Coordinate in a two-dimensional array.
            radius:  Lenght of radius.

        Returns:
            Array of indices of neighbours.
        """
        if points is None:
            points = self.pos
        return np.asarray(self.tree.query_ball_point(points, radius, np.inf))

    def nhb(self, radius: float, points: Array | None = None) -> Array:
        """Compute neighbourhood within ``radius``

        If ``points`` is given, return the neighbourhood around each unit in
        ``points``. Otherwise, return the neighbourhodd for each unit on the
        grid.

        Args:
            point:   Coordinate in a two-dimensional array.
            radius:  Lenght of radius.

        Returns:
            Array of positions of neighbours.
        """
        if points is None:
            points = self.pos
        idx = self.nhb_idx(radius, points)
        return self.pos[idx]

    def __iter__(self):
        for row, col in zip(self.rows.flat, self.cols.flat):
            yield row, col
