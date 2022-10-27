import numpy as np
from scipy.spatial import cKDTree

from . typealias import Array, Coord, Shape


class SomGrid:
    def __init__(self, shape: Shape) -> None:
        if not all(isinstance(val, int) and val >= 1 for val in shape):
            raise ValueError("Dimensions must be integer > 0.")
        self.shape = shape
        self.pos = np.asarray(list(np.ndindex(shape)), dtype=int)
        self.tree = cKDTree(self.pos)
        self.rows, self.cols = np.indices(shape)

    def nhb_idx(self, point: Coord, radius: float) -> Array:
        """Compute the neighbourhood within ``radius`` around ``point``.

        Args:
            point:   Coordinate in a two-dimensional array.
            radius:  Lenght of radius.

        Returns:
            Array of indices of neighbours.
        """
        return np.asarray(self.tree.query_ball_point(point, radius, np.inf))

    def nhb(self, point: Coord, radius: float) -> Array:
        """Compute neighbourhood within ``radius`` around ``pouint``.

        Args:
            point:   Coordinate in a two-dimensional array.
            radius:  Lenght of radius.

        Returns:
            Array of positions of neighbours.
        """
        idx = self.nhb_idx(point, radius)
        return self.pos[idx]

    def __iter__(self):
        for row, col in zip(self.rows.flat, self.cols.flat):
            yield row, col

    def rc(self):
        return iter(self)

    def cr(self):
        for row, col in zip(self.rows.flat, self.cols.flat):
            yield col, row

