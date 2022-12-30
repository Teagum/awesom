"""
Implementation of the ``Weights``
"""
import numpy as np


class Weights:
    """Weights
    """
    def __init__(self, dx: int, dy: int, dw: int) -> None:
        self.dx = dx
        self.dy = dy
        self.dw = dw
        self.shape = (self.dx, self.dw, self.dw)
        self.n_units = self.dx * self.dy
        self._weights = np.empty((self.n_units, self.dw), dtype=np.float64)

    def __getitem__(self, key):
        return self._weights[key]
