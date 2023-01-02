"""
Implementation of the ``Weights``
"""
from typing import Any, cast

import numpy as np

from awesom import utilities
from awesom.typealias import FloatArray


class Weights:
    """Weights
    """
    def __init__(self, dx: int, dy: int, dw: int) -> None:
        self.dx = dx
        self.dy = dy
        self.dw = dw
        self.shape = (self.dx, self.dw, self.dw)
        self.n_units = self.dx * self.dy
        self.vectors = np.empty((self.n_units, self.dw), dtype=np.float64)

    def __getitem__(self, key: Any) -> FloatArray:
        return cast(FloatArray, self.vectors[key])

    def init_pca(self, training_data: FloatArray | None = None,
                 adapt: bool = True) -> None:
        """Initialize weights using PCA method

        Compute initial SOM weights by sampling from the first two principal
        components of the input data set.

        Args:
            trainig_data:   Input data set
            adapt:  If ``True``, the largest value of ``shape`` is applied to the
                    principal component with the largest sigular value. This
                    orients the map, such that map dimension with the most units
                    coincides with principal component with the largest variance.

        Returns:
            Array of weights.
        """
        if training_data is None:
            data = np.random.randint(-100, 100, (300, self.dw)).astype(float)
            _, vects, trans_data = utilities.pca(data, 2)
        else:
            _, vects, trans_data = utilities.pca(training_data, 2)

        if adapt:
            shape = tuple(sorted((self.dx, self.dy), reverse=True))
        else:
            shape = (self.dx, self.dy)

        data_min = trans_data.min(axis=0)
        data_max = trans_data.max(axis=0)
        dim_x = np.linspace(data_min[0], data_max[0], shape[0])
        dim_y = np.linspace(data_min[1], data_max[1], shape[1])

        grid_x, grid_y = np.meshgrid(dim_x, dim_y)
        points = np.vstack((grid_x.ravel(), grid_y.ravel()))
        self.vectors[...] = points.T @ vects + training_data.mean(axis=0)
