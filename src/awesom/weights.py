"""
Implementation of the ``Weights``
"""
from typing import Any, cast

import numpy as np

import awesom.utilities as utils
from awesom.typing import FilePath, FloatArray, IntArray


class Weights:
    """Weights
    """
    def __init__(self, dx: int, dy: int, dw: int,
                 seed: int | IntArray | None = None) -> None:
        self.dx = dx
        self.dy = dy
        self.dw = dw
        self.shape = (self.dx, self.dy, self.dw)
        self.n_units = self.dx * self.dy
        self._vectors = np.empty((self.n_units, self.dw), dtype=np.float64)
        self._rng = np.random.default_rng(seed)


    def __getitem__(self, key: Any) -> FloatArray:
        return cast(FloatArray, self._vectors[key])


    @property
    def vectors(self) -> FloatArray:
        """Return weight vectors"""
        return self._vectors


    def update(self, buff: FloatArray) -> None:
        """Update the weight vectors

        Args:
            buff:   Weight updates
        """
        if buff.dtype != self._vectors.dtype:
            raise TypeError(f"Update buffer has incompatible type <{buff.dtype}>. "
                            f"Expected <{self._vectors.dtype}>.")

        if buff.shape != self._vectors.shape:
            raise TypeError(f"Update buffer has incompatible shape <{buff.shape}>. "
                            f"Expected <{self._vectors.shape}>")

        np.add(buff, self._vectors, out=self._vectors)


    def init_pca(self, training_data: FloatArray | None = None,
                 adapt: bool = True) -> None:
        """Initialize weights using PCA method

        Compute initial SOM weights by sampling from the first two principal
        components of the input data set.

        Args:
            trainig_data:  Input data set
            adapt:  If ``True``, the largest value of ``shape`` is applied to the
                    principal component with the largest sigular value. This
                    orients the map, such that map dimension with the most units
                    coincides with principal component with the largest variance.
        """
        if training_data is None:
            training_data = self._rng.integers(-100, 100, (300, self.dw)).astype(float)
        _, vects, trans_data = utils.pca(training_data, 2)

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
        self._vectors[...] = points.T @ vects + training_data.mean(axis=0)


    def init_rnd(self, training_data: FloatArray | None = None) -> None:
        """Compute initial SOM weights by sampling uniformly from the data space.

        Args:
            dims:  Dimensions of SOM
            data:  Input data set. If ``None``, sample from [-10, 10]
        """
        if training_data is not None:
            data_limits = np.column_stack((training_data.min(axis=0),
                                           training_data.max(axis=0)))
        else:
            data_limits = self._rng.integers(-10, 10, (self.dw, 2))
            data_limits.sort()
        weights = [self._rng.uniform(dmin, dmax, self.dx*self.dy)
                   for (dmin, dmax) in data_limits]
        self._vectors[...] = np.column_stack(weights)


    def init_stv(self) -> None:
        """Initialize with stochastic vectors
        """
        nvt = self.dx * self.dy
        self._vectors[...] = utils.sample_st_vector(nvt, self.dw)


    def init_stm(self) -> None:
        """Initialize with stochastic matrices
        """
        nvt = self.dx * self.dy
        self._vectors[...] = utils.sample_st_matrix(nvt, self.dw)


    def save_vectors(self, path: FilePath) -> None:
        """Store weight vector in a portable `.npy` file

        Args:
            path:  File path
        """
        np.save(path, self._vectors, allow_pickle=False)
