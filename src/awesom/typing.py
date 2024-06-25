"""
Type aliases
"""
import pathlib
from typing import Any, Callable

from matplotlib import axes
from mpl_toolkits import mplot3d
import numpy as np
import numpy.typing as npt


IntArray = np.ndarray[Any, np.dtype[np.int_]]
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

Axis = axes.Axes
Axes3D = mplot3d.axes3d.Axes3D

Coord = tuple[int, int]
Shape = tuple[int, int]
SomDims = tuple[int, int, int]

Metric = str | Callable[[FloatArray, FloatArray], float]

FilePath = pathlib.Path | str

DistFunc = Callable[
    [IntArray, npt.ArrayLike, float, FloatArray | None],
    FloatArray]
