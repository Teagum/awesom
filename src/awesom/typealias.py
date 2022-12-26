"""
Type aliases
"""
import pathlib
from typing import Any, Callable

from matplotlib import axes
import numpy as np


Array = np.ndarray
IntArray = np.ndarray[Any, np.dtype[np.int_]]
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

Axis = axes.Axes

Coord = tuple[int, int]
Shape = tuple[int, int]
SomDims = tuple[int, int, int]

Metric = str | Callable[[Array, Array], float]
WeightInit = str | Callable[[Array, Shape], Array]

FilePath = pathlib.Path | str
