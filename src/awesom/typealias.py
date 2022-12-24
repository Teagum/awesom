"""
Type aliases
"""
import pathlib
from typing import Callable

from matplotlib import axes
import numpy as np


Array = np.ndarray
Axis = axes.Axes

Coord = tuple[int, int]
Shape = tuple[int, int]
SomDims = tuple[int, int, int]

Metric = str | Callable[[Array, Array], float]
WeightInit = str | Callable[[Array, Shape], Array]

FilePath = pathlib.Path | str
