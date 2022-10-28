"""
Type aliases
"""
import pathlib
from typing import Callable, Union

import matplotlib as mpl
import numpy as np


Array = np.ndarray
Axis = mpl.axes.Axes

Coord = tuple[int, int]
Shape = tuple[int, int]
SomDims = tuple[int, int, int]

Metric = Union[Callable[[Array, Array], float], str]
WeightInit = Union[Callable[[Array, Shape], Array], str]

FilePath = pathlib.Path | str
