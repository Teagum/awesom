"""
Type aliases
"""
from typing import Callable, Union

import numpy as np


Array = np.ndarray

Coord = tuple[int, int]
Shape = tuple[int, int]
SomDims = tuple[int, int, int]

Metric = Union[Callable[[Array, Array], float], str]
WeightInit = Union[Callable[[Array, Shape], Array], str]
