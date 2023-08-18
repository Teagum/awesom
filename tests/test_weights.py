"""
Tests for weights.py module
"""

import numpy as np
from hypothesis import given

from awesom.typing import SomDims
from awesom.weights import Weights

from .utils import som_dims


@given(som_dims)
def test_update(dim: SomDims) -> None:
    weights = Weights(*dim)
    vals = np.zeros((dim[0]*dim[1], dim[2]))
    weights.update(vals)
    assert weights.vectors.dtype.type is np.double
