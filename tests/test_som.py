"""
Tests for som.py module
"""
from hypothesis import given

from awesom.som import SomBase
from awesom.typing import SomDims
from awesom.weights import Weights

from .utils import som_dims


@given(som_dims)
def test_som_base(dims: SomDims) -> None:
    weights = Weights(*dims)
    som = SomBase(dims, 100, 0.1, 10, "gaussian", weights)
    assert som
