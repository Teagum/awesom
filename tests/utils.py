from hypothesis.extra.numpy import array_shapes
from hypothesis.strategies import integers, tuples

som_dims = array_shapes(min_dims=3, max_dims=3, min_side=100, max_side=100)
