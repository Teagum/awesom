"""
Topologies for self-organizing maps.
"""

import numpy as np


def vn_neighbourhood(x, y, dx, dy, flat=False):
    """ Compute Von Neuman Neighbourhood.

    Compute the Von Neuman Neighbourhood of index (x, y) given an array with
    dimension (dx, dy). The Von Neumann Neighbourhood is the 4-neighbourhood,
    which includes the four direct neighbours of index (x, y) given a rect-
    angular array.

    Params:
        x    (int)       x-Index for which to compute the neighbourhood.
        y    (int)       y-Index for which to compute the neighbourhood.
        dx   (int)       Size of enclosing array's x-axis.
        dy   (int)       Size of enclosing array's y-axis.
        flat (bool)      Return flat index if True. Default is False.

    Return:
        1d-array of ints if flat, 2d-array otherwise.
    """
    nh = []

    if x-1 >= 0:
        nh.append((x-1, y))
    if x+1 < dx:
        nh.append((x+1, y))
    if y-1 >= 0:
        nh.append((x, y-1))
    if y+1 < dy:
        nh.append((x, y+1))

    if flat:
        nh = np.array(nh)
        return np.ravel_multi_index(nh.T, (dx, dy))
    else:
        return nh
