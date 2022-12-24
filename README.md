[![mypy](https://github.com/Teagum/blossom/actions/workflows/mypy.yml/badge.svg)](https://github.com/Teagum/blossom/actions/workflows/mypy.yml)
[![pylint](https://github.com/Teagum/blossom/actions/workflows/pylint.yml/badge.svg)](https://github.com/Teagum/blossom/actions/workflows/pylint.yml)

# Awesom
Self-organizing map framework for Python


```python
import matplotlib.pyplot as plt

from awesom import datasets
from awesom import plot as asp
from awesom.som import IncrementalMap


X, y = datasets.norm_circle(5, 500, 1, radius=4)

som = IncrementalMap((7, 7, X.shape[1]), 100, 0.04, 4)
som.fit(X)

fig, ax = plt.subplots(1, 1)
asp.data_2d(ax, X, y)
asp.wire(ax, som)
```

![SOM wire plot](https://user-images.githubusercontent.com/11088297/209104159-958cfbef-15f5-4259-9c15-bfebcb76058e.png "Input dataspce with wire plot")
