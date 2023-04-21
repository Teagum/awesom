#!/usr/bin/env python3
import sys
import time

from sklearn.datasets import load_iris
from awesom.som import IncrementalMap
from awesom.typing import FloatArray


SHAPE = (30, 30, 4)
N_ITER = 100
ETA = 0.1
NHR = 10


def main(argv=None):
    if argv is None:
        argv = sys.argv

    training_data, training_labels = load_iris(return_X_y=True)

    som = IncrementalMap(SHAPE, N_ITER, ETA, NHR)
    timeit(som, training_data)

    return 0


def timeit(inst, data: FloatArray, repeat: int = 5) -> None:
    checks = []
    for _ in range(repeat):
        start = time.perf_counter()
        inst.fit(data)
        pace = time.perf_counter() - start
        checks.append(pace)

    perf = sum(checks) / repeat
    print(f"{inst.__class__.__name__:20}{perf:.5} s")


if __name__ == "__main__":
    sys.exit(main())
