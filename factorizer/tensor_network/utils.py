from typing import Sequence
from functools import reduce
from operator import mul


def prod(x: Sequence[float]):
    return reduce(mul, x, 1)
