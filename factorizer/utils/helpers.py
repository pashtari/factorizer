from typing import Any, Text, Tuple, Union, Sequence, Callable
import inspect
from contextlib import contextmanager
from itertools import accumulate
from functools import partial, reduce
from operator import mul


@contextmanager
def null_context():
    yield


class UniversalTuple(tuple):
    def __contains__(self, other: Any) -> bool:
        return True


def as_tuple(obj: Any) -> Tuple[Any, ...]:
    """Convert to tuple."""
    if not isinstance(obj, Sequence) or isinstance(obj, Text):
        obj = (obj,)

    return tuple(obj)


def prod(x: Sequence[float]):
    return reduce(mul, x, 1)


def cumprod(x: Sequence[float]):
    return list(accumulate(x, mul))


def has_args(obj: Any, keywords: Union[str, Sequence[str]]) -> bool:
    """
    Return a boolean indicating whether the given callable `obj` has
    the `keywords` in its signature.
    """
    if not callable(obj):
        return False

    sig = inspect.signature(obj)
    return all(key in sig.parameters for key in as_tuple(keywords))


def wrap_class(obj: Union[Sequence, Callable]) -> Callable:
    assert isinstance(
        obj, (Sequence, Callable)
    ), f"{obj} should be a sequence or callable."

    if isinstance(obj, Sequence) and isinstance(obj[0], Callable):
        args = []
        kwargs = {}
        for i, a in enumerate(obj):
            if i == 0:
                callable_obj = a
            elif isinstance(a, Sequence):
                args.extend(a)
            elif isinstance(a, dict):
                kwargs.update(a)

        return partial(callable_obj, *args, **kwargs)
    else:
        return obj


def is_wrappable_class(obj: Any) -> bool:
    if isinstance(obj, Callable):
        out = True
    elif isinstance(obj, Sequence):
        flags = []
        for i, a in enumerate(obj):
            if i == 0:
                flags.append(isinstance(a, Callable))
            else:
                flags.append(isinstance(a, (Sequence, dict)))

        out = all(flags)
    else:
        out = False

    return out
