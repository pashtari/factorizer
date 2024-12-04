from typing import Any, Sequence, Callable, Type, Iterable
import inspect
from itertools import accumulate
from functools import partial
from operator import mul

from torch import nn

PositionalArgs = tuple[Any, ...]
KeywordArgs = dict[str, Any]
ArgsType = PositionalArgs | KeywordArgs

PartialFunctionType = tuple[Callable[..., Any] | ArgsType, ...]
PartialModuleType = tuple[Type[nn.Module] | ArgsType, ...]

# Below is more accurate but works only on python 3.11+
# PartialFunctionType = tuple[Callable[..., Any], *Sequence[ArgsType]]
# PartialModuleType = tuple[Type[nn.Module], *Sequence[ArgsType]]


class Universaltuple(tuple):
    """
    A custom tuple that always returns True when checking for membership.

    Examples:
        ut = Universaltuple((1, 2, 3))
        print(4 in ut)  # Output: True
        print("anything" in ut)  # Output: True
    """

    def __contains__(self, other: Any) -> bool:
        """Overrides __contains__ to always return True."""
        return True


def as_tuple(obj: Any) -> tuple[Any, ...]:
    """
    Converts an object to a tuple.

    If the object is a sequence but not a string, it is converted directly.
    Otherwise, it is wrapped in a single-element tuple.

    Args:
        obj (Any): The object to convert.

    Returns:
        tuple[Any, ...]: The converted tuple.
    """
    if not isinstance(obj, Sequence) or isinstance(obj, str):
        return (obj,)
    return tuple(obj)


def cumprod(x: Iterable[float]) -> list[float]:
    """
    Calculates the cumulative product of an iterable of numbers.

    Args:
        x (Iterable[float]): An iterable of numbers.

    Returns:
        list[float]: A list of the cumulative products.
    """
    return list(accumulate(x, mul))


def has_args(obj: Any, keywords: str | Sequence[str]) -> bool:
    """
    Checks if a callable object has specific keyword arguments.

    Args:
        obj (Any): The callable object to inspect.
        keywords (str | Sequence[str]): A single keyword or a sequence of
            keywords to check against the callable's arguments.

    Returns:
        bool: True if the callable object has all the specified keywords in its
            arguments, False otherwise.
    """
    if not callable(obj):
        return False

    try:
        sig = inspect.signature(obj)
    except ValueError:
        return False

    return all(key in sig.parameters for key in as_tuple(keywords))


def partialize(obj: PartialFunctionType) -> Callable:
    """
    Wraps into a partial callable with given arguments.

    If `obj` is a tuple, the first element is assumed to be a callable and
    the subsequent elements are positional and/or keyword arguments.
    Otherwise, returns the callable as-is.

    Args:
        obj (PartialFunctionType): A tuple with the first element as a
            callable and subsequent elements as positional arguments
            (in tuple form) and/or keyword arguments (in dict form), or just a callable.

    Returns:
        Callable: A callable, either the original object or a partial function.

    Raises:
        TypeError: If `obj` is neither a callable nor a valid tuple format.
    """
    if callable(obj):
        return obj

    if isinstance(obj, Sequence) and callable(obj[0]):
        callable_obj = obj[0]
        args = []
        kwargs = {}

        for item in obj[1:]:
            if isinstance(item, dict):
                kwargs.update(item)
            elif isinstance(item, Sequence) and not isinstance(item, str):
                args.extend(item)
            else:
                args.append(item)

        return partial(callable_obj, *args, **kwargs)

    raise TypeError(f"Expected a callable or valid tuple, got {type(obj).__name__}")


def is_partializable(obj: Any) -> bool:
    """
    Checks if an object can be turned into a callable by `partialize`.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: True if the object can be wrapped, False otherwise.
    """
    if callable(obj):
        return True

    if isinstance(obj, Sequence) and obj and callable(obj[0]):
        return True

    return False
