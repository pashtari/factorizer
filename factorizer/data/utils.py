from typing import Callable, Sequence, Union
import inspect
from functools import partial, partialmethod


def partialclass(obj, *args, **kwargs):
    class NewCls(obj):
        __init__ = partialmethod(obj.__init__, *args, **kwargs)

    NewCls.__name__ = obj.__name__
    locals()[obj.__name__] = NewCls
    del NewCls
    return locals()[obj.__name__]


def wrap_class(obj: Union[Sequence, Callable]):
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

        if inspect.isclass(callable_obj):
            return partialclass(callable_obj, *args, **kwargs)
        else:
            return partial(callable_obj, *args, **kwargs)
    else:
        return obj
