"""
Broadcasting support for infinite arrays.
"""

import numpy as np
from typing import Callable, Tuple, Union, Iterator, Any
from .arrays import InfiniteArray
from .ranges import OneToInf
from ._utils import get_infinity, is_infinity

# Get ∞ from the module's namespace and add to local namespace
_INF = get_infinity()
globals()['∞'] = _INF


class BroadcastArray(InfiniteArray):
    """Lazy broadcasted array that computes values on-demand."""
    
    def __init__(self, func: Callable, shape: Tuple, dtype=None):
        super().__init__(dtype)
        self._func = func
        self._shape = shape
        self._ndim = len(shape) if isinstance(shape, tuple) else 1
        
        if self._ndim == 1 and (isinstance(shape, tuple) and len(shape) == 1 and is_infinity(shape[0])):
            self._indices = OneToInf()
        else:
            self._indices = OneToInf()
    
    def __getitem__(self, key: Union[int, slice, Tuple]) -> Any:
        if isinstance(key, int):
            # For 1D, key is 0-based index, but we compute based on 1-based position
            result = self._func(key)
            if hasattr(result, '__iter__') and not isinstance(result, str):
                try:
                    return result[0] if len(result) == 1 else result
                except:
                    return result
            return result
        elif isinstance(key, slice):
            return self
        elif isinstance(key, tuple):
            # Multi-dimensional indexing
            if len(key) == 1:
                return self.__getitem__(key[0])
            result = self._func(key)
            return result
        return self._func(key)
    
    def __iter__(self) -> Iterator:
        i = 0
        while True:
            try:
                yield self.__getitem__(i)
                i += 1
            except (IndexError, StopIteration):
                break
            except Exception as e:
                # If function fails, stop iteration
                break
    
    def shape(self) -> Tuple:
        return self._shape

