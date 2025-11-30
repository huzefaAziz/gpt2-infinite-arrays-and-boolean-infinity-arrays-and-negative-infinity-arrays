"""
Caching functionality for infinite arrays to enable mutation.
"""

import numpy as np
from typing import Dict, Union, Tuple, Iterator, Any, Optional
from .arrays import InfiniteArray
from .ranges import OneToInf
from ._utils import get_infinity, is_infinity

# Get ∞ from the module's namespace and add to local namespace
_INF = get_infinity()
globals()['∞'] = _INF


class CachedArray(InfiniteArray):
    """Cached version of an infinite array that allows mutation."""
    
    def __init__(self, array: InfiniteArray, dtype=None):
        super().__init__(dtype or array.dtype)
        self._base_array = array
        self._cache: Dict[Union[int, Tuple], Any] = {}
        self._shape = array.shape()
        self._ndim = len(self._shape) if isinstance(self._shape, tuple) else 1
        
        if self._ndim == 1 and (isinstance(self._shape, tuple) and len(self._shape) == 1 and is_infinity(self._shape[0])):
            self._indices = OneToInf()
        else:
            self._indices = OneToInf()
    
    def __getitem__(self, key: Union[int, slice, Tuple]) -> Any:
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        # Otherwise get from base array
        value = self._base_array[key]
        
        # Cache it (but don't cache slices)
        if not isinstance(key, slice):
            self._cache[key] = value
        
        return value
    
    def __setitem__(self, key: Union[int, Tuple], value: Any) -> None:
        """Set an item in the cached array."""
        if isinstance(key, slice):
            raise ValueError("Cannot set slice of infinite array")
        self._cache[key] = value
    
    def __iter__(self) -> Iterator:
        i = 0
        while True:
            try:
                yield self.__getitem__(i)
                i += 1
            except (IndexError, StopIteration):
                break
            except Exception:
                break
    
    def shape(self) -> Tuple:
        return self._shape
    
    def __repr__(self) -> str:
        # Show first few elements
        items = []
        try:
            for i, item in enumerate(self):
                if i >= 12:
                    items.append("⋮")
                    break
                items.append(str(item))
        except:
            items.append("...")
        
        return f"{self.__class__.__name__}{self.shape()}:\n  " + "\n  ".join(items)


def cache(array: InfiniteArray) -> CachedArray:
    """Convert an infinite array to a cached (mutable) version."""
    return CachedArray(array)

