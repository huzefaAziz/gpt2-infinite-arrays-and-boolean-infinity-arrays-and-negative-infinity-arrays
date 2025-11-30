"""
Diagonal infinite arrays.
"""

import numpy as np
from typing import Tuple, Union, Iterator, Any
from .arrays import InfiniteArray
from .ranges import OneToInf, InfUnitRange
from ._utils import get_infinity

# Get ∞ from the module's namespace and add to local namespace
_INF = get_infinity()
globals()['∞'] = _INF


class InfiniteDiagonal(InfiniteArray):
    """Infinite diagonal matrix with values from a sequence."""
    
    def __init__(self, values: Union[Iterator, InfiniteArray], dtype=None):
        super().__init__(dtype)
        self._values = values
        self._shape = (_INF, _INF)
        self._ndim = 2
        self._indices = (OneToInf(), OneToInf())
        
        # If values is iterable, create iterator
        if hasattr(values, '__iter__') and not isinstance(values, InfiniteArray):
            self._value_iter = iter(values)
            self._value_cache = {}
        elif hasattr(values, '__getitem__'):
            # For array-like objects, use __getitem__
            self._value_iter = None
            self._value_cache = {}
        else:
            raise TypeError("values must be iterable or array-like")
    
    def _get_value(self, i: int) -> Any:
        """Get the i-th diagonal value (1-based)."""
        if hasattr(self._values, '__getitem__'):
            try:
                # Try 0-based indexing
                return self._values[i - 1]
            except (IndexError, TypeError):
                # Try 1-based indexing
                try:
                    return self._values[i]
                except (IndexError, TypeError):
                    # Fall back to iteration
                    if i in self._value_cache:
                        return self._value_cache[i]
                    if self._value_iter is None:
                        self._value_iter = iter(self._values)
                    # Advance to position i
                    while len(self._value_cache) < i:
                        try:
                            val = next(self._value_iter)
                            self._value_cache[len(self._value_cache) + 1] = val
                        except StopIteration:
                            break
                    return self._value_cache.get(i, 0)
        else:
            # Use iteration
            if i in self._value_cache:
                return self._value_cache[i]
            if self._value_iter is None:
                self._value_iter = iter(self._values)
            # Advance to position i
            while len(self._value_cache) < i:
                try:
                    val = next(self._value_iter)
                    self._value_cache[len(self._value_cache) + 1] = val
                except StopIteration:
                    return 0
            return self._value_cache.get(i, 0)
    
    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[float, 'InfiniteDiagonal']:
        if isinstance(key, int):
            # Return diagonal element
            return self._get_value(key + 1)  # Convert 0-based to 1-based
        elif isinstance(key, tuple) and len(key) == 2:
            row, col = key
            # Diagonal element if row == col
            if row == col:
                return self._get_value(row + 1)  # Convert 0-based to 1-based
            else:
                # Off-diagonal elements are zero
                return 0.0
        elif isinstance(key, slice):
            return self
        return 0.0
    
    def __iter__(self) -> Iterator:
        # Iterator over rows (each row is itself an iterator)
        for i in range(100):  # Limit for display purposes
            yield self._get_row(i)
    
    def _get_row(self, row_idx: int) -> Iterator:
        """Get a row as an iterator."""
        for col_idx in range(100):  # Limit for display purposes
            if row_idx == col_idx:
                yield self._get_value(row_idx + 1)
            else:
                yield 0.0
    
    def shape(self) -> Tuple:
        return self._shape
    
    def __repr__(self) -> str:
        # Show a matrix representation
        rows = []
        n = 15  # Number of rows/cols to show
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(str(self._get_value(i + 1)))
                elif j == n - 1:
                    row.append("…")
                    break
                else:
                    row.append("⋅")
            rows.append("  ".join(row))
        
        return f"{self.__class__.__name__}{self.shape()}:\n" + "\n".join(rows) + "\n⋮"

