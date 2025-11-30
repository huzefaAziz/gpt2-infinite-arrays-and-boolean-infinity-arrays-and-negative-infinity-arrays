"""
Infinite range types for indexing infinite arrays.
"""

import itertools
from typing import Iterator, Optional, Union, overload
from .infinity import Infinity
from ._utils import get_infinity

# Get ∞ from the module's namespace and add to local namespace
_INF = get_infinity()
globals()['∞'] = _INF


class OneToInf:
    """Infinite range starting from 1: 1, 2, 3, ..."""
    
    def __init__(self):
        self.start = 1
    
    def __repr__(self) -> str:
        return "OneToInf()"
    
    def __str__(self) -> str:
        return "OneToInf()"
    
    def __iter__(self) -> Iterator[int]:
        return itertools.count(1)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[int, 'OneToInf']:
        if isinstance(key, int):
            if key < 0:
                raise IndexError("negative indices not supported for OneToInf")
            return key + 1  # Convert 0-based to 1-based
        elif isinstance(key, slice):
            # For slicing, we can't return a finite slice, so return self
            return self
        raise TypeError(f"OneToInf indices must be integers or slices, not {type(key)}")
    
    def __len__(self) -> int:
        raise TypeError("len() of infinite OneToInf is undefined")
    
    def __contains__(self, item: int) -> bool:
        return isinstance(item, int) and item >= 1
    
    def index(self, value: int) -> int:
        """Return 0-based index of value."""
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{value} not in OneToInf")
        return value - 1
    
    def count(self, value: int) -> int:
        """Return count of value (always 1 if in range, 0 otherwise)."""
        return 1 if value in self else 0


class InfUnitRange:
    """Infinite unit range starting from a given value: start, start+1, start+2, ..."""
    
    def __init__(self, start: int = 1, step: int = 1):
        self.start = start
        self.step = step
    
    def __repr__(self) -> str:
        if self.step == 1:
            return f"InfUnitRange({self.start})"
        return f"InfUnitRange({self.start}, step={self.step})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __iter__(self) -> Iterator[int]:
        return itertools.count(self.start, self.step)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[int, 'InfUnitRange']:
        if isinstance(key, int):
            if key < 0:
                raise IndexError("negative indices not supported for InfUnitRange")
            return self.start + key * self.step
        elif isinstance(key, slice):
            return self
        raise TypeError(f"InfUnitRange indices must be integers or slices, not {type(key)}")
    
    def __len__(self) -> int:
        raise TypeError("len() of infinite InfUnitRange is undefined")
    
    def __contains__(self, item: int) -> bool:
        if not isinstance(item, int):
            return False
        if self.step == 1:
            return item >= self.start
        # For non-unit steps, check if item is reachable
        return (item - self.start) % self.step == 0 and item >= self.start
    
    def index(self, value: int) -> int:
        """Return 0-based index of value."""
        if value not in self:
            raise ValueError(f"{value} not in {self}")
        return (value - self.start) // self.step
    
    def count(self, value: int) -> int:
        """Return count of value (always 1 if in range, 0 otherwise)."""
        return 1 if value in self else 0


class InfStepRange:
    """Infinite step range: start, start+step, start+2*step, ..."""
    
    def __init__(self, start: int, step: int):
        self.start = start
        self.step = step
    
    def __repr__(self) -> str:
        return f"InfStepRange({self.start}, {self.step})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __iter__(self) -> Iterator[int]:
        return itertools.count(self.start, self.step)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[int, 'InfStepRange']:
        if isinstance(key, int):
            if key < 0:
                raise IndexError("negative indices not supported for InfStepRange")
            return self.start + key * self.step
        elif isinstance(key, slice):
            return self
        raise TypeError(f"InfStepRange indices must be integers or slices, not {type(key)}")
    
    def __len__(self) -> int:
        raise TypeError("len() of infinite InfStepRange is undefined")
    
    def __contains__(self, item: int) -> bool:
        if not isinstance(item, int):
            return False
        return (item - self.start) % self.step == 0 and (
            (self.step > 0 and item >= self.start) or
            (self.step < 0 and item <= self.start)
        )
    
    def index(self, value: int) -> int:
        """Return 0-based index of value."""
        if value not in self:
            raise ValueError(f"{value} not in {self}")
        return (value - self.start) // self.step
    
    def count(self, value: int) -> int:
        """Return count of value (always 1 if in range, 0 otherwise)."""
        return 1 if value in self else 0

