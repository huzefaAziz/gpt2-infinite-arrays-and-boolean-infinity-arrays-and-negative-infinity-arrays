"""
Infinite array types.
"""

import numpy as np
from typing import Tuple, Union, Iterator, Optional, Any
from abc import ABC, abstractmethod
from .ranges import OneToInf, InfUnitRange
from .infinity import Infinity
from ._utils import get_infinity, is_infinity

# Get ∞ from the module's namespace and add to local namespace
_INF = get_infinity()
globals()['∞'] = _INF


class InfiniteArray(ABC):
    """Base class for infinite arrays."""
    
    def __init__(self, dtype=None):
        self.dtype = dtype or np.float64
    
    @abstractmethod
    def __getitem__(self, key: Union[int, slice, Tuple]) -> Any:
        """Get item or slice from the array."""
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator:
        """Iterate over array elements."""
        pass
    
    @abstractmethod
    def shape(self) -> Tuple:
        """Return the shape of the array."""
        pass
    
    def __len__(self) -> int:
        """Length is undefined for infinite arrays."""
        raise TypeError("len() of infinite array is undefined")
    
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
    
    def __array__(self, dtype=None):
        """Convert to numpy array (will fail for infinite arrays)."""
        raise ValueError("Cannot convert infinite array to numpy array")
    
    def __add__(self, other):
        """Element-wise addition."""
        from .broadcasting import BroadcastArray
        return BroadcastArray(lambda i: self[i] + other[i] if hasattr(other, '__getitem__') else self[i] + other, self.shape())
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        """Element-wise subtraction."""
        from .broadcasting import BroadcastArray
        return BroadcastArray(lambda i: self[i] - other[i] if hasattr(other, '__getitem__') else self[i] - other, self.shape())
    
    def __rsub__(self, other):
        from .broadcasting import BroadcastArray
        return BroadcastArray(lambda i: other - self[i], self.shape())
    
    def __mul__(self, other):
        """Element-wise multiplication."""
        from .broadcasting import BroadcastArray
        return BroadcastArray(lambda i: self[i] * other[i] if hasattr(other, '__getitem__') else self[i] * other, self.shape())
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Element-wise division."""
        from .broadcasting import BroadcastArray
        return BroadcastArray(lambda i: self[i] / other[i] if hasattr(other, '__getitem__') else self[i] / other, self.shape())
    
    def __rtruediv__(self, other):
        from .broadcasting import BroadcastArray
        return BroadcastArray(lambda i: other / self[i], self.shape())


class Ones(InfiniteArray):
    """Infinite array filled with ones."""
    
    def __init__(self, shape: Union[Infinity, Tuple, None] = None, dtype=None):
        if shape is None or is_infinity(shape):
            shape = (_INF,)
        elif isinstance(shape, int):
            shape = (shape,)
        
        super().__init__(dtype)
        self._shape = shape
        self._ndim = len(shape)
        
        # For 1D case, use OneToInf for indices
        if self._ndim == 1 and is_infinity(self._shape[0]):
            self._indices = OneToInf()
        else:
            self._indices = OneToInf()  # Default for 1D
    
    def __getitem__(self, key: Union[int, slice, Tuple]) -> Union[float, 'Ones']:
        if isinstance(key, int):
            return np.ones(1, dtype=self.dtype)[0]
        elif isinstance(key, slice):
            return self
        elif isinstance(key, tuple):
            # Multi-dimensional indexing
            return np.ones(1, dtype=self.dtype)[0]
        return np.ones(1, dtype=self.dtype)[0]
    
    def __iter__(self) -> Iterator[float]:
        while True:
            yield np.ones(1, dtype=self.dtype)[0]
    
    def shape(self) -> Tuple:
        return self._shape
    
    def __repr__(self) -> str:
        if self._ndim == 1 and is_infinity(self._shape[0]):
            items = ["1.0"] * 12 + ["⋮"]
            shape_str = f"with indices {self._indices}:"
        else:
            shape_str = f"{self.shape()}"
            items = ["1.0"] * 12 + ["⋮"]
        return f"{self.__class__.__name__}{self.shape()}:\n  " + "\n  ".join(items)


class Zeros(InfiniteArray):
    """Infinite array filled with zeros."""
    
    def __init__(self, shape: Union[Infinity, Tuple, None] = None, dtype=None):
        if shape is None or is_infinity(shape):
            shape = (_INF,)
        elif isinstance(shape, int):
            shape = (shape,)
        
        super().__init__(dtype)
        self._shape = shape
        self._ndim = len(shape)
        
        if self._ndim == 1 and is_infinity(self._shape[0]):
            self._indices = OneToInf()
        else:
            self._indices = OneToInf()
    
    def __getitem__(self, key: Union[int, slice, Tuple]) -> Union[float, 'Zeros']:
        if isinstance(key, int):
            return np.zeros(1, dtype=self.dtype)[0]
        elif isinstance(key, slice):
            return self
        elif isinstance(key, tuple):
            return np.zeros(1, dtype=self.dtype)[0]
        return np.zeros(1, dtype=self.dtype)[0]
    
    def __iter__(self) -> Iterator[float]:
        while True:
            yield np.zeros(1, dtype=self.dtype)[0]
    
    def shape(self) -> Tuple:
        return self._shape


class Fill(InfiniteArray):
    """Infinite array filled with a constant value."""
    
    def __init__(self, value: Any, shape: Union[Infinity, Tuple, None] = None, dtype=None):
        if shape is None or is_infinity(shape):
            shape = (_INF,)
        elif isinstance(shape, int):
            shape = (shape,)
        
        super().__init__(dtype)
        self._value = value
        self._shape = shape
        self._ndim = len(shape)
        
        if self._ndim == 1 and is_infinity(self._shape[0]):
            self._indices = OneToInf()
        else:
            self._indices = OneToInf()
    
    def __getitem__(self, key: Union[int, slice, Tuple]) -> Any:
        if isinstance(key, int):
            return self._value
        elif isinstance(key, slice):
            return self
        elif isinstance(key, tuple):
            return self._value
        return self._value
    
    def __iter__(self) -> Iterator:
        while True:
            yield self._value
    
    def shape(self) -> Tuple:
        return self._shape

