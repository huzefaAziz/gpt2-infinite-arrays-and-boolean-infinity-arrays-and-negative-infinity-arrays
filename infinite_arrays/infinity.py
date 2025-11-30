"""
Infinity constant and related utilities.
"""

import sys
from typing import Any


class Infinity:
    """Represents infinity for array dimensions."""
    
    def __repr__(self) -> str:
        return "∞"
    
    def __str__(self) -> str:
        return "∞"
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Infinity)
    
    def __ne__(self, other: Any) -> bool:
        return not isinstance(other, Infinity)
    
    def __hash__(self) -> int:
        return hash(float('inf'))
    
    def __lt__(self, other: Any) -> bool:
        return False
    
    def __le__(self, other: Any) -> bool:
        return isinstance(other, Infinity)
    
    def __gt__(self, other: Any) -> bool:
        return not isinstance(other, Infinity)
    
    def __ge__(self, other: Any) -> bool:
        return True
    
    def __add__(self, other: Any) -> 'Infinity':
        return self
    
    def __radd__(self, other: Any) -> 'Infinity':
        return self
    
    def __sub__(self, other: Any) -> 'Infinity':
        return self
    
    def __rsub__(self, other: Any) -> 'Infinity':
        return self
    
    def __mul__(self, other: Any) -> 'Infinity':
        if other == 0:
            raise ValueError("∞ * 0 is undefined")
        return self
    
    def __rmul__(self, other: Any) -> 'Infinity':
        if other == 0:
            raise ValueError("0 * ∞ is undefined")
        return self
    
    def __truediv__(self, other: Any) -> 'Infinity':
        if isinstance(other, Infinity):
            raise ValueError("∞ / ∞ is undefined")
        return self
    
    def __rtruediv__(self, other: Any) -> float:
        return 0.0
    
    def __int__(self) -> int:
        return sys.maxsize
    
    def __float__(self) -> float:
        return float('inf')
    
    def __index__(self) -> int:
        raise TypeError("∞ cannot be used as an index directly")


# Global infinity instance
# Store with ASCII name, then add to module namespace with Unicode symbol
_INFINITY = Infinity()
# Make it accessible as ∞ in this module's namespace
globals()['∞'] = _INFINITY

