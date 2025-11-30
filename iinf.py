import sys
import os
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the current directory to Python path so we can import infinite_arrays
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now we can import the module
import numpy as np
import math
from typing import Any
import infinite_arrays
from infinite_arrays import Ones, InfiniteDiagonal, cache, BroadcastArray, Zeros, Fill, CachedArray
from infinite_arrays._utils import get_infinity

_INF = infinite_arrays.__dict__.get('∞', get_infinity())
globals()['∞'] = _INF


class bool_inf:
    """
    Boolean infinity - represents infinite truth values.
    Can represent True, False, or an infinite boolean state.
    """
    
    _SENTINEL = object()
    
    def __init__(self, value=_SENTINEL):
        """Create a new bool_inf instance."""
        if value is self._SENTINEL:
            # Default: create infinite truth
            self._value = True
            self._is_inf = True
        else:
            self._value = bool(value)
            self._is_inf = False
    
    def __bool__(self):
        """Return the boolean value."""
        return self._value
    
    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, bool_inf):
            return self._value == other._value and self._is_inf == other._is_inf
        return self._value == bool(other)
    
    def __hash__(self):
        """Hash value."""
        return hash((self._value, self._is_inf))
    
    def __repr__(self):
        if self._is_inf:
            return "∞T" if self._value else "∞F"
        return f"bool_inf({self._value})"
    
    def __str__(self):
        if self._is_inf:
            return "∞T" if self._value else "∞F"
        return str(self._value)
    
    def __and__(self, other):
        """Logical AND operation."""
        if isinstance(other, (bool, bool_inf)):
            if self._is_inf:
                if self._value:
                    # ∞T & X = X
                    return bool_inf(other) if isinstance(other, bool_inf) else bool_inf(other)
                else:
                    # ∞F & X = False
                    return bool_inf(False)
            # Regular boolean AND
            return bool_inf(self._value and (bool(other) if not isinstance(other, bool_inf) else other._value))
        return super().__and__(other)
    
    def __or__(self, other):
        """Logical OR operation."""
        if isinstance(other, (bool, bool_inf)):
            if self._is_inf:
                if self._value:
                    # ∞T | X = True
                    return bool_inf(True)
                else:
                    # ∞F | X = X
                    return bool_inf(other) if isinstance(other, bool_inf) else bool_inf(other)
            # Regular boolean OR
            return bool_inf(self._value or (bool(other) if not isinstance(other, bool_inf) else other._value))
        return super().__or__(other)
    
    def __xor__(self, other):
        """Logical XOR operation."""
        if isinstance(other, (bool, bool_inf)):
            if self._is_inf:
                # Infinite XOR behaves like regular XOR with infinite value
                other_val = other._value if isinstance(other, bool_inf) else bool(other)
                return bool_inf(self._value ^ other_val)
            other_val = other._value if isinstance(other, bool_inf) else bool(other)
            return bool_inf(self._value ^ other_val)
        return super().__xor__(other)
    
    def __invert__(self):
        """Logical NOT operation."""
        return bool_inf(not self._value)
    
    def __bool__(self):
        """Return the boolean value."""
        return self._value
    
    class _bool_inf_array(InfiniteDiagonal):
        """
        Internal infinite diagonal array implementation with boolean values.
        """
        
        def __init__(self, values=None, use_inf: bool = False, dtype=None):
            """
            Initialize boolean infinity infinite diagonal array.
            
            Args:
                values: Iterable of boolean values (if None, generates sequence)
                use_inf: If True, all diagonal values are bool_inf(∞)
                dtype: Data type (default: bool)
            """
            if values is None:
                if use_inf:
                    # Create infinite sequence of bool_inf values
                    def bool_inf_gen():
                        while True:
                            yield bool_inf()
                    values = bool_inf_gen()
                    dtype = dtype or bool
                else:
                    # Create alternating True/False sequence
                    def bool_gen():
                        i = 0
                        while True:
                            yield bool(i % 2 == 0)
                            i += 1
                    values = bool_gen()
                    dtype = dtype or bool
            else:
                dtype = dtype or bool
            
            # Initialize InfiniteDiagonal with the sequence
            super().__init__(values, dtype=dtype)
            self._use_inf = use_inf
        
        def _get_value(self, i: int) -> Any:
            """Get the i-th diagonal value with bool_inf conversion if needed."""
            # Get value from parent
            val = super()._get_value(i)
            
            # Convert to bool_inf if use_inf flag is set and value isn't already bool_inf
            if self._use_inf and not isinstance(val, bool_inf):
                return bool_inf(val)
            
            return val
        
        def __repr__(self):
            """Override to show boolean representation."""
            rows = []
            n = 10  # Number of rows/cols to show
            for i in range(n):
                row = []
                for j in range(n):
                    if i == j:
                        val = self._get_value(i + 1)
                        if isinstance(val, bool_inf):
                            row.append(str(val))
                        else:
                            row.append(str(bool(val)))
                    elif j == n - 1:
                        row.append("…")
                        break
                    else:
                        row.append("⋅")
                rows.append("  ".join(row))
            
            return f"{self.__class__.__name__}{self.shape()}:\n" + "\n".join(rows) + "\n⋮"
    
    class bool_inf_array(CachedArray):
        """
        Cached infinite diagonal array with boolean infinity values.
        Wraps _bool_inf_array with caching for fast access.
        """
        
        def __init__(self, 
                     values=None, 
                     use_inf: bool = False,
                     pattern: str = "alternating",
                     dtype=None):
            """
            Initialize cached boolean infinity infinite diagonal array.
            
            Args:
                values: Iterable of boolean values (if None, generates sequence)
                use_inf: If True, all diagonal values are bool_inf(∞)
                pattern: Pattern type ("alternating", "all_true", "all_false", "custom")
                dtype: Data type (default: bool)
            """
            # Create the base array
            if values is None:
                if pattern == "alternating":
                    def bool_gen():
                        i = 0
                        while True:
                            yield bool(i % 2 == 0)
                            i += 1
                    values = bool_gen()
                elif pattern == "all_true":
                    def bool_gen():
                        while True:
                            yield True
                    values = bool_gen()
                elif pattern == "all_false":
                    def bool_gen():
                        while True:
                            yield False
                    values = bool_gen()
                elif pattern == "infinite":
                    def bool_gen():
                        while True:
                            yield bool_inf()
                    values = bool_gen()
                    use_inf = True
                else:
                    # Default to alternating
                    def bool_gen():
                        i = 0
                        while True:
                            yield bool(i % 2 == 0)
                            i += 1
                    values = bool_gen()
            
            # Create the base _bool_inf_array
            base_array = bool_inf._bool_inf_array(values=values, use_inf=use_inf, dtype=dtype)
            
            # Initialize CachedArray with the base array
            super().__init__(base_array, dtype=dtype or bool)
            self._pattern = pattern
            self._use_inf = use_inf
        
        def get_diagonal_value(self, index: int):
            """Get diagonal value at index."""
            return self[index, index]
        
        def set_diagonal_value(self, index: int, value):
            """Set diagonal value at index (cached array allows mutation)."""
            self[index, index] = bool_inf(value) if self._use_inf else bool(value)
        
        def __and__(self, other):
            """Element-wise logical AND."""
            from infinite_arrays.broadcasting import BroadcastArray
            if hasattr(other, '__getitem__'):
                return BroadcastArray(
                    lambda i: self[i, i] and other[i] if hasattr(other, '__getitem__') else self[i, i] and other,
                    self.shape()
                )
            return BroadcastArray(lambda i: self[i, i] and other, self.shape())
        
        def __or__(self, other):
            """Element-wise logical OR."""
            from infinite_arrays.broadcasting import BroadcastArray
            if hasattr(other, '__getitem__'):
                return BroadcastArray(
                    lambda i: self[i, i] or other[i] if hasattr(other, '__getitem__') else self[i, i] or other,
                    self.shape()
                )
            return BroadcastArray(lambda i: self[i, i] or other, self.shape())
        
        def __xor__(self, other):
            """Element-wise logical XOR."""
            from infinite_arrays.broadcasting import BroadcastArray
            if hasattr(other, '__getitem__'):
                return BroadcastArray(
                    lambda i: self[i, i] ^ other[i] if hasattr(other, '__getitem__') else self[i, i] ^ other,
                    self.shape()
                )
            return BroadcastArray(lambda i: self[i, i] ^ other, self.shape())
        
        def __invert__(self):
            """Element-wise logical NOT."""
            from infinite_arrays.broadcasting import BroadcastArray
            return BroadcastArray(lambda i: not self[i, i], self.shape())
        
        @classmethod
        def from_pattern(cls, pattern: str = "alternating"):
            """
            Create bool_inf_array from a pattern.
            
            Args:
                pattern: "alternating", "all_true", "all_false", or "infinite"
            """
            return cls(pattern=pattern)
        
        @classmethod
        def from_infinity(cls):
            """Create bool_inf_array filled with bool_inf(∞) values."""
            return cls(use_inf=True, pattern="infinite")
        
        @classmethod
        def from_values(cls, values):
            """Create bool_inf_array from iterable of boolean values."""
            return cls(values=values)


# Convenience functions
def create_bool_inf(value=True):
    """Create a bool_inf instance."""
    return bool_inf(value)

def create_bool_inf_array(pattern="alternating"):
    """Create a bool_inf_array instance."""
    return bool_inf.bool_inf_array.from_pattern(pattern)


if __name__ == "__main__":
    print("Testing bool_inf (boolean infinity) and bool_inf_array")
    print("=" * 60)
    
    # Test bool_inf class
    print("\n1. Creating bool_inf instances:")
    b1 = bool_inf()  # Infinite truth
    b2 = bool_inf(True)
    b3 = bool_inf(False)
    print(f"   bool_inf() = {b1}")
    print(f"   bool_inf(True) = {b2}")
    print(f"   bool_inf(False) = {b3}")
    
    print("\n2. Testing bool_inf logical operations:")
    print(f"   {b1} & True = {b1 & True}")
    print(f"   {b1} | False = {b1 | False}")
    print(f"   {b2} ^ {b3} = {b2 ^ b3}")
    print(f"   ~{b1} = {~b1}")
    
    # Test bool_inf_array with alternating pattern
    print("\n3. Creating bool_inf_array with alternating pattern:")
    bool_arr = bool_inf.bool_inf_array.from_pattern("alternating")
    print("   First 10 diagonal elements:")
    for i in range(10):
        val = bool_arr.get_diagonal_value(i)
        print(f"     [{i},{i}] = {val}")
    
    # Test bool_inf_array with all True
    print("\n4. Creating bool_inf_array with all True pattern:")
    true_arr = bool_inf.bool_inf_array.from_pattern("all_true")
    print("   First 10 diagonal elements:")
    for i in range(10):
        val = true_arr.get_diagonal_value(i)
        print(f"     [{i},{i}] = {val}")
    
    # Test bool_inf_array with infinity values
    print("\n5. Creating bool_inf_array with bool_inf(∞) values:")
    inf_arr = bool_inf.bool_inf_array.from_infinity()
    print("   First 10 diagonal elements:")
    for i in range(10):
        val = inf_arr.get_diagonal_value(i)
        print(f"     [{i},{i}] = {val}")
    
    print("\n6. Matrix representation of bool_inf_array:")
    print(bool_arr)
    
    print("\n7. Testing mutation (cached array allows setting values):")
    bool_arr.set_diagonal_value(5, False)
    print(f"   After setting [5,5] = False: {bool_arr.get_diagonal_value(5)}")