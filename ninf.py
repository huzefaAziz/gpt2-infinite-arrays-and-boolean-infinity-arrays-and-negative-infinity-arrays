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
import infinite_arrays
from infinite_arrays import Ones, InfiniteDiagonal, cache, BroadcastArray, Zeros, Fill

class ninf(int):
    """
    Negative infinity or numeric infinity as an integer-like value.
    Represents negative infinity in integer operations.
    """
    
    _SENTINEL = object()
    
    def __new__(cls, value=_SENTINEL):
        """Create a new ninf instance."""
        if value is cls._SENTINEL:
            # Default: create negative infinity
            instance = super().__new__(cls, -sys.maxsize)
            instance._value = -sys.maxsize
            instance._is_inf = True
        else:
            instance = super().__new__(cls, value)
            instance._value = int(value)
            instance._is_inf = False
        return instance
    
    def __repr__(self):
        return "-∞" if getattr(self, '_is_inf', False) else f"ninf({self._value})"
    
    def __str__(self):
        return "-∞" if getattr(self, '_is_inf', False) else str(self._value)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self._is_inf:
                if other == float('inf') or other == sys.maxsize:
                    raise ValueError("Cannot add +∞ to -∞")
                return ninf()  # -∞ + any finite = -∞
            return ninf(self._value + other)
        return super().__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if self._is_inf:
                if other == float('-inf'):
                    raise ValueError("Cannot subtract -∞ from -∞")
                return ninf()  # -∞ - any finite = -∞
            return ninf(self._value - other)
        return super().__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return 0
            if self._is_inf:
                if other < 0:
                    # -∞ * negative = +∞, but we return positive max
                    return sys.maxsize
                return ninf()  # -∞ * positive = -∞
            result = self._value * other
            return ninf(result)
        return super().__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            if self._is_inf:
                return ninf()  # -∞ / any finite = -∞
            return ninf(self._value / other)
        return super().__truediv__(other)
    
    @classmethod
    def create_sequence(cls, start: int = 0, step: int = 1, use_ninf: bool = False):
        """
        Create a sequence generator for ninf_array.
        
        Args:
            start: Starting value
            step: Step size
            use_ninf: If True, use ninf(-∞) for all values
        """
        if use_ninf:
            def generator():
                while True:
                    yield ninf()
            return generator()
        else:
            def generator():
                i = start
                while True:
                    yield i
                    i += step
            return generator()
    
    class ninf_array(InfiniteDiagonal):
        """
        Infinite diagonal array with numeric infinity values.
        Can use ninf values or generate sequences based on ninf.
        """
        
        def __init__(self, 
                     values=None, 
                     use_ninf: bool = False,
                     start: int = 0,
                     step: int = 1,
                     dtype=None):
            """
            Initialize ninf infinite diagonal array.
            
            Args:
                values: Iterable of values (if None, generates sequence)
                use_ninf: If True, all diagonal values are ninf(-∞)
                start: Starting value for sequence (if values is None)
                step: Step size for sequence (if values is None)
                dtype: Data type (default: int for ninf, float otherwise)
            """
            if values is None:
                if use_ninf:
                    # Create infinite sequence of ninf values
                    def ninf_gen():
                        while True:
                            yield ninf()
                    values = ninf_gen()
                    dtype = dtype or int
                else:
                    # Create arithmetic sequence starting from start
                    def seq_gen():
                        i = start
                        while True:
                            yield i
                            i += step
                    values = seq_gen()
                    dtype = dtype or (int if isinstance(start, int) else float)
            elif use_ninf:
                # Convert all values to ninf
                def convert_gen():
                    for v in values:
                        yield ninf(v) if not isinstance(v, ninf) else v
                values = convert_gen()
                dtype = dtype or int
            
            # Initialize InfiniteDiagonal with the sequence
            super().__init__(values, dtype=dtype)
            self._use_ninf = use_ninf
            self._start = start
            self._step = step
        
        def __repr__(self):
            # Override to show ninf representation
            rows = []
            n = 10  # Number of rows/cols to show
            for i in range(n):
                row = []
                for j in range(n):
                    if i == j:
                        val = self._get_value(i + 1)
                        if isinstance(val, ninf):
                            row.append(str(val))
                        else:
                            row.append(str(val))
                    elif j == n - 1:
                        row.append("…")
                        break
                    else:
                        row.append("⋅")
                rows.append("  ".join(row))
            
            return f"{self.__class__.__name__}{self.shape()}:\n" + "\n".join(rows) + "\n⋮"
        
        def get_diagonal_value(self, index: int):
            """Get diagonal value at index."""
            return self._get_value(index + 1)  # Convert 0-based to 1-based
        
        @classmethod
        def from_ninf(cls, length=None):
            """
            Create ninf_array filled with ninf(-∞) values.
            
            Args:
                length: Not used (array is infinite), kept for API consistency
            """
            return cls(use_ninf=True)
        
        @classmethod
        def from_sequence(cls, start: int = 0, step: int = 1):
            """
            Create ninf_array from arithmetic sequence.
            
            Args:
                start: Starting value
                step: Step size
            """
            return cls(use_ninf=False, start=start, step=step)
        
        @classmethod
        def from_ninf_sequence(cls, start: int = 0, step: int = 1):
            """
            Create ninf_array where diagonal values are ninf(start + n*step).
            
            Args:
                start: Starting value (will be wrapped in ninf)
                step: Step size
            """
            def ninf_seq_gen():
                i = start
                while True:
                    yield ninf(i)
                    i += step
            return cls(values=ninf_seq_gen())


# Convenience functions
def create_ninf():
    """Create a ninf(-∞) instance."""
    return ninf()

def create_ninf_array(use_ninf=False, start=0, step=1):
    """Create a ninf_array instance."""
    return ninf.ninf_array(use_ninf=use_ninf, start=start, step=step)


if __name__ == "__main__":
    print("Testing ninf (negative infinity) and ninf_array")
    print("=" * 60)
    
    # Test ninf class
    print("\n1. Creating ninf(-∞) instance:")
    n = ninf()
    print(f"   ninf() = {n}")
    print(f"   Repr: {repr(n)}")
    
    print("\n2. Testing ninf arithmetic:")
    print(f"   ninf() + 5 = {n + 5}")
    print(f"   ninf() * 2 = {n * 2}")
    print(f"   ninf() / 3 = {n / 3}")
    
    # Test ninf_array with ninf values
    print("\n3. Creating ninf_array with ninf(-∞) values:")
    ninf_arr = ninf.ninf_array.from_ninf()
    print("   First 10 diagonal elements:")
    for i in range(10):
        val = ninf_arr.get_diagonal_value(i)
        print(f"     [{i},{i}] = {val}")
    
    # Test ninf_array with arithmetic sequence
    print("\n4. Creating ninf_array from sequence (start=0, step=1):")
    seq_arr = ninf.ninf_array.from_sequence(start=0, step=1)
    print("   First 10 diagonal elements:")
    for i in range(10):
        val = seq_arr.get_diagonal_value(i)
        print(f"     [{i},{i}] = {val}")
    
    # Test ninf_array with ninf sequence
    print("\n5. Creating ninf_array from ninf sequence (start=10, step=-2):")
    ninf_seq_arr = ninf.ninf_array.from_ninf_sequence(start=10, step=-2)
    print("   First 10 diagonal elements:")
    for i in range(10):
        val = ninf_seq_arr.get_diagonal_value(i)
        print(f"     [{i},{i}] = {val}")
    
    print("\n6. Matrix representation of ninf_array:")
    print(ninf_arr)
    