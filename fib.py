"""
Fast recursive Fibonacci sequence using infinite_arrays library.
Uses caching for memoization to achieve O(n) time complexity.
"""

import sys
import os

# Add the infinite_arrays directory to the path so we can import it without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infinite_arrays'))

from infinite_arrays import cache, CachedArray, InfiniteArray
from infinite_arrays.infinity import Infinity
from infinite_arrays._utils import get_infinity
import numpy as np
from typing import Tuple

_INF = get_infinity()
globals()['∞'] = _INF


class FibonacciArray(InfiniteArray):
    """
    Infinite array representing the Fibonacci sequence.
    Uses recursive computation with caching for O(n) time complexity.
    """
    
    def __init__(self):
        super().__init__(dtype=np.int64)
        self._shape = (_INF,)
        # Cache for memoization
        self._cache = {}
        # Base cases
        self._cache[0] = 0
        self._cache[1] = 1
    
    def __getitem__(self, key):
        """Get Fibonacci number at index n (0-based: F(0)=0, F(1)=1, F(2)=1, ...)"""
        if isinstance(key, slice):
            return self
        
        n = key
        
        # Check cache first
        if n in self._cache:
            return self._cache[n]
        
        # Recursive computation: F(n) = F(n-1) + F(n-2)
        if n < 0:
            raise IndexError("Fibonacci sequence is only defined for non-negative indices")
        
        # Recursively compute using cached values
        fib_n = self[n - 1] + self[n - 2]
        self._cache[n] = fib_n
        return fib_n
    
    def __iter__(self):
        """Iterate over Fibonacci numbers"""
        n = 0
        while True:
            yield self[n]
            n += 1
    
    def shape(self) -> Tuple:
        return self._shape


def fibonacci(n: int) -> int:
    """
    Get the nth Fibonacci number (0-indexed).
    
    Args:
        n: Index in the Fibonacci sequence (0-based)
    
    Returns:
        The nth Fibonacci number
    
    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    fib = FibonacciArray()
    return fib[n]


def fibonacci_sequence(length: int = 100):
    """
    Generate a sequence of Fibonacci numbers.
    
    Args:
        length: Number of Fibonacci numbers to generate
    
    Returns:
        List of Fibonacci numbers
    """
    fib = FibonacciArray()
    return [fib[i] for i in range(length)]


if __name__ == "__main__":
    # Example usage
    print("Fibonacci Sequence (first 20 numbers):")
    fib = FibonacciArray()
    
    for i in range(100):
        print(f"F({i}) = {fib[i]}")
    
    print("\n" + "="*50)
    print("\nTesting large Fibonacci numbers:")
    
    # Test large values - these are fast due to caching
    test_indices = [50, 100, 200, 500]
    for n in test_indices:
        result = fibonacci(n)
        print(f"F({n}) = {result}")

