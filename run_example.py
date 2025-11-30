"""
Standalone script to use InfiniteArrays without installation.
This script adds the current directory to the Python path so the
infinite_arrays module can be imported directly.
"""

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

# Get the infinity symbol from the module
# Note: Python doesn't allow ∞ as an identifier in source code,
# but you can access it from the module or use None (which defaults to infinity)
_inf = infinite_arrays.__dict__['∞']

print("=" * 60)
print("InfiniteArrays - Using without installation")
print("=" * 60)

print("\n1. Creating an infinite vector of ones:")
print("-" * 60)
# Use None which defaults to infinity, or pass _inf directly
x = Ones(_inf)  # or Ones(None) works too
print(x)
print(f"x[0] = {x[0]}")
print(f"x[5] = {x[5]}")

print("\n2. Creating an infinite diagonal matrix:")
print("-" * 60)
D = InfiniteDiagonal(range(1, 10000))
print(f"D[0, 0] = {D[0, 0]}")  # Diagonal element
print(f"D[1, 1] = {D[1, 1]}")
print(f"D[0, 1] = {D[0, 1]}")  # Off-diagonal element
print(f"D[2, 2] = {D[2, 2]}")

print("\n3. Broadcasting operations:")
print("-" * 60)
# Create a custom broadcasted array
def compute_value(i):
    """Compute exp(-i) + 2 for index i (0-based)."""
    return math.exp(-(i + 1)) + 2  # i+1 to match Julia's 1-based behavior

result = BroadcastArray(compute_value, (_inf,))
print("First few values of exp(-i) + 2:")
for i in range(10):
    print(f"  result[{i}] = {result[i]:.10f}")

print("\n4. Element-wise operations:")
print("-" * 60)
x = Ones(_inf)
y = x + 2  # Add 2 to each element
print(f"y[0] = {y[0]}")  # Should be 3.0

z = x * 3  # Multiply each element by 3
print(f"z[0] = {z[0]}")  # Should be 3.0

print("\n5. Cached (mutable) arrays:")
print("-" * 60)
C = cache(Ones(_inf))
print(f"Before: C[0] = {C[0]}")
C[0] = 3.0
print(f"After: C[0] = {C[0]}")
print(f"C[1] = {C[1]}")  # Should still be 1.0

print("\n6. Other infinite array types:")
print("-" * 60)
zeros = Zeros(_inf)
print(f"zeros[0] = {zeros[0]}")

filled = Fill(42, _inf)
print(f"filled[0] = {filled[0]}")
print(f"filled[5] = {filled[5]}")

print("\n7. Accessing elements:")
print("-" * 60)
x = Ones(_inf)
print("First 10 elements:")
for i in range(10):
    print(f"  x[{i}] = {x[i]}")

print("\n" + "=" * 60)
print("Examples completed!")
print("=" * 60)

