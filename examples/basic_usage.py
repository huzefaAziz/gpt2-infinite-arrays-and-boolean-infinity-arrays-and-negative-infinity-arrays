"""
Basic usage examples for InfiniteArrays.
"""

import numpy as np
import math
from infinite_arrays import ∞, Ones, InfiniteDiagonal, cache, BroadcastArray, Zeros, Fill

print("=" * 60)
print("InfiniteArrays - Basic Usage Examples")
print("=" * 60)

print("\n1. Creating an infinite vector of ones:")
print("-" * 60)
x = Ones(∞)
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

result = BroadcastArray(compute_value, (∞,))
print("First few values of exp(-i) + 2:")
for i in range(10):
    print(f"  result[{i}] = {result[i]:.10f}")

print("\n4. Element-wise operations:")
print("-" * 60)
x = Ones(∞)
y = x + 2  # Add 2 to each element
print(f"y[0] = {y[0]}")  # Should be 3.0

z = x * 3  # Multiply each element by 3
print(f"z[0] = {z[0]}")  # Should be 3.0

print("\n5. Cached (mutable) arrays:")
print("-" * 60)
C = cache(Ones(∞))
print(f"Before: C[0] = {C[0]}")
C[0] = 3.0
print(f"After: C[0] = {C[0]}")
print(f"C[1] = {C[1]}")  # Should still be 1.0

print("\n6. Other infinite array types:")
print("-" * 60)
zeros = Zeros(∞)
print(f"zeros[0] = {zeros[0]}")

filled = Fill(42, ∞)
print(f"filled[0] = {filled[0]}")
print(f"filled[5] = {filled[5]}")

print("\n7. Accessing elements:")
print("-" * 60)
x = Ones(∞)
print("First 10 elements:")
for i in range(10):
    print(f"  x[{i}] = {x[i]}")

print("\n" + "=" * 60)
print("Examples completed!")
print("=" * 60)

