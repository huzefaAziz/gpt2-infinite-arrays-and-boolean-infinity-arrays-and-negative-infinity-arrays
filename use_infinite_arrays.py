"""
How to use InfiniteArrays in your own Python file without installation.
Just add these lines at the top of your file:
"""

import sys
import os
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the current directory to Python path (where infinite_arrays folder is located)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import infinite_arrays - no installation needed!
import infinite_arrays
from infinite_arrays import Ones, InfiniteDiagonal, cache, BroadcastArray, Zeros, Fill

# Get the infinity symbol from the module (optional - you can also use None)
# Note: Python doesn't allow ∞ as an identifier in source code
_inf = infinite_arrays.__dict__['∞']

# Example usage:
if __name__ == "__main__":
    # Create infinite vector of ones (you can use None or _inf - both work!)
    x = Ones(None)  # None defaults to infinity
    print(f"x[0] = {x[0]}")
    print(f"x[5] = {x[5]}")
    
    # Create infinite diagonal matrix
    D = InfiniteDiagonal(range(1, 10000))
    print(f"D[0, 0] = {D[0, 0]}")
    print(f"D[1, 1] = {D[1, 1]}")
    print(f"D[0, 1] = {D[0, 1]}")  # Off-diagonal is 0
    
    # Create cached (mutable) array
    C = cache(Ones(None))
    C[0] = 3.0
    print(f"C[0] = {C[0]}")
    print(f"C[1] = {C[1]}")  # Still 1.0
    
    # Other examples
    zeros = Zeros(None)
    print(f"zeros[0] = {zeros[0]}")
    
    filled = Fill(42, None)
    print(f"filled[0] = {filled[0]}")
    
    print("\n[OK] InfiniteArrays is working without installation!")

