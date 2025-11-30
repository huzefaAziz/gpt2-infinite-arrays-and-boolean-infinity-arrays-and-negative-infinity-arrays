"""
Example usage of the Infinite-dimensional QR (IQR) algorithm.

This demonstrates how to use the IQR algorithm to compute spectra of
infinite-dimensional operators, as described in:
Colbrook, M.J. & Hansen, A.C. "On the infinite-dimensional QR algorithm"
Numer. Math. 143, 17-83 (2019).
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from infinite_arrays.iqr import (
    InfiniteOperator,
    iqr_algorithm,
    iqr_spectrum,
    create_diagonal_operator,
    create_tridiagonal_operator,
)

print("=" * 70)
print("Infinite-dimensional QR Algorithm Examples")
print("=" * 70)

# Example 1: Diagonal operator
print("\n1. Diagonal Operator (eigenvalues = 1, 2, 3, ...)")
print("-" * 70)
diag_op = create_diagonal_operator(lambda i: float(i + 1))
result = iqr_algorithm(diag_op, n=20, max_iter=500, tol=1e-12)
print(f"Computed {len(result['eigenvalues'])} eigenvalues")
print(f"Iterations: {result['iterations']}, Converged: {result['converged']}")
print("First 10 eigenvalues:")
for i, ev in enumerate(result['eigenvalues'][:10]):
    print(f"  λ_{i+1} = {ev:.10f} (expected: {i+1:.1f})")

# Example 2: Tridiagonal operator (discrete Laplacian)
print("\n2. Tridiagonal Operator (discrete Laplacian-like)")
print("-" * 70)
tridiag_op = create_tridiagonal_operator(
    main_diag=lambda i: 2.0,
    upper_diag=lambda i: -1.0,
    lower_diag=lambda i: -1.0
)
result = iqr_algorithm(tridiag_op, n=50, max_iter=1000, tol=1e-10)
print(f"Computed {len(result['eigenvalues'])} eigenvalues")
print(f"Iterations: {result['iterations']}, Converged: {result['converged']}")
print("First 10 eigenvalues:")
for i, ev in enumerate(result['eigenvalues'][:10]):
    print(f"  λ_{i+1} = {ev:.10f}")

# Example 3: Custom operator
print("\n3. Custom Operator (matrix with specific structure)")
print("-" * 70)
def custom_matrix_func(i, j):
    """Custom operator: T[i,j] = 1/(1 + |i-j|)"""
    return 1.0 / (1.0 + abs(i - j))

custom_op = InfiniteOperator(custom_matrix_func)
result = iqr_algorithm(custom_op, n=30, max_iter=500, tol=1e-10)
print(f"Computed {len(result['eigenvalues'])} eigenvalues")
print(f"Iterations: {result['iterations']}, Converged: {result['converged']}")
print("First 10 eigenvalues (sorted by magnitude):")
for i, ev in enumerate(result['eigenvalues'][:10]):
    print(f"  λ_{i+1} = {ev:.10f}")

# Example 4: Adaptive spectrum computation
print("\n4. Adaptive Spectrum Computation (multiple truncation sizes)")
print("-" * 70)
spectrum_result = iqr_spectrum(
    diag_op,
    n_range=[20, 50, 100],
    max_iter=500,
    tol=1e-10,
    adaptive=True
)
print(f"Recommended truncation size: {spectrum_result['recommended_n']}")
print(f"Converged: {spectrum_result['converged']}")
print("Eigenvalues from largest truncation:")
for i, ev in enumerate(spectrum_result['eigenvalues'][:10]):
    print(f"  λ_{i+1} = {ev:.10f}")

# Example 5: Computing eigenvectors
print("\n5. Computing Eigenvectors")
print("-" * 70)
result_with_evecs = iqr_algorithm(
    diag_op,
    n=10,
    max_iter=500,
    tol=1e-12,
    compute_eigenvectors=True
)
print(f"Computed {len(result_with_evecs['eigenvalues'])} eigenvalues and eigenvectors")
print("First eigenvalue and its eigenvector:")
ev0 = result_with_evecs['eigenvalues'][0]
evec0 = result_with_evecs['eigenvectors'][:, 0]
print(f"  λ_1 = {ev0:.10f}")
print(f"  v_1 = {evec0[:5]}... (first 5 components)")

# Example 6: Hermitian operator
print("\n6. Hermitian Operator Example")
print("-" * 70)
def hermitian_matrix_func(i, j):
    """Hermitian operator: T[i,j] = exp(-|i-j|) * (1 + 0.1j if i<j else 1 - 0.1j)"""
    if i == j:
        return 1.0
    elif i < j:
        return np.exp(-abs(i - j)) * (1.0 + 0.1j)
    else:
        return np.exp(-abs(i - j)) * (1.0 - 0.1j)

hermitian_op = InfiniteOperator(hermitian_matrix_func, dtype=complex)
result = iqr_algorithm(hermitian_op, n=25, max_iter=500, tol=1e-10)
print(f"Computed {len(result['eigenvalues'])} eigenvalues")
print("First 10 eigenvalues (complex):")
for i, ev in enumerate(result['eigenvalues'][:10]):
    print(f"  λ_{i+1} = {ev:.10f}")

print("\n" + "=" * 70)
print("Examples completed!")
print("=" * 70)

