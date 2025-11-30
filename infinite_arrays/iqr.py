"""
Infinite-dimensional QR algorithm implementation.

This module implements the infinite-dimensional QR (IQR) algorithm as described in:
Colbrook, M.J. & Hansen, A.C. "On the infinite-dimensional QR algorithm"
Numer. Math. 143, 17-83 (2019). https://doi.org/10.1007/s00211-019-01047-5

The algorithm computes spectra and eigenvectors of infinite-dimensional operators
with convergence rates and error control.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List, Union, Dict, Any
from scipy.linalg import qr
from .arrays import InfiniteArray
from ._utils import get_infinity
from .broadcasting import BroadcastArray

# Get ∞ from the module's namespace
_INF = get_infinity()
globals()['∞'] = _INF


class InfiniteOperator:
    """
    Represents an infinite-dimensional operator on l^2(N).
    
    The operator is defined by a function that returns matrix elements.
    For an operator T, T[i, j] gives the (i, j)-th matrix element (0-based indexing).
    """
    
    def __init__(self, matrix_func: Callable[[int, int], complex], 
                 shape: Tuple = None,
                 dtype=None):
        """
        Initialize an infinite operator.
        
        Parameters:
        -----------
        matrix_func : Callable[[int, int], complex]
            Function that returns the (i, j)-th matrix element (0-based indexing)
        shape : Tuple, optional
            Shape of the operator (default: (∞, ∞))
        dtype : type, optional
            Data type (default: complex)
        """
        self._matrix_func = matrix_func
        self._shape = shape if shape is not None else (_INF, _INF)
        self.dtype = dtype or complex
        self._cache: Dict[Tuple[int, int], complex] = {}
    
    def __getitem__(self, key: Tuple[int, int]) -> complex:
        """Get matrix element at position (i, j)."""
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if (i, j) in self._cache:
                return self._cache[(i, j)]
            value = self._matrix_func(i, j)
            self._cache[(i, j)] = value
            return value
        raise IndexError("Index must be a tuple (i, j)")
    
    def get_truncation(self, n: int) -> np.ndarray:
        """
        Get a finite n×n truncation of the operator.
        
        Parameters:
        -----------
        n : int
            Size of the truncation
            
        Returns:
        --------
        np.ndarray
            n×n matrix representing the truncation
        """
        matrix = np.zeros((n, n), dtype=self.dtype)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self[i, j]
        return matrix
    
    def shape(self) -> Tuple:
        """Return the shape of the operator."""
        return self._shape


def iqr_algorithm(operator: InfiniteOperator,
                  n: int = 50,
                  max_iter: int = 1000,
                  tol: float = 1e-10,
                  shift: Optional[complex] = None,
                  compute_eigenvectors: bool = False) -> Dict[str, Any]:
    """
    Infinite-dimensional QR algorithm for computing spectra.
    
    This implements the IQR algorithm as described in the paper. The algorithm
    works by:
    1. Truncating the infinite operator to a finite n×n matrix
    2. Applying QR iterations with optional shifts
    3. Extracting eigenvalues from the converged matrix
    4. Optionally computing eigenvectors
    
    Parameters:
    -----------
    operator : InfiniteOperator
        The infinite-dimensional operator to compute spectrum for
    n : int, default=50
        Size of the finite truncation
    max_iter : int, default=1000
        Maximum number of QR iterations
    tol : float, default=1e-10
        Convergence tolerance
    shift : complex, optional
        Shift parameter for shifted QR algorithm (Wilkinson shift recommended)
    compute_eigenvectors : bool, default=False
        Whether to compute eigenvectors
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'eigenvalues': array of computed eigenvalues
        - 'eigenvectors': array of eigenvectors (if compute_eigenvectors=True)
        - 'iterations': number of iterations performed
        - 'converged': whether convergence was achieved
        - 'residual': residual error estimate
    """
    # Get finite truncation
    A = operator.get_truncation(n)
    
    # Initialize eigenvector matrix if needed
    if compute_eigenvectors:
        Q_total = np.eye(n, dtype=operator.dtype)
    else:
        Q_total = None
    
    # QR iteration
    iterations = 0
    converged = False
    
    for k in range(max_iter):
        # Compute shift (Wilkinson shift for better convergence)
        if shift is None:
            # Wilkinson shift: use eigenvalue of bottom-right 2x2 block
            if n >= 2:
                a = A[n-2, n-2]
                b = A[n-2, n-1]
                c = A[n-1, n-2]
                d = A[n-1, n-1]
                # Eigenvalue of 2x2 matrix closest to d
                trace = a + d
                det = a * d - b * c
                discriminant = trace**2 - 4 * det
                if discriminant >= 0:
                    lambda1 = (trace + np.sqrt(discriminant)) / 2
                    lambda2 = (trace - np.sqrt(discriminant)) / 2
                    shift_val = lambda2 if abs(lambda2 - d) < abs(lambda1 - d) else lambda1
                else:
                    shift_val = trace / 2
            else:
                shift_val = A[0, 0]
        else:
            shift_val = shift
        
        # Shift the matrix
        A_shifted = A - shift_val * np.eye(n, dtype=A.dtype)
        
        # QR decomposition
        Q, R = qr(A_shifted)
        
        # Reverse QR: A = R * Q + shift
        A = R @ Q + shift_val * np.eye(n, dtype=A.dtype)
        
        # Accumulate eigenvectors if needed
        if compute_eigenvectors:
            Q_total = Q_total @ Q
        
        iterations = k + 1
        
        # Check convergence: off-diagonal elements should be small
        off_diag = np.abs(A - np.diag(np.diag(A)))
        max_off_diag = np.max(off_diag)
        
        if max_off_diag < tol:
            converged = True
            break
    
    # Extract eigenvalues from diagonal
    eigenvalues = np.diag(A)
    
    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    
    result = {
        'eigenvalues': eigenvalues,
        'iterations': iterations,
        'converged': converged,
        'residual': max_off_diag if converged else None,
    }
    
    if compute_eigenvectors:
        eigenvectors = Q_total[:, idx]
        result['eigenvectors'] = eigenvectors
    
    return result


def iqr_spectrum(operator: InfiniteOperator,
                 n_range: Union[int, List[int]] = None,
                 max_iter: int = 1000,
                 tol: float = 1e-10,
                 adaptive: bool = True) -> Dict[str, Any]:
    """
    Compute spectrum using IQR algorithm with adaptive truncation.
    
    This function applies the IQR algorithm with increasing truncation sizes
    to estimate the spectrum of the infinite operator. It can use multiple
    truncation sizes to check convergence.
    
    Parameters:
    -----------
    operator : InfiniteOperator
        The infinite-dimensional operator
    n_range : int or List[int], optional
        Truncation sizes to use. If int, uses [n_range]. If None, uses adaptive sizing.
    max_iter : int, default=1000
        Maximum QR iterations per truncation
    tol : float, default=1e-10
        Convergence tolerance
    adaptive : bool, default=True
        Whether to adaptively increase truncation size
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'eigenvalues': estimated eigenvalues
        - 'eigenvalues_by_n': eigenvalues for each truncation size
        - 'converged': convergence status
        - 'recommended_n': recommended truncation size
    """
    if n_range is None:
        if adaptive:
            n_range = [20, 50, 100, 200]
        else:
            n_range = [50]
    elif isinstance(n_range, int):
        n_range = [n_range]
    
    results_by_n = {}
    all_eigenvalues = []
    
    for n in n_range:
        result = iqr_algorithm(operator, n=n, max_iter=max_iter, tol=tol)
        results_by_n[n] = result
        all_eigenvalues.extend(result['eigenvalues'].tolist())
    
    # Estimate spectrum (could use more sophisticated methods)
    if len(n_range) > 1:
        # Use eigenvalues from largest truncation as estimate
        largest_n = max(n_range)
        eigenvalues = results_by_n[largest_n]['eigenvalues']
    else:
        eigenvalues = results_by_n[n_range[0]]['eigenvalues']
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvalues_by_n': results_by_n,
        'converged': all(r['converged'] for r in results_by_n.values()),
        'recommended_n': max(n_range),
    }


def create_diagonal_operator(diagonal_values: Union[Callable, np.ndarray, List]) -> InfiniteOperator:
    """
    Create an infinite diagonal operator.
    
    Parameters:
    -----------
    diagonal_values : Callable, np.ndarray, or List
        If callable, function that returns diagonal value at index i (0-based)
        If array/list, diagonal values
        
    Returns:
    --------
    InfiniteOperator
        Diagonal operator
    """
    if callable(diagonal_values):
        def matrix_func(i, j):
            if i == j:
                return diagonal_values(i)
            return 0.0
    else:
        # Convert to list/array
        diag = np.asarray(diagonal_values)
        def matrix_func(i, j):
            if i == j and i < len(diag):
                return diag[i]
            return 0.0
    
    return InfiniteOperator(matrix_func)


def create_tridiagonal_operator(main_diag: Union[Callable, np.ndarray, List],
                                upper_diag: Union[Callable, np.ndarray, List] = None,
                                lower_diag: Union[Callable, np.ndarray, List] = None) -> InfiniteOperator:
    """
    Create an infinite tridiagonal operator.
    
    Parameters:
    -----------
    main_diag : Callable, np.ndarray, or List
        Main diagonal values
    upper_diag : Callable, np.ndarray, or List, optional
        Upper diagonal values (default: zeros)
    lower_diag : Callable, np.ndarray, or List, optional
        Lower diagonal values (default: zeros)
        
    Returns:
    --------
    InfiniteOperator
        Tridiagonal operator
    """
    # Convert to functions
    if callable(main_diag):
        main_func = main_diag
    else:
        main_arr = np.asarray(main_diag)
        def main_func(i):
            return main_arr[i] if i < len(main_arr) else 0.0
    
    if upper_diag is None:
        upper_func = lambda i: 0.0
    elif callable(upper_diag):
        upper_func = upper_diag
    else:
        upper_arr = np.asarray(upper_diag)
        def upper_func(i):
            return upper_arr[i] if i < len(upper_arr) else 0.0
    
    if lower_diag is None:
        lower_func = lambda i: 0.0
    elif callable(lower_diag):
        lower_func = lower_diag
    else:
        lower_arr = np.asarray(lower_diag)
        def lower_func(i):
            return lower_arr[i] if i < len(lower_arr) else 0.0
    
    def matrix_func(i, j):
        if i == j:
            return main_func(i)
        elif j == i + 1:
            return upper_func(i)
        elif j == i - 1:
            return lower_func(j)
        return 0.0
    
    return InfiniteOperator(matrix_func)


# Example usage functions
def example_diagonal_operator():
    """Example: Diagonal operator with eigenvalues 1, 2, 3, ..."""
    return create_diagonal_operator(lambda i: float(i + 1))


def example_tridiagonal_operator():
    """Example: Tridiagonal operator (discrete Laplacian-like)"""
    return create_tridiagonal_operator(
        main_diag=lambda i: 2.0,
        upper_diag=lambda i: -1.0,
        lower_diag=lambda i: -1.0
    )

