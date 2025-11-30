"""
InfiniteArrays - A Python package for representing arrays with infinite dimension sizes.

This package provides lazy infinite arrays designed to work with NumPy and other array types.
"""

from .infinity import Infinity
from .ranges import OneToInf, InfUnitRange, InfStepRange
from .arrays import InfiniteArray, Ones, Zeros, Fill
from .broadcasting import BroadcastArray
from .cache import cache, CachedArray
from .diagonal import InfiniteDiagonal
from .iqr import (
    InfiniteOperator,
    iqr_algorithm,
    iqr_spectrum,
    create_diagonal_operator,
    create_tridiagonal_operator,
)

# Import the infinity module to access the infinity instance
from . import infinity as _infinity_module

# Get the infinity instance using __dict__ (since we can't use ∞ in code)
# The infinity module stores it as '∞' in its globals()
_infinity = _infinity_module.__dict__.get('∞', _infinity_module._INFINITY)

# Make it available in this module's namespace
# We need to add it to globals() since we can't use ∞ as an identifier
globals()['∞'] = _infinity

__version__ = "0.1.0"
__all__ = [
    "∞",
    "Infinity",
    "OneToInf",
    "InfUnitRange",
    "InfStepRange",
    "InfiniteArray",
    "Ones",
    "Zeros",
    "Fill",
    "BroadcastArray",
    "cache",
    "CachedArray",
    "InfiniteDiagonal",
    "InfiniteOperator",
    "iqr_algorithm",
    "iqr_spectrum",
    "create_diagonal_operator",
    "create_tridiagonal_operator",
]
