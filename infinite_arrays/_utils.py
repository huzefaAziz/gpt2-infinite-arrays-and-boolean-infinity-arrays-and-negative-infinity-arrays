"""
Internal utility functions for infinite arrays.
"""

from .infinity import Infinity


def get_infinity():
    """Get the infinity instance from the infinity module."""
    from . import infinity as _infinity_module
    # Try to get ∞ from module dict, or fall back to _INFINITY
    return _infinity_module.__dict__.get('∞', getattr(_infinity_module, '_INFINITY', None))


def is_infinity(value):
    """Check if a value is infinity."""
    return isinstance(value, Infinity) or value is get_infinity()

