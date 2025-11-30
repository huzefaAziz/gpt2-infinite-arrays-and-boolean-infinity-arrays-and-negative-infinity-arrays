"""
Basic tests for InfiniteArrays.
"""

import pytest
import numpy as np
from infinite_arrays import (
    ∞, Ones, Zeros, Fill, InfiniteDiagonal,
    cache, CachedArray, BroadcastArray,
    OneToInf, InfUnitRange, InfStepRange
)


def test_ones():
    """Test Ones infinite array."""
    x = Ones(∞)
    assert x[0] == 1.0
    assert x[5] == 1.0
    assert x[100] == 1.0


def test_zeros():
    """Test Zeros infinite array."""
    x = Zeros(∞)
    assert x[0] == 0.0
    assert x[5] == 0.0


def test_fill():
    """Test Fill infinite array."""
    x = Fill(42, ∞)
    assert x[0] == 42
    assert x[5] == 42


def test_diagonal():
    """Test InfiniteDiagonal."""
    D = InfiniteDiagonal(range(1, 10000))
    assert D[0, 0] == 1
    assert D[1, 1] == 2
    assert D[2, 2] == 3
    assert D[0, 1] == 0.0  # Off-diagonal


def test_cache():
    """Test caching functionality."""
    x = Ones(∞)
    C = cache(x)
    assert isinstance(C, CachedArray)
    assert C[0] == 1.0
    C[0] = 3.0
    assert C[0] == 3.0
    assert C[1] == 1.0  # Other elements unchanged


def test_broadcast_array():
    """Test BroadcastArray."""
    def func(i):
        return i * 2
    
    arr = BroadcastArray(func, (∞,))
    assert arr[0] == 0
    assert arr[1] == 2
    assert arr[5] == 10


def test_one_to_inf():
    """Test OneToInf range."""
    r = OneToInf()
    assert 1 in r
    assert 0 not in r
    assert r[0] == 1  # 0-based index
    assert r[5] == 6


def test_inf_unit_range():
    """Test InfUnitRange."""
    r = InfUnitRange(5)
    assert 5 in r
    assert 4 not in r
    assert r[0] == 5
    assert r[1] == 6


def test_inf_step_range():
    """Test InfStepRange."""
    r = InfStepRange(0, 2)
    assert 0 in r
    assert 2 in r
    assert 1 not in r
    assert r[0] == 0
    assert r[1] == 2


def test_infinity_comparison():
    """Test infinity constant."""
    assert ∞ == ∞
    assert ∞ is ∞
    assert str(∞) == "∞"


def test_array_operations():
    """Test array operations."""
    x = Ones(∞)
    y = x + 2
    assert y[0] == 3.0
    
    z = x * 3
    assert z[0] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

