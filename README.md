# Infinite Arrays Projects

A collection of fast, infinite data structures built using the `infinite_arrays` library. These projects demonstrate various applications of infinite arrays including recursive sequences, AI model integration, and specialized numeric/boolean infinity types.

## Projects Overview

This repository contains four main projects:

1. **[fib.py](#fibpy---fibonacci-sequence)** - Fast recursive Fibonacci sequence with memoization
2. **[vinf.py](#vinfpy---gpt2-infinite-arrays)** - GPT2-based infinite text/embedding arrays
3. **[iinf.py](#iinfpy---boolean-infinity-arrays)** - Boolean infinity infinite diagonal arrays
4. **[ninf.py](#ninfpy---negative-infinity-arrays)** - Negative infinity numeric arrays

## Table of Contents

- [Installation](#installation)
- [fib.py - Fibonacci Sequence](#fibpy---fibonacci-sequence)
- [vinf.py - GPT2 Infinite Arrays](#vinfpy---gpt2-infinite-arrays)
- [iinf.py - Boolean Infinity Arrays](#iinfpy---boolean-infinity-arrays)
- [ninf.py - Negative Infinity Arrays](#ninfpy---negative-infinity-arrays)
- [Common Features](#common-features)
- [Examples](#examples)
- [License](#license)

## Installation

All projects use the `infinite_arrays` library **without requiring installation**. The library is included in the `infinite_arrays/` directory and automatically added to the Python path.

### Basic Requirements

```bash
# Core dependencies (required for all projects)
pip install numpy

# For vinf.py (GPT2 arrays)
pip install torch transformers

# Optional: For better performance
pip install scipy
```

### No Installation Required

The projects are designed to work out-of-the-box. Simply ensure:
- Python 3.7+
- The `infinite_arrays/` directory is present
- Required packages are installed (see above)

## fib.py - Fibonacci Sequence

Fast recursive Fibonacci sequence implementation with O(n) time complexity using memoization.

### Features

- ✅ Infinite Fibonacci sequence
- ✅ O(n) time complexity with caching
- ✅ Recursive computation with memoization
- ✅ No installation required

### Usage

```python
from fib import FibonacciArray, fibonacci, fibonacci_sequence

# Create Fibonacci array
fib = FibonacciArray()

# Get specific Fibonacci number
print(fib[10])  # 55
print(fib[100])  # Very large number, fast due to caching

# Use convenience function
print(fibonacci(10))  # 55

# Generate sequence
seq = fibonacci_sequence(20)
print(seq)  # [0, 1, 1, 2, 3, 5, 8, 13, ...]

# Iterate infinitely
for i, f in enumerate(fib):
    print(f"F({i}) = {f}")
    if i >= 10:
        break
```

### Example Output

```
F(0) = 0
F(1) = 1
F(2) = 1
F(3) = 2
F(4) = 3
F(5) = 5
F(6) = 8
F(7) = 13
F(8) = 21
F(9) = 34
F(10) = 55
```

## vinf.py - GPT2 Infinite Arrays

Infinite arrays powered by GPT2 language model for generating infinite sequences of tokens, text, or embeddings.

### Features

- ✅ GPT2-powered infinite text generation
- ✅ Multiple modes: tokens, text, embeddings
- ✅ Infinite diagonal matrix representation
- ✅ Vector-style access for embeddings
- ✅ Automatic caching for fast access

### Requirements

```bash
pip install torch transformers
```

### Usage

```python
from vinf import gpt2, VInfiniteArray

# Create GPT2 infinite diagonal array (token mode)
gpt2_tokens = gpt2(model_name="gpt2", prompt="The future of AI is", mode="token")
print(gpt2_tokens[0, 0])  # Token ID

# Create GPT2 infinite diagonal array (text mode)
gpt2_text = gpt2(model_name="gpt2", prompt="The future of AI is", mode="text")
print(gpt2_text[0, 0])  # Text token

# Generate text sequence
generated = gpt2_text.generate(30)
print(generated)  # Generated text

# Create vector infinite array (embeddings)
vinf = VInfiniteArray(model_name="gpt2", prompt="Hello world")
embedding = vinf[0]  # Get embedding vector
print(f"Embedding shape: {embedding.shape}")  # (768,) for GPT2
```

### Modes

1. **Token Mode** (`mode="token"`): Returns token IDs
2. **Text Mode** (`mode="text"`): Returns text strings
3. **Embedding Mode** (`mode="embedding"`): Returns embedding vectors

### Example Output

```
First 10 diagonal elements (text):
  [0,0] = 'The'
  [1,1] = ' future'
  [2,2] = ' of'
  [3,3] = ' AI'
  [4,4] = ' is'
  ...
```

## iinf.py - Boolean Infinity Arrays

Infinite diagonal arrays with boolean infinity values, supporting logical operations and various generation patterns.

### Features

- ✅ Boolean infinity types (∞T, ∞F)
- ✅ Infinite diagonal matrices
- ✅ Cached arrays with mutation support
- ✅ Multiple generation patterns
- ✅ Element-wise logical operations
- ✅ Comparison operators (==, <, >, <=, >=)

### Usage

```python
from iinf import bool_inf, create_bool_inf_array

# Create boolean infinity value
b = bool_inf()  # Returns ∞T (infinite truth)

# Create infinite diagonal array with alternating pattern
bool_arr = bool_inf.bool_inf_array.from_pattern("alternating")
print(bool_arr.get_diagonal_value(0))  # True
print(bool_arr.get_diagonal_value(1))  # False

# Create array with all True
true_arr = bool_inf.bool_inf_array.from_pattern("all_true")

# Create array with infinity values
inf_arr = bool_inf.bool_inf_array.from_infinity()

# Mutate values (cached arrays allow setting)
bool_arr.set_diagonal_value(5, False)

# Logical operations
result = bool_arr & true_arr  # Element-wise AND
result = bool_arr | true_arr  # Element-wise OR
result = ~bool_arr           # Element-wise NOT

# Comparison operations
b1 = bool_inf()      # ∞T
b2 = bool_inf(True)  # True
b3 = bool_inf(False) # False
b4 = bool_inf(False)
b4._is_inf = True    # ∞F

# Ordering: ∞F < False < True < ∞T
print(b3 < b2)       # True (False < True)
print(b4 < b3)       # True (∞F < False)
print(b1 > b2)       # True (∞T > True)
print(b2 == b2)      # True (equality)
print(b3 <= b2)      # True (False <= True)
print(b1 >= b2)      # True (∞T >= True)

# Compare with regular bool
print(b3 < True)     # True
print(b1 > False)    # True
```

### Patterns

- `"alternating"`: True, False, True, False, ...
- `"all_true"`: All True values
- `"all_false"`: All False values
- `"infinite"`: All `bool_inf(∞)` values

### Comparison Operators

The `bool_inf` class supports all comparison operators with a well-defined ordering:

**Ordering**: `∞F < False < True < ∞T`

- **Equality (`==`)**: Compares both value and infinity state
- **Less than (`<`)**: Returns `True` if left operand is less than right
- **Greater than (`>`)**: Returns `True` if left operand is greater than right
- **Less or equal (`<=`)**: Combination of `<` and `==`
- **Greater or equal (`>=`)**: Combination of `>` and `==`

All comparison operators work with:
- `bool_inf` instances
- Regular Python `bool` values

### Example Output

```
First 10 diagonal elements:
  [0,0] = True
  [1,1] = False
  [2,2] = True
  [3,3] = False
  [4,4] = True
  ...
```

## ninf.py - Negative Infinity Arrays

Infinite diagonal arrays with negative infinity numeric values, supporting arithmetic operations and sequences.

### Features

- ✅ Negative infinity numeric type (`-∞`)
- ✅ Infinite diagonal matrices
- ✅ Arithmetic sequence generation
- ✅ Arithmetic operations support

### Usage

```python
from ninf import ninf, create_ninf_array

# Create negative infinity value
n = ninf()  # Returns -∞

# Create infinite diagonal array with ninf(-∞) values
ninf_arr = ninf.ninf_array.from_ninf()

# Create array from arithmetic sequence
seq_arr = ninf.ninf_array.from_sequence(start=0, step=1)

# Create array with ninf sequence
ninf_seq_arr = ninf.ninf_array.from_ninf_sequence(start=10, step=-2)

# Arithmetic operations
result = ninf() + 5    # -∞ + 5 = -∞
result = ninf() * 2    # -∞ * 2 = -∞
result = ninf() / 3    # -∞ / 3 = -∞
```

### Factory Methods

- `from_ninf()`: All diagonal elements are `ninf(-∞)`
- `from_sequence(start, step)`: Arithmetic sequence
- `from_ninf_sequence(start, step)`: Sequence of `ninf` values

### Example Output

```
First 10 diagonal elements:
  [0,0] = -∞
  [1,1] = -∞
  [2,2] = -∞
  ...
```

## Common Features

All projects share these common features:

### Infinite Arrays

- **Lazy Evaluation**: Values computed on-demand
- **Caching**: Fast repeated access
- **Memory Efficient**: Only stores computed values
- **No Installation**: Works without installing infinite_arrays

### Diagonal Matrix Access

All diagonal array projects support:

```python
# Single index (diagonal element)
arr[5]  # Element at [5, 5]

# 2D index (row, col)
arr[5, 5]  # Diagonal element
arr[5, 3]  # Off-diagonal (returns 0 or default)
```

### Iteration

```python
# Iterate over elements
for i, value in enumerate(array):
    print(f"Element {i}: {value}")
    if i >= 10:
        break
```

## Examples

### Complete Example: Fibonacci

```python
from fib import FibonacciArray

fib = FibonacciArray()

# Print first 20 Fibonacci numbers
for i in range(20):
    print(f"F({i}) = {fib[i]}")

# Compute large Fibonacci numbers efficiently
print(f"F(100) = {fib[100]}")
print(f"F(500) = {fib[500]}")
```

### Complete Example: GPT2 Arrays

```python
from vinf import gpt2

# Generate infinite text sequence
gpt2_text = gpt2(
    model_name="gpt2",
    prompt="The future of AI",
    mode="text"
)

# Get first 10 tokens
for i in range(10):
    print(f"Token {i}: {gpt2_text[i, i]}")

# Generate continuous text
text = gpt2_text.generate(50)
print(f"\nGenerated text: {text}")
```

### Complete Example: Boolean Arrays

```python
from iinf import bool_inf

# Create alternating boolean array
arr = bool_inf.bool_inf_array.from_pattern("alternating")

# Access elements
print(arr[0, 0])  # True
print(arr[1, 1])  # False

# Mutate
arr.set_diagonal_value(10, True)

# Logical operations
all_true = bool_inf.bool_inf_array.from_pattern("all_true")
result = arr & all_true

# Comparison operations
b1 = bool_inf()      # ∞T
b2 = bool_inf(True)   # True
b3 = bool_inf(False)  # False

# Demonstrate ordering: ∞F < False < True < ∞T
print(f"{b3} < {b2} = {b3 < b2}")  # False < True = True
print(f"{b1} > {b2} = {b1 > b2}")  # ∞T > True = True
print(f"{b2} == {b2} = {b2 == b2}")  # True == True = True
```

### Complete Example: Negative Infinity Arrays

```python
from ninf import ninf

# Create negative infinity
n = ninf()
print(n)  # -∞

# Create array with arithmetic sequence
arr = ninf.ninf_array.from_sequence(start=0, step=1)

# Access elements
for i in range(10):
    print(f"Element {i}: {arr.get_diagonal_value(i)}")
```

## Performance

All implementations use lazy evaluation and caching for optimal performance:

- **Fibonacci**: O(n) time complexity with O(n) space
- **GPT2 Arrays**: Tokens cached after first generation
- **Boolean Arrays**: Cached array allows O(1) access
- **Negative Infinity**: Efficient arithmetic operations

## Requirements Summary

| Project | Required Packages |
|---------|------------------|
| `fib.py` | `numpy` |
| `vinf.py` | `numpy`, `torch`, `transformers` |
| `iinf.py` | `numpy` |
| `ninf.py` | `numpy` |

## File Structure

```
.
├── README.md
├── fib.py                 # Fibonacci sequence
├── vinf.py                # GPT2 infinite arrays
├── iinf.py                # Boolean infinity arrays
├── ninf.py                # Negative infinity arrays
└── infinite_arrays/       # Library (no installation needed)
    ├── __init__.py
    ├── arrays.py
    ├── cache.py
    ├── diagonal.py
    └── ...
```

## Contributing

Contributions are welcome! Each project is self-contained and can be extended independently.

## License

This project uses the `infinite_arrays` library. Please refer to the library's license for usage terms.

## Acknowledgments

- Built using the `infinite_arrays` library
- GPT2 integration powered by Hugging Face Transformers
- Infinite array concepts inspired by functional programming and lazy evaluation

---

**Note**: All projects are designed to work without installing the `infinite_arrays` library. The library directory is automatically added to the Python path at runtime.
