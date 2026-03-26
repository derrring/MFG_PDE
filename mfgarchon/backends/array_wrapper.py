"""
Backend Array Wrapper for Transparent Acceleration.

This module provides a high-level wrapper that makes backend arrays behave like
NumPy arrays, allowing existing solvers to benefit from GPU/JIT acceleration
WITHOUT modifying their internal code.

Strategy:
- Wrap backend arrays with NumPy-compatible interface
- Intercept operations and dispatch to backend
- Automatic conversion at boundaries (input/output)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .base_backend import BaseBackend


class BackendArray:
    """
    Wrapper that makes backend arrays behave like NumPy arrays.

    This allows existing NumPy-based code to transparently use
    accelerated backends (JAX/Numba/Torch) without modifications.

    Example:
        ```python
        backend = create_backend("jax")
        wrapper = BackendArrayWrapper(backend)

        # Existing NumPy code works unchanged
        x = wrapper.array([1, 2, 3])  # Actually JAX array
        y = x + 2  # JAX operation
        z = np.sin(x)  # Dispatched to JAX

        # Automatic conversion back to NumPy
        result = wrapper.to_numpy(z)
        ```
    """

    def __init__(self, data, backend: BaseBackend):
        """Initialize backend array wrapper."""
        self._data = data
        self._backend = backend
        self._module = backend.array_module

    @property
    def data(self):
        """Access underlying backend array."""
        return self._data

    @property
    def shape(self):
        """Array shape."""
        return self._data.shape

    @property
    def dtype(self):
        """Array dtype."""
        return self._data.dtype

    @property
    def ndim(self):
        """Number of dimensions."""
        return self._data.ndim

    @property
    def size(self):
        """Total number of elements."""
        return self._data.size

    # Arithmetic operations (automatically dispatch to backend)
    def __add__(self, other):
        other_data = other._data if isinstance(other, BackendArray) else other
        return BackendArray(self._data + other_data, self._backend)

    def __sub__(self, other):
        other_data = other._data if isinstance(other, BackendArray) else other
        return BackendArray(self._data - other_data, self._backend)

    def __mul__(self, other):
        other_data = other._data if isinstance(other, BackendArray) else other
        return BackendArray(self._data * other_data, self._backend)

    def __truediv__(self, other):
        other_data = other._data if isinstance(other, BackendArray) else other
        return BackendArray(self._data / other_data, self._backend)

    def __pow__(self, other):
        other_data = other._data if isinstance(other, BackendArray) else other
        return BackendArray(self._data**other_data, self._backend)

    # Reverse operations
    def __radd__(self, other):
        return BackendArray(other + self._data, self._backend)

    def __rsub__(self, other):
        return BackendArray(other - self._data, self._backend)

    def __rmul__(self, other):
        return BackendArray(other * self._data, self._backend)

    def __rtruediv__(self, other):
        return BackendArray(other / self._data, self._backend)

    # Indexing
    def __getitem__(self, key):
        return BackendArray(self._data[key], self._backend)

    def __setitem__(self, key, value):
        value_data = value._data if isinstance(value, BackendArray) else value
        self._data[key] = value_data

    # Array methods
    def copy(self):
        """Create a copy of the array."""
        if hasattr(self._data, "copy"):
            return BackendArray(self._data.copy(), self._backend)
        return BackendArray(self._module.array(self._data), self._backend)

    def reshape(self, *shape):
        """Reshape array."""
        return BackendArray(self._data.reshape(*shape), self._backend)

    def flatten(self):
        """Flatten array."""
        return BackendArray(self._data.flatten(), self._backend)

    def sum(self, axis=None):
        """Sum along axis."""
        result = self._module.sum(self._data, axis=axis)
        if axis is None or np.isscalar(result):
            return result  # Scalar, return as-is
        return BackendArray(result, self._backend)

    def mean(self, axis=None):
        """Mean along axis."""
        result = self._module.mean(self._data, axis=axis)
        if axis is None or np.isscalar(result):
            return result
        return BackendArray(result, self._backend)

    def max(self, axis=None):
        """Maximum along axis."""
        result = self._module.max(self._data, axis=axis)
        if axis is None or np.isscalar(result):
            return result
        return BackendArray(result, self._backend)

    def min(self, axis=None):
        """Minimum along axis."""
        result = self._module.min(self._data, axis=axis)
        if axis is None or np.isscalar(result):
            return result
        return BackendArray(result, self._backend)

    # NumPy compatibility
    def __array__(self):
        """NumPy array interface - automatic conversion."""
        return self._backend.to_numpy(self._data)

    def __repr__(self):
        return f"BackendArray({self._backend.name}, shape={self.shape}, dtype={self.dtype})"


class BackendArrayWrapper:
    """
    High-level wrapper that provides NumPy-like interface with backend acceleration.

    This is the main class that solvers interact with. It handles:
    - Array creation with backend
    - Automatic wrapping/unwrapping
    - NumPy function interception
    """

    def __init__(self, backend: BaseBackend):
        """Initialize wrapper with backend."""
        self.backend = backend
        self.module = backend.array_module

    # Array creation
    def array(self, data, dtype=None):
        """Create backend array from data."""
        backend_array = self.backend.array(data, dtype=dtype)
        return BackendArray(backend_array, self.backend)

    def zeros(self, shape, dtype=None):
        """Create array of zeros."""
        backend_array = self.backend.zeros(shape, dtype=dtype)
        return BackendArray(backend_array, self.backend)

    def ones(self, shape, dtype=None):
        """Create array of ones."""
        backend_array = self.backend.ones(shape, dtype=dtype)
        return BackendArray(backend_array, self.backend)

    def linspace(self, start, stop, num):
        """Create linearly spaced array."""
        backend_array = self.backend.linspace(start, stop, num)
        return BackendArray(backend_array, self.backend)

    # Conversion
    def from_numpy(self, array: np.ndarray):
        """Convert NumPy array to backend array."""
        backend_array = self.backend.from_numpy(array)
        return BackendArray(backend_array, self.backend)

    def to_numpy(self, array: BackendArray | Any) -> np.ndarray:
        """Convert backend array to NumPy."""
        if isinstance(array, BackendArray):
            return self.backend.to_numpy(array._data)
        return np.asarray(array)

    # NumPy function interception
    def __getattr__(self, name):
        """
        Intercept NumPy functions and dispatch to backend.

        This allows code like `wrapper.sin(x)` to use backend's sine.
        """
        if hasattr(self.module, name):
            backend_func = getattr(self.module, name)

            def wrapped_func(*args, **kwargs):
                # Unwrap BackendArray arguments
                unwrapped_args = [arg._data if isinstance(arg, BackendArray) else arg for arg in args]

                # Call backend function
                result = backend_func(*unwrapped_args, **kwargs)

                # Wrap result if it's an array
                if hasattr(result, "shape"):
                    return BackendArray(result, self.backend)
                return result

            return wrapped_func

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


def create_array_wrapper(backend: str | BaseBackend) -> BackendArrayWrapper:
    """
    Create backend array wrapper.

    Args:
        backend: Backend name ("numpy", "jax", "numba", "torch") or instance

    Returns:
        BackendArrayWrapper instance

    Example:
        ```python
        # Create JAX-accelerated wrapper
        jax_wrapper = create_array_wrapper("jax")

        # Use in existing NumPy code
        x = jax_wrapper.zeros((100, 100))  # JAX array
        y = jax_wrapper.sin(x)              # JAX sine
        result = jax_wrapper.to_numpy(y)    # Back to NumPy
        ```
    """
    if isinstance(backend, str):
        from . import create_backend

        backend_instance = create_backend(backend)
    else:
        backend_instance = backend

    return BackendArrayWrapper(backend_instance)


# Convenience function for monkey-patching NumPy operations
def patch_numpy_for_backend(backend: BaseBackend, module_namespace: dict):
    """
    Monkey-patch NumPy functions in a module to use backend.

    EXPERIMENTAL: This modifies the module's namespace to redirect
    NumPy calls to the backend. Use with caution.

    Args:
        backend: Backend instance
        module_namespace: Module's __dict__ or globals()

    Example:
        ```python
        # In a solver module
        backend = create_backend("jax")
        patch_numpy_for_backend(backend, globals())

        # Now all np.* calls use JAX
        x = np.zeros(10)  # Actually JAX zeros!
        ```
    """
    wrapper = BackendArrayWrapper(backend)

    # Replace common NumPy functions
    patched_functions = {
        "zeros": wrapper.zeros,
        "ones": wrapper.ones,
        "array": wrapper.array,
        "linspace": wrapper.linspace,
        "sin": lambda x: wrapper.sin(x),
        "cos": lambda x: wrapper.cos(x),
        "exp": lambda x: wrapper.exp(x),
        "log": lambda x: wrapper.log(x),
        "sqrt": lambda x: wrapper.sqrt(x),
        # Add more as needed
    }

    for name, func in patched_functions.items():
        if name in module_namespace:
            module_namespace[f"_original_{name}"] = module_namespace[name]
        module_namespace[name] = func
