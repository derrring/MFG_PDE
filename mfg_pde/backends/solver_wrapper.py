"""
Solver Wrapper for Transparent Backend Acceleration.

This module provides high-level wrappers that make existing solvers
work with backend acceleration WITHOUT modifying their source code.

Strategy:
1. Wrap solver's solve method
2. Convert inputs to backend arrays
3. Run solver (accelerated operations)
4. Convert outputs back to NumPy

This is a **zero-modification** approach to backend integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .base_backend import BaseBackend

from .array_wrapper import create_array_wrapper


class AcceleratedSolverWrapper:
    """
    Wrapper that adds backend acceleration to existing solvers.

    This wrapper:
    - Intercepts solver method calls
    - Converts NumPy arrays to backend arrays
    - Executes solver with accelerated arrays
    - Converts results back to NumPy

    The solver itself is unchanged!
    """

    def __init__(self, solver: Any, backend: str | BaseBackend):
        """
        Initialize accelerated wrapper.

        Args:
            solver: Original solver instance (HJBFDMSolver, FPParticleSolver, etc.)
            backend: Backend for acceleration ("jax", "numba", "torch", or instance)
        """
        self._solver = solver
        self._backend_name = backend if isinstance(backend, str) else backend.name
        self._wrapper = create_array_wrapper(backend)
        self._backend = self._wrapper.backend

        # Store original methods for later restoration
        self._original_methods = {}

    def wrap(self) -> Any:
        """
        Apply wrapper to solver.

        This modifies the solver instance to use backend arrays.

        Returns:
            The modified solver (same instance, for chaining)
        """
        # Wrap the main solve methods (Issue #643: getattr+callable pattern)
        solve_hjb = getattr(self._solver, "solve_hjb_system", None)
        if solve_hjb is not None and callable(solve_hjb):
            self._wrap_method("solve_hjb_system")

        solve_fp = getattr(self._solver, "solve_fp_system", None)
        if solve_fp is not None and callable(solve_fp):
            self._wrap_method("solve_fp_system")

        solve = getattr(self._solver, "solve", None)
        if solve is not None and callable(solve):
            self._wrap_method("solve")

        return self._solver

    def _wrap_method(self, method_name: str):
        """Wrap a solver method with backend conversion."""
        original_method = getattr(self._solver, method_name)
        self._original_methods[method_name] = original_method

        def wrapped_method(*args, **kwargs):
            # Convert inputs from NumPy to backend
            backend_args = [self._wrapper.from_numpy(arg) if isinstance(arg, np.ndarray) else arg for arg in args]

            backend_kwargs = {
                k: self._wrapper.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
            }

            # Call original method with backend arrays
            result = original_method(*backend_args, **backend_kwargs)

            # Convert outputs back to NumPy
            # Issue #643: hasattr(r, "shape") is acceptable for array duck-typing
            # (NumPy, PyTorch, JAX arrays all have .shape attribute)
            if isinstance(result, tuple):
                return tuple(self._wrapper.to_numpy(r) if hasattr(r, "shape") else r for r in result)
            elif hasattr(result, "shape"):
                return self._wrapper.to_numpy(result)
            else:
                return result

        setattr(self._solver, method_name, wrapped_method)

    def unwrap(self):
        """Restore original solver methods."""
        for method_name, original_method in self._original_methods.items():
            setattr(self._solver, method_name, original_method)
        self._original_methods.clear()


def accelerate_solver(solver: Any, backend: str) -> Any:
    """
    Convenience function to accelerate a solver.

    Args:
        solver: Solver instance
        backend: Backend name ("jax", "numba", "torch")

    Returns:
        Accelerated solver (same instance, modified)

    Example:
        ```python
        # Original solver (NumPy-based)
        hjb_solver = HJBFDMSolver(problem)

        # Accelerate with JAX (no code changes!)
        accelerate_solver(hjb_solver, "jax")

        # Now runs on GPU if available
        result = hjb_solver.solve_hjb_system(M, U_final, U_prev)
        ```
    """
    wrapper = AcceleratedSolverWrapper(solver, backend)
    return wrapper.wrap()


class BackendContextManager:
    """
    Context manager for temporary backend acceleration.

    Allows using backend acceleration in a specific code block,
    then automatically restoring original behavior.

    Example:
        ```python
        hjb_solver = HJBFDMSolver(problem)

        # Normal NumPy execution
        result1 = hjb_solver.solve_hjb_system(M, U, U_prev)

        # Temporarily accelerated
        with BackendContextManager(hjb_solver, "jax"):
            result2 = hjb_solver.solve_hjb_system(M, U, U_prev)  # JAX!

        # Back to NumPy
        result3 = hjb_solver.solve_hjb_system(M, U, U_prev)
        ```
    """

    def __init__(self, solver: Any, backend: str):
        """Initialize context manager."""
        self._wrapper = AcceleratedSolverWrapper(solver, backend)

    def __enter__(self):
        """Enter context: apply acceleration."""
        return self._wrapper.wrap()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: restore original."""
        self._wrapper.unwrap()


# Decorator for method-level acceleration
def with_backend(backend: str):
    """
    Decorator to run a method with backend acceleration.

    Example:
        ```python
        class MySolver:
            @with_backend("jax")
            def solve(self, M, U):
                # This method runs with JAX arrays
                # All NumPy operations automatically use JAX
                result = np.linalg.solve(A, b)  # JAX linalg
                return result
        ```
    """

    def decorator(method):
        def wrapped(self, *args, **kwargs):
            wrapper_obj = create_array_wrapper(backend)

            # Convert inputs
            backend_args = [wrapper_obj.from_numpy(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
            backend_kwargs = {
                k: wrapper_obj.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
            }

            # Execute
            result = method(self, *backend_args, **backend_kwargs)

            # Convert outputs
            # Issue #643: hasattr(r, "shape") is acceptable for array duck-typing
            if isinstance(result, tuple):
                return tuple(wrapper_obj.to_numpy(r) if hasattr(r, "shape") else r for r in result)
            elif hasattr(result, "shape"):
                return wrapper_obj.to_numpy(result)
            return result

        return wrapped

    return decorator
