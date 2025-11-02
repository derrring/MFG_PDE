#!/usr/bin/env python3
"""
Hamiltonian Signature Adapter

Provides automatic detection and conversion between different Hamiltonian signatures,
enabling backward compatibility while standardizing on a unified signature.

Standard Signature: hamiltonian(x, m, p, t)
    - x: spatial coordinates (state)
    - m: density (state)
    - p: momentum/gradient (control)
    - t: time (parameter)

Supported Legacy Signatures:
    - (x, p, m, t): Legacy 1D ordering
    - (t, x, p, m): Neural network convention
    - Custom signatures can be added via explicit mapping

Example:
    >>> from mfg_pde.utils import HamiltonianAdapter
    >>>
    >>> # Legacy signature function
    >>> def old_hamiltonian(x, p, m, t=0):
    ...     return 0.5 * p**2 + m
    >>>
    >>> # Create adapter
    >>> adapter = HamiltonianAdapter(old_hamiltonian)
    >>>
    >>> # Call with standard signature
    >>> H = adapter(x=1.0, m=0.5, p=2.0, t=0.0)  # Automatically converts
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray
else:
    import numpy as np  # noqa: TC002


class HamiltonianAdapter:
    """
    Adapter to support multiple Hamiltonian signatures.

    Automatically detects the signature of a Hamiltonian function and provides
    a unified interface using the standard signature: hamiltonian(x, m, p, t).

    This enables backward compatibility with legacy code while encouraging
    migration to the standard signature.

    Attributes:
        func: Original Hamiltonian function
        signature_type: Detected signature type ("standard", "legacy", "neural", "unknown")
        parameter_names: List of parameter names from original function
        uses_standard: True if function already uses standard signature
    """

    def __init__(self, hamiltonian_func: Callable, signature_hint: str | None = None):
        """
        Initialize Hamiltonian adapter.

        Args:
            hamiltonian_func: Hamiltonian function to adapt
            signature_hint: Optional hint for signature type
                - None: Auto-detect (default)
                - "standard": Force standard (x, m, p, t)
                - "legacy": Force legacy (x, p, m, t)
                - "neural": Force neural (t, x, p, m)

        Raises:
            ValueError: If signature cannot be detected and no hint provided
        """
        self.func = hamiltonian_func
        self.signature_hint = signature_hint

        # Detect signature
        self.signature_type, self.parameter_names = self._detect_signature(hamiltonian_func, signature_hint)

        # Check if already using standard signature
        self.uses_standard = self.signature_type == "standard"

        # Emit warning for non-standard signatures
        if not self.uses_standard and self.signature_type != "unknown":
            warnings.warn(
                f"Hamiltonian uses non-standard signature '{self.signature_type}'. "
                f"Consider migrating to standard signature: hamiltonian(x, m, p, t). "
                f"See docs/development/HAMILTONIAN_SIGNATURE_ANALYSIS_2025-11-02.md",
                FutureWarning,
                stacklevel=3,
            )

    def _detect_signature(self, func: Callable, hint: str | None = None) -> tuple[str, list[str]]:
        """
        Detect Hamiltonian signature from function parameters.

        Args:
            func: Hamiltonian function
            hint: Optional signature hint

        Returns:
            Tuple of (signature_type, parameter_names)
        """
        # If hint provided, trust it
        if hint in ["standard", "legacy", "neural"]:
            return hint, []

        # Get function signature
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
        except (ValueError, TypeError):
            return "unknown", []

        # Remove 'self' if present (for methods)
        if params and params[0] == "self":
            params = params[1:]

        # Empty or too few parameters
        if len(params) < 3:
            return "unknown", params

        # Match against known patterns
        # Standard: (x, m, p, t) or (x, m, p) with t optional
        if params[:3] == ["x", "m", "p"]:
            return "standard", params

        # Legacy: (x, p, m, t) or (x, p, m) with t optional
        if params[:3] == ["x", "p", "m"]:
            return "legacy", params

        # Neural network: (t, x, p, m)
        if len(params) >= 4 and params[:4] == ["t", "x", "p", "m"]:
            return "neural", params

        # Check for common variations
        # Sometimes parameters have descriptive names like x_position, momentum, etc.
        if len(params) >= 3:
            # Try to infer from common naming patterns
            first_three = [p.lower() for p in params[:3]]

            # Standard variations
            if any(name in first_three[0] for name in ["x", "pos", "state"]) and any(
                name in first_three[1] for name in ["m", "rho", "density"]
            ):
                if any(name in first_three[2] for name in ["p", "grad", "momentum"]):
                    return "standard", params

            # Legacy variations
            if any(name in first_three[0] for name in ["x", "pos", "state"]) and any(
                name in first_three[1] for name in ["p", "grad", "momentum"]
            ):
                if any(name in first_three[2] for name in ["m", "rho", "density"]):
                    return "legacy", params

        return "unknown", params

    def __call__(
        self,
        x: float | NDArray[np.float64],
        m: float | NDArray[np.float64],
        p: float | NDArray[np.float64],
        t: float = 0.0,
    ) -> float | NDArray[np.float64]:
        """
        Call Hamiltonian with standard signature.

        Automatically converts to the original function's signature.

        Args:
            x: Spatial coordinates (scalar or array)
            m: Density value(s)
            p: Momentum/gradient (scalar or array)
            t: Time (default 0.0)

        Returns:
            Hamiltonian value(s)

        Raises:
            ValueError: If signature cannot be converted
        """
        if self.signature_type == "standard":
            # Already standard, call directly
            return self.func(x, m, p, t)

        elif self.signature_type == "legacy":
            # Legacy: (x, p, m, t)
            return self.func(x, p, m, t)

        elif self.signature_type == "neural":
            # Neural: (t, x, p, m)
            return self.func(t, x, p, m)

        else:
            # Unknown signature - try standard first
            try:
                return self.func(x, m, p, t)
            except TypeError as e:
                # Try legacy as fallback
                try:
                    return self.func(x, p, m, t)
                except TypeError:
                    raise ValueError(
                        f"Hamiltonian signature could not be determined automatically. "
                        f"Detected parameters: {self.parameter_names}. "
                        f"Please use standard signature: hamiltonian(x, m, p, t) "
                        f"or provide signature_hint='standard'/'legacy'/'neural'."
                    ) from e

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the adapted Hamiltonian.

        Returns:
            Dictionary with signature information
        """
        return {
            "signature_type": self.signature_type,
            "parameter_names": self.parameter_names,
            "uses_standard": self.uses_standard,
            "function_name": getattr(self.func, "__name__", "<unknown>"),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HamiltonianAdapter("
            f"signature={self.signature_type}, "
            f"params={self.parameter_names}, "
            f"func={getattr(self.func, '__name__', '<lambda>')}"
            f")"
        )


def create_hamiltonian_adapter(
    hamiltonian_func: Callable | None = None, signature_hint: str | None = None
) -> HamiltonianAdapter | None:
    """
    Factory function to create HamiltonianAdapter.

    Args:
        hamiltonian_func: Hamiltonian function to adapt (optional)
        signature_hint: Optional signature hint

    Returns:
        HamiltonianAdapter instance, or None if hamiltonian_func is None

    Example:
        >>> def legacy_hamiltonian(x, p, m, t=0):
        ...     return 0.5 * p**2
        >>>
        >>> adapter = create_hamiltonian_adapter(legacy_hamiltonian)
        >>> H = adapter(x=1.0, m=0.5, p=2.0, t=0.0)
    """
    if hamiltonian_func is None:
        return None

    return HamiltonianAdapter(hamiltonian_func, signature_hint=signature_hint)


# Convenience function for direct use in solvers
def adapt_hamiltonian(
    hamiltonian_func: Callable,
    x: float | NDArray[np.float64],
    m: float | NDArray[np.float64],
    p: float | NDArray[np.float64],
    t: float = 0.0,
    signature_hint: str | None = None,
) -> float | NDArray[np.float64]:
    """
    Adapt and call Hamiltonian in one step.

    Convenience function for one-off Hamiltonian evaluations without
    creating a persistent adapter object.

    Args:
        hamiltonian_func: Hamiltonian function
        x: Spatial coordinates
        m: Density value(s)
        p: Momentum/gradient
        t: Time (default 0.0)
        signature_hint: Optional signature hint

    Returns:
        Hamiltonian value(s)

    Example:
        >>> def old_H(x, p, m):
        ...     return 0.5 * p**2 + m
        >>>
        >>> H = adapt_hamiltonian(old_H, x=1.0, m=0.5, p=2.0)
    """
    adapter = HamiltonianAdapter(hamiltonian_func, signature_hint=signature_hint)
    return adapter(x, m, p, t)
