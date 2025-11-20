"""
Integration utilities for MFG_PDE.

Provides consistent integration functions that work across NumPy versions,
with preference for NumPy 2.0+ standard but fallbacks for older versions.
"""

from __future__ import annotations

# Import from the comprehensive NumPy compatibility module
from mfg_pde.utils.numpy_compat import get_numpy_info, trapezoid


# Re-export for backward compatibility
def get_integration_info() -> dict[str, str]:
    """Get information about available integration methods."""
    return get_numpy_info()


# Export the main functions
__all__ = ["get_integration_info", "trapezoid"]


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing integration utilities...")

    import numpy as np

    # Test get_integration_info
    info = get_integration_info()
    assert isinstance(info, dict)
    assert "numpy_version" in info
    print(f"  NumPy version: {info['numpy_version']}")

    # Test trapezoid integration
    x = np.linspace(0, 1, 100)
    y = x**2  # Integral should be 1/3
    result = trapezoid(y, x)
    expected = 1 / 3
    error = abs(result - expected)
    assert error < 1e-3, f"Integration error {error} too large"
    print(f"  Trapezoid integration: {result:.6f} (expected {expected:.6f}, error {error:.2e})")

    print("Smoke tests passed!")
