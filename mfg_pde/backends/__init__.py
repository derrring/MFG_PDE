"""
MFG_PDE Computation Backends

This module provides different computational backends for MFG solving,
including NumPy (default) and JAX (high-performance with GPU support).
"""

import warnings
from typing import Any, Dict, Optional

# Backend registry
_BACKENDS = {}
_DEFAULT_BACKEND = "numpy"


def register_backend(name: str, backend_class):
    """Register a computational backend."""
    _BACKENDS[name] = backend_class


def get_available_backends() -> Dict[str, bool]:
    """Get list of available backends with their availability status."""
    backends = {"numpy": True}  # NumPy is always available

    # Check JAX availability
    try:
        import jax

        backends["jax"] = True
    except ImportError:
        backends["jax"] = False

    return backends


def create_backend(backend_name: str = "auto", **kwargs):
    """
    Create a computational backend instance.

    Args:
        backend_name: Backend to use ("numpy", "jax", or "auto")
        **kwargs: Backend-specific configuration

    Returns:
        Backend instance
    """
    if backend_name == "auto":
        available = get_available_backends()
        if available.get("jax", False):
            backend_name = "jax"
        else:
            backend_name = "numpy"

    if backend_name not in _BACKENDS:
        if backend_name == "jax":
            # Try to register JAX backend
            try:
                from .jax_backend import JAXBackend

                register_backend("jax", JAXBackend)
            except ImportError:
                raise ImportError(
                    "JAX backend requested but not available. "
                    "Install with: pip install 'mfg_pde[jax]'"
                )
        elif backend_name == "numpy":
            from .numpy_backend import NumPyBackend

            register_backend("numpy", NumPyBackend)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    return _BACKENDS[backend_name](**kwargs)


def get_backend_info() -> Dict[str, Any]:
    """Get information about available backends."""
    available = get_available_backends()
    info = {
        "available_backends": available,
        "default_backend": _DEFAULT_BACKEND,
        "registered_backends": list(_BACKENDS.keys()),
    }

    # Add JAX-specific info if available
    if available.get("jax", False):
        try:
            import jax

            info["jax_info"] = {
                "version": jax.__version__,
                "devices": [str(d) for d in jax.devices()],
                "default_device": str(jax.devices()[0]),
                "has_gpu": any("gpu" in str(d).lower() for d in jax.devices()),
            }
        except Exception:
            info["jax_info"] = {"error": "JAX available but info retrieval failed"}

    return info


# Initialize default backends
try:
    from .numpy_backend import NumPyBackend

    register_backend("numpy", NumPyBackend)
except ImportError:
    warnings.warn("NumPy backend not available")

try:
    from .jax_backend import JAXBackend

    register_backend("jax", JAXBackend)
except ImportError:
    pass  # JAX is optional
