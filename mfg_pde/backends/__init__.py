"""
MFG_PDE Computation Backends

This module provides different computational backends for MFG solving:
- PyTorch: CUDA/MPS acceleration with neural network support
- JAX: XLA compilation with GPU/TPU support
- Numba: CPU JIT compilation for imperative algorithms
- NumPy: CPU baseline for compatibility

Device priority: CUDA > MPS > JAX GPU > Numba > NumPy
"""

from __future__ import annotations

import warnings
from typing import Any

# Backend registry
_BACKENDS = {}
_DEFAULT_BACKEND = "numpy"


def register_backend(name: str, backend_class):
    """Register a computational backend."""
    _BACKENDS[name] = backend_class


def get_available_backends() -> dict[str, bool]:
    """Get list of available backends with their availability status."""
    backends = {"numpy": True}  # NumPy is always available

    # Check PyTorch availability
    try:
        import torch

        backends["torch"] = True
        backends["torch_cuda"] = torch.cuda.is_available()
        backends["torch_mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        backends["torch"] = False
        backends["torch_cuda"] = False
        backends["torch_mps"] = False

    # Check JAX availability
    try:
        import jax

        backends["jax"] = True
        backends["jax_gpu"] = any("gpu" in str(d).lower() for d in jax.devices())
    except ImportError:
        backends["jax"] = False
        backends["jax_gpu"] = False

    # Check Numba availability
    try:
        import importlib.util

        numba_spec = importlib.util.find_spec("numba")
        backends["numba"] = numba_spec is not None
    except ImportError:
        backends["numba"] = False

    return backends


def create_backend(backend_name: str = "auto", **kwargs):
    """
    Create a computational backend instance.

    Args:
        backend_name: Backend to use ("torch", "jax", "numba", "numpy", or "auto")
        **kwargs: Backend-specific configuration

    Returns:
        Backend instance
    """
    if backend_name == "auto":
        available = get_available_backends()

        # Priority: CUDA > MPS > JAX GPU > JAX CPU > Numba > NumPy
        if available.get("torch_cuda", False):
            backend_name = "torch"
            kwargs.setdefault("device", "cuda")
        elif available.get("torch_mps", False):
            backend_name = "torch"
            kwargs.setdefault("device", "mps")
        elif available.get("jax_gpu", False):
            backend_name = "jax"
            kwargs.setdefault("device", "gpu")
        elif available.get("jax", False):
            backend_name = "jax"
        elif available.get("numba", False):
            backend_name = "numba"
        else:
            backend_name = "numpy"

    if backend_name not in _BACKENDS:
        if backend_name == "torch":
            # Try to register PyTorch backend
            try:
                from .torch_backend import TorchBackend

                register_backend("torch", TorchBackend)
            except ImportError:
                raise ImportError(
                    "PyTorch backend requested but not available. Install with: pip install torch"
                ) from None
        elif backend_name == "jax":
            # Try to register JAX backend
            try:
                from .jax_backend import JAXBackend

                register_backend("jax", JAXBackend)
            except ImportError:
                raise ImportError(
                    "JAX backend requested but not available. Install with: pip install 'mfg_pde[jax]'"
                ) from None
        elif backend_name == "numba":
            # Try to register Numba backend
            try:
                from .numba_backend import NumbaBackend

                register_backend("numba", NumbaBackend)
            except ImportError:
                raise ImportError(
                    "Numba backend requested but not available. Install with: pip install numba"
                ) from None
        elif backend_name == "numpy":
            from .numpy_backend import NumPyBackend

            register_backend("numpy", NumPyBackend)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    return _BACKENDS[backend_name](**kwargs)


def get_backend_info() -> dict[str, Any]:
    """Get information about available backends."""
    available = get_available_backends()
    info = {
        "available_backends": available,
        "default_backend": _DEFAULT_BACKEND,
        "registered_backends": list(_BACKENDS.keys()),
    }

    # Add PyTorch-specific info if available
    if available.get("torch", False):
        try:
            import torch

            info["torch_info"] = {
                "version": torch.__version__,
                "cuda_available": available.get("torch_cuda", False),
                "mps_available": available.get("torch_mps", False),
            }

            if available.get("torch_cuda", False):
                info["torch_info"].update(
                    {
                        "cuda_version": torch.version.cuda,
                        "cuda_device_count": torch.cuda.device_count(),
                        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    }
                )

        except Exception:
            info["torch_info"] = {"error": "PyTorch available but info retrieval failed"}

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
    from .torch_backend import TorchBackend

    register_backend("torch", TorchBackend)
except ImportError:
    pass  # PyTorch is optional

try:
    from .jax_backend import JAXBackend

    register_backend("jax", JAXBackend)
except ImportError:
    pass  # JAX is optional


# Backward compatibility functions for legacy code
def get_legacy_backend_list():
    """Legacy function for backward compatibility."""
    warnings.warn(
        "get_legacy_backend_list is deprecated. Use get_available_backends() instead.", DeprecationWarning, stacklevel=2
    )
    return get_available_backends()


# Ensure essential backends are always available for compatibility
def ensure_numpy_backend():
    """Ensure NumPy backend is always available for compatibility."""
    if "numpy" not in _BACKENDS:
        try:
            from .numpy_backend import NumPyBackend

            register_backend("numpy", NumPyBackend)
        except ImportError as e:
            raise ImportError("NumPy backend is required for MFG_PDE compatibility") from e


# Auto-initialize on import
ensure_numpy_backend()
