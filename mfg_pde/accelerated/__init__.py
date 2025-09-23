"""
GPU-accelerated solvers for MFG_PDE using JAX.

This module provides high-performance implementations of MFG solvers
using JAX for automatic differentiation, vectorization, and GPU acceleration.

Features:
- Automatic differentiation for sensitivity analysis
- GPU acceleration with CUDA/ROCm support
- Vectorized operations for batch processing
- Just-in-time compilation for optimal performance
- Memory-efficient implementations for large problems

Dependencies:
- jax: For automatic differentiation and GPU acceleration
- jaxlib: JAX backend library
- optax: JAX-based optimization library (optional)
"""

from __future__ import annotations

import warnings

# Check JAX availability
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    from jax.config import config

    HAS_JAX = True

    # Configure JAX for 64-bit precision (important for scientific computing)
    config.update("jax_enable_x64", True)

    # Check GPU availability
    try:
        gpu_devices = jax.devices("gpu")
        HAS_GPU = len(gpu_devices) > 0
        DEFAULT_DEVICE = gpu_devices[0] if HAS_GPU else jax.devices("cpu")[0]
    except:
        HAS_GPU = False
        DEFAULT_DEVICE = jax.devices("cpu")[0]

except ImportError:
    HAS_JAX = False
    HAS_GPU = False
    DEFAULT_DEVICE = None

    # Create dummy JAX objects for graceful fallback
    class DummyJAX:
        def __getattr__(self, name):
            def dummy_func(*args, **kwargs):
                raise ImportError("JAX is required for GPU-accelerated solvers. Install with: pip install jax jaxlib")

            return dummy_func

    jax = DummyJAX()
    jnp = DummyJAX()

    def jit(f):  # No-op decorator
        return f

    def vmap(f, *args, **kwargs):  # No-op decorator
        return f

    def grad(f):  # No-op decorator
        return f


def check_jax_availability() -> bool:
    """Check if JAX is available for GPU acceleration."""
    return HAS_JAX


def check_gpu_availability() -> bool:
    """Check if GPU devices are available."""
    return HAS_GPU


def get_device_info() -> dict:
    """Get information about available compute devices."""
    if not HAS_JAX:
        return {"jax_available": False, "gpu_available": False}

    info = {
        "jax_available": True,
        "gpu_available": HAS_GPU,
        "default_device": str(DEFAULT_DEVICE),
        "all_devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
    }

    if HAS_GPU:
        info["gpu_devices"] = [str(device) for device in jax.devices("gpu")]
        info["gpu_memory"] = []

        try:
            # Get GPU memory info if available
            for device in jax.devices("gpu"):
                memory_info = device.memory_stats() if hasattr(device, "memory_stats") else {}
                info["gpu_memory"].append({"device": str(device), "memory_info": memory_info})
        except:
            pass

    return info


def configure_jax_for_scientific_computing():
    """Configure JAX settings optimized for scientific computing."""
    if not HAS_JAX:
        warnings.warn("JAX not available, skipping configuration")
        return

    # Enable 64-bit precision for numerical accuracy
    jax.config.update("jax_enable_x64", True)

    # Configure memory management
    if HAS_GPU:
        # Pre-allocate GPU memory to avoid fragmentation
        jax.config.update("jax_cuda_visible_devices", "0")

        # Enable memory debugging if needed
        # jax.config.update("jax_debug_nans", True)
        # jax.config.update("jax_debug_infs", True)


def print_acceleration_info():
    """Print information about acceleration capabilities."""
    print("MFG_PDE Acceleration Status")
    print("=" * 40)

    info = get_device_info()

    if info["jax_available"]:
        print(f"JAX Version: {info['jax_version']}")
        print(f"Default Device: {info['default_device']}")
        print(f"All Devices: {len(info['all_devices'])} available")

        for device in info["all_devices"]:
            device_type = "GPU" if "gpu" in device.lower() else "CPU"
            print(f"   {device_type}: {device}")

        if info["gpu_available"]:
            print(f"GPU Acceleration: Enabled ({len(info['gpu_devices'])} GPU(s))")
        else:
            print("GPU Acceleration: Not available (using CPU)")

    else:
        print("JAX: Not installed")
        print("GPU Acceleration: Not available")
        print("To enable GPU acceleration, install JAX:")
        print("   pip install jax jaxlib  # CPU version")
        print("   pip install jax[cuda]  # CUDA version")
        print("   pip install jax[cuda12_pip]  # CUDA 12 version")


# Configure JAX on import
configure_jax_for_scientific_computing()

# Export key components
__all__ = [
    "DEFAULT_DEVICE",
    "HAS_GPU",
    "HAS_JAX",
    "check_gpu_availability",
    "check_jax_availability",
    "get_device_info",
    "grad",
    "jax",
    "jit",
    "jnp",
    "print_acceleration_info",
    "vmap",
]

# Conditional imports for JAX-based solvers
if HAS_JAX:
    try:
        # Individual imports for explicit exports
        from .jax_fp_solver import JAXFokkerPlanckSolver  # noqa: F401
        from .jax_hjb_solver import JAXHJBSolver  # noqa: F401
        from .jax_mfg_solver import JAXMFGSolver  # noqa: F401
        from .jax_solvers import *  # noqa: F403
        from .jax_utils import *  # noqa: F403

        __all__.extend(["JAXFokkerPlanckSolver", "JAXHJBSolver", "JAXMFGSolver"])

    except ImportError as e:
        warnings.warn(f"Could not import JAX solvers: {e}")

# Show acceleration status on import
if __name__ != "__main__":
    import os

    if os.getenv("MFG_PDE_SHOW_ACCELERATION_INFO", "false").lower() == "true":
        print_acceleration_info()
