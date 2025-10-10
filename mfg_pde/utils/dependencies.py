"""
Optional dependency checking with helpful error messages.

Provides utilities for checking optional dependencies and displaying
clear, actionable error messages when dependencies are missing.
"""

import functools
from typing import Literal

DependencyGroup = Literal["neural", "reinforcement", "numerical", "performance", "gpu", "all"]

# Mapping of packages to installation information
DEPENDENCY_MAP = {
    "torch": {
        "install_group": "neural",
        "install_cmd": "pip install mfg-pde[neural]",
        "alternative": "pip install torch",
        "used_by": ["RL algorithms", "neural operators", "GPU acceleration"],
    },
    "jax": {
        "install_group": "performance",
        "install_cmd": "pip install mfg-pde[performance]",
        "alternative": "pip install jax jaxlib",
        "used_by": ["JAX backend", "autodiff", "GPU kernels"],
    },
    "gymnasium": {
        "install_group": "reinforcement",
        "install_cmd": "pip install mfg-pde[reinforcement]",
        "alternative": "pip install gymnasium",
        "used_by": ["RL environments", "MFG games"],
    },
    "igraph": {
        "install_group": "numerical",
        "install_cmd": "pip install mfg-pde[numerical]",
        "alternative": "pip install igraph",
        "used_by": ["Network MFG", "graph algorithms"],
    },
    "networkx": {
        "install_group": "numerical",
        "install_cmd": "pip install mfg-pde[numerical]",
        "alternative": "pip install networkx",
        "used_by": ["Network visualization", "graph analysis"],
    },
    "plotly": {
        "install_group": "all",
        "install_cmd": "pip install mfg-pde[all]",
        "alternative": "pip install plotly",
        "used_by": ["Interactive visualizations", "3D plots"],
    },
    "bokeh": {
        "install_group": "all",
        "install_cmd": "pip install mfg-pde[all]",
        "alternative": "pip install bokeh",
        "used_by": ["Interactive plots", "dashboard"],
    },
    "polars": {
        "install_group": "performance",
        "install_cmd": "pip install mfg-pde[performance]",
        "alternative": "pip install polars",
        "used_by": ["Data analysis", "parameter sweeps"],
    },
}


def is_available(package: str) -> bool:
    """
    Check if a package is available without raising an error.

    Args:
        package: Package name to check

    Returns:
        True if package can be imported, False otherwise

    Example:
        >>> if is_available('torch'):
        ...     import torch
        ...     # Use PyTorch features
    """
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def check_dependency(package: str, feature: str | None = None) -> bool:
    """
    Check if an optional dependency is available.

    Args:
        package: Package name to check (e.g., 'torch', 'jax')
        feature: Optional feature description for error message

    Returns:
        True if package is available

    Raises:
        ImportError: If package is missing, with helpful installation instructions

    Example:
        >>> check_dependency('torch', feature='neural operators')
        True  # If torch is installed

        >>> check_dependency('torch')  # If torch is missing
        ImportError: torch required.
        Used by: RL algorithms, neural operators, GPU acceleration

        Install options:
          1. pip install mfg-pde[neural]
          2. pip install torch
    """
    if is_available(package):
        return True

    # Build helpful error message
    if package in DEPENDENCY_MAP:
        info = DEPENDENCY_MAP[package]
        feature_msg = f" for {feature}" if feature else ""

        msg = (
            f"{package} required{feature_msg}.\n"
            f"Used by: {', '.join(info['used_by'])}\n\n"
            f"Install options:\n"
            f"  1. {info['install_cmd']}\n"
            f"  2. {info['alternative']}"
        )
    else:
        # Unknown package, provide basic message
        feature_msg = f" for {feature}" if feature else ""
        msg = f"{package} required{feature_msg} but not installed.\nInstall with: pip install {package}"

    raise ImportError(msg)


def require_dependencies(*packages: str, feature: str | None = None):
    """
    Decorator to require dependencies for a function or class.

    Args:
        *packages: Package names required
        feature: Optional feature description for error messages

    Returns:
        Decorator function

    Example:
        >>> @require_dependencies('torch', feature='Mean Field DDPG')
        ... def create_ddpg_agent(env):
        ...     import torch
        ...     # Implementation uses PyTorch

        >>> @require_dependencies('jax', 'jaxlib')
        ... class JAXBackend:
        ...     # Implementation uses JAX
    """

    def decorator(func_or_class):
        @functools.wraps(func_or_class)
        def wrapper(*args, **kwargs):
            # Check all required dependencies
            for pkg in packages:
                check_dependency(pkg, feature=feature)
            return func_or_class(*args, **kwargs)

        return wrapper

    return decorator


def get_available_features() -> dict[str, bool]:
    """
    Get availability status of all optional features.

    Returns:
        Dictionary mapping feature names to availability status

    Example:
        >>> features = get_available_features()
        >>> if features['pytorch']:
        ...     # Use PyTorch features
        >>> if features['jax']:
        ...     # Use JAX features
    """
    return {
        "pytorch": is_available("torch"),
        "jax": is_available("jax"),
        "gymnasium": is_available("gymnasium"),
        "igraph": is_available("igraph"),
        "networkx": is_available("networkx"),
        "plotly": is_available("plotly"),
        "bokeh": is_available("bokeh"),
        "polars": is_available("polars"),
        "numba": is_available("numba"),
        "cupy": is_available("cupy"),
    }


def show_optional_features() -> None:
    """
    Display status of all optional features.

    Prints a formatted table showing which optional dependencies
    are installed and available.

    Example:
        >>> import mfg_pde
        >>> mfg_pde.show_optional_features()
        MFG_PDE Optional Features
        ==================================================
        pytorch        : ✓ Available
        jax            : ✗ Not installed
        gymnasium      : ✓ Available
        ...
    """
    features = get_available_features()

    print("MFG_PDE Optional Features")
    print("=" * 50)

    for name, available in sorted(features.items()):
        status = "✓ Available" if available else "✗ Not installed"
        print(f"{name:15s}: {status}")

    print("=" * 50)
    print("\nInstallation options:")
    print("  All features:     pip install mfg-pde[all]")
    print("  Neural networks:  pip install mfg-pde[neural]")
    print("  Reinforcement:    pip install mfg-pde[reinforcement]")
    print("  Performance:      pip install mfg-pde[performance]")
    print("  GPU support:      pip install mfg-pde[gpu]")


# Pre-check common dependencies for module-level flags
TORCH_AVAILABLE = is_available("torch")
JAX_AVAILABLE = is_available("jax")
GYMNASIUM_AVAILABLE = is_available("gymnasium")
IGRAPH_AVAILABLE = is_available("igraph")
NETWORKX_AVAILABLE = is_available("networkx")
PLOTLY_AVAILABLE = is_available("plotly")
BOKEH_AVAILABLE = is_available("bokeh")
POLARS_AVAILABLE = is_available("polars")
NUMBA_AVAILABLE = is_available("numba")
CUPY_AVAILABLE = is_available("cupy")
