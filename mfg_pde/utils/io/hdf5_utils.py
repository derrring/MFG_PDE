"""
HDF5 utilities for MFG_PDE solver data persistence.

This module provides comprehensive HDF5 support for saving and loading
MFG solver results, including:
- Solution arrays (U, M)
- Grid information (time and spatial grids)
- Convergence metadata
- Problem configuration
- Checkpointing for resume capability

HDF5 is the recommended format for solver outputs due to:
- Native support for multidimensional NumPy arrays
- Efficient compression
- Rich metadata capabilities
- Scientific computing standard
- Hierarchical data organization

Examples:
    Save solver result:
        >>> from mfg_pde.utils.io.hdf5_utils import save_solution
        >>> save_solution(U, M, metadata, 'solution.h5')

    Load solver result:
        >>> from mfg_pde.utils.io.hdf5_utils import load_solution
        >>> U, M, metadata = load_solution('solution.h5')

    Save with SolverResult:
        >>> result.save_hdf5('solution.h5')
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Optional dependency with graceful fallback
try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


def _check_hdf5_available() -> None:
    """Check if h5py is available, raise helpful error if not."""
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")


def save_solution(
    U: NDArray,
    M: NDArray,
    metadata: dict[str, Any],
    filename: str | Path,
    *,
    compression: str = "gzip",
    compression_opts: int = 4,
    x_grid: NDArray | None = None,
    t_grid: NDArray | None = None,
) -> None:
    """
    Save MFG solver solution to HDF5 file.

    Saves the value function U(t,x), density M(t,x), and associated metadata
    to an HDF5 file with compression.

    Args:
        U: Value/control function array, shape (Nt+1, Nx+1)
        M: Density/distribution array, shape (Nt+1, Nx+1)
        metadata: Dictionary with convergence info and solver parameters
        filename: Output HDF5 file path
        compression: Compression algorithm ('gzip', 'lzf', None)
        compression_opts: Compression level (1-9 for gzip)
        x_grid: Optional spatial grid coordinates
        t_grid: Optional temporal grid coordinates

    Raises:
        ImportError: If h5py not installed
        ValueError: If U and M shapes don't match

    Example:
        >>> save_solution(U, M, {'converged': True, 'iterations': 47},
        ...               'solution.h5')
    """
    _check_hdf5_available()

    if U.shape != M.shape:
        raise ValueError(f"U and M must have same shape: U{U.shape} vs M{M.shape}")

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        # Create main solution group
        solutions = f.create_group("solutions")
        solutions.create_dataset(
            "U",
            data=U,
            compression=compression,
            compression_opts=compression_opts,
        )
        solutions.create_dataset(
            "M",
            data=M,
            compression=compression,
            compression_opts=compression_opts,
        )

        # Save grid information if provided
        if x_grid is not None or t_grid is not None:
            grids = f.create_group("grids")
            if x_grid is not None:
                grids.create_dataset("x", data=x_grid)
            if t_grid is not None:
                grids.create_dataset("t", data=t_grid)

        # Save metadata as attributes
        meta_group = f.create_group("metadata")
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            elif isinstance(value, np.ndarray):
                # Save array metadata as datasets
                meta_group.create_dataset(key, data=value)
            elif value is None:
                meta_group.attrs[key] = "None"
            else:
                # Convert to string for complex types
                meta_group.attrs[key] = str(value)

        # Add file metadata
        f.attrs["format_version"] = "1.0"
        f.attrs["package"] = "mfg_pde"


def load_solution(
    filename: str | Path,
) -> tuple[NDArray, NDArray, dict[str, Any]]:
    """
    Load MFG solver solution from HDF5 file.

    Args:
        filename: HDF5 file path to load

    Returns:
        Tuple of (U, M, metadata) where:
        - U: Value function array
        - M: Density array
        - metadata: Dictionary with convergence info and parameters

    Raises:
        ImportError: If h5py not installed
        FileNotFoundError: If file doesn't exist
        KeyError: If file missing required datasets

    Example:
        >>> U, M, metadata = load_solution('solution.h5')
        >>> print(f"Converged: {metadata['converged']}")
    """
    _check_hdf5_available()

    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        # Load solution arrays
        U = f["solutions/U"][:]
        M = f["solutions/M"][:]

        # Load metadata
        metadata = {}
        if "metadata" in f:
            meta_group = f["metadata"]

            # Load attributes
            for key, value in meta_group.attrs.items():
                if value == "None":
                    metadata[key] = None
                else:
                    metadata[key] = value

            # Load array metadata
            for key in meta_group:
                metadata[key] = meta_group[key][:]

        # Load grids if present
        if "grids" in f:
            grids = f["grids"]
            if "x" in grids:
                metadata["x_grid"] = grids["x"][:]
            if "t" in grids:
                metadata["t_grid"] = grids["t"][:]

    return U, M, metadata


def save_checkpoint(
    solver_state: dict[str, Any],
    filename: str | Path,
    *,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save complete solver state for resuming computation.

    Saves all solver state including current iterates, convergence history,
    and configuration for resuming interrupted computations.

    Args:
        solver_state: Dictionary containing:
            - U: Current U iterate
            - M: Current M iterate
            - iteration: Current iteration number
            - residuals_u: Convergence history for U
            - residuals_m: Convergence history for M
            - config: Solver configuration (optional)
        filename: Output HDF5 file path
        compression: Compression algorithm
        compression_opts: Compression level

    Example:
        >>> state = {
        ...     'U': solver.U,
        ...     'M': solver.M,
        ...     'iteration': solver.iterations_run,
        ...     'residuals_u': solver.l2distu_abs,
        ...     'residuals_m': solver.l2distm_abs
        ... }
        >>> save_checkpoint(state, 'checkpoint.h5')
    """
    _check_hdf5_available()

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        # Save current state
        state = f.create_group("state")

        # Save main arrays
        for key in ["U", "M"]:
            if key in solver_state:
                state.create_dataset(
                    key,
                    data=solver_state[key],
                    compression=compression,
                    compression_opts=compression_opts,
                )

        # Save convergence history
        history = state.create_group("history")
        for key in ["residuals_u", "residuals_m"]:
            if key in solver_state:
                history.create_dataset(key, data=solver_state[key])

        # Save scalar state
        if "iteration" in solver_state:
            state.attrs["iteration"] = solver_state["iteration"]

        # Save configuration if present
        if "config" in solver_state:
            config_group = f.create_group("config")
            config = solver_state["config"]
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    config_group.attrs[key] = value
                elif isinstance(value, dict):
                    # Nested config
                    nested = config_group.create_group(key)
                    for k, v in value.items():
                        if isinstance(v, (str, int, float, bool)):
                            nested.attrs[k] = v

        # Add checkpoint metadata
        f.attrs["checkpoint_type"] = "mfg_solver"
        f.attrs["format_version"] = "1.0"


def load_checkpoint(filename: str | Path) -> dict[str, Any]:
    """
    Load solver checkpoint for resuming computation.

    Args:
        filename: HDF5 checkpoint file path

    Returns:
        Dictionary with solver state including U, M, iteration, residuals

    Raises:
        ImportError: If h5py not installed
        FileNotFoundError: If checkpoint file doesn't exist

    Example:
        >>> state = load_checkpoint('checkpoint.h5')
        >>> solver.U = state['U']
        >>> solver.M = state['M']
        >>> solver.resume(from_iteration=state['iteration'])
    """
    _check_hdf5_available()

    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    state = {}

    with h5py.File(filepath, "r") as f:
        # Load main arrays
        if "state" in f:
            state_group = f["state"]

            for key in ["U", "M"]:
                if key in state_group:
                    state[key] = state_group[key][:]

            # Load iteration number
            if "iteration" in state_group.attrs:
                state["iteration"] = state_group.attrs["iteration"]

            # Load convergence history
            if "history" in state_group:
                history = state_group["history"]
                for key in history:
                    state[key] = history[key][:]

        # Load configuration
        if "config" in f:
            config = {}
            config_group = f["config"]

            for key, value in config_group.attrs.items():
                config[key] = value

            for key in config_group:
                nested_config = {}
                nested_group = config_group[key]
                for k, v in nested_group.attrs.items():
                    nested_config[k] = v
                config[key] = nested_config

            state["config"] = config

    return state


def get_hdf5_info(filename: str | Path) -> dict[str, Any]:
    """
    Get information about HDF5 file contents without loading data.

    Args:
        filename: HDF5 file path

    Returns:
        Dictionary with file structure and metadata

    Example:
        >>> info = get_hdf5_info('solution.h5')
        >>> print(f"Solution shape: {info['U_shape']}")
        >>> print(f"Converged: {info['metadata']['converged']}")
    """
    _check_hdf5_available()

    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    info = {}

    with h5py.File(filepath, "r") as f:
        # File-level attributes
        info["format_version"] = f.attrs.get("format_version", "unknown")

        # Dataset shapes
        if "solutions" in f:
            solutions = f["solutions"]
            if "U" in solutions:
                info["U_shape"] = solutions["U"].shape
                info["U_dtype"] = solutions["U"].dtype
            if "M" in solutions:
                info["M_shape"] = solutions["M"].shape
                info["M_dtype"] = solutions["M"].dtype

        # Metadata
        if "metadata" in f:
            info["metadata"] = dict(f["metadata"].attrs)

        # Grids
        if "grids" in f:
            grids = f["grids"]
            if "x" in grids:
                info["x_grid_size"] = grids["x"].shape[0]
            if "t" in grids:
                info["t_grid_size"] = grids["t"].shape[0]

    return info


# Convenience function aliases
save_hdf5 = save_solution
load_hdf5 = load_solution

__all__ = [
    "HDF5_AVAILABLE",
    "get_hdf5_info",
    "load_checkpoint",
    "load_hdf5",
    "load_solution",
    "save_checkpoint",
    "save_hdf5",
    "save_solution",
]
