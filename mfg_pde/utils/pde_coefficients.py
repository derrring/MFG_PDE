"""
Utilities for handling PDE coefficients (drift, diffusion) in MFG solvers.

This module provides unified handling of scalar, array, and callable coefficients,
eliminating code duplication across HJB, FP, and coupling solvers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.core.mfg_problem import MFGProblem


class CoefficientField:
    """
    Unified interface for scalar, array, and callable PDE coefficients.

    Handles extraction and validation of diffusion and drift coefficients
    at specific timesteps during PDE solving.

    Parameters
    ----------
    field : None | float | ndarray | Callable
        The coefficient field:
        - None: Use default from problem
        - float: Constant coefficient
        - ndarray: Precomputed spatially/temporally varying coefficient
        - Callable: State-dependent coefficient with signature (t, x, m) -> float | ndarray
    default_value : float | ndarray
        Default value to use when field is None (typically problem.sigma or problem.drift)
    field_name : str
        Name of coefficient for error messages (e.g., "diffusion_field", "drift_field")
    dimension : int
        Spatial dimension (1 for 1D, 2 for 2D, etc.)

    Examples
    --------
    Scalar diffusion:
    >>> field = CoefficientField(0.1, problem.sigma, "diffusion_field", dimension=1)
    >>> sigma = field.evaluate_at(timestep=5, grid=x_coords, density=m)

    Array diffusion:
    >>> sigma_array = np.ones((Nt, Nx)) * 0.1
    >>> field = CoefficientField(sigma_array, problem.sigma, "diffusion_field", dimension=1)
    >>> sigma = field.evaluate_at(timestep=5, grid=x_coords, density=m)

    Callable diffusion:
    >>> def porous_medium(t, x, m):
    ...     return 0.1 * m
    >>> field = CoefficientField(porous_medium, problem.sigma, "diffusion_field", dimension=1)
    >>> sigma = field.evaluate_at(timestep=5, grid=x_coords, density=m)
    """

    def __init__(
        self,
        field: None | float | np.ndarray | Callable,
        default_value: float | np.ndarray,
        field_name: str = "coefficient",
        dimension: int = 1,
    ):
        self.field = field
        self.default = default_value
        self.name = field_name
        self.dimension = dimension

        # Cache type checks
        self._is_none = field is None
        self._is_scalar = isinstance(field, (int, float))
        self._is_array = isinstance(field, np.ndarray)
        self._is_callable = callable(field)

    def evaluate_at(
        self,
        timestep_idx: int,
        grid: np.ndarray | tuple[np.ndarray, ...],
        density: np.ndarray,
        dt: float | None = None,
    ) -> float | np.ndarray:
        """
        Evaluate coefficient at specific timestep and state.

        Parameters
        ----------
        timestep_idx : int
            Timestep index for evaluation
        grid : ndarray | tuple[ndarray, ...]
            Spatial grid coordinates:
            - 1D: ndarray of shape (Nx,)
            - nD: tuple of coordinate arrays
        density : ndarray
            Density field at current timestep
        dt : float | None
            Timestep size (needed for computing physical time)

        Returns
        -------
        float | ndarray
            Evaluated coefficient (scalar or array matching density shape)
        """
        if self._is_none:
            return self.default

        elif self._is_scalar:
            return float(self.field)

        elif self._is_callable:
            return self._evaluate_callable(timestep_idx, grid, density, dt)

        elif self._is_array:
            return self._extract_from_array(timestep_idx, density.shape)

        else:
            raise TypeError(f"{self.name} must be None, float, ndarray, or Callable, got {type(self.field)}")

    def _evaluate_callable(
        self,
        timestep_idx: int,
        grid: np.ndarray | tuple[np.ndarray, ...],
        density: np.ndarray,
        dt: float | None,
    ) -> float | np.ndarray:
        """Evaluate callable coefficient with validation."""
        # Compute physical time
        t_current = timestep_idx * dt if dt is not None else timestep_idx

        # Call the coefficient function
        result = self.field(t_current, grid, density)

        # Validate and convert output
        expected_shape = density.shape
        validated_result = self._validate_callable_output(result, expected_shape, timestep_idx)

        return validated_result

    def _validate_callable_output(self, output: Any, expected_shape: tuple, timestep_idx: int) -> float | np.ndarray:
        """
        Validate callable coefficient output.

        Parameters
        ----------
        output : Any
            Output from callable coefficient
        expected_shape : tuple
            Expected shape (matching density)
        timestep_idx : int
            Current timestep index for error messages

        Returns
        -------
        float | ndarray
            Validated output (scalar or array)

        Raises
        ------
        TypeError
            If output is not float or ndarray
        ValueError
            If output shape doesn't match expected or contains NaN/Inf
        """
        # Handle scalar output
        if isinstance(output, (int, float)):
            return float(output)

        # Handle array output
        elif isinstance(output, np.ndarray):
            # Check shape
            if output.shape != expected_shape:
                raise ValueError(
                    f"Callable {self.name} returned array with shape {output.shape}, "
                    f"expected {expected_shape} (matching density shape) at timestep {timestep_idx}"
                )

            # Check for NaN/Inf
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                raise ValueError(
                    f"Callable {self.name} returned NaN or Inf values at timestep {timestep_idx}. "
                    f"Check your coefficient function implementation."
                )

            return output

        else:
            raise TypeError(
                f"Callable {self.name} must return float or ndarray, got {type(output)} at timestep {timestep_idx}"
            )

    def _extract_from_array(self, timestep_idx: int, expected_shape: tuple) -> np.ndarray:
        """
        Extract coefficient from precomputed array.

        Handles both spatially-varying (ndim = dimension) and
        spatiotemporal (ndim = dimension + 1) arrays.

        Parameters
        ----------
        timestep_idx : int
            Current timestep index
        expected_shape : tuple
            Expected spatial shape

        Returns
        -------
        ndarray
            Extracted coefficient array

        Raises
        ------
        ValueError
            If array dimensions are incompatible
        """
        field_ndim = self.field.ndim

        # Spatially varying only: shape matches expected_shape
        if field_ndim == self.dimension:
            if self.field.shape != expected_shape:
                raise ValueError(
                    f"Spatial {self.name} array has shape {self.field.shape}, "
                    f"expected {expected_shape} (matching grid shape)"
                )
            return self.field

        # Spatiotemporal: shape is (Nt, *spatial_shape)
        elif field_ndim == self.dimension + 1:
            # Extract at timestep
            extracted = self.field[timestep_idx, ...]

            if extracted.shape != expected_shape:
                raise ValueError(
                    f"Spatiotemporal {self.name} array at timestep {timestep_idx} "
                    f"has shape {extracted.shape}, expected {expected_shape}"
                )
            return extracted

        else:
            raise ValueError(
                f"{self.name} array must have {self.dimension} dimensions (spatial) or "
                f"{self.dimension + 1} dimensions (spatiotemporal), got {field_ndim} dimensions"
            )

    def is_callable(self) -> bool:
        """Check if coefficient is callable (state-dependent)."""
        return self._is_callable

    def is_constant(self) -> bool:
        """Check if coefficient is constant (None or scalar)."""
        return self._is_none or self._is_scalar

    def is_array(self) -> bool:
        """Check if coefficient is precomputed array."""
        return self._is_array


def get_spatial_grid(problem: MFGProblem) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Get spatial grid coordinates for coefficient evaluation.

    Provides unified interface for both legacy 1D API (xmin, xmax, Nx)
    and modern geometry-based API.

    Parameters
    ----------
    problem : MFGProblem
        MFG problem instance

    Returns
    -------
    ndarray | tuple[ndarray, ...]
        Spatial coordinates:
        - 1D: ndarray of shape (Nx,)
        - nD: tuple of coordinate arrays for each dimension

    Examples
    --------
    Legacy 1D problem:
    >>> grid = get_spatial_grid(problem)  # ndarray of x-coordinates

    Geometry-based 2D problem:
    >>> grid = get_spatial_grid(problem)  # (x_coords, y_coords)
    """
    # Modern geometry-based API
    if hasattr(problem, "geometry") and hasattr(problem.geometry, "coordinates"):
        return problem.geometry.coordinates

    # Legacy 1D API
    elif hasattr(problem, "xmin") and hasattr(problem, "xmax"):
        Nx = problem.Nx + 1 if hasattr(problem, "Nx") else None
        if Nx is None:
            raise AttributeError("Problem must have either geometry.coordinates or (xmin, xmax, Nx) attributes")
        return np.linspace(problem.xmin, problem.xmax, Nx)

    else:
        raise AttributeError("Problem must have either geometry.coordinates or (xmin, xmax, Nx) attributes")
