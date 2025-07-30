"""
Advanced array and tensor validation for MFG_PDE using Pydantic.

This module provides Pydantic models for validating complex array shapes,
numerical properties, and physical constraints specific to MFG problems.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator, validator
from ..utils.integration import trapezoid


class ArrayValidationConfig(BaseModel):
    """Configuration for array validation tolerances and checks."""

    mass_conservation_rtol: float = Field(
        1e-3, gt=0.0, description="Relative tolerance for mass conservation"
    )
    smoothness_threshold: float = Field(
        1e3, gt=0.0, description="Threshold for smoothness checking"
    )
    cfl_max: float = Field(
        0.5, gt=0.0, le=1.0, description="Maximum CFL number allowed"
    )

    class Config:
        validate_assignment = True


class MFGGridConfig(BaseModel):
    """
    MFG grid configuration with automatic validation.

    Validates grid dimensions and computes derived quantities with
    stability checks for numerical methods.
    """

    Nx: int = Field(..., ge=10, le=2000, description="Number of spatial grid points")
    Nt: int = Field(..., ge=10, le=20000, description="Number of time grid points")
    xmin: float = Field(0.0, description="Spatial domain minimum")
    xmax: float = Field(1.0, description="Spatial domain maximum")
    T: float = Field(1.0, gt=0.0, le=100.0, description="Final time")
    sigma: float = Field(0.1, gt=0.0, le=10.0, description="Diffusion coefficient")

    @validator("xmax")
    def validate_domain(cls, v, values):
        """Validate spatial domain is well-defined."""
        xmin = values.get("xmin", 0.0)
        if v <= xmin:
            raise ValueError(f"xmax ({v}) must be > xmin ({xmin})")
        return v

    @property
    def dx(self) -> float:
        """Spatial grid spacing."""
        return (self.xmax - self.xmin) / self.Nx

    @property
    def dt(self) -> float:
        """Time grid spacing."""
        return self.T / self.Nt

    @property
    def cfl_number(self) -> float:
        """CFL number for stability analysis."""
        return self.sigma**2 * self.dt / (self.dx**2)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Expected shape for solution arrays (Nt+1, Nx+1)."""
        return (self.Nt + 1, self.Nx + 1)

    @validator("sigma")
    def validate_cfl_stability(cls, v, values):
        """Validate CFL condition for numerical stability."""
        Nx = values.get("Nx")
        Nt = values.get("Nt")
        xmin = values.get("xmin", 0.0)
        xmax = values.get("xmax", 1.0)
        T = values.get("T", 1.0)

        if Nx and Nt and xmax > xmin and T > 0:
            dx = (xmax - xmin) / Nx
            dt = T / Nt
            cfl = v**2 * dt / (dx**2)

            if cfl > 0.5:
                warnings.warn(
                    f"CFL number {cfl:.3f} > 0.5 may cause instability", UserWarning
                )

        return v

    class Config:
        validate_assignment = True


class MFGArrays(BaseModel):
    """
    MFG solution arrays with comprehensive shape and property validation.

    Validates U (HJB solution) and M (FP density) arrays for correct shapes,
    physical constraints, and numerical properties.
    """

    U_solution: np.ndarray = Field(..., description="HJB solution array")
    M_solution: np.ndarray = Field(..., description="FP density array")
    grid_config: MFGGridConfig = Field(..., description="Grid configuration")
    validation_config: ArrayValidationConfig = Field(
        default_factory=ArrayValidationConfig, description="Validation configuration"
    )

    @validator("U_solution")
    def validate_U_solution(cls, v, values):
        """Validate HJB solution array properties."""
        grid_config = values.get("grid_config")

        if grid_config:
            expected_shape = grid_config.grid_shape

            # Shape validation
            if v.shape != expected_shape:
                raise ValueError(
                    f"U solution shape {v.shape} != expected {expected_shape}"
                )

            # Data type validation
            if not np.issubdtype(v.dtype, np.floating):
                raise ValueError("U solution must be floating point array")

            # NaN/Inf validation
            if np.any(np.isnan(v)):
                raise ValueError("U solution contains NaN values")
            if np.any(np.isinf(v)):
                raise ValueError("U solution contains infinite values")

            # Smoothness check (optional warning)
            if v.size > 4:  # Need at least 2x2 for second differences
                try:
                    # Check for large second differences indicating non-smoothness
                    if v.ndim == 2 and min(v.shape) >= 3:
                        second_diff_x = np.diff(v, n=2, axis=1)
                        second_diff_t = np.diff(v, n=2, axis=0)

                        max_diff_x = np.max(np.abs(second_diff_x))
                        max_diff_t = np.max(np.abs(second_diff_t))

                        if max_diff_x > 1e3 or max_diff_t > 1e3:
                            warnings.warn(
                                f"U solution may be non-smooth (max second diff: x={max_diff_x:.2e}, t={max_diff_t:.2e})",
                                UserWarning,
                            )
                except Exception:
                    # Skip smoothness check if it fails
                    pass

        return v

    @validator("M_solution")
    def validate_M_solution(cls, v, values):
        """Validate FP density array properties."""
        grid_config = values.get("grid_config")
        validation_config = values.get("validation_config", ArrayValidationConfig())

        if grid_config:
            expected_shape = grid_config.grid_shape

            # Shape validation
            if v.shape != expected_shape:
                raise ValueError(
                    f"M solution shape {v.shape} != expected {expected_shape}"
                )

            # Data type validation
            if not np.issubdtype(v.dtype, np.floating):
                raise ValueError("M solution must be floating point array")

            # NaN/Inf validation
            if np.any(np.isnan(v)):
                raise ValueError("M solution contains NaN values")
            if np.any(np.isinf(v)):
                raise ValueError("M solution contains infinite values")

            # Physical constraint: non-negativity
            if np.any(v < 0):
                negative_count = np.sum(v < 0)
                min_value = np.min(v)
                raise ValueError(
                    f"Density M must be non-negative everywhere. "
                    f"Found {negative_count} negative values (min: {min_value:.2e})"
                )

            # Mass conservation check
            dx = grid_config.dx
            for t_idx in range(v.shape[0]):
                mass_at_t = trapezoid(v[t_idx], dx=dx)
                if not np.isclose(
                    mass_at_t, 1.0, rtol=validation_config.mass_conservation_rtol
                ):
                    if t_idx == 0:
                        # Initial condition
                        warnings.warn(
                            f"Initial mass not normalized: {mass_at_t:.6f} (should be 1.0)",
                            UserWarning,
                        )
                    elif t_idx == v.shape[0] - 1:
                        # Final condition - more strict
                        warnings.warn(
                            f"Final mass not conserved: {mass_at_t:.6f} (should be 1.0)",
                            UserWarning,
                        )

        return v

    @model_validator(mode="after")
    def validate_solution_consistency(self):
        """Validate consistency between U and M solutions."""
        U_solution = self.U_solution
        M_solution = self.M_solution

        if U_solution is not None and M_solution is not None:
            # Shape consistency
            if U_solution.shape != M_solution.shape:
                raise ValueError(
                    f"U and M shape mismatch: {U_solution.shape} vs {M_solution.shape}"
                )

        return self

    def get_solution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for both solutions."""
        stats = {}

        # U solution statistics
        stats["U"] = {
            "shape": self.U_solution.shape,
            "dtype": str(self.U_solution.dtype),
            "min": float(np.min(self.U_solution)),
            "max": float(np.max(self.U_solution)),
            "mean": float(np.mean(self.U_solution)),
            "std": float(np.std(self.U_solution)),
        }

        # M solution statistics
        stats["M"] = {
            "shape": self.M_solution.shape,
            "dtype": str(self.M_solution.dtype),
            "min": float(np.min(self.M_solution)),
            "max": float(np.max(self.M_solution)),
            "mean": float(np.mean(self.M_solution)),
            "std": float(np.std(self.M_solution)),
        }

        # Mass conservation analysis
        dx = self.grid_config.dx
        mass_history = []
        for t_idx in range(self.M_solution.shape[0]):
            mass_at_t = trapezoid(self.M_solution[t_idx], dx=dx)
            mass_history.append(mass_at_t)

        stats["mass_conservation"] = {
            "initial_mass": mass_history[0],
            "final_mass": mass_history[-1],
            "mass_drift": mass_history[-1] - mass_history[0],
            "max_mass": max(mass_history),
            "min_mass": min(mass_history),
        }

        # CFL analysis
        stats["numerical_stability"] = {
            "cfl_number": self.grid_config.cfl_number,
            "dx": self.grid_config.dx,
            "dt": self.grid_config.dt,
            "sigma": self.grid_config.sigma,
        }

        return stats

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        json_encoders = {
            np.ndarray: lambda v: {
                "shape": v.shape,
                "dtype": str(v.dtype),
                "summary_stats": {
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "mean": float(np.mean(v)),
                },
            }
        }


class CollocationConfig(BaseModel):
    """
    Collocation points configuration with advanced validation.

    Validates collocation points for shape, domain constraints,
    and compatibility with grid configuration.
    """

    points: np.ndarray = Field(..., description="Collocation points array")
    grid_config: MFGGridConfig = Field(..., description="Grid configuration")

    @validator("points")
    def validate_collocation_points(cls, v, values):
        """Validate collocation points properties."""
        # Shape validation
        if v.ndim != 2 or v.shape[1] != 1:
            raise ValueError(
                f"Collocation points must be Nx1 array, got shape {v.shape}"
            )

        # Domain validation
        grid_config = values.get("grid_config")
        if grid_config:
            xmin, xmax = grid_config.xmin, grid_config.xmax
            if np.any(v < xmin) or np.any(v > xmax):
                raise ValueError(f"Collocation points must be in [{xmin}, {xmax}]")

            # Count validation relative to grid
            num_points = len(v)
            min_points = max(5, grid_config.Nx // 20)
            max_points = grid_config.Nx

            if num_points < min_points:
                warnings.warn(
                    f"Few collocation points ({num_points} < {min_points}) may reduce accuracy",
                    UserWarning,
                )

            if num_points > max_points:
                warnings.warn(
                    f"Many collocation points ({num_points} > {max_points}) may be inefficient",
                    UserWarning,
                )

        # Uniqueness check
        if len(np.unique(v)) != len(v):
            warnings.warn("Duplicate collocation points detected", UserWarning)

        # Distribution check
        if len(v) > 1:
            spacing = np.diff(np.sort(v.flatten()))
            min_spacing = np.min(spacing)
            max_spacing = np.max(spacing)

            if max_spacing / min_spacing > 10:
                warnings.warn(
                    f"Irregular collocation point distribution (spacing ratio: {max_spacing/min_spacing:.1f})",
                    UserWarning,
                )

        return v

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration with array validation.

    Combines solver configuration with validated arrays and metadata
    for comprehensive experiment tracking and validation.
    """

    # Core configurations
    grid_config: MFGGridConfig = Field(..., description="Grid configuration")
    arrays: Optional[MFGArrays] = Field(
        None, description="Solution arrays (if available)"
    )
    collocation: Optional[CollocationConfig] = Field(
        None, description="Collocation configuration"
    )

    # Experiment metadata
    experiment_name: str = Field(..., min_length=1, description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    researcher: str = Field("", description="Researcher name")
    tags: list = Field(default_factory=list, description="Experiment tags")

    # Output configuration
    output_dir: str = Field("./results", description="Output directory")
    save_arrays: bool = Field(True, description="Whether to save solution arrays")
    save_plots: bool = Field(True, description="Whether to save visualization plots")

    @validator("experiment_name")
    def validate_experiment_name(cls, v):
        """Validate experiment name is filesystem-safe."""
        import re

        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", v):
            raise ValueError(
                "Experiment name must contain only letters, numbers, _, -, and ."
            )
        return v

    @model_validator(mode="after")
    def validate_experiment_consistency(self):
        """Validate consistency between all configurations."""
        grid_config = self.grid_config
        arrays = self.arrays
        collocation = self.collocation

        # Ensure array configuration matches grid configuration
        if arrays and arrays.grid_config != grid_config:
            warnings.warn(
                "Array grid configuration differs from experiment grid configuration",
                UserWarning,
            )

        # Ensure collocation configuration matches grid configuration
        if collocation and collocation.grid_config != grid_config:
            warnings.warn(
                "Collocation grid configuration differs from experiment grid configuration",
                UserWarning,
            )

        return self

    def to_notebook_metadata(self) -> Dict[str, Any]:
        """Convert to metadata suitable for notebook reporting."""
        metadata = {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "researcher": self.researcher,
            "tags": self.tags,
            "grid_config": self.grid_config.dict(),
        }

        if self.arrays:
            metadata["solution_statistics"] = self.arrays.get_solution_statistics()

        if self.collocation:
            metadata["collocation_points"] = len(self.collocation.points)

        return metadata

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
