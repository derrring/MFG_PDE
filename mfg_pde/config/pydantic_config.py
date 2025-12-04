"""
Enhanced Pydantic-based configuration classes for MFG_PDE solvers.

This module provides MFGSolverConfig - the master solver configuration with
automatic validation, serialization, and advanced numerical stability checks.

NOTE: The internal config classes (_NewtonConfig, _GFDMConfig, etc.) are prefixed
with underscore to indicate they are internal to MFGSolverConfig and should not
be used directly. For method-specific configs, use mfg_methods.py instead.

The canonical config classes are:
- mfg_pde.config.mfg_methods: Method-specific configs (FDMConfig, GFDMConfig, etc.)
- mfg_pde.config.core: Core solver configs (SolverConfig, PicardConfig, etc.)
- This module: MFGSolverConfig (master solver config for legacy API)
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _NewtonConfig(BaseModel):
    """
    Internal Newton method configuration for MFGSolverConfig.

    NOTE: This is an internal class. For public API, use mfg_methods.NewtonConfig.

    Provides comprehensive validation for Newton iteration parameters including
    numerical stability checks and convergence criteria validation.
    """

    max_iterations: int = Field(30, ge=1, le=1000, description="Maximum number of Newton iterations")
    tolerance: float = Field(1e-6, gt=1e-15, le=1e-1, description="Convergence tolerance for Newton method")
    damping_factor: float = Field(1.0, gt=0.0, le=1.0, description="Damping parameter for Newton updates")
    line_search: bool = Field(False, description="Whether to use line search for step size control")
    verbose: bool = Field(False, description="Whether to print Newton iteration details")

    @field_validator("tolerance")
    @classmethod
    def validate_numerical_stability(cls, v: float) -> float:
        """Validate tolerance for numerical stability."""
        if v < 1e-12:
            warnings.warn(
                f"Very strict Newton tolerance ({v:.2e}) may cause numerical instability",
                UserWarning,
            )
        return v

    @field_validator("damping_factor")
    @classmethod
    def validate_damping_convergence(cls, v: float) -> float:
        """Validate damping factor for convergence properties."""
        if v < 0.1:
            warnings.warn(
                f"Very small damping factor ({v:.3f}) may cause slow convergence",
                UserWarning,
            )
        return v

    @classmethod
    def fast(cls) -> _NewtonConfig:
        """Create configuration optimized for speed."""
        return cls(max_iterations=10, tolerance=1e-4, damping_factor=0.8, line_search=False, verbose=False)

    @classmethod
    def accurate(cls) -> _NewtonConfig:
        """Create configuration optimized for accuracy."""
        return cls(max_iterations=50, tolerance=1e-8, damping_factor=1.0, line_search=False, verbose=False)

    @classmethod
    def research(cls) -> _NewtonConfig:
        """Create configuration optimized for research with detailed logging."""
        return cls(max_iterations=100, tolerance=1e-10, damping_factor=1.0, line_search=True, verbose=True)

    model_config = ConfigDict(validate_assignment=True)


class _PicardConfig(BaseModel):
    """
    Internal Picard iteration configuration for MFGSolverConfig.

    NOTE: This is an internal class. For public API, use core.PicardConfig.

    Provides validation for Picard method parameters including convergence
    hierarchy checks and numerical stability validation.
    """

    max_iterations: int = Field(20, ge=1, le=500, description="Maximum number of Picard iterations")
    tolerance: float = Field(1e-3, gt=1e-12, le=1.0, description="Convergence tolerance for Picard method")
    damping_factor: float = Field(0.5, gt=0.0, le=1.0, description="Damping parameter for Picard updates")
    adaptive_damping: bool = Field(False, description="Whether to use adaptive damping")
    verbose: bool = Field(False, description="Whether to print Picard iteration details")

    @field_validator("tolerance")
    @classmethod
    def validate_convergence_feasibility(cls, v: float, info: Any) -> float:
        """Validate tolerance is achievable within iteration limits."""
        max_iter = info.data.get("max_iterations", 20) if info.data else 20
        if v < 1e-8 and max_iter < 50:
            warnings.warn(
                f"Strict tolerance ({v:.2e}) with few iterations ({max_iter}) may not converge",
                UserWarning,
            )
        return v

    @classmethod
    def fast(cls) -> _PicardConfig:
        """Create configuration optimized for speed."""
        return cls(max_iterations=10, tolerance=1e-2, damping_factor=0.8, adaptive_damping=False, verbose=False)

    @classmethod
    def accurate(cls) -> _PicardConfig:
        """Create configuration optimized for accuracy."""
        return cls(max_iterations=100, tolerance=1e-6, damping_factor=0.3, adaptive_damping=True, verbose=False)

    model_config = ConfigDict(validate_assignment=True)


class _GFDMConfig(BaseModel):
    """
    Internal GFDM configuration for MFGSolverConfig.

    NOTE: This is an internal class. For public API, use mfg_methods.GFDMConfig.

    Provides validation for Generalized Finite Difference Method parameters
    including weight function validation and constraint checking.
    """

    delta: float = Field(0.1, gt=0.0, le=1.0, description="GFDM delta parameter")
    weight_function: str = Field("gaussian", description="Weight function type")
    use_qp_constraints: bool = Field(True, description="Whether to use QP constraints")
    constraint_tolerance: float = Field(1e-8, gt=0.0, le=1e-3, description="QP constraint tolerance")
    max_neighbors: int = Field(10, ge=3, le=50, description="Maximum number of neighbors for GFDM")

    @field_validator("weight_function")
    @classmethod
    def validate_weight_function(cls, v: str) -> str:
        """Validate weight function type."""
        allowed_functions = {"gaussian", "linear", "cubic", "quintic"}
        if v not in allowed_functions:
            raise ValueError(f"Weight function must be one of {allowed_functions}")
        return v

    @field_validator("delta")
    @classmethod
    def validate_delta_stability(cls, v: float) -> float:
        """Validate delta parameter for stability."""
        if v < 0.01:
            warnings.warn(
                f"Very small delta ({v:.3f}) may cause numerical instability",
                UserWarning,
            )
        if v > 0.5:
            warnings.warn(f"Large delta ({v:.3f}) may reduce accuracy", UserWarning)
        return v

    model_config = ConfigDict(validate_assignment=True)


class _ParticleConfig(BaseModel):
    """
    Internal particle method configuration for MFGSolverConfig.

    NOTE: This is an internal class. For public API, use mfg_methods.ParticleConfig.

    Provides validation for particle-based methods including count validation,
    boundary condition checking, and KDE parameter validation.
    """

    num_particles: int = Field(5000, ge=100, le=1000000, description="Number of particles")
    kde_bandwidth: float = Field(0.01, gt=0.0, le=1.0, description="KDE bandwidth parameter")
    boundary_treatment: str = Field("reflection", description="Boundary treatment method")
    resampling_method: str = Field("systematic", description="Particle resampling method")
    adaptive_particles: bool = Field(False, description="Whether to use adaptive particle count")

    @field_validator("boundary_treatment")
    @classmethod
    def validate_boundary_treatment(cls, v: str) -> str:
        """Validate boundary treatment method."""
        allowed_treatments = {"reflection", "absorption", "periodic"}
        if v not in allowed_treatments:
            raise ValueError(f"Boundary treatment must be one of {allowed_treatments}")
        return v

    @field_validator("resampling_method")
    @classmethod
    def validate_resampling_method(cls, v: str) -> str:
        """Validate resampling method."""
        allowed_methods = {"systematic", "multinomial", "residual", "stratified"}
        if v not in allowed_methods:
            raise ValueError(f"Resampling method must be one of {allowed_methods}")
        return v

    @field_validator("num_particles")
    @classmethod
    def validate_particle_count(cls, v: int) -> int:
        """Validate particle count for computational efficiency."""
        if v < 1000:
            warnings.warn(f"Few particles ({v}) may give poor density approximation", UserWarning)
        if v > 100000:
            warnings.warn(f"Many particles ({v}) may be computationally expensive", UserWarning)
        return v

    model_config = ConfigDict(validate_assignment=True)


class _HJBConfig(BaseModel):
    """
    Internal HJB solver configuration for MFGSolverConfig.

    NOTE: This is an internal class. For public API, use mfg_methods.HJBConfig.

    Combines Newton and GFDM configurations with validation for
    compatibility and numerical stability.
    """

    newton: _NewtonConfig = Field(
        default_factory=lambda: _NewtonConfig(
            max_iterations=30, tolerance=1e-6, damping_factor=1.0, line_search=False, verbose=False
        ),
        description="Newton method configuration",
    )
    gfdm: _GFDMConfig = Field(
        default_factory=lambda: _GFDMConfig(
            delta=0.1, weight_function="gaussian", use_qp_constraints=True, constraint_tolerance=1e-10, max_neighbors=20
        ),
        description="GFDM configuration",
    )
    solver_type: str = Field("gfdm_qp", description="HJB solver type")

    @field_validator("solver_type")
    @classmethod
    def validate_solver_type(cls, v: str) -> str:
        """Validate HJB solver type."""
        allowed_types = {"gfdm_qp", "gfdm_tuned", "fdm", "semi_lagrangian"}
        if v not in allowed_types:
            raise ValueError(f"HJB solver type must be one of {allowed_types}")
        return v

    @model_validator(mode="after")
    def validate_newton_gfdm_compatibility(self) -> _HJBConfig:
        """Validate compatibility between Newton and GFDM configurations."""
        newton_config = self.newton
        gfdm_config = self.gfdm

        if newton_config and gfdm_config:
            # Check tolerance hierarchy
            if newton_config.tolerance > gfdm_config.constraint_tolerance * 100:
                warnings.warn(
                    "Newton tolerance much larger than GFDM constraint tolerance - may affect convergence",
                    UserWarning,
                )

        return self

    model_config = ConfigDict(validate_assignment=True)


class _FPConfig(BaseModel):
    """
    Internal FP solver configuration for MFGSolverConfig.

    NOTE: This is an internal class. For public API, use mfg_methods.FPConfig.

    Combines particle configuration with FP-specific parameters
    and cross-validation for numerical stability.
    """

    particle: _ParticleConfig = Field(
        default_factory=lambda: _ParticleConfig(
            num_particles=5000,
            kde_bandwidth=0.01,
            boundary_treatment="reflection",
            resampling_method="systematic",
            adaptive_particles=False,
        ),
        description="Particle method configuration",
    )
    solver_type: str = Field("particle", description="FP solver type")
    time_integration: str = Field("euler", description="Time integration method")

    @field_validator("solver_type")
    @classmethod
    def validate_solver_type(cls, v: str) -> str:
        """Validate FP solver type."""
        allowed_types = {"particle", "fdm", "hybrid"}
        if v not in allowed_types:
            raise ValueError(f"FP solver type must be one of {allowed_types}")
        return v

    @field_validator("time_integration")
    @classmethod
    def validate_time_integration(cls, v: str) -> str:
        """Validate time integration method."""
        allowed_methods = {"euler", "rk4", "implicit_euler", "crank_nicolson"}
        if v not in allowed_methods:
            raise ValueError(f"Time integration must be one of {allowed_methods}")
        return v

    model_config = ConfigDict(validate_assignment=True)


class MFGSolverConfig(BaseModel):
    """
    Master MFG solver configuration with comprehensive validation.

    Combines all solver configurations with cross-validation for
    numerical stability, convergence properties, and physical constraints.

    NOTE: This class uses internal config classes (_NewtonConfig, _HJBConfig, etc.)
    that are specific to the legacy MFGSolverConfig API. For method-specific configs,
    use the canonical classes from mfg_methods.py (HJBConfig, FPConfig, etc.).
    """

    # Core solver configurations (using internal classes)
    newton: _NewtonConfig = Field(
        default_factory=lambda: _NewtonConfig(
            max_iterations=30, tolerance=1e-6, damping_factor=1.0, line_search=False, verbose=False
        ),
        description="Newton method configuration",
    )
    picard: _PicardConfig = Field(
        default_factory=lambda: _PicardConfig(
            max_iterations=20, tolerance=1e-3, damping_factor=0.5, adaptive_damping=False, verbose=False
        ),
        description="Picard iteration configuration",
    )
    hjb: _HJBConfig = Field(
        default_factory=lambda: _HJBConfig(solver_type="gfdm_qp"), description="HJB solver configuration"
    )
    fp: _FPConfig = Field(
        default_factory=lambda: _FPConfig(solver_type="fdm", time_integration="implicit_euler"),
        description="FP solver configuration",
    )

    # Global solver settings
    return_structured: bool = Field(True, description="Whether to return structured result objects")
    enable_warm_start: bool = Field(False, description="Whether to enable warm start capability")
    convergence_tolerance: float = Field(1e-5, gt=1e-12, le=1e-1, description="Global convergence tolerance")
    strict_convergence_errors: bool = Field(
        True,
        description="Whether to raise exceptions for convergence failures (True) or issue warnings (False)",
    )

    # Metadata and tracking
    experiment_name: str | None = Field(None, description="Name for experiment tracking")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Configuration creation timestamp")

    @model_validator(mode="after")
    def validate_tolerance_hierarchy(self) -> MFGSolverConfig:
        """Validate tolerance hierarchy across all methods."""
        newton_tol = self.newton.tolerance
        picard_tol = self.picard.tolerance
        global_tol = self.convergence_tolerance

        # Newton should be strictest, then global, then Picard
        if newton_tol > global_tol:
            warnings.warn(
                f"Newton tolerance ({newton_tol:.2e}) > global tolerance ({global_tol:.2e})",
                UserWarning,
            )

        if global_tol > picard_tol:
            warnings.warn(
                f"Global tolerance ({global_tol:.2e}) > Picard tolerance ({picard_tol:.2e})",
                UserWarning,
            )

        return self

    @field_validator("convergence_tolerance")
    @classmethod
    def validate_global_convergence(cls, v: float) -> float:
        """Validate global convergence tolerance is reasonable."""
        if v < 1e-10:
            warnings.warn(
                f"Very strict global tolerance ({v:.2e}) may be difficult to achieve",
                UserWarning,
            )
        return v

    def to_notebook_metadata(self) -> dict[str, Any]:
        """Convert configuration to notebook metadata format."""
        return {
            "experiment_name": self.experiment_name,
            "config_version": "2.0_pydantic",
            "created_at": self.created_at.isoformat(),
            "solver_config": self.dict(exclude={"created_at"}),
            "validation_passed": True,
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to legacy parameter format for backward compatibility."""
        return {
            # Newton parameters (new naming)
            "max_newton_iterations": self.newton.max_iterations,
            "newton_tolerance": self.newton.tolerance,
            # Picard parameters (new naming)
            "max_picard_iterations": self.picard.max_iterations,
            "picard_tolerance": self.picard.tolerance,
            # Legacy parameter names (for backward compatibility)
            "NiterNewton": self.newton.max_iterations,
            "l2errBoundNewton": self.newton.tolerance,
            "max_iterations": self.picard.max_iterations,
            "l2errBoundPicard": self.picard.tolerance,
            "l2errBound": self.convergence_tolerance,
            # Additional parameters
            "return_structured": self.return_structured,
            "enable_warm_start": self.enable_warm_start,
        }

    @classmethod
    def fast(cls) -> MFGSolverConfig:
        """Create configuration optimized for speed."""
        return cls(
            newton=_NewtonConfig.fast(),
            picard=_PicardConfig.fast(),
            convergence_tolerance=1e-3,
            return_structured=True,
            enable_warm_start=False,
            strict_convergence_errors=False,
            experiment_name="fast_config",
        )

    @classmethod
    def accurate(cls) -> MFGSolverConfig:
        """Create configuration optimized for accuracy."""
        return cls(
            newton=_NewtonConfig.accurate(),
            picard=_PicardConfig.accurate(),
            convergence_tolerance=1e-7,
            return_structured=True,
            enable_warm_start=False,
            strict_convergence_errors=True,
            experiment_name="accurate_config",
        )

    @classmethod
    def research(cls) -> MFGSolverConfig:
        """Create configuration optimized for research."""
        return cls(
            newton=_NewtonConfig.research(),
            picard=_PicardConfig.accurate(),
            convergence_tolerance=1e-8,
            return_structured=True,
            enable_warm_start=True,
            strict_convergence_errors=True,
            experiment_name="research_config",
        )

    model_config = ConfigDict(validate_assignment=True, env_prefix="MFG_")


# Convenience factory functions for backward compatibility
def create_fast_config() -> MFGSolverConfig:
    """Create fast configuration using Pydantic validation."""
    return MFGSolverConfig.fast()


def create_accurate_config() -> MFGSolverConfig:
    """Create accurate configuration using Pydantic validation."""
    return MFGSolverConfig.accurate()


def create_research_config() -> MFGSolverConfig:
    """Create research configuration using Pydantic validation."""
    return MFGSolverConfig.research()


def extract_legacy_parameters(config: MFGSolverConfig, **kwargs: Any) -> dict[str, Any]:
    """
    Extract legacy parameters from Pydantic config with automatic validation.

    Args:
        config: Pydantic MFGSolverConfig instance
        **kwargs: Additional parameters to override

    Returns:
        Dictionary with legacy parameter names and validated values
    """
    # Start with validated Pydantic configuration
    legacy_params = config.to_legacy_dict()

    # Override with any additional kwargs (with validation warnings)
    for key, value in kwargs.items():
        if key in legacy_params:
            original_value = legacy_params[key]
            if original_value != value:
                warnings.warn(
                    f"Overriding validated parameter {key}: {original_value} -> {value}",
                    UserWarning,
                )
        legacy_params[key] = value

    return legacy_params


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# These aliases allow existing code to import the internal config classes
# using their original names. New code should use mfg_methods.py classes.

NewtonConfig = _NewtonConfig
PicardConfig = _PicardConfig
GFDMConfig = _GFDMConfig
ParticleConfig = _ParticleConfig
HJBConfig = _HJBConfig
FPConfig = _FPConfig
