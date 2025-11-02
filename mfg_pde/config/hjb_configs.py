"""
HJB solver configuration classes.

This module provides configuration for all HJB solver methods:
- FDM (Finite Difference Method)
- GFDM (Generalized Finite Difference Method / meshfree)
- Semi-Lagrangian
- WENO (Weighted Essentially Non-Oscillatory)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class NewtonConfig(BaseModel):
    """
    Newton iteration configuration for HJB nonlinear solver.

    Attributes
    ----------
    max_iterations : int
        Maximum number of Newton iterations (default: 10)
    tolerance : float
        Convergence tolerance (default: 1e-6)
    relaxation : float
        Relaxation parameter for Newton updates (default: 1.0)
    """

    max_iterations: int = Field(default=10, ge=1)
    tolerance: float = Field(default=1e-6, gt=0)
    relaxation: float = Field(default=1.0, gt=0, le=1.0)


class FDMHJBConfig(BaseModel):
    """
    Finite Difference Method specific configuration.

    Attributes
    ----------
    scheme : Literal["central", "upwind", "lax_friedrichs"]
        Spatial discretization scheme (default: central)
    time_stepping : Literal["explicit", "implicit", "crank_nicolson"]
        Time stepping method (default: implicit)
    """

    scheme: Literal["central", "upwind", "lax_friedrichs"] = "central"
    time_stepping: Literal["explicit", "implicit", "crank_nicolson"] = "implicit"


class GFDMConfig(BaseModel):
    """
    Generalized Finite Difference Method (meshfree/particle collocation) configuration.

    GFDM is used for particle-based/meshfree discretizations, particularly
    useful for high-dimensional problems and complex geometries.

    Attributes
    ----------
    delta : float
        Support radius for local cloud of points (default: 0.1)
    stencil_size : int
        Number of neighbors in local stencil (default: 20)
    qp_optimization_level : Literal["none", "auto", "always"]
        Quadratic programming monotonicity enforcement (default: auto)
    monotonicity_check : bool
        Check monotonicity violations (default: True)
    adaptive_qp : bool
        Use adaptive QP threshold (default: False)
    qp_threshold : float
        Threshold for monotonicity violations before QP (default: 1e-6)
    """

    delta: float = Field(default=0.1, gt=0)
    stencil_size: int = Field(default=20, ge=5)
    qp_optimization_level: Literal["none", "auto", "always"] = "auto"
    monotonicity_check: bool = True
    adaptive_qp: bool = False
    qp_threshold: float = Field(default=1e-6, gt=0)

    @model_validator(mode="after")
    def validate_qp_settings(self) -> GFDMConfig:
        """Validate QP-related settings are consistent."""
        if self.qp_optimization_level == "none" and self.adaptive_qp:
            raise ValueError("adaptive_qp cannot be True when qp_optimization_level is 'none'")
        return self


class SLConfig(BaseModel):
    """
    Semi-Lagrangian method configuration.

    Semi-Lagrangian methods follow characteristics backward in time,
    making them well-suited for convection-dominated problems.

    Attributes
    ----------
    interpolation_method : Literal["linear", "cubic", "rbf"]
        Interpolation method for foot-of-characteristic (default: cubic)
    rk_order : Literal[1, 2, 3, 4]
        Runge-Kutta order for characteristic tracing (default: 2)
    cfl_number : float
        CFL number for stability (default: 0.5)
    """

    interpolation_method: Literal["linear", "cubic", "rbf"] = "cubic"
    rk_order: Literal[1, 2, 3, 4] = 2
    cfl_number: float = Field(default=0.5, gt=0, le=1.0)


class WENOConfig(BaseModel):
    """
    WENO (Weighted Essentially Non-Oscillatory) scheme configuration.

    WENO schemes provide high-order accuracy while maintaining shock-capturing
    capability, useful for problems with discontinuities.

    Attributes
    ----------
    weno_order : Literal[3, 5, 7]
        WENO scheme order (default: 5)
    flux_splitting : Literal["lax_friedrichs", "roe", "local_lax"]
        Flux splitting method (default: lax_friedrichs)
    epsilon : float
        Smoothness indicator parameter (default: 1e-6)
    """

    weno_order: Literal[3, 5, 7] = 5
    flux_splitting: Literal["lax_friedrichs", "roe", "local_lax"] = "lax_friedrichs"
    epsilon: float = Field(default=1e-6, gt=0)


class HJBConfig(BaseModel):
    """
    HJB solver configuration.

    This configuration specifies which HJB solver method to use and its
    method-specific parameters.

    Attributes
    ----------
    method : Literal["fdm", "gfdm", "semi_lagrangian", "weno"]
        HJB solver method (default: fdm)
    accuracy_order : int
        Numerical accuracy order (1-5, default: 2)
    boundary_conditions : Literal["dirichlet", "neumann", "periodic"]
        Boundary condition type (default: neumann)
    newton : NewtonConfig
        Newton iteration configuration
    fdm_config : FDMHJBConfig | None
        FDM-specific configuration (auto-populated if method="fdm")
    gfdm_config : GFDMConfig | None
        GFDM-specific configuration (auto-populated if method="gfdm")
    sl_config : SLConfig | None
        Semi-Lagrangian specific configuration (auto-populated if method="semi_lagrangian")
    weno_config : WENOConfig | None
        WENO-specific configuration (auto-populated if method="weno")

    Examples
    --------
    >>> # FDM solver
    >>> config = HJBConfig(method="fdm", accuracy_order=2)

    >>> # GFDM solver with custom settings
    >>> config = HJBConfig(
    ...     method="gfdm",
    ...     gfdm_config=GFDMConfig(delta=0.15, stencil_size=25)
    ... )

    >>> # Semi-Lagrangian with RBF interpolation
    >>> config = HJBConfig(
    ...     method="semi_lagrangian",
    ...     sl_config=SLConfig(interpolation_method="rbf", rk_order=4)
    ... )
    """

    method: Literal["fdm", "gfdm", "semi_lagrangian", "weno"] = "fdm"
    accuracy_order: int = Field(default=2, ge=1, le=5)
    boundary_conditions: Literal["dirichlet", "neumann", "periodic"] = "neumann"
    newton: NewtonConfig = Field(default_factory=NewtonConfig)

    # Method-specific configs (auto-populated based on method)
    fdm_config: FDMHJBConfig | None = None
    gfdm_config: GFDMConfig | None = None
    sl_config: SLConfig | None = None
    weno_config: WENOConfig | None = None

    @model_validator(mode="after")
    def validate_method_config(self) -> HJBConfig:
        """Auto-populate method-specific config if not provided."""
        if self.method == "fdm" and self.fdm_config is None:
            self.fdm_config = FDMHJBConfig()
        elif self.method == "gfdm" and self.gfdm_config is None:
            self.gfdm_config = GFDMConfig()
        elif self.method == "semi_lagrangian" and self.sl_config is None:
            self.sl_config = SLConfig()
        elif self.method == "weno" and self.weno_config is None:
            self.weno_config = WENOConfig()
        return self

    def get_method_config(self) -> FDMHJBConfig | GFDMConfig | SLConfig | WENOConfig:
        """
        Get the active method-specific configuration.

        Returns
        -------
        FDMHJBConfig | GFDMConfig | SLConfig | WENOConfig
            Method-specific configuration object

        Raises
        ------
        ValueError
            If method-specific config is not available
        """
        if self.method == "fdm":
            if self.fdm_config is None:
                raise ValueError("FDM config not initialized")
            return self.fdm_config
        elif self.method == "gfdm":
            if self.gfdm_config is None:
                raise ValueError("GFDM config not initialized")
            return self.gfdm_config
        elif self.method == "semi_lagrangian":
            if self.sl_config is None:
                raise ValueError("Semi-Lagrangian config not initialized")
            return self.sl_config
        elif self.method == "weno":
            if self.weno_config is None:
                raise ValueError("WENO config not initialized")
            return self.weno_config
        else:
            raise ValueError(f"Unknown method: {self.method}")
