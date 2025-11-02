"""
Fokker-Planck solver configuration classes.

This module provides configuration for all FP solver methods:
- FDM (Finite Difference Method)
- Particle (particle-based methods)
- Network (graph/network methods)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class FDMFPConfig(BaseModel):
    """
    FDM-based FP solver configuration.

    Attributes
    ----------
    scheme : Literal["upwind", "central", "lax_friedrichs"]
        Spatial discretization scheme (default: upwind)
    time_stepping : Literal["explicit", "implicit"]
        Time stepping method (default: implicit)
    """

    scheme: Literal["upwind", "central", "lax_friedrichs"] = "upwind"
    time_stepping: Literal["explicit", "implicit"] = "implicit"


class ParticleConfig(BaseModel):
    """
    Particle-based FP solver configuration.

    Particle methods sample the density using stochastic differential equations
    and estimate the density via kernel density estimation (KDE) or directly
    on collocation points.

    Attributes
    ----------
    num_particles : int
        Number of particles to sample (default: 5000)
    kde_bandwidth : float | Literal["auto"]
        KDE bandwidth parameter (default: auto)
    normalization : Literal["none", "initial_only", "all"]
        Density normalization strategy (default: initial_only)
    mode : Literal["hybrid", "collocation"]
        Particle mode (default: hybrid):
        - hybrid: Sample particles, output to grid via KDE
        - collocation: Use external particles, output on particles (meshfree)
    external_particles : Any
        External collocation points (ndarray, required for collocation mode)
        NOTE: This field is excluded from serialization (use only programmatically)

    Examples
    --------
    >>> # Hybrid mode (sample particles, output to grid)
    >>> config = ParticleConfig(num_particles=5000, normalization="initial_only")

    >>> # Collocation mode (meshfree with external particles)
    >>> import numpy as np
    >>> points = np.random.uniform(0, 1, (1000, 2))
    >>> config = ParticleConfig(
    ...     mode="collocation",
    ...     external_particles=points
    ... )
    """

    num_particles: int = Field(default=5000, gt=0)
    kde_bandwidth: float | Literal["auto"] = "auto"
    normalization: Literal["none", "initial_only", "all"] = "initial_only"
    mode: Literal["hybrid", "collocation"] = "hybrid"
    external_particles: Any = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_collocation_mode(self) -> ParticleConfig:
        """Validate that external_particles is provided for collocation mode."""
        if self.mode == "collocation" and self.external_particles is None:
            raise ValueError(
                "external_particles must be provided when mode='collocation'. "
                "Note: external_particles must be set programmatically (not via YAML) "
                "as it contains numpy arrays."
            )
        if self.mode == "hybrid" and self.external_particles is not None:
            raise ValueError(
                "external_particles should not be provided for hybrid mode. "
                "Use mode='collocation' if you want to use external particles."
            )
        return self

    @model_validator(mode="after")
    def validate_kde_bandwidth(self) -> ParticleConfig:
        """Validate KDE bandwidth parameter."""
        if isinstance(self.kde_bandwidth, float) and self.kde_bandwidth <= 0:
            raise ValueError("kde_bandwidth must be positive when specified as float")
        return self


class NetworkConfig(BaseModel):
    """
    Network/graph-based FP solver configuration.

    For MFG problems defined on graphs or networks (e.g., traffic networks).

    Attributes
    ----------
    discretization_method : Literal["finite_volume", "finite_element"]
        Discretization method on network (default: finite_volume)
    """

    discretization_method: Literal["finite_volume", "finite_element"] = "finite_volume"


class FPConfig(BaseModel):
    """
    Fokker-Planck solver configuration.

    This configuration specifies which FP solver method to use and its
    method-specific parameters.

    Attributes
    ----------
    method : Literal["fdm", "particle", "network"]
        FP solver method (default: fdm)
    fdm_config : FDMFPConfig | None
        FDM-specific configuration (auto-populated if method="fdm")
    particle_config : ParticleConfig | None
        Particle-specific configuration (auto-populated if method="particle")
    network_config : NetworkConfig | None
        Network-specific configuration (auto-populated if method="network")

    Examples
    --------
    >>> # FDM solver
    >>> config = FPConfig(method="fdm")

    >>> # Particle solver with high particle count
    >>> config = FPConfig(
    ...     method="particle",
    ...     particle_config=ParticleConfig(num_particles=10000)
    ... )

    >>> # Collocation mode (meshfree)
    >>> import numpy as np
    >>> points = np.random.uniform(0, 1, (1000, 2))
    >>> config = FPConfig(
    ...     method="particle",
    ...     particle_config=ParticleConfig(mode="collocation", external_particles=points)
    ... )
    """

    method: Literal["fdm", "particle", "network"] = "fdm"

    # Method-specific configs (auto-populated based on method)
    fdm_config: FDMFPConfig | None = None
    particle_config: ParticleConfig | None = None
    network_config: NetworkConfig | None = None

    @model_validator(mode="after")
    def validate_method_config(self) -> FPConfig:
        """Auto-populate method-specific config if not provided."""
        if self.method == "fdm" and self.fdm_config is None:
            self.fdm_config = FDMFPConfig()
        elif self.method == "particle" and self.particle_config is None:
            self.particle_config = ParticleConfig()
        elif self.method == "network" and self.network_config is None:
            self.network_config = NetworkConfig()
        return self

    def get_method_config(self) -> FDMFPConfig | ParticleConfig | NetworkConfig:
        """
        Get the active method-specific configuration.

        Returns
        -------
        FDMFPConfig | ParticleConfig | NetworkConfig
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
        elif self.method == "particle":
            if self.particle_config is None:
                raise ValueError("Particle config not initialized")
            return self.particle_config
        elif self.method == "network":
            if self.network_config is None:
                raise ValueError("Network config not initialized")
            return self.network_config
        else:
            raise ValueError(f"Unknown method: {self.method}")
