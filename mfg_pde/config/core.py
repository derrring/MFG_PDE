"""
Core solver configuration classes.

This module provides the unified solver configuration system for MFG_PDE.
Configurations specify HOW to solve problems (algorithmic choices), not WHAT
problems to solve (mathematical definitions - those are MFGProblem instances).

Key Principle
-------------
- MFGProblem (Python code): Mathematical definition (g, H, ρ₀, geometry)
- SolverConfig (YAML/Python): Algorithmic choices (method, tolerance, backend)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from pathlib import Path


# Type alias for solver configuration base class
# All solver configs inherit from Pydantic BaseModel
BaseConfig = BaseModel


class LoggingConfig(BaseModel):
    """
    Configuration for logging and progress reporting.

    Attributes
    ----------
    level : Literal["DEBUG", "INFO", "WARNING", "ERROR"]
        Logging level (default: INFO)
    progress_bar : bool
        Show progress bar during solving (default: True)
    save_intermediate : bool
        Save intermediate results during iteration (default: False)
    output_dir : str | None
        Directory for saving intermediate results (default: None)
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    progress_bar: bool = True
    save_intermediate: bool = False
    output_dir: str | None = None

    @model_validator(mode="after")
    def validate_output_dir(self) -> LoggingConfig:
        """Validate that output_dir is provided if save_intermediate is True."""
        if self.save_intermediate and self.output_dir is None:
            raise ValueError("output_dir must be provided when save_intermediate is True")
        return self


class BackendConfig(BaseModel):
    """
    Configuration for computational backend.

    Attributes
    ----------
    type : Literal["numpy", "jax", "pytorch"]
        Backend type (default: numpy)
    device : Literal["cpu", "gpu", "auto"]
        Compute device (default: cpu)
    precision : Literal["float32", "float64"]
        Floating point precision (default: float64)
    """

    type: Literal["numpy", "jax", "pytorch"] = "numpy"
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    precision: Literal["float32", "float64"] = "float64"

    @model_validator(mode="after")
    def validate_device(self) -> BackendConfig:
        """Validate device compatibility with backend."""
        if self.type == "numpy" and self.device == "gpu":
            raise ValueError("NumPy backend does not support GPU device. Use JAX or PyTorch.")
        return self


class PicardConfig(BaseModel):
    """
    Configuration for Picard (fixed-point) iteration with damping.

    Attributes
    ----------
    max_iterations : int
        Maximum number of iterations (default: 100)
    tolerance : float
        Convergence tolerance (default: 1e-6)
    damping_factor : float
        Damping factor θ ∈ (0, 1] for update: u^{n+1} = θu_new + (1-θ)u^n
        - 1.0: No damping (faster but may diverge)
        - 0.5: Moderate damping (balanced, default)
        - <0.3: Heavy damping (slower but more stable)
    anderson_memory : int
        Anderson acceleration memory depth (0 = disabled, default: 0)
    verbose : bool
        Print iteration progress (default: True)
    """

    max_iterations: int = Field(default=100, ge=1)
    tolerance: float = Field(default=1e-6, gt=0)
    damping_factor: float = Field(default=0.5, gt=0, le=1.0)
    anderson_memory: int = Field(default=0, ge=0)
    verbose: bool = True

    @model_validator(mode="after")
    def validate_anderson(self) -> PicardConfig:
        """Validate Anderson acceleration parameters."""
        if self.anderson_memory < 0:
            raise ValueError("anderson_memory must be non-negative")
        if self.anderson_memory > self.max_iterations:
            raise ValueError("anderson_memory cannot exceed max_iterations")
        return self


class SolverConfig(BaseModel):
    """
    Unified solver configuration.

    This class specifies HOW to solve an MFG problem (algorithmic choices),
    not WHAT problem to solve (mathematical definition).

    The problem definition (terminal cost g, Hamiltonian H, initial density ρ₀)
    is specified via MFGProblem instances.

    Attributes
    ----------
    hjb : HJBConfig
        HJB solver configuration
    fp : FPConfig
        Fokker-Planck solver configuration
    picard : PicardConfig
        Picard iteration configuration
    backend : BackendConfig
        Computational backend configuration
    logging : LoggingConfig
        Logging configuration

    Examples
    --------
    >>> # From YAML file
    >>> config = SolverConfig.from_yaml("config.yaml")

    >>> # Programmatically
    >>> config = SolverConfig(
    ...     hjb=HJBConfig(method="fdm", accuracy_order=2),
    ...     fp=FPConfig(method="particle", num_particles=5000),
    ...     picard=PicardConfig(max_iterations=50, tolerance=1e-6)
    ... )

    >>> # With builder
    >>> from mfg_pde.config import ConfigBuilder
    >>> config = (
    ...     ConfigBuilder()
    ...     .solver_hjb(method="fdm", accuracy_order=2)
    ...     .solver_fp(method="particle", num_particles=5000)
    ...     .picard(max_iterations=50)
    ...     .build()
    ... )
    """

    hjb: HJBConfig = Field(default_factory=lambda: HJBConfig())
    fp: FPConfig = Field(default_factory=lambda: FPConfig())
    picard: PicardConfig = Field(default_factory=PicardConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Parameters
        ----------
        path : str | Path
            Output file path

        Examples
        --------
        >>> config = SolverConfig(...)
        >>> config.to_yaml("experiments/baseline.yaml")
        """
        from .io import save_solver_config

        save_solver_config(self, path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SolverConfig:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        path : str | Path
            Path to YAML configuration file

        Returns
        -------
        SolverConfig
            Validated solver configuration

        Examples
        --------
        >>> config = SolverConfig.from_yaml("experiments/baseline.yaml")
        """
        from .io import load_solver_config

        return load_solver_config(path)

    def model_dump_yaml(self) -> dict:
        """
        Dump configuration as dictionary suitable for YAML serialization.

        Returns
        -------
        dict
            Configuration as nested dictionary
        """
        return self.model_dump(exclude_none=True, mode="json")


# Forward references will be resolved after HJBConfig and FPConfig are imported
from .fp_configs import FPConfig  # noqa: E402
from .hjb_configs import HJBConfig  # noqa: E402

# Update forward references
SolverConfig.model_rebuild()
