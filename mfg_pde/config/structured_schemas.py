"""
Structured Configuration Schemas for MFG_PDE using OmegaConf.

This module provides type-safe configuration schemas using dataclasses,
solving the common problem of OmegaConf failing type checking by providing
static type information that mypy can understand.

NAMING CONVENTION (v0.16+)
==========================
All OmegaConf dataclass schemas use the `*Schema` suffix to distinguish them
from Pydantic `*Config` classes in `core.py`. This follows the architecture:

- **Pydantic** (`core.py`): `*Config` suffix - Runtime validation, API safety
- **OmegaConf** (`structured_schemas.py`): `*Schema` suffix - YAML management, experiments

See `docs/development/PYDANTIC_OMEGACONF_COOPERATION.md` for full architecture.

Type Safety Solution (Issue #28)
================================
These dataclass schemas are the CORE solution for OmegaConf type checking errors.

BEFORE (Type errors):
    conf: DictConfig = OmegaConf.load("config.yaml")
    print(conf.problem.T)  # Mypy error: "DictConfig has no attribute 'problem'"

AFTER (Type safe):
    schema = OmegaConf.structured(MFGSchema)
    file_conf = OmegaConf.load("config.yaml")
    conf: MFGSchema = OmegaConf.merge(schema, file_conf)
    print(conf.problem.T)  # Type safe, autocompletes!

DO NOT modify these schemas without understanding the full typing implications.
Reference Issue #28 for complete implementation context.

Inspired by the structured configs pattern recommended for type-safe OmegaConf usage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundaryConditionsSchema:
    """Boundary conditions configuration schema."""

    type: str = "periodic"
    left_value: float | None = None
    right_value: float | None = None


@dataclass
class InitialConditionSchema:
    """Initial condition configuration schema."""

    type: str = "gaussian"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainSchema:
    """Domain configuration schema."""

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None


@dataclass
class ProblemSchema:
    """MFG problem configuration schema."""

    name: str = "base_mfg_problem"
    type: str = "standard"
    T: float = 1.0
    Nx: int = 50
    Nt: int = 30
    domain: DomainSchema = field(default_factory=DomainSchema)
    initial_condition: InitialConditionSchema = field(default_factory=InitialConditionSchema)
    boundary_conditions: BoundaryConditionsSchema = field(default_factory=BoundaryConditionsSchema)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class NewtonSchema:
    """Newton solver configuration schema."""

    max_iterations: int = 20
    tolerance: float = 1e-8
    line_search: bool = True


@dataclass
class HJBSchema:
    """HJB solver configuration schema."""

    method: str = "gfdm"
    boundary_handling: str = "penalty"
    penalty_weight: float = 1000.0
    newton: NewtonSchema = field(default_factory=NewtonSchema)


@dataclass
class FPSchema:
    """Fokker-Planck solver configuration schema."""

    method: str = "fdm"
    upwind_scheme: str = "central"


@dataclass
class SolverSchema:
    """Solver configuration schema."""

    type: str = "fixed_point"
    max_iterations: int = 100
    tolerance: float = 1e-6
    damping: float = 0.5
    backend: str = "numpy"
    hjb: HJBSchema = field(default_factory=HJBSchema)
    fp: FPSchema = field(default_factory=FPSchema)


@dataclass
class LoggingSchema:
    """Logging configuration schema."""

    level: str = "INFO"
    file: str | None = None


@dataclass
class VisualizationSchema:
    """Visualization configuration schema."""

    enabled: bool = True
    save_plots: bool = True
    plot_dir: str = "plots"
    formats: list[str] = field(default_factory=lambda: ["png", "html"])
    dpi: int = 300


@dataclass
class ExperimentSchema:
    """Experiment configuration schema."""

    name: str = "parameter_sweep"
    description: str = "Parameter sweep experiment"
    output_dir: str = "results"
    logging: LoggingSchema = field(default_factory=LoggingSchema)
    visualization: VisualizationSchema = field(default_factory=VisualizationSchema)
    sweeps: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class MFGSchema:
    """Complete MFG configuration schema combining all components."""

    problem: ProblemSchema = field(default_factory=ProblemSchema)
    solver: SolverSchema = field(default_factory=SolverSchema)
    experiment: ExperimentSchema = field(default_factory=ExperimentSchema)


# Specialized configurations for common scenarios


@dataclass
class BeachProblemSchema:
    """Towel on Beach problem configuration schema."""

    problem: ProblemSchema = field(
        default_factory=lambda: ProblemSchema(
            name="towel_on_beach",
            type="spatial_competition",
            T=2.0,
            Nx=80,
            Nt=40,
            parameters={
                "stall_position": 0.6,
                "crowd_aversion": 1.5,
                "noise_level": 0.1,
            },
        )
    )
    solver: SolverSchema = field(default_factory=SolverSchema)
    experiment: ExperimentSchema = field(default_factory=ExperimentSchema)


# Type aliases for convenience (canonical names)
StructuredMFGConfig = MFGSchema
StructuredBeachConfig = BeachProblemSchema

# =============================================================================
# BACKWARD COMPATIBILITY: Deprecated *Config aliases (v0.16 - v0.17)
# =============================================================================
# These aliases maintain backward compatibility during the migration period.
# They will be removed in v0.18. Use the *Schema names instead.

# Deprecated name → New name mapping
_DEPRECATED_MAP: dict[str, str] = {
    "BoundaryConditionsConfig": "BoundaryConditionsSchema",
    "InitialConditionConfig": "InitialConditionSchema",
    "DomainConfig": "DomainSchema",
    "ProblemConfig": "ProblemSchema",
    "NewtonConfig": "NewtonSchema",
    "HJBConfig": "HJBSchema",
    "FPConfig": "FPSchema",
    "SolverConfig": "SolverSchema",
    "LoggingConfig": "LoggingSchema",
    "VisualizationConfig": "VisualizationSchema",
    "ExperimentConfig": "ExperimentSchema",
    "MFGConfig": "MFGSchema",
    "BeachProblemConfig": "BeachProblemSchema",
}


def __getattr__(name: str) -> Any:
    """
    Module-level __getattr__ to support deprecated *Config imports.

    Allows imports of old names (e.g., HJBConfig) but warns the user
    to migrate to the new *Schema names.
    """
    if name in _DEPRECATED_MAP:
        new_name = _DEPRECATED_MAP[name]
        warnings.warn(
            f"'{name}' is deprecated and will be removed in v0.18. "
            f"Please import '{new_name}' from mfg_pde.config.structured_schemas instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export list (new canonical names)
__all__ = [
    # Canonical *Schema names (v0.16+)
    "BoundaryConditionsSchema",
    "InitialConditionSchema",
    "DomainSchema",
    "ProblemSchema",
    "NewtonSchema",
    "HJBSchema",
    "FPSchema",
    "SolverSchema",
    "LoggingSchema",
    "VisualizationSchema",
    "ExperimentSchema",
    "MFGSchema",
    "BeachProblemSchema",
    # Type aliases
    "StructuredMFGConfig",
    "StructuredBeachConfig",
]
