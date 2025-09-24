"""
Structured Configuration Schemas for MFG_PDE using OmegaConf.

This module provides type-safe configuration schemas using dataclasses,
solving the common problem of OmegaConf failing type checking by providing
static type information that mypy can understand.

⚠️ CRITICAL - Issue #28 Type Safety Solution ⚠️
=============================================
These dataclass schemas are the CORE solution for OmegaConf type checking errors.

BEFORE (Type errors):
    conf: DictConfig = OmegaConf.load("config.yaml")
    print(conf.problem.T)  # ❌ Mypy error: "DictConfig has no attribute 'problem'"

AFTER (Type safe):
    schema = OmegaConf.structured(MFGConfig)
    file_conf = OmegaConf.load("config.yaml")
    conf: MFGConfig = OmegaConf.merge(schema, file_conf)
    print(conf.problem.T)  # ✅ Type safe, autocompletes!

DO NOT modify these schemas without understanding the full typing implications.
Reference Issue #28 for complete implementation context.

Inspired by the structured configs pattern recommended for type-safe OmegaConf usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


@dataclass
class BoundaryConditionsConfig:
    """Boundary conditions configuration schema."""

    type: str = "periodic"
    left_value: float | None = None
    right_value: float | None = None


@dataclass
class InitialConditionConfig:
    """Initial condition configuration schema."""

    type: str = "gaussian"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainConfig:
    """Domain configuration schema."""

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None


@dataclass
class ProblemConfig:
    """MFG problem configuration schema."""

    name: str = "base_mfg_problem"
    type: str = "standard"
    T: float = 1.0
    Nx: int = 50
    Nt: int = 30
    domain: DomainConfig = field(default_factory=DomainConfig)
    initial_condition: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    boundary_conditions: BoundaryConditionsConfig = field(default_factory=BoundaryConditionsConfig)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class NewtonConfig:
    """Newton solver configuration schema."""

    max_iterations: int = 20
    tolerance: float = 1e-8
    line_search: bool = True


@dataclass
class HJBConfig:
    """HJB solver configuration schema."""

    method: str = "gfdm"
    boundary_handling: str = "penalty"
    penalty_weight: float = 1000.0
    newton: NewtonConfig = field(default_factory=NewtonConfig)


@dataclass
class FPConfig:
    """Fokker-Planck solver configuration schema."""

    method: str = "fdm"
    upwind_scheme: str = "central"


@dataclass
class SolverConfig:
    """Solver configuration schema."""

    type: str = "fixed_point"
    max_iterations: int = 100
    tolerance: float = 1e-6
    damping: float = 0.5
    backend: str = "numpy"
    hjb: HJBConfig = field(default_factory=HJBConfig)
    fp: FPConfig = field(default_factory=FPConfig)


@dataclass
class LoggingConfig:
    """Logging configuration schema."""

    level: str = "INFO"
    file: str | None = None


@dataclass
class VisualizationConfig:
    """Visualization configuration schema."""

    enabled: bool = True
    save_plots: bool = True
    plot_dir: str = "plots"
    formats: list[str] = field(default_factory=lambda: ["png", "html"])
    dpi: int = 300


@dataclass
class ExperimentConfig:
    """Experiment configuration schema."""

    name: str = "parameter_sweep"
    description: str = "Parameter sweep experiment"
    output_dir: str = "results"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    sweeps: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class MFGConfig:
    """Complete MFG configuration schema combining all components."""

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# Specialized configurations for common scenarios


@dataclass
class BeachProblemConfig:
    """Towel on Beach problem configuration schema."""

    problem: ProblemConfig = field(
        default_factory=lambda: ProblemConfig(
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
    solver: SolverConfig = field(default_factory=SolverConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# Type aliases for convenience
StructuredMFGConfig = MFGConfig
StructuredBeachConfig = BeachProblemConfig
