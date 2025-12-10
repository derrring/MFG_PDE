"""
Configuration management for MFG_PDE solvers.

This module provides the unified solver configuration system using Pydantic.
Configurations specify HOW to solve problems (algorithmic choices), not WHAT
problems to solve (mathematical definitions - those are MFGProblem instances).

Quick Start
-----------
>>> from mfg_pde.config import SolverConfig, HJBConfig, FPConfig
>>> config = SolverConfig(
...     hjb=HJBConfig(method="fdm", accuracy_order=2),
...     fp=FPConfig(method="particle", num_particles=5000),
... )

>>> # Or load from YAML
>>> from mfg_pde.config import load_solver_config
>>> config = load_solver_config("experiments/baseline.yaml")

Key Principle
-------------
Configuration is for SOLVERS (how to solve), not PROBLEMS (what to solve):
- MFGProblem (Python): Mathematical definition (g, H, rho_0, geometry)
- SolverConfig (YAML/Python): Algorithmic choices (method, tolerance, backend)
"""

# =============================================================================
# CORE CONFIG CLASSES (Pydantic-based)
# =============================================================================

from .array_validation import (
    ArrayValidationConfig,
    CollocationConfig,
    ExperimentConfig,
    MFGArrays,
    MFGGridConfig,
)
from .core import (
    BackendConfig,
    BaseConfig,
    LoggingConfig,
    MFGSolverConfig,
    PicardConfig,
    SolverConfig,  # Backward compatibility alias for MFGSolverConfig
)

# YAML I/O
from .io import load_solver_config, save_solver_config, validate_yaml_config

# MFG method configurations (unified)
from .mfg_methods import (
    FDMConfig,
    FPConfig,
    GFDMConfig,
    HJBConfig,
    NetworkConfig,
    NewtonConfig,
    ParticleConfig,
    SLConfig,
    WENOConfig,
)

# =============================================================================
# OPTIONAL: OmegaConf-based configuration management
# =============================================================================

try:
    # Bridge utilities for Pydantic-OmegaConf interoperability
    from .bridge import (  # noqa: F401
        bridge_to_pydantic,
        load_effective_config,
        save_effective_config,
    )
    from .omegaconf_manager import (  # noqa: F401
        OmegaConfManager,
        create_omega_manager,
        create_parameter_sweep_configs,
        load_beach_config,
        load_experiment_config,
    )

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Core classes
    "BaseConfig",
    "SolverConfig",
    "PicardConfig",
    "BackendConfig",
    "LoggingConfig",
    # Method configs (unified)
    "FDMConfig",
    "GFDMConfig",
    "SLConfig",
    "WENOConfig",
    "ParticleConfig",
    "NetworkConfig",
    "NewtonConfig",
    # Composite solver configs
    "HJBConfig",
    "FPConfig",
    # I/O
    "load_solver_config",
    "save_solver_config",
    "validate_yaml_config",
    # MFG solver config
    "MFGSolverConfig",
    # Array validation
    "ArrayValidationConfig",
    "CollocationConfig",
    "ExperimentConfig",
    "MFGArrays",
    "MFGGridConfig",
]

# Add OmegaConf functionality if available
if OMEGACONF_AVAILABLE:
    __all__.extend(
        [
            "OmegaConfManager",
            "create_omega_manager",
            "create_parameter_sweep_configs",
            "load_beach_config",
            "load_experiment_config",
            # Bridge utilities
            "bridge_to_pydantic",
            "save_effective_config",
            "load_effective_config",
        ]
    )
