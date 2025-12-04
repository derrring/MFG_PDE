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

# Array validation (specialized utility)
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
    PicardConfig,
    SolverConfig,
)

# FP solver configurations
from .fp_configs import (
    FDMFPConfig,
    FPConfig,
    NetworkConfig,
    ParticleConfig,
)

# HJB solver configurations
from .hjb_configs import (
    FDMHJBConfig,
    GFDMConfig,
    HJBConfig,
    NewtonConfig,
    SLConfig,
    WENOConfig,
)

# YAML I/O
from .io import load_solver_config, save_solver_config, validate_yaml_config

# Pydantic-based MFG solver config (for backward compatibility)
from .pydantic_config import (
    MFGSolverConfig,
    create_accurate_config,
    create_fast_config,
    create_research_config,
    extract_legacy_parameters,
)

# =============================================================================
# OPTIONAL: OmegaConf-based configuration management
# =============================================================================

try:
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
    # HJB configs
    "HJBConfig",
    "NewtonConfig",
    "FDMHJBConfig",
    "GFDMConfig",
    "SLConfig",
    "WENOConfig",
    # FP configs
    "FPConfig",
    "FDMFPConfig",
    "ParticleConfig",
    "NetworkConfig",
    # I/O
    "load_solver_config",
    "save_solver_config",
    "validate_yaml_config",
    # Backward compatibility
    "MFGSolverConfig",
    "create_fast_config",
    "create_accurate_config",
    "create_research_config",
    "extract_legacy_parameters",
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
        ]
    )
