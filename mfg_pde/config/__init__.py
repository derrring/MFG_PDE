"""
Configuration management for MFG_PDE solvers.

This module provides structured configuration objects for all solver components,
replacing scattered constructor parameters with organized, validated config classes.

The module includes both original dataclass-based configurations and enhanced
Pydantic-based configurations with automatic validation and serialization.
"""

# Advanced array and tensor validation
from .array_validation import ArrayValidationConfig, CollocationConfig, ExperimentConfig, MFGArrays, MFGGridConfig

# Enhanced Pydantic-based configurations (recommended for new code)
from .pydantic_config import (
    FPConfig,
    GFDMConfig,
    HJBConfig,
    MFGSolverConfig,
    NewtonConfig,
    ParticleConfig,
    PicardConfig,
    create_accurate_config,
    create_fast_config,
    create_research_config,
    extract_legacy_parameters,
)

# Original dataclass-based configurations (for backward compatibility)
from .solver_config import FPConfig as DataclassFPConfig
from .solver_config import GFDMConfig as DataclassGFDMConfig
from .solver_config import HJBConfig as DataclassHJBConfig
from .solver_config import MFGSolverConfig as DataclassMFGSolverConfig
from .solver_config import NewtonConfig as DataclassNewtonConfig
from .solver_config import ParticleConfig as DataclassParticleConfig
from .solver_config import PicardConfig as DataclassPicardConfig
from .solver_config import create_accurate_config as create_accurate_config_dataclass
from .solver_config import create_default_config as create_default_config_dataclass
from .solver_config import create_fast_config as create_fast_config_dataclass
from .solver_config import create_research_config as create_research_config_dataclass

# OmegaConf-based configuration management (if available)
try:
    from .omegaconf_manager import (
        OmegaConfManager,
        create_omega_manager,
        create_parameter_sweep_configs,
        load_beach_config,
        load_experiment_config,
    )

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

# Default to Pydantic configurations for new code
__all__ = [
    # Pydantic configurations (recommended)
    "NewtonConfig",
    "PicardConfig",
    "GFDMConfig",
    "ParticleConfig",
    "HJBConfig",
    "FPConfig",
    "MFGSolverConfig",
    "create_fast_config",
    "create_accurate_config",
    "create_research_config",
    "extract_legacy_parameters",
    # Array validation
    "ArrayValidationConfig",
    "MFGGridConfig",
    "MFGArrays",
    "CollocationConfig",
    "ExperimentConfig",
    # Dataclass configurations (backward compatibility)
    "DataclassNewtonConfig",
    "DataclassPicardConfig",
    "DataclassGFDMConfig",
    "DataclassParticleConfig",
    "DataclassHJBConfig",
    "DataclassFPConfig",
    "DataclassMFGSolverConfig",
    "create_default_config_dataclass",
    "create_fast_config_dataclass",
    "create_accurate_config_dataclass",
    "create_research_config_dataclass",
]

# Add OmegaConf functionality if available
if OMEGACONF_AVAILABLE:
    __all__.extend(
        [
            "OmegaConfManager",
            "create_omega_manager",
            "load_beach_config",
            "load_experiment_config",
            "create_parameter_sweep_configs",
        ]
    )
