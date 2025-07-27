"""
Configuration management for MFG_PDE solvers.

This module provides structured configuration objects for all solver components,
replacing scattered constructor parameters with organized, validated config classes.

The module includes both original dataclass-based configurations and enhanced 
Pydantic-based configurations with automatic validation and serialization.
"""

# Original dataclass-based configurations (for backward compatibility)
from .solver_config import (
    NewtonConfig as DataclassNewtonConfig,
    PicardConfig as DataclassPicardConfig,
    ParticleConfig as DataclassParticleConfig,
    GFDMConfig as DataclassGFDMConfig,
    MFGSolverConfig as DataclassMFGSolverConfig,
    HJBConfig as DataclassHJBConfig,
    FPConfig as DataclassFPConfig,
    create_default_config as create_default_config_dataclass,
    create_fast_config as create_fast_config_dataclass,
    create_accurate_config as create_accurate_config_dataclass,
    create_research_config as create_research_config_dataclass,
)

# Enhanced Pydantic-based configurations (recommended for new code)
from .pydantic_config import (
    NewtonConfig,
    PicardConfig,
    GFDMConfig,
    ParticleConfig,
    HJBConfig,
    FPConfig,
    MFGSolverConfig,
    create_fast_config,
    create_accurate_config,
    create_research_config,
    extract_legacy_parameters,
)

# Advanced array and tensor validation
from .array_validation import (
    ArrayValidationConfig,
    MFGGridConfig,
    MFGArrays,
    CollocationConfig,
    ExperimentConfig,
)

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
