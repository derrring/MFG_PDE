"""
Configuration management for MFG_PDE solvers.

This module provides the unified solver configuration system with three usage patterns:
1. YAML files (recommended) - For experiments and reproducibility
2. Builder API (programmatic) - For dynamic configuration
3. Presets (domain-specific) - Ready-to-use configurations

Quick Start
-----------
>>> # From YAML file
>>> from mfg_pde.config import load_solver_config
>>> config = load_solver_config("experiments/baseline.yaml")

>>> # Using builder API
>>> from mfg_pde.config import ConfigBuilder
>>> config = (
...     ConfigBuilder()
...     .solver_hjb(method="fdm", accuracy_order=2)
...     .solver_fp_particle(num_particles=5000)
...     .picard(max_iterations=50)
...     .build()
... )

>>> # Using presets
>>> from mfg_pde.config import presets
>>> config = presets.accurate_solver()
>>> config = presets.crowd_dynamics_solver()

Key Principle
-------------
Configuration is for SOLVERS (how to solve), not PROBLEMS (what to solve):
- MFGProblem (Python): Mathematical definition (g, H, ρ₀, geometry)
- SolverConfig (YAML/Python): Algorithmic choices (method, tolerance, backend)
"""

# =============================================================================
# NEW UNIFIED CONFIG SYSTEM (Phase 3.2 - RECOMMENDED)
# =============================================================================

# Core configuration classes
# Presets module (use via: from mfg_pde.config import presets)
# Issue deprecation warnings for old configuration systems
import warnings as _warnings

from . import presets

# =============================================================================
# OLD CONFIG SYSTEMS (Backward Compatibility - DEPRECATED)
# =============================================================================
# Advanced array and tensor validation (specialized, orthogonal to main system)
from .array_validation import (
    ArrayValidationConfig,
    CollocationConfig,
    ExperimentConfig,
    MFGArrays,
    MFGGridConfig,
)

# Builder API
from .builder import ConfigBuilder
from .core import BackendConfig, LoggingConfig, PicardConfig, SolverConfig

# FP solver configurations
from .fp_configs import FDMFPConfig, FPConfig, NetworkConfig, ParticleConfig

# HJB solver configurations
from .hjb_configs import FDMHJBConfig, GFDMConfig, HJBConfig, NewtonConfig, SLConfig, WENOConfig

# YAML I/O
from .io import load_solver_config, save_solver_config, validate_yaml_config

# Legacy compatibility (with deprecation warnings)
from .legacy import (
    accurate_config,
    create_accurate_config,
    create_default_config,
    create_fast_config,
    create_research_config,
    crowd_dynamics_config,
    educational_config,
    epidemic_config,
    fast_config,
    financial_config,
    large_scale_config,
    production_config,
    research_config,
    traffic_config,
)

# Old modern_config.py (DEPRECATED - unified config replaces these, will be removed in v0.15.0)
from .modern_config import PresetConfig, create_config
from .modern_config import SolverConfig as ModernSolverConfig

# Old Pydantic-based configurations (DEPRECATED - unified config replaces these, will be removed in v0.15.0)
from .pydantic_config import FPConfig as PydanticFPConfig
from .pydantic_config import GFDMConfig as PydanticGFDMConfig
from .pydantic_config import HJBConfig as PydanticHJBConfig
from .pydantic_config import (
    MFGSolverConfig,  # Backward compatibility - unaliased import
    extract_legacy_parameters,
)
from .pydantic_config import MFGSolverConfig as PydanticMFGSolverConfig
from .pydantic_config import NewtonConfig as PydanticNewtonConfig
from .pydantic_config import ParticleConfig as PydanticParticleConfig
from .pydantic_config import PicardConfig as PydanticPicardConfig

# Old dataclass-based configurations (DEPRECATED - will be removed in v0.15.0)
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

_warnings.warn(
    "The following legacy configuration imports are deprecated and will be removed in v0.15.0:\n"
    "  - modern_config: PresetConfig, create_config, ModernSolverConfig\n"
    "  - pydantic_config: PydanticFPConfig, PydanticGFDMConfig, PydanticHJBConfig, etc.\n"
    "  - solver_config: DataclassFPConfig, DataclassGFDMConfig, DataclassHJBConfig, etc.\n"
    "Please migrate to the unified configuration system:\n"
    "  from mfg_pde.config import SolverConfig, FPConfig, HJBConfig, create_solver_config",
    DeprecationWarning,
    stacklevel=2,
)

# OmegaConf-based configuration management (optional, specialized)
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
    # -------------------------------------------------------------------------
    # NEW UNIFIED CONFIG SYSTEM (RECOMMENDED)
    # -------------------------------------------------------------------------
    # Core classes
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
    # Builder & I/O
    "ConfigBuilder",
    "load_solver_config",
    "save_solver_config",
    "validate_yaml_config",
    # Presets (use via: presets.fast_solver(), etc.)
    "presets",
    # -------------------------------------------------------------------------
    # LEGACY COMPATIBILITY (DEPRECATED - will be removed in v1.0.0)
    # -------------------------------------------------------------------------
    # Dataclass-style (with deprecation warnings)
    "create_fast_config",
    "create_accurate_config",
    "create_research_config",
    "create_default_config",
    "create_fast_config_dataclass",
    "create_accurate_config_dataclass",
    "create_research_config_dataclass",
    "create_default_config_dataclass",
    # Modern-style (with deprecation warnings)
    "fast_config",
    "accurate_config",
    "research_config",
    "crowd_dynamics_config",
    "traffic_config",
    "epidemic_config",
    "financial_config",
    "large_scale_config",
    "production_config",
    "educational_config",
    # Old config classes (deprecated)
    "MFGSolverConfig",  # Backward compatibility - Pydantic version
    "PydanticFPConfig",
    "PydanticGFDMConfig",
    "PydanticHJBConfig",
    "PydanticMFGSolverConfig",
    "PydanticNewtonConfig",
    "PydanticParticleConfig",
    "PydanticPicardConfig",
    "DataclassFPConfig",
    "DataclassGFDMConfig",
    "DataclassHJBConfig",
    "DataclassMFGSolverConfig",
    "DataclassNewtonConfig",
    "DataclassParticleConfig",
    "DataclassPicardConfig",
    "ModernSolverConfig",
    "PresetConfig",
    "create_config",
    "extract_legacy_parameters",
    # -------------------------------------------------------------------------
    # SPECIALIZED (orthogonal to main config system)
    # -------------------------------------------------------------------------
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
