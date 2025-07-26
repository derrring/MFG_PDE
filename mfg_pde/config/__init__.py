"""
Configuration management for MFG_PDE solvers.

This module provides structured configuration objects for all solver components,
replacing scattered constructor parameters with organized, validated config classes.
"""

from .solver_config import (
    NewtonConfig, PicardConfig, ParticleConfig, GFDMConfig,
    MFGSolverConfig, HJBConfig, FPConfig, 
    create_default_config, create_fast_config, create_accurate_config, create_research_config
)

__all__ = [
    'NewtonConfig', 'PicardConfig', 'ParticleConfig', 'GFDMConfig',
    'MFGSolverConfig', 'HJBConfig', 'FPConfig',
    'create_default_config', 'create_fast_config', 'create_accurate_config', 'create_research_config'
]