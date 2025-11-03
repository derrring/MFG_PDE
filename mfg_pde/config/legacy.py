"""
Legacy configuration support for backward compatibility.

This module provides compatibility shims for old configuration functions.
All functions here are deprecated and will be removed in v2.0.0.

Migration Path
--------------
v0.9.0 (current): Deprecation warnings added
v1.0.0 (+3 months): Warnings become prominent
v2.0.0 (+6 months): Old functions removed

Users should migrate to:
- Presets: from mfg_pde.config import presets
- Builder: from mfg_pde.config import ConfigBuilder
- YAML: SolverConfig.from_yaml("config.yaml")
"""

from __future__ import annotations

from warnings import warn

from .core import SolverConfig  # noqa: TC001
from .presets import accurate_solver, fast_solver, research_solver


def create_fast_config(*args, **kwargs) -> SolverConfig:
    """
    DEPRECATED: Create fast solver configuration.

    This function is deprecated and will be removed in v2.0.0.
    Use presets.fast_solver() instead.

    Returns
    -------
    SolverConfig
        Fast solver configuration

    Examples
    --------
    OLD (deprecated):
    >>> config = create_fast_config()

    NEW (recommended):
    >>> from mfg_pde.config import presets
    >>> config = presets.fast_solver()
    """
    warn(
        "create_fast_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.fast_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fast_solver()


def create_accurate_config(*args, **kwargs) -> SolverConfig:
    """
    DEPRECATED: Create accurate solver configuration.

    This function is deprecated and will be removed in v2.0.0.
    Use presets.accurate_solver() instead.

    Returns
    -------
    SolverConfig
        Accurate solver configuration

    Examples
    --------
    OLD (deprecated):
    >>> config = create_accurate_config()

    NEW (recommended):
    >>> from mfg_pde.config import presets
    >>> config = presets.accurate_solver()
    """
    warn(
        "create_accurate_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.accurate_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return accurate_solver()


def create_research_config(*args, **kwargs) -> SolverConfig:
    """
    DEPRECATED: Create research solver configuration.

    This function is deprecated and will be removed in v2.0.0.
    Use presets.research_solver() instead.

    Returns
    -------
    SolverConfig
        Research solver configuration

    Examples
    --------
    OLD (deprecated):
    >>> config = create_research_config()

    NEW (recommended):
    >>> from mfg_pde.config import presets
    >>> config = presets.research_solver()
    """
    warn(
        "create_research_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.research_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return research_solver()


def create_default_config(*args, **kwargs) -> SolverConfig:
    """
    DEPRECATED: Create default solver configuration.

    This function is deprecated and will be removed in v2.0.0.
    Use presets.fast_solver() or presets.default_solver() instead.

    Returns
    -------
    SolverConfig
        Default solver configuration

    Examples
    --------
    OLD (deprecated):
    >>> config = create_default_config()

    NEW (recommended):
    >>> from mfg_pde.config import presets
    >>> config = presets.default_solver()
    """
    warn(
        "create_default_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.default_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fast_solver()


# Legacy aliases from modern_config.py
def fast_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.fast_solver() instead."""
    warn(
        "fast_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.fast_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fast_solver()


def accurate_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.accurate_solver() instead."""
    warn(
        "accurate_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.accurate_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return accurate_solver()


def research_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.research_solver() instead."""
    warn(
        "research_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.research_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return research_solver()


def crowd_dynamics_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.crowd_dynamics_solver() instead."""
    warn(
        "crowd_dynamics_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.crowd_dynamics_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .presets import crowd_dynamics_solver

    return crowd_dynamics_solver()


def traffic_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.traffic_flow_solver() instead."""
    warn(
        "traffic_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.traffic_flow_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .presets import traffic_flow_solver

    return traffic_flow_solver()


def epidemic_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.epidemic_solver() instead."""
    warn(
        "epidemic_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.epidemic_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .presets import epidemic_solver

    return epidemic_solver()


def financial_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.financial_solver() instead."""
    warn(
        "financial_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.financial_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .presets import financial_solver

    return financial_solver()


def large_scale_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.large_scale_solver() instead."""
    warn(
        "large_scale_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.large_scale_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .presets import large_scale_solver

    return large_scale_solver()


def production_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.production_solver() instead."""
    warn(
        "production_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.production_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .presets import production_solver

    return production_solver()


def educational_config(*args, **kwargs) -> SolverConfig:
    """DEPRECATED: Use presets.educational_solver() instead."""
    warn(
        "educational_config() is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.educational_solver()' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .presets import educational_solver

    return educational_solver()


__all__ = [  # noqa: RUF022 - Intentionally organized by category, not alphabetically
    # Dataclass-style (solver_config.py)
    "create_fast_config",
    "create_accurate_config",
    "create_research_config",
    "create_default_config",
    # Modern-style (modern_config.py)
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
]
