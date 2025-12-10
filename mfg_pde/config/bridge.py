"""
Bridge utilities for Pydantic-OmegaConf interoperability.

This module provides generic adapters between OmegaConf DictConfigs and Pydantic
models, enabling seamless conversion between the two configuration systems.

Architecture Overview:
- **Pydantic** (`*Config`): Runtime validation, API safety, type strictness
- **OmegaConf** (`*Schema`): YAML management, CLI overrides, parameter sweeps

The bridge functions allow:
1. Converting OmegaConf configs to validated Pydantic models
2. Saving effective (resolved) configs for reproducibility

See `docs/development/PYDANTIC_OMEGACONF_COOPERATION.md` for the full guide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def bridge_to_pydantic[T: BaseModel](
    omega_cfg: DictConfig,
    pydantic_cls: type[T],
    *,
    strict: bool = True,
) -> T:
    """
    Convert an OmegaConf DictConfig to a validated Pydantic model.

    This generic adapter handles the common pattern of loading experiment
    configuration from YAML (via OmegaConf) and validating it with Pydantic.

    Parameters
    ----------
    omega_cfg : DictConfig
        OmegaConf configuration, typically loaded from YAML.
    pydantic_cls : type[T]
        Target Pydantic model class.
    strict : bool, optional
        If True, use strict validation (no type coercion). Default True.

    Returns
    -------
    T
        Validated Pydantic model instance.

    Raises
    ------
    ValidationError
        If the config fails Pydantic validation.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from mfg_pde.config import MFGSolverConfig
    >>> from mfg_pde.config.bridge import bridge_to_pydantic
    >>>
    >>> # Load from YAML
    >>> omega_cfg = OmegaConf.load("experiment.yaml")
    >>>
    >>> # Convert to validated Pydantic model
    >>> config = bridge_to_pydantic(omega_cfg, MFGSolverConfig)
    >>> print(config.tolerance)  # Type-safe access
    """
    from omegaconf import OmegaConf

    # Resolve interpolations and missing values
    OmegaConf.resolve(omega_cfg)

    # Convert to plain Python dict
    container: dict[str, Any] = OmegaConf.to_container(omega_cfg, resolve=True)  # type: ignore[assignment]

    # Validate with Pydantic
    if strict:
        return pydantic_cls.model_validate(container, strict=True)
    return pydantic_cls.model_validate(container)


def save_effective_config(
    config: BaseModel,
    output_dir: str | Path,
    *,
    filename: str = "resolved_config.json",
    include_defaults: bool = True,
) -> Path:
    """
    Save the effective (resolved) Pydantic config to a JSON file.

    This function saves the complete configuration with all defaults filled in,
    enabling full reproducibility of experiment runs.

    Parameters
    ----------
    config : BaseModel
        Pydantic configuration model (e.g., MFGSolverConfig).
    output_dir : str | Path
        Directory to save the config file.
    filename : str, optional
        Output filename. Default "resolved_config.json".
    include_defaults : bool, optional
        If True, include fields with default values. Default True.

    Returns
    -------
    Path
        Path to the saved config file.

    Examples
    --------
    >>> from mfg_pde.config import MFGSolverConfig
    >>> from mfg_pde.config.bridge import save_effective_config
    >>>
    >>> config = MFGSolverConfig(tolerance=1e-8, max_iterations=200)
    >>> path = save_effective_config(config, "results/experiment_001")
    >>> print(f"Config saved to {path}")

    Notes
    -----
    The output JSON includes all fields, including those with default values,
    ensuring the exact configuration can be reconstructed later.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / filename

    # Export to JSON-serializable dict
    if include_defaults:
        config_dict = config.model_dump(mode="json")
    else:
        config_dict = config.model_dump(mode="json", exclude_defaults=True)

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    return config_path


def load_effective_config[T: BaseModel](
    config_path: str | Path,
    pydantic_cls: type[T],
) -> T:
    """
    Load a previously saved effective config from JSON.

    Parameters
    ----------
    config_path : str | Path
        Path to the JSON config file.
    pydantic_cls : type[T]
        Target Pydantic model class.

    Returns
    -------
    T
        Validated Pydantic model instance.

    Examples
    --------
    >>> from mfg_pde.config import MFGSolverConfig
    >>> from mfg_pde.config.bridge import load_effective_config
    >>>
    >>> config = load_effective_config(
    ...     "results/experiment_001/resolved_config.json",
    ...     MFGSolverConfig
    ... )
    """
    with open(config_path) as f:
        config_dict = json.load(f)

    return pydantic_cls.model_validate(config_dict)


__all__ = [
    "bridge_to_pydantic",
    "save_effective_config",
    "load_effective_config",
]
