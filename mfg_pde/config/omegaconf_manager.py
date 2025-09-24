"""
OmegaConf-based configuration management for MFG_PDE with Structured Configs.

This module provides YAML-based configuration management using OmegaConf's
structured configs pattern, solving the common type checking issues by
providing static type information that mypy can understand.

⚠️ IMPORTANT NOTE - Issue #28 Solution ⚠️
====================================
This module implements the COMPLETE solution for OmegaConf type checking problems.
If you encounter mypy errors like:
  - "DictConfig has no attribute 'problem'"
  - "Cannot assign to a type [misc]"
  - "Name already defined (possibly by an import) [no-redef]"

DO NOT modify the import section! Instead:
1. Use the structured config methods: load_structured_mfg_config(), load_mfg_config()
2. Import proper types: from .structured_schemas import MFGConfig, TypedMFGConfig
3. Reference Issue #28 for complete implementation details
4. The current import pattern with try/except and stub classes is CORRECT

Features:
- Type-safe configuration schemas using dataclasses
- YAML-based configuration with interpolation
- Configuration composition and inheritance
- Integration with existing Pydantic configs
- Parameter sweeps and experiment management
- Full mypy compatibility via structured configs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # For type checking only
    import contextlib

    with contextlib.suppress(ImportError):
        from omegaconf import DictConfig, ListConfig, OmegaConf
        from omegaconf.errors import ConfigAttributeError, UnsupportedInterpolationType

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.errors import ConfigAttributeError

    try:
        from omegaconf.errors import UnsupportedInterpolationType
    except ImportError:
        # Fallback for older versions
        class UnsupportedInterpolationError(Exception):
            pass

        UnsupportedInterpolationType = UnsupportedInterpolationError

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    # Create fallback types
    DictConfig = dict
    ListConfig = list
    OmegaConf = type(None)
    ConfigAttributeError = AttributeError
    UnsupportedInterpolationType = Exception

from .pydantic_config import MFGSolverConfig, create_fast_config
from .structured_schemas import (
    BeachProblemConfig,
    MFGConfig,
)

logger = logging.getLogger(__name__)

# Type aliases for structured configs
if TYPE_CHECKING:
    from omegaconf import DictConfig as OmegaConfig

    # Use structured schemas for type safety
    TypedMFGConfig = MFGConfig
    TypedBeachConfig = BeachProblemConfig
else:
    OmegaConfig = Any
    TypedMFGConfig = Any
    TypedBeachConfig = Any


class OmegaConfManager:
    """
    Configuration manager using OmegaConf for YAML-based configuration.

    Features:
    - YAML configuration files with interpolation
    - Configuration composition and inheritance
    - Integration with existing Pydantic configs
    - Parameter sweeps and experiment management
    - Type validation and schema enforcement
    """

    def __init__(self, config_dir: str | Path | None = None):
        """
        Initialize OmegaConf configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        # Re-check availability at runtime
        try:
            from omegaconf import DictConfig as _DictConfig
            from omegaconf import OmegaConf as _OmegaConf

            self._OmegaConf = _OmegaConf
            self._DictConfig = _DictConfig
            self._omegaconf_available = True
        except ImportError as err:
            raise ImportError("OmegaConf is not available. Install with: pip install omegaconf") from err

        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
        self.config_dir.mkdir(exist_ok=True)
        self._schema_cache: dict[str, OmegaConfig] = {}

        # Initialize default configurations
        self._create_default_configs()

    def _create_default_configs(self) -> None:
        """Create default configuration files if they don't exist."""

        # Base MFG problem configuration
        base_mfg_config = {
            "problem": {
                "name": "base_mfg_problem",
                "T": 1.0,
                "Nx": 50,
                "Nt": 30,
                "domain": {"x_min": 0.0, "x_max": 1.0},
                "initial_condition": {
                    "type": "gaussian",
                    "parameters": {"center": 0.5, "width": 0.1},
                },
                "boundary_conditions": {
                    "m": {"type": "no_flux"},
                    "u": {"type": "neumann", "left_value": 0.0, "right_value": 0.0},
                },
            }
        }

        # Solver configuration
        solver_config = {
            "solver": {
                "type": "fixed_point",
                "max_iterations": 100,
                "tolerance": 1e-6,
                "damping": 0.5,
                "backend": "numpy",
                "hjb": {
                    "method": "gfdm",
                    "boundary_handling": "penalty",
                    "penalty_weight": 1000.0,
                    "newton": {
                        "max_iterations": 20,
                        "tolerance": 1e-8,
                        "line_search": True,
                    },
                },
                "fp": {"method": "fdm", "upwind_scheme": "central"},
            }
        }

        # Towel on Beach problem configuration
        beach_config = {
            "problem": {
                "name": "towel_on_beach",
                "type": "spatial_competition",
                "T": 2.0,
                "Nx": 80,
                "Nt": 40,
                "domain": {"x_min": 0.0, "x_max": 1.0},
                "parameters": {
                    "stall_position": 0.6,
                    "crowd_aversion": "${lambda}",  # Interpolation placeholder
                    "noise_level": 0.1,
                },
                "initial_condition": {
                    "type": "${init_type}",  # Can be overridden
                    "parameters": {
                        "gaussian": {"center": "${init_center:0.2}", "width": 0.05},
                        "uniform": {},
                        "bimodal": {"centers": [0.3, 0.7], "widths": [0.03, 0.03]},
                    },
                },
            }
        }

        # Experiment configuration
        experiment_config = {
            "experiment": {
                "name": "parameter_sweep",
                "description": "Parameter sweep experiment",
                "output_dir": "results/${experiment.name}",
                "logging": {
                    "level": "INFO",
                    "file": "${experiment.output_dir}/experiment.log",
                },
                "visualization": {
                    "enabled": True,
                    "save_plots": True,
                    "plot_dir": "${experiment.output_dir}/plots",
                    "formats": ["png", "html"],
                    "dpi": 300,
                },
                "sweeps": {
                    "lambda": [0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
                    "init_type": ["gaussian", "uniform", "bimodal"],
                },
            }
        }

        # Write default configs
        configs = {
            "base_mfg.yaml": base_mfg_config,
            "solver.yaml": solver_config,
            "beach_problem.yaml": beach_config,
            "experiment.yaml": experiment_config,
        }

        for filename, config in configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                self._OmegaConf.save(self._OmegaConf.create(config), config_path)
                logger.info(f"Created default config: {config_path}")

    def load_config(self, config_path: str | Path, **overrides: Any) -> OmegaConfig:
        """
        Load configuration from YAML file with optional overrides.

        Args:
            config_path: Path to configuration file
            **overrides: Configuration overrides

        Returns:
            OmegaConf DictConfig object
        """
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load base configuration
        config = self._OmegaConf.load(config_path)

        # Apply overrides
        if overrides:
            override_config = self._OmegaConf.create(overrides)
            config = self._OmegaConf.merge(config, override_config)

        # Resolve interpolations (modifies config in place)
        try:
            self._OmegaConf.resolve(config)
        except (UnsupportedInterpolationType, Exception) as e:
            logger.warning(f"Could not resolve all interpolations: {e}")

        # Cast to DictConfig for type consistency (our configs should be dictionaries)
        from typing import cast

        return cast("OmegaConfig", config)

    def compose_config(self, *config_paths: str | Path, **overrides: Any) -> OmegaConfig:
        """
        Compose configuration from multiple YAML files.

        Args:
            *config_paths: Paths to configuration files (in merge order)
            **overrides: Final configuration overrides

        Returns:
            Composed OmegaConf DictConfig object
        """
        configs = []
        for path in config_paths:
            config = self.load_config(path)
            configs.append(config)

        # Merge configurations
        composed_config = self._OmegaConf.merge(*configs)

        # Apply final overrides
        if overrides:
            override_config = self._OmegaConf.create(overrides)
            composed_config = self._OmegaConf.merge(composed_config, override_config)

        # Cast to DictConfig for type consistency
        from typing import cast

        return cast("OmegaConfig", composed_config)

    def create_pydantic_config(self, omega_config: OmegaConfig) -> MFGSolverConfig:
        """
        Convert OmegaConf configuration to Pydantic MFGSolverConfig.

        Args:
            omega_config: OmegaConf configuration

        Returns:
            Pydantic MFGSolverConfig object
        """
        # Extract solver configuration
        solver_config = omega_config.get("solver", {})
        # Convert OmegaConf to Python dict (use to_container for newer OmegaConf versions)
        if hasattr(self._OmegaConf, "to_container"):
            raw_dict = self._OmegaConf.to_container(solver_config, resolve=True)
        else:
            # Fallback for older OmegaConf versions
            raw_dict = dict(solver_config) if hasattr(solver_config, "items") else {}

        # Ensure we have a proper dictionary with string keys
        from typing import cast

        solver_dict: dict[str, Any] = cast("dict[str, Any]", raw_dict if isinstance(raw_dict, dict) else {})

        # Map OmegaConf structure to Pydantic structure
        pydantic_dict = self._map_omega_to_pydantic(solver_dict)

        try:
            return MFGSolverConfig(**pydantic_dict)
        except Exception as e:
            logger.warning(f"Could not create Pydantic config directly: {e}")
            # Fall back to default configuration
            return create_fast_config()

    def _map_omega_to_pydantic(self, omega_dict: dict[str, Any]) -> dict[str, Any]:
        """Map OmegaConf structure to Pydantic structure."""
        # This is a simplified mapping - extend as needed
        return {
            "max_iterations": omega_dict.get("max_iterations", 100),
            "tolerance": omega_dict.get("tolerance", 1e-6),
            "damping": omega_dict.get("damping", 0.5),
            # Add more mappings as needed
        }

    def save_config(self, config: OmegaConfig, output_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: OmegaConf configuration
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._OmegaConf.save(config, output_path)
        logger.info(f"Configuration saved to: {output_path}")

    def create_parameter_sweep(self, base_config: OmegaConfig, sweep_params: dict[str, list[Any]]) -> list[OmegaConfig]:
        """
        Create parameter sweep configurations.

        Args:
            base_config: Base configuration
            sweep_params: Dictionary of parameter names to value lists

        Returns:
            List of configurations for parameter sweep
        """
        import itertools

        # Get all combinations
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        combinations = list(itertools.product(*param_values))

        sweep_configs = []
        for combo in combinations:
            # Create configuration for this combination
            config = self._OmegaConf.create(base_config)

            # Apply parameter values using OmegaConf dot notation
            for param_name, param_value in zip(param_names, combo, strict=False):
                # Use OmegaConf.select for navigation and direct assignment
                # First ensure the nested structure exists
                keys = param_name.split(".")
                for i in range(len(keys) - 1):
                    partial_key = ".".join(keys[: i + 1])
                    if self._OmegaConf.select(config, partial_key) is None:
                        # Create the nested key by setting an empty dict
                        parent_key = ".".join(keys[:i]) if i > 0 else None
                        if parent_key:
                            parent = self._OmegaConf.select(config, parent_key)
                            setattr(parent, keys[i], {})
                        else:
                            setattr(config, keys[i], {})

                # Now set the final value
                if len(keys) == 1:
                    setattr(config, keys[0], param_value)
                else:
                    parent_key = ".".join(keys[:-1])
                    parent = self._OmegaConf.select(config, parent_key)
                    setattr(parent, keys[-1], param_value)

            # Add metadata
            if "experiment" not in config:
                config.experiment = {}
            config.experiment["current_params"] = dict(zip(param_names, combo, strict=False))

            sweep_configs.append(config)

        return sweep_configs

    def validate_config(self, config: OmegaConfig, schema_name: str | None = None) -> bool:
        """
        Validate configuration against schema.

        Args:
            config: Configuration to validate
            schema_name: Optional schema name for validation

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - check required fields
            required_fields = ["problem", "solver"]
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Type validation
            if not isinstance(config.problem.T, int | float):
                logger.error("problem.T must be numeric")
                return False

            if not isinstance(config.problem.Nx, int):
                logger.error("problem.Nx must be integer")
                return False

            # Add more validation as needed
            return True

        except (AttributeError, ConfigAttributeError) as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def get_config_template(self, template_name: str) -> OmegaConfig:
        """
        Get configuration template by name.

        Args:
            template_name: Name of configuration template

        Returns:
            Configuration template
        """
        templates = {
            "beach_problem": "beach_problem.yaml",
            "base_mfg": "base_mfg.yaml",
            "solver": "solver.yaml",
            "experiment": "experiment.yaml",
        }

        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")

        return self.load_config(templates[template_name])

    # === STRUCTURED CONFIG METHODS (Type-Safe) ===

    def load_structured_config(
        self, config_path: str | Path, schema_cls: type = MFGConfig, **overrides: Any
    ) -> TypedMFGConfig:
        """
        Load configuration using structured schema for full type safety.

        This method solves the common OmegaConf type checking problem by using
        structured configs that provide static type information to mypy.

        Args:
            config_path: Path to configuration file
            schema_cls: Structured schema class (dataclass)
            **overrides: Configuration overrides

        Returns:
            Fully typed configuration object with autocompletion support

        Example:
            >>> manager = OmegaConfManager()
            >>> config = manager.load_structured_config("config.yaml")
            >>> print(config.problem.T)  # ✅ Type safe, autocompletes
        """
        # Create structured schema with defaults
        schema = self._OmegaConf.structured(schema_cls)

        # Load file configuration
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if config_path.exists():
            file_config = self._OmegaConf.load(config_path)
        else:
            logger.warning(f"Config file not found: {config_path}, using schema defaults")
            file_config = self._OmegaConf.create({})

        # Apply overrides
        if overrides:
            override_config = self._OmegaConf.create(overrides)
            file_config = self._OmegaConf.merge(file_config, override_config)

        # Merge schema with file config (file config takes precedence)
        config = self._OmegaConf.merge(schema, file_config)

        # Resolve interpolations
        try:
            self._OmegaConf.resolve(config)
        except (UnsupportedInterpolationType, Exception) as e:
            logger.warning(f"Could not resolve all interpolations: {e}")

        return config

    def load_mfg_config(self, config_path: str | Path = "config.yaml", **overrides: Any) -> TypedMFGConfig:
        """
        Load complete MFG configuration with full type safety.

        Args:
            config_path: Path to configuration file
            **overrides: Configuration overrides

        Returns:
            Fully typed MFG configuration object
        """
        return self.load_structured_config(config_path, MFGConfig, **overrides)

    def load_beach_config_structured(
        self, config_path: str | Path = "beach_problem.yaml", **overrides: Any
    ) -> TypedBeachConfig:
        """
        Load Beach problem configuration with full type safety.

        Args:
            config_path: Path to configuration file
            **overrides: Configuration overrides

        Returns:
            Fully typed Beach problem configuration object
        """
        return self.load_structured_config(config_path, BeachProblemConfig, **overrides)  # type: ignore[return-value]

    def create_default_mfg_config(self) -> TypedMFGConfig:
        """
        Create default MFG configuration using structured schema.

        Returns:
            Default MFG configuration with full type safety
        """
        return self._OmegaConf.structured(MFGConfig)

    def validate_structured_config(self, config: TypedMFGConfig) -> bool:
        """
        Validate structured configuration against schema.

        Args:
            config: Structured configuration object

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - structured configs provide automatic validation
            if not hasattr(config, "problem") or not hasattr(config, "solver"):
                logger.error("Missing required configuration sections")
                return False

            # Type validation is automatic with structured configs
            if not isinstance(config.problem.T, int | float):
                logger.error("problem.T must be numeric")
                return False

            if not isinstance(config.problem.Nx, int):
                logger.error("problem.Nx must be integer")
                return False

            return True

        except (AttributeError, Exception) as e:
            logger.error(f"Configuration validation error: {e}")
            return False


def create_omega_manager(config_dir: str | Path | None = None) -> OmegaConfManager:
    """
    Create OmegaConf configuration manager.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        OmegaConfManager instance
    """
    return OmegaConfManager(config_dir)


# Convenience functions
def load_beach_config(**overrides: Any) -> OmegaConfig:
    """Load Towel on Beach configuration with overrides."""
    manager = create_omega_manager()
    return manager.load_config("beach_problem.yaml", **overrides)


def load_experiment_config(**overrides: Any) -> OmegaConfig:
    """Load experiment configuration with overrides."""
    manager = create_omega_manager()
    return manager.load_config("experiment.yaml", **overrides)


def create_parameter_sweep_configs(lambda_values: list[float], init_types: list[str]) -> list[OmegaConfig]:
    """Create parameter sweep configurations for beach problem."""
    manager = create_omega_manager()
    base_config = manager.load_config("beach_problem.yaml")

    sweep_params = {
        "problem.parameters.crowd_aversion": lambda_values,
        "problem.initial_condition.type": init_types,
    }

    return manager.create_parameter_sweep(base_config, sweep_params)  # type: ignore


# === NEW STRUCTURED CONFIG CONVENIENCE FUNCTIONS (Type-Safe) ===


def load_structured_mfg_config(config_path: str | Path = "config.yaml", **overrides: Any) -> TypedMFGConfig:
    """
    Load MFG configuration with full type safety.

    This function provides the new type-safe way to load configurations,
    solving the common OmegaConf type checking issues.

    Args:
        config_path: Path to configuration file
        **overrides: Configuration overrides

    Returns:
        Fully typed MFG configuration with autocompletion support

    Example:
        >>> config = load_structured_mfg_config("my_config.yaml")
        >>> print(config.problem.T)  # ✅ Type safe, autocompletes
        >>> print(config.solver.max_iterations)  # ✅ Full IDE support
    """
    manager = create_omega_manager()
    return manager.load_mfg_config(config_path, **overrides)


def load_structured_beach_config(config_path: str | Path = "beach_problem.yaml", **overrides: Any) -> TypedBeachConfig:
    """
    Load Beach problem configuration with full type safety.

    Args:
        config_path: Path to configuration file
        **overrides: Configuration overrides

    Returns:
        Fully typed Beach problem configuration
    """
    manager = create_omega_manager()
    return manager.load_beach_config_structured(config_path, **overrides)


def create_default_structured_config() -> TypedMFGConfig:
    """
    Create default MFG configuration using structured schema.

    Returns:
        Default configuration with full type safety
    """
    manager = create_omega_manager()
    return manager.create_default_mfg_config()
