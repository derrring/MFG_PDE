"""
Enhanced MFG Solver Factory with Pydantic Configuration Support

Provides factory patterns for creating optimized solver configurations with
Pydantic validation, automatic serialization, and enhanced error checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

try:
    from pydantic import ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from ..alg.mfg_solvers.adaptive_particle_collocation_solver import AdaptiveParticleCollocationSolver
from ..alg.mfg_solvers.config_aware_fixed_point_iterator import ConfigAwareFixedPointIterator
from ..alg.mfg_solvers.enhanced_particle_collocation_solver import MonitoredParticleCollocationSolver
from ..config.pydantic_config import (
    MFGSolverConfig,
    create_accurate_config,
    create_fast_config,
    create_research_config,
)

if TYPE_CHECKING:
    from ..core.mfg_problem import MFGProblem

from ..utils.logging import get_logger

SolverType = Literal["fixed_point", "particle_collocation", "adaptive_particle", "monitored_particle"]


class PydanticSolverFactory:
    """
    Enhanced factory for creating MFG solvers with Pydantic validation.

    Provides easy creation patterns with automatic configuration validation,
    serialization support, and enhanced error checking for research workflows.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

        if not PYDANTIC_AVAILABLE:
            self.logger.warning("Pydantic not available - falling back to basic factory")

    def create_validated_solver(
        self,
        problem: MFGProblem,
        solver_type: SolverType = "fixed_point",
        config: MFGSolverConfig | None = None,
        config_preset: str = "balanced",
        **kwargs,
    ) -> ConfigAwareFixedPointIterator | MonitoredParticleCollocationSolver | AdaptiveParticleCollocationSolver:
        """
        Create MFG solver with comprehensive Pydantic validation.

        Args:
            problem: MFG problem instance
            solver_type: Type of solver to create
            config: Pydantic MFGSolverConfig (optional)
            config_preset: Configuration preset if config not provided
            **kwargs: Additional parameters (validated automatically)

        Returns:
            Configured and validated MFG solver

        Raises:
            ValidationError: If configuration validation fails
            ValueError: If solver type is invalid
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic required for validated solver creation")

        try:
            # Create or validate configuration
            if config is None:
                config = self._create_preset_config(config_preset, **kwargs)
            else:
                # Validate existing configuration
                if not isinstance(config, MFGSolverConfig):
                    raise ValueError("config must be MFGSolverConfig instance")

                # Update with any additional kwargs
                if kwargs:
                    config = self._update_config_with_kwargs(config, **kwargs)

            # Log configuration details
            self.logger.info(f"Creating {solver_type} solver with validated configuration")
            self.logger.debug(f"Configuration: {config.dict()}")

            # Create solver based on type
            if solver_type == "fixed_point":
                return self._create_validated_fixed_point_solver(problem, config)
            elif solver_type == "particle_collocation":
                return self._create_validated_particle_collocation_solver(problem, config)
            elif solver_type == "adaptive_particle":
                return self._create_validated_adaptive_particle_solver(problem, config)
            elif solver_type == "monitored_particle":
                return self._create_validated_monitored_particle_solver(problem, config)
            else:
                raise ValueError(f"Unknown solver type: {solver_type}")

        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ValidationError(f"Solver configuration validation failed: {e}")
        except Exception as e:
            self.logger.error(f"Solver creation failed: {e}")
            raise RuntimeError(f"Failed to create solver: {e}")

    def _create_preset_config(self, preset: str, **kwargs) -> MFGSolverConfig:
        """Create configuration from preset with validation."""
        preset_configs = {
            "fast": create_fast_config,
            "accurate": create_accurate_config,
            "research": create_research_config,
            "balanced": create_accurate_config,  # Default to accurate
        }

        if preset not in preset_configs:
            available = list(preset_configs.keys())
            raise ValueError(f"Unknown config preset '{preset}'. Available: {available}")

        # Create base configuration
        config = preset_configs[preset]()

        # Update with kwargs if provided
        if kwargs:
            config = self._update_config_with_kwargs(config, **kwargs)

        return config

    def _update_config_with_kwargs(self, config: MFGSolverConfig, **kwargs) -> MFGSolverConfig:
        """Update Pydantic configuration with additional parameters."""
        try:
            # Convert to dict and update
            config_dict = config.dict()

            # Handle nested updates
            for key, value in kwargs.items():
                if key in ["newton_tolerance", "newton_max_iterations"]:
                    # Newton parameters
                    newton_key = key.replace("newton_", "")
                    if newton_key == "max_iterations":
                        newton_key = "max_iterations"
                    elif newton_key == "tolerance":
                        newton_key = "tolerance"
                    config_dict["newton"][newton_key] = value

                elif key in [
                    "picard_tolerance",
                    "picard_max_iterations",
                    "picard_damping_factor",
                ]:
                    # Picard parameters
                    picard_key = key.replace("picard_", "").replace("damping_factor", "damping_factor")
                    if picard_key == "max_iterations":
                        picard_key = "max_iterations"
                    elif picard_key == "tolerance":
                        picard_key = "tolerance"
                    config_dict["picard"][picard_key] = value

                elif key in ["num_particles", "kde_bandwidth"]:
                    # Particle parameters
                    config_dict["fp"]["particle"][key] = value

                elif key == "convergence_tolerance":
                    config_dict["convergence_tolerance"] = value

                elif key in [
                    "return_structured",
                    "enable_warm_start",
                    "experiment_name",
                ]:
                    config_dict[key] = value

                else:
                    # Try to set directly
                    config_dict[key] = value

            # Recreate configuration with validation
            return MFGSolverConfig(**config_dict)

        except ValidationError as e:
            self.logger.warning(f"Parameter update validation failed: {e}")
            # Try to update individual fields
            updated_config = config.copy(deep=True)

            for key, value in kwargs.items():
                try:
                    if hasattr(updated_config, key):
                        setattr(updated_config, key, value)
                    else:
                        self.logger.warning(f"Unknown parameter: {key}")
                except Exception as field_error:
                    self.logger.warning(f"Failed to set {key}={value}: {field_error}")

            return updated_config

    def _create_validated_fixed_point_solver(
        self, problem: MFGProblem, config: MFGSolverConfig
    ) -> ConfigAwareFixedPointIterator:
        """Create validated fixed point iterator solver."""
        try:
            # Create HJB and FP solvers (simplified for now)
            from ..alg.fp_solvers.fp_particle import FPParticleSolver
            from ..alg.hjb_solvers.hjb_gfdm import HJBGFDMSolver

            # Create collocation points
            collocation_points = np.linspace(0, 1, 10).reshape(-1, 1)

            # Create HJB solver with config (using tuned QP optimization level)
            hjb_solver = HJBGFDMSolver(
                problem=problem,
                collocation_points=collocation_points,
                qp_optimization_level="tuned",
                qp_usage_target=0.1,
            )

            # Create FP solver with config
            fp_solver = FPParticleSolver(problem=problem, num_particles=config.fp.particle.num_particles)

            # Create fixed point iterator with validated config
            solver = ConfigAwareFixedPointIterator(
                problem=problem,
                hjb_solver=hjb_solver,
                fp_solver=fp_solver,
                config=config,
            )

            self.logger.info("Successfully created validated fixed point solver")
            return solver

        except Exception as e:
            self.logger.error(f"Failed to create fixed point solver: {e}")
            raise RuntimeError(f"Fixed point solver creation failed: {e}")

    def _create_validated_particle_collocation_solver(
        self, problem: MFGProblem, config: MFGSolverConfig
    ) -> ParticleCollocationSolver:
        """Create validated particle collocation solver."""
        try:
            from ..alg.mfg_solvers.particle_collocation_solver import ParticleCollocationSolver

            # Create collocation points
            collocation_points = np.linspace(0, 1, 10).reshape(-1, 1)

            # Extract legacy parameters for compatibility
            legacy_params = config.to_legacy_dict()

            solver = ParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=config.fp.particle.num_particles,
                **legacy_params,
            )

            self.logger.info("Successfully created validated particle collocation solver")
            return solver

        except Exception as e:
            self.logger.error(f"Failed to create particle collocation solver: {e}")
            raise RuntimeError(f"Particle collocation solver creation failed: {e}")

    def _create_validated_adaptive_particle_solver(
        self, problem: MFGProblem, config: MFGSolverConfig
    ) -> AdaptiveParticleCollocationSolver:
        """Create validated adaptive particle solver."""
        try:
            # Create collocation points
            collocation_points = np.linspace(0, 1, 10).reshape(-1, 1)

            # Extract legacy parameters
            legacy_params = config.to_legacy_dict()

            solver = AdaptiveParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=config.fp.particle.num_particles,
                verbose=False,
                **legacy_params,
            )

            self.logger.info("Successfully created validated adaptive particle solver")
            return solver

        except Exception as e:
            self.logger.error(f"Failed to create adaptive particle solver: {e}")
            raise RuntimeError(f"Adaptive particle solver creation failed: {e}")

    def _create_validated_monitored_particle_solver(
        self, problem: MFGProblem, config: MFGSolverConfig
    ) -> MonitoredParticleCollocationSolver:
        """Create validated monitored particle solver."""
        try:
            # Create collocation points
            collocation_points = np.linspace(0, 1, 10).reshape(-1, 1)

            # Extract legacy parameters
            legacy_params = config.to_legacy_dict()

            solver = MonitoredParticleCollocationSolver(
                problem=problem,
                collocation_points=collocation_points,
                num_particles=config.fp.particle.num_particles,
                **legacy_params,
            )

            self.logger.info("Successfully created validated monitored particle solver")
            return solver

        except Exception as e:
            self.logger.error(f"Failed to create monitored particle solver: {e}")
            raise RuntimeError(f"Monitored particle solver creation failed: {e}")


# Global factory instance
_pydantic_factory = PydanticSolverFactory()


def create_validated_solver(
    problem: MFGProblem,
    solver_type: SolverType = "fixed_point",
    config: MFGSolverConfig | None = None,
    config_preset: str = "balanced",
    **kwargs,
) -> ConfigAwareFixedPointIterator | MonitoredParticleCollocationSolver | AdaptiveParticleCollocationSolver:
    """
    Convenience function for creating validated MFG solvers.

    Args:
        problem: MFG problem instance
        solver_type: Type of solver to create
        config: Optional Pydantic configuration
        config_preset: Configuration preset name
        **kwargs: Additional configuration parameters

    Returns:
        Validated MFG solver instance
    """
    return _pydantic_factory.create_validated_solver(
        problem=problem,
        solver_type=solver_type,
        config=config,
        config_preset=config_preset,
        **kwargs,
    )


def create_fast_validated_solver(
    problem: MFGProblem, solver_type: SolverType = "fixed_point", **kwargs
) -> ConfigAwareFixedPointIterator | MonitoredParticleCollocationSolver | AdaptiveParticleCollocationSolver:
    """Create fast solver with Pydantic validation."""
    return create_validated_solver(problem=problem, solver_type=solver_type, config_preset="fast", **kwargs)


def create_accurate_validated_solver(
    problem: MFGProblem, solver_type: SolverType = "fixed_point", **kwargs
) -> ConfigAwareFixedPointIterator | MonitoredParticleCollocationSolver | AdaptiveParticleCollocationSolver:
    """Create accurate solver with Pydantic validation."""
    return create_validated_solver(problem=problem, solver_type=solver_type, config_preset="accurate", **kwargs)


def create_research_validated_solver(
    problem: MFGProblem,
    solver_type: SolverType = "fixed_point",
    experiment_name: str | None = None,
    **kwargs,
) -> ConfigAwareFixedPointIterator | MonitoredParticleCollocationSolver | AdaptiveParticleCollocationSolver:
    """Create research solver with Pydantic validation and experiment tracking."""
    return create_validated_solver(
        problem=problem,
        solver_type=solver_type,
        config_preset="research",
        experiment_name=experiment_name,
        **kwargs,
    )


def validate_solver_config(config_dict: dict[str, Any]) -> MFGSolverConfig:
    """
    Validate a configuration dictionary and return Pydantic model.

    Args:
        config_dict: Configuration parameters as dictionary

    Returns:
        Validated MFGSolverConfig instance

    Raises:
        ValidationError: If validation fails
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic required for configuration validation")

    return MFGSolverConfig(**config_dict)


def load_solver_config(config_path: str) -> MFGSolverConfig:
    """
    Load and validate solver configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Validated MFGSolverConfig instance
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic required for configuration loading")

    return MFGSolverConfig.parse_file(config_path)


def save_solver_config(config: MFGSolverConfig, config_path: str) -> None:
    """
    Save validated solver configuration to JSON file.

    Args:
        config: Pydantic MFGSolverConfig instance
        config_path: Output path for JSON file
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic required for configuration saving")

    with open(config_path, "w") as f:
        f.write(config.json(indent=2))
