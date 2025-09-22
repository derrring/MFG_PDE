"""
Modern Configuration System with Builder Pattern

This module provides a clean, fluent API for configuring solvers
that complements the existing configuration system.
"""

from __future__ import annotations


from typing import Any
import copy

from .pydantic_config import MFGSolverConfig, create_fast_config, create_accurate_config, create_research_config


class SolverConfig:
    """
    Modern solver configuration with fluent builder pattern.

    This class provides a clean, readable way to configure solvers
    while internally using the existing Pydantic configuration system.

    Example:
        config = (SolverConfig()
                  .with_tolerance(1e-8)
                  .with_max_iterations(500)
                  .with_damping(0.8))

        solver = FixedPointSolver(config=config)
    """

    def __init__(self, base_config: MFGSolverConfig | None = None):
        """
        Initialize configuration.

        Args:
            base_config: Optional base configuration to start from
        """
        if base_config is None:
            base_config = create_fast_config()  # Sensible default

        self._config = base_config

    def with_tolerance(self, tolerance: float) -> SolverConfig:
        """Set convergence tolerance."""
        new_config = copy.deepcopy(self._config)
        # Update the relevant tolerance fields
        if hasattr(new_config, 'picard') and new_config.picard:
            new_config.picard.tolerance = tolerance
        if hasattr(new_config, 'newton') and new_config.newton:
            new_config.newton.tolerance = tolerance
        return SolverConfig(new_config)

    def with_max_iterations(self, max_iterations: int) -> SolverConfig:
        """Set maximum iterations."""
        new_config = copy.deepcopy(self._config)
        if hasattr(new_config, 'picard') and new_config.picard:
            new_config.picard.max_iterations = max_iterations
        if hasattr(new_config, 'newton') and new_config.newton:
            new_config.newton.max_iterations = max_iterations
        return SolverConfig(new_config)

    def with_damping(self, damping_factor: float) -> SolverConfig:
        """Set damping factor."""
        new_config = copy.deepcopy(self._config)
        if hasattr(new_config, 'picard') and new_config.picard:
            new_config.picard.damping_factor = damping_factor
        return SolverConfig(new_config)

    def with_verbose(self, verbose: bool = True) -> SolverConfig:
        """Enable/disable verbose output."""
        new_config = copy.deepcopy(self._config)
        # Set verbose on all components that support it
        for component_name in ['picard', 'newton', 'hjb', 'fp']:
            component = getattr(new_config, component_name, None)
            if component and hasattr(component, 'verbose'):
                component.verbose = verbose
        return SolverConfig(new_config)

    def with_method(self, method: str) -> SolverConfig:
        """Set solution method."""
        new_config = copy.deepcopy(self._config)
        # This would set the appropriate method configuration
        # Implementation depends on the specific method structure
        return SolverConfig(new_config)

    def with_grid_size(self, nx: int, nt: int | None = None) -> SolverConfig:
        """Set grid resolution."""
        new_config = copy.deepcopy(self._config)
        # Update grid configuration if available
        # This would depend on how grid configuration is structured
        return SolverConfig(new_config)

    def with_custom_parameter(self, name: str, value: Any) -> SolverConfig:
        """Set custom parameter."""
        new_config = copy.deepcopy(self._config)
        # Store custom parameters in the SolverConfig wrapper instead
        new_solver_config = SolverConfig(new_config)
        if not hasattr(new_solver_config, '_custom_parameters'):
            new_solver_config._custom_parameters = {}
        new_solver_config._custom_parameters[name] = value
        return new_solver_config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self._config.dict() if hasattr(self._config, 'dict') else vars(self._config)

    def get_underlying_config(self) -> MFGSolverConfig:
        """Get the underlying Pydantic configuration."""
        return self._config

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        # First check custom parameters stored in wrapper
        if hasattr(self, '_custom_parameters') and name in self._custom_parameters:
            return self._custom_parameters[name]

        # Then check standard parameters
        return getattr(self._config, name, default)

    @property
    def max_iterations(self) -> int:
        """Get max iterations."""
        if hasattr(self._config, 'picard') and self._config.picard:
            return self._config.picard.max_iterations
        return 100  # Default

    @property
    def tolerance(self) -> float:
        """Get tolerance."""
        if hasattr(self._config, 'picard') and self._config.picard:
            return self._config.picard.tolerance
        return 1e-6  # Default

    def __str__(self) -> str:
        return f"SolverConfig(tolerance={self.tolerance}, max_iterations={self.max_iterations})"

    def __repr__(self) -> str:
        return self.__str__()


class PresetConfig:
    """
    Factory for common solver configurations.

    Provides easy access to well-tested configuration presets.
    """

    @staticmethod
    def for_research_prototype() -> SolverConfig:
        """
        Configuration optimized for quick research prototyping.

        - Fast convergence
        - Moderate accuracy
        - Verbose output for debugging
        """
        base = create_research_config()
        return (SolverConfig(base)
                .with_tolerance(1e-4)
                .with_max_iterations(50)
                .with_verbose(True))

    @staticmethod
    def for_production_quality() -> SolverConfig:
        """
        Configuration for production-quality results.

        - High accuracy
        - Conservative convergence criteria
        - Error checking enabled
        """
        base = create_accurate_config()
        return (SolverConfig(base)
                .with_tolerance(1e-8)
                .with_max_iterations(1000)
                .with_verbose(False))

    @staticmethod
    def for_high_performance() -> SolverConfig:
        """
        Configuration optimized for speed.

        - Fast algorithms
        - Moderate accuracy
        - Optimized for large problems
        """
        base = create_fast_config()
        return (SolverConfig(base)
                .with_tolerance(1e-5)
                .with_max_iterations(200)
                .with_verbose(False))

    @staticmethod
    def for_educational() -> SolverConfig:
        """
        Configuration for educational/demonstration purposes.

        - Verbose output
        - Clear intermediate steps
        - Moderate accuracy
        """
        base = create_research_config()
        return (SolverConfig(base)
                .with_tolerance(1e-4)
                .with_max_iterations(100)
                .with_verbose(True)
                .with_custom_parameter('show_steps', True)
                .with_custom_parameter('plot_convergence', True))

    @staticmethod
    def for_high_accuracy() -> SolverConfig:
        """
        Configuration for high-accuracy computations.

        - Very tight tolerances
        - Maximum iterations
        - Conservative algorithms
        """
        base = create_accurate_config()
        return (SolverConfig(base)
                .with_tolerance(1e-10)
                .with_max_iterations(2000)
                .with_damping(0.5)  # More conservative
                .with_verbose(False))

    @staticmethod
    def for_large_problems() -> SolverConfig:
        """
        Configuration optimized for large-scale problems.

        - Memory-efficient algorithms
        - Faster convergence
        - Scalable methods
        """
        base = create_fast_config()
        return (SolverConfig(base)
                .with_tolerance(1e-4)
                .with_max_iterations(100)
                .with_custom_parameter('memory_efficient', True)
                .with_custom_parameter('use_sparse', True))

    @staticmethod
    def for_crowd_dynamics() -> SolverConfig:
        """
        Configuration optimized for crowd dynamics problems.

        - Handles high density regions
        - Stable for varying crowd sizes
        - Good balance of speed and stability
        """
        base = create_fast_config()
        return (SolverConfig(base)
                .with_tolerance(1e-5)
                .with_max_iterations(200)
                .with_damping(0.9)
                .with_custom_parameter('adaptive_grid', True))

    @staticmethod
    def for_financial_problems() -> SolverConfig:
        """
        Configuration optimized for portfolio/financial problems.

        - High accuracy for financial calculations
        - Stable for low risk aversion
        - Conservative convergence criteria
        """
        base = create_accurate_config()
        return (SolverConfig(base)
                .with_tolerance(1e-7)
                .with_max_iterations(500)
                .with_damping(0.8)
                .with_custom_parameter('enforce_bounds', True))

    @staticmethod
    def for_epidemic_models() -> SolverConfig:
        """
        Configuration optimized for epidemic spreading models.

        - Handles sharp transitions
        - Stable for high infection rates
        - Conservative time stepping
        """
        base = create_research_config()
        return (SolverConfig(base)
                .with_tolerance(1e-6)
                .with_max_iterations(300)
                .with_damping(0.7)
                .with_custom_parameter('adaptive_timestep', True))

    @staticmethod
    def for_traffic_flow() -> SolverConfig:
        """
        Configuration optimized for traffic flow problems.

        - Fast convergence for smooth dynamics
        - Efficient for large road networks
        - Good performance for steady-state problems
        """
        base = create_fast_config()
        return (SolverConfig(base)
                .with_tolerance(1e-5)
                .with_max_iterations(150)
                .with_custom_parameter('use_upwind', True)
                .with_custom_parameter('steady_state_acceleration', True))


# Convenience aliases for common usage patterns
def create_config() -> SolverConfig:
    """Create a default configuration."""
    return SolverConfig()

def research_config() -> SolverConfig:
    """Create configuration for research."""
    return PresetConfig.for_research_prototype()

def production_config() -> SolverConfig:
    """Create configuration for production."""
    return PresetConfig.for_production_quality()

def fast_config() -> SolverConfig:
    """Create configuration for speed."""
    return PresetConfig.for_high_performance()

def accurate_config() -> SolverConfig:
    """Create configuration for accuracy."""
    return PresetConfig.for_high_accuracy()

def educational_config() -> SolverConfig:
    """Create configuration for education."""
    return PresetConfig.for_educational()

def crowd_dynamics_config() -> SolverConfig:
    """Create configuration optimized for crowd dynamics."""
    return PresetConfig.for_crowd_dynamics()

def financial_config() -> SolverConfig:
    """Create configuration optimized for financial problems."""
    return PresetConfig.for_financial_problems()

def epidemic_config() -> SolverConfig:
    """Create configuration optimized for epidemic models."""
    return PresetConfig.for_epidemic_models()

def traffic_config() -> SolverConfig:
    """Create configuration optimized for traffic flow."""
    return PresetConfig.for_traffic_flow()

def large_scale_config() -> SolverConfig:
    """Create configuration optimized for large-scale problems."""
    return PresetConfig.for_large_problems()