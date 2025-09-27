"""
Legacy configuration compatibility wrappers

Provides compatibility for old config APIs with deprecation warnings.
"""

from mfg_pde.config import accurate_config, fast_config, research_config

from . import DeprecatedAPI, deprecated


@deprecated("Use fast_config(), accurate_config(), or research_config() instead")
class LegacyConfig(DeprecatedAPI):
    """
    DEPRECATED: Legacy configuration wrapper.

    Use the new config system instead:

    Old:
        config = LegacyConfig(max_iterations=500, tolerance=1e-6)

    New:
        config = accurate_config().with_max_iterations(500).with_tolerance(1e-6)

    Or even simpler:
        result = solve_mfg("crowd_dynamics", accuracy="high")
    """

    def __init__(self, **kwargs):
        super().__init__("New config system or solve_mfg() with accuracy parameter")
        self.params = kwargs

    def to_new_config(self):
        """Convert to new config system."""
        # Determine best base config
        if self.params.get("fast", False):
            base_config = fast_config()
        elif self.params.get("high_accuracy", False):
            base_config = accurate_config()
        elif self.params.get("debug", False):
            base_config = research_config()
        else:
            base_config = accurate_config()

        # Apply custom parameters
        if "max_iterations" in self.params:
            base_config = base_config.with_max_iterations(self.params["max_iterations"])

        if "tolerance" in self.params:
            base_config = base_config.with_tolerance(self.params["tolerance"])

        if "damping_parameter" in self.params:
            base_config = base_config.with_damping(self.params["damping_parameter"])

        return base_config


# Legacy config class aliases
@deprecated("Use SolverConfig from new API instead")
class SolverConfig(LegacyConfig):
    pass


@deprecated("Use solve_mfg() with accuracy='research' instead")
class DebugConfig(LegacyConfig):
    def __init__(self, **kwargs):
        super().__init__(debug=True, **kwargs)


@deprecated("Use solve_mfg() with accuracy='fast' instead")
class FastConfig(LegacyConfig):
    def __init__(self, **kwargs):
        super().__init__(fast=True, **kwargs)


@deprecated("Use solve_mfg() with accuracy='high' instead")
class AccurateConfig(LegacyConfig):
    def __init__(self, **kwargs):
        super().__init__(high_accuracy=True, **kwargs)


# Legacy factory functions
@deprecated("Use solve_mfg() with automatic configuration instead")
def create_enhanced_config(**kwargs):
    """DEPRECATED: Use new automatic configuration system."""
    return LegacyConfig(**kwargs).to_new_config()


@deprecated("Use solve_mfg() with backend parameter instead")
def configure_backend(backend_name, **kwargs):
    """DEPRECATED: Use backend parameter in solve_mfg() or FixedPointSolver."""
    import warnings

    warnings.warn(
        "configure_backend is deprecated. Use backend parameter in solve_mfg() or FixedPointSolver instead.",
        DeprecationWarning,
        stacklevel=2,
    )


@deprecated("Use new logging system with hooks instead")
def configure_research_logging(session_name, **kwargs):
    """DEPRECATED: Use DebugHook or custom hooks for logging."""
    import warnings

    warnings.warn(
        "configure_research_logging is deprecated. Use DebugHook or custom hooks for logging. "
        "Example: solver.solve(problem, hooks=DebugHook(log_level='INFO'))",
        DeprecationWarning,
        stacklevel=2,
    )
