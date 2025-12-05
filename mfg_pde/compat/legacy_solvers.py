"""
Legacy solver compatibility wrappers

Provides compatibility for old solver APIs with deprecation warnings.

DEPRECATED: This entire module is deprecated and will be removed in v1.0.0.
Use problem.solve() API instead.
"""

from . import DeprecatedAPI


class LegacyMFGSolver(DeprecatedAPI):
    """
    DEPRECATED: Legacy MFG solver wrapper.

    Use the new problem.solve() API instead:

    Old:
        solver = LegacyMFGSolver(config)
        result = solver.solve(problem)

    New:
        result = problem.solve()
    """

    def __init__(self, config=None):
        super().__init__("problem.solve()")
        self.config = config or {}

    def solve(self, problem):
        """Solve MFG problem using legacy interface."""
        # Use problem.solve() API
        return problem.solve()

    def _convert_config(self, config):
        """Convert legacy config to new API parameters."""
        mapping = {
            "max_iterations": "max_iterations",
            "tolerance": "tolerance",
            "damping_parameter": "damping",
            "backend": "backend",
        }

        kwargs = {}
        for old_key, new_key in mapping.items():
            if old_key in config:
                kwargs[new_key] = config[old_key]

        return kwargs

    def _infer_problem_type(self, problem):
        """Try to infer problem type from legacy problem object."""
        if hasattr(problem, "problem_type"):
            return problem.problem_type

        # Check for common patterns
        class_name = problem.__class__.__name__.lower()
        if "crowd" in class_name:
            return "crowd_dynamics"
        elif "portfolio" in class_name or "merton" in class_name:
            return "portfolio_optimization"
        elif "traffic" in class_name:
            return "traffic_flow"
        elif "epidemic" in class_name:
            return "epidemic"

        return None


# Legacy solver class aliases (all deprecated - use problem.solve() instead)
class EnhancedParticleCollocationSolver(LegacyMFGSolver):
    """DEPRECATED: Use problem.solve() instead."""


class FixedPointIterator(LegacyMFGSolver):
    """DEPRECATED: Use problem.solve() instead."""


class AdaptiveMFGSolver(LegacyMFGSolver):
    """DEPRECATED: Use problem.solve() instead."""


class DebugMFGSolver(LegacyMFGSolver):
    """DEPRECATED: Use problem.solve() instead."""
