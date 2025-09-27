"""
Legacy solver compatibility wrappers

Provides compatibility for old solver APIs with deprecation warnings.
"""

from mfg_pde.simple import solve_mfg
from mfg_pde.solvers import FixedPointSolver

from . import DeprecatedAPI, deprecated


@deprecated("Use solve_mfg('crowd_dynamics') or FixedPointSolver() instead")
class LegacyMFGSolver(DeprecatedAPI):
    """
    DEPRECATED: Legacy MFG solver wrapper.

    Use the new API instead:

    Old:
        solver = LegacyMFGSolver(config)
        result = solver.solve(problem)

    New (Simple):
        result = solve_mfg("crowd_dynamics")

    New (Advanced):
        solver = FixedPointSolver()
        result = solver.solve(problem)
    """

    def __init__(self, config=None):
        super().__init__("solve_mfg() or FixedPointSolver()")
        self.config = config or {}

    def solve(self, problem):
        """Solve MFG problem using legacy interface."""
        # Convert legacy config to new API
        kwargs = self._convert_config(self.config)

        # Try to determine problem type
        problem_type = self._infer_problem_type(problem)

        if problem_type:
            # Use simple API
            return solve_mfg(problem_type, **kwargs)
        else:
            # Use core objects API
            solver = FixedPointSolver(**kwargs)
            return solver.solve(problem)

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


# Legacy solver class aliases with deprecation warnings
@deprecated("Use solve_mfg() instead")
class EnhancedParticleCollocationSolver(LegacyMFGSolver):
    pass


@deprecated("Use FixedPointSolver() instead")
class FixedPointIterator(LegacyMFGSolver):
    pass


@deprecated("Use FixedPointSolver() with hooks instead")
class AdaptiveMFGSolver(LegacyMFGSolver):
    pass


@deprecated("Use solve_mfg() with accuracy='research' instead")
class DebugMFGSolver(LegacyMFGSolver):
    pass
