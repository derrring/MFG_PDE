"""
Legacy solver compatibility wrappers

Provides compatibility for old solver APIs with deprecation warnings.
"""

from mfg_pde.factory import create_fast_solver

from . import DeprecatedAPI, deprecated


@deprecated("Use create_fast_solver() or FixedPointSolver() instead")
class LegacyMFGSolver(DeprecatedAPI):
    """
    DEPRECATED: Legacy MFG solver wrapper.

    Use the new factory API instead:

    Old:
        solver = LegacyMFGSolver(config)
        result = solver.solve(problem)

    New (Factory API):
        solver = create_fast_solver(problem, solver_type="fixed_point")
        result = solver.solve()

    New (Direct):
        solver = FixedPointSolver()
        result = solver.solve(problem)
    """

    def __init__(self, config=None):
        super().__init__("create_fast_solver() or FixedPointSolver()")
        self.config = config or {}

    def solve(self, problem):
        """Solve MFG problem using legacy interface."""
        # Use factory API as default
        solver = create_fast_solver(problem, solver_type="fixed_point")
        return solver.solve()

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
