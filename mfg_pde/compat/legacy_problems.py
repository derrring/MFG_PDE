"""
Legacy problem compatibility wrappers

Provides compatibility for old problem APIs with deprecation warnings.
"""

from . import DeprecatedAPI, deprecated


@deprecated("Use create_mfg_problem() or solve_mfg() instead")
class LegacyMFGProblem(DeprecatedAPI):
    """
    DEPRECATED: Legacy MFG problem wrapper.

    Use the new API instead:

    Old:
        problem = LegacyMFGProblem(domain, time_horizon)

    New (Simple):
        result = solve_mfg("crowd_dynamics", domain_size=5.0, time_horizon=2.0)

    New (Advanced):
        problem = create_mfg_problem("crowd_dynamics", domain=(0, 5), time_horizon=2.0)
    """

    def __init__(self, domain_bounds=(0, 1), time_horizon=1.0, **kwargs):
        super().__init__("create_mfg_problem() or solve_mfg()")
        self._domain_bounds = domain_bounds
        self._time_horizon = time_horizon
        self.metadata = kwargs

    def get_domain_bounds(self):
        return self._domain_bounds

    def get_time_horizon(self):
        return self._time_horizon

    def evaluate_hamiltonian(self, x, p, m, t):
        # Default simple Hamiltonian
        return 0.5 * p**2 + 0.1 * m

    def get_initial_density(self):
        import numpy as np

        x = np.linspace(*self._domain_bounds, 101)
        density = np.exp(-10 * (x - 0.2) ** 2)
        return density / np.trapezoid(density, x)

    def get_terminal_value(self):
        import numpy as np

        x = np.linspace(*self._domain_bounds, 101)
        return 0.5 * (x - self._domain_bounds[1]) ** 2


# Legacy problem class aliases
@deprecated("Use solve_mfg('crowd_dynamics') instead")
class ExampleMFGProblem(LegacyMFGProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = "crowd_dynamics"


@deprecated("Use create_mfg_problem('crowd_dynamics') instead")
class CrowdDynamicsProblem(LegacyMFGProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = "crowd_dynamics"


@deprecated("Use create_mfg_problem('portfolio_optimization') instead")
class PortfolioOptimizationProblem(LegacyMFGProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = "portfolio_optimization"


@deprecated("Use create_mfg_problem('traffic_flow') instead")
class TrafficFlowProblem(LegacyMFGProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = "traffic_flow"


@deprecated("Use create_mfg_problem('custom') instead")
class CustomMFGProblem(LegacyMFGProblem):
    pass
