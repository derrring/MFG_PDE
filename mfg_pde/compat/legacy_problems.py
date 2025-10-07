"""
Legacy problem compatibility wrappers

Provides compatibility for old problem APIs with deprecation warnings.
"""

from . import DeprecatedAPI, deprecated


@deprecated("Use MFGProblem class with factory API instead")
class LegacyMFGProblem(DeprecatedAPI):
    """
    DEPRECATED: Legacy MFG problem wrapper.

    Use the new factory API instead:

    Old:
        problem = LegacyMFGProblem(domain, time_horizon)

    New (Factory API):
        from mfg_pde import MFGProblem
        from mfg_pde.factory import create_fast_solver

        class CustomProblem(MFGProblem):
            def __init__(self):
                super().__init__(T=2.0, Nt=20, xmin=0.0, xmax=5.0, Nx=50)

            def g(self, x):
                return 0.5 * (x - 5.0)**2

            def rho0(self, x):
                return np.exp(-10 * (x - 1.0)**2)

        problem = CustomProblem()
        solver = create_fast_solver(problem, solver_type="fixed_point")
        result = solver.solve()
    """

    def __init__(self, domain_bounds=(0, 1), time_horizon=1.0, **kwargs):
        super().__init__("MFGProblem class with factory API")
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
@deprecated("Use MFGProblem class with factory API instead")
class ExampleMFGProblem(LegacyMFGProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = "crowd_dynamics"


@deprecated("Use MFGProblem class with factory API instead")
class CrowdDynamicsProblem(LegacyMFGProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = "crowd_dynamics"


@deprecated("Use MFGProblem class with factory API instead")
class PortfolioOptimizationProblem(LegacyMFGProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = "portfolio_optimization"


@deprecated("Use MFGProblem class with factory API instead")
class TrafficFlowProblem(LegacyMFGProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = "traffic_flow"


@deprecated("Use MFGProblem class with factory API instead")
class CustomMFGProblem(LegacyMFGProblem):
    pass
