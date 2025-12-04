"""
Model Factory Module - Domain-Specific MFG Problems

This module provides flexible, parameterized MFG problem classes for common domains.
Examples in examples/models/ show concrete realizations with fixed parameters.

Planned Models (see Issue #354):
- LQProblem: Linear-Quadratic baseline (well-understood, validation)
- CrowdDynamicsProblem: Crowd evacuation/movement
- TrafficFlowProblem: Traffic network optimization
- FinanceProblem: Portfolio/trading MFG
- EpidemicProblem: Disease spread dynamics

Usage (future):
    from mfg_pde.factory.models import CrowdDynamicsProblem
    from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver

    problem = CrowdDynamicsProblem(
        room_bounds=(0, 10, 0, 10),
        exits=[(10, 5)],
        congestion_model="soft",
    )
    solver = HJBFDMSolver(problem)  # User configures solver directly
"""

__all__: list[str] = []

# Placeholder - models will be implemented in future PRs
# See GitHub Issue #354 for implementation plan
