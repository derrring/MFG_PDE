"""
Eikonal Equation Solvers for Level Set Methods.

This module provides Fast Marching Method (FMM) and Fast Sweeping Method (FSM)
for solving the Eikonal equation:

    |∇T(x)| = 1/F(x)

where T is the arrival time (or distance) and F is the speed function.

Key Applications:
- Signed distance function computation (reinitialization)
- Geodesic distance on implicit surfaces
- Optimal path planning

Algorithms:
- **Fast Marching Method (FMM)**: O(N log N) heap-based, optimal for general domains
- **Fast Sweeping Method (FSM)**: O(N) per sweep, efficient for simple domains

Mathematical Background:
    The Eikonal equation arises from the limit of the level set equation:
        ∂φ/∂t + F|∇φ| = 0

    At steady state with F = 1:
        |∇T| = 1  (signed distance function property)

    For general speed F(x) > 0:
        |∇T| = 1/F  (travel time from interface)

References:
- Sethian (1996): A fast marching level set method for monotonically advancing fronts
- Tsitsiklis (1995): Efficient algorithms for globally optimal trajectories
- Zhao (2005): A fast sweeping method for Eikonal equations

Created: 2026-02-06 (Issue #664 - Eikonal Solver Implementation)
"""

from mfg_pde.geometry.level_set.eikonal.corner_update import (
    eikonal_corner_update,
    identify_boundary_points,
    identify_corner_points,
)
from mfg_pde.geometry.level_set.eikonal.fast_marching import FastMarchingMethod
from mfg_pde.geometry.level_set.eikonal.fast_sweeping import FastSweepingMethod
from mfg_pde.geometry.level_set.eikonal.godunov_update import (
    godunov_update_1d,
    godunov_update_2d,
    godunov_update_nd,
)
from mfg_pde.geometry.level_set.eikonal.protocol import EikonalSolver

__all__ = [
    "EikonalSolver",
    "FastMarchingMethod",
    "FastSweepingMethod",
    "eikonal_corner_update",
    "godunov_update_1d",
    "godunov_update_2d",
    "godunov_update_nd",
    "identify_boundary_points",
    "identify_corner_points",
]
