"""
Validation benchmarks for MFG_PDE solvers.

This module contains validation tests comparing different solver implementations
on canonical MFG problems with known properties or established baselines.

Phase 3 Validation Suite:
- Problem 1: 2D crowd motion (FDM vs GFDM)
- Problem 2: 2D obstacle navigation (FDM vs GFDM)
- Problem 3: 2D congestion evacuation (FDM vs GFDM)
- Problem 4: 3D simple diffusion (optional)

Each validation test measures:
- L² solution error (vs baseline)
- Mass conservation
- Convergence rate
- Computational cost
"""

__all__ = [
    "test_2d_congestion_evacuation",
    "test_2d_crowd_motion",
    "test_2d_obstacle_navigation",
]
