"""
Numerical utilities for MFG computations.

This module provides numerical algorithms and helper functions commonly needed
in MFG research projects, including particle interpolation, spatial operations,
and computational utilities.
"""

from mfg_pde.utils.numerical.nonlinear_solvers import FixedPointSolver, NewtonSolver
from mfg_pde.utils.numerical.particle_interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)

__all__ = [
    "FixedPointSolver",
    "NewtonSolver",
    "estimate_kde_bandwidth",
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
]
