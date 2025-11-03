"""
Numerical utilities for MFG computations.

This module provides numerical algorithms and helper functions commonly needed
in MFG research projects, including particle interpolation, signed distance
functions, spatial operations, and computational utilities.
"""

from mfg_pde.utils.numerical.nonlinear_solvers import FixedPointSolver, NewtonSolver
from mfg_pde.utils.numerical.particle_interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)
from mfg_pde.utils.numerical.sdf_utils import (
    sdf_box,
    sdf_complement,
    sdf_difference,
    sdf_gradient,
    sdf_intersection,
    sdf_smooth_intersection,
    sdf_smooth_union,
    sdf_sphere,
    sdf_union,
)

__all__ = [
    # Nonlinear solvers
    "FixedPointSolver",
    "NewtonSolver",
    # Particle interpolation
    "estimate_kde_bandwidth",
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
    # Signed distance functions
    "sdf_box",
    "sdf_complement",
    "sdf_difference",
    "sdf_gradient",
    "sdf_intersection",
    "sdf_smooth_intersection",
    "sdf_smooth_union",
    "sdf_sphere",
    "sdf_union",
]
