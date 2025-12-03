"""
Particle Boundary Condition Utilities.

This module provides functions for applying boundary conditions to particles
in particle-based MFG solvers. Supports both GPU and CPU backends.

Key Functions:
- apply_boundary_conditions_gpu: GPU-accelerated BC application
- apply_boundary_conditions_numpy: CPU fallback

Boundary Types:
- periodic: Wrap particles around domain boundaries
- no_flux: Reflect particles at boundaries (mass conserving)
- dirichlet: Absorbing boundaries (clamp to domain)

Examples:
    >>> import numpy as np
    >>> from mfg_pde.utils.numerical.particle import apply_boundary_conditions_numpy
    >>>
    >>> particles = np.array([-0.5, 0.5, 1.5, 2.5])
    >>> particles = apply_boundary_conditions_numpy(particles, 0.0, 2.0, "periodic")
    >>> print(particles)  # [1.5, 0.5, 1.5, 0.5]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.backends.backend_protocol import BaseBackend


def apply_boundary_conditions_gpu(
    particles,
    xmin: float,
    xmax: float,
    bc_type: str,
    backend: BaseBackend,
):
    """
    Apply boundary conditions to particles on GPU.

    Supports periodic, no-flux (reflecting), and Dirichlet boundaries.
    All operations performed in parallel on GPU.

    Parameters
    ----------
    particles : backend tensor
        Particle positions, shape (N,)
    xmin : float
        Lower boundary
    xmax : float
        Upper boundary
    bc_type : str
        Boundary condition type: 'periodic', 'no_flux', or 'dirichlet'
    backend : BaseBackend
        Backend providing tensor operations

    Returns
    -------
    backend tensor
        Particles with boundary conditions applied, shape (N,)

    Notes
    -----
    - periodic: Particles wrap around the domain
    - no_flux: Particles reflect off boundaries (mass conserving)
    - dirichlet: Particles are clamped to domain (absorbing)
    """
    xp = backend.array_module
    Lx = xmax - xmin

    if bc_type == "periodic":
        # Wrap around: x -> xmin + (x - xmin) mod Lx
        particles = xmin + ((particles - xmin) % Lx)

    elif bc_type == "no_flux":
        # Reflecting boundaries
        # Left violation: x < xmin -> reflect to 2*xmin - x
        # Right violation: x > xmax -> reflect to 2*xmax - x

        # Use where (ternary) for GPU efficiency
        left_violations = particles < xmin
        right_violations = particles > xmax

        # Reflect left
        particles = xp.where(left_violations, 2 * xmin - particles, particles)

        # Reflect right
        particles = xp.where(right_violations, 2 * xmax - particles, particles)

    elif bc_type == "dirichlet":
        # Absorbing: clamp to domain
        if hasattr(xp, "clip"):
            particles = xp.clip(particles, xmin, xmax)
        else:
            # JAX doesn't have clip, use clamp
            particles = xp.minimum(xp.maximum(particles, xmin), xmax)

    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")

    return particles


def apply_boundary_conditions_numpy(
    particles: NDArray[np.floating],
    xmin: float,
    xmax: float,
    bc_type: str,
) -> NDArray[np.floating]:
    """
    CPU fallback for applying boundary conditions.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N,)
    xmin : float
        Lower boundary
    xmax : float
        Upper boundary
    bc_type : str
        Boundary condition type: 'periodic', 'no_flux', or 'dirichlet'

    Returns
    -------
    np.ndarray
        Particles with boundary conditions applied

    Examples
    --------
    >>> particles = np.array([-0.5, 0.5, 1.5, 2.5])
    >>> result = apply_boundary_conditions_numpy(particles, 0.0, 2.0, "periodic")
    >>> # particles wrap around: -0.5 -> 1.5, 2.5 -> 0.5
    """
    # Make a copy to avoid modifying input
    particles = particles.copy()
    Lx = xmax - xmin

    if bc_type == "periodic":
        particles = xmin + ((particles - xmin) % Lx)

    elif bc_type == "no_flux":
        left_violations = particles < xmin
        right_violations = particles > xmax

        particles[left_violations] = 2 * xmin - particles[left_violations]
        particles[right_violations] = 2 * xmax - particles[right_violations]

    elif bc_type == "dirichlet":
        particles = np.clip(particles, xmin, xmax)

    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")

    return particles


__all__ = [
    "apply_boundary_conditions_gpu",
    "apply_boundary_conditions_numpy",
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for particle boundary utilities."""
    print("Testing particle boundary utilities...")

    # Test periodic BC
    particles = np.array([-0.5, 0.5, 1.5, 2.5])
    result = apply_boundary_conditions_numpy(particles.copy(), 0.0, 2.0, "periodic")
    assert np.allclose(result, [1.5, 0.5, 1.5, 0.5])
    print("  Periodic BC: passed")

    # Test no-flux (reflecting) BC
    particles = np.array([-0.5, 0.5, 1.5, 2.5])
    result = apply_boundary_conditions_numpy(particles.copy(), 0.0, 2.0, "no_flux")
    assert np.allclose(result, [0.5, 0.5, 1.5, 1.5])  # -0.5 -> 0.5, 2.5 -> 1.5
    print("  No-flux BC: passed")

    # Test Dirichlet (absorbing) BC
    particles = np.array([-0.5, 0.5, 1.5, 2.5])
    result = apply_boundary_conditions_numpy(particles.copy(), 0.0, 2.0, "dirichlet")
    assert np.allclose(result, [0.0, 0.5, 1.5, 2.0])  # clamped
    print("  Dirichlet BC: passed")

    # Test that original is not modified
    original = np.array([2.5])
    result = apply_boundary_conditions_numpy(original, 0.0, 2.0, "no_flux")
    assert original[0] == 2.5, "Original should not be modified"
    print("  No mutation: passed")

    print("\nAll smoke tests passed!")
