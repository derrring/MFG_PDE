"""
Protocol definition for Eikonal solvers.

This module defines the EikonalSolver protocol that both FastMarchingMethod
and FastSweepingMethod implement.

Created: 2026-02-06 (Issue #664)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid


@runtime_checkable
class EikonalSolver(Protocol):
    """
    Protocol for Eikonal equation solvers.

    Eikonal solvers compute the solution to:
        |grad T(x)| = 1/F(x)

    where T is the arrival time (distance) and F is the speed function.

    Key methods:
    - solve(): General Eikonal equation with arbitrary speed and frozen values
    - compute_signed_distance(): Specialized for SDF computation from level set
    """

    geometry: TensorProductGrid

    def solve(
        self,
        speed: NDArray[np.float64] | float,
        frozen_mask: NDArray[np.bool_],
        frozen_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve the Eikonal equation |grad T| = 1/F.

        Args:
            speed: Speed function F(x) > 0, scalar or array matching grid shape.
                   T satisfies |grad T| = 1/F.
            frozen_mask: Boolean array marking points with known values (True = frozen).
                        These points act as boundary conditions.
            frozen_values: Values at frozen points. Non-frozen values are ignored.

        Returns:
            Solution array T with T[frozen_mask] = frozen_values[frozen_mask]
            and |grad T| = 1/F elsewhere.
        """
        ...

    def compute_signed_distance(
        self,
        phi_initial: NDArray[np.float64],
        subcell_accuracy: bool = True,
    ) -> NDArray[np.float64]:
        """
        Compute signed distance function from initial level set.

        Solves |grad phi| = 1 while preserving the zero level set of phi_initial.

        Args:
            phi_initial: Initial level set function. The zero level set {phi = 0}
                        is the interface from which distances are measured.
            subcell_accuracy: If True, use linear interpolation to locate the
                             interface with subcell precision (reduces grid bias).

        Returns:
            Signed distance function phi_sdf with:
            - |grad phi_sdf| = 1 everywhere
            - sign(phi_sdf) = sign(phi_initial)
            - {phi_sdf = 0} = {phi_initial = 0} (interface preserved)
        """
        ...
