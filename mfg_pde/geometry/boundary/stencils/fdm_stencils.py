"""
FDM Boundary Stencils - Finite Difference Method boundary stencil coefficients.

This module provides stencil coefficients for various differential operators at
domain boundaries, supporting Dirichlet, Neumann, Robin, and periodic BCs.

Mathematical Background
=======================

For the standard 3-point Laplacian stencil at interior points:

    D * d^2u/dx^2 ≈ D * (u[i-1] - 2*u[i] + u[i+1]) / dx^2

At boundaries, we modify the stencil based on the BC type:

Neumann (du/dn = g)
-------------------
Use ghost point reflection. At left boundary (i=0):
- Ghost point u[-1] = u[1] - 2*dx*g  (for du/dx = g at x=0)
- For homogeneous Neumann (g=0): u[-1] = u[1]
- Stencil becomes: D * (2*u[1] - 2*u[0]) / dx^2 = 2D/dx^2 * (u[1] - u[0])

For MASS CONSERVATION in Fokker-Planck:
- We need row sum = 0
- Conservative stencil: D/dx^2 * (u[1] - u[0])
- Diagonal: D/dx^2, Neighbor: -D/dx^2, Row sum = 0

Dirichlet (u = g)
-----------------
Strong enforcement eliminates DOF:
- Row becomes: 1*u[0] = g
- Diagonal: 1, RHS: g

Robin (alpha*u + beta*du/dn = g)
--------------------------------
Combines Dirichlet and Neumann using ghost point.

Usage
=====

Get stencil for Laplacian at left boundary with no-flux BC::

    from mfg_pde.geometry.boundary.stencils import FDMBoundaryStencils
    from mfg_pde.geometry.boundary import BCType

    stencil = FDMBoundaryStencils.diffusion_laplacian(
        bc_type=BCType.NEUMANN,
        position="left",
        dx=0.1,
        diffusion_coeff=0.05,  # sigma^2/2 for FP
    )

    # Use in matrix assembly
    A[0, 0] += stencil.diagonal      # +D/dx^2
    A[0, 1] = stencil.neighbor       # -D/dx^2
    # Row sum = 0 -> mass conserving
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .base import BoundaryStencil, OperatorType

if TYPE_CHECKING:
    from mfg_pde.geometry.boundary.types import BCType


class FDMBoundaryStencils:
    """
    Factory class for FDM boundary stencil coefficients.

    Provides static methods to generate stencil coefficients for different
    combinations of BC types and differential operators.
    """

    @staticmethod
    def diffusion_laplacian(
        bc_type: BCType | str,
        position: Literal["left", "right"],
        dx: float,
        diffusion_coeff: float,
        bc_value: float = 0.0,
    ) -> BoundaryStencil:
        """
        Get stencil for diffusion (Laplacian) operator at boundary.

        Implements the discretization of D * d^2u/dx^2 at boundary points
        with proper handling of various BC types.

        Parameters
        ----------
        bc_type : BCType | str
            Type of boundary condition. Supported: NEUMANN, DIRICHLET, ROBIN.

        position : {"left", "right"}
            Which boundary (left = x_min, right = x_max).

        dx : float
            Grid spacing.

        diffusion_coeff : float
            Diffusion coefficient D. For Fokker-Planck, this is sigma^2/2.

        bc_value : float, optional
            BC value (g in du/dn=g or u=g). Default 0 (homogeneous).

        Returns
        -------
        BoundaryStencil
            Stencil coefficients for matrix assembly.

        Notes
        -----
        For mass conservation in Fokker-Planck equations, use NEUMANN BC type
        with bc_value=0 (no-flux). The resulting stencil has row sum = 0.

        Examples
        --------
        No-flux BC for Fokker-Planck at left boundary::

            stencil = FDMBoundaryStencils.diffusion_laplacian(
                bc_type=BCType.NEUMANN,
                position="left",
                dx=0.1,
                diffusion_coeff=0.05,  # sigma^2/2
            )

            assert stencil.is_conservative()  # Row sum = 0
        """
        # Normalize bc_type to string for comparison
        bc_type_str = bc_type.name if hasattr(bc_type, "name") else str(bc_type).upper()

        dx_sq = dx * dx
        neighbor_offset = 1 if position == "left" else -1

        if bc_type_str in ("NEUMANN", "NO_FLUX"):
            # Neumann / No-flux: du/dn = bc_value (typically 0)
            # Ghost point reflection: u_ghost = u_boundary - 2*dx*bc_value
            # For homogeneous (bc_value=0): u_ghost = u_boundary
            #
            # Standard Laplacian at boundary:
            #   D * (u_ghost - 2*u_boundary + u_neighbor) / dx^2
            # With u_ghost = u_boundary (homogeneous):
            #   D * (u_boundary - 2*u_boundary + u_neighbor) / dx^2
            #   = D * (u_neighbor - u_boundary) / dx^2
            #
            # Conservative form (row sum = 0):
            #   diagonal = D/dx^2
            #   neighbor = -D/dx^2
            #   row sum = D/dx^2 - D/dx^2 = 0

            diagonal = diffusion_coeff / dx_sq
            neighbor = -diffusion_coeff / dx_sq

            # Non-homogeneous Neumann: add RHS term from ghost point
            # u_ghost = u_boundary - 2*dx*g  (for outward normal)
            # Sign depends on position
            if position == "left":
                # Left boundary: outward normal points left (negative x)
                # du/dn = -du/dx = g => du/dx = -g
                rhs_value = -2.0 * diffusion_coeff * bc_value / dx
            else:
                # Right boundary: outward normal points right (positive x)
                # du/dn = du/dx = g
                rhs_value = 2.0 * diffusion_coeff * bc_value / dx

            return BoundaryStencil(
                diagonal=diagonal,
                neighbor=neighbor,
                neighbor_offset=neighbor_offset,
                rhs_value=rhs_value,
                eliminates_dof=False,
                preserves_conservation=(bc_value == 0.0),  # Only homogeneous is conservative
                bc_type="neumann",
                operator_type=OperatorType.LAPLACIAN,
            )

        elif bc_type_str == "DIRICHLET":
            # Strong Dirichlet: u = bc_value
            # Eliminates DOF: row becomes [1, 0, ...] = bc_value
            return BoundaryStencil(
                diagonal=1.0,
                neighbor=0.0,
                neighbor_offset=neighbor_offset,
                rhs_value=bc_value,
                eliminates_dof=True,
                preserves_conservation=False,  # Dirichlet breaks conservation
                bc_type="dirichlet",
                operator_type=OperatorType.LAPLACIAN,
            )

        elif bc_type_str == "ROBIN":
            # Robin: alpha*u + beta*du/dn = g
            # Default: alpha=1, beta=1 (can be extended)
            # Using ghost point: u_ghost = u_boundary - 2*dx/beta * (g - alpha*u_boundary)
            #
            # For simplicity, implement the standard form with alpha=beta=1:
            # u + du/dn = g
            # Ghost point: u_ghost = u_boundary - 2*dx*(g - u_boundary)
            #            = u_boundary*(1 + 2*dx) - 2*dx*g
            #
            # Laplacian: D*(u_ghost - 2*u_boundary + u_neighbor)/dx^2
            # After substitution, diagonal and neighbor coefficients emerge.

            # Simplified Robin with alpha=beta=1
            alpha = 1.0
            beta = 1.0

            # Ghost point coefficient for u_boundary
            ghost_coeff = 1.0 + 2.0 * dx * alpha / beta

            # Effective stencil after substituting ghost point
            diagonal = diffusion_coeff * (ghost_coeff - 2.0) / dx_sq
            neighbor = diffusion_coeff / dx_sq
            rhs_value = -2.0 * diffusion_coeff * bc_value / (beta * dx)

            return BoundaryStencil(
                diagonal=diagonal,
                neighbor=neighbor,
                neighbor_offset=neighbor_offset,
                rhs_value=rhs_value,
                eliminates_dof=False,
                preserves_conservation=False,
                bc_type="robin",
                operator_type=OperatorType.LAPLACIAN,
            )

        elif bc_type_str == "PERIODIC":
            # Periodic: u[0] = u[N], u[-1] = u[N-1]
            # Handled differently - connects boundaries
            # This requires knowing total grid size, so we return a marker
            raise NotImplementedError(
                "Periodic BC stencils require knowledge of grid size. "
                "Use FDMBoundaryStencils.periodic_laplacian() with grid_size parameter."
            )

        else:
            raise ValueError(f"Unsupported BC type for diffusion_laplacian: {bc_type_str}")

    @staticmethod
    def advection_upwind(
        bc_type: BCType | str,
        position: Literal["left", "right"],
        dx: float,
        velocity: float,
        bc_value: float = 0.0,
    ) -> BoundaryStencil:
        """
        Get stencil for advection operator at boundary using upwind discretization.

        Implements the discretization of v * du/dx at boundary points.
        Uses first-order upwind for stability.

        Parameters
        ----------
        bc_type : BCType | str
            Type of boundary condition.

        position : {"left", "right"}
            Which boundary.

        dx : float
            Grid spacing.

        velocity : float
            Advection velocity v. Sign determines upwind direction.

        bc_value : float, optional
            BC value. For no-flux in FP, this controls the advective flux.

        Returns
        -------
        BoundaryStencil
            Stencil coefficients for matrix assembly.

        Notes
        -----
        For Fokker-Planck no-flux BC, the advective flux alpha*m should be zero
        at the boundary. This is enforced by setting the advection stencil to
        zero at no-flux boundaries.
        """
        bc_type_str = bc_type.name if hasattr(bc_type, "name") else str(bc_type).upper()

        neighbor_offset = 1 if position == "left" else -1

        if bc_type_str in ("NEUMANN", "NO_FLUX"):
            # No-flux for Fokker-Planck: advective flux = 0 at boundary
            # This means we don't add advection contribution at boundary
            # The density can't flow out through advection
            return BoundaryStencil(
                diagonal=0.0,
                neighbor=0.0,
                neighbor_offset=neighbor_offset,
                rhs_value=0.0,
                eliminates_dof=False,
                preserves_conservation=True,  # Zero flux preserves mass
                bc_type="neumann",
                operator_type=OperatorType.ADVECTION,
            )

        elif bc_type_str == "DIRICHLET":
            # Dirichlet for advection: use ghost point with prescribed value
            # Upwind scheme: if v > 0, use backward difference
            #                if v < 0, use forward difference

            if position == "left":
                # Left boundary
                if velocity >= 0:
                    # Flow coming in from left (ghost region)
                    # Use backward difference: v*(u[0] - u_ghost)/dx
                    # u_ghost = bc_value
                    diagonal = velocity / dx
                    neighbor = 0.0
                    rhs_value = velocity * bc_value / dx
                else:
                    # Flow going out to the left
                    # Use forward difference: v*(u[1] - u[0])/dx
                    diagonal = -velocity / dx
                    neighbor = velocity / dx
                    rhs_value = 0.0
            else:  # position == "right"
                if velocity >= 0:
                    # Flow going out to the right
                    # Use backward difference: v*(u[N] - u[N-1])/dx
                    diagonal = velocity / dx
                    neighbor = -velocity / dx
                    rhs_value = 0.0
                else:
                    # Flow coming in from right (ghost region)
                    # Use forward difference: v*(u_ghost - u[N])/dx
                    # u_ghost = bc_value
                    diagonal = -velocity / dx
                    neighbor = 0.0
                    rhs_value = -velocity * bc_value / dx

            return BoundaryStencil(
                diagonal=diagonal,
                neighbor=neighbor,
                neighbor_offset=neighbor_offset,
                rhs_value=rhs_value,
                eliminates_dof=False,
                preserves_conservation=False,
                bc_type="dirichlet",
                operator_type=OperatorType.ADVECTION,
            )

        else:
            raise ValueError(f"Unsupported BC type for advection_upwind: {bc_type_str}")

    @staticmethod
    def fokker_planck_noflux(
        position: Literal["left", "right"],
        dx: float,
        sigma: float,
        alpha: float = 0.0,
    ) -> BoundaryStencil:
        """
        Get combined stencil for Fokker-Planck equation with no-flux BC.

        The Fokker-Planck equation is:
            dm/dt = -div(alpha * m) + (sigma^2/2) * Laplacian(m)

        No-flux BC requires:
            j * n = 0  where j = alpha*m - (sigma^2/2)*grad(m)

        This method returns a conservative stencil that ensures zero net flux
        at the boundary, preserving total mass.

        Parameters
        ----------
        position : {"left", "right"}
            Which boundary.

        dx : float
            Grid spacing.

        sigma : float
            Diffusion coefficient (noise intensity). The diffusion coefficient
            is sigma^2/2.

        alpha : float, optional
            Advection velocity at boundary. For no-flux, this is typically 0
            or the value is such that the total flux is zero.

        Returns
        -------
        BoundaryStencil
            Combined conservative stencil for FP equation.

        Notes
        -----
        This is a convenience method that ensures the complete FP operator
        at the boundary is mass-conserving. The stencil combines diffusion
        and advection terms in a way that guarantees row sum = 0.

        For pure diffusion (alpha=0), this is equivalent to:
            diffusion_laplacian(NEUMANN, position, dx, sigma^2/2, bc_value=0)
        """
        diffusion_coeff = sigma * sigma / 2.0
        neighbor_offset = 1 if position == "left" else -1
        dx_sq = dx * dx

        # Diffusion part: conservative Neumann
        diff_diagonal = diffusion_coeff / dx_sq
        diff_neighbor = -diffusion_coeff / dx_sq

        # Advection part: zero at no-flux boundary
        # For proper no-flux, advective flux should be zero
        adv_diagonal = 0.0
        adv_neighbor = 0.0

        # Combined (note: signs depend on how FP is discretized)
        # Standard form: dm/dt = -div(alpha*m) + D*Laplacian(m)
        # The advection enters with negative sign in the operator
        diagonal = diff_diagonal - adv_diagonal
        neighbor = diff_neighbor - adv_neighbor

        return BoundaryStencil(
            diagonal=diagonal,
            neighbor=neighbor,
            neighbor_offset=neighbor_offset,
            rhs_value=0.0,
            eliminates_dof=False,
            preserves_conservation=True,
            bc_type="no_flux",
            operator_type=OperatorType.FOKKER_PLANCK,
        )


if __name__ == "__main__":
    """Quick smoke test for FDM boundary stencils."""
    from mfg_pde.geometry.boundary.types import BCType

    print("FDM Boundary Stencils - Smoke Test")
    print("=" * 50)

    # Test 1: Neumann BC for diffusion (mass-conserving)
    print("\n1. Neumann BC for diffusion (no-flux):")
    stencil = FDMBoundaryStencils.diffusion_laplacian(
        bc_type=BCType.NEUMANN,
        position="left",
        dx=0.1,
        diffusion_coeff=0.05,
    )
    print(f"   Diagonal: {stencil.diagonal:.4f}")
    print(f"   Neighbor: {stencil.neighbor:.4f}")
    print(f"   Row sum: {stencil.row_sum():.6f}")
    print(f"   Is conservative: {stencil.is_conservative()}")
    assert stencil.is_conservative(), "Neumann stencil should be conservative"
    print("   PASS")

    # Test 2: Dirichlet BC
    print("\n2. Dirichlet BC for diffusion:")
    stencil = FDMBoundaryStencils.diffusion_laplacian(
        bc_type=BCType.DIRICHLET,
        position="left",
        dx=0.1,
        diffusion_coeff=0.05,
        bc_value=1.0,
    )
    print(f"   Diagonal: {stencil.diagonal:.4f}")
    print(f"   Neighbor: {stencil.neighbor:.4f}")
    print(f"   RHS value: {stencil.rhs_value:.4f}")
    print(f"   Eliminates DOF: {stencil.eliminates_dof}")
    assert stencil.eliminates_dof, "Dirichlet should eliminate DOF"
    print("   PASS")

    # Test 3: Fokker-Planck no-flux
    print("\n3. Fokker-Planck no-flux BC:")
    stencil = FDMBoundaryStencils.fokker_planck_noflux(
        position="left",
        dx=0.1,
        sigma=0.3,
    )
    print(f"   Diagonal: {stencil.diagonal:.4f}")
    print(f"   Neighbor: {stencil.neighbor:.4f}")
    print(f"   Row sum: {stencil.row_sum():.6f}")
    print(f"   Preserves conservation: {stencil.preserves_conservation}")
    assert stencil.is_conservative(), "FP no-flux should be conservative"
    print("   PASS")

    # Test 4: Advection with no-flux
    print("\n4. Advection with no-flux BC:")
    stencil = FDMBoundaryStencils.advection_upwind(
        bc_type=BCType.NEUMANN,
        position="left",
        dx=0.1,
        velocity=1.0,
    )
    print(f"   Diagonal: {stencil.diagonal:.4f}")
    print(f"   Neighbor: {stencil.neighbor:.4f}")
    print(f"   Is conservative: {stencil.is_conservative()}")
    assert stencil.diagonal == 0.0, "No-flux advection diagonal should be zero"
    assert stencil.neighbor == 0.0, "No-flux advection neighbor should be zero"
    print("   PASS")

    print("\n" + "=" * 50)
    print("All smoke tests passed!")
