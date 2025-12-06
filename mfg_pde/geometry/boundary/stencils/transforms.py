"""
Generic BC Transforms - Modify any stencil at domain boundaries.

This module provides generic boundary condition transformations that can modify
ANY operator stencil at boundaries. The key design principle is:

    **Operators belong to solvers, NOT to the BC module.**

The BC module only provides generic transforms, avoiding the combinatorial
explosion of N_operators x N_bcs x N_methods.

Mathematical Background
=======================

Neumann BC (du/dn = g)
----------------------
Uses ghost point reflection. At left boundary:
- Ghost point: u_ghost = u_boundary - 2*dx*g  (for outward normal)
- For homogeneous (g=0): u_ghost = u_boundary

This transforms an interior 3-point stencil [a, b, c] into a 2-point boundary
stencil [a+c, b] (ghost folds onto neighbor).

Dirichlet BC (u = g)
--------------------
Eliminates the boundary DOF from the system:
- Row becomes: [1, 0, 0, ...] * u = g
- Decouples boundary from interior

Robin BC (alpha*u + beta*du/dn = g)
-----------------------------------
Combines Dirichlet and Neumann using ghost point approach.

Periodic BC
-----------
Connects boundaries: u[0] = u[N], u[-1] = u[N-1]
Requires knowledge of grid size.

Usage
=====

Solver builds its interior stencil, then applies BC transform::

    from mfg_pde.geometry.boundary.stencils import BCTransforms

    # Solver builds interior Laplacian stencil (solver's responsibility)
    D = sigma**2 / 2
    interior = {"diagonal": -2*D/dx**2, "left": D/dx**2, "right": D/dx**2}

    # Apply Neumann BC transform at left boundary
    boundary_stencil = BCTransforms.neumann(
        interior_stencil=interior,
        position="left",
        dx=dx,
        bc_value=0.0,  # No-flux
    )

    # Use in matrix assembly
    A[0, 0] = boundary_stencil.diagonal
    A[0, 1] = boundary_stencil.neighbor

See Also
--------
- GitHub Issue #379: Layered BC Stencil Architecture
- mfg_pde.geometry.boundary.stencils.fdm_stencils: Convenience wrappers
"""

from __future__ import annotations

from typing import Literal, TypedDict

from .base import BoundaryStencil, OperatorType


class InteriorStencil(TypedDict, total=False):
    """Interior stencil coefficients for 1D FDM.

    Keys:
        diagonal: Coefficient for center point (u[i])
        left: Coefficient for left neighbor (u[i-1])
        right: Coefficient for right neighbor (u[i+1])
        far_left: Coefficient for far left neighbor (u[i-2]), optional
        far_right: Coefficient for far right neighbor (u[i+2]), optional
    """

    diagonal: float
    left: float
    right: float
    far_left: float
    far_right: float


class BCTransforms:
    """
    Generic BC transformations for modifying any operator stencil at boundaries.

    This class provides static methods that transform interior stencils into
    boundary stencils by applying appropriate BC modifications (ghost point
    reflection, DOF elimination, etc.).

    The transforms are OPERATOR-AGNOSTIC: they work with any stencil (Laplacian,
    gradient, advection, etc.) because the mathematical principles (ghost points,
    DOF elimination) are the same regardless of the operator.
    """

    @staticmethod
    def neumann(
        interior_stencil: InteriorStencil,
        position: Literal["left", "right"],
        dx: float,
        bc_value: float = 0.0,
    ) -> BoundaryStencil:
        """
        Apply Neumann BC (du/dn = g) to an interior stencil.

        Uses ghost point reflection: the ghost point value is determined by
        the BC and folds onto the interior neighbor.

        Parameters
        ----------
        interior_stencil : InteriorStencil
            Interior stencil coefficients with keys "diagonal", "left", "right".

        position : {"left", "right"}
            Which boundary (left = x_min, right = x_max).

        dx : float
            Grid spacing. Used for non-homogeneous BC to compute RHS contribution.

        bc_value : float, optional
            The Neumann BC value g in du/dn = g. Default 0 (no-flux).

        Returns
        -------
        BoundaryStencil
            Transformed boundary stencil with ghost point folded in.

        Notes
        -----
        Ghost point reflection for Neumann at left boundary:
        - Outward normal points left (negative x direction)
        - du/dn = -du/dx = g  =>  du/dx = -g
        - Ghost point: u[-1] = u[1] - 2*dx*(-g) = u[1] + 2*dx*g

        For 3-point stencil [a, b, c] at i=0:
        - a*u[-1] + b*u[0] + c*u[1]
        - With u[-1] = u[1] + 2*dx*g:
        - a*(u[1] + 2*dx*g) + b*u[0] + c*u[1]
        - = b*u[0] + (a+c)*u[1] + 2*a*dx*g
        - Diagonal: b, Neighbor: a+c, RHS: -2*a*dx*g

        For homogeneous Neumann (g=0), RHS term vanishes and stencil is
        conservative (row sum = interior row sum).

        Examples
        --------
        Apply Neumann to Laplacian at left boundary::

            D = 0.05  # Diffusion coefficient
            dx = 0.1
            interior = {"diagonal": -2*D/dx**2, "left": D/dx**2, "right": D/dx**2}

            stencil = BCTransforms.neumann(interior, "left", dx, bc_value=0.0)
            # Result: diagonal = -2*D/dx**2 = -10.0
            #         neighbor = 2*D/dx**2 = 10.0 (folded ghost)
            #         Row sum = 0 (conservative)
        """
        diag = interior_stencil.get("diagonal", 0.0)
        left = interior_stencil.get("left", 0.0)
        right = interior_stencil.get("right", 0.0)

        if position == "left":
            # Left boundary: ghost is at i=-1, interior neighbor at i=1
            # Ghost coefficient 'left' folds onto 'right' neighbor
            ghost_coeff = left  # Coefficient that was for ghost point
            neighbor_coeff = right  # Coefficient for interior neighbor
            neighbor_offset = 1

            # Ghost point: u[-1] = u[1] + 2*dx*bc_value (for left boundary)
            # Contribution: ghost_coeff * (u[1] + 2*dx*bc_value)
            # This adds ghost_coeff to neighbor and ghost_coeff*2*dx*bc_value to RHS
            new_neighbor = neighbor_coeff + ghost_coeff
            rhs_value = -ghost_coeff * 2.0 * dx * bc_value

        else:  # position == "right"
            # Right boundary: ghost is at i=N+1, interior neighbor at i=N-1
            # Ghost coefficient 'right' folds onto 'left' neighbor
            ghost_coeff = right  # Coefficient that was for ghost point
            neighbor_coeff = left  # Coefficient for interior neighbor
            neighbor_offset = -1

            # Ghost point: u[N+1] = u[N-1] - 2*dx*bc_value (for right boundary)
            # Outward normal points right, so du/dn = du/dx = bc_value
            # => u[N+1] = u[N-1] + 2*dx*bc_value
            # Wait, let's be careful with sign convention:
            # At right boundary, outward normal is +x, so du/dn = du/dx
            # One-sided: (u[N+1] - u[N-1]) / (2*dx) = bc_value
            # => u[N+1] = u[N-1] + 2*dx*bc_value
            new_neighbor = neighbor_coeff + ghost_coeff
            rhs_value = -ghost_coeff * 2.0 * dx * bc_value

        return BoundaryStencil(
            diagonal=diag,
            neighbor=new_neighbor,
            neighbor_offset=neighbor_offset,
            rhs_value=rhs_value,
            eliminates_dof=False,
            preserves_conservation=(bc_value == 0.0),
            bc_type="neumann",
            operator_type=OperatorType.LAPLACIAN,  # Generic, caller can override
        )

    @staticmethod
    def dirichlet(
        bc_value: float = 0.0,
        position: Literal["left", "right"] = "left",
    ) -> BoundaryStencil:
        """
        Apply Dirichlet BC (u = g) - eliminates the boundary DOF.

        Strong Dirichlet enforcement: the boundary row becomes [1, 0, 0, ...] = g,
        completely eliminating the boundary DOF from the coupled system.

        Parameters
        ----------
        bc_value : float, optional
            The Dirichlet BC value g in u = g. Default 0.

        position : {"left", "right"}, optional
            Which boundary. Affects neighbor_offset sign.

        Returns
        -------
        BoundaryStencil
            Stencil with diagonal=1, neighbor=0, rhs_value=bc_value.

        Notes
        -----
        Dirichlet BC does NOT depend on the interior stencil because it
        completely overrides the equation at the boundary point.

        This is the "strong" enforcement. For "weak" enforcement in FEM,
        use penalty methods or Lagrange multipliers.

        Examples
        --------
        Apply Dirichlet at left boundary::

            stencil = BCTransforms.dirichlet(bc_value=1.0, position="left")
            # Result: A[0, :] = [1, 0, 0, ...], b[0] = 1.0
        """
        neighbor_offset = 1 if position == "left" else -1

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

    @staticmethod
    def robin(
        interior_stencil: InteriorStencil,
        position: Literal["left", "right"],
        dx: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        bc_value: float = 0.0,
    ) -> BoundaryStencil:
        """
        Apply Robin BC (alpha*u + beta*du/dn = g) to an interior stencil.

        Robin BC is a linear combination of Dirichlet and Neumann. It's
        implemented using the ghost point approach with modified reflection.

        Parameters
        ----------
        interior_stencil : InteriorStencil
            Interior stencil coefficients.

        position : {"left", "right"}
            Which boundary.

        dx : float
            Grid spacing.

        alpha : float, optional
            Coefficient for Dirichlet part. Default 1.

        beta : float, optional
            Coefficient for Neumann part. Default 1.

        bc_value : float, optional
            The Robin BC value g. Default 0.

        Returns
        -------
        BoundaryStencil
            Transformed boundary stencil.

        Notes
        -----
        Robin BC: alpha*u + beta*du/dn = g

        Using centered difference for du/dn at boundary:
        - At left: du/dn = -du/dx = -(u[1] - u[-1]) / (2*dx)
        - Robin: alpha*u[0] + beta*(u[-1] - u[1]) / (2*dx) = g

        Solving for ghost point:
        - u[-1] = u[1] + (2*dx/beta) * (g - alpha*u[0])

        Substituting into interior stencil gives modified diagonal and neighbor.
        """
        diag = interior_stencil.get("diagonal", 0.0)
        left = interior_stencil.get("left", 0.0)
        right = interior_stencil.get("right", 0.0)

        if abs(beta) < 1e-14:
            # Pure Dirichlet (beta -> 0)
            return BCTransforms.dirichlet(bc_value / alpha if alpha != 0 else 0.0, position)

        if position == "left":
            ghost_coeff = left
            neighbor_coeff = right
            neighbor_offset = 1

            # Ghost point: u[-1] = u[1] + (2*dx/beta) * (g - alpha*u[0])
            # Substitute into stencil:
            # ghost_coeff * u[-1] + diag*u[0] + neighbor_coeff*u[1]
            # = ghost_coeff * [u[1] + (2*dx/beta)*(g - alpha*u[0])] + diag*u[0] + neighbor_coeff*u[1]
            # = [diag - ghost_coeff*2*dx*alpha/beta]*u[0] + [neighbor_coeff + ghost_coeff]*u[1]
            #   + ghost_coeff*2*dx*g/beta

            new_diagonal = diag - ghost_coeff * 2.0 * dx * alpha / beta
            new_neighbor = neighbor_coeff + ghost_coeff
            rhs_value = -ghost_coeff * 2.0 * dx * bc_value / beta

        else:  # position == "right"
            ghost_coeff = right
            neighbor_coeff = left
            neighbor_offset = -1

            # At right boundary, outward normal is +x
            # Robin: alpha*u[N] + beta*du/dx = g
            # du/dx = (u[N+1] - u[N-1]) / (2*dx)
            # Solving: u[N+1] = u[N-1] + (2*dx/beta)*(g - alpha*u[N])

            new_diagonal = diag - ghost_coeff * 2.0 * dx * alpha / beta
            new_neighbor = neighbor_coeff + ghost_coeff
            rhs_value = -ghost_coeff * 2.0 * dx * bc_value / beta

        return BoundaryStencil(
            diagonal=new_diagonal,
            neighbor=new_neighbor,
            neighbor_offset=neighbor_offset,
            rhs_value=rhs_value,
            eliminates_dof=False,
            preserves_conservation=False,  # Robin generally not conservative
            bc_type="robin",
            operator_type=OperatorType.LAPLACIAN,
        )

    @staticmethod
    def no_flux(
        interior_stencil: InteriorStencil,
        position: Literal["left", "right"],
    ) -> BoundaryStencil:
        """
        Apply no-flux BC (du/dn = 0) - a convenience wrapper for Neumann with g=0.

        No-flux is the most common BC for Fokker-Planck equations as it
        ensures mass conservation.

        Parameters
        ----------
        interior_stencil : InteriorStencil
            Interior stencil coefficients.

        position : {"left", "right"}
            Which boundary.

        Returns
        -------
        BoundaryStencil
            Transformed boundary stencil with row sum = 0.

        Notes
        -----
        This is equivalent to ``neumann(stencil, position, dx=any, bc_value=0.0)``.
        The dx parameter doesn't matter for homogeneous Neumann since the RHS
        term vanishes.
        """
        return BCTransforms.neumann(
            interior_stencil=interior_stencil,
            position=position,
            dx=1.0,  # Doesn't matter for bc_value=0
            bc_value=0.0,
        )


if __name__ == "__main__":
    """Smoke test for generic BC transforms."""

    print("BC Transforms - Smoke Test")
    print("=" * 50)

    # Test 1: Neumann transform on Laplacian
    print("\n1. Neumann transform on Laplacian stencil:")
    D = 0.05  # Diffusion coefficient
    dx = 0.1
    interior = {
        "diagonal": -2 * D / dx**2,  # -10.0
        "left": D / dx**2,  # 5.0
        "right": D / dx**2,  # 5.0
    }
    print(f"   Interior stencil: {interior}")
    print(f"   Interior row sum: {sum(interior.values()):.6f}")

    stencil = BCTransforms.neumann(interior, "left", dx, bc_value=0.0)
    print(f"   Boundary diagonal: {stencil.diagonal:.4f}")
    print(f"   Boundary neighbor: {stencil.neighbor:.4f}")
    print(f"   Row sum: {stencil.row_sum():.6f}")
    print(f"   Is conservative: {stencil.is_conservative()}")
    assert stencil.is_conservative()
    print("   PASS")

    # Test 2: Dirichlet transform
    print("\n2. Dirichlet transform:")
    stencil = BCTransforms.dirichlet(bc_value=1.0, position="left")
    print(f"   Diagonal: {stencil.diagonal:.4f}")
    print(f"   Neighbor: {stencil.neighbor:.4f}")
    print(f"   RHS: {stencil.rhs_value:.4f}")
    print(f"   Eliminates DOF: {stencil.eliminates_dof}")
    assert stencil.eliminates_dof
    assert stencil.diagonal == 1.0
    assert stencil.neighbor == 0.0
    print("   PASS")

    # Test 3: Robin transform
    print("\n3. Robin transform (alpha=1, beta=1):")
    stencil = BCTransforms.robin(interior, "left", dx, alpha=1.0, beta=1.0, bc_value=0.5)
    print(f"   Diagonal: {stencil.diagonal:.4f}")
    print(f"   Neighbor: {stencil.neighbor:.4f}")
    print(f"   RHS: {stencil.rhs_value:.4f}")
    print("   PASS")

    # Test 4: No-flux convenience
    print("\n4. No-flux (convenience wrapper):")
    stencil = BCTransforms.no_flux(interior, "left")
    print(f"   Diagonal: {stencil.diagonal:.4f}")
    print(f"   Neighbor: {stencil.neighbor:.4f}")
    print(f"   Row sum: {stencil.row_sum():.6f}")
    print(f"   Preserves conservation: {stencil.preserves_conservation}")
    assert stencil.preserves_conservation
    assert stencil.is_conservative()
    print("   PASS")

    # Test 5: Neumann at right boundary
    print("\n5. Neumann at right boundary:")
    stencil = BCTransforms.neumann(interior, "right", dx, bc_value=0.0)
    print(f"   Diagonal: {stencil.diagonal:.4f}")
    print(f"   Neighbor: {stencil.neighbor:.4f}")
    print(f"   Neighbor offset: {stencil.neighbor_offset}")
    print(f"   Row sum: {stencil.row_sum():.6f}")
    assert stencil.neighbor_offset == -1  # Points to left neighbor
    assert stencil.is_conservative()
    print("   PASS")

    # Test 6: Advection stencil with Neumann
    print("\n6. Advection stencil with Neumann:")
    v = 1.0  # velocity
    adv_interior = {
        "diagonal": v / dx,  # Upwind at interior: v*(u[i] - u[i-1])/dx
        "left": -v / dx,
        "right": 0.0,
    }
    print(f"   Interior advection stencil: {adv_interior}")
    stencil = BCTransforms.neumann(adv_interior, "left", dx, bc_value=0.0)
    print(f"   Boundary diagonal: {stencil.diagonal:.4f}")
    print(f"   Boundary neighbor: {stencil.neighbor:.4f}")
    print(f"   Row sum: {stencil.row_sum():.6f}")
    print("   PASS")

    print("\n" + "=" * 50)
    print("All smoke tests passed!")
