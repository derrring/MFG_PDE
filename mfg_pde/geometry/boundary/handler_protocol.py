"""
Boundary Handler Protocol for Solver BC Integration.

Defines a common interface for boundary condition handling across all MFG solvers
(FDM, GFDM, Particle, FEM, Semi-Lagrangian).

Issue #545: Eliminate inconsistent BC handling patterns across solvers.

Design Principles:
1. **Geometry provides context**: Boundary points, normals, segments
2. **Solver applies enforcement**: Ghost nodes, penalties, reflections, interpolation
3. **Uniform workflow**: All solvers follow the same BC retrieval pattern

Typical Usage:
    ```python
    # 1. Retrieve BC from geometry (centralized in solver base class)
    bc = problem.geometry.get_boundary_conditions()

    # 2. Get boundary context from geometry
    boundary_indices = geometry.get_boundary_indices(solver.points)
    normals = geometry.get_normals(boundary_indices)

    # 3. Match points to BC segments
    for idx in boundary_indices:
        for segment in bc.segments:
            if segment.matches_point(solver.points[idx], ...):
                bc_type = segment.bc_type
                bc_value = segment.value

                # 4. Apply BC using solver-specific method
                solver.apply_boundary_condition(idx, bc_type, bc_value)
    ```

Mathematical Context:
    - Dirichlet BC: u(x,t) = g(x,t) on ∂Ω
    - Neumann BC: ∂u/∂n(x,t) = h(x,t) on ∂Ω
    - Periodic BC: u(x_left) = u(x_right)
    - Robin BC: α u + β ∂u/∂n = γ on ∂Ω
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mfg_pde.geometry.boundary.conditions import BoundaryConditions


@runtime_checkable
class BoundaryHandler(Protocol):
    """
    Protocol for solver boundary condition handling.

    This protocol defines the minimal interface that solvers must implement
    to handle boundary conditions in a consistent way.

    Responsibilities:
    - Identify boundary points in solver's discretization
    - Apply BCs using solver-specific enforcement method
    - Provide BC type information for algorithmic decisions

    NOT Responsible for:
    - BC segment matching (handled by BCSegment.matches_point())
    - Boundary normal computation (handled by geometry)
    - BC value interpolation (handled by BoundaryConditions)
    """

    def get_boundary_indices(self) -> NDArray[np.integer]:
        """
        Identify boundary points in solver's discretization.

        Returns:
            Array of integer indices identifying boundary points in the
            solver's point set (grid points, collocation points, particles, etc.).

        Notes:
            - For grid-based methods: Indices of grid boundary
            - For meshfree methods: Indices of collocation points near ∂Ω
            - For particle methods: Indices of particles on/near boundary

        Example:
            ```python
            boundary_idx = solver.get_boundary_indices()
            # array([0, 1, 2, ..., N-3, N-2, N-1])  # Boundary points
            ```
        """
        ...

    def apply_boundary_conditions(
        self,
        values: NDArray,
        bc: BoundaryConditions,
        time: float = 0.0,
    ) -> NDArray:
        """
        Apply boundary conditions to solution values.

        This is the main method that solvers implement to enforce BCs
        using their specific discretization method.

        Args:
            values: Solution values at all discretization points (N,)
            bc: Boundary conditions object from geometry
            time: Current time for time-dependent BCs

        Returns:
            Modified solution values with BCs enforced (N,)

        Enforcement Methods by Solver Type:
            - **FDM**: Ghost cells, one-sided stencils
            - **GFDM**: Ghost nodes, stencil rotation, penalty weights
            - **Particle**: Reflection, resampling
            - **FEM**: Essential BCs (modify system), Natural BCs (weak form)
            - **Semi-Lagrangian**: Interpolation with BC constraints

        Example:
            ```python
            # FDM solver
            u = np.zeros(N)
            u_with_bc = solver.apply_boundary_conditions(u, bc, time=t)
            # u_with_bc[boundary_idx] now satisfies Dirichlet/Neumann conditions
            ```
        """
        ...

    def get_bc_type_for_point(self, point_idx: int) -> str:
        """
        Determine BC type (dirichlet/neumann/periodic) for a specific point.

        Args:
            point_idx: Index of point in solver's discretization

        Returns:
            BC type string: "dirichlet", "neumann", "periodic", or "none"

        Notes:
            - Used for algorithmic decisions (stencil selection, interpolation)
            - Queries geometry BC segments to determine type
            - Returns "none" if point is not on boundary

        Example:
            ```python
            bc_type = solver.get_bc_type_for_point(idx=0)
            if bc_type == "neumann":
                # Use gradient-based stencil
                deriv_stencil = build_neumann_stencil(idx)
            ```
        """
        ...


# Optional extension for advanced BC handling
@runtime_checkable
class AdvancedBoundaryHandler(BoundaryHandler, Protocol):
    """
    Extended protocol for solvers with advanced BC features.

    Adds optional methods for:
    - Normal vector computation
    - Mixed/Robin BC support
    - Time-dependent BC caching
    """

    def get_boundary_normals(self) -> NDArray:
        """
        Compute outward-pointing normal vectors at boundary points.

        Returns:
            Array of normal vectors (n_boundary, dimension)

        Notes:
            - For rectangular domains: ±e_i directions
            - For general domains: Computed from SDF or mesh geometry
            - Used for Neumann BC and coordinate rotation (GFDM)

        Example:
            ```python
            normals = solver.get_boundary_normals()
            # array([[-1, 0], [-1, 0], ..., [1, 0], [1, 0]])  # 2D box
            ```
        """
        ...

    def apply_robin_bc(
        self,
        values: NDArray,
        alpha: float,
        beta: float,
        gamma: float,
        time: float = 0.0,
    ) -> NDArray:
        """
        Apply Robin boundary conditions: α u + β ∂u/∂n = γ.

        Args:
            values: Solution values (N,)
            alpha: Coefficient for u term
            beta: Coefficient for ∂u/∂n term
            gamma: Right-hand side value
            time: Current time

        Returns:
            Modified solution values with Robin BC enforced (N,)

        Notes:
            - Generalization of Dirichlet (β=0) and Neumann (α=0)
            - Requires both value and gradient enforcement
            - Not all solvers support Robin BCs efficiently
        """
        ...


def validate_boundary_handler(solver) -> bool:
    """
    Runtime check if solver implements BoundaryHandler protocol.

    Args:
        solver: Solver instance to validate

    Returns:
        True if solver implements the protocol, False otherwise

    Example:
        ```python
        from mfg_pde.geometry.boundary import validate_boundary_handler

        if validate_boundary_handler(my_solver):
            # Safe to use unified BC workflow
            boundary_idx = my_solver.get_boundary_indices()
        else:
            # Fallback to solver-specific BC handling
            pass
        ```
    """
    return isinstance(solver, BoundaryHandler)
