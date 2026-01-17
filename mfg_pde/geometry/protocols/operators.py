"""
Operator trait protocols for geometry capabilities.

This module defines Protocol classes for geometry operator capabilities.
Geometries implement these protocols to advertise that they can provide
specific differential operators (Laplacian, gradient, divergence, advection).

The operator abstraction pattern allows solvers to be geometry-agnostic:
- Solver requests operators (e.g., Laplacian, gradient)
- Geometry provides operator as LinearOperator or callable
- Solver uses operator without knowing geometry details

Benefits:
- New geometries require no solver changes
- Solvers work across FDM, FEM, GFDM, meshfree methods
- Testable (mock operators for unit tests)
- Mathematical clarity (code mirrors equations)

Example:
    def solve_heat_equation(u0, geometry: SupportsLaplacian, dt, steps):
        laplacian = geometry.get_laplacian_operator()
        u = u0.copy()
        for _ in range(steps):
            u = u + dt * laplacian(u)  # Forward Euler
        return u

Created: 2026-01-17 (Issue #590 - Phase 1.1)
Part of: Geometry & BC Architecture Implementation (Issue #589)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from scipy.sparse.linalg import LinearOperator

    from mfg_pde.geometry.boundary import BoundaryConditions


@runtime_checkable
class SupportsLaplacian(Protocol):
    """
    Geometry can provide Laplacian operator (diffusion).

    The Laplacian operator L satisfies L[u] ≈ ∇²u at interior points,
    with boundary conditions applied appropriately.

    Mathematical Definition:
        L[u] = ∇²u = ∑ᵢ ∂²u/∂xᵢ²

    Discretization-Specific Implementations:
        - TensorProductGrid: Finite difference stencil (-1, 2, -1) / dx²
        - UnstructuredMesh: FEM stiffness matrix assembly
        - GraphGeometry: Graph Laplacian matrix
        - ImplicitDomain: Meshfree RBF or GFDM Laplacian

    Returns:
        LinearOperator or callable that applies Laplacian to a field.
        The operator should handle BC automatically if bc parameter provided.
    """

    def get_laplacian_operator(
        self,
        order: int = 2,
        bc: BoundaryConditions | None = None,
    ) -> LinearOperator | Callable[[NDArray], NDArray]:
        """
        Return discrete Laplacian operator.

        Args:
            order: Discretization order (2, 4, 6, ...). Higher order more accurate
                   but requires wider stencils/more neighbors.
            bc: Boundary conditions to incorporate. If None, uses geometry's
                default BC or assumes periodic/natural BC.

        Returns:
            LinearOperator or callable L where L @ u or L(u) computes ∇²u.

        Raises:
            ValueError: If order not supported by this geometry
            TypeError: If bc type incompatible with geometry method

        Example:
            >>> grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[51, 51])
            >>> laplacian = grid.get_laplacian_operator(order=2, bc=neumann_bc(dimension=2))
            >>> u = np.random.rand(51, 51)
            >>> Lu = laplacian @ u.ravel()  # or Lu = laplacian(u)

        Note:
            - Output shape matches input: (N,) → (N,) for flattened fields
            - For matrix-free operators, use scipy.sparse.linalg.LinearOperator
            - Geometry determines whether to return explicit matrix or matrix-free operator
        """
        ...


@runtime_checkable
class SupportsGradient(Protocol):
    """
    Geometry can compute spatial gradients (advection, Hamiltonian).

    The gradient operator ∇ computes spatial derivatives in all dimensions.

    Mathematical Definition:
        ∇u = (∂u/∂x₁, ∂u/∂x₂, ..., ∂u/∂xₐ)

    Discretization-Specific Implementations:
        - TensorProductGrid: Finite difference stencils (forward, backward, centered)
        - UnstructuredMesh: FEM gradient recovery or direct element gradients
        - ImplicitDomain: SDF-based gradients or meshfree differentiation
        - GraphGeometry: Not available (discrete graph has no continuous gradient)

    Returns:
        LinearOperator or callable that computes gradients.
        Output shape: (dimension, num_points) for vectorized computation.
    """

    def get_gradient_operator(
        self,
        direction: int | None = None,
        order: int = 2,
        scheme: Literal["centered", "forward", "backward", "upwind"] = "centered",
    ) -> LinearOperator | Callable[[NDArray], NDArray] | tuple[LinearOperator | Callable, ...]:
        """
        Return discrete gradient operator.

        Args:
            direction: Specific direction (0=x, 1=y, 2=z). If None, return all directions.
            order: Discretization order (2, 4, 6, ...)
            scheme: Differencing scheme
                - "centered": Centered differences (most accurate, requires interior points)
                - "forward": Forward differences (for boundary points, upwind with v>0)
                - "backward": Backward differences (for boundary points, upwind with v<0)
                - "upwind": Automatic upwind selection (requires velocity field)

        Returns:
            If direction is None:
                tuple of operators/callables (∂/∂x₁, ∂/∂x₂, ..., ∂/∂xₐ)
            If direction specified:
                Single operator/callable for ∂/∂xᵢ

        Raises:
            ValueError: If direction >= geometry.dimension
            ValueError: If scheme not supported
            NotImplementedError: If geometry doesn't support gradients (e.g., GraphGeometry)

        Example:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> grad_x, grad_y = grid.get_gradient_operator()
            >>> u = np.random.rand(Nx, Ny)
            >>> ux = grad_x @ u.ravel()
            >>> uy = grad_y @ u.ravel()
            >>> grad_u = np.stack([ux, uy], axis=0)  # Shape: (2, Nx*Ny)

        Note:
            - Sign convention: ∇u[d, i] = ∂u/∂xₐ at point i
            - For upwind scheme, velocity field must be provided separately or
              use get_advection_operator() which handles upwinding automatically
        """
        ...


@runtime_checkable
class SupportsDivergence(Protocol):
    """
    Geometry can compute divergence operator (mass conservation, incompressibility).

    The divergence operator div computes the sum of partial derivatives.

    Mathematical Definition:
        div(v) = ∑ᵢ ∂vᵢ/∂xᵢ

    For scalar fields, divergence of gradient gives Laplacian:
        div(∇u) = ∇²u

    Discretization-Specific Implementations:
        - TensorProductGrid: Finite difference divergence stencil
        - UnstructuredMesh: FEM divergence from basis functions
        - ImplicitDomain: Meshfree divergence operator

    Returns:
        LinearOperator or callable that computes divergence of vector field.
    """

    def get_divergence_operator(
        self,
        order: int = 2,
    ) -> LinearOperator | Callable[[NDArray], NDArray]:
        """
        Return discrete divergence operator.

        Args:
            order: Discretization order (2, 4, 6, ...)

        Returns:
            Operator that computes div(v) from vector field v.
            Input shape: (dimension, num_points)
            Output shape: (num_points,)

        Example:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> div_op = grid.get_divergence_operator(order=2)
            >>> v = np.random.rand(2, Nx*Ny)  # Vector field (vx, vy)
            >>> div_v = div_op @ v.ravel()  # or div_v = div_op(v)

        Note:
            - For conservation laws: ∂m/∂t + div(J) = 0
            - For vector calculus identity: div(grad(u)) = Laplacian(u)
            - Some geometries may return divergence as adjoint of gradient
        """
        ...


@runtime_checkable
class SupportsAdvection(Protocol):
    """
    Geometry can provide advection operator (transport, Fokker-Planck).

    The advection operator computes the transport term v·∇u or div(u·v) depending
    on formulation (conservative vs non-conservative).

    Mathematical Forms:
        Non-conservative: v·∇u = ∑ᵢ vᵢ ∂u/∂xᵢ
        Conservative: div(u·v) = u·div(v) + v·∇u

    For Fokker-Planck equation:
        ∂m/∂t + div(m·α) - σ²/2·∇²m = 0

    The advection term div(m·α) requires conservative formulation.

    Discretization-Specific Implementations:
        - TensorProductGrid: Upwind finite differences (1st-order, WENO, etc.)
        - UnstructuredMesh: FEM advection matrix with stabilization (SUPG)
        - ImplicitDomain: Meshfree advection with upwinding

    Returns:
        LinearOperator or callable that applies advection to a field.
    """

    def get_advection_operator(
        self,
        velocity_field: NDArray,
        scheme: Literal["upwind", "centered", "weno", "lax_friedrichs"] = "upwind",
        conservative: bool = True,
    ) -> LinearOperator | Callable[[NDArray], NDArray]:
        """
        Return discrete advection operator for given velocity field.

        Args:
            velocity_field: Velocity/drift field (dimension, num_points) or (num_points,) for 1D.
                            Determines direction and magnitude of transport.
            scheme: Advection scheme
                - "upwind": 1st-order upwind (stable, dissipative)
                - "centered": Centered differences (2nd-order, may oscillate)
                - "weno": Weighted ENO (high-order, for shocks)
                - "lax_friedrichs": Lax-Friedrichs flux (stable, dissipative)
            conservative: If True, compute div(u·v). If False, compute v·∇u.
                          For mass conservation (FP equation), use conservative=True.

        Returns:
            Operator A where A @ u or A(u) computes advection term.
            For conservative form: div(u·v)
            For non-conservative form: v·∇u

        Raises:
            ValueError: If velocity_field shape incompatible with geometry
            ValueError: If scheme not supported

        Example:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> alpha = np.random.rand(2, Nx*Ny)  # Drift field
            >>> adv_op = grid.get_advection_operator(alpha, scheme='upwind', conservative=True)
            >>> m = np.random.rand(Nx*Ny)  # Density
            >>> div_m_alpha = adv_op @ m  # Conservative advection

        Note:
            - For FP equation, always use conservative=True to preserve mass
            - Upwind scheme selects direction based on velocity sign (automatic stabilization)
            - WENO scheme requires wider stencils (ghost depth ≥ 2)
        """
        ...


@runtime_checkable
class SupportsInterpolation(Protocol):
    """
    Geometry can interpolate field values at arbitrary points.

    Interpolation is needed for:
    - Semi-Lagrangian methods (characteristic foot point evaluation)
    - Particle-in-cell methods (field to particle mapping)
    - Post-processing (visualization at non-grid points)

    Discretization-Specific Implementations:
        - TensorProductGrid: Multi-linear or high-order tensor product interpolation
        - UnstructuredMesh: Barycentric interpolation within elements
        - ImplicitDomain: RBF interpolation or natural neighbor interpolation
        - GraphGeometry: Not available (discrete points only)

    Returns:
        Callable that interpolates field to query points.
    """

    def get_interpolation_operator(
        self,
        query_points: NDArray,
        order: int = 1,
        extrapolation_mode: Literal["constant", "nearest", "boundary"] = "boundary",
    ) -> Callable[[NDArray], NDArray]:
        """
        Return interpolation operator for given query points.

        Args:
            query_points: Points at which to interpolate, shape (num_query, dimension)
            order: Interpolation order
                - 1: Linear (TensorProductGrid: trilinear, UnstructuredMesh: barycentric)
                - 2: Quadratic
                - 3: Cubic
            extrapolation_mode: How to handle points outside domain
                - "constant": Use constant value (e.g., 0)
                - "nearest": Use nearest boundary value
                - "boundary": Project to boundary and use boundary value

        Returns:
            Callable I where I(u) returns interpolated values at query_points.
            Input: u of shape (num_grid_points,)
            Output: interpolated values of shape (num_query,)

        Raises:
            ValueError: If query_points outside domain and extrapolation not allowed
            NotImplementedError: If order not supported by this geometry

        Example:
            >>> grid = TensorProductGrid(dimension=2, ...)
            >>> # Characteristic foot points from semi-Lagrangian
            >>> foot_points = grid_points - dt * velocity
            >>> interp = grid.get_interpolation_operator(foot_points, order=1)
            >>> u_foot = interp(u)  # Evaluate u at foot points

        Note:
            - For semi-Lagrangian: query_points are characteristic foot points
            - For particle methods: query_points are particle positions
            - Interpolation operator is linear in field values
        """
        ...
