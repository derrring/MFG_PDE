from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import lstsq

# BC types for BoundaryCapable protocol implementation (Issue #527)
from scipy.optimize import approx_fprime

from mfg_pde.geometry.boundary import BCType, DiscretizationType

# Legacy operator for backward compatibility (deprecated)
from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator

# New GFDM infrastructure (Strategy Pattern)
from mfg_pde.utils.numerical.gfdm_strategies import (
    DirectCollocationHandler,
    TaylorOperator,
    create_operator,
)
from mfg_pde.utils.numerical.qp_utils import QPCache, QPSolver

from .base_hjb import BaseHJBSolver
from .gfdm_boundary_mixin import GFDMBoundaryMixin
from .gfdm_interpolation_mixin import GFDMInterpolationMixin
from .gfdm_stencil_mixin import GFDMStencilMixin
from .hjb_gfdm_monotonicity import MonotonicityMixin

# Optional QP solver imports
CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None
OSQP_AVAILABLE = importlib.util.find_spec("osqp") is not None

if TYPE_CHECKING:
    from mfg_pde.core.derivatives import DerivativeTensors
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions


class HJBGFDMSolver(GFDMInterpolationMixin, GFDMStencilMixin, GFDMBoundaryMixin, MonotonicityMixin, BaseHJBSolver):
    """
    Generalized Finite Difference Method (GFDM) solver for HJB equations using collocation.

    This solver implements meshfree collocation for HJB equations using:
    1. δ-neighborhood search for local support
    2. Taylor expansion with weighted least squares for derivative approximation
    3. Newton iteration for nonlinear HJB equations
    4. Support for various boundary conditions
    5. Optional QP constraints for monotonicity preservation

    QP Optimization Levels:
    - "none": GFDM without QP constraints
    - "auto": Adaptive QP with M-matrix checking for monotonicity preservation
    - "always": Force QP at every point (for debugging and analysis)

    Note: Monotonicity and QP constraint functionality is provided by MonotonicityMixin.

    Implements BoundaryCapable protocol for unified BC handling (Issue #527).

    Collocation Point Strategies (Issue #529):
        Use FIXED collocation points throughout the MFG solve. Moving points
        during iteration causes convergence stall due to interpolation noise
        and stencil weight fluctuations.

        IMPORTANT: Fully Lagrangian MFG (moving collocation with the flow)
        is MATHEMATICALLY INVALID because the optimal control alpha* = -grad(u)
        requires grad(u) at FIXED spatial locations.

        See ``docs/theory/adaptive_collocation_analysis.md`` for detailed analysis
        of three collocation strategies and why only fixed collocation is valid.
    """

    # BoundaryCapable protocol: Supported BC types
    _SUPPORTED_BC_TYPES: frozenset = frozenset(
        {
            BCType.DIRICHLET,
            BCType.NEUMANN,
            BCType.NO_FLUX,  # Same as Neumann with g=0
        }
    )

    @property
    def supported_bc_types(self) -> frozenset:
        """BC types this solver supports (BoundaryCapable protocol)."""
        return self._SUPPORTED_BC_TYPES

    @property
    def discretization_type(self) -> DiscretizationType:
        """Discretization method (BoundaryCapable protocol)."""
        return DiscretizationType.GFDM

    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        weight_scale: float = 1.0,
        max_newton_iterations: int | None = None,
        newton_tolerance: float | None = None,
        # Deprecated parameters for backward compatibility
        NiterNewton: int | None = None,
        l2errBoundNewton: float | None = None,
        boundary_indices: np.ndarray | None = None,
        boundary_conditions: dict | BoundaryConditions | None = None,
        # QP optimization level
        qp_optimization_level: str = "none",  # "none", "auto", or "always"
        qp_usage_target: float = 0.1,  # Unused, kept for backward compatibility
        qp_solver: str = "osqp",  # "osqp" or "scipy"
        qp_warm_start: bool = True,  # Enable QP warm-starting
        qp_constraint_mode: str = "indirect",  # "indirect" or "hamiltonian"
        # Adaptive neighborhood parameters
        adaptive_neighborhoods: bool = False,
        k_min: int | None = None,
        max_delta_multiplier: float = 5.0,
        # Hybrid neighborhood parameters
        k_neighbors: int | None = None,
        neighborhood_mode: str = "hybrid",
        # New GFDM infrastructure parameters
        derivative_method: str = "taylor",  # "taylor" or "rbf"
        rbf_kernel: str = "phs3",  # For RBF-FD: "phs3", "phs5", "gaussian"
        rbf_poly_degree: int = 2,  # Polynomial augmentation degree for RBF-FD
        use_new_infrastructure: bool = True,  # Use new Strategy Pattern (recommended)
        # Local Coordinate Rotation for boundary accuracy (Issue #531)
        use_local_coordinate_rotation: bool = False,
        # Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        use_ghost_nodes: bool = False,
        # Wind-Dependent BC for viscosity solution compatibility
        use_wind_dependent_bc: bool = False,
    ):
        """
        Initialize the GFDM HJB solver.

        Args:
            problem: MFG problem instance
            collocation_points: (N_points, d) array of collocation points
            delta: Neighborhood radius for collocation
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("wendland", "cubic_spline", "gaussian", "inverse_distance", "uniform")
            weight_scale: Scale parameter for weight function
            max_newton_iterations: Maximum Newton iterations (new parameter name)
            newton_tolerance: Newton convergence tolerance (new parameter name)
            NiterNewton: DEPRECATED - use max_newton_iterations
            l2errBoundNewton: DEPRECATED - use newton_tolerance
            boundary_indices: Indices of boundary collocation points
            boundary_conditions: Dictionary or BoundaryConditions object specifying boundary conditions
            use_monotone_constraints: DEPRECATED - Use qp_optimization_level instead.
                If explicitly set to True, will override qp_optimization_level.
            qp_optimization_level: QP optimization level (controls QP behavior):
                - "none": No QP constraints
                - "auto": Adaptive QP with M-matrix checking (recommended)
                - "always": Force QP at every point (debugging/analysis)
            qp_usage_target: Deprecated parameter, kept for backward compatibility
            qp_solver: QP solver backend (default "osqp"):
                - "osqp": Use OSQP solver (fast convex QP, 5-10× faster than scipy)
                - "scipy": Use scipy.optimize.minimize (SLSQP or L-BFGS-B)
            qp_warm_start: Enable warm-starting for QP solves (default True).
                When True, uses previous QP solution as initial guess for next solve.
                Provides 2-3× additional speedup for OSQP on similar QP problems.
                Only applies to OSQP solver (scipy does not support efficient warm-starting).
            qp_constraint_mode: Type of monotonicity constraints (default "indirect"):
                - "indirect": Constraints on Taylor coefficients (simpler, approximate)
                - "hamiltonian": Direct Hamiltonian gradient constraints dH/du_j >= 0
                  (stricter, better monotonicity guarantees, requires gamma parameter)
            adaptive_neighborhoods: Enable adaptive delta enlargement to guarantee well-posed problems.
                When enabled, points with insufficient neighbors get locally enlarged delta.
                Maintains theoretical soundness while ensuring practical robustness.
                Recommended for irregular particle distributions.
            k_min: Minimum number of neighbors required per point (auto-computed from taylor_order if None).
                For Taylor order p in d dimensions, need C(d+p, p) - 1 derivatives.
            max_delta_multiplier: Maximum allowed delta enlargement factor (default 5.0, conservative).
                Limits delta growth to preserve GFDM locality. For very irregular distributions,
                consider increasing to 10.0 (achieves 98%+ success) or increasing base delta instead.
                Trade-off: Smaller limit = better theory, larger limit = better robustness.
            k_neighbors: Number of neighbors for neighborhood selection (auto-computed if None).
                When None, computed from Taylor order to ensure well-posed least squares.
            neighborhood_mode: Neighborhood selection strategy:
                - "radius": Use all points within delta (classic behavior)
                - "knn": Use exactly k nearest neighbors
                - "hybrid": Use delta, but ensure at least k neighbors (default, most robust)
            derivative_method: Method for computing spatial derivatives:
                - "taylor": Standard GFDM with Taylor polynomial basis (default)
                - "rbf": RBF-FD with polyharmonic splines (better conditioning)
            rbf_kernel: Kernel for RBF-FD method (only used when derivative_method="rbf"):
                - "phs3": r³ polyharmonic spline (most common)
                - "phs5": r⁵ polyharmonic spline (higher accuracy)
                - "gaussian": Gaussian RBF (requires shape parameter tuning)
            rbf_poly_degree: Polynomial augmentation degree for RBF-FD (default 2)
            use_new_infrastructure: Use new Strategy Pattern infrastructure (default True).
                When True, uses TaylorOperator/LocalRBFOperator + DirectCollocationHandler.
                When False, uses legacy GFDMOperator (deprecated, for backward compatibility).
            use_local_coordinate_rotation: Enable Local Coordinate Rotation (LCR) for
                boundary stencils (default False, Issue #531). When True, rotates
                neighbor offsets at boundary points to align with the boundary normal,
                improving numerical conditioning for normal derivative computation.
                Recommended for domains with complex boundaries or when boundary
                stencils show poor conditioning. Only affects boundary points.
            use_ghost_nodes: Enable Ghost Nodes method for Neumann boundary conditions
                (default False, Issue #531 - Terminal BC compatibility). When True,
                creates mirrored "ghost" neighbors outside the domain for boundary points,
                enforcing ∂u/∂n = 0 structurally through symmetric stencils rather than
                via row replacement. This eliminates terminal cost/BC incompatibility issues
                in MFG problems. Recommended when terminal cost violates Neumann BC
                (e.g., g(x) = ||x - x_exit||² with Neumann BC at walls). Mutually exclusive
                with use_local_coordinate_rotation (ghost nodes take precedence).
            use_wind_dependent_bc: Enable wind-dependent boundary conditions (default False).
                When True (requires use_ghost_nodes=True), ghost nodes are only enforced
                when characteristics flow INTO the boundary (∇u·n > 0). When flow is OUT
                (∇u·n < 0), uses extrapolation instead. This implements the viscosity solution
                approach where BCs are weak constraints, only enforced when the PDE solution
                "wants" to violate them. Recommended for evacuation/exit problems where agents
                need to cross boundaries. Based on Lions & Souganidis theory of discontinuous
                viscosity solutions.
        """
        super().__init__(problem)

        # Handle deprecated QP level names
        if qp_optimization_level in ["smart", "tuned", "basic"]:
            import warnings

            warnings.warn(
                f"qp_optimization_level='{qp_optimization_level}' is deprecated. Use 'auto' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            qp_optimization_level = "auto"

        # Store QP optimization level
        self.qp_optimization_level = qp_optimization_level

        # Set method name based on QP optimization level
        if qp_optimization_level == "none":
            self.hjb_method_name = "GFDM"
        elif qp_optimization_level == "auto":
            self.hjb_method_name = "GFDM-QP"
        elif qp_optimization_level == "always":
            self.hjb_method_name = "GFDM-QP-Always"
        else:
            self.hjb_method_name = f"GFDM-{qp_optimization_level}"

        import warnings

        # Handle backward compatibility
        if NiterNewton is not None:
            warnings.warn(
                "Parameter 'NiterNewton' is deprecated. Use 'max_newton_iterations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if max_newton_iterations is None:
                max_newton_iterations = NiterNewton

        if l2errBoundNewton is not None:
            warnings.warn(
                "Parameter 'l2errBoundNewton' is deprecated. Use 'newton_tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if newton_tolerance is None:
                newton_tolerance = l2errBoundNewton

        # Set defaults if still None
        if max_newton_iterations is None:
            max_newton_iterations = 30
        if newton_tolerance is None:
            newton_tolerance = 1e-6

        # Collocation parameters
        self.collocation_points = collocation_points
        self.n_points = collocation_points.shape[0]
        self.dimension = collocation_points.shape[1]
        self.delta = delta
        self.taylor_order = taylor_order
        self.weight_function = weight_function
        self.weight_scale = weight_scale

        # Newton parameters (store with new names)
        self.max_newton_iterations = max_newton_iterations
        self.newton_tolerance = newton_tolerance

        # Keep old names for backward compatibility (without warnings when accessed)
        self.NiterNewton = max_newton_iterations
        self.l2errBoundNewton = newton_tolerance

        # Boundary condition parameters
        self.boundary_indices = boundary_indices if boundary_indices is not None else np.array([])
        # Get BC from parameter, or from problem if not provided
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        elif hasattr(self.problem, "get_boundary_conditions"):
            self.boundary_conditions = self.problem.get_boundary_conditions() or {}
        else:
            self.boundary_conditions = {}
        self.interior_indices = np.setdiff1d(np.arange(self.n_points), self.boundary_indices)

        # QP optimization level (single source of truth for QP control)
        self.qp_optimization_level = qp_optimization_level

        # QP usage target (deprecated, kept for backward compatibility)
        self.qp_usage_target = qp_usage_target

        # QP solver selection
        self.qp_solver = qp_solver
        self.qp_warm_start = qp_warm_start
        self.qp_constraint_mode = qp_constraint_mode

        # Initialize unified QP solver from qp_utils
        # Map qp_solver parameter to QPSolver backend
        # Use "auto" to allow fallback to scipy when OSQP not installed
        qp_backend = "auto" if qp_solver == "osqp" else "scipy-slsqp"
        self._qp_cache = QPCache(max_size=1000)
        self._qp_solver_instance = QPSolver(
            backend=qp_backend,
            enable_warm_start=qp_warm_start,
            cache=self._qp_cache,
        )

        # Legacy warm-start cache (kept for backward compatibility, but unused)
        self._qp_warm_start_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

        # Initialize QP diagnostic statistics (extended from QPSolver.stats)
        # These are GFDM-specific stats not covered by QPSolver
        self.qp_stats = {
            "total_qp_solves": 0,
            "qp_times": [],
            "violations_detected": 0,
            "points_checked": 0,
            "qp_successes": 0,
            "qp_failures": 0,
            "qp_fallbacks": 0,
            "slsqp_solves": 0,
            "lbfgsb_solves": 0,
            "osqp_solves": 0,
            "osqp_failures": 0,
        }
        self._current_point_idx = 0

        # Adaptive neighborhood parameters
        self.adaptive_neighborhoods = adaptive_neighborhoods
        self.max_delta_multiplier = max_delta_multiplier

        # Cache grid size info from geometry
        self._n_spatial_grid_points = self._compute_n_spatial_grid_points()

        # Cache domain bounds from geometry
        self.domain_bounds = self._get_domain_bounds()

        # Compute k_min from Taylor order if not provided
        from math import comb

        n_derivatives_required = comb(self.dimension + taylor_order, taylor_order) - 1
        if k_min is None:
            self.k_min = n_derivatives_required
        else:
            # Ensure k_min is at least what's required for Taylor expansion
            if k_min < n_derivatives_required:
                import warnings

                warnings.warn(
                    f"k_min={k_min} is less than required for Taylor order {taylor_order} "
                    f"in {self.dimension}D (need {n_derivatives_required}). "
                    f"Using k_min={n_derivatives_required} instead.",
                    UserWarning,
                    stacklevel=2,
                )
                self.k_min = n_derivatives_required
            else:
                self.k_min = k_min

        # Initialize adaptive neighborhood statistics
        self.adaptive_stats = {
            "n_adapted": 0,
            "adaptive_enlargements": [],
            "max_delta_used": delta,
        }

        # Store new infrastructure parameters
        self._use_new_infrastructure = use_new_infrastructure
        self._derivative_method = derivative_method
        self._rbf_kernel = rbf_kernel
        self._rbf_poly_degree = rbf_poly_degree

        # Local Coordinate Rotation for boundary accuracy (Issue #531)
        self._use_local_coordinate_rotation = use_local_coordinate_rotation

        # Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        self._use_ghost_nodes = use_ghost_nodes

        # Wind-Dependent BC for viscosity solution compatibility
        self._use_wind_dependent_bc = use_wind_dependent_bc

        # Hyperviscosity parameter for wind-dependent BC stabilization
        # epsilon > 0 adds damping: u_ghost = 2u_b - u_m - epsilon*(u_b - u_m)
        # Recommended: 0.0 (no damping) to 0.3 (moderate damping)
        self._wind_bc_hyperviscosity = 0.0  # Default: no hyperviscosity

        # Check for mutual exclusivity (ghost nodes takes precedence)
        if self._use_ghost_nodes and self._use_local_coordinate_rotation:
            import warnings

            warnings.warn(
                "Both use_ghost_nodes and use_local_coordinate_rotation are enabled. "
                "Ghost nodes take precedence and LCR will be disabled for boundary points.",
                UserWarning,
                stacklevel=2,
            )

        # Wind-dependent BC requires ghost nodes
        if self._use_wind_dependent_bc and not self._use_ghost_nodes:
            raise ValueError(
                "use_wind_dependent_bc=True requires use_ghost_nodes=True. "
                "Wind-dependent BC is a modification of the ghost nodes method."
            )

        # DEBUG: Print wind-BC configuration once at initialization
        if self._use_wind_dependent_bc:
            import sys

            print(f"\n[Wind-BC INIT] Enabled with {len(boundary_indices)} boundary points", flush=True, file=sys.stderr)

        # Create differential operator using Strategy Pattern (recommended)
        # or legacy GFDMOperator (for backward compatibility)
        if use_new_infrastructure:
            # New infrastructure: TaylorOperator or LocalRBFOperator
            if derivative_method == "taylor":
                self._gfdm_operator = TaylorOperator(
                    points=collocation_points,
                    delta=delta,
                    taylor_order=taylor_order,
                    weight_function=weight_function,
                    k_neighbors=k_neighbors,
                    neighborhood_mode=neighborhood_mode,
                )
            elif derivative_method == "rbf":
                self._gfdm_operator = create_operator(
                    points=collocation_points,
                    delta=delta,
                    method="rbf",
                    kernel=rbf_kernel,
                    poly_degree=rbf_poly_degree,
                    k_neighbors=k_neighbors,
                    neighborhood_mode=neighborhood_mode,
                )
            else:
                raise ValueError(f"Unknown derivative_method: {derivative_method}")

            # Initialize BC handler with Row Replacement pattern
            self._bc_handler = DirectCollocationHandler()

            # Compute boundary normals for Neumann BC
            self._boundary_normals = self._compute_boundary_normals()

            # Create unified BC config (single source of truth)
            self._bc_config = self._create_bc_config()
        else:
            # Legacy: GFDMOperator (deprecated, for backward compatibility)
            warnings.warn(
                "use_new_infrastructure=False is deprecated. "
                "The legacy GFDMOperator will be removed in v0.18.0. "
                "Use use_new_infrastructure=True (default) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._gfdm_operator = GFDMOperator(
                points=collocation_points,
                delta=delta,
                taylor_order=taylor_order,
                weight_function=weight_function,
                weight_scale=weight_scale,
                boundary_indices=self.boundary_indices,
                domain_bounds=self.domain_bounds,
                boundary_type=None,  # No ghost particles
                k_neighbors=k_neighbors,
                neighborhood_mode=neighborhood_mode,
            )
            self._bc_handler = None
            self._boundary_normals = None
            self._bc_config = None

        # Get multi-indices from operator (both TaylorOperator and GFDMOperator have .multi_indices)
        self.multi_indices = self._gfdm_operator.multi_indices
        self.n_derivatives = len(self.multi_indices)

        # Store spatial shape for grid<->collocation interpolation
        # This is needed for _map_grid_to_collocation and _map_collocation_to_grid
        # get_grid_shape() returns node counts (Nx+1, Ny+1), not cell counts
        self._output_spatial_shape = tuple(self.problem.geometry.get_grid_shape())

        # Build neighborhood structure - uses GFDMOperator's neighborhoods as base,
        # only extends for points needing adaptive delta enlargement
        self._build_neighborhood_structure()

        # Apply Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        # Ghost nodes take precedence over LCR if both are enabled
        # This must be called BEFORE Taylor matrices are built, since it augments neighborhoods
        if self._use_ghost_nodes:
            self._apply_ghost_nodes_to_neighborhoods()
        elif self._use_local_coordinate_rotation:
            # Apply Local Coordinate Rotation for boundary stencils (Issue #531)
            # This modifies neighborhoods by adding rotated_offsets for better normal derivatives
            self._apply_local_coordinate_rotation()

        # Build reverse neighborhood map for sparse Jacobian (point j -> rows affected)
        self._build_reverse_neighborhoods()

        # Build Taylor matrices for extended neighborhoods
        self._build_taylor_matrices()

    def _compute_n_spatial_grid_points(self) -> int:
        """Compute total number of spatial grid points from geometry."""
        grid_shape = self.problem.geometry.get_grid_shape()
        return int(np.prod(grid_shape))

    def _get_boundary_condition_property(self, property_name: str, default: Any = None) -> Any:
        """Helper method to get boundary condition properties from either dict or dataclass."""
        if hasattr(self.boundary_conditions, property_name):
            # BoundaryConditions dataclass
            return getattr(self.boundary_conditions, property_name)
        elif isinstance(self.boundary_conditions, dict):
            # Dictionary format
            return self.boundary_conditions.get(property_name, default)
        else:
            return default

    def _get_domain_bounds(self) -> list[tuple[float, float]]:
        """Get domain bounds from geometry or legacy xmin/xmax attributes.

        Returns:
            List of (min, max) tuples for each dimension.
        """
        # Prefer geometry (modern interface)
        if hasattr(self.problem, "geometry") and self.problem.geometry is not None:
            geom = self.problem.geometry
            # Use get_bounds() method which returns (min_coords, max_coords) arrays
            if hasattr(geom, "get_bounds"):
                bounds_result = geom.get_bounds()
                if bounds_result is not None:
                    min_coords, max_coords = bounds_result
                    return [(float(min_coords[d]), float(max_coords[d])) for d in range(len(min_coords))]
            # Fallback to .bounds property if available
            if hasattr(geom, "bounds"):
                return list(geom.bounds)

        # Fallback to legacy 1D xmin/xmax
        xmin = getattr(self.problem, "xmin", None)
        xmax = getattr(self.problem, "xmax", None)
        if xmin is not None and xmax is not None:
            return [(float(xmin), float(xmax))]

        # Last resort: infer from collocation points (vectorized)
        mins = self.collocation_points.min(axis=0)  # shape: (d,)
        maxs = self.collocation_points.max(axis=0)  # shape: (d,)
        return list(zip(mins.astype(float).tolist(), maxs.astype(float).tolist(), strict=True))

    # =========================================================================
    # Boundary Methods: Provided by GFDMBoundaryMixin (gfdm_boundary_mixin.py)
    # _compute_outward_normal, _compute_boundary_normals, _create_bc_config
    # _build_neumann_bc_weights, _count_deep_neighbors, _build_rotation_matrix
    # _apply_local_coordinate_rotation, _rotate_derivatives_back
    # _create_ghost_neighbors, _apply_ghost_nodes_to_neighborhoods
    # _get_values_with_ghosts
    # =========================================================================

    def _compute_gradient_at_point(self, u_values: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Compute gradient ∇u at a single point using GFDM weights.

        Args:
            u_values: Solution vector at all collocation points
            point_idx: Index of point where gradient is computed

        Returns:
            Gradient vector ∇u, shape (dimension,)
        """
        # Get neighborhood for this point
        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]

        # Get Taylor weights for derivatives
        weights = neighborhood["weights"]

        # Extract gradient weights (first-order derivatives: columns 1 to dimension+1)
        # weights structure: [u, u_x, u_y, u_xx, u_xy, u_yy, ...] for 2D
        grad_weights = weights[:, 1 : 1 + self.dimension]  # Shape: (n_neighbors, dimension)

        # Get neighbor values (using standard ghost mirroring, no wind-dependent check)
        # This avoids circular dependency: we need gradient to check wind direction,
        # but we can't check wind direction while computing the gradient!
        # Solution: use standard ghosts here, wind-dependent BC applies at derivative computation
        if self._use_ghost_nodes and hasattr(self, "_ghost_node_map"):
            # Standard ghost mirroring
            ghost_to_mirror = {}
            for ghost_info in self._ghost_node_map.values():
                ghost_to_mirror.update(ghost_info["ghost_to_mirror"])

            u_neighbors_list = []
            for idx in neighbor_indices:
                if idx < 0:
                    mirror_idx = ghost_to_mirror.get(int(idx))
                    u_neighbors_list.append(u_values[mirror_idx] if mirror_idx is not None else 0.0)
                else:
                    u_neighbors_list.append(u_values[int(idx)])
            u_neighbors = np.array(u_neighbors_list)
        else:
            u_neighbors = u_values[neighbor_indices]

        # Compute gradient: ∇u = Σ w_i * u_i for each component
        grad_u = grad_weights.T @ u_neighbors  # Shape: (dimension,)

        return grad_u

    def _build_differentiation_matrices(self) -> None:
        """
        Pre-compute sparse differentiation matrices for vectorized derivative computation.

        Builds:
        - D_grad: List of sparse matrices (n_points x n_points) for each gradient component
        - D_lap: Sparse matrix (n_points x n_points) for Laplacian

        After this, derivatives can be computed via matrix-vector multiplication:
            grad_u[d] = D_grad[d] @ u
            lap_u = D_lap @ u

        This converts O(n * k^2) per-point computation to O(n * k) matrix multiplication.
        """
        from scipy.sparse import lil_matrix

        n = self.n_points
        d = self.dimension

        # Initialize sparse matrices in LIL format (efficient for construction)
        D_grad_lil = [lil_matrix((n, n)) for _ in range(d)]
        D_lap_lil = lil_matrix((n, n))

        # Pre-compute LCR boundary points set for fast lookup
        lcr_boundary_set = set()
        if self._use_local_coordinate_rotation and hasattr(self, "_boundary_rotations"):
            lcr_boundary_set = set(self._boundary_rotations.keys())

        for i in range(n):
            # For LCR boundary points, use our Taylor matrices with rotation
            if i in lcr_boundary_set:
                weights = self._compute_derivative_weights_from_taylor(i)
            else:
                weights = self._gfdm_operator.get_derivative_weights(i)

            if weights is None:
                continue

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]  # shape: (d, n_neighbors)
            lap_weights = weights["lap_weights"]  # shape: (n_neighbors,)

            # Fill gradient matrices
            for dim in range(d):
                # Neighbor contributions (skip ghost particles with j < 0)
                real_grad_sum = 0.0
                for k, j in enumerate(neighbor_indices):
                    if j >= 0:
                        D_grad_lil[dim][i, j] = grad_weights[dim, k]
                        real_grad_sum += grad_weights[dim, k]
                # Center contribution (sum rule: center weight = -sum of REAL neighbor weights)
                # Note: Must exclude ghost particle weights to maintain row sum = 0
                center_weight = -real_grad_sum
                D_grad_lil[dim][i, i] += center_weight

            # Fill Laplacian matrix (same fix: exclude ghost weights from center)
            real_lap_sum = 0.0
            for k, j in enumerate(neighbor_indices):
                if j >= 0:
                    D_lap_lil[i, j] = lap_weights[k]
                    real_lap_sum += lap_weights[k]
            D_lap_lil[i, i] += -real_lap_sum

        # Convert to CSR format for efficient matrix-vector multiplication
        self._D_grad = [D.tocsr() for D in D_grad_lil]
        self._D_lap = D_lap_lil.tocsr()

    def _compute_derivatives_vectorized(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients and Laplacian for all points via sparse matrix multiplication.

        Args:
            u: Function values at collocation points, shape (n_points,)

        Returns:
            grad_u: Gradient at all points, shape (n_points, dimension)
            lap_u: Laplacian at all points, shape (n_points,)
        """
        if not hasattr(self, "_D_grad") or self._D_grad is None:
            self._build_differentiation_matrices()

        grad_u = np.column_stack([D @ u for D in self._D_grad])
        lap_u = self._D_lap @ u

        # Note: LCR rotation is now applied in _compute_derivative_weights_from_taylor()
        # so gradients are already in the original coordinate frame

        return grad_u, lap_u

    def approximate_derivatives(self, u_values: np.ndarray, point_idx: int) -> dict[tuple[int, ...], float]:
        """
        Approximate derivatives at collocation point using weighted least squares.

        Args:
            u_values: Function values at collocation points
            point_idx: Index of the collocation point

        Returns:
            Dictionary mapping derivative multi-indices to approximated values
        """
        # Track current point for debugging/statistics
        self._current_point_idx = point_idx

        # Get QP level and check for ghost particles
        qp_level = getattr(self, "qp_optimization_level", "none")
        neighborhood = self.neighborhoods[point_idx]
        has_ghost = neighborhood.get("has_ghost", False)

        # Fast path: delegate to GFDMOperator when no ghost particles and no QP needed
        if qp_level == "none" and not has_ghost:
            return self._gfdm_operator.approximate_derivatives_at_point(u_values, point_idx)

        # Slow path: handle ghost particles and/or QP constraints
        if self.taylor_matrices[point_idx] is None:
            return {}

        taylor_data = self.taylor_matrices[point_idx]

        # Extract function values at neighborhood points, handling ghost nodes/particles
        neighbor_indices = neighborhood["indices"]
        u_center = u_values[point_idx]

        # Use ghost-aware value retrieval if ghost nodes method is active
        if self._use_ghost_nodes:
            u_neighbors = self._get_values_with_ghosts(u_values, neighbor_indices, point_idx=point_idx)
        else:
            # Handle legacy ghost particles based on BC type
            # - Neumann/no-flux: u_ghost = u_center (mirror value)
            # - Dirichlet: u_ghost = BC value (if available)
            bc_type = self._get_boundary_condition_property("type", "neumann")
            bc_values = self._get_boundary_condition_property("values", None)

            u_neighbors = []
            for idx in neighbor_indices:  # type: ignore[attr-defined]
                if idx >= 0:
                    # Regular neighbor
                    u_neighbors.append(u_values[idx])
                else:
                    # Legacy ghost particle: value depends on BC type
                    if bc_type == "dirichlet" and bc_values is not None:
                        # Dirichlet BC: use prescribed value
                        # Note: bc_values may be scalar, array, or callable
                        if callable(bc_values):
                            x_pos = self.collocation_points[point_idx]
                            u_neighbors.append(bc_values(x_pos))
                        elif hasattr(bc_values, "__getitem__"):
                            # Array-like: use value at this point
                            u_neighbors.append(bc_values[point_idx] if point_idx < len(bc_values) else 0.0)
                        else:
                            # Scalar
                            u_neighbors.append(float(bc_values))
                    else:
                        # Neumann/no-flux: mirror u_center
                        u_neighbors.append(u_center)

        u_neighbors = np.array(u_neighbors)  # type: ignore[assignment]

        # Right-hand side: u(x_neighbor) - u(x_center) for Taylor expansion
        # u(x_j) - u(x_0) ≈ ∇u·(x_j - x_0) where A matrix uses (x_j - x_0)
        # For ghost particles: u_ghost = u_center → b = 0, enforcing ∂u/∂n = 0
        b = u_neighbors - u_center

        if qp_level == "always":
            # "always" level: Force QP at every point without checking M-matrix
            derivative_coeffs = self._solve_monotone_constrained_qp(taylor_data, b, point_idx)  # type: ignore[arg-type]
        elif qp_level == "auto":
            # "auto" level: Adaptive QP with M-matrix checking
            # First try unconstrained solution to check if constraints are needed
            unconstrained_coeffs = self._solve_unconstrained_fallback(taylor_data, b)  # type: ignore[arg-type]

            # Check if unconstrained solution violates monotonicity (M-matrix property)
            self.qp_stats["points_checked"] += 1
            needs_constraints = self._check_monotonicity_violation(unconstrained_coeffs, point_idx)

            if needs_constraints:
                # Apply constrained QP to enforce monotonicity
                self.qp_stats["violations_detected"] += 1
                derivative_coeffs = self._solve_monotone_constrained_qp(taylor_data, b, point_idx)  # type: ignore[arg-type]
            else:
                # Use faster unconstrained solution
                derivative_coeffs = unconstrained_coeffs
        elif taylor_data.get("use_svd", False):  # type: ignore[attr-defined]
            # Use SVD: solve using pseudoinverse with truncated SVD
            sqrt_W = taylor_data["sqrt_W"]
            U = taylor_data["U"]
            S = taylor_data["S"]
            Vt = taylor_data["Vt"]

            # Compute sqrt(W) @ b
            Wb = sqrt_W @ b

            # SVD solution: x = V @ S^{-1} @ U^T @ Wb
            UT_Wb = U.T @ Wb
            S_inv_UT_Wb = UT_Wb / S  # Element-wise division
            derivative_coeffs = Vt.T @ S_inv_UT_Wb

        elif taylor_data.get("use_qr", False):  # type: ignore[attr-defined]
            # Use QR decomposition: solve R @ x = Q^T @ sqrt(W) @ b
            sqrt_W = taylor_data["sqrt_W"]
            Q = taylor_data["Q"]
            R = taylor_data["R"]

            Wb = sqrt_W @ b
            QT_Wb = Q.T @ Wb

            try:
                derivative_coeffs = np.linalg.solve(R, QT_Wb)
            except np.linalg.LinAlgError:
                # Fallback to least squares if R is singular
                A_matrix = taylor_data.get("A")  # type: ignore[attr-defined]
                if A_matrix is not None:
                    lstsq_result = lstsq(A_matrix, b)
                    derivative_coeffs = lstsq_result[0] if lstsq_result is not None else np.zeros(len(b))
                else:
                    derivative_coeffs = np.zeros(len(b))

        elif taylor_data.get("AtWA_inv") is not None:  # type: ignore[attr-defined]
            # Use precomputed normal equations
            derivative_coeffs = taylor_data["AtWA_inv"] @ taylor_data["AtW"] @ b
        else:
            # Final fallback to direct least squares
            A_matrix = taylor_data.get("A")  # type: ignore[attr-defined]
            if A_matrix is not None:
                lstsq_result = lstsq(A_matrix, b)
                derivative_coeffs = lstsq_result[0] if lstsq_result is not None else np.zeros(len(b))
            else:
                derivative_coeffs = np.zeros(len(b))

        # Handle case where coefficient computation failed
        if derivative_coeffs is None:
            derivative_coeffs = np.zeros(len(self.multi_indices))

        # Map coefficients to multi-indices
        derivatives = {}
        for k, beta in enumerate(self.multi_indices):
            derivatives[beta] = derivative_coeffs[k]

        # Apply inverse rotation for LCR boundary points (Issue #531)
        # Derivatives were computed in rotated frame, need to rotate back
        if (
            self._use_local_coordinate_rotation
            and hasattr(self, "_boundary_rotations")
            and point_idx in self._boundary_rotations
        ):
            derivatives = self._rotate_derivatives_back(derivatives, self._boundary_rotations[point_idx])

        return derivatives

    def compute_all_derivatives(
        self, u: np.ndarray, use_qp: bool | None = None
    ) -> dict[int, dict[tuple[int, ...], float]]:
        """
        Compute derivatives at all collocation points using precomputed Taylor matrices.

        When to use this vs GFDMOperator:
        - Use this method when you need QP constraints for monotonicity (M-matrix)
        - Use GFDMOperator for general GFDM needs (FP solver, one-off computations)

        Example:
            # For QP-constrained derivatives (HJB specific):
            solver = HJBGFDMSolver(problem, points, qp_optimization_level="auto")
            derivs = solver.compute_all_derivatives(u, use_qp=True)

            # For general GFDM (simpler, no QP):
            from mfg_pde.utils.numerical import GFDMOperator
            gfdm = GFDMOperator(points, delta=0.1)
            grad = gfdm.gradient(u)
            lap = gfdm.laplacian(u)

        Args:
            u: Function values at collocation points, shape (n_points,)
            use_qp: Override QP constraint behavior for this call.
                None: Use solver's qp_optimization_level setting
                True: Force QP constraints at all points
                False: Disable QP constraints for this call

        Returns:
            Dictionary mapping point index to derivative dictionary.
            derivatives[i] = {(1,): du/dx, (2,): d²u/dx², ...} for 1D
            derivatives[i] = {(1,0): du/dx, (0,1): du/dy, (2,0): d²u/dx², ...} for 2D
        """
        # Optionally override QP level for this computation
        saved_qp_level = None
        if use_qp is not None:
            saved_qp_level = self.qp_optimization_level
            self.qp_optimization_level = "always" if use_qp else "none"

        try:
            all_derivatives: dict[int, dict[tuple[int, ...], float]] = {}
            for i in range(self.n_points):
                all_derivatives[i] = self.approximate_derivatives(u, i)
            return all_derivatives
        finally:
            # Restore QP level if overridden
            if saved_qp_level is not None:
                self.qp_optimization_level = saved_qp_level

    # Note: _solve_monotone_constrained_qp moved to MonotonicityMixin
    # Note: _solve_unconstrained_fallback moved to MonotonicityMixin
    # Note: print_qp_diagnostics moved to MonotonicityMixin
    # Note: _compute_fd_weights_from_taylor moved to MonotonicityMixin

    def _approximate_all_derivatives_cached(self, u: np.ndarray) -> dict[int, dict[tuple[int, ...], float]]:
        """Compute all derivatives at once (for caching between residual/Jacobian)."""
        all_derivs: dict[int, dict[tuple[int, ...], float]] = {}
        for i in range(self.n_points):
            all_derivs[i] = self.approximate_derivatives(u, i)
        return all_derivs

    def _compute_hjb_residual_vectorized(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        grad_u: np.ndarray,
        lap_u: np.ndarray,
    ) -> np.ndarray:
        """
        Compute HJB residual using vectorized operations.

        Args:
            u_current: Current solution at collocation points
            u_n_plus_1: Solution at next time step
            m_n_plus_1: Density at collocation points
            grad_u: Pre-computed gradient, shape (n_points, dimension)
            lap_u: Pre-computed Laplacian, shape (n_points,)

        Returns:
            Residual vector, shape (n_points,)
        """
        dt = self.problem.T / self.problem.Nt
        u_t = (u_n_plus_1 - u_current) / dt

        # Compute Hamiltonian for all points
        # For standard LQ: H = |grad_u|^2 / (2*lambda) + V + gamma*m
        lambda_val = self._get_lambda_value()
        gamma_val = getattr(self.problem, "gamma", 0.0)

        # |grad_u|^2 for all points
        grad_norm_sq = np.sum(grad_u**2, axis=1)

        # Kinetic term: |p|^2 / (2*lambda)
        H_kinetic = grad_norm_sq / (2 * lambda_val)

        # Potential term (if exists)
        if hasattr(self.problem, "f_potential") and self.problem.f_potential is not None:
            # Need to interpolate potential to collocation points
            H_potential = self._interpolate_potential_to_collocation()
        else:
            H_potential = np.zeros(self.n_points)

        # Interaction term: gamma * m
        H_interaction = gamma_val * m_n_plus_1

        H_total = H_kinetic + H_potential + H_interaction

        # Running cost term L(x) if provided
        # This implements: H(x,p,m) = |p|^2/(2*lambda) + gamma*m + L(x)
        if hasattr(self, "_running_cost") and self._running_cost is not None:
            H_total = H_total + self._running_cost

        # Diffusion term: (sigma^2 / 2) * Laplacian
        sigma = getattr(self.problem, "diffusion", 0.0) or getattr(self.problem, "sigma", 0.0)
        diffusion_term = 0.5 * sigma**2 * lap_u

        # HJB residual: -u_t + H - diffusion = 0
        residual = -u_t + H_total - diffusion_term

        return residual

    def _interpolate_potential_to_collocation(self) -> np.ndarray:
        """
        Interpolate potential field to collocation points (cached).

        Handles arbitrary dimensions by building grid axes from bounds.
        """
        if hasattr(self, "_potential_at_collocation"):
            return self._potential_at_collocation

        if not hasattr(self.problem, "f_potential") or self.problem.f_potential is None:
            self._potential_at_collocation = np.zeros(self.n_points)
            return self._potential_at_collocation

        from scipy.interpolate import RegularGridInterpolator

        potential = self.problem.f_potential
        bounds = self.domain_bounds

        # Validate potential shape matches dimension
        if potential.ndim != self.dimension:
            raise ValueError(
                f"Potential array has {potential.ndim} dimensions but problem is "
                f"{self.dimension}D. Potential shape: {potential.shape}"
            )

        # Build grid axes for each dimension
        axes = []
        for d in range(self.dimension):
            xmin, xmax = bounds[d]
            axes.append(np.linspace(xmin, xmax, potential.shape[d]))

        # Handle 1D special case (needs flattening)
        if self.dimension == 1:
            potential = potential.flatten()

        # Create interpolator and evaluate at collocation points
        interp = RegularGridInterpolator(tuple(axes), potential, bounds_error=False, fill_value=0.0)
        self._potential_at_collocation = interp(self.collocation_points)

        return self._potential_at_collocation

    def _compute_hjb_residual_with_cache(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
        cached_derivs: dict[int, dict[tuple[int, ...], float]],
    ) -> np.ndarray:
        """Compute HJB residual using pre-computed derivatives."""
        from mfg_pde.core.derivatives import from_multi_index_dict

        residual = np.zeros(self.n_points)
        dimension = self.problem.dimension
        dt = self.problem.T / self.problem.Nt
        u_t = (u_n_plus_1 - u_current) / dt

        for i in range(self.n_points):
            x_pos = self.collocation_points[i]
            derivs = cached_derivs[i]
            p_derivs = from_multi_index_dict(derivs, dimension=dimension)
            laplacian = p_derivs.laplacian or 0.0

            H = self.problem.H(i, m_n_plus_1[i], derivs=p_derivs, x_position=x_pos)

            # Add running cost L(x) if provided
            if hasattr(self, "_running_cost") and self._running_cost is not None:
                H = H + self._running_cost[i]

            sigma_val = self._get_sigma_value(i)
            diffusion_term = 0.5 * sigma_val**2 * laplacian
            residual[i] = -u_t[i] + H - diffusion_term

        return residual

    def _compute_hjb_jacobian_vectorized(
        self,
        grad_u: np.ndarray,
    ):
        """
        Compute sparse Jacobian using vectorized operations with pre-computed differentiation matrices.

        For standard LQ Hamiltonian H = |p|²/(2λ), dH/dp = p/λ.
        Jacobian structure: J = (1/dt)I + (1/λ) * Σ_d diag(p_d) @ D_grad[d] - (σ²/2) * D_lap

        Args:
            grad_u: Pre-computed gradient, shape (n_points, dimension)

        Returns:
            Sparse Jacobian matrix in CSR format
        """
        from scipy.sparse import diags, eye

        if not hasattr(self, "_D_grad") or self._D_grad is None:
            self._build_differentiation_matrices()

        n = self.n_points
        d = self.dimension
        dt = self.problem.T / self.problem.Nt
        lambda_val = self._get_lambda_value()
        sigma = getattr(self.problem, "diffusion", 0.0) or getattr(self.problem, "sigma", 0.0)
        diffusion_coeff = 0.5 * sigma**2

        # Time derivative term: (1/dt) * I
        jacobian = (1.0 / dt) * eye(n, format="csr")

        # Hamiltonian gradient term: (1/λ) * Σ_d diag(p_d) @ D_grad[d]
        # For LQ: dH/dp = p/λ, so ∂(dH/dp · ∇u)/∂u_j = (p/λ) · (∂∇u/∂u_j)
        for dim in range(d):
            p_d = grad_u[:, dim] / lambda_val  # dH/dp_d = p_d / λ
            jacobian = jacobian + diags(p_d, format="csr") @ self._D_grad[dim]

        # Diffusion term: -(σ²/2) * D_lap
        jacobian = jacobian - diffusion_coeff * self._D_lap

        return jacobian

    def _compute_hjb_jacobian_sparse(
        self,
        u_current: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
        cached_derivs: dict[int, dict[tuple[int, ...], float]],
    ):
        """Compute sparse Jacobian using pre-computed derivatives and GFDM weights."""
        from scipy.sparse import lil_matrix

        from mfg_pde.core.derivatives import from_multi_index_dict, to_multi_index_dict

        n = self.n_points
        d = self.problem.dimension
        dt = self.problem.T / self.problem.Nt

        # Pre-cache all derivative weights (avoids repeated method call overhead)
        if not hasattr(self, "_cached_derivative_weights"):
            self._cached_derivative_weights = [self._gfdm_operator.get_derivative_weights(i) for i in range(n)]

        # Use LIL format for efficient construction
        jacobian = lil_matrix((n, n))

        for i in range(n):
            weights = self._cached_derivative_weights[i]
            if weights is None:
                jacobian[i, i] = 1.0 / dt  # Fallback: identity
                continue

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]
            lap_weights = weights["lap_weights"]

            p_derivs = from_multi_index_dict(cached_derivs[i], dimension=d)

            dH_dp = self.problem.dH_dp(
                x_idx=i,
                m_at_x=m_n_plus_1[i],
                derivs=to_multi_index_dict(p_derivs),
                t_idx=time_idx,
            )
            if dH_dp is None:
                dH_dp = self._compute_dH_dp_fd(i, m_n_plus_1[i], p_derivs, time_idx)

            sigma_val = self._get_sigma_value(i)
            diffusion_coeff = 0.5 * sigma_val**2

            # Neighbor contributions
            for k, j in enumerate(neighbor_indices):
                if j < 0:
                    continue  # Skip ghost particles
                val = np.dot(dH_dp, grad_weights[:, k]) - diffusion_coeff * lap_weights[k]
                jacobian[i, j] = val

            # Center point contribution
            center_grad_weight = -np.sum(grad_weights, axis=1)
            center_lap_weight = -np.sum(lap_weights)
            jacobian[i, i] += np.dot(dH_dp, center_grad_weight) - diffusion_coeff * center_lap_weight
            jacobian[i, i] += 1.0 / dt  # Time derivative

        return jacobian.tocsr()

    def _apply_boundary_conditions_to_sparse_system(self, jacobian_sparse, residual: np.ndarray, time_idx: int):
        """
        Apply boundary conditions to sparse Jacobian using Row Replacement pattern.

        This implements the "Direct Collocation" approach recommended in GFDM literature:
        - For interior points: PDE equation rows (already set)
        - For boundary points: Replace PDE rows with BC equation rows

        For Dirichlet: Row becomes identity (u_i = g)
        For Neumann: Row becomes normal derivative operator (∂u/∂n = 0)
        """
        if len(self.boundary_indices) == 0:
            return jacobian_sparse, residual

        # Convert to LIL format for efficient row modification (O(nnz) not O(n²))
        jac_lil = jacobian_sparse.tolil()
        residual_bc = residual.copy()

        # Apply BC directly on sparse matrix (avoid dense conversion)
        bc_type = self._get_boundary_condition_property("type", "neumann")
        bc_values = self._get_boundary_condition_property("values", {})
        normals = self._bc_config.get("normals", None) if self._bc_config else None
        dimension = self.dimension

        for local_idx, i in enumerate(self.boundary_indices):
            # Clear row (LIL supports efficient row clearing)
            jac_lil[i, :] = 0.0

            if bc_type == "dirichlet":
                # Dirichlet: u = g
                jac_lil[i, i] = 1.0
                if isinstance(bc_values, dict):
                    residual_bc[i] = bc_values.get(i, 0.0)
                elif callable(bc_values):
                    residual_bc[i] = bc_values(self.collocation_points[i])
                else:
                    residual_bc[i] = float(bc_values) if bc_values else 0.0

            elif bc_type in ("neumann", "no_flux"):
                # Neumann: du/dn = g
                # Skip row replacement if ghost nodes are active - BC is enforced structurally
                if self._use_ghost_nodes and hasattr(self, "_ghost_node_map") and i in self._ghost_node_map:
                    # Ghost nodes enforce Neumann BC through symmetric stencils
                    # Keep the original PDE row (already set in Jacobian)
                    # Restore the row from the original Jacobian (undo the clearing above)
                    jac_lil[i, :] = jacobian_sparse[i, :].toarray().flatten()
                    # Residual also stays as-is (PDE residual)
                    continue

                # For LCR boundary points, use our LCR-corrected weights (Issue #531)
                if (
                    self._use_local_coordinate_rotation
                    and hasattr(self, "_boundary_rotations")
                    and i in self._boundary_rotations
                ):
                    weights = self._compute_derivative_weights_from_taylor(i)
                else:
                    weights = self._gfdm_operator.get_derivative_weights(i)

                if weights is None:
                    jac_lil[i, i] = 1.0
                    residual_bc[i] = 0.0
                    continue

                neighbor_indices = weights["neighbor_indices"]
                grad_weights = weights["grad_weights"]

                # Get normal vector
                if normals is not None and local_idx < len(normals):
                    normal = normals[local_idx]
                else:
                    normal = self._compute_outward_normal(i)

                # Normal derivative: du/dn = n . grad(u)
                center_weight = 0.0
                for k, j in enumerate(neighbor_indices):
                    if j >= 0 and j != i:
                        weight = sum(normal[d] * grad_weights[d, k] for d in range(dimension))
                        jac_lil[i, j] = weight
                        center_weight -= weight

                jac_lil[i, i] = center_weight

                if isinstance(bc_values, dict):
                    residual_bc[i] = bc_values.get(i, 0.0)
                else:
                    residual_bc[i] = 0.0  # No-flux default

        return jac_lil.tocsr(), residual_bc

    def _get_lambda_value(self) -> float:
        """
        Get control cost parameter lambda with validation.

        Returns:
            Positive lambda value

        Raises:
            ValueError: If lambda <= 0

        Notes:
            Lambda appears in the Hamiltonian as H = |p|²/(2λ) + ...
            Division by lambda requires λ > 0.
        """
        lambda_val = getattr(self.problem, "lambda_", 1.0)
        if lambda_val is None:
            lambda_val = 1.0
        if lambda_val <= 0:
            raise ValueError(
                f"Control cost parameter lambda_ must be positive, got {lambda_val}. "
                f"Set problem.lambda_ to a positive value."
            )
        return float(lambda_val)

    def _get_sigma_value(self, point_idx: int | None = None) -> float:
        """
        Get diffusion coefficient value, handling both numeric and callable sigma.

        Args:
            point_idx: Collocation point index (for callable sigma evaluation)

        Returns:
            Numeric sigma value

        Handles three cases:
        1. problem.nu exists (legacy attribute)
        2. problem.sigma is callable → evaluate at collocation point
        3. problem.sigma is numeric → use directly (fallback: 1.0)
        """
        if hasattr(self.problem, "nu"):
            # Legacy attribute name from some problem formulations
            return float(self.problem.nu)
        elif callable(getattr(self.problem, "sigma", None)):
            # Callable sigma: evaluate at current point if available
            if point_idx is not None and point_idx < len(self.collocation_points):
                x = self.collocation_points[point_idx]
                return float(self.problem.sigma(x))
            else:
                # Fallback: use representative value (center of domain)
                return 1.0
        else:
            # Numeric sigma: use directly (with fallback to default)
            return float(getattr(self.problem, "sigma", 1.0))

    # Note: _check_monotonicity_violation moved to MonotonicityMixin
    # Note: _check_m_matrix_property moved to MonotonicityMixin
    # Note: _build_monotonicity_constraints moved to MonotonicityMixin
    # Note: _build_hamiltonian_gradient_constraints moved to MonotonicityMixin

    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        show_progress: bool = True,
        diffusion_field: float | np.ndarray | None = None,
        running_cost: np.ndarray | None = None,
        # Deprecated parameter names for backward compatibility
        M_density_evolution_from_FP: np.ndarray | None = None,
        U_final_condition_at_T: np.ndarray | None = None,
        U_from_prev_picard: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the HJB system using GFDM collocation method.

        Args:
            M_density: (Nt, *spatial_shape) density from FP solver
            U_terminal: (*spatial_shape,) terminal condition u(T,x)
            U_coupling_prev: (Nt, *spatial_shape) previous coupling iteration estimate
            show_progress: Whether to display progress bar for timesteps
            running_cost: (n_points,) time-independent running cost L(x) at collocation points.
                This is added to the Hamiltonian at each backward time step:
                H(x,p,m) = |p|^2/(2*lambda) + gamma*m + L(x)
            diffusion_field: Optional diffusion coefficient override

        Returns:
            (Nt, *spatial_shape) solution array
        """
        # Handle deprecated parameter names with warnings
        if M_density_evolution_from_FP is not None:
            if M_density is not None:
                raise ValueError("Cannot specify both 'M_density' and deprecated 'M_density_evolution_from_FP'")
            warnings.warn(
                "Parameter 'M_density_evolution_from_FP' is deprecated. Use 'M_density' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            M_density = M_density_evolution_from_FP

        if U_final_condition_at_T is not None:
            if U_terminal is not None:
                raise ValueError("Cannot specify both 'U_terminal' and deprecated 'U_final_condition_at_T'")
            warnings.warn(
                "Parameter 'U_final_condition_at_T' is deprecated. Use 'U_terminal' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            U_terminal = U_final_condition_at_T

        if U_from_prev_picard is not None:
            if U_coupling_prev is not None:
                raise ValueError("Cannot specify both 'U_coupling_prev' and deprecated 'U_from_prev_picard'")
            warnings.warn(
                "Parameter 'U_from_prev_picard' is deprecated. Use 'U_coupling_prev' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            U_coupling_prev = U_from_prev_picard

        # Validate required parameters
        if U_terminal is None:
            raise ValueError("U_terminal is required")

        from mfg_pde.utils.progress import RichProgressBar

        # Determine n_time_points from available data or problem configuration
        # n_time_points = Nt + 1 (number of time knots including t=0 and t=T)
        if M_density is not None:
            n_time_points = M_density.shape[0]
        else:
            n_time_points = self.problem.Nt + 1

        # For standalone HJB (no MFG coupling), use defaults
        if M_density is None:
            # Default: uniform density (no coupling effect)
            M_density = np.ones((n_time_points, *U_terminal.shape))
        if U_coupling_prev is None:
            # Default: zero coupling (pure HJB)
            U_coupling_prev = np.zeros((n_time_points, *U_terminal.shape))

        # Store original spatial shape for reshaping output
        self._output_spatial_shape = M_density.shape[1:]

        # Store running cost for use in residual computation
        # Running cost L(x) is added to Hamiltonian at each backward step
        if running_cost is not None:
            if running_cost.shape[0] != self.n_points:
                raise ValueError(
                    f"running_cost must have shape (n_points,) = ({self.n_points},), got shape {running_cost.shape}"
                )
            self._running_cost = running_cost.copy()
        else:
            self._running_cost = None

        # Detect if input is already in collocation format (pure meshfree mode)
        # Grid format: M_density.shape = (Nt, Nx, Ny, ...)
        # Collocation format: M_density.shape = (Nt, n_points)
        is_meshfree_input = M_density.ndim == 2 and M_density.shape[1] == self.n_points

        # For GFDM, we work directly with collocation points
        U_solution_collocation = np.zeros((n_time_points, self.n_points))

        if is_meshfree_input:
            # Pure meshfree mode: input already at collocation points
            M_collocation = M_density.copy()
            U_prev_collocation = U_coupling_prev.copy()
            # U_terminal should also be at collocation points
            U_solution_collocation[n_time_points - 1, :] = U_terminal.copy()
        else:
            # Hybrid mode: map grid data to collocation points
            M_collocation = self._map_grid_to_collocation_batch(M_density)
            U_prev_collocation = self._map_grid_to_collocation_batch(U_coupling_prev)
            # Set final condition at t=T (last time index = n_time_points - 1)
            U_solution_collocation[n_time_points - 1, :] = self._map_grid_to_collocation(U_terminal.flatten())

        # Backward time stepping: Nt steps from index (n_time_points-2) down to 0
        # This covers all Nt intervals in the backward direction
        timestep_range = range(n_time_points - 2, -1, -1)
        if show_progress:
            timestep_range = RichProgressBar(
                timestep_range,
                desc="HJB (backward)",
                unit="step",
                disable=False,
            )

        for n in timestep_range:
            U_solution_collocation[n, :] = self._solve_timestep(
                U_solution_collocation[n + 1, :],
                U_prev_collocation[n, :],
                M_collocation[n, :],  # FIXED: Use m^n, not m^{n+1} (same-time coupling)
                n,
            )

            # Update progress bar with QP statistics if available
            if show_progress and hasattr(timestep_range, "set_postfix"):
                postfix = {}
                if self.qp_optimization_level in ["auto", "always"] and hasattr(self, "qp_stats"):
                    postfix["qp_solves"] = self.qp_stats.get("total_qp_solves", 0)
                if postfix:
                    timestep_range.set_postfix(**postfix)

        # Return format depends on input mode
        if is_meshfree_input:
            # Pure meshfree: return collocation data directly
            return U_solution_collocation
        else:
            # Hybrid mode: map back to grid
            U_solution = self._map_collocation_to_grid_batch(U_solution_collocation)
            return U_solution

    def _solve_timestep(
        self,
        u_n_plus_1: np.ndarray,
        u_prev_picard: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """Solve HJB at one time step using Newton iteration."""
        from scipy.sparse.linalg import spsolve

        u_current = u_n_plus_1.copy()

        # Check if we can use vectorized path (standard LQ Hamiltonian)
        is_custom = getattr(self.problem, "is_custom", False)
        use_vectorized = not is_custom and self.qp_optimization_level == "none"

        for _newton_iter in range(self.max_newton_iterations):
            if use_vectorized:
                # Fast vectorized path for standard LQ Hamiltonian
                grad_u, lap_u = self._compute_derivatives_vectorized(u_current)

                # Vectorized residual
                residual = self._compute_hjb_residual_vectorized(u_current, u_n_plus_1, m_n_plus_1, grad_u, lap_u)

                # Check convergence
                if np.linalg.norm(residual) < self.newton_tolerance:
                    break

                # Vectorized Jacobian
                jacobian_sparse = self._compute_hjb_jacobian_vectorized(grad_u)
            else:
                # Original per-point path for custom Hamiltonians or QP mode
                all_derivs = self._approximate_all_derivatives_cached(u_current)

                residual = self._compute_hjb_residual_with_cache(
                    u_current, u_n_plus_1, m_n_plus_1, time_idx, all_derivs
                )

                if np.linalg.norm(residual) < self.newton_tolerance:
                    break

                jacobian_sparse = self._compute_hjb_jacobian_sparse(u_current, m_n_plus_1, time_idx, all_derivs)

            # Apply boundary conditions (sparse-aware)
            jacobian_bc, residual_bc = self._apply_boundary_conditions_to_sparse_system(
                jacobian_sparse, residual, time_idx
            )

            # Newton update using sparse solver
            try:
                delta_u = spsolve(jacobian_bc, -residual_bc)
            except Exception:
                # Fallback to dense solver
                delta_u = np.linalg.lstsq(jacobian_bc.toarray(), -residual_bc, rcond=None)[0]

            # Limit step size to prevent extreme updates
            max_step = 10.0  # Reasonable limit for value function updates
            step_norm = np.linalg.norm(delta_u)
            if step_norm > max_step:
                delta_u = delta_u * (max_step / step_norm)

            u_current += delta_u

            # Apply boundary conditions to solution
            u_current = self._apply_boundary_conditions_to_solution(u_current, time_idx)

        return u_current

    def _compute_hjb_residual(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """Compute HJB residual at collocation points."""
        from mfg_pde.core.derivatives import from_multi_index_dict

        residual = np.zeros(self.n_points)
        dimension = self.problem.dimension

        # Time derivative approximation (backward Euler)
        # For backward-in-time problems: ∂u/∂t ≈ (u_{n+1} - u_n) / dt
        # where t_{n+1} > t_n (future time is at n+1)
        # problem.Nt = number of time intervals, so dt = T / Nt
        dt = self.problem.T / self.problem.Nt
        u_t = (u_n_plus_1 - u_current) / dt

        for i in range(self.n_points):
            # Get spatial coordinates for this collocation point
            x_pos = self.collocation_points[i]

            # Approximate derivatives using GFDM
            derivs = self.approximate_derivatives(u_current, i)

            # Convert multi-index derivatives dict to DerivativeTensors (nD support)
            p_derivs = from_multi_index_dict(derivs, dimension=dimension)
            laplacian = p_derivs.laplacian or 0.0

            # Hamiltonian (user-provided)
            # problem.H() signature: H(x_idx, m_at_x, derivs, x_position=...)
            # Pass x_position explicitly since collocation points may differ from geometry grid
            H = self.problem.H(i, m_n_plus_1[i], derivs=p_derivs, x_position=x_pos)

            # Diffusion coefficient
            # NOTE: HJB uses (σ²/2) factor from control theory (Pontryagin maximum principle)
            # This differs from FP equation which uses σ² (standard diffusion form)
            # Both forms are correct - they arise from different derivations of MFG system
            #
            # BUG #15 FIX: Handle both callable sigma(x) and numeric sigma
            # Use helper method to get sigma value (handles nu, callable sigma, numeric sigma)
            sigma_val = self._get_sigma_value(i)

            diffusion_term = 0.5 * sigma_val**2 * laplacian

            # HJB residual: -u_t + H - (sigma²/2)Δu = 0
            # Note: For backward-in-time problems, the HJB equation has -∂u/∂t
            residual[i] = -u_t[i] + H - diffusion_term

        return residual

    def _compute_dH_dp_fd(
        self,
        point_idx: int,
        m_at_x: float,
        derivs: DerivativeTensors,
        time_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute dH/dp - analytical for standard LQ Hamiltonian, FD otherwise.

        For standard LQ Hamiltonian H = |∇u|²/(2λ), dH/dp = p/λ analytically.

        Args:
            point_idx: Collocation point index
            m_at_x: Density value at the point
            derivs: DerivativeTensors with current gradient/hessian
            time_idx: Time index (currently unused, kept for API compatibility)

        Returns:
            dH/dp array, shape (dim,)
        """
        p = derivs.grad if derivs.grad is not None else np.zeros(self.problem.dimension)

        # Fast path: for standard LQ Hamiltonian H = |p|²/(2λ), dH/dp = p/λ
        lambda_val = getattr(self.problem, "lambda_", None)
        if lambda_val is not None and lambda_val > 0:
            # Check if using standard (non-custom) Hamiltonian
            is_custom = getattr(self.problem, "is_custom", False)
            if not is_custom:
                return p / lambda_val

        # Fallback: finite differences for custom Hamiltonians using scipy
        x_pos = self.collocation_points[point_idx]
        hess = derivs.hess if derivs.hess is not None else np.zeros((len(p), len(p)))

        def H_of_p(p_vec: np.ndarray) -> float:
            """Hamiltonian as function of momentum p only."""
            from mfg_pde.core.derivatives import DerivativeTensors

            d = DerivativeTensors.from_arrays(grad=p_vec, hess=hess)
            return self.problem.H(point_idx, m_at_x, derivs=d, x_position=x_pos)

        # Use scipy's approx_fprime for gradient computation
        return approx_fprime(p, H_of_p, epsilon=1e-7)

    def _compute_hjb_jacobian_analytic(
        self,
        u_current: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """
        Compute Jacobian using analytic formula with GFDM weights.

        Formula: ∂R_i/∂u_j = (1/dt)δ_{ij} + (∂H/∂p)·(∂p_i/∂u_j) - (σ²/2)·(∂Δu_i/∂u_j)

        Uses user-provided dH_dp if available, otherwise FD on H.
        """
        from mfg_pde.core.derivatives import from_multi_index_dict, to_multi_index_dict

        n = self.n_points
        d = self.problem.dimension
        dt = self.problem.T / self.problem.Nt
        jacobian = np.zeros((n, n))

        for i in range(n):
            # Get derivative weights from GFDM
            weights = self._gfdm_operator.get_derivative_weights(i)
            if weights is None:
                continue  # Skip points without valid Taylor data

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]  # shape (d, n_neighbors)
            lap_weights = weights["lap_weights"]  # shape (n_neighbors,)

            # Compute derivatives at point i
            derivs_dict = self.approximate_derivatives(u_current, i)

            # Build DerivativeTensors using nD infrastructure
            p_derivs = from_multi_index_dict(derivs_dict, dimension=d)

            # Get ∂H/∂p (user-provided or FD fallback)
            dH_dp = self.problem.dH_dp(
                x_idx=i,
                m_at_x=m_n_plus_1[i],
                derivs=to_multi_index_dict(p_derivs),
                t_idx=time_idx,
            )

            if dH_dp is None:
                # Fallback: compute via FD on H
                dH_dp = self._compute_dH_dp_fd(i, m_n_plus_1[i], p_derivs, time_idx)

            # Get sigma for diffusion term
            sigma_val = self._get_sigma_value(i)
            diffusion_coeff = 0.5 * sigma_val**2

            # Build row i of Jacobian
            # For neighbors: use GFDM weights directly
            for k, j in enumerate(neighbor_indices):
                if j < 0:
                    continue  # Skip ghost particles

                # ∂R_i/∂u_j = (∂H/∂p) · grad_weights[:, k] - diffusion * lap_weights[k]
                jacobian[i, j] = np.dot(dH_dp, grad_weights[:, k]) - diffusion_coeff * lap_weights[k]

            # For center point contribution: weights are -sum(row weights)
            # because b = u_neighbors - u_center, so ∂b/∂u_center = -1
            center_grad_weight = -np.sum(grad_weights, axis=1)
            center_lap_weight = -np.sum(lap_weights)
            jacobian[i, i] += np.dot(dH_dp, center_grad_weight) - diffusion_coeff * center_lap_weight

            # Time derivative contribution: (1/dt) on diagonal
            jacobian[i, i] += 1.0 / dt

        return jacobian

    def _compute_hjb_jacobian(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,  # FIXED: Added actual u_n_plus_1 parameter
        u_prev_picard: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton iteration.

        Uses analytic Jacobian if dH_dp is available (faster),
        otherwise falls back to finite differences on residual.

        Bug #15 fix: Disable QP in Jacobian computation to reduce QP calls from ~750k to ~7.5k.
        Jacobian only affects Newton convergence rate, not final monotonicity (enforced by residual).
        """
        # Try analytic Jacobian first (faster if dH_dp available or FD on H)
        # Analytic Jacobian uses GFDM weights directly - O(n·k) vs O(n²) for FD
        # Works for any dimension since GFDM weights are dimension-agnostic
        return self._compute_hjb_jacobian_analytic(u_current, m_n_plus_1, time_idx)

        # Fallback: numerical finite differences on full residual
        n = self.n_points
        jacobian = np.zeros((n, n))

        # Finite difference step
        eps = 1e-7

        # Bug #15 fix: Temporarily disable QP for Jacobian computation
        saved_qp_level = self.qp_optimization_level
        self.qp_optimization_level = "none"

        try:
            # Compute base residual ONCE (was incorrectly inside loop before)
            residual_base = self._compute_hjb_residual(u_current, u_n_plus_1, m_n_plus_1, time_idx)

            # Compute Jacobian by columns (perturbing each u[j])
            # Sparse optimization: only compute affected rows (neighbors of j)
            for j in range(n):
                u_plus = u_current.copy()
                u_plus[j] += eps

                # Get rows affected by perturbing u[j] (j and its neighbors)
                affected_rows = self._get_affected_rows(j)

                # Compute residual only at affected points
                residual_plus = self._compute_hjb_residual(u_plus, u_n_plus_1, m_n_plus_1, time_idx)

                # Only update affected rows (sparse pattern)
                for i in affected_rows:
                    jacobian[i, j] = (residual_plus[i] - residual_base[i]) / eps
        finally:
            # Restore QP for residual evaluation
            self.qp_optimization_level = saved_qp_level

        return jacobian

    def _apply_boundary_conditions_to_solution(self, u: np.ndarray, time_idx: int) -> np.ndarray:
        """Apply boundary conditions directly to solution array."""
        if len(self.boundary_indices) == 0:
            return u

        # Use unified BC config (single source of truth) when using new infrastructure
        if self._use_new_infrastructure and self._bc_config is not None:
            bc_type = self._bc_config["type"]
            bc_values = self._bc_config["values"]
        else:
            # Legacy path: inconsistent defaults preserved for backward compatibility
            bc_type_val = self._get_boundary_condition_property("type", "neumann")
            bc_type = bc_type_val.lower() if isinstance(bc_type_val, str) else "neumann"
            bc_values = self._get_boundary_condition_property("value", 0.0)

        if bc_type == "dirichlet":
            # For collocation points on boundaries, enforce Dirichlet values
            if callable(bc_values):
                # Time-dependent or space-dependent BC
                current_time = self.problem.T * time_idx / self.problem.Nt
                for i in self.boundary_indices:
                    u[i] = bc_values(self.collocation_points[i], current_time)
            else:
                # Constant BC value
                u[self.boundary_indices] = bc_values
        # For Neumann: no direct solution modification (enforced via residual)

        return u

    def _apply_boundary_conditions_to_system(
        self, jacobian: np.ndarray, residual: np.ndarray, time_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply boundary conditions to the linear system J·δu = -R.

        For Dirichlet BC: Set row to identity and residual to zero.
        For Neumann BC: Row Replacement with normal derivative operator.
        """
        if len(self.boundary_indices) == 0:
            return jacobian, residual

        jacobian_bc = jacobian.copy()
        residual_bc = residual.copy()

        # Use new infrastructure with DirectCollocationHandler
        if self._use_new_infrastructure and self._bc_handler is not None:
            self._bc_handler.apply_to_matrix(
                A=jacobian_bc,
                b=residual_bc,
                boundary_indices=self.boundary_indices,
                operator=self._gfdm_operator,
                bc_config=self._bc_config,
            )
            return jacobian_bc, residual_bc

        # Legacy path (deprecated)
        bc_type_val = self._get_boundary_condition_property("type", "neumann")
        bc_type = bc_type_val.lower() if isinstance(bc_type_val, str) else "neumann"

        if bc_type == "dirichlet":
            for i in self.boundary_indices:
                jacobian_bc[i, :] = 0.0
                jacobian_bc[i, i] = 1.0
                residual_bc[i] = 0.0

        return jacobian_bc, residual_bc


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBGFDMSolver...")

    import numpy as np

    from mfg_pde import MFGProblem

    # Test 1D problem with uniform collocation points matching problem grid
    problem_1d = MFGProblem(Nx=20, Nt=10, T=1.0, diffusion=0.1)

    # Use problem grid points as collocation points to avoid index mismatch
    collocation_points = problem_1d.xSpace.reshape(-1, 1)

    solver_1d = HJBGFDMSolver(
        problem_1d,
        collocation_points=collocation_points,
        delta=0.15,
        taylor_order=2,
        weight_function="wendland",
    )

    # Test solver initialization
    assert solver_1d.dimension == 1
    assert solver_1d.n_points == problem_1d.Nx + 1
    assert solver_1d.delta == 0.15
    assert solver_1d.taylor_order == 2
    assert solver_1d.hjb_method_name == "GFDM"
    print("  [1D] Solver initialized")
    print(f"       Collocation points: {solver_1d.n_points}, Delta: {solver_1d.delta}")

    # Test derivative computation API (1D)
    # f(x) = x^2 -> df/dx = 2x, d²f/dx² = 2
    x = collocation_points[:, 0]
    u_1d = x**2

    # Test compute_all_derivatives
    all_derivs_1d = solver_1d.compute_all_derivatives(u_1d)
    assert len(all_derivs_1d) == solver_1d.n_points
    # Interior points should have derivatives
    mid_idx = solver_1d.n_points // 2
    assert (1,) in all_derivs_1d[mid_idx], f"Missing gradient key (1,) at point {mid_idx}"
    print(f"  [1D] compute_all_derivatives: {len(all_derivs_1d)} points")
    print(f"       Multi-indices: {solver_1d.multi_indices}")

    # Test 2D problem
    print("\n  [2D] Testing 2D solver...")

    # Create 2D collocation points (grid)
    Nx_2d = 10
    x_grid = np.linspace(0, 1, Nx_2d)
    y_grid = np.linspace(0, 1, Nx_2d)
    xx, yy = np.meshgrid(x_grid, y_grid)
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])

    problem_2d = MFGProblem(Nx=Nx_2d, Nt=5, T=1.0, diffusion=0.1, dimension=2)

    solver_2d = HJBGFDMSolver(
        problem_2d,
        collocation_points=points_2d,
        delta=0.2,
        taylor_order=2,
        weight_function="wendland",
    )
    print(f"       Collocation points: {solver_2d.n_points}, Delta: {solver_2d.delta}")
    print(f"       Multi-indices: {solver_2d.multi_indices}")

    # f(x,y) = x² + y² -> gradient = [2x, 2y], laplacian = 4
    u_2d = points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2

    # Test compute_all_derivatives
    all_derivs_2d = solver_2d.compute_all_derivatives(u_2d)
    assert len(all_derivs_2d) == solver_2d.n_points

    # Find interior point (center of grid)
    mid_idx_2d = 55  # Center of 10x10 grid
    derivs_mid = all_derivs_2d[mid_idx_2d]
    print(f"  [2D] Derivatives at interior point {mid_idx_2d}:")
    print(f"       Keys: {list(derivs_mid.keys())}")

    # Check expected derivatives for f(x,y) = x² + y² at interior point
    if derivs_mid:
        grad_x = derivs_mid.get((1, 0), 0.0)
        grad_y = derivs_mid.get((0, 1), 0.0)
        lap_xx = derivs_mid.get((2, 0), 0.0)
        lap_yy = derivs_mid.get((0, 2), 0.0)
        print(f"       du/dx = {grad_x:.4f} (expected: {2 * points_2d[mid_idx_2d, 0]:.4f})")
        print(f"       du/dy = {grad_y:.4f} (expected: {2 * points_2d[mid_idx_2d, 1]:.4f})")
        print(f"       d²u/dx² = {lap_xx:.4f} (expected: 2.0)")
        print(f"       d²u/dy² = {lap_yy:.4f} (expected: 2.0)")

    print("\nNote: For gradient/laplacian utilities, use mfg_pde.utils.numerical.gfdm_operators")
    print("Smoke tests passed!")
