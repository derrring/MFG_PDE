from __future__ import annotations

import importlib.util
import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist

from mfg_pde.utils.numerical.differential_utils import (
    compute_dH_dp,
)

# Legacy operator for backward compatibility (deprecated)
from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator

# New GFDM infrastructure (Strategy Pattern)
from mfg_pde.utils.numerical.gfdm_strategies import (
    DirectCollocationHandler,
    TaylorOperator,
    create_operator,
)
from mfg_pde.utils.numerical.particle.interpolation import (
    interpolate_grid_to_particles,
)
from mfg_pde.utils.numerical.qp_utils import QPCache, QPSolver

from .base_hjb import BaseHJBSolver
from .hjb_gfdm_monotonicity import MonotonicityMixin

# Optional QP solver imports
CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None
OSQP_AVAILABLE = importlib.util.find_spec("osqp") is not None

if TYPE_CHECKING:
    from mfg_pde.core.derivatives import DerivativeTensors
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions


class HJBGFDMSolver(MonotonicityMixin, BaseHJBSolver):
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
    """

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
    ):
        """
        Initialize the GFDM HJB solver.

        Args:
            problem: MFG problem instance
            collocation_points: (N_points, d) array of collocation points
            delta: Neighborhood radius for collocation
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("gaussian", "inverse_distance", "uniform")
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

    def _compute_outward_normal(self, point_idx: int) -> np.ndarray:
        """
        Compute outward normal vector at a boundary point.

        For rectangular domains, the normal is determined by which boundary
        the point lies on. For corner points (on multiple boundaries), returns
        the average of all boundary normals.

        Args:
            point_idx: Index of boundary point

        Returns:
            Unit outward normal vector, shape (dimension,)
        """
        point = self.collocation_points[point_idx]
        normal = np.zeros(self.dimension)
        tol = 1e-10

        for d in range(self.dimension):
            low, high = self.domain_bounds[d]
            if abs(point[d] - low) < tol:
                normal[d] -= 1.0  # At lower boundary, normal points inward (negative)
            if abs(point[d] - high) < tol:
                normal[d] += 1.0  # At upper boundary, normal points outward (positive)

        # Normalize (handles corners where multiple boundaries meet)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
        else:
            # Fallback: no clear boundary detected, use zero vector
            normal = np.zeros(self.dimension)

        return normal

    def _compute_boundary_normals(self) -> np.ndarray | None:
        """
        Compute outward normal vectors for all boundary points.

        Tries to use geometry infrastructure (BoundaryConditions.get_outward_normal)
        first, falls back to axis-aligned computation for rectangular domains.

        Returns:
            Array of shape (n_boundary, dimension) with unit normal vectors,
            or None if no boundary points.
        """
        if len(self.boundary_indices) == 0:
            return None

        normals = np.zeros((len(self.boundary_indices), self.dimension))

        # Try to use geometry infrastructure first (supports SDF domains)
        bc_obj = None
        if hasattr(self.problem, "geometry") and self.problem.geometry is not None:
            geom = self.problem.geometry
            if hasattr(geom, "boundary_conditions"):
                bc_obj = geom.boundary_conditions

        for local_idx, global_idx in enumerate(self.boundary_indices):
            point = self.collocation_points[global_idx]

            # Try BoundaryConditions.get_outward_normal (works for SDF domains)
            if bc_obj is not None and hasattr(bc_obj, "get_outward_normal"):
                normal = bc_obj.get_outward_normal(point)
                if normal is not None:
                    normals[local_idx] = normal
                    continue

            # Fallback: axis-aligned computation for rectangular domains
            normals[local_idx] = self._compute_outward_normal(global_idx)

        return normals

    def _create_bc_config(self) -> dict:
        """
        Create unified BC configuration dict (single source of truth).

        This ensures consistent BC handling between:
        - _apply_boundary_conditions_to_sparse_system (Jacobian)
        - _apply_boundary_conditions_to_solution (solution)
        - DirectCollocationHandler.apply_to_residual (residual)

        Returns:
            BC configuration dict with keys: type, values, normals
        """
        # Resolve BC type from boundary_conditions (single source of truth)
        bc_type = self._get_boundary_condition_property("type", None)

        if bc_type is None:
            # Infer from BC object or default to neumann for MFG
            if hasattr(self.boundary_conditions, "default_bc"):
                bc_type = self.boundary_conditions.default_bc.value.lower()
            else:
                bc_type = "neumann"  # Default for MFG (no-flux)

        if isinstance(bc_type, str):
            bc_type = bc_type.lower()

        # Get BC values (for Dirichlet or non-zero Neumann)
        bc_value = self._get_boundary_condition_property("value", 0.0)

        # Build values dict for per-point BC values
        if callable(bc_value):
            # Time-dependent or space-dependent BC - will be evaluated at runtime
            bc_values = bc_value
        else:
            # Constant value for all boundary points
            bc_values = bc_value

        return {
            "type": bc_type,
            "values": bc_values,
            "normals": self._boundary_normals,
        }

    def _build_neumann_bc_weights(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Build GFDM weights for normal derivative at boundary points.

        For Neumann BC (∂u/∂n = 0), we need weights such that:
            ∂u/∂n ≈ Σ_j w_j u_j = 0

        Returns:
            Dictionary mapping boundary point index to (neighbor_indices, weights)
        """
        bc_weights = {}

        for i in self.boundary_indices:
            weights_data = self._gfdm_operator.get_derivative_weights(i)
            if weights_data is None:
                continue

            neighbor_indices = weights_data["neighbor_indices"]
            grad_weights = weights_data["grad_weights"]  # shape: (d, n_neighbors)

            # Get outward normal at this point
            normal = self._compute_outward_normal(i)

            # Normal derivative weights: ∂u/∂n = n · ∇u = Σ_d n_d (∂u/∂x_d)
            # = Σ_d n_d (Σ_j w^d_j u_j) = Σ_j (Σ_d n_d w^d_j) u_j
            normal_weights = np.zeros(len(neighbor_indices))
            for k in range(len(neighbor_indices)):
                for d in range(self.dimension):
                    normal_weights[k] += normal[d] * grad_weights[d, k]

            # Center contribution: sum rule requires center weight = -Σ_j neighbor_weights
            # But this is already handled in grad_weights via _build_differentiation_matrices
            # We need to extract the center weight separately
            center_weight = -np.sum(normal_weights[neighbor_indices >= 0])

            bc_weights[i] = (neighbor_indices, normal_weights, center_weight)

        return bc_weights

    def _build_neighborhood_structure(self):
        """
        Build δ-neighborhood structure for all collocation points.

        Uses GFDMOperator's neighborhoods (without ghost particles - pure one-sided stencils)
        and only extends for points needing adaptive delta enlargement.
        """
        self.neighborhoods = {}

        # For adaptive delta, we need pairwise distances
        if self.adaptive_neighborhoods:
            distances = cdist(self.collocation_points, self.collocation_points)
        else:
            distances = None

        for i in range(self.n_points):
            # Start with GFDMOperator's neighborhood (pure one-sided stencils, no ghost particles)
            base_neighborhood = self._gfdm_operator.get_neighborhood(i)
            n_neighbors = base_neighborhood["size"]

            # Check if adaptive delta enlargement is needed
            needs_adaptation = self.adaptive_neighborhoods and n_neighbors < self.k_min and distances is not None

            if needs_adaptation:
                # Adaptive delta enlargement for insufficient neighbors
                neighbor_indices = base_neighborhood["indices"].copy()
                neighbor_points = base_neighborhood["points"].copy()
                neighbor_distances = base_neighborhood["distances"].copy()

                # Only count real neighbors (not ghost particles) for k_min check
                real_neighbor_count = np.sum(neighbor_indices >= 0)

                delta_current = self.delta
                delta_multiplier = 1.0
                was_adapted = False
                max_delta = self.delta * self.max_delta_multiplier

                while real_neighbor_count < self.k_min and delta_current < max_delta:
                    # Enlarge delta by 20% increments
                    delta_multiplier *= 1.2
                    delta_current = self.delta * delta_multiplier

                    # Recompute neighborhood with enlarged delta
                    neighbor_mask = distances[i, :] < delta_current
                    neighbor_indices = np.where(neighbor_mask)[0]
                    real_neighbor_count = len(neighbor_indices)
                    was_adapted = True

                # Update neighborhood data for enlarged delta
                if was_adapted:
                    neighbor_points = self.collocation_points[neighbor_indices]
                    neighbor_distances = distances[i, neighbor_indices]

                    # Re-add ghost particles if this is a boundary point
                    if base_neighborhood.get("has_ghost", False):
                        ghost_particles, ghost_distances = self._gfdm_operator._create_ghost_particles(i)
                        if ghost_particles:
                            neighbor_points = np.vstack(
                                [neighbor_points] + [gp.reshape(1, -1) for gp in ghost_particles]
                            )
                            neighbor_distances = np.concatenate([neighbor_distances, np.array(ghost_distances)])
                            neighbor_indices = np.concatenate(
                                [neighbor_indices, np.array([-1 - j for j in range(len(ghost_particles))])]
                            )

                # Track maximum delta used
                if delta_current > self.adaptive_stats["max_delta_used"]:
                    self.adaptive_stats["max_delta_used"] = delta_current

                # Record adaptive enlargement
                if was_adapted:
                    self.adaptive_stats["n_adapted"] += 1
                    self.adaptive_stats["adaptive_enlargements"].append(
                        {
                            "point_idx": i,
                            "base_delta": self.delta,
                            "adapted_delta": delta_current,
                            "delta_multiplier": delta_multiplier,
                            "n_neighbors": len(neighbor_indices),
                        }
                    )

                # Warn if still insufficient neighbors
                if real_neighbor_count < self.k_min:
                    import warnings as _warnings

                    _warnings.warn(
                        f"Point {i}: Could not find {self.k_min} neighbors even with "
                        f"delta={delta_current:.4f} ({delta_multiplier:.2f}x base). "
                        f"Only found {real_neighbor_count} neighbors. GFDM approximation may be poor.",
                        UserWarning,
                        stacklevel=3,
                    )

                # Store adapted neighborhood
                self.neighborhoods[i] = {
                    "indices": np.array(neighbor_indices) if isinstance(neighbor_indices, list) else neighbor_indices,
                    "points": np.array(neighbor_points) if isinstance(neighbor_points, list) else neighbor_points,
                    "distances": np.array(neighbor_distances)
                    if isinstance(neighbor_distances, list)
                    else neighbor_distances,
                    "size": len(neighbor_indices),
                    "has_ghost": base_neighborhood.get("has_ghost", False),
                    "ghost_count": base_neighborhood.get("ghost_count", 0),
                    "adapted": True,
                }
            else:
                # Use GFDMOperator's neighborhood directly (no adaptation needed)
                self.neighborhoods[i] = {
                    "indices": base_neighborhood["indices"],
                    "points": base_neighborhood["points"],
                    "distances": base_neighborhood["distances"],
                    "size": base_neighborhood["size"],
                    "has_ghost": base_neighborhood.get("has_ghost", False),
                    "ghost_count": base_neighborhood.get("ghost_count", 0),
                    "adapted": False,
                }

        # Report adaptive neighborhood statistics if enabled
        if self.adaptive_neighborhoods:
            n_adapted = self.adaptive_stats["n_adapted"]
            if n_adapted > 0:
                pct_adapted = 100.0 * n_adapted / self.n_points
                avg_multiplier = np.mean([e["delta_multiplier"] for e in self.adaptive_stats["adaptive_enlargements"]])
                max_multiplier = np.max([e["delta_multiplier"] for e in self.adaptive_stats["adaptive_enlargements"]])

                import warnings

                warnings.warn(
                    f"Adaptive neighborhoods: {n_adapted}/{self.n_points} points ({pct_adapted:.1f}%) "
                    f"required delta enlargement. Base delta: {self.delta:.4f}, "
                    f"Max delta used: {self.adaptive_stats['max_delta_used']:.4f} "
                    f"({max_multiplier:.2f}x base), Avg multiplier: {avg_multiplier:.2f}x. "
                    f"Consider increasing base delta for better theoretical accuracy.",
                    UserWarning,
                    stacklevel=2,
                )

    def _build_reverse_neighborhoods(self) -> None:
        """
        Build reverse neighborhood map: for each point j, find all points i that have j in their neighborhood.

        This enables sparse Jacobian computation - when perturbing u[j], only residuals at
        points in reverse_neighborhoods[j] are affected.

        Complexity: O(n * k) where k is average neighborhood size.
        """
        self._reverse_neighborhoods: dict[int, list[int]] = {j: [] for j in range(self.n_points)}

        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            neighbor_indices = neighborhood["indices"]

            # For each neighbor j of point i, add i to j's reverse neighborhood
            for j in neighbor_indices:
                if 0 <= j < self.n_points:  # Exclude ghost particles (negative indices)
                    self._reverse_neighborhoods[j].append(i)

        # Convert to arrays for faster access
        self._reverse_neighborhoods = {j: np.array(rows, dtype=int) for j, rows in self._reverse_neighborhoods.items()}

    def _get_affected_rows(self, j: int) -> np.ndarray:
        """
        Get rows affected by perturbing u[j] for sparse Jacobian.

        Perturbing u[j] affects:
        1. Row j (the point itself - its residual depends on its own value)
        2. All rows i where j is in the neighborhood of i (j affects i's derivative approximation)

        Returns:
            Array of row indices affected by u[j]
        """
        # Get points that have j in their neighborhood
        affected = self._reverse_neighborhoods.get(j, np.array([], dtype=int))

        # Always include j itself (residual at j depends on u[j])
        if j not in affected:
            affected = np.append(affected, j)

        return affected

    def _build_taylor_matrices(self):
        """
        Pre-compute Taylor expansion matrices A for all collocation points.

        Uses GFDMOperator's Taylor matrices when the neighborhood wasn't adapted,
        only rebuilds for points that needed adaptive delta enlargement.
        """
        self.taylor_matrices: dict[int, np.ndarray] = {}

        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            n_neighbors_raw = neighborhood["size"]
            n_neighbors = int(n_neighbors_raw) if isinstance(n_neighbors_raw, int | float) else 0

            if n_neighbors < self.n_derivatives:
                self.taylor_matrices[i] = None
                continue

            # Check if we can reuse GFDMOperator's Taylor matrices
            # (only if neighborhood wasn't adapted)
            if not neighborhood.get("adapted", False):
                base_taylor = self._gfdm_operator.get_taylor_data(i)
                if base_taylor is not None:
                    # Compute condition number safely
                    S = base_taylor["S"]
                    if len(S) > 0 and S[-1] > 1e-15:
                        cond_num = S[0] / S[-1]
                    else:
                        cond_num = np.inf

                    # Wrap GFDMOperator's format to HJBGFDMSolver's expected format
                    self.taylor_matrices[i] = {  # type: ignore[assignment]
                        "A": base_taylor["A"],
                        "W": base_taylor["W"],
                        "sqrt_W": base_taylor["sqrt_W"],
                        "U": base_taylor["U"],
                        "S": S,
                        "Vt": base_taylor["Vt"],
                        "rank": base_taylor["rank"],
                        "condition_number": cond_num,
                        "use_svd": True,
                        "use_qr": False,
                    }
                    continue

            # Need to build Taylor matrices for adapted neighborhoods
            # Build Taylor expansion matrix A
            A = np.zeros((n_neighbors, self.n_derivatives))
            center_point = self.collocation_points[i]
            neighbor_points = neighborhood["points"]

            for j, neighbor_point in enumerate(neighbor_points):  # type: ignore[var-annotated,arg-type]
                delta_x = neighbor_point - center_point

                for k, beta in enumerate(self.multi_indices):
                    # Compute (x - x_center)^β / β!
                    term = 1.0
                    factorial = 1.0

                    for dim in range(self.dimension):
                        if beta[dim] > 0:
                            term *= delta_x[dim] ** beta[dim]
                            factorial *= math.factorial(beta[dim])

                    A[j, k] = term / factorial

            # Compute weights and store matrices
            weights = self._compute_weights(np.asarray(neighborhood["distances"]))
            W = np.diag(weights)

            # Use SVD or QR decomposition to avoid condition number amplification
            sqrt_W = np.sqrt(W)
            WA = sqrt_W @ A

            # Try SVD first (most robust)
            try:
                # SVD decomposition: WA = U @ S @ V^T
                U, S, Vt = np.linalg.svd(WA, full_matrices=False)

                # Condition number check and regularization
                condition_number = S[0] / S[-1] if S[-1] > 1e-15 else np.inf

                # Truncate small singular values for regularization
                tolerance = 1e-12
                rank = np.sum(tolerance < S)

                self.taylor_matrices[i] = {  # type: ignore[assignment]
                    "A": A,
                    "W": W,
                    "sqrt_W": sqrt_W,
                    "U": U[:, :rank],
                    "S": S[:rank],
                    "Vt": Vt[:rank, :],
                    "rank": rank,
                    "condition_number": condition_number,
                    "use_svd": True,
                    "use_qr": False,
                }

            except np.linalg.LinAlgError:
                # Fallback to QR decomposition if SVD fails
                try:
                    # QR decomposition: WA = Q @ R
                    Q, R = np.linalg.qr(WA)
                    self.taylor_matrices[i] = {  # type: ignore[assignment]
                        "A": A,
                        "W": W,
                        "sqrt_W": sqrt_W,
                        "Q": Q,
                        "R": R,
                        "use_qr": True,
                        "use_svd": False,
                    }
                except np.linalg.LinAlgError:
                    # Final fallback to normal equations if QR also fails
                    self.taylor_matrices[i] = {  # type: ignore[assignment]
                        "A": A,
                        "W": W,
                        "AtW": A.T @ W,
                        "AtWA_inv": None,
                        "use_qr": False,
                        "use_svd": False,
                    }

                    try:
                        AtWA = A.T @ W @ A
                        if np.linalg.det(AtWA) > 1e-12:
                            self.taylor_matrices[i]["AtWA_inv"] = np.linalg.inv(AtWA)
                    except (np.linalg.LinAlgError, FloatingPointError):
                        # Cannot compute inverse - leave AtWA_inv as None
                        pass

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute weights based on distance and weight function using smoothing kernels.

        Uses the unified kernel API from mfg_pde.utils.numerical.kernels.
        """
        from mfg_pde.utils.numerical.kernels import (
            GaussianKernel,
            WendlandKernel,
        )

        if self.weight_function == "gaussian":
            # Use GaussianKernel with smoothing length = weight_scale
            kernel = GaussianKernel()
            return kernel(distances, h=self.weight_scale)

        elif self.weight_function == "inverse_distance":
            # Keep legacy inverse distance weights (not a standard kernel)
            return 1.0 / (distances + 1e-12)

        elif self.weight_function == "uniform":
            # Keep legacy uniform weights (trivial case)
            return np.ones_like(distances)

        elif self.weight_function == "wendland":
            # Use Wendland C^4 kernel: (1 - r/h)_+^6 (35q² + 18q + 3)
            kernel = WendlandKernel(k=2)  # k=2 → C^4 continuity
            # Support radius = delta (neighborhood size)
            return kernel(distances, h=self.delta)

        else:
            raise ValueError(
                f"Unknown weight function: {self.weight_function}. "
                f"Available options: 'gaussian', 'inverse_distance', 'uniform', 'wendland'"
            )

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

        for i in range(n):
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

        # Extract function values at neighborhood points, handling ghost particles
        neighbor_indices = neighborhood["indices"]
        u_center = u_values[point_idx]

        # Handle ghost particles based on BC type
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
                # Ghost particle: value depends on BC type
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

        # For GFDM, we work directly with collocation points
        # Map grid data to collocation points
        # Output shape: (n_time_points, n_points) - same as input
        U_solution_collocation = np.zeros((n_time_points, self.n_points))
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

        # Map back to grid
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

        # Fallback: finite differences for custom Hamiltonians
        x_pos = self.collocation_points[point_idx]

        def H_wrapper(x_idx, m, derivs):
            return self.problem.H(x_idx, m, derivs=derivs, x_position=x_pos)

        hess = derivs.hess

        return compute_dH_dp(
            H_func=H_wrapper,
            x_idx=point_idx,
            m=m_at_x,
            p=p,
            hess=hess,
        )

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

    def _map_grid_to_collocation(self, u_grid: np.ndarray) -> np.ndarray:
        """
        Map values from regular grid to collocation points using interpolation.

        Uses scipy RegularGridInterpolator via particle interpolation utilities.

        Args:
            u_grid: Values on regular grid, flattened to 1D

        Returns:
            Values at collocation points, shape (n_points,)
        """
        # Get spatial shape from stored attribute or infer from grid size
        if hasattr(self, "_output_spatial_shape"):
            spatial_shape = self._output_spatial_shape
        else:
            # Assume 1D if not set
            spatial_shape = (len(u_grid),)

        # Reshape to spatial dimensions
        u_grid_shaped = u_grid.reshape(spatial_shape)

        # Squeeze singleton dimensions for proper interpolation
        # e.g., (21, 1) -> (21,) for 1D problems
        u_grid_squeezed = np.squeeze(u_grid_shaped)

        # Format bounds for interpolation utility (matches squeezed dimensions)
        grid_bounds = self._format_grid_bounds_for_interpolation()

        return interpolate_grid_to_particles(
            u_grid_squeezed,
            grid_bounds=grid_bounds,
            particle_positions=self.collocation_points,
            method="linear",
        )

    def _map_collocation_to_grid(self, u_collocation: np.ndarray) -> np.ndarray:
        """
        Map values from collocation points to regular grid using cached interpolation.

        Uses pre-computed interpolation matrix for efficiency (O(n*m) vs O(n^3)).

        Args:
            u_collocation: Values at collocation points, shape (n_points,)

        Returns:
            Values on regular grid, flattened to 1D
        """
        # Use cached interpolation matrix if available
        if hasattr(self, "_interp_matrix") and self._interp_matrix is not None:
            return self._interp_matrix @ u_collocation

        # Fallback: build interpolation matrix on first call
        self._build_interpolation_matrix()
        return self._interp_matrix @ u_collocation

    def _build_interpolation_matrix(self) -> None:
        """
        Pre-compute interpolation matrix for collocation->grid mapping.

        Uses scipy LinearNDInterpolator with barycentric weights for efficiency.
        The matrix is sparse due to triangulation locality.
        """
        from scipy.spatial import Delaunay

        # Get spatial shape
        if hasattr(self, "_output_spatial_shape"):
            spatial_shape = self._output_spatial_shape
        else:
            Nx_grid = self._n_spatial_grid_points
            spatial_shape = (Nx_grid + 1,)

        non_singleton = tuple(s for s in spatial_shape if s > 1)
        squeezed_shape = non_singleton if non_singleton else (spatial_shape[0],)

        grid_bounds = self._format_grid_bounds_for_interpolation()
        ndim = len(squeezed_shape)

        # Create grid points
        if ndim == 1:
            xmin, xmax = grid_bounds if not isinstance(grid_bounds[0], (tuple, list)) else grid_bounds[0]
            x = np.linspace(xmin, xmax, squeezed_shape[0])
            grid_points = x.reshape(-1, 1)
        elif ndim == 2:
            (xmin, xmax), (ymin, ymax) = grid_bounds
            x = np.linspace(xmin, xmax, squeezed_shape[0])
            y = np.linspace(ymin, ymax, squeezed_shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")
            grid_points = np.column_stack([X.ravel(), Y.ravel()])
        else:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = grid_bounds
            x = np.linspace(xmin, xmax, squeezed_shape[0])
            y = np.linspace(ymin, ymax, squeezed_shape[1])
            z = np.linspace(zmin, zmax, squeezed_shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        n_grid = grid_points.shape[0]
        n_coll = self.n_points

        # Build interpolation matrix using Delaunay triangulation
        # Each grid point is expressed as weighted sum of collocation points
        try:
            tri = Delaunay(self.collocation_points)
            simplex_indices = tri.find_simplex(grid_points)

            # Build sparse matrix
            from scipy.sparse import lil_matrix

            interp_matrix = lil_matrix((n_grid, n_coll))

            for i, (pt, simplex_idx) in enumerate(zip(grid_points, simplex_indices, strict=True)):
                if simplex_idx >= 0:
                    # Point inside triangulation - use barycentric coords
                    simplex = tri.simplices[simplex_idx]
                    b = tri.transform[simplex_idx, :ndim].dot(pt - tri.transform[simplex_idx, ndim])
                    bary = np.append(b, 1 - b.sum())
                    for j, weight in zip(simplex, bary, strict=True):
                        interp_matrix[i, j] = weight
                else:
                    # Point outside - use nearest neighbor
                    from scipy.spatial import cKDTree

                    if not hasattr(self, "_coll_tree"):
                        self._coll_tree = cKDTree(self.collocation_points)
                    _, nearest_idx = self._coll_tree.query(pt)
                    interp_matrix[i, nearest_idx] = 1.0

            self._interp_matrix = interp_matrix.tocsr()
        except Exception:
            # Fallback: use dense nearest-neighbor matrix
            from scipy.spatial import cKDTree

            tree = cKDTree(self.collocation_points)
            _, nearest_indices = tree.query(grid_points)
            interp_matrix = np.zeros((n_grid, n_coll))
            interp_matrix[np.arange(n_grid), nearest_indices] = 1.0
            self._interp_matrix = interp_matrix

    def _format_grid_bounds_for_interpolation(self) -> tuple:
        """Format domain_bounds for interpolation utility."""
        if len(self.domain_bounds) == 1:
            # 1D: return (xmin, xmax)
            return self.domain_bounds[0]
        else:
            # nD: return tuple of (min, max) tuples
            return tuple(self.domain_bounds)

    def _map_grid_to_collocation_batch(self, U_grid: np.ndarray) -> np.ndarray:
        """
        Batch version of _map_grid_to_collocation.

        Args:
            U_grid: Shape (Nt, ...) where ... represents spatial dimensions

        Returns:
            U_collocation: Shape (Nt, n_points)
        """
        Nt = U_grid.shape[0]
        U_collocation = np.zeros((Nt, self.n_points))

        for n in range(Nt):
            u_grid_flat = U_grid[n].flatten()
            U_collocation[n, :] = self._map_grid_to_collocation(u_grid_flat)

        return U_collocation

    def _map_collocation_to_grid_batch(self, U_collocation: np.ndarray) -> np.ndarray:
        """
        Batch version of _map_collocation_to_grid.

        Args:
            U_collocation: Shape (Nt, n_points)

        Returns:
            U_grid: Shape (Nt, ...) where ... matches original spatial shape
        """
        Nt = U_collocation.shape[0]

        # Get the original spatial shape
        if hasattr(self, "_output_spatial_shape"):
            spatial_shape = self._output_spatial_shape
        else:
            Nx_grid = self._n_spatial_grid_points
            spatial_shape = (Nx_grid + 1,)

        n_spatial_points = int(np.prod(spatial_shape))

        # Map each timestep
        U_grid_flat = np.zeros((Nt, n_spatial_points))
        for n in range(Nt):
            U_grid_flat[n, :] = self._map_collocation_to_grid(U_collocation[n, :])

        # Reshape to original spatial dimensions
        output_shape = (Nt, *tuple(spatial_shape))
        return U_grid_flat.reshape(output_shape)


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
