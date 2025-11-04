from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist

from .base_hjb import BaseHJBSolver

# Optional QP solver imports
CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None
OSQP_AVAILABLE = importlib.util.find_spec("osqp") is not None

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions
    from mfg_pde.types.solver_types import MultiIndexTuple


class HJBGFDMSolver(BaseHJBSolver):
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
        # Adaptive neighborhood parameters
        adaptive_neighborhoods: bool = False,
        k_min: int | None = None,
        max_delta_multiplier: float = 5.0,
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
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else {}
        self.interior_indices = np.setdiff1d(np.arange(self.n_points), self.boundary_indices)

        # QP optimization level (single source of truth for QP control)
        self.qp_optimization_level = qp_optimization_level

        # QP usage target (deprecated, kept for backward compatibility)
        self.qp_usage_target = qp_usage_target

        # QP solver selection
        self.qp_solver = qp_solver
        self.qp_warm_start = qp_warm_start

        # Warm-start cache: stores previous QP solutions per point
        # Key: point_idx, Value: (solution vector, dual variables)
        self._qp_warm_start_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

        # Initialize QP diagnostic statistics
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

        # Pre-compute GFDM structure
        self._build_neighborhood_structure()
        self._build_taylor_matrices()

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

    def _build_neighborhood_structure(self):
        """Build δ-neighborhood structure for all collocation points with ghost particles for boundaries."""
        self.neighborhoods = {}

        # Compute pairwise distances
        distances = cdist(self.collocation_points, self.collocation_points)

        for i in range(self.n_points):
            # Find neighbors within delta radius (with adaptive enlargement if enabled)
            delta_current = self.delta
            delta_multiplier = 1.0
            was_adapted = False

            # Standard delta-radius neighborhood
            neighbor_mask = distances[i, :] < delta_current
            neighbor_indices = np.where(neighbor_mask)[0]
            n_neighbors = len(neighbor_indices)

            # Adaptive delta enlargement if enabled and insufficient neighbors
            if self.adaptive_neighborhoods and n_neighbors < self.k_min:
                max_delta = self.delta * self.max_delta_multiplier

                while n_neighbors < self.k_min and delta_current < max_delta:
                    # Enlarge delta by 20% increments
                    delta_multiplier *= 1.2
                    delta_current = self.delta * delta_multiplier

                    # Recompute neighborhood
                    neighbor_mask = distances[i, :] < delta_current
                    neighbor_indices = np.where(neighbor_mask)[0]
                    n_neighbors = len(neighbor_indices)
                    was_adapted = True

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
                            "n_neighbors": n_neighbors,
                        }
                    )

                # Warn if still insufficient neighbors
                if n_neighbors < self.k_min:
                    import warnings

                    warnings.warn(
                        f"Point {i}: Could not find {self.k_min} neighbors even with "
                        f"delta={delta_current:.4f} ({delta_multiplier:.2f}x base). "
                        f"Only found {n_neighbors} neighbors. GFDM approximation may be poor.",
                        UserWarning,
                        stacklevel=3,
                    )

            # Extract final neighborhood
            neighbor_points = self.collocation_points[neighbor_indices]
            neighbor_distances = distances[i, neighbor_indices]

            # Check if this is a boundary point and add ghost particles if needed
            is_boundary_point = i in self.boundary_indices
            ghost_particles = []
            ghost_distances = []

            if (
                is_boundary_point
                and hasattr(self.boundary_conditions, "type")
                and getattr(self.boundary_conditions, "type", None) == "no_flux"
            ):
                # Add ghost particles for no-flux boundary conditions
                current_point = self.collocation_points[i]

                # For 1D case, add ghost particles beyond boundaries
                if self.dimension == 1:
                    x = current_point[0]

                    # Check if near left boundary
                    if abs(x - self.problem.xmin) < 1e-10:
                        # Add ghost particle symmetrically reflected across left boundary
                        # For a point at x=0, place ghost at x = -h where h is a small distance
                        h = 0.1 * self.delta  # Distance from boundary
                        ghost_x = self.problem.xmin - h
                        ghost_point = np.array([ghost_x])
                        ghost_distance = h

                        if ghost_distance < self.delta:
                            ghost_particles.append(ghost_point)
                            ghost_distances.append(ghost_distance)

                    # Check if near right boundary
                    if abs(x - self.problem.xmax) < 1e-10:
                        # Add ghost particle symmetrically reflected across right boundary
                        # For a point at x=1, place ghost at x = 1+h
                        h = 0.1 * self.delta  # Distance from boundary
                        ghost_x = self.problem.xmax + h
                        ghost_point = np.array([ghost_x])
                        ghost_distance = h

                        if ghost_distance < self.delta:
                            ghost_particles.append(ghost_point)
                            ghost_distances.append(ghost_distance)

            # Combine regular neighbors and ghost particles
            all_points = list(neighbor_points)
            all_distances = list(neighbor_distances)
            all_indices = list(neighbor_indices)

            for j, ghost_point in enumerate(ghost_particles):
                all_points.append(ghost_point)
                all_distances.append(ghost_distances[j])
                all_indices.append(-1 - j)  # Negative indices for ghost particles

            # Store neighborhood information
            self.neighborhoods[i] = {
                "indices": np.array(all_indices),
                "points": np.array(all_points),
                "distances": np.array(all_distances),
                "size": len(all_points),
                "has_ghost": len(ghost_particles) > 0,
                "ghost_count": len(ghost_particles),
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

    def _get_multi_index_set(self, d: int, p: int) -> list[MultiIndexTuple]:
        """Generate multi-index set B(d,p) = {β ∈ N^d : 0 < |β| ≤ p} with lexicographical ordering."""
        multi_indices = []

        def generate_indices(current_index, remaining_dims, remaining_order):
            if remaining_dims == 0:
                if sum(current_index) > 0 and sum(current_index) <= p:
                    multi_indices.append(tuple(current_index))
                return

            for i in range(remaining_order + 1):
                generate_indices([*current_index, i], remaining_dims - 1, remaining_order - i)

        generate_indices([], d, p)

        # Sort by lexicographical ordering: first by order |β|, then lexicographically
        multi_indices.sort(key=lambda beta: (sum(beta), beta))
        return multi_indices

    def _build_taylor_matrices(self):
        """Pre-compute Taylor expansion matrices A for all collocation points."""
        self.taylor_matrices: dict[int, np.ndarray] = {}
        self.multi_indices = self._get_multi_index_set(self.dimension, self.taylor_order)
        self.n_derivatives = len(self.multi_indices)

        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            n_neighbors_raw = neighborhood["size"]
            n_neighbors = int(n_neighbors_raw) if isinstance(n_neighbors_raw, int | float) else 0

            if n_neighbors < self.n_derivatives:
                self.taylor_matrices[i] = None
                continue

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
            # Instead of forming A^T W A, use SVD or QR decomposition of sqrt(W) @ A
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

        Uses the unified smoothing kernel API from mfg_pde.utils.numerical.smoothing_kernels.
        """
        from mfg_pde.utils.numerical.smoothing_kernels import (
            GaussianKernel,
            WendlandC4Kernel,
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
            # Use WendlandC4Kernel: (1 - r/h)_+^6 (35q² + 18q + 3)
            kernel = WendlandC4Kernel()
            # Support radius = delta (neighborhood size)
            return kernel(distances, h=self.delta)

        else:
            raise ValueError(
                f"Unknown weight function: {self.weight_function}. "
                f"Available options: 'gaussian', 'inverse_distance', 'uniform', 'wendland'"
            )

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

        if self.taylor_matrices[point_idx] is None:
            return {}

        taylor_data = self.taylor_matrices[point_idx]
        neighborhood = self.neighborhoods[point_idx]

        # Extract function values at neighborhood points, handling ghost particles
        neighbor_indices = neighborhood["indices"]
        u_center = u_values[point_idx]

        # Handle ghost particles for no-flux boundary conditions
        u_neighbors = []
        for idx in neighbor_indices:  # type: ignore[attr-defined]
            if idx >= 0:
                # Regular neighbor
                u_neighbors.append(u_values[idx])
            else:
                # Ghost particle: enforce no-flux condition u_ghost = u_center
                u_neighbors.append(u_center)

        u_neighbors = np.array(u_neighbors)  # type: ignore[assignment]

        # Right-hand side: u(x_neighbor) - u(x_center) for Taylor expansion
        # u(x_j) - u(x_0) ≈ ∇u·(x_j - x_0) where A matrix uses (x_j - x_0)
        # For ghost particles: u_ghost = u_center → b = 0, enforcing ∂u/∂n = 0
        b = u_neighbors - u_center

        # Solve weighted least squares with optional monotonicity constraints
        # Single source of truth: qp_optimization_level
        qp_level = getattr(self, "qp_optimization_level", "none")

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

        # Map coefficients to multi-indices
        derivatives = {}
        for k, beta in enumerate(self.multi_indices):
            derivatives[beta] = derivative_coeffs[k]

        return derivatives

    def _solve_monotone_constrained_qp(self, taylor_data: dict, b: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Solve constrained quadratic programming problem for monotone derivative approximation.

        Solves: min ||W^(1/2) A x - W^(1/2) b||^2
        subject to: monotonicity constraints on finite difference weights

        Args:
            taylor_data: Dictionary containing precomputed matrices
            b: Right-hand side vector
            point_idx: Index of collocation point (for diagnostics)

        Returns:
            derivative_coeffs: Coefficients for derivative approximation
        """
        import time

        t0 = time.time()

        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fallback to unconstrained if scipy not available
            self.qp_stats["qp_fallbacks"] += 1
            elapsed = time.time() - t0
            self.qp_stats["qp_times"].append(elapsed)
            return self._solve_unconstrained_fallback(taylor_data, b)

        A = taylor_data["A"]
        W = taylor_data["W"]
        sqrt_W = taylor_data["sqrt_W"]
        _n_neighbors, _n_coeffs = A.shape

        # Define objective function: ||W^(1/2) A x - W^(1/2) b||^2
        def objective(x):
            residual = sqrt_W @ (A @ x - b)
            return 0.5 * np.dot(residual, residual)

        # Define gradient of objective
        def gradient(x):
            residual = A @ x - b
            return A.T @ W @ residual

        # Define Hessian of objective
        def hessian(x):
            return A.T @ W @ A

        # Analyze the stencil structure to determine appropriate constraints
        center_point = self.collocation_points[point_idx]
        neighborhood = self.neighborhoods[point_idx]
        neighbor_points = neighborhood["points"]
        neighbor_indices = neighborhood["indices"]

        # Set up constraints for monotonicity
        constraints = []

        # Implement proper monotonicity constraints for finite difference weights
        if self.dimension == 1:
            # For 1D, we can analyze the finite difference stencil more precisely
            monotonicity_constraints = self._build_monotonicity_constraints(
                A,
                neighbor_indices,  # type: ignore[arg-type]
                neighbor_points,  # type: ignore[arg-type]
                center_point,
                point_idx,
            )
            constraints.extend(monotonicity_constraints)

        # Set up bounds for optimization variables
        bounds = []

        # For each coefficient, determine appropriate bounds based on physics
        # Make bounds much more realistic for stable numerical computation
        for _k, beta in enumerate(self.multi_indices):
            if sum(beta) == 0:  # Constant term - no physical constraint
                bounds.append((None, None))
            elif sum(beta) == 1:  # First derivative terms - reasonable for MFG
                bounds.append((-20.0, 20.0))  # type: ignore[arg-type]  # Realistic gradient bounds
            elif sum(beta) == 2:  # Second derivative terms - key for monotonicity
                if self.dimension == 1 and beta == (2,):
                    # For 1D Laplacian: moderate diffusion bounds
                    bounds.append((-100.0, 100.0))  # type: ignore[arg-type]  # Realistic diffusion bounds
                else:
                    bounds.append((-50.0, 50.0))  # type: ignore[arg-type]  # Conservative cross-derivative bounds
            else:
                bounds.append((-2.0, 2.0))  # type: ignore[arg-type]  # Tight bounds for higher order terms

        # Only add monotonicity constraints when they are really needed
        # Check if this point is near boundaries or critical regions
        center_point = self.collocation_points[point_idx]
        near_boundary = (
            abs(center_point[0] - self.problem.xmin) < 0.1 * self.delta
            or abs(center_point[0] - self.problem.xmax) < 0.1 * self.delta
        )

        # Add conservative constraint only if near boundary and needed
        if near_boundary and self.dimension == 1:

            def constraint_stability(x):
                """Mild stability constraint near boundaries"""
                # Ensure second derivative term doesn't become extreme
                for k, beta in enumerate(self.multi_indices):
                    if sum(beta) == 2 and beta == (2,):
                        return 50.0 - abs(x[k])  # Should be positive (|coeff| < 50)
                return 1.0  # Always satisfied if no second derivative

            constraints.append({"type": "ineq", "fun": constraint_stability})

        # Initial guess: unconstrained solution
        x0 = self._solve_unconstrained_fallback(taylor_data, b)

        # Try OSQP first if requested
        if self.qp_solver == "osqp":
            try:
                # Compute P and q for OSQP formulation
                # P = A'WA (Hessian), q = -A'Wb (linear term)
                P = A.T @ W @ A
                q = -A.T @ W @ b

                # Call OSQP solver
                result_x = self._solve_qp_with_osqp(P, q, bounds, constraints, x0, point_idx)

                # Track statistics
                self.qp_stats["total_qp_solves"] += 1
                self.qp_stats["osqp_solves"] += 1
                self.qp_stats["qp_successes"] += 1
                elapsed = time.time() - t0
                self.qp_stats["qp_times"].append(elapsed)

                return result_x

            except ImportError:
                # OSQP not available - fall back to scipy
                import warnings

                warnings.warn("OSQP not available, falling back to scipy", stacklevel=2)
                self.qp_solver = "scipy"  # Switch permanently

            except (RuntimeError, Exception):
                # OSQP failed - fall back to scipy for this solve
                self.qp_stats["osqp_failures"] += 1
                # Continue to scipy solver below

        # Solve constrained optimization with robust settings (scipy fallback)
        if self.qp_solver == "scipy":
            try:
                # Try fast L-BFGS-B first if only bounds constraints
                if len(constraints) == 0:
                    self.qp_stats["lbfgsb_solves"] += 1
                    result = minimize(
                        objective,
                        x0,
                        method="L-BFGS-B",  # Faster for bounds-only problems
                        jac=gradient,
                        bounds=bounds,
                        options={
                            "maxiter": 50,
                            "ftol": 1e-6,
                            "gtol": 1e-6,
                        },  # More robust settings
                    )
                else:
                    # Use SLSQP for general constraints with robust settings
                    self.qp_stats["slsqp_solves"] += 1
                    result = minimize(
                        objective,
                        x0,
                        method="SLSQP",  # Better for equality/inequality constraints
                        jac=gradient,
                        bounds=bounds,
                        constraints=constraints,
                        options={
                            "maxiter": 40,
                            "ftol": 1e-6,
                            "eps": 1.4901161193847656e-08,
                            "disp": False,
                        },
                    )

                # Track QP solve statistics
                self.qp_stats["total_qp_solves"] += 1
                elapsed = time.time() - t0
                self.qp_stats["qp_times"].append(elapsed)

                if result.success:
                    self.qp_stats["qp_successes"] += 1
                    return result.x
                else:
                    # Fallback to unconstrained if optimization fails
                    self.qp_stats["qp_failures"] += 1
                    self.qp_stats["qp_fallbacks"] += 1
                    return x0

            except Exception:
                # Fallback to unconstrained if any error occurs
                self.qp_stats["qp_fallbacks"] += 1
                elapsed = time.time() - t0
                self.qp_stats["qp_times"].append(elapsed)
                return x0

    def _solve_qp_with_osqp(
        self,
        P: np.ndarray,
        q: np.ndarray,
        bounds: list,
        monotonicity_constraints: list,
        x0: np.ndarray,
        point_idx: int,
    ) -> np.ndarray:
        """
        Solve QP using OSQP solver for improved performance.

        Solves: minimize (1/2) x' P x + q' x
                subject to l <= A x <= u

        Args:
            P: Quadratic cost matrix (Hessian)
            q: Linear cost vector
            bounds: List of (lower, upper) bound tuples for each variable
            monotonicity_constraints: List of constraint dicts (scipy format)
            x0: Initial guess (fallback if warm-start not available)
            point_idx: Collocation point index (for warm-start caching)

        Returns:
            Solution vector x

        Raises:
            ImportError: If OSQP not available
            RuntimeError: If OSQP solve fails
        """
        import osqp

        import scipy.sparse as sp

        n = len(x0)

        # Build constraint matrix and bounds
        # Start with variable bounds (identity matrix rows)
        constraint_rows = []
        l_bounds = []
        u_bounds = []

        for i, (lb, ub) in enumerate(bounds):
            # Add identity row for this variable's bounds
            row = np.zeros(n)
            row[i] = 1.0
            constraint_rows.append(row)
            l_bounds.append(lb if lb is not None else -np.inf)
            u_bounds.append(ub if ub is not None else np.inf)

        # Add monotonicity constraints (linearized if needed)
        # For now, use Option A: bounds-only (skip nonlinear constraints)
        # This is sufficient for most cases and much simpler

        if constraint_rows:
            A_constraint = sp.csc_matrix(np.vstack(constraint_rows))
            lower_bounds = np.array(l_bounds)
            upper_bounds = np.array(u_bounds)
        else:
            # No constraints
            A_constraint = sp.csc_matrix((0, n))
            lower_bounds = np.array([])
            upper_bounds = np.array([])

        # Convert P to sparse (CSC format for OSQP)
        P_sparse = sp.csc_matrix(P)

        # Set up OSQP problem
        prob = osqp.OSQP()
        prob.setup(
            P=P_sparse,
            q=q,
            A=A_constraint,
            l=lower_bounds,
            u=upper_bounds,
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10000,
            polish=False,  # Disabled: polishing prints verbose messages even with verbose=False
        )

        # Apply warm-starting if enabled and cached solution available
        if self.qp_warm_start and point_idx in self._qp_warm_start_cache:
            x_prev, y_prev = self._qp_warm_start_cache[point_idx]
            # Warm-start with previous solution
            # OSQP uses both primal (x) and dual (y) variables for warm-starting
            if y_prev is not None and len(y_prev) == len(lower_bounds):
                prob.warm_start(x=x_prev, y=y_prev)
            else:
                # Only primal warm-start if dual not compatible
                prob.warm_start(x=x_prev)

        # Solve
        result = prob.solve()

        # Check status
        if result.info.status != "solved":
            raise RuntimeError(f"OSQP failed with status: {result.info.status}")

        # Cache solution for next warm-start
        if self.qp_warm_start:
            self._qp_warm_start_cache[point_idx] = (result.x.copy(), result.y.copy())

        return result.x

    def _solve_unconstrained_fallback(self, taylor_data: dict, b: np.ndarray) -> np.ndarray:
        """Fallback to unconstrained solution using SVD or normal equations."""
        if taylor_data.get("use_svd", False):
            sqrt_W = taylor_data["sqrt_W"]
            U = taylor_data["U"]
            S = taylor_data["S"]
            Vt = taylor_data["Vt"]

            Wb = sqrt_W @ b
            UT_Wb = U.T @ Wb
            S_inv_UT_Wb = UT_Wb / S
            return Vt.T @ S_inv_UT_Wb
        elif taylor_data.get("AtWA_inv") is not None:
            return taylor_data["AtWA_inv"] @ taylor_data["AtW"] @ b
        else:
            A = taylor_data["A"]
            from scipy.linalg import lstsq

            if A is not None and b is not None:
                lstsq_result = lstsq(A, b)
                coeffs = lstsq_result[0] if lstsq_result is not None else np.zeros(len(b))
            else:
                coeffs = np.zeros(len(b) if b is not None else 1)
            return coeffs

    def print_qp_diagnostics(self) -> None:
        """
        Print comprehensive QP diagnostic statistics.

        Reports QP solve counts, timings, success rates, and solver usage.
        Useful for understanding QP performance and bottlenecks.
        """
        if self.qp_stats is None or not self.qp_stats.get("total_qp_solves", 0):
            print("\nQP Diagnostics: No QP solves recorded")
            return

        print("\n" + "=" * 80)
        print(f"QP DIAGNOSTICS - {self.hjb_method_name}")
        print("=" * 80)

        # Basic counts
        total_solves = self.qp_stats["total_qp_solves"]
        print("\nQP Solve Summary:")
        print(f"  Total QP solves:        {total_solves}")
        print(
            f"  Successful solves:      {self.qp_stats['qp_successes']} ({100 * self.qp_stats['qp_successes'] / max(total_solves, 1):.1f}%)"
        )
        print(
            f"  Failed solves:          {self.qp_stats['qp_failures']} ({100 * self.qp_stats['qp_failures'] / max(total_solves, 1):.1f}%)"
        )
        print(f"  Fallbacks:              {self.qp_stats['qp_fallbacks']}")

        # M-matrix checking (for "auto" level)
        if self.qp_stats["points_checked"] > 0:
            print("\nM-Matrix Violation Detection ('auto' level):")
            print(f"  Points checked:         {self.qp_stats['points_checked']}")
            print(
                f"  Violations detected:    {self.qp_stats['violations_detected']} ({100 * self.qp_stats['violations_detected'] / max(self.qp_stats['points_checked'], 1):.1f}%)"
            )

        # Solver breakdown
        osqp = self.qp_stats.get("osqp_solves", 0)
        slsqp = self.qp_stats["slsqp_solves"]
        lbfgsb = self.qp_stats["lbfgsb_solves"]
        print("\nSolver Usage:")
        print(f"  OSQP:                   {osqp} ({100 * osqp / max(total_solves, 1):.1f}%)")
        if self.qp_stats.get("osqp_failures", 0) > 0:
            print(f"    OSQP failures:        {self.qp_stats['osqp_failures']}")
        print(f"  scipy (SLSQP):          {slsqp} ({100 * slsqp / max(total_solves, 1):.1f}%)")
        print(f"  scipy (L-BFGS-B):       {lbfgsb} ({100 * lbfgsb / max(total_solves, 1):.1f}%)")

        # Timing statistics
        if self.qp_stats["qp_times"]:
            import numpy as np

            times = np.array(self.qp_stats["qp_times"])
            print("\nQP Solve Timing:")
            print(f"  Total time:             {np.sum(times):.2f} s")
            print(f"  Mean time per solve:    {np.mean(times) * 1000:.2f} ms")
            print(f"  Median time per solve:  {np.median(times) * 1000:.2f} ms")
            print(f"  Min time per solve:     {np.min(times) * 1000:.2f} ms")
            print(f"  Max time per solve:     {np.max(times) * 1000:.2f} ms")
            print(f"  Std dev:                {np.std(times) * 1000:.2f} ms")

        print("=" * 80 + "\n")

    def _compute_fd_weights_from_taylor(self, taylor_data: dict, derivative_idx: int) -> np.ndarray | None:
        """
        Compute finite difference weights for a specific derivative.

        For GFDM with weighted least squares, given:
        - A: Taylor expansion matrix [n_neighbors, n_derivs]
        - W: Weight matrix [n_neighbors, n_neighbors]
        - We solve: min ||sqrt(W) @ (A @ D - b)||^2 to get D from b

        To get weights w such that D^β = w @ b (where b = u_center - u_neighbors):
        We need the β-th row of the solution operator (A^T W A)^{-1} A^T W

        Args:
            taylor_data: Precomputed Taylor matrices
            derivative_idx: Index of derivative in multi_indices

        Returns:
            w: Array of finite difference weights [n_neighbors]
                or None if computation fails
        """
        try:
            if taylor_data.get("use_svd"):
                # Use SVD decomposition
                # We have: sqrt(W) @ A = U @ diag(S) @ Vt
                # Solution operator: D = (A^T W A)^{-1} A^T W @ b
                #                      = Vt.T @ diag(1/S^2) @ Vt @ Vt.T @ diag(S) @ U.T @ sqrt(W) @ b
                #                      = Vt.T @ diag(1/S) @ U.T @ sqrt(W) @ b
                # Weights for derivative β are β-th row of: Vt.T @ diag(1/S) @ U.T @ sqrt(W)

                U = taylor_data["U"]
                S = taylor_data["S"]
                Vt = taylor_data["Vt"]
                sqrt_W = taylor_data["sqrt_W"]

                # Compute: weights_matrix = Vt.T @ diag(1/S) @ U.T @ sqrt(W)
                # Shape: [n_derivs, n_neighbors]
                weights_matrix = Vt.T @ np.diag(1.0 / S) @ U.T @ sqrt_W

                # Extract β-th row
                weights = weights_matrix[derivative_idx, :]
                return weights

            elif taylor_data.get("use_qr"):
                # Use QR decomposition - fall back to normal equations
                A = taylor_data["A"]
                W = taylor_data["W"]
                try:
                    # Compute (A^T W A)^{-1} A^T W and extract row
                    AtWA_inv = np.linalg.inv(A.T @ W @ A)
                    weights_matrix = AtWA_inv @ A.T @ W
                    weights = weights_matrix[derivative_idx, :]
                    return weights
                except np.linalg.LinAlgError:
                    return None

            elif taylor_data.get("AtWA_inv") is not None:
                # Direct normal equations
                AtWA_inv = taylor_data["AtWA_inv"]
                W = taylor_data["W"]
                A = taylor_data["A"]
                weights_matrix = AtWA_inv @ A.T @ W
                weights = weights_matrix[derivative_idx, :]
                return weights

            else:
                return None

        except Exception:
            return None

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

    def _check_monotonicity_violation(
        self, D_coeffs: np.ndarray, point_idx: int = 0, use_adaptive: bool | None = None
    ) -> bool:
        """
        Check if unconstrained Taylor coefficients violate monotonicity.

        Unified method supporting both basic (strict) and adaptive (threshold-based) modes.

        Args:
            D_coeffs: Taylor derivative coefficients from unconstrained solve
            point_idx: Collocation point index (for debugging)
            use_adaptive: Override adaptive mode (deprecated parameter, always uses basic M-matrix check)

        Returns:
            True if QP constraints are needed

        Mathematical Criteria (see docs/development/QP_MONOTONICITY_CRITERIA.md):
            1. Laplacian negativity: D₂ < 0 (diffusion dominance)
            2. Gradient boundedness: |D₁| ≤ C·σ²·|D₂| (prevent advection dominance)
            3. Higher-order control: Σ|Dₖ| < |D₂| for order ≥ 3 (truncation error)

        Modes:
            - BASIC: Return True if ANY criterion violated
            - ADAPTIVE: Return True if violation_severity > adaptive_threshold
        """
        # Find multi-index locations
        laplacian_idx = None
        gradient_idx = None

        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 2 and all(b <= 2 for b in beta):
                if laplacian_idx is None:
                    laplacian_idx = k
            elif sum(beta) == 1:
                if gradient_idx is None:
                    gradient_idx = k

        if laplacian_idx is None:
            return False  # Cannot check monotonicity without Laplacian term

        # Extract coefficients
        D_laplacian = D_coeffs[laplacian_idx]
        tolerance = 1e-12
        laplacian_mag = abs(D_laplacian) + 1e-10

        # Criterion 1: Laplacian Negativity (Diffusion Dominance)
        violation_1 = D_laplacian >= -tolerance

        # Criterion 2: Gradient Boundedness (Prevent Advection Dominance)
        violation_2 = False
        if gradient_idx is not None:
            D_gradient = D_coeffs[gradient_idx]
            sigma = self._get_sigma_value(point_idx)
            scale_factor = 10.0 * max(sigma**2, 0.1)
            gradient_mag = abs(D_gradient)
            violation_2 = gradient_mag > scale_factor * laplacian_mag

        # Criterion 3: Higher-Order Control (Truncation Error)
        higher_order_norm = sum(abs(D_coeffs[k]) for k in range(len(D_coeffs)) if sum(self.multi_indices[k]) >= 3)
        violation_3 = higher_order_norm > laplacian_mag

        # Basic violation check (any criterion violated)
        has_violation = violation_1 or violation_2 or violation_3

        # Always use basic M-matrix check (adaptive parameter deprecated)
        if not use_adaptive:
            # BASIC MODE: Strict enforcement of all criteria
            return has_violation

        # ADAPTIVE MODE: Threshold-Based Decision
        # Compute quantitative severity (0 = no violation, >0 = increasing severity)
        severity = 0.0

        # Severity 1: How positive is Laplacian (should be negative)
        if violation_1:
            severity = max(severity, D_laplacian + tolerance)

        # Severity 2: Excess gradient relative to diffusion
        if violation_2:
            D_gradient = D_coeffs[gradient_idx]
            sigma = self._get_sigma_value(point_idx)
            scale_factor = 10.0 * max(sigma**2, 0.1)
            gradient_mag = abs(D_gradient)
            excess_gradient = gradient_mag / laplacian_mag - scale_factor
            severity = max(severity, excess_gradient)

        # Severity 3: Excess higher-order terms
        if violation_3:
            excess_higher_order = higher_order_norm / laplacian_mag - 1.0
            severity = max(severity, excess_higher_order)

        # Decision: use QP if M-matrix property is violated (severity > 0)
        # This is the basic adaptive M-matrix checking
        needs_qp = severity > 0.0

        return needs_qp

    def _check_m_matrix_property(
        self, weights: np.ndarray, point_idx: int, tolerance: float = 1e-12
    ) -> tuple[bool, dict]:
        """
        Verify M-matrix property for finite difference weights.

        For a monotone scheme, the Laplacian weights must satisfy:
        - Diagonal (center): w_center ≤ 0
        - Off-diagonal (neighbors): w_j ≥ -tolerance for j ≠ center

        Args:
            weights: Finite difference weights [n_neighbors]
            point_idx: Index of collocation point
            tolerance: Small tolerance for numerical errors

        Returns:
            is_monotone: True if M-matrix property satisfied
            diagnostics: Dictionary with detailed information
        """
        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]

        # Find center point index in neighborhood
        center_idx_in_neighbors = None
        center_point = self.collocation_points[point_idx]

        for j, idx in enumerate(neighbor_indices):
            if idx == -1:  # Ghost particle
                # Check if this ghost is actually at center location
                if np.allclose(neighborhood["points"][j], center_point):
                    center_idx_in_neighbors = j
                    break
            elif idx == point_idx:
                # Direct match
                center_idx_in_neighbors = j
                break
            elif np.allclose(self.collocation_points[idx], center_point):
                # Position match
                center_idx_in_neighbors = j
                break

        if center_idx_in_neighbors is None:
            # Center not found in neighborhood - unusual but possible
            # Consider all weights as neighbors
            w_center = 0.0
            neighbor_weights = weights
        else:
            w_center = weights[center_idx_in_neighbors]
            neighbor_weights = np.delete(weights, center_idx_in_neighbors)

        # Check M-matrix conditions
        center_ok = w_center <= tolerance  # Should be ≤ 0 (allow small positive for numerical error)
        neighbors_ok = np.all(neighbor_weights >= -tolerance)  # Should be ≥ 0

        is_monotone = center_ok and neighbors_ok

        # Compute diagnostics
        min_neighbor_weight = np.min(neighbor_weights) if len(neighbor_weights) > 0 else 0.0
        num_violations = np.sum(neighbor_weights < -tolerance)

        diagnostics = {
            "is_monotone": is_monotone,
            "center_ok": center_ok,
            "neighbors_ok": neighbors_ok,
            "w_center": float(w_center),
            "min_neighbor_weight": float(min_neighbor_weight),
            "max_neighbor_weight": float(np.max(neighbor_weights)) if len(neighbor_weights) > 0 else 0.0,
            "num_violations": int(num_violations),
            "num_neighbors": len(neighbor_weights),
            "violation_severity": float(abs(min_neighbor_weight)) if min_neighbor_weight < -tolerance else 0.0,
        }

        return is_monotone, diagnostics

    def _build_monotonicity_constraints(
        self,
        A: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_points: np.ndarray,
        center_point: np.ndarray,
        point_idx: int,
    ) -> list[dict]:
        """
        Build M-matrix monotonicity constraints for finite difference weights.

        For a monotone scheme approximating the Laplacian ∂²u/∂x², the GFDM
        finite difference weights w_j must satisfy the M-matrix property:
        - Diagonal weight (center): w_center ≤ 0
        - Off-diagonal weights (neighbors): w_j ≥ 0 for j ≠ center

        Strategy:
            This implementation uses INDIRECT constraints on Taylor coefficients D
            rather than direct Hamiltonian gradient constraints ∂H/∂u_j ≥ 0.

            The indirect approach is simpler but approximate. For stricter monotonicity,
            consider implementing direct Hamiltonian gradient constraints (see Section 4.4
            of particle-collocation theory document).

            TODO (Future Enhancement): Implement direct Hamiltonian gradient constraints
            -------------------------------------------------------------------------
            For Hamiltonian H = 1/2|∇u|² + γm|∇u|² + V(x), enforce:

                ∂H_h/∂u_j ≥ 0  for all j ≠ j_0

            where H_h is the numerical Hamiltonian. This gives:

                (1 + 2γm) (Σ_l c_{j_0,l} u_l) · c_{j_0,j} ≥ 0

            These are LINEAR constraints on the finite difference coefficients c_{j_0,j},
            which can be derived from the Taylor coefficients D through the relation:

                w = β-th row of (A^T W A)^{-1} A^T W

            This approach is more direct and theoretically rigorous than the current
            indirect constraints on D.

            See docs/theory/numerical_methods/[PRIVATE]_particle_collocation_qp_monotone.md
            Section 4.5 for implementation details.

        Constraint Categories:
            1. Diffusion dominance: ∂²u/∂x² coefficient should be negative
            2. Gradient boundedness: ∂u/∂x shouldn't overwhelm diffusion
            3. Truncation error control: Higher derivatives should be small

        References:
            Section 4.3-4.5 of particle-collocation theory document
        """
        constraints = []

        if self.dimension == 1:
            # Find indices for derivatives
            laplacian_idx = None
            first_deriv_idx = None

            for k, beta in enumerate(self.multi_indices):
                if beta == (2,):
                    laplacian_idx = k
                elif beta == (1,):
                    first_deriv_idx = k

            if laplacian_idx is None:
                # No second derivative in Taylor expansion - skip constraints
                return constraints

            # ===================================================================
            # CONSTRAINT 1: Negative Laplacian (Diffusion Dominance)
            # ===================================================================
            # Physical motivation: For elliptic operators σ²/2 ∂²u/∂x²,
            # the diffusion term should have negative coefficient to produce
            # proper M-matrix structure (diagonal ≤ 0).

            def constraint_laplacian_negative(x):
                """Enforce Laplacian coefficient is negative (diffusion dominance)"""
                # For proper elliptic discretization: ∂²u/∂x² < 0
                return -x[laplacian_idx]  # Should be positive (≥ 0)

            constraints.append({"type": "ineq", "fun": constraint_laplacian_negative})

            # ===================================================================
            # CONSTRAINT 2: Gradient Boundedness (Prevent Advection Dominance)
            # ===================================================================
            # Physical motivation: Prevent first-order (advection) terms from
            # overwhelming second-order (diffusion) terms, which can break
            # M-matrix structure.
            #
            # ADAPTIVE SCALING: Use problem's diffusion coefficient σ to set
            # physically meaningful bounds. For diffusion operator σ²/2 ∂²u/∂x²,
            # gradient scale should be bounded relative to σ²|∂²u/∂x²|.

            if first_deriv_idx is not None:
                # Get diffusion coefficient from problem
                sigma = self._get_sigma_value(point_idx)
                sigma_sq = sigma**2

                def constraint_gradient_bounded(x):
                    """Ensure first derivative doesn't dominate second derivative"""
                    # Physical scaling: |∂u/∂x| ~ O(1), |∂²u/∂x²| ~ O(σ²)
                    # For proper elliptic operator balance: |∇u| ≤ C·σ²|Δu|
                    laplacian_mag = abs(x[laplacian_idx]) + 1e-10
                    gradient_mag = abs(x[first_deriv_idx])

                    # Adaptive bound: gradient shouldn't exceed σ²-scaled Laplacian
                    # Factor 10.0 allows for reasonable advection while preventing dominance
                    scale_factor = 10.0 * max(sigma_sq, 0.1)  # Min scale for very small σ
                    return scale_factor * laplacian_mag - gradient_mag

                constraints.append({"type": "ineq", "fun": constraint_gradient_bounded})

            # ===================================================================
            # CONSTRAINT 3: Higher-Order Term Control (Truncation Error)
            # ===================================================================
            # Physical motivation: Third and higher derivatives represent
            # truncation error. Keeping them small relative to the Laplacian
            # improves accuracy and stability.

            def constraint_higher_order_small(x):
                """Keep higher-order terms small (truncation error control)"""
                higher_order_norm = 0.0
                for k, beta in enumerate(self.multi_indices):
                    if sum(beta) >= 3:  # Third and higher derivatives
                        higher_order_norm += abs(x[k])
                # Higher-order terms should be smaller than second derivative
                laplacian_mag = abs(x[laplacian_idx]) + 1e-10
                return laplacian_mag - higher_order_norm  # Should be positive

            constraints.append({"type": "ineq", "fun": constraint_higher_order_small})

        return constraints

    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Solve the HJB system using GFDM collocation method.

        Args:
            M_density_evolution_from_FP: (Nt, Nx) density evolution from FP solver
            U_final_condition_at_T: (Nx,) final condition for value function
            U_from_prev_picard: (Nt, Nx) value function from previous Picard iteration
            show_progress: Whether to display progress bar for timesteps

        Returns:
            (Nt, Nx) solution array
        """
        from mfg_pde.utils.progress import tqdm

        Nt, _Nx = M_density_evolution_from_FP.shape

        # For GFDM, we work directly with collocation points
        # Map grid data to collocation points
        U_solution_collocation = np.zeros((Nt, self.n_points))
        M_collocation = self._map_grid_to_collocation_batch(M_density_evolution_from_FP)
        U_prev_collocation = self._map_grid_to_collocation_batch(U_from_prev_picard)

        # Set final condition
        U_solution_collocation[Nt - 1, :] = self._map_grid_to_collocation(U_final_condition_at_T)

        # Backward time stepping with progress bar
        timestep_range = range(Nt - 2, -1, -1)
        if show_progress:
            timestep_range = tqdm(
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
        u_current = u_n_plus_1.copy()

        for _newton_iter in range(self.max_newton_iterations):
            # Compute residual
            residual = self._compute_hjb_residual(u_current, u_n_plus_1, m_n_plus_1, time_idx)

            # Check convergence
            if np.linalg.norm(residual) < self.newton_tolerance:
                break

            # Compute Jacobian
            jacobian = self._compute_hjb_jacobian(u_current, u_n_plus_1, u_prev_picard, m_n_plus_1, time_idx)

            # Apply boundary conditions
            jacobian_bc, residual_bc = self._apply_boundary_conditions_to_system(jacobian, residual, time_idx)

            # Newton update with step size limiting
            try:
                delta_u = np.linalg.solve(jacobian_bc, -residual_bc)
            except np.linalg.LinAlgError:
                delta_u = np.linalg.pinv(jacobian_bc) @ (-residual_bc)

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
        residual = np.zeros(self.n_points)

        # Time derivative approximation (backward Euler)
        # For backward-in-time problems: ∂u/∂t ≈ (u_{n+1} - u_n) / dt
        # where t_{n+1} > t_n (future time is at n+1)
        dt = self.problem.T / (self.problem.Nt - 1)
        u_t = (u_n_plus_1 - u_current) / dt

        for i in range(self.n_points):
            # Get spatial coordinates
            x = self.collocation_points[i]

            # Approximate derivatives using GFDM
            derivs = self.approximate_derivatives(u_current, i)

            # Extract gradient and Hessian
            # For 1D: derivs[(1,)] is du/dx, derivs[(2,)] is d²u/dx²
            # For 2D: derivs[(1,0)] is du/dx, derivs[(0,1)] is du/dy, etc.
            d = self.problem.d  # Spatial dimension
            if d == 1:
                p = derivs.get((1,), 0.0)
                laplacian = derivs.get((2,), 0.0)
            elif d == 2:
                p_x = derivs.get((1, 0), 0.0)
                p_y = derivs.get((0, 1), 0.0)
                p = np.array([p_x, p_y])
                laplacian = derivs.get((2, 0), 0.0) + derivs.get((0, 2), 0.0)
            else:
                msg = f"Dimension {d} not implemented"
                raise NotImplementedError(msg)

            # Hamiltonian (user-provided)
            H = self.problem.H(x, p, m_n_plus_1[i])

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

        Uses finite differences to approximate ∂R/∂u where R is the residual.

        Bug #15 fix: Disable QP in Jacobian computation to reduce QP calls from ~750k to ~7.5k.
        Jacobian only affects Newton convergence rate, not final monotonicity (enforced by residual).
        """
        n = self.n_points
        jacobian = np.zeros((n, n))

        # Finite difference step
        eps = 1e-7

        # Bug #15 fix: Temporarily disable QP for Jacobian computation
        # This reduces QP invocations while maintaining monotonicity (enforced by residual)
        # Jacobian finite differences are a numerical approximation tool for Newton's method
        # and don't need monotonicity constraints. Only the residual evaluation needs QP.
        saved_qp_level = self.qp_optimization_level
        self.qp_optimization_level = "none"

        try:
            # Compute Jacobian by columns (perturbing each u[j])
            for j in range(n):
                u_plus = u_current.copy()
                u_plus[j] += eps

                # FIXED: Use actual u_n_plus_1 (not perturbed in Newton iteration)
                residual_plus = self._compute_hjb_residual(u_plus, u_n_plus_1, m_n_plus_1, time_idx)
                residual_base = self._compute_hjb_residual(u_current, u_n_plus_1, m_n_plus_1, time_idx)

                jacobian[:, j] = (residual_plus - residual_base) / eps
        finally:
            # Restore QP for residual evaluation
            self.qp_optimization_level = saved_qp_level

        return jacobian

    def _apply_boundary_conditions_to_solution(self, u: np.ndarray, time_idx: int) -> np.ndarray:
        """Apply boundary conditions directly to solution array."""
        # Check boundary condition type
        bc_type_val = self._get_boundary_condition_property("type", "dirichlet")

        # Convert to lowercase for case-insensitive comparison
        if isinstance(bc_type_val, str):
            bc_type = bc_type_val.lower()
        else:
            bc_type = "dirichlet"

        if bc_type == "dirichlet":
            # For collocation points on boundaries, enforce Dirichlet values
            if len(self.boundary_indices) > 0:
                bc_value = self._get_boundary_condition_property("value", 0.0)
                if callable(bc_value):
                    # Time-dependent or space-dependent BC
                    current_time = self.problem.T * time_idx / (self.problem.Nt + 1)
                    for i in self.boundary_indices:
                        u[i] = bc_value(self.collocation_points[i], current_time)
                else:
                    # Constant BC value
                    u[self.boundary_indices] = bc_value
            return u
        elif bc_type == "neumann":
            # Neumann conditions typically enforced weakly through residual
            return u
        else:
            # Default: no modification
            return u

    def _apply_boundary_conditions_to_system(
        self, jacobian: np.ndarray, residual: np.ndarray, time_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply boundary conditions to the linear system J·δu = -R.

        For Dirichlet BC: Set row to identity and residual to zero.
        For Neumann BC: Usually handled weakly (no modification).
        """
        jacobian_bc = jacobian.copy()
        residual_bc = residual.copy()

        bc_type_val = self._get_boundary_condition_property("type", "dirichlet")
        if isinstance(bc_type_val, str):
            bc_type = bc_type_val.lower()
        else:
            bc_type = "dirichlet"

        if bc_type == "dirichlet":
            # Enforce Dirichlet BC by setting identity rows in Jacobian
            if len(self.boundary_indices) > 0:
                for i in self.boundary_indices:
                    # Set Jacobian row to identity (δu_i = 0 enforced)
                    jacobian_bc[i, :] = 0.0
                    jacobian_bc[i, i] = 1.0
                    # Set residual to zero (no update for boundary values)
                    residual_bc[i] = 0.0
            return jacobian_bc, residual_bc
        else:
            return jacobian_bc, residual_bc

    def _map_grid_to_collocation(self, u_grid: np.ndarray) -> np.ndarray:
        """
        Map values from regular grid to collocation points.

        Uses nearest neighbor or linear interpolation.
        """
        # Placeholder implementation: Assume 1D for simplicity
        # For production: use scipy.interpolate or custom interpolation
        u_collocation = np.zeros(self.n_points)
        for i in range(self.n_points):
            # Find nearest grid point or interpolate
            # Simple approach: nearest neighbor
            idx = self._map_collocation_index_to_grid_index(i)
            u_collocation[i] = u_grid[idx]
        return u_collocation

    def _map_collocation_to_grid(self, u_collocation: np.ndarray) -> np.ndarray:
        """
        Map values from collocation points to regular grid.

        Uses inverse interpolation or reconstruction.
        """
        # Placeholder: Assume 1D and that grid == collocation points
        Nx = getattr(self.problem, "Nx", self.n_points)
        u_grid = np.zeros(Nx)
        for j in range(Nx):
            # Find nearest collocation point
            # Simple approach: direct copy if aligned
            if j < self.n_points:
                u_grid[j] = u_collocation[j]
        return u_grid

    def _map_grid_to_collocation_batch(self, U_grid: np.ndarray) -> np.ndarray:
        """Batch version of _map_grid_to_collocation."""
        Nt, _Nx = U_grid.shape
        U_collocation = np.zeros((Nt, self.n_points))
        for n in range(Nt):
            U_collocation[n, :] = self._map_grid_to_collocation(U_grid[n, :])
        return U_collocation

    def _map_collocation_to_grid_batch(self, U_collocation: np.ndarray) -> np.ndarray:
        """Batch version of _map_collocation_to_grid."""
        Nt, _n_points = U_collocation.shape
        Nx = getattr(self.problem, "Nx", self.n_points)
        U_grid = np.zeros((Nt, Nx))
        for n in range(Nt):
            U_grid[n, :] = self._map_collocation_to_grid(U_collocation[n, :])
        return U_grid

    def _map_collocation_index_to_grid_index(self, collocation_idx: int) -> int:
        """
        Map collocation point index to nearest grid index.

        Placeholder: Assumes collocation points are aligned with grid.
        """
        # Simple 1D mapping
        Nx = getattr(self.problem, "Nx", self.n_points)
        if self.n_points == Nx:
            return collocation_idx
        else:
            # Scale index
            return int(collocation_idx * Nx / self.n_points)


class MonotonicityStats:
    """Track M-matrix property satisfaction statistics across solve."""

    def __init__(self):
        self.total_points = 0
        self.monotone_points = 0
        self.violations_by_point: dict[int, list[dict]] = {}
        self.worst_violations: list[dict] = []

    def record_point(self, point_idx: int, is_monotone: bool, diagnostics: dict):
        """Record M-matrix verification result for a single point."""
        self.total_points += 1
        if is_monotone:
            self.monotone_points += 1
        else:
            if point_idx not in self.violations_by_point:
                self.violations_by_point[point_idx] = []
            self.violations_by_point[point_idx].append(diagnostics)
            self.worst_violations.append({"point_idx": point_idx, "severity": diagnostics["violation_severity"]})

    def get_success_rate(self) -> float:
        """Compute percentage of points satisfying M-matrix property."""
        if self.total_points == 0:
            return 0.0
        return 100.0 * self.monotone_points / self.total_points

    def get_summary(self) -> dict:
        """Get summary statistics for M-matrix verification."""
        success_rate = self.get_success_rate()
        num_violating_points = len(self.violations_by_point)
        max_violation = 0.0
        if self.worst_violations:
            max_violation = max(v["severity"] for v in self.worst_violations)

        return {
            "success_rate": success_rate,
            "monotone_points": self.monotone_points,
            "total_points": self.total_points,
            "num_violating_points": num_violating_points,
            "max_violation_severity": max_violation,
        }
