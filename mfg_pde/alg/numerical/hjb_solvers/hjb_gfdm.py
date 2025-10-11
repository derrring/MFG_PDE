from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist

from .base_hjb import BaseHJBSolver

# Optional QP solver imports (merged from tuned QP solver)
CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None
OSQP_AVAILABLE = importlib.util.find_spec("osqp") is not None

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions
    from mfg_pde.types.internal import MultiIndexTuple


class HJBGFDMSolver(BaseHJBSolver):
    """
    Generalized Finite Difference Method (GFDM) solver for HJB equations using collocation.

    This solver implements meshfree collocation for HJB equations using:
    1. δ-neighborhood search for local support
    2. Taylor expansion with weighted least squares for derivative approximation
    3. Newton iteration for nonlinear HJB equations
    4. Support for various boundary conditions
    5. Enhanced QP optimization levels (smart/tuned) for performance optimization

    QP Optimization Levels:
    - "none": Basic GFDM without QP optimization
    - "basic": Standard monotonicity constraints
    - "smart": Moderate QP usage optimization with context awareness
    - "tuned": Aggressive QP usage optimization targeting ~10% usage rate
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
        use_monotone_constraints: bool = False,
        # Enhanced QP options (merged from tuned QP solver)
        qp_optimization_level: str = "none",  # "none", "basic", "smart", "tuned"
        qp_usage_target: float = 0.1,
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
            use_monotone_constraints: Enable constrained QP for monotonicity preservation
            qp_optimization_level: QP optimization level ("none", "basic", "smart", "tuned")
            qp_usage_target: Target QP usage rate for optimization (default 0.1 = 10%)
        """
        super().__init__(problem)

        # Store QP optimization level early for method naming
        self.qp_optimization_level = qp_optimization_level

        # Set method name based on QP optimization level
        if qp_optimization_level == "none":
            self.hjb_method_name = "GFDM"
        elif qp_optimization_level == "basic":
            self.hjb_method_name = "GFDM-Basic"
        elif qp_optimization_level == "smart":
            self.hjb_method_name = "GFDM-Smart-QP"
        elif qp_optimization_level == "tuned":
            self.hjb_method_name = "GFDM-Tuned-QP"
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

        # Monotonicity constraint option
        self.use_monotone_constraints = use_monotone_constraints

        # Enhanced QP optimization features (merged from tuned QP solver)
        self.qp_usage_target = qp_usage_target

        # Initialize QP-related attributes based on optimization level
        if qp_optimization_level in ["smart", "tuned"]:
            self._init_enhanced_qp_features()
        else:
            self.enhanced_qp_stats = None
            self._current_point_idx = 0
            self._current_time_ratio = 0.0
            self._current_newton_iter = 0

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
            # Find neighbors within delta radius
            neighbor_mask = distances[i, :] < self.delta
            neighbor_indices = np.where(neighbor_mask)[0]
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
        """Compute weights based on distance and weight function."""
        if self.weight_function == "gaussian":
            return np.exp(-(distances**2) / self.weight_scale**2)
        elif self.weight_function == "inverse_distance":
            return 1.0 / (distances + 1e-12)
        elif self.weight_function == "uniform":
            return np.ones_like(distances)
        elif self.weight_function == "wendland":
            # Wendland's compactly supported kernel following equation (8):
            # w_{j_0,j_l} = (1/c_d) * (1 - ||X_{k,j_0} - X_{k,j_l}||/c)_+^4
            # where c = delta (support radius) and ()_+ = max(0, .)
            c = self.delta
            normalized_distances = distances / c

            # Compute (1 - r/c)_+^4
            weights = np.maximum(0, 1 - normalized_distances) ** 4

            # For 1D case, the normalization constant c_d can be computed analytically
            # ∫_{-δ}^{δ} (1 - |x|/δ)_+^4 dx = δ * ∫_{-1}^{1} (1 - |t|)_+^4 dt = δ * (2/5) = 2δ/5
            if self.dimension == 1:
                c_d = 2 * c / 5
            else:
                # For higher dimensions, use empirical normalization
                # or compute c_d numerically, but for now use simple normalization
                c_d = 1.0

            # Apply normalization: w = (1/c_d) * (1 - r/c)_+^4
            weights = weights / c_d

            return weights
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
        # Inject context for enhanced QP features
        if self.qp_optimization_level in ["smart", "tuned"]:
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

        # Right-hand side: u(x_center) - u(x_neighbor) following equation (6) in the mathematical framework
        # For ghost particles, this becomes u_center - u_center = 0, enforcing ∂u/∂n = 0
        b = u_center - u_neighbors

        # Solve weighted least squares with optional monotonicity constraints
        use_monotone_qp = hasattr(self, "use_monotone_constraints") and self.use_monotone_constraints

        if use_monotone_qp:
            # First try unconstrained solution to check if constraints are needed
            unconstrained_coeffs = self._solve_unconstrained_fallback(taylor_data, b)  # type: ignore[arg-type]

            # Check if unconstrained solution violates monotonicity
            # Use enhanced QP logic if available, otherwise use basic check
            if self.qp_optimization_level in ["smart", "tuned"]:
                needs_constraints = self._enhanced_check_monotonicity_violation(unconstrained_coeffs)
            else:
                needs_constraints = self._check_monotonicity_violation(unconstrained_coeffs)

            if needs_constraints:
                # Use constrained QP for monotonicity (enhanced version if available)
                if self.qp_optimization_level in ["smart", "tuned"] and CVXPY_AVAILABLE:
                    derivative_coeffs = self._enhanced_solve_monotone_constrained_qp(taylor_data, b, point_idx)  # type: ignore[arg-type]
                else:
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

        Returns:
            derivative_coeffs: Coefficients for derivative approximation
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fallback to unconstrained if scipy not available
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

        # Solve constrained optimization with robust settings
        try:
            # Try fast L-BFGS-B first if only bounds constraints
            if len(constraints) == 0:
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

            if result.success:
                return result.x
            else:
                # Fallback to unconstrained if optimization fails
                return x0

        except Exception:
            # Fallback to unconstrained if any error occurs
            return x0

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
    ) -> list[dict]:
        """
        Build M-matrix monotonicity constraints for finite difference weights.

        For a monotone scheme approximating the Laplacian ∂²u/∂x², the GFDM
        finite difference weights w_j must satisfy the M-matrix property:
        - Diagonal weight (center): w_center ≤ 0
        - Off-diagonal weights (neighbors): w_j ≥ 0 for j ≠ center

        Note: In the current QP formulation, we optimize over Taylor coefficients
        D (derivatives at center), not directly over weights w. Therefore, these
        constraints are approximate - they enforce conditions on D that typically
        lead to proper M-matrix structure in w.

        References:
            Section 4.3 of particle-collocation theory document
        """
        constraints = []

        if self.dimension == 1:
            # Find indices for second derivative (Laplacian)
            laplacian_idx = None
            for k, beta in enumerate(self.multi_indices):
                if beta == (2,):
                    laplacian_idx = k
                    break

            if laplacian_idx is None:
                # No second derivative in Taylor expansion - skip constraints
                return constraints

            # IMPROVED CONSTRAINTS (still indirect, but better motivated):
            # For elliptic operators like Laplacian with diffusion σ²/2 ∂²u/∂x²,
            # proper monotone schemes have:
            # 1. Second derivative coefficient should be negative (diffusion effect)
            # 2. First derivative coefficient should be bounded (not dominate)
            # 3. Higher derivatives should be small (truncation error)

            # For elliptic operators: σ²/2 ∂²u/∂x² (diffusion)
            # We enforce negative Laplacian coefficient for proper monotone discretization

            def constraint_laplacian_negative(x):
                """Enforce Laplacian coefficient is negative (diffusion)"""
                # For proper elliptic discretization: ∂²u/∂x² < 0 for diffusion
                return -x[laplacian_idx]  # Should be positive

            constraints.append({"type": "ineq", "fun": constraint_laplacian_negative})

            # Find first derivative index
            first_deriv_idx = None
            for k, beta in enumerate(self.multi_indices):
                if beta == (1,):
                    first_deriv_idx = k
                    break

            if first_deriv_idx is not None:

                def constraint_gradient_bounded(x):
                    """Ensure first derivative doesn't dominate second derivative"""
                    # |∂u/∂x| should be O(1) while |∂²u/∂x²| ~ O(σ²)
                    # Prevent gradient from overwhelming diffusion
                    laplacian_mag = abs(x[laplacian_idx]) + 1e-10
                    gradient_mag = abs(x[first_deriv_idx])
                    # Gradient shouldn't be more than 10× the Laplacian scale
                    return 10.0 * laplacian_mag - gradient_mag

                constraints.append({"type": "ineq", "fun": constraint_gradient_bounded})

            # Control higher-order derivatives (truncation error)
            def constraint_higher_order_small(x):
                """Keep higher-order terms small (truncation error)"""
                higher_order_norm = 0.0
                for k, beta in enumerate(self.multi_indices):
                    if sum(beta) >= 3:  # Third and higher derivatives
                        higher_order_norm += abs(x[k])
                # Higher-order terms should be smaller than second derivative
                laplacian_mag = abs(x[laplacian_idx]) + 1e-10
                return laplacian_mag - higher_order_norm  # Should be positive

            constraints.append({"type": "ineq", "fun": constraint_higher_order_small})

        return constraints


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
