from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist

from .base_hjb import BaseHJBSolver

# Optional QP solver imports (merged from tuned QP solver)
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

OSQP_AVAILABLE = importlib.util.find_spec("osqp") is not None

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions
    from mfg_pde.types.internal import DerivativeDict, GradientDict, MultiIndexTuple


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

            except:
                # Fallback to QR decomposition
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
                except:
                    # Final fallback to normal equations
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
                    except:
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
                A, neighbor_indices, neighbor_points, center_point  # type: ignore[arg-type]
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

    def _build_monotonicity_constraints(
        self,
        A: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_points: np.ndarray,
        center_point: np.ndarray,
    ) -> list[dict]:
        """
        Build monotonicity constraints for finite difference weights.

        For a monotone scheme, the finite difference weights should satisfy:
        - Off-diagonal weights should be non-negative
        - Diagonal weight should be negative (for Laplacian)
        - Sum property should be maintained for consistency
        """
        constraints = []

        if self.dimension == 1:
            # For 1D Laplacian approximation, we want weights w_i such that:
            # Σ w_i u_i ≈ ∂²u/∂x² at center point
            # Monotonicity requires: w_center < 0, w_neighbors ≥ 0

            # Find the index of the second derivative term
            second_deriv_idx = None
            for k, beta in enumerate(self.multi_indices):
                if beta == (2,):
                    second_deriv_idx = k
                    break

            if second_deriv_idx is not None:
                # The finite difference weights are related to Taylor coefficients
                # For second derivative: w = A_inv @ e_k where e_k selects second derivative
                # We need w_i ≥ 0 for off-diagonal terms

                def constraint_positive_neighbors(x):
                    """Ensure finite difference weights for neighbors are non-negative"""
                    # Convert Taylor coefficients to finite difference weights
                    # This is an approximation - exact relationship depends on stencil
                    second_deriv_coeff = x[second_deriv_idx]

                    # For well-behaved stencils, large negative second derivative coefficient
                    # corresponds to positive neighbor weights
                    return second_deriv_coeff + 10.0  # Should be reasonable

                constraints.append({"type": "ineq", "fun": constraint_positive_neighbors})

        return constraints

    def _check_monotonicity_violation(self, coeffs: np.ndarray) -> bool:
        """
        Check if the unconstrained solution violates monotonicity requirements.

        Returns True if constrained optimization is needed.
        """
        # Simple heuristic: check if any coefficients are extremely large
        max_coeff = np.max(np.abs(coeffs))

        # If coefficients are reasonable, no need for constraints
        if max_coeff < 100.0:
            return False

        # Check for oscillatory patterns in derivative coefficients
        for k, beta in enumerate(self.multi_indices):
            if sum(beta) >= 2:  # Second or higher derivatives
                if abs(coeffs[k]) > 200.0:  # Too large
                    return True

        return False

    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the HJB system using GFDM collocation method.

        Args:
            M_density_evolution_from_FP: (Nt, Nx) density evolution from FP solver
            U_final_condition_at_T: (Nx,) final condition for value function
            U_from_prev_picard: (Nt, Nx) value function from previous Picard iteration

        Returns:
            (Nt, Nx) solution array
        """
        Nt, _Nx = M_density_evolution_from_FP.shape

        # For GFDM, we work directly with collocation points
        # Map grid data to collocation points
        U_solution_collocation = np.zeros((Nt, self.n_points))
        M_collocation = self._map_grid_to_collocation_batch(M_density_evolution_from_FP)
        U_prev_collocation = self._map_grid_to_collocation_batch(U_from_prev_picard)

        # Set final condition
        U_solution_collocation[Nt - 1, :] = self._map_grid_to_collocation(U_final_condition_at_T)

        # Backward time stepping
        for n in range(Nt - 2, -1, -1):
            U_solution_collocation[n, :] = self._solve_timestep(
                U_solution_collocation[n + 1, :],
                U_prev_collocation[n, :],
                M_collocation[n + 1, :],
                n,
            )

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
        # Inject temporal context for enhanced QP features
        if self.qp_optimization_level in ["smart", "tuned"]:
            total_time_steps = getattr(self.problem, "Nt", 50) + 1
            self._current_time_ratio = time_idx / max(1, total_time_steps - 1)

        u_current = u_n_plus_1.copy()

        for newton_iter in range(self.max_newton_iterations):
            # Inject Newton iteration context for enhanced QP features
            if self.qp_optimization_level in ["smart", "tuned"]:
                self._current_newton_iter = newton_iter
            # Compute residual
            residual = self._compute_hjb_residual(u_current, u_n_plus_1, m_n_plus_1, time_idx)

            # Check convergence
            if np.linalg.norm(residual) < self.newton_tolerance:
                break

            # Compute Jacobian
            jacobian = self._compute_hjb_jacobian(u_current, u_prev_picard, m_n_plus_1, time_idx)

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
        dt = self.problem.Dt
        sigma = self.problem.sigma

        for i in range(self.n_points):
            # Time derivative term
            residual[i] += (u_current[i] - u_n_plus_1[i]) / dt

            # Approximate spatial derivatives
            derivatives = self.approximate_derivatives(u_current, i)

            # Diffusion term: -σ²/2 * Δu
            if self.dimension == 1:
                if (2,) in derivatives:
                    residual[i] -= (sigma**2 / 2.0) * derivatives[(2,)]
            elif self.dimension == 2:
                laplacian = 0.0
                if (2, 0) in derivatives:
                    laplacian += derivatives[(2, 0)]
                if (0, 2) in derivatives:
                    laplacian += derivatives[(0, 2)]
                residual[i] -= (sigma**2 / 2.0) * laplacian

            # Hamiltonian term
            p_values = self._extract_gradient(derivatives)
            # Map collocation index to grid index for Hamiltonian evaluation
            grid_idx = self._map_collocation_index_to_grid_index(i)
            hamiltonian = self.problem.H(x_idx=grid_idx, m_at_x=m_n_plus_1[i], p_values=p_values, t_idx=time_idx)
            residual[i] += hamiltonian

        return residual

    def _extract_gradient(self, derivatives: DerivativeDict) -> GradientDict:
        """Extract gradient components for Hamiltonian."""
        p_values = {}

        if self.dimension == 1:
            if (1,) in derivatives:
                p_values["forward"] = derivatives[(1,)]
                p_values["backward"] = derivatives[(1,)]
        elif self.dimension == 2:
            if (1, 0) in derivatives:
                p_values["x"] = derivatives[(1, 0)]
            if (0, 1) in derivatives:
                p_values["y"] = derivatives[(0, 1)]

        return p_values

    def _compute_hjb_jacobian(
        self,
        u_current: np.ndarray,
        u_prev_picard: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """Compute HJB Jacobian matrix."""
        jacobian = np.zeros((self.n_points, self.n_points))
        dt = self.problem.Dt
        sigma = self.problem.sigma
        eps = 1e-7

        # Time derivative contribution (diagonal)
        jacobian[np.diag_indices(self.n_points)] += 1.0 / dt

        # Spatial derivative contributions
        for i in range(self.n_points):
            taylor_data = self.taylor_matrices[i]
            if taylor_data is None:
                continue

            neighborhood = self.neighborhoods[i]
            neighbor_indices = neighborhood["indices"]

            # Diffusion term contribution
            # Compute derivative matrix using SVD, QR or normal equations
            if taylor_data.get("use_svd", False):  # type: ignore[attr-defined]
                # For SVD approach: derivative matrix = V @ S^{-1} @ U^T @ sqrt(W)
                try:
                    U = taylor_data["U"]
                    S = taylor_data["S"]
                    Vt = taylor_data["Vt"]
                    sqrt_W = taylor_data["sqrt_W"]

                    # Compute pseudoinverse matrix: V @ S^{-1} @ U^T @ sqrt(W)
                    S_inv_UT = (1.0 / S[:, np.newaxis]) * U.T  # Broadcasting for S^{-1} @ U^T
                    derivative_matrix = Vt.T @ S_inv_UT @ sqrt_W
                except:
                    derivative_matrix = None
            elif taylor_data.get("use_qr", False):  # type: ignore[attr-defined]
                # For QR approach: derivative matrix = R^{-1} @ Q^T @ sqrt(W)
                try:
                    Q = taylor_data["Q"]
                    R = taylor_data["R"]
                    sqrt_W = taylor_data["sqrt_W"]

                    R_inv = np.linalg.inv(R)
                    derivative_matrix = R_inv @ Q.T @ sqrt_W
                except:
                    derivative_matrix = None
            elif taylor_data.get("AtWA_inv") is not None:  # type: ignore[attr-defined]
                derivative_matrix = taylor_data["AtWA_inv"] @ taylor_data["AtW"]
            else:
                derivative_matrix = None

            if derivative_matrix is not None:
                # Find second derivative indices
                for k, beta in enumerate(self.multi_indices):
                    if (self.dimension == 1 and beta == (2,)) or (self.dimension == 2 and beta in [(2, 0), (0, 2)]):
                        # Check bounds for derivative_matrix access
                        if k < derivative_matrix.shape[0]:
                            for j_local, j_global in enumerate(neighbor_indices):  # type: ignore[var-annotated,arg-type]
                                if j_local < derivative_matrix.shape[1] and j_global >= 0:
                                    # Only apply to real particles (ghost particles have negative indices)
                                    coeff = derivative_matrix[k, j_local]
                                    jacobian[i, j_global] -= (sigma**2 / 2.0) * coeff

            # Hamiltonian Jacobian (numerical)
            for j in neighbor_indices:  # type: ignore[attr-defined]
                # Skip ghost particles (negative indices)
                if j < 0:
                    continue

                u_pert_plus = u_current.copy()
                u_pert_minus = u_current.copy()
                u_pert_plus[j] += eps
                u_pert_minus[j] -= eps

                deriv_plus = self.approximate_derivatives(u_pert_plus, i)
                deriv_minus = self.approximate_derivatives(u_pert_minus, i)

                p_plus = self._extract_gradient(deriv_plus)
                p_minus = self._extract_gradient(deriv_minus)

                try:
                    H_plus = self.problem.H(i, m_n_plus_1[i], p_plus, time_idx)
                    H_minus = self.problem.H(i, m_n_plus_1[i], p_minus, time_idx)
                    jacobian[i, j] += (H_plus - H_minus) / (2 * eps)
                except:
                    pass

        return jacobian

    def get_decomposition_info(self) -> dict:
        """Get information about the decomposition methods used."""
        total_points = self.n_points
        svd_count = sum(
            1  # type: ignore[misc]
            for i in range(total_points)
            if self.taylor_matrices[i] is not None and self.taylor_matrices[i].get("use_svd", False)  # type: ignore[attr-defined]
        )
        qr_count = sum(
            1  # type: ignore[misc]
            for i in range(total_points)
            if self.taylor_matrices[i] is not None and self.taylor_matrices[i].get("use_qr", False)  # type: ignore[attr-defined]
        )
        normal_count = total_points - svd_count - qr_count

        # Get condition numbers for SVD points
        condition_numbers = []
        ranks = []
        for i in range(total_points):
            if self.taylor_matrices[i] is not None and self.taylor_matrices[i].get("use_svd", False):  # type: ignore[attr-defined]
                condition_numbers.append(self.taylor_matrices[i].get("condition_number", np.inf))  # type: ignore[attr-defined]
                ranks.append(self.taylor_matrices[i].get("rank", 0))  # type: ignore[attr-defined]

        info = {
            "total_points": total_points,
            "svd_points": svd_count,
            "qr_points": qr_count,
            "normal_equation_points": normal_count,
            "svd_percentage": svd_count / total_points * 100 if total_points > 0 else 0,
            "condition_numbers": condition_numbers,
            "ranks": ranks,
            "avg_condition_number": (np.mean(condition_numbers) if condition_numbers else np.inf),
            "min_rank": min(ranks) if ranks else 0,
            "max_rank": max(ranks) if ranks else 0,
        }

        return info

    def _apply_boundary_conditions_to_solution(self, u: np.ndarray, time_idx: int) -> np.ndarray:
        """Apply boundary conditions to solution vector."""
        u_modified = u.copy()

        for boundary_idx in self.boundary_indices:
            if boundary_idx >= len(u_modified):
                continue

            bc_type = self._get_boundary_condition_property("type", "dirichlet")

            if bc_type == "dirichlet":
                bc_function = self._get_boundary_condition_property("function")
                bc_value = self._get_boundary_condition_property("value")
                left_value = self._get_boundary_condition_property("left_value")
                right_value = self._get_boundary_condition_property("right_value")

                if bc_function is not None:
                    point = self.collocation_points[boundary_idx]
                    t = time_idx * self.problem.Dt
                    u_modified[boundary_idx] = bc_function(t, point)
                elif bc_value is not None:
                    u_modified[boundary_idx] = bc_value
                elif left_value is not None or right_value is not None:
                    # Use left/right values for boundary points
                    point = self.collocation_points[boundary_idx]
                    if point[0] <= self.problem.xmin + 1e-10 and left_value is not None:
                        u_modified[boundary_idx] = left_value
                    elif point[0] >= self.problem.xmax - 1e-10 and right_value is not None:
                        u_modified[boundary_idx] = right_value

            elif bc_type == "no_flux":
                # For no-flux boundaries, we don't modify the solution directly
                # The boundary condition ∂u/∂x = 0 is enforced in the system matrix
                # So we leave the solution value as computed by the Newton iteration
                pass

        return u_modified

    def _apply_boundary_conditions_to_system(
        self, jacobian: np.ndarray, residual: np.ndarray, time_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions to system matrix and residual."""
        jacobian_modified = jacobian.copy()
        residual_modified = residual.copy()

        for boundary_idx in self.boundary_indices:
            if boundary_idx >= len(residual_modified):
                continue

            bc_type = self._get_boundary_condition_property("type", "dirichlet")

            if bc_type == "dirichlet":
                # Replace equation with identity
                jacobian_modified[boundary_idx, :] = 0.0
                jacobian_modified[boundary_idx, boundary_idx] = 1.0

                # Set target value using helper method
                bc_function = self._get_boundary_condition_property("function")
                bc_value = self._get_boundary_condition_property("value")

                if bc_function is not None:
                    point = self.collocation_points[boundary_idx]
                    t = time_idx * self.problem.Dt
                    target_value = bc_function(t, point)
                elif bc_value is not None:
                    target_value = bc_value
                else:
                    target_value = 0.0

                residual_modified[boundary_idx] = target_value

            elif bc_type == "no_flux":
                # No-flux boundary condition is now handled automatically through ghost particles
                # The ghost particle method enforces ∂u/∂n = 0 by setting u_ghost = u_center
                # No additional modification to the system is needed
                pass

        return jacobian_modified, residual_modified

    def _map_grid_to_collocation(self, u_grid: np.ndarray) -> np.ndarray:
        """Map grid values to collocation points."""
        if u_grid.shape[0] != self.n_points:
            # Simple nearest neighbor mapping
            grid_x = np.linspace(0, 1, self.problem.Nx + 1)
            u_collocation = np.zeros(self.n_points)

            for i in range(self.n_points):
                distances = np.abs(grid_x - self.collocation_points[i, 0])
                nearest_idx = np.argmin(distances)
                u_collocation[i] = u_grid[nearest_idx]

            return u_collocation
        else:
            return u_grid

    def _map_collocation_to_grid(self, u_collocation: np.ndarray) -> np.ndarray:
        """Map collocation point values to regular grid."""
        if u_collocation.shape[0] != self.problem.Nx + 1:
            grid_x = np.linspace(0, 1, self.problem.Nx + 1)
            u_grid = np.zeros(self.problem.Nx + 1)

            for i in range(self.problem.Nx + 1):
                distances = np.abs(self.collocation_points[:, 0] - grid_x[i])
                nearest_idx = np.argmin(distances)
                u_grid[i] = u_collocation[nearest_idx]

            return u_grid
        else:
            return u_collocation

    def _map_grid_to_collocation_batch(self, U_grid: np.ndarray) -> np.ndarray:
        """Map batch of grid arrays to collocation points."""
        U_collocation = np.zeros((U_grid.shape[0], self.n_points))
        for t in range(U_grid.shape[0]):
            U_collocation[t, :] = self._map_grid_to_collocation(U_grid[t, :])
        return U_collocation

    def _map_collocation_to_grid_batch(self, U_collocation: np.ndarray) -> np.ndarray:
        """Map batch of collocation arrays to grid."""
        U_grid = np.zeros((U_collocation.shape[0], self.problem.Nx + 1))
        for t in range(U_collocation.shape[0]):
            U_grid[t, :] = self._map_collocation_to_grid(U_collocation[t, :])
        return U_grid

    def _map_collocation_index_to_grid_index(self, collocation_idx: int) -> int:
        """Map a collocation point index to the nearest grid index."""
        # Collocation points are in [0, 1], map to [xmin, xmax]
        collocation_x = self.collocation_points[collocation_idx, 0]
        # Scale to physical domain
        physical_x = self.problem.xmin + collocation_x * (self.problem.xmax - self.problem.xmin)

        # Find nearest grid index
        grid_x = np.linspace(self.problem.xmin, self.problem.xmax, self.problem.Nx + 1)
        distances = np.abs(grid_x - physical_x)
        grid_idx = np.argmin(distances)

        return int(grid_idx)

    # Enhanced QP Features (merged from tuned QP solver)
    def _init_enhanced_qp_features(self):
        """Initialize enhanced QP features for smart/tuned optimization levels."""
        self.enhanced_qp_stats = {  # type: ignore[assignment]
            "total_qp_decisions": 0,
            "qp_activated": 0,
            "qp_skipped": 0,
            "extreme_violations": 0,
            "boundary_qp_activations": 0,
            "interior_qp_activations": 0,
            "early_time_qp_activations": 0,
            "late_time_qp_activations": 0,
            "threshold_adaptations": 0,
            "total_solve_time": 0.0,
        }

        # Context tracking
        self._current_point_idx = 0
        self._current_time_ratio = 0.0
        self._current_newton_iter = 0
        self._problem_difficulty = self._assess_problem_difficulty()

        # Boundary point identification
        self.boundary_point_set = set(self.boundary_indices) if len(self.boundary_indices) > 0 else set()

        # Compute thresholds based on optimization level
        if self.qp_optimization_level == "tuned":
            self._enhanced_thresholds = self._compute_tuned_thresholds()
            self._decision_threshold = 15.0  # More aggressive for tuned
        else:  # smart
            self._enhanced_thresholds = self._compute_smart_thresholds()
            self._decision_threshold = 5.0  # Less aggressive for smart

        self._threshold_adaptation_rate = 0.02

        print(f"Enhanced QP ({self.qp_optimization_level}) initialized:")
        print(f"  Target QP usage rate: {self.qp_usage_target:.1%}")
        print(f"  Decision threshold: {self._decision_threshold}")
        print(f"  CVXPY available: {'YES' if CVXPY_AVAILABLE else 'NO'}")
        print(f"  Boundary points: {len(self.boundary_point_set)}")
        print(f"  Problem difficulty: {self._problem_difficulty:.2f}")

    def _assess_problem_difficulty(self) -> float:
        """Assess problem difficulty for QP threshold tuning."""
        difficulty = 0.0

        sigma = getattr(self.problem, "sigma", 0.1)
        difficulty += min(1.0, sigma / 0.3)

        T = getattr(self.problem, "T", 1.0)
        difficulty += min(1.0, T / 3.0)

        Nx = getattr(self.problem, "Nx", 50)
        difficulty += min(1.0, (Nx - 20) / 80)

        coefCT = getattr(self.problem, "coefCT", 0.02)
        difficulty += min(1.0, coefCT / 0.1)

        return min(2.0, difficulty)

    def _compute_smart_thresholds(self) -> dict[str, float]:
        """Compute smart QP thresholds (moderate restrictions)."""
        base_threshold = 50.0
        difficulty_multiplier = 1.0 + self._problem_difficulty

        return {
            "extreme_violation": base_threshold * difficulty_multiplier * 10,  # 500-1500
            "severe_violation": base_threshold * difficulty_multiplier * 5,  # 250-750
            "moderate_violation": base_threshold * difficulty_multiplier * 2,  # 100-300
            "mild_violation": base_threshold * difficulty_multiplier * 1,  # 50-150
            "gradient_threshold": base_threshold * difficulty_multiplier * 0.5,  # 25-75
            "variation_threshold": base_threshold * difficulty_multiplier * 0.3,  # 15-45
        }

    def _compute_tuned_thresholds(self) -> dict[str, float]:
        """Compute tuned QP thresholds (aggressive restrictions for ~10% usage)."""
        base_threshold = 100.0
        difficulty_multiplier = (1.0 + self._problem_difficulty) * 3.0

        return {
            "extreme_violation": base_threshold * difficulty_multiplier * 20,  # 6000-12000
            "severe_violation": base_threshold * difficulty_multiplier * 10,  # 3000-6000
            "moderate_violation": base_threshold * difficulty_multiplier * 5,  # 1500-3000
            "mild_violation": base_threshold * difficulty_multiplier * 3,  # 900-1800
            "gradient_threshold": base_threshold * difficulty_multiplier * 2,  # 600-1200
            "variation_threshold": base_threshold * difficulty_multiplier * 1,  # 300-600
        }

    def _enhanced_check_monotonicity_violation(self, coeffs: np.ndarray) -> bool:
        """Enhanced monotonicity violation check with smart/tuned QP logic."""
        if self.qp_optimization_level not in ["smart", "tuned"] or self.enhanced_qp_stats is None:
            # Fallback to basic check
            return self._check_monotonicity_violation(coeffs)

        self.enhanced_qp_stats["total_qp_decisions"] += 1

        # Fast rejection for obviously valid solutions
        if np.any(~np.isfinite(coeffs)):
            self.enhanced_qp_stats["qp_activated"] += 1
            self.enhanced_qp_stats["extreme_violations"] += 1
            return True

        # Get thresholds
        if not hasattr(self, "_enhanced_thresholds") or self._enhanced_thresholds is None:
            # Fallback to basic check if thresholds not available
            return self._check_monotonicity_violation(coeffs)

        thresholds = self._enhanced_thresholds

        # Compute solution characteristics
        max_coeff = np.max(np.abs(coeffs))
        std_coeff = np.std(coeffs) if len(coeffs) > 1 else 0.0

        # Violation scoring
        violation_score = 0.0

        # Score 1: Coefficient magnitude
        if max_coeff > thresholds["extreme_violation"]:
            violation_score += 20.0
            self.enhanced_qp_stats["extreme_violations"] += 1
        elif max_coeff > thresholds["severe_violation"]:
            violation_score += 10.0
        elif max_coeff > thresholds["moderate_violation"]:
            violation_score += 3.0
        elif max_coeff > thresholds["mild_violation"] and self.qp_optimization_level == "smart":
            violation_score += 1.0  # Only for smart level

        # Score 2: Higher-order derivatives
        if len(coeffs) > 2:
            higher_order_max = np.max(np.abs(coeffs[2:]))
            if higher_order_max > thresholds["extreme_violation"]:
                violation_score += 10.0
            elif higher_order_max > thresholds["severe_violation"]:
                violation_score += 5.0

        # Score 3: Gradient magnitude
        if len(coeffs) >= 2:
            gradient_magnitude = np.linalg.norm(coeffs[:2])
            if gradient_magnitude > thresholds["gradient_threshold"]:
                violation_score += 5.0 if self.qp_optimization_level == "tuned" else 2.0

        # Score 4: Coefficient variation
        if std_coeff > thresholds["variation_threshold"]:
            violation_score += 2.0 if self.qp_optimization_level == "tuned" else 1.0

        # Context-based adjustments
        if self._current_point_idx in self.boundary_point_set:
            violation_score *= 1.2 if self.qp_optimization_level == "tuned" else 1.5
        else:
            violation_score *= 0.5 if self.qp_optimization_level == "tuned" else 0.7

        # Temporal context
        if self._current_time_ratio < 0.1:
            violation_score *= 1.1 if self.qp_optimization_level == "tuned" else 1.3
        elif self._current_time_ratio < 0.3:
            violation_score *= 1.0
        else:
            violation_score *= 0.6 if self.qp_optimization_level == "tuned" else 0.8

        # Newton iteration context
        newton_factor = 0.8**self._current_newton_iter
        violation_score *= newton_factor

        # Adaptive threshold
        if not hasattr(self, "_decision_threshold") or self._decision_threshold is None:
            current_decision_threshold = 5.0  # Default threshold
        else:
            current_decision_threshold = self._decision_threshold

        # Continuous threshold adaptation
        if self.enhanced_qp_stats is not None and self.enhanced_qp_stats["total_qp_decisions"] > 50:
            current_qp_rate = self.enhanced_qp_stats["qp_activated"] / self.enhanced_qp_stats["total_qp_decisions"]
            target_rate = self.qp_usage_target

            # Only adapt if attributes are available
            if (
                hasattr(self, "_decision_threshold")
                and hasattr(self, "_threshold_adaptation_rate")
                and self._decision_threshold is not None
                and self._threshold_adaptation_rate is not None
            ):
                if current_qp_rate > target_rate * 1.2:
                    self._decision_threshold *= 1.0 + self._threshold_adaptation_rate * 2
                    self.enhanced_qp_stats["threshold_adaptations"] += 1
                elif current_qp_rate > target_rate * 1.1:
                    self._decision_threshold *= 1.0 + self._threshold_adaptation_rate
                    self.enhanced_qp_stats["threshold_adaptations"] += 1
                elif current_qp_rate < target_rate * 0.5:
                    self._decision_threshold *= 1.0 - self._threshold_adaptation_rate
                    self.enhanced_qp_stats["threshold_adaptations"] += 1

                current_decision_threshold = self._decision_threshold

        # Final decision
        needs_qp = violation_score > current_decision_threshold

        # Update statistics
        if needs_qp:
            self.enhanced_qp_stats["qp_activated"] += 1
            if (
                hasattr(self, "boundary_point_set")
                and self.boundary_point_set is not None
                and self._current_point_idx in self.boundary_point_set
            ):
                self.enhanced_qp_stats["boundary_qp_activations"] += 1
            else:
                self.enhanced_qp_stats["interior_qp_activations"] += 1

            if self._current_time_ratio < 0.3:
                self.enhanced_qp_stats["early_time_qp_activations"] += 1
            elif self._current_time_ratio > 0.7:
                self.enhanced_qp_stats["late_time_qp_activations"] += 1
        else:
            self.enhanced_qp_stats["qp_skipped"] += 1

        return needs_qp

    def _enhanced_solve_monotone_constrained_qp(self, taylor_data: dict, b: np.ndarray, point_idx: int) -> np.ndarray:
        """Enhanced QP solve using CVXPY when available."""
        if not CVXPY_AVAILABLE:
            return self._solve_monotone_constrained_qp(taylor_data, b, point_idx)

        try:
            A = taylor_data.get("A", np.eye(len(b)))
            n_vars = A.shape[1]

            x = cp.Variable(n_vars)

            if "sqrt_W" in taylor_data:
                sqrt_W = taylor_data["sqrt_W"]
                objective = cp.Minimize(cp.sum_squares(sqrt_W @ A @ x - sqrt_W @ b))
            else:
                objective = cp.Minimize(cp.sum_squares(A @ x - b))

            # Adaptive constraints based on problem difficulty
            constraints = []

            # Check if point is in boundary set
            is_boundary = (
                hasattr(self, "boundary_point_set")
                and self.boundary_point_set is not None
                and point_idx in self.boundary_point_set
            )

            # Get problem difficulty with fallback
            problem_difficulty = getattr(self, "_problem_difficulty", 1.0)
            if problem_difficulty is None:
                problem_difficulty = 1.0

            if is_boundary:
                bound_scale = 5.0 * (1.0 + problem_difficulty)
                constraints.extend([x >= -bound_scale, x <= bound_scale])
            else:
                bound_scale = 20.0 * (1.0 + problem_difficulty)
                constraints.extend([x >= -bound_scale, x <= bound_scale])

            problem = cp.Problem(objective, constraints)

            if OSQP_AVAILABLE:
                problem.solve(
                    solver=cp.OSQP,
                    verbose=False,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    max_iter=1000,
                )
            else:
                problem.solve(solver=cp.ECOS, verbose=False, max_iters=1000)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return x.value if x.value is not None else np.zeros(n_vars)
            else:
                return self._solve_unconstrained_fallback(taylor_data, b)

        except Exception:
            return self._solve_monotone_constrained_qp(taylor_data, b, point_idx)

    def get_enhanced_qp_report(self) -> dict[str, Any]:
        """Generate enhanced QP performance report."""
        if self.enhanced_qp_stats is None:
            return {"message": "Enhanced QP not enabled. Use qp_optimization_level='smart' or 'tuned'."}

        stats = self.enhanced_qp_stats.copy()

        # Calculate rates
        total_decisions = stats["total_qp_decisions"]
        if total_decisions > 0:
            stats["qp_usage_rate"] = stats["qp_activated"] / total_decisions
            stats["qp_skip_rate"] = stats["qp_skipped"] / total_decisions
            stats["extreme_violation_rate"] = stats["extreme_violations"] / total_decisions
        else:
            stats["qp_usage_rate"] = 0.0
            stats["qp_skip_rate"] = 0.0
            stats["extreme_violation_rate"] = 0.0

        # Context analysis
        total_qp_activated = stats["qp_activated"]
        if total_qp_activated > 0:
            stats["boundary_qp_percentage"] = stats["boundary_qp_activations"] / total_qp_activated * 100
            stats["interior_qp_percentage"] = stats["interior_qp_activations"] / total_qp_activated * 100
            stats["early_time_percentage"] = stats["early_time_qp_activations"] / total_qp_activated * 100
            stats["late_time_percentage"] = stats["late_time_qp_activations"] / total_qp_activated * 100
        else:
            stats["boundary_qp_percentage"] = 0.0
            stats["interior_qp_percentage"] = 0.0
            stats["early_time_percentage"] = 0.0
            stats["late_time_percentage"] = 0.0

        # Optimization assessment
        target_rate = self.qp_usage_target
        current_rate = stats["qp_usage_rate"]

        if current_rate <= target_rate * 1.2:
            stats["optimization_quality"] = "EXCELLENT"
            stats["optimization_effectiveness"] = min(1.0, target_rate / max(0.01, current_rate))
        elif current_rate <= target_rate * 2.0:
            stats["optimization_quality"] = "GOOD"
            stats["optimization_effectiveness"] = target_rate / current_rate
        elif current_rate <= target_rate * 3.0:
            stats["optimization_quality"] = "FAIR"
            stats["optimization_effectiveness"] = target_rate / current_rate
        else:
            stats["optimization_quality"] = "POOR"
            stats["optimization_effectiveness"] = target_rate / current_rate

        # Add configuration info
        stats["optimization_level"] = self.qp_optimization_level
        stats["target_qp_rate"] = target_rate
        stats["problem_difficulty"] = getattr(self, "_problem_difficulty", 0.0)
        stats["final_decision_threshold"] = getattr(self, "_decision_threshold", 0.0)
        stats["cvxpy_available"] = CVXPY_AVAILABLE

        return stats

    def print_enhanced_qp_summary(self):
        """Print comprehensive enhanced QP performance summary."""
        if self.enhanced_qp_stats is None:
            print("Enhanced QP not enabled. Use qp_optimization_level='smart' or 'tuned'.")
            return

        stats = self.get_enhanced_qp_report()

        print(f"\n{'=' * 70}")
        print(f"ENHANCED QP ({self.qp_optimization_level.upper()}) PERFORMANCE SUMMARY")
        print(f"{'=' * 70}")

        print("Configuration:")
        print(f"  Optimization Level: {stats['optimization_level']}")
        print(f"  Target QP Usage Rate: {stats['target_qp_rate']:.1%}")
        print(f"  Problem Difficulty: {stats['problem_difficulty']:.2f}")
        print(f"  Final Decision Threshold: {stats['final_decision_threshold']:.1f}")
        print(f"  Threshold Adaptations: {stats['threshold_adaptations']}")

        print("\nQP Decision Statistics:")
        print(f"  Total QP Decisions: {stats['total_qp_decisions']}")
        print(f"  QP Activated: {stats['qp_activated']} ({stats['qp_usage_rate']:.1%})")
        print(f"  QP Skipped: {stats['qp_skipped']} ({stats['qp_skip_rate']:.1%})")
        print(f"  Extreme Violations: {stats['extreme_violations']} ({stats['extreme_violation_rate']:.1%})")

        if stats["qp_activated"] > 0:
            print("\nContext Analysis:")
            print(f"  Boundary Point QP: {stats['boundary_qp_percentage']:.1f}%")
            print(f"  Interior Point QP: {stats['interior_qp_percentage']:.1f}%")
            print(f"  Early Time QP: {stats['early_time_percentage']:.1f}%")
            print(f"  Late Time QP: {stats['late_time_percentage']:.1f}%")

        print("\nOptimization Results:")
        print(f"  Optimization Quality: {stats['optimization_quality']}")
        print(f"  Optimization Effectiveness: {stats['optimization_effectiveness']:.1%}")

        if stats["qp_skip_rate"] > 0:
            estimated_speedup = 1 / (1 - stats["qp_skip_rate"] * 0.9)
            print(f"  Estimated Speedup: {estimated_speedup:.1f}x")

        # Target assessment
        target_rate = stats["target_qp_rate"]
        current_rate = stats["qp_usage_rate"]

        if current_rate <= target_rate * 1.2:
            print("  Status: TARGET ACHIEVED")
        elif current_rate <= target_rate * 2.0:
            print("  Status: CLOSE TO TARGET")
        else:
            print("  Status: NEEDS FURTHER TUNING")

        print(f"{'=' * 70}")
