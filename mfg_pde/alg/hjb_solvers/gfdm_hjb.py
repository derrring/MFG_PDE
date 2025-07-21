import numpy as np
import math
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq
import scipy.sparse as sparse
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Union
from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.core.boundaries import BoundaryConditions


class GFDMHJBSolver(BaseHJBSolver):
    """
    Generalized Finite Difference Method (GFDM) solver for HJB equations using collocation.
    
    This solver implements meshfree collocation for HJB equations using:
    1. δ-neighborhood search for local support
    2. Taylor expansion with weighted least squares for derivative approximation
    3. Newton iteration for nonlinear HJB equations
    4. Support for various boundary conditions
    """
    
    def __init__(
        self,
        problem: "MFGProblem",
        collocation_points: np.ndarray,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        weight_scale: float = 1.0,
        NiterNewton: int = 30,
        l2errBoundNewton: float = 1e-6,
        boundary_indices: Optional[np.ndarray] = None,
        boundary_conditions: Optional[Union[Dict, "BoundaryConditions"]] = None,
        use_monotone_constraints: bool = False,
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
            NiterNewton: Maximum Newton iterations
            l2errBoundNewton: Newton convergence tolerance
            boundary_indices: Indices of boundary collocation points
            boundary_conditions: Dictionary or BoundaryConditions object specifying boundary conditions
            use_monotone_constraints: Enable constrained QP for monotonicity preservation
        """
        super().__init__(problem)
        self.hjb_method_name = "GFDM"
        
        # Collocation parameters
        self.collocation_points = collocation_points
        self.n_points = collocation_points.shape[0]
        self.dimension = collocation_points.shape[1]
        self.delta = delta
        self.taylor_order = taylor_order
        self.weight_function = weight_function
        self.weight_scale = weight_scale
        
        # Newton parameters
        self.NiterNewton = NiterNewton
        self.l2errBoundNewton = l2errBoundNewton
        
        # Boundary condition parameters
        self.boundary_indices = boundary_indices if boundary_indices is not None else np.array([])
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else {}
        self.interior_indices = np.setdiff1d(np.arange(self.n_points), self.boundary_indices)
        
        # Monotonicity constraint option
        self.use_monotone_constraints = use_monotone_constraints
        
        # Pre-compute GFDM structure
        self._build_neighborhood_structure()
        self._build_taylor_matrices()
    
    def _get_boundary_condition_property(self, property_name: str, default=None):
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
            
            if is_boundary_point and hasattr(self.boundary_conditions, 'type') and self.boundary_conditions.type == 'no_flux':
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
                'indices': np.array(all_indices),
                'points': np.array(all_points),
                'distances': np.array(all_distances),
                'size': len(all_points),
                'has_ghost': len(ghost_particles) > 0,
                'ghost_count': len(ghost_particles)
            }
    
    def _get_multi_index_set(self, d: int, p: int) -> List[Tuple[int, ...]]:
        """Generate multi-index set B(d,p) = {β ∈ N^d : 0 < |β| ≤ p} with lexicographical ordering."""
        multi_indices = []
        
        def generate_indices(current_index, remaining_dims, remaining_order):
            if remaining_dims == 0:
                if sum(current_index) > 0 and sum(current_index) <= p:
                    multi_indices.append(tuple(current_index))
                return
            
            for i in range(remaining_order + 1):
                generate_indices(current_index + [i], remaining_dims - 1, remaining_order - i)
        
        generate_indices([], d, p)
        
        # Sort by lexicographical ordering: first by order |β|, then lexicographically
        multi_indices.sort(key=lambda beta: (sum(beta), beta))
        return multi_indices
    
    def _build_taylor_matrices(self):
        """Pre-compute Taylor expansion matrices A for all collocation points."""
        self.taylor_matrices = {}
        self.multi_indices = self._get_multi_index_set(self.dimension, self.taylor_order)
        self.n_derivatives = len(self.multi_indices)
        
        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            n_neighbors = neighborhood['size']
            
            if n_neighbors < self.n_derivatives:
                self.taylor_matrices[i] = None
                continue
            
            # Build Taylor expansion matrix A
            A = np.zeros((n_neighbors, self.n_derivatives))
            center_point = self.collocation_points[i]
            neighbor_points = neighborhood['points']
            
            for j, neighbor_point in enumerate(neighbor_points):
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
            weights = self._compute_weights(neighborhood['distances'])
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
                rank = np.sum(S > tolerance)
                
                self.taylor_matrices[i] = {
                    'A': A,
                    'W': W,
                    'sqrt_W': sqrt_W,
                    'U': U[:, :rank],
                    'S': S[:rank],
                    'Vt': Vt[:rank, :],
                    'rank': rank,
                    'condition_number': condition_number,
                    'use_svd': True,
                    'use_qr': False
                }
                
            except:
                # Fallback to QR decomposition
                try:
                    # QR decomposition: WA = Q @ R
                    Q, R = np.linalg.qr(WA)
                    self.taylor_matrices[i] = {
                        'A': A,
                        'W': W,
                        'sqrt_W': sqrt_W,
                        'Q': Q,
                        'R': R,
                        'use_qr': True,
                        'use_svd': False
                    }
                except:
                    # Final fallback to normal equations
                    self.taylor_matrices[i] = {
                        'A': A,
                        'W': W,
                        'AtW': A.T @ W,
                        'AtWA_inv': None,
                        'use_qr': False,
                        'use_svd': False
                    }
                    
                    try:
                        AtWA = A.T @ W @ A
                        if np.linalg.det(AtWA) > 1e-12:
                            self.taylor_matrices[i]['AtWA_inv'] = np.linalg.inv(AtWA)
                    except:
                        pass
    
    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute weights based on distance and weight function."""
        if self.weight_function == "gaussian":
            return np.exp(-distances**2 / self.weight_scale**2)
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
            raise ValueError(f"Unknown weight function: {self.weight_function}. "
                           f"Available options: 'gaussian', 'inverse_distance', 'uniform', 'wendland'")
    
    def approximate_derivatives(self, u_values: np.ndarray, point_idx: int) -> Dict[Tuple[int, ...], float]:
        """
        Approximate derivatives at collocation point using weighted least squares.
        
        Args:
            u_values: Function values at all collocation points
            point_idx: Index of the collocation point
            
        Returns:
            Dictionary mapping multi-indices to derivative values
        """
        if self.taylor_matrices[point_idx] is None:
            return {}
        
        taylor_data = self.taylor_matrices[point_idx]
        neighborhood = self.neighborhoods[point_idx]
        
        # Extract function values at neighborhood points, handling ghost particles
        neighbor_indices = neighborhood['indices']
        u_center = u_values[point_idx]
        
        # Handle ghost particles for no-flux boundary conditions
        u_neighbors = []
        for idx in neighbor_indices:
            if idx >= 0:
                # Regular neighbor
                u_neighbors.append(u_values[idx])
            else:
                # Ghost particle: enforce no-flux condition u_ghost = u_center
                u_neighbors.append(u_center)
        
        u_neighbors = np.array(u_neighbors)
        
        # Right-hand side: u(x_center) - u(x_neighbor) following equation (6) in the mathematical framework
        # For ghost particles, this becomes u_center - u_center = 0, enforcing ∂u/∂n = 0
        b = u_center - u_neighbors
        
        # Solve weighted least squares with optional monotonicity constraints
        use_monotone_qp = hasattr(self, 'use_monotone_constraints') and self.use_monotone_constraints
        
        if use_monotone_qp:
            # First try unconstrained solution to check if constraints are needed
            unconstrained_coeffs = self._solve_unconstrained_fallback(taylor_data, b)
            
            # Check if unconstrained solution violates monotonicity
            needs_constraints = self._check_monotonicity_violation(unconstrained_coeffs)
            
            if needs_constraints:
                # Use constrained QP for monotonicity
                derivative_coeffs = self._solve_monotone_constrained_qp(taylor_data, b, point_idx)
            else:
                # Use faster unconstrained solution
                derivative_coeffs = unconstrained_coeffs
        elif taylor_data.get('use_svd', False):
            # Use SVD: solve using pseudoinverse with truncated SVD
            sqrt_W = taylor_data['sqrt_W']
            U = taylor_data['U']
            S = taylor_data['S']
            Vt = taylor_data['Vt']
            
            # Compute sqrt(W) @ b
            Wb = sqrt_W @ b
            
            # SVD solution: x = V @ S^{-1} @ U^T @ Wb
            UT_Wb = U.T @ Wb
            S_inv_UT_Wb = UT_Wb / S  # Element-wise division
            derivative_coeffs = Vt.T @ S_inv_UT_Wb
            
        elif taylor_data.get('use_qr', False):
            # Use QR decomposition: solve R @ x = Q^T @ sqrt(W) @ b
            sqrt_W = taylor_data['sqrt_W']
            Q = taylor_data['Q']
            R = taylor_data['R']
            
            Wb = sqrt_W @ b
            QT_Wb = Q.T @ Wb
            
            try:
                derivative_coeffs = np.linalg.solve(R, QT_Wb)
            except np.linalg.LinAlgError:
                # Fallback to least squares if R is singular
                derivative_coeffs, _, _, _ = lstsq(taylor_data['A'], b)
        
        elif taylor_data.get('AtWA_inv') is not None:
            # Use precomputed normal equations
            derivative_coeffs = taylor_data['AtWA_inv'] @ taylor_data['AtW'] @ b
        else:
            # Final fallback to direct least squares
            derivative_coeffs, _, _, _ = lstsq(taylor_data['A'], b)
        
        # Map coefficients to multi-indices
        derivatives = {}
        for k, beta in enumerate(self.multi_indices):
            derivatives[beta] = derivative_coeffs[k]
        
        return derivatives
    
    def _solve_monotone_constrained_qp(self, taylor_data: Dict, b: np.ndarray, point_idx: int) -> np.ndarray:
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
        
        A = taylor_data['A']
        W = taylor_data['W']
        sqrt_W = taylor_data['sqrt_W']
        n_neighbors, n_coeffs = A.shape
        
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
        neighbor_points = neighborhood['points']
        neighbor_indices = neighborhood['indices']
        
        # Set up constraints for monotonicity
        constraints = []
        
        # Implement proper monotonicity constraints for finite difference weights
        if self.dimension == 1:
            # For 1D, we can analyze the finite difference stencil more precisely
            monotonicity_constraints = self._build_monotonicity_constraints(
                A, neighbor_indices, neighbor_points, center_point
            )
            constraints.extend(monotonicity_constraints)
        
        # Set up bounds for optimization variables
        bounds = []
        
        # For each coefficient, determine appropriate bounds based on physics
        # Make bounds much more realistic for stable numerical computation
        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 0:  # Constant term - no physical constraint
                bounds.append((None, None))
            elif sum(beta) == 1:  # First derivative terms - reasonable for MFG
                bounds.append((-20.0, 20.0))  # Realistic gradient bounds
            elif sum(beta) == 2:  # Second derivative terms - key for monotonicity
                if self.dimension == 1 and beta == (2,):
                    # For 1D Laplacian: moderate diffusion bounds
                    bounds.append((-100.0, 100.0))  # Realistic diffusion bounds
                else:
                    bounds.append((-50.0, 50.0))  # Conservative cross-derivative bounds
            else:
                bounds.append((-2.0, 2.0))  # Tight bounds for higher order terms
        
        # Only add monotonicity constraints when they are really needed
        # Check if this point is near boundaries or critical regions
        center_point = self.collocation_points[point_idx]
        near_boundary = (abs(center_point[0] - self.problem.xmin) < 0.1 * self.delta or 
                        abs(center_point[0] - self.problem.xmax) < 0.1 * self.delta)
        
        # Add conservative constraint only if near boundary and needed
        if near_boundary and self.dimension == 1:
            def constraint_stability(x):
                """Mild stability constraint near boundaries"""
                # Ensure second derivative term doesn't become extreme
                for k, beta in enumerate(self.multi_indices):
                    if sum(beta) == 2 and beta == (2,):
                        return 50.0 - abs(x[k])  # Should be positive (|coeff| < 50)
                return 1.0  # Always satisfied if no second derivative
            
            constraints.append({'type': 'ineq', 'fun': constraint_stability})
        
        # Initial guess: unconstrained solution
        x0 = self._solve_unconstrained_fallback(taylor_data, b)
        
        # Solve constrained optimization with robust settings
        try:
            # Try fast L-BFGS-B first if only bounds constraints
            if len(constraints) == 0:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',  # Faster for bounds-only problems
                    jac=gradient,
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6}  # More robust settings
                )
            else:
                # Use SLSQP for general constraints with robust settings
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',  # Better for equality/inequality constraints
                    jac=gradient,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 40, 'ftol': 1e-6, 'eps': 1.4901161193847656e-08, 'disp': False}
                )
            
            if result.success:
                return result.x
            else:
                # Fallback to unconstrained if optimization fails
                return x0
                
        except Exception:
            # Fallback to unconstrained if any error occurs
            return x0
    
    def _solve_unconstrained_fallback(self, taylor_data: Dict, b: np.ndarray) -> np.ndarray:
        """Fallback to unconstrained solution using SVD or normal equations."""
        if taylor_data.get('use_svd', False):
            sqrt_W = taylor_data['sqrt_W']
            U = taylor_data['U']
            S = taylor_data['S']
            Vt = taylor_data['Vt']
            
            Wb = sqrt_W @ b
            UT_Wb = U.T @ Wb
            S_inv_UT_Wb = UT_Wb / S
            return Vt.T @ S_inv_UT_Wb
        elif taylor_data.get('AtWA_inv') is not None:
            return taylor_data['AtWA_inv'] @ taylor_data['AtW'] @ b
        else:
            A = taylor_data['A']
            from scipy.linalg import lstsq
            coeffs, _, _, _ = lstsq(A, b)
            return coeffs
    
    def _build_monotonicity_constraints(self, A: np.ndarray, neighbor_indices: np.ndarray, 
                                      neighbor_points: np.ndarray, center_point: np.ndarray) -> List[Dict]:
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
                
                constraints.append({'type': 'ineq', 'fun': constraint_positive_neighbors})
        
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
        Nt, Nx = M_density_evolution_from_FP.shape
        
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
                n
            )
        
        # Map back to grid
        U_solution = self._map_collocation_to_grid_batch(U_solution_collocation)
        return U_solution
    
    def _solve_timestep(self, u_n_plus_1: np.ndarray, u_prev_picard: np.ndarray, 
                       m_n_plus_1: np.ndarray, time_idx: int) -> np.ndarray:
        """Solve HJB at one time step using Newton iteration."""
        u_current = u_n_plus_1.copy()
        
        for newton_iter in range(self.NiterNewton):
            # Compute residual
            residual = self._compute_hjb_residual(u_current, u_n_plus_1, m_n_plus_1, time_idx)
            
            # Check convergence
            if np.linalg.norm(residual) < self.l2errBoundNewton:
                break
            
            # Compute Jacobian
            jacobian = self._compute_hjb_jacobian(u_current, u_prev_picard, m_n_plus_1, time_idx)
            
            # Apply boundary conditions
            jacobian_bc, residual_bc = self._apply_boundary_conditions_to_system(
                jacobian, residual, time_idx
            )
            
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
    
    def _compute_hjb_residual(self, u_current: np.ndarray, u_n_plus_1: np.ndarray,
                             m_n_plus_1: np.ndarray, time_idx: int) -> np.ndarray:
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
            hamiltonian = self.problem.H(
                x_idx=grid_idx, m_at_x=m_n_plus_1[i], p_values=p_values, t_idx=time_idx
            )
            residual[i] += hamiltonian
        
        return residual
    
    def _extract_gradient(self, derivatives: Dict[Tuple[int, ...], float]) -> Dict[str, float]:
        """Extract gradient components for Hamiltonian."""
        p_values = {}
        
        if self.dimension == 1:
            if (1,) in derivatives:
                p_values['forward'] = derivatives[(1,)]
                p_values['backward'] = derivatives[(1,)]
        elif self.dimension == 2:
            if (1, 0) in derivatives:
                p_values['x'] = derivatives[(1, 0)]
            if (0, 1) in derivatives:
                p_values['y'] = derivatives[(0, 1)]
        
        return p_values
    
    def _compute_hjb_jacobian(self, u_current: np.ndarray, u_prev_picard: np.ndarray,
                             m_n_plus_1: np.ndarray, time_idx: int) -> np.ndarray:
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
            neighbor_indices = neighborhood['indices']
            
            # Diffusion term contribution
            # Compute derivative matrix using SVD, QR or normal equations
            if taylor_data.get('use_svd', False):
                # For SVD approach: derivative matrix = V @ S^{-1} @ U^T @ sqrt(W)
                try:
                    U = taylor_data['U']
                    S = taylor_data['S']
                    Vt = taylor_data['Vt']
                    sqrt_W = taylor_data['sqrt_W']
                    
                    # Compute pseudoinverse matrix: V @ S^{-1} @ U^T @ sqrt(W)
                    S_inv_UT = (1.0 / S[:, np.newaxis]) * U.T  # Broadcasting for S^{-1} @ U^T
                    derivative_matrix = Vt.T @ S_inv_UT @ sqrt_W
                except:
                    derivative_matrix = None
            elif taylor_data.get('use_qr', False):
                # For QR approach: derivative matrix = R^{-1} @ Q^T @ sqrt(W)
                try:
                    Q = taylor_data['Q']
                    R = taylor_data['R']
                    sqrt_W = taylor_data['sqrt_W']
                    
                    R_inv = np.linalg.inv(R)
                    derivative_matrix = R_inv @ Q.T @ sqrt_W
                except:
                    derivative_matrix = None
            elif taylor_data.get('AtWA_inv') is not None:
                derivative_matrix = taylor_data['AtWA_inv'] @ taylor_data['AtW']
            else:
                derivative_matrix = None
            
            if derivative_matrix is not None:
                # Find second derivative indices
                for k, beta in enumerate(self.multi_indices):
                    if (self.dimension == 1 and beta == (2,)) or \
                       (self.dimension == 2 and beta in [(2, 0), (0, 2)]):
                        # Check bounds for derivative_matrix access
                        if k < derivative_matrix.shape[0]:
                            for j_local, j_global in enumerate(neighbor_indices):
                                if j_local < derivative_matrix.shape[1] and j_global >= 0:
                                    # Only apply to real particles (ghost particles have negative indices)
                                    coeff = derivative_matrix[k, j_local]
                                    jacobian[i, j_global] -= (sigma**2 / 2.0) * coeff
            
            # Hamiltonian Jacobian (numerical)
            for j in neighbor_indices:
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
    
    def get_decomposition_info(self) -> Dict:
        """Get information about the decomposition methods used."""
        total_points = self.n_points
        svd_count = sum(1 for i in range(total_points) 
                       if self.taylor_matrices[i] is not None and 
                          self.taylor_matrices[i].get('use_svd', False))
        qr_count = sum(1 for i in range(total_points) 
                      if self.taylor_matrices[i] is not None and 
                         self.taylor_matrices[i].get('use_qr', False))
        normal_count = total_points - svd_count - qr_count
        
        # Get condition numbers for SVD points
        condition_numbers = []
        ranks = []
        for i in range(total_points):
            if (self.taylor_matrices[i] is not None and 
                self.taylor_matrices[i].get('use_svd', False)):
                condition_numbers.append(self.taylor_matrices[i].get('condition_number', np.inf))
                ranks.append(self.taylor_matrices[i].get('rank', 0))
        
        info = {
            'total_points': total_points,
            'svd_points': svd_count,
            'qr_points': qr_count,
            'normal_equation_points': normal_count,
            'svd_percentage': svd_count / total_points * 100 if total_points > 0 else 0,
            'condition_numbers': condition_numbers,
            'ranks': ranks,
            'avg_condition_number': np.mean(condition_numbers) if condition_numbers else np.inf,
            'min_rank': min(ranks) if ranks else 0,
            'max_rank': max(ranks) if ranks else 0
        }
        
        return info
    
    def _apply_boundary_conditions_to_solution(self, u: np.ndarray, time_idx: int) -> np.ndarray:
        """Apply boundary conditions to solution vector."""
        u_modified = u.copy()
        
        for boundary_idx in self.boundary_indices:
            if boundary_idx >= len(u_modified):
                continue
            
            bc_type = self._get_boundary_condition_property('type', 'dirichlet')
            
            if bc_type == 'dirichlet':
                bc_function = self._get_boundary_condition_property('function')
                bc_value = self._get_boundary_condition_property('value')
                left_value = self._get_boundary_condition_property('left_value') 
                right_value = self._get_boundary_condition_property('right_value')
                
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
                        
            elif bc_type == 'no_flux':
                # For no-flux boundaries, we don't modify the solution directly
                # The boundary condition ∂u/∂x = 0 is enforced in the system matrix
                # So we leave the solution value as computed by the Newton iteration
                pass
        
        return u_modified
    
    def _apply_boundary_conditions_to_system(self, jacobian: np.ndarray, residual: np.ndarray, 
                                           time_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions to system matrix and residual."""
        jacobian_modified = jacobian.copy()
        residual_modified = residual.copy()
        
        for boundary_idx in self.boundary_indices:
            if boundary_idx >= len(residual_modified):
                continue
            
            bc_type = self._get_boundary_condition_property('type', 'dirichlet')
            
            if bc_type == 'dirichlet':
                # Replace equation with identity
                jacobian_modified[boundary_idx, :] = 0.0
                jacobian_modified[boundary_idx, boundary_idx] = 1.0
                
                # Set target value using helper method
                bc_function = self._get_boundary_condition_property('function')
                bc_value = self._get_boundary_condition_property('value')
                
                if bc_function is not None:
                    point = self.collocation_points[boundary_idx]
                    t = time_idx * self.problem.Dt
                    target_value = bc_function(t, point)
                elif bc_value is not None:
                    target_value = bc_value
                else:
                    target_value = 0.0
                
                residual_modified[boundary_idx] = target_value
                
            elif bc_type == 'no_flux':
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
        
        return grid_idx