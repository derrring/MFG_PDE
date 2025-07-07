import numpy as np
import math
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq
import scipy.sparse as sparse
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


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
        weight_function: str = "gaussian",
        weight_scale: float = 1.0,
        NiterNewton: int = 30,
        l2errBoundNewton: float = 1e-6,
        boundary_indices: Optional[np.ndarray] = None,
        boundary_conditions: Optional[Dict] = None,
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
            boundary_conditions: Dictionary specifying boundary conditions
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
        
        # Pre-compute GFDM structure
        self._build_neighborhood_structure()
        self._build_taylor_matrices()
    
    def _build_neighborhood_structure(self):
        """Build δ-neighborhood structure for all collocation points."""
        self.neighborhoods = {}
        
        # Compute pairwise distances
        distances = cdist(self.collocation_points, self.collocation_points)
        
        for i in range(self.n_points):
            # Find neighbors within delta radius
            neighbor_mask = distances[i, :] < self.delta
            neighbor_indices = np.where(neighbor_mask)[0]
            
            # Store neighborhood information
            self.neighborhoods[i] = {
                'indices': neighbor_indices,
                'points': self.collocation_points[neighbor_indices],
                'distances': distances[i, neighbor_indices],
                'size': len(neighbor_indices)
            }
    
    def _get_multi_index_set(self, d: int, p: int) -> List[Tuple[int, ...]]:
        """Generate multi-index set B(d,p) = {β ∈ N^d : 0 < |β| ≤ p}."""
        multi_indices = []
        
        def generate_indices(current_index, remaining_dims, remaining_order):
            if remaining_dims == 0:
                if sum(current_index) > 0 and sum(current_index) <= p:
                    multi_indices.append(tuple(current_index))
                return
            
            for i in range(remaining_order + 1):
                generate_indices(current_index + [i], remaining_dims - 1, remaining_order - i)
        
        generate_indices([], d, p)
        multi_indices.sort()
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
            
            self.taylor_matrices[i] = {
                'A': A,
                'W': W,
                'AtW': A.T @ W,
                'AtWA_inv': None
            }
            
            # Pre-compute (A^T W A)^{-1} for least squares
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
        else:
            raise ValueError(f"Unknown weight function: {self.weight_function}")
    
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
        
        # Extract function values at neighborhood points
        neighbor_indices = neighborhood['indices']
        u_center = u_values[point_idx]
        u_neighbors = u_values[neighbor_indices]
        
        # Right-hand side: u(x_neighbor) - u(x_center) for proper Taylor expansion
        b = u_neighbors - u_center
        
        # Solve weighted least squares
        if taylor_data['AtWA_inv'] is not None:
            derivative_coeffs = taylor_data['AtWA_inv'] @ taylor_data['AtW'] @ b
        else:
            # Fallback to lstsq
            derivative_coeffs, _, _, _ = lstsq(taylor_data['A'], b)
        
        # Map coefficients to multi-indices
        derivatives = {}
        for k, beta in enumerate(self.multi_indices):
            derivatives[beta] = derivative_coeffs[k]
        
        return derivatives
    
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
            if taylor_data['AtWA_inv'] is not None:
                derivative_matrix = taylor_data['AtWA_inv'] @ taylor_data['AtW']
                
                # Find second derivative indices
                for k, beta in enumerate(self.multi_indices):
                    if (self.dimension == 1 and beta == (2,)) or \
                       (self.dimension == 2 and beta in [(2, 0), (0, 2)]):
                        for j_local, j_global in enumerate(neighbor_indices):
                            coeff = derivative_matrix[k, j_local]
                            jacobian[i, j_global] -= (sigma**2 / 2.0) * coeff
            
            # Hamiltonian Jacobian (numerical)
            for j in neighbor_indices:
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
    
    def _apply_boundary_conditions_to_solution(self, u: np.ndarray, time_idx: int) -> np.ndarray:
        """Apply boundary conditions to solution vector."""
        u_modified = u.copy()
        
        for boundary_idx in self.boundary_indices:
            if boundary_idx >= len(u_modified):
                continue
            
            bc_type = self.boundary_conditions.get('type', 'dirichlet')
            
            if bc_type == 'dirichlet':
                if 'function' in self.boundary_conditions:
                    point = self.collocation_points[boundary_idx]
                    t = time_idx * self.problem.Dt
                    u_modified[boundary_idx] = self.boundary_conditions['function'](t, point)
                elif 'value' in self.boundary_conditions:
                    u_modified[boundary_idx] = self.boundary_conditions['value']
        
        return u_modified
    
    def _apply_boundary_conditions_to_system(self, jacobian: np.ndarray, residual: np.ndarray, 
                                           time_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions to system matrix and residual."""
        jacobian_modified = jacobian.copy()
        residual_modified = residual.copy()
        
        for boundary_idx in self.boundary_indices:
            if boundary_idx >= len(residual_modified):
                continue
            
            bc_type = self.boundary_conditions.get('type', 'dirichlet')
            
            if bc_type == 'dirichlet':
                # Replace equation with identity
                jacobian_modified[boundary_idx, :] = 0.0
                jacobian_modified[boundary_idx, boundary_idx] = 1.0
                
                # Set target value
                if 'function' in self.boundary_conditions:
                    point = self.collocation_points[boundary_idx]
                    t = time_idx * self.problem.Dt
                    target_value = self.boundary_conditions['function'](t, point)
                elif 'value' in self.boundary_conditions:
                    target_value = self.boundary_conditions['value']
                else:
                    target_value = 0.0
                
                residual_modified[boundary_idx] = target_value
        
        return jacobian_modified, residual_modified
    
    def _map_grid_to_collocation(self, u_grid: np.ndarray) -> np.ndarray:
        """Map grid values to collocation points."""
        if u_grid.shape[0] != self.n_points:
            # Simple nearest neighbor mapping
            grid_x = np.linspace(0, 1, self.problem.Nx)
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
        if u_collocation.shape[0] != self.problem.Nx:
            grid_x = np.linspace(0, 1, self.problem.Nx)
            u_grid = np.zeros(self.problem.Nx)
            
            for i in range(self.problem.Nx):
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
        U_grid = np.zeros((U_collocation.shape[0], self.problem.Nx))
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
        grid_x = np.linspace(self.problem.xmin, self.problem.xmax, self.problem.Nx)
        distances = np.abs(grid_x - physical_x)
        grid_idx = np.argmin(distances)
        
        return grid_idx