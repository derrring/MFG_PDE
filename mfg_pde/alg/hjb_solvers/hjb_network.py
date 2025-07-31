"""
HJB Solver for Networks/Graphs.

This module implements Hamilton-Jacobi-Bellman equation solvers
for Mean Field Games on network structures.

Mathematical formulation:
∂u/∂t + H_i(m, ∇_G u, t) = 0  at node i
u(T, i) = g(i)                  terminal condition

where:
- H_i: Hamiltonian at node i  
- ∇_G: Discrete gradient on graph
- Network-specific boundary conditions

Key algorithms:
- Explicit time stepping on networks
- Implicit schemes with graph Laplacians
- Policy iteration for network control problems
- Value iteration on discrete state spaces
"""

from typing import Dict, List, Optional, TYPE_CHECKING, Union, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from ...core.network_mfg_problem import NetworkMFGProblem


class NetworkHJBSolver(BaseHJBSolver):
    """
    HJB solver for Mean Field Games on networks.
    
    Solves the discrete HJB equation:
    ∂u/∂t + H_i(m, ∇_G u, t) = 0
    
    with network-specific Hamiltonians and discrete operators.
    """
    
    def __init__(self,
                 problem: "NetworkMFGProblem",
                 scheme: str = "explicit",
                 cfl_factor: float = 0.5,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        """
        Initialize network HJB solver.
        
        Args:
            problem: Network MFG problem instance
            scheme: Time discretization ("explicit", "implicit", "semi_implicit")
            cfl_factor: CFL stability factor for explicit schemes
            max_iterations: Maximum iterations for implicit schemes
            tolerance: Convergence tolerance
        """
        super().__init__(problem)
        
        self.network_problem = problem
        self.scheme = scheme
        self.cfl_factor = cfl_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Network properties
        self.num_nodes = problem.num_nodes
        self.adjacency_matrix = problem.get_adjacency_matrix()
        self.laplacian_matrix = problem.get_laplacian_matrix()
        
        # Time discretization
        self.dt = problem.T / problem.Nt
        self.times = np.linspace(0, problem.T, problem.Nt + 1)
        
        # Solver name
        self.hjb_method_name = f"NetworkHJB_{scheme}"
        
        # Initialize discrete operators
        self._initialize_network_operators()
    
    def _initialize_network_operators(self):
        """Initialize network-specific discrete operators."""
        # Graph gradient operator (simplified representation)
        self.gradient_ops = {}
        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)
            self.gradient_ops[i] = neighbors
        
        # Stability constraint for explicit schemes
        if self.scheme == "explicit":
            max_degree = np.max(np.array(self.adjacency_matrix.sum(axis=1)).flatten())
            self.dt_stable = self.cfl_factor / (max_degree + 1e-12)
            if self.dt > self.dt_stable:
                print(f"Warning: dt={self.dt:.2e} > dt_stable={self.dt_stable:.2e}")
    
    def solve_hjb_system(self,
                        M_density_evolution: np.ndarray,
                        U_final_condition_at_T: np.ndarray,
                        U_from_prev_picard: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve HJB system on network with given density evolution.
        
        Args:
            M_density_evolution: (Nt+1, num_nodes) density evolution from FP
            U_final_condition_at_T: Terminal condition u(T, i)
            U_from_prev_picard: Previous Picard iterate for Jacobian
            
        Returns:
            (Nt+1, num_nodes) value function evolution
        """
        Nt = self.network_problem.Nt
        U = np.zeros((Nt + 1, self.num_nodes))
        
        # Set terminal condition
        U[Nt, :] = U_final_condition_at_T
        
        # Backward time stepping
        for n in range(Nt - 1, -1, -1):
            t = self.times[n]
            
            # Current density
            m_current = M_density_evolution[n, :]
            
            # Solve single time step
            if self.scheme == "explicit":
                U[n, :] = self._explicit_step(U[n + 1, :], m_current, t)
            elif self.scheme == "implicit":
                U[n, :] = self._implicit_step(U[n + 1, :], m_current, t)
            elif self.scheme == "semi_implicit":
                U[n, :] = self._semi_implicit_step(U[n + 1, :], m_current, t)
            else:
                raise ValueError(f"Unknown scheme: {self.scheme}")
            
            # Apply boundary conditions
            U[n, :] = self.network_problem.apply_boundary_conditions(U[n, :], t)
        
        return U
    
    def _explicit_step(self, u_next: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Explicit time step for network HJB."""
        u_current = u_next.copy()
        
        # For each node, compute Hamiltonian and update
        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i]
            
            # Compute network Hamiltonian
            hamiltonian = self.network_problem.hamiltonian(i, neighbors, m, u_next, t)
            
            # Forward Euler step
            u_current[i] = u_next[i] - self.dt * hamiltonian
        
        return u_current
    
    def _implicit_step(self, u_next: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Implicit time step for network HJB using fixed point iteration."""
        u_current = u_next.copy()
        
        # Fixed point iteration for implicit step
        for iteration in range(self.max_iterations):
            u_old = u_current.copy()
            
            # Update each node
            for i in range(self.num_nodes):
                neighbors = self.gradient_ops[i]
                hamiltonian = self.network_problem.hamiltonian(i, neighbors, m, u_current, t)
                
                # Implicit update
                u_current[i] = u_next[i] - self.dt * hamiltonian
            
            # Check convergence
            error = np.max(np.abs(u_current - u_old))
            if error < self.tolerance:
                break
        
        return u_current
    
    def _semi_implicit_step(self, u_next: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Semi-implicit time step handling diffusion implicitly."""
        # Split Hamiltonian into diffusion and reaction parts
        u_current = u_next.copy()
        
        # Explicit treatment of non-linear terms
        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i]
            hamiltonian = self.network_problem.hamiltonian(i, neighbors, m, u_next, t)
            u_current[i] = u_next[i] - self.dt * hamiltonian
        
        # Implicit treatment of diffusion (graph Laplacian)
        diffusion_coeff = getattr(self.network_problem.components, 'diffusion_coefficient', 1.0)
        if diffusion_coeff > 0:
            # Solve (I + dt * D * L) u = u_temp
            L = self.laplacian_matrix
            system_matrix = sp.identity(self.num_nodes) + self.dt * diffusion_coeff * L
            
            try:
                u_current = spsolve(system_matrix, u_current)
            except Exception:
                # Fallback to explicit if implicit solve fails
                u_current = u_current - self.dt * diffusion_coeff * (L @ u_current)
        
        return u_current
    
    def backward_step(self, u_next: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """
        Single backward time step (interface compatibility).
        
        Args:
            u_next: Value function at next time step
            m_current: Current density distribution
            dt: Time step size
            
        Returns:
            Updated value function
        """
        # Temporarily adjust dt for this step
        original_dt = self.dt
        self.dt = dt
        
        t = 0.0  # Dummy time (should be passed properly in full implementation)
        
        if self.scheme == "explicit":
            result = self._explicit_step(u_next, m_current, t)
        elif self.scheme == "implicit":
            result = self._implicit_step(u_next, m_current, t)
        else:
            result = self._semi_implicit_step(u_next, m_current, t)
        
        # Restore original dt
        self.dt = original_dt
        
        return result


class NetworkPolicyIterationHJBSolver(NetworkHJBSolver):
    """
    Policy iteration solver for network HJB equations.
    
    Alternates between:
    1. Policy evaluation: Solve linear system for current policy
    2. Policy improvement: Update control policy
    """
    
    def __init__(self,
                 problem: "NetworkMFGProblem",
                 max_policy_iterations: int = 50,
                 policy_tolerance: float = 1e-6,
                 **kwargs):
        """
        Initialize policy iteration HJB solver.
        
        Args:
            problem: Network MFG problem
            max_policy_iterations: Maximum policy iteration steps
            policy_tolerance: Policy convergence tolerance
            **kwargs: Additional arguments for base solver
        """
        super().__init__(problem, scheme="implicit", **kwargs)
        
        self.max_policy_iterations = max_policy_iterations
        self.policy_tolerance = policy_tolerance
        self.hjb_method_name = "NetworkHJB_PolicyIteration"
        
        # Current policy (action for each node)
        self.current_policy = {}
    
    def solve_hjb_system(self,
                        M_density_evolution: np.ndarray,
                        U_final_condition_at_T: np.ndarray,
                        U_from_prev_picard: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve HJB using policy iteration."""
        Nt = self.network_problem.Nt
        U = np.zeros((Nt + 1, self.num_nodes))
        
        # Set terminal condition
        U[Nt, :] = U_final_condition_at_T
        
        # Backward time stepping with policy iteration
        for n in range(Nt - 1, -1, -1):
            t = self.times[n]
            m_current = M_density_evolution[n, :]
            
            U[n, :] = self._policy_iteration_step(U[n + 1, :], m_current, t)
            
            # Apply boundary conditions
            U[n, :] = self.network_problem.apply_boundary_conditions(U[n, :], t)
        
        return U
    
    def _policy_iteration_step(self, u_next: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Single time step using policy iteration."""
        u_current = u_next.copy()
        
        # Initialize policy (greedy with respect to u_next)
        self._initialize_policy(u_next, m, t)
        
        # Policy iteration loop
        for policy_iter in range(self.max_policy_iterations):
            # Policy evaluation: solve linear system
            u_new = self._policy_evaluation(u_next, m, t)
            
            # Policy improvement
            old_policy = self.current_policy.copy()
            self._policy_improvement(u_new, m, t)
            
            # Check policy convergence
            if self._policies_equal(old_policy, self.current_policy):
                u_current = u_new
                break
            
            u_current = u_new
        
        return u_current
    
    def _initialize_policy(self, u: np.ndarray, m: np.ndarray, t: float):
        """Initialize policy greedily."""
        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i]
            
            if not neighbors:
                self.current_policy[i] = i  # Stay at current node
                continue
            
            # Choose neighbor that minimizes cost
            best_action = i
            best_cost = float('inf')
            
            for neighbor in neighbors:
                cost = self._compute_action_cost(i, neighbor, u, m, t)
                if cost < best_cost:
                    best_cost = cost
                    best_action = neighbor
            
            self.current_policy[i] = best_action
    
    def _policy_evaluation(self, u_next: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Evaluate current policy by solving linear system."""
        # Build linear system for policy evaluation
        A = sp.lil_matrix((self.num_nodes, self.num_nodes))
        b = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            action = self.current_policy[i]
            
            # Policy evaluation equation
            A[i, i] = 1.0 / self.dt
            
            if action != i:  # Moving to a different node
                edge_cost = self.network_problem.edge_cost(i, action, t)
                A[i, action] -= 1.0 / self.dt
                b[i] = edge_cost
            
            # Add potential and coupling terms
            potential = self.network_problem.node_potential(i, t)
            coupling = self.network_problem.density_coupling(i, m, t)
            b[i] += potential + coupling + u_next[i] / self.dt
        
        # Solve linear system
        A = A.tocsr()
        u_evaluated = spsolve(A, b)
        
        return u_evaluated
    
    def _policy_improvement(self, u: np.ndarray, m: np.ndarray, t: float):
        """Improve policy greedily."""
        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i] + [i]  # Include staying at node
            
            best_action = self.current_policy[i]
            best_cost = self._compute_action_cost(i, best_action, u, m, t)
            
            for action in neighbors:
                cost = self._compute_action_cost(i, action, u, m, t)
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
            
            self.current_policy[i] = best_action
    
    def _compute_action_cost(self, node: int, action: int, u: np.ndarray, 
                           m: np.ndarray, t: float) -> float:
        """Compute cost of taking action from node."""
        if action == node:
            # Cost of staying at node
            return (self.network_problem.node_potential(node, t) + 
                   self.network_problem.density_coupling(node, m, t))
        else:
            # Cost of moving to action node
            edge_cost = self.network_problem.edge_cost(node, action, t)
            return edge_cost + u[action]
    
    def _policies_equal(self, policy1: Dict[int, int], policy2: Dict[int, int]) -> bool:
        """Check if two policies are equal."""
        if len(policy1) != len(policy2):
            return False
        
        for node in policy1:
            if policy1[node] != policy2.get(node):
                return False
        
        return True


# Factory function for network HJB solvers
def create_network_hjb_solver(problem: "NetworkMFGProblem",
                             solver_type: str = "explicit",
                             **kwargs) -> NetworkHJBSolver:
    """
    Create network HJB solver with specified type.
    
    Args:
        problem: Network MFG problem
        solver_type: Type of solver ("explicit", "implicit", "semi_implicit", "policy_iteration")
        **kwargs: Additional solver parameters
        
    Returns:
        Configured network HJB solver
    """
    if solver_type == "policy_iteration":
        return NetworkPolicyIterationHJBSolver(problem, **kwargs)
    else:
        return NetworkHJBSolver(problem, scheme=solver_type, **kwargs)