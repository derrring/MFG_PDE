"""
Fokker-Planck Solver for Networks/Graphs.

This module implements Fokker-Planck equation solvers for Mean Field Games
on network structures, handling discrete density evolution on graphs.

Mathematical formulation:
∂m/∂t - div_G(m ∇_G H_p) - σ²/2 Δ_G m = 0  on network
m(0, i) = m_0(i)                             initial condition

where:
- div_G: Graph divergence operator
- Δ_G: Graph Laplacian operator
- H_p: Derivative of Hamiltonian w.r.t. momentum
- σ²: Diffusion coefficient on network

Key algorithms:
- Explicit/implicit time stepping on networks
- Flow-based density evolution along edges
- Conservation-preserving schemes
- Network boundary condition handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .base_fp import BaseFPSolver

if TYPE_CHECKING:
    from ...core.network_mfg_problem import NetworkMFGProblem


class NetworkFPSolver(BaseFPSolver):
    """
    Fokker-Planck solver for Mean Field Games on networks.

    Solves the discrete FP equation:
    ∂m/∂t - div_G(m ∇_G H_p) - σ²/2 Δ_G m = 0

    with network-specific operators and mass conservation.
    """

    def __init__(
        self,
        problem: NetworkMFGProblem,
        scheme: str = "explicit",
        diffusion_coefficient: float = 0.1,
        cfl_factor: float = 0.5,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        enforce_mass_conservation: bool = True,
    ):
        """
        Initialize network FP solver.

        Args:
            problem: Network MFG problem instance
            scheme: Time discretization ("explicit", "implicit", "upwind")
            diffusion_coefficient: Network diffusion coefficient σ²/2
            cfl_factor: CFL stability factor for explicit schemes
            max_iterations: Maximum iterations for implicit schemes
            tolerance: Convergence tolerance
            enforce_mass_conservation: Whether to enforce mass conservation
        """
        super().__init__(problem)

        self.network_problem = problem
        self.scheme = scheme
        self.diffusion_coefficient = diffusion_coefficient
        self.cfl_factor = cfl_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.enforce_mass_conservation = enforce_mass_conservation

        # Network properties
        self.num_nodes = problem.num_nodes
        self.adjacency_matrix = problem.get_adjacency_matrix()
        self.laplacian_matrix = problem.get_laplacian_matrix()

        # Time discretization
        self.dt = problem.T / problem.Nt
        self.times = np.linspace(0, problem.T, problem.Nt + 1)

        # Solver name
        self.fp_method_name = f"NetworkFP_{scheme}"

        # Initialize discrete operators
        self._initialize_network_operators()

    def _initialize_network_operators(self):
        """Initialize network-specific discrete operators."""
        # Graph divergence operator structure
        self.divergence_ops = {}
        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)
            self.divergence_ops[i] = neighbors

        # Stability constraint for explicit schemes
        if self.scheme == "explicit":
            max_degree = np.max(np.array(self.adjacency_matrix.sum(axis=1)).flatten())
            self.dt_stable = self.cfl_factor / (max_degree * self.diffusion_coefficient + 1e-12)
            if self.dt > self.dt_stable:
                print(f"Warning: dt={self.dt:.2e} > dt_stable={self.dt_stable:.2e}")

    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """
        Solve FP system on network with given control field.

        Args:
            m_initial_condition: Initial density m_0(i)
            U_solution_for_drift: (Nt+1, num_nodes) value function for drift

        Returns:
            (Nt+1, num_nodes) density evolution
        """
        Nt = self.network_problem.Nt
        M = np.zeros((Nt + 1, self.num_nodes))

        # Set initial condition
        M[0, :] = m_initial_condition

        # Normalize initial condition if needed
        if self.enforce_mass_conservation:
            total_mass = np.sum(M[0, :])
            if total_mass > 1e-12:
                M[0, :] /= total_mass

        # Forward time stepping
        for n in range(Nt):
            t = self.times[n]

            # Current value function for drift computation
            u_current = U_solution_for_drift[n, :]

            # Solve single time step
            if self.scheme == "explicit":
                M[n + 1, :] = self._explicit_step(M[n, :], u_current, t)
            elif self.scheme == "implicit":
                M[n + 1, :] = self._implicit_step(M[n, :], u_current, t)
            elif self.scheme == "upwind":
                M[n + 1, :] = self._upwind_step(M[n, :], u_current, t)
            else:
                raise ValueError(f"Unknown scheme: {self.scheme}")

            # Enforce non-negativity and mass conservation
            M[n + 1, :] = np.maximum(M[n + 1, :], 0)

            if self.enforce_mass_conservation:
                total_mass = np.sum(M[n + 1, :])
                if total_mass > 1e-12:
                    M[n + 1, :] /= total_mass

        return M

    def _explicit_step(self, m_current: np.ndarray, u_current: np.ndarray, t: float) -> np.ndarray:
        """Explicit time step for network FP equation."""
        m_next = m_current.copy()

        # Compute flows for each node
        for i in range(self.num_nodes):
            neighbors = self.divergence_ops[i]

            # Diffusion term: -σ²/2 * Δ_G m
            diffusion_term = 0.0
            for j in neighbors:
                edge_weight = (
                    self.network_problem.network_data.get_edge_weight(i, j)
                    if self.network_problem.network_data is not None
                    else 1.0  # Default edge weight
                )
                diffusion_term += edge_weight * (m_current[j] - m_current[i])
            diffusion_term *= self.diffusion_coefficient

            # Drift term: -div_G(m ∇_G H_p)
            drift_term = self._compute_drift_term(i, m_current, u_current, t)

            # Forward Euler step
            m_next[i] = m_current[i] + self.dt * (diffusion_term + drift_term)

        return m_next

    def _implicit_step(self, m_current: np.ndarray, u_current: np.ndarray, t: float) -> np.ndarray:
        """Implicit time step for network FP equation."""
        # Build linear system for implicit step
        A = sp.lil_matrix((self.num_nodes, self.num_nodes))
        b = m_current.copy()

        for i in range(self.num_nodes):
            neighbors = self.divergence_ops[i]

            # Diagonal term
            A[i, i] = 1.0

            # Diffusion terms (implicit)
            for j in neighbors:
                edge_weight = (
                    self.network_problem.network_data.get_edge_weight(i, j)
                    if self.network_problem.network_data is not None
                    else 1.0  # Default edge weight
                )
                coeff = self.dt * self.diffusion_coefficient * edge_weight

                A[i, i] += coeff
                A[i, j] -= coeff

            # Drift terms (explicit for simplicity)
            drift_term = self._compute_drift_term(i, m_current, u_current, t)
            b[i] += self.dt * drift_term

        # Solve linear system
        A = A.tocsr()
        try:
            m_next = spsolve(A, b)
            # Ensure result is numpy array
            if hasattr(m_next, "toarray"):
                m_next = m_next.toarray().flatten()
            m_next = np.asarray(m_next)
        except Exception:
            # Fallback to explicit if implicit solve fails
            print("Warning: Implicit solve failed, falling back to explicit")
            m_next = self._explicit_step(m_current, u_current, t)

        return m_next

    def _upwind_step(self, m_current: np.ndarray, u_current: np.ndarray, t: float) -> np.ndarray:
        """Upwind scheme for network FP equation (conservation-preserving)."""
        m_next = m_current.copy()

        # Compute flows between neighboring nodes
        for i in range(self.num_nodes):
            neighbors = self.divergence_ops[i]

            net_flow = 0.0

            for j in neighbors:
                # Compute flow from j to i
                flow_ji = self._compute_edge_flow(j, i, m_current, u_current, t)
                # Compute flow from i to j
                flow_ij = self._compute_edge_flow(i, j, m_current, u_current, t)

                # Net flow into node i
                net_flow += flow_ji - flow_ij

            # Update with net flow
            m_next[i] = m_current[i] + self.dt * net_flow

        return m_next

    def _compute_drift_term(self, node: int, m: np.ndarray, u: np.ndarray, t: float) -> float:
        """Compute drift term for FP equation at given node."""
        neighbors = self.divergence_ops[node]

        if not neighbors:
            return 0.0

        # Simplified drift computation based on value function gradient
        drift = 0.0
        for neighbor in neighbors:
            # Gradient of value function
            du = u[neighbor] - u[node]

            # Flow along gradient (simplified)
            edge_weight = (
                self.network_problem.network_data.get_edge_weight(node, neighbor)
                if self.network_problem.network_data is not None
                else 1.0  # Default edge weight
            )
            drift += edge_weight * m[node] * du

        return drift

    def _compute_edge_flow(self, node_from: int, node_to: int, m: np.ndarray, u: np.ndarray, t: float) -> float:
        """Compute flow along edge from node_from to node_to."""
        if node_from == node_to:
            return 0.0

        # Check if edge exists
        edge_weight = (
            self.network_problem.network_data.get_edge_weight(node_from, node_to)
            if self.network_problem.network_data is not None
            else 1.0  # Default edge weight
        )
        if edge_weight == 0:
            return 0.0

        # Upwind density selection
        du = u[node_to] - u[node_from]

        if du > 0:  # Flow from node_from to node_to
            density = m[node_from]
        else:  # Flow from node_to to node_from
            density = m[node_to]
            du = -du

        # Flow magnitude
        flow = edge_weight * density * du

        return flow

    def forward_step(self, m_prev: np.ndarray, u_current: np.ndarray, dt: float) -> np.ndarray:
        """
        Single forward time step (interface compatibility).

        Args:
            m_prev: Density at previous time step
            u_current: Current value function
            dt: Time step size

        Returns:
            Updated density distribution
        """
        # Temporarily adjust dt for this step
        original_dt = self.dt
        self.dt = dt

        t = 0.0  # Dummy time (should be passed properly in full implementation)

        if self.scheme == "explicit":
            result = self._explicit_step(m_prev, u_current, t)
        elif self.scheme == "implicit":
            result = self._implicit_step(m_prev, u_current, t)
        else:
            result = self._upwind_step(m_prev, u_current, t)

        # Restore original dt
        self.dt = original_dt

        # Enforce constraints
        result = np.maximum(result, 0)
        if self.enforce_mass_conservation:
            total_mass = np.sum(result)
            if total_mass > 1e-12:
                result /= total_mass

        return result


class NetworkFlowFPSolver(NetworkFPSolver):
    """
    Flow-based FP solver for networks using edge-based density evolution.

    This solver explicitly tracks density flows along network edges,
    providing better conservation properties and physical interpretation.
    """

    def __init__(self, problem: NetworkMFGProblem, **kwargs):
        """Initialize flow-based network FP solver."""
        super().__init__(problem, **kwargs)

        self.fp_method_name = "NetworkFlowFP"
        self.num_edges = problem.num_edges

        # Edge-based data structures
        self.edge_list = self._build_edge_list()
        self.node_to_edges = self._build_node_edge_mapping()

    def _build_edge_list(self) -> list[tuple[int, int]]:
        """Build list of edges from adjacency matrix."""
        edges = []
        rows, cols = self.adjacency_matrix.nonzero()

        # For undirected graphs, avoid duplicate edges
        for i, j in zip(rows, cols, strict=False):
            if i < j:  # Only include each edge once
                edges.append((i, j))

        return edges

    def _build_node_edge_mapping(self) -> dict[int, list[int]]:
        """Build mapping from nodes to incident edges."""
        node_edges = {i: [] for i in range(self.num_nodes)}

        for edge_idx, (i, j) in enumerate(self.edge_list):
            node_edges[i].append(edge_idx)
            node_edges[j].append(edge_idx)

        return node_edges

    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """Solve FP system using flow-based approach."""
        Nt = self.network_problem.Nt
        M = np.zeros((Nt + 1, self.num_nodes))
        M[0, :] = m_initial_condition

        # Normalize initial condition
        if self.enforce_mass_conservation:
            total_mass = np.sum(M[0, :])
            if total_mass > 1e-12:
                M[0, :] /= total_mass

        # Forward time stepping using flow conservation
        for n in range(Nt):
            t = self.times[n]
            u_current = U_solution_for_drift[n, :]

            M[n + 1, :] = self._flow_based_step(M[n, :], u_current, t)

            # Enforce constraints
            M[n + 1, :] = np.maximum(M[n + 1, :], 0)
            if self.enforce_mass_conservation:
                total_mass = np.sum(M[n + 1, :])
                if total_mass > 1e-12:
                    M[n + 1, :] /= total_mass

        return M

    def _flow_based_step(self, m_current: np.ndarray, u_current: np.ndarray, t: float) -> np.ndarray:
        """Flow-based time step preserving mass conservation."""
        m_next = m_current.copy()

        # Compute flows along all edges
        edge_flows = np.zeros(len(self.edge_list))

        for edge_idx, (i, j) in enumerate(self.edge_list):
            # Flow from i to j
            flow_ij = self._compute_directed_edge_flow(i, j, m_current, u_current, t)
            edge_flows[edge_idx] = flow_ij

        # Update node densities based on edge flows
        for node in range(self.num_nodes):
            net_flow = 0.0

            for edge_idx in self.node_to_edges[node]:
                i, j = self.edge_list[edge_idx]
                flow = edge_flows[edge_idx]

                if i == node:
                    net_flow -= flow  # Outgoing flow
                else:
                    net_flow += flow  # Incoming flow

            m_next[node] = m_current[node] + self.dt * net_flow

        return m_next

    def _compute_directed_edge_flow(
        self, node_from: int, node_to: int, m: np.ndarray, u: np.ndarray, t: float
    ) -> float:
        """Compute directed flow from node_from to node_to."""
        # Value function difference
        du = u[node_to] - u[node_from]

        # Edge weight
        edge_weight = (
            self.network_problem.network_data.get_edge_weight(node_from, node_to)
            if self.network_problem.network_data is not None
            else 1.0  # Default edge weight
        )

        # Drift-driven flow
        drift_flow = edge_weight * m[node_from] * max(du, 0)

        # Diffusion flow
        diffusion_flow = self.diffusion_coefficient * edge_weight * (m[node_to] - m[node_from])

        return drift_flow + diffusion_flow


# Factory function for network FP solvers
def create_network_fp_solver(problem: NetworkMFGProblem, solver_type: str = "explicit", **kwargs) -> NetworkFPSolver:
    """
    Create network FP solver with specified type.

    Args:
        problem: Network MFG problem
        solver_type: Type of solver ("explicit", "implicit", "upwind", "flow")
        **kwargs: Additional solver parameters

    Returns:
        Configured network FP solver
    """
    if solver_type == "flow":
        return NetworkFlowFPSolver(problem, **kwargs)
    else:
        return NetworkFPSolver(problem, scheme=solver_type, **kwargs)
