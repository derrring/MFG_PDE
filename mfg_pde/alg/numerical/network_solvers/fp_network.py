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

from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.extensions.topology import NetworkMFGProblem


class FPNetworkSolver(BaseFPSolver):
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

    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,
        show_progress: bool = True,
        # Deprecated parameter name for backward compatibility
        m_initial_condition: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve FP system on network with unified drift and diffusion API.

        Args:
            M_initial: Initial density m₀(i)
            m_initial_condition: DEPRECATED, use M_initial
            drift_field: Drift field specification (optional):
                - None: Zero drift (pure diffusion on network)
                - np.ndarray: Precomputed drift (e.g., -∇U/λ for MFG)
                - Callable: Function α(t, x, m) -> drift (Phase 2)
            diffusion_field: Diffusion specification (optional):
                - None: Use self.diffusion_coefficient (backward compatible)
                - float: Constant diffusion on network
                - np.ndarray/Callable: Phase 2
            show_progress: Whether to display progress bar for timesteps

        Returns:
            (Nt+1, num_nodes) density evolution
        """
        import warnings

        # Handle deprecated parameter name
        if m_initial_condition is not None:
            if M_initial is not None:
                raise ValueError(
                    "Cannot specify both M_initial and m_initial_condition. "
                    "Use M_initial (m_initial_condition is deprecated)."
                )
            warnings.warn(
                "Parameter 'm_initial_condition' is deprecated. Use 'M_initial' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            M_initial = m_initial_condition

        # Validate required parameter
        if M_initial is None:
            raise ValueError("M_initial is required")

        # Handle drift_field parameter
        if drift_field is None:
            # Zero drift (pure diffusion): create zero U field for internal use
            Nt = self.network_problem.Nt
            effective_U = np.zeros((Nt + 1, self.num_nodes))
        elif isinstance(drift_field, np.ndarray):
            # Precomputed drift field (including MFG drift = -∇U/λ)
            effective_U = drift_field
        elif callable(drift_field):
            # Custom drift function - Phase 2
            raise NotImplementedError(
                "FPNetworkSolver does not yet support callable drift_field. "
                "Pass precomputed drift as np.ndarray. "
                "Support for callable drift coming in Phase 2."
            )
        else:
            raise TypeError(f"drift_field must be None, np.ndarray, or Callable, got {type(drift_field)}")

        # Handle diffusion_field parameter
        if diffusion_field is None:
            # Use self.diffusion_coefficient (backward compatible)
            effective_diffusion = self.diffusion_coefficient
        elif isinstance(diffusion_field, (int, float)):
            # Constant diffusion
            effective_diffusion = float(diffusion_field)
        elif isinstance(diffusion_field, np.ndarray) or callable(diffusion_field):
            # Spatially varying or state-dependent - Phase 2
            raise NotImplementedError(
                "FPNetworkSolver does not yet support spatially varying or callable diffusion_field. "
                "Pass constant diffusion as float. Support coming in Phase 2."
            )
        else:
            raise TypeError(
                f"diffusion_field must be None, float, np.ndarray, or Callable, got {type(diffusion_field)}"
            )

        # Temporarily override diffusion_coefficient if custom diffusion provided
        original_diffusion = self.diffusion_coefficient
        if diffusion_field is not None:
            self.diffusion_coefficient = effective_diffusion

        try:
            Nt = self.network_problem.Nt
            M = np.zeros((Nt + 1, self.num_nodes))

            # Set initial condition
            M[0, :] = M_initial

            # Normalize initial condition if needed
            if self.enforce_mass_conservation:
                total_mass = np.sum(M[0, :])
                if total_mass > 1e-12:
                    M[0, :] /= total_mass

            # Forward time stepping with progress bar
            from mfg_pde.utils.progress import tqdm

            timestep_range = range(Nt)
            if show_progress:
                timestep_range = tqdm(
                    timestep_range,
                    desc="FP (forward)",
                    unit="step",
                    disable=False,
                )

            for n in timestep_range:
                t = self.times[n]

                # Current value function for drift computation
                u_current = effective_U[n, :]

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
        finally:
            # Restore original diffusion coefficient
            self.diffusion_coefficient = original_diffusion

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


# Alias for backward compatibility
NetworkFPSolver = FPNetworkSolver


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing NetworkFPSolver...")

    # Test class availability
    assert NetworkFPSolver is not None
    print("  NetworkFPSolver class available")

    # Note: Full smoke test requires NetworkMFGProblem setup
    # See examples/networks/ for usage examples

    print("Smoke tests passed!")
