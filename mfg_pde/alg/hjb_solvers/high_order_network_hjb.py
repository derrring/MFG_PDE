"""
Network HJB Solvers with High-Order Discretization Schemes.

This module implements numerical methods for HJB equations on networks,
featuring upwind schemes and high-order discretization techniques.

Key features:
- Network-adapted upwind schemes
- High-order discretization methods
- Non-global continuity boundary conditions
- Monotone schemes for network Hamilton-Jacobi equations
- Adaptive time stepping
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .hjb_network import NetworkHJBSolver

if TYPE_CHECKING:
    from ...core.network_mfg_problem import NetworkMFGProblem


class HighOrderNetworkHJBSolver(NetworkHJBSolver):
    """
    HJB solver with high-order discretization schemes for networks.

    Implements numerical methods including network-adapted upwind schemes,
    Lax-Friedrichs, and Godunov methods with adaptive time stepping.
    """

    def __init__(
        self,
        problem: "NetworkMFGProblem",
        scheme: str = "network_upwind",
        order: int = 2,
        adaptive_time_step: bool = False,
        monotone_scheme: bool = True,
        **kwargs,
    ):
        """
        Initialize high-order network HJB solver.

        Args:
            problem: Network MFG problem instance
            scheme: Discretization scheme ("network_upwind", "lax_friedrichs", "godunov")
            order: Order of accuracy (1 or 2)
            adaptive_time_step: Use adaptive time stepping
            monotone_scheme: Ensure monotonicity preservation
            **kwargs: Additional arguments for base solver
        """
        super().__init__(problem, **kwargs)

        self.discretization_scheme = scheme
        self.order = order
        self.adaptive_time_step = adaptive_time_step
        self.monotone_scheme = monotone_scheme

        # High-order discretization parameters
        self.artificial_viscosity = kwargs.get("artificial_viscosity", 0.01)
        self.flux_limiter = kwargs.get("flux_limiter", "minmod")

        # Solver name
        self.hjb_method_name = f"HighOrderNetworkHJB_{scheme}_O{order}"

        # Initialize high-order operators
        self._initialize_high_order_operators()

    def _initialize_high_order_operators(self):
        """Initialize high-order discretization operators."""
        # Build gradient reconstruction matrices
        self.gradient_matrices = self._build_gradient_matrices()

        # Build flux matrices for different schemes
        self.flux_matrices = self._build_flux_matrices()

        # Initialize adaptive time step parameters
        if self.adaptive_time_step:
            self.dt_min = self.dt * 0.1
            self.dt_max = self.dt * 2.0
            self.dt_current = self.dt

    def _build_gradient_matrices(self) -> Dict[str, csr_matrix]:
        """Build gradient reconstruction matrices for higher-order schemes."""
        gradient_matrices = {}

        # First-order gradient (standard)
        rows, cols, data = [], [], []
        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)
            degree = len(neighbors)

            if degree > 0:
                # Central node
                rows.append(i)
                cols.append(i)
                data.append(-degree)

                # Neighbors
                for j in neighbors:
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)

        gradient_matrices["first_order"] = csr_matrix(
            (data, (rows, cols)), shape=(self.num_nodes, self.num_nodes)
        )

        # Second-order gradient (if requested)
        if self.order >= 2:
            gradient_matrices["second_order"] = self._build_second_order_gradient()

        return gradient_matrices

    def _build_second_order_gradient(self) -> csr_matrix:
        """Build second-order gradient matrix using network topology."""
        rows, cols, data = [], [], []

        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)

            if len(neighbors) >= 2:
                # Use least squares approach for second-order reconstruction
                for j in neighbors:
                    # Second-order weights (simplified)
                    weight = 1.0 / len(neighbors)

                    rows.append(i)
                    cols.append(j)
                    data.append(weight)

                    rows.append(i)
                    cols.append(i)
                    data.append(-weight)

        return csr_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes))

    def _build_flux_matrices(self) -> Dict[str, csr_matrix]:
        """Build flux matrices for different discretization schemes."""
        flux_matrices = {}

        # Network upwind flux
        flux_matrices["upwind"] = self._build_upwind_flux_matrix()

        # Lax-Friedrichs flux
        flux_matrices["lax_friedrichs"] = self._build_lax_friedrichs_flux_matrix()

        # Godunov flux (simplified)
        flux_matrices["godunov"] = self._build_godunov_flux_matrix()

        return flux_matrices

    def _build_upwind_flux_matrix(self) -> csr_matrix:
        """Build upwind flux matrix for network Hamilton-Jacobi equations."""
        rows, cols, data = [], [], []

        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)

            for j in neighbors:
                edge_weight = self.network_data.get_edge_weight(i, j)

                # Upwind weighting based on edge direction
                upwind_weight = edge_weight

                rows.extend([i, i])
                cols.extend([i, j])
                data.extend([-upwind_weight, upwind_weight])

        return csr_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes))

    def _build_lax_friedrichs_flux_matrix(self) -> csr_matrix:
        """Build Lax-Friedrichs flux matrix with artificial viscosity."""
        rows, cols, data = [], [], []

        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)
            degree = len(neighbors)

            if degree > 0:
                # Central node contribution
                central_weight = -degree * (1.0 + self.artificial_viscosity)
                rows.append(i)
                cols.append(i)
                data.append(central_weight)

                # Neighbor contributions
                for j in neighbors:
                    edge_weight = self.network_data.get_edge_weight(i, j)
                    neighbor_weight = edge_weight * (
                        1.0 + self.artificial_viscosity / degree
                    )

                    rows.append(i)
                    cols.append(j)
                    data.append(neighbor_weight)

        return csr_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes))

    def _build_godunov_flux_matrix(self) -> csr_matrix:
        """Build Godunov flux matrix (simplified for networks)."""
        # For now, use upwind as base for Godunov
        return self._build_upwind_flux_matrix()

    def solve_hjb_system(
        self,
        M_density_evolution: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve HJB system using high-order discretization schemes.

        Args:
            M_density_evolution: (Nt+1, num_nodes) density evolution
            U_final_condition_at_T: Terminal condition
            U_from_prev_picard: Previous Picard iterate

        Returns:
            (Nt+1, num_nodes) value function evolution
        """
        Nt = self.network_problem.Nt
        U = np.zeros((Nt + 1, self.num_nodes))

        # Set terminal condition
        U[Nt, :] = U_final_condition_at_T

        # Backward time stepping with advanced schemes
        for n in range(Nt - 1, -1, -1):
            t = self.times[n]
            m_current = M_density_evolution[n, :]

            # Choose time step (adaptive if enabled)
            if self.adaptive_time_step:
                dt_step = self._compute_adaptive_time_step(U[n + 1, :], m_current)
            else:
                dt_step = self.dt

            # Apply high-order discretization scheme
            if self.discretization_scheme == "network_upwind":
                U[n, :] = self._network_upwind_step(U[n + 1, :], m_current, t, dt_step)
            elif self.discretization_scheme == "lax_friedrichs":
                U[n, :] = self._lax_friedrichs_step(U[n + 1, :], m_current, t, dt_step)
            elif self.discretization_scheme == "godunov":
                U[n, :] = self._godunov_step(U[n + 1, :], m_current, t, dt_step)
            else:
                # Fallback to standard scheme
                U[n, :] = self._explicit_step(U[n + 1, :], m_current, t)

            # Apply boundary conditions with non-global continuity
            U[n, :] = self._apply_non_global_boundary_conditions(U[n, :], t)

            # Ensure monotonicity if requested
            if self.monotone_scheme:
                U[n, :] = self._enforce_monotonicity(U[n, :], U[n + 1, :])

        return U

    def _network_upwind_step(
        self, u_next: np.ndarray, m: np.ndarray, t: float, dt: float
    ) -> np.ndarray:
        """Network-adapted upwind scheme for HJB equation."""
        u_current = u_next.copy()

        # Apply upwind flux matrix
        flux_matrix = self.flux_matrices["upwind"]

        # Compute spatial derivatives using upwind differences
        spatial_term = flux_matrix @ u_next

        # Add Hamiltonian terms
        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i]

            # Compute Hamiltonian with upwind velocity
            hamiltonian = self._compute_upwind_hamiltonian(i, neighbors, m, u_next, t)

            # Update with upwind scheme
            u_current[i] = u_next[i] - dt * (hamiltonian + spatial_term[i])

        return u_current

    def _lax_friedrichs_step(
        self, u_next: np.ndarray, m: np.ndarray, t: float, dt: float
    ) -> np.ndarray:
        """Lax-Friedrichs scheme with artificial viscosity."""
        u_current = u_next.copy()

        # Apply Lax-Friedrichs flux matrix
        flux_matrix = self.flux_matrices["lax_friedrichs"]
        spatial_term = flux_matrix @ u_next

        # Add viscosity term
        viscosity_term = self.artificial_viscosity * (self.laplacian_matrix @ u_next)

        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i]
            hamiltonian = self.network_problem.hamiltonian(i, neighbors, m, u_next, t)

            u_current[i] = u_next[i] - dt * (
                hamiltonian + spatial_term[i] - viscosity_term[i]
            )

        return u_current

    def _godunov_step(
        self, u_next: np.ndarray, m: np.ndarray, t: float, dt: float
    ) -> np.ndarray:
        """Godunov scheme for network Hamilton-Jacobi equations."""
        u_current = u_next.copy()

        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i]

            # Solve local Riemann problems (simplified)
            riemann_fluxes = []
            for j in neighbors:
                flux = self._solve_network_riemann_problem(i, j, u_next, m, t)
                riemann_fluxes.append(flux)

            # Update using Godunov fluxes
            total_flux = sum(riemann_fluxes)
            hamiltonian = self.network_problem.hamiltonian(i, neighbors, m, u_next, t)

            u_current[i] = u_next[i] - dt * (hamiltonian + total_flux)

        return u_current

    def _solve_network_riemann_problem(
        self, node_i: int, node_j: int, u: np.ndarray, m: np.ndarray, t: float
    ) -> float:
        """Solve Riemann problem between two nodes (simplified)."""
        # Simplified Riemann solver for networks
        ui, uj = u[node_i], u[node_j]

        # Compute flux based on upwind principle
        if ui >= uj:
            # Flow from i to j
            flux = self.network_data.get_edge_weight(node_i, node_j) * (ui - uj)
        else:
            # Flow from j to i
            flux = -self.network_data.get_edge_weight(node_i, node_j) * (uj - ui)

        return flux

    def _compute_upwind_hamiltonian(
        self, node: int, neighbors: List[int], m: np.ndarray, p: np.ndarray, t: float
    ) -> float:
        """Compute Hamiltonian using upwind discretization."""
        # Enhanced Hamiltonian with upwind treatment
        control_cost = 0.0

        for neighbor in neighbors:
            edge_weight = self.network_data.get_edge_weight(node, neighbor)

            # Upwind gradient
            dp = p[neighbor] - p[node]

            # Upwind velocity selection
            if dp > 0:
                velocity = max(0, dp)  # Forward flow
            else:
                velocity = min(0, dp)  # Backward flow

            control_cost += 0.5 * edge_weight * velocity**2

        # Add potential and coupling
        potential = self.network_problem.node_potential(node, t)
        coupling = self.network_problem.density_coupling(node, m, t)

        return control_cost + potential + coupling

    def _compute_adaptive_time_step(self, u: np.ndarray, m: np.ndarray) -> float:
        """Compute adaptive time step based on CFL condition."""
        # Compute maximum velocity on network
        max_velocity = 0.0

        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)
            for j in neighbors:
                velocity = abs(u[j] - u[i])
                max_velocity = max(max_velocity, velocity)

        # CFL condition for networks
        cfl_dt = self.cfl_factor / (max_velocity + 1e-12)

        # Clamp to reasonable bounds
        dt_adaptive = np.clip(cfl_dt, self.dt_min, self.dt_max)

        return dt_adaptive

    def _apply_non_global_boundary_conditions(
        self, u: np.ndarray, t: float
    ) -> np.ndarray:
        """Apply boundary conditions with non-global continuity."""
        u_bc = u.copy()

        # Standard boundary conditions
        u_bc = self.network_problem.apply_boundary_conditions(u_bc, t)

        # Non-global continuity treatment
        if hasattr(self.network_problem.components, "non_global_continuity"):
            if self.network_problem.components.non_global_continuity:
                u_bc = self._apply_non_global_continuity(u_bc, t)

        return u_bc

    def _apply_non_global_continuity(self, u: np.ndarray, t: float) -> np.ndarray:
        """Apply non-global continuity conditions on network edges."""
        u_modified = u.copy()

        # Allow discontinuities across certain edges
        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)

            for j in neighbors:
                # Check if this edge allows discontinuity
                edge_weight = self.network_data.get_edge_weight(i, j)

                # Simple criterion: allow discontinuity on weak edges
                if edge_weight < 0.5:  # Threshold for discontinuity
                    # Don't enforce continuity across this edge
                    continue
                else:
                    # Enforce some level of continuity
                    avg_value = 0.5 * (u[i] + u[j])
                    u_modified[i] = 0.8 * u[i] + 0.2 * avg_value
                    u_modified[j] = 0.8 * u[j] + 0.2 * avg_value

        return u_modified

    def _enforce_monotonicity(
        self, u_current: np.ndarray, u_next: np.ndarray
    ) -> np.ndarray:
        """Enforce monotonicity preservation in time."""
        if not self.monotone_scheme:
            return u_current

        # Simple monotonicity enforcement
        u_monotone = u_current.copy()

        # Ensure solution doesn't oscillate wildly
        max_change = 0.1 * np.max(np.abs(u_next))
        change = u_current - u_next

        # Limit excessive changes
        excessive_change = np.abs(change) > max_change
        u_monotone[excessive_change] = (
            u_next[excessive_change] + np.sign(change[excessive_change]) * max_change
        )

        return u_monotone


# Factory function for high-order network HJB solvers
def create_high_order_network_hjb_solver(
    problem: "NetworkMFGProblem", scheme: str = "network_upwind", **kwargs
) -> HighOrderNetworkHJBSolver:
    """
    Create high-order network HJB solver with upwind and Godunov schemes.

    Args:
        problem: Network MFG problem
        scheme: Discretization scheme type
        **kwargs: Additional solver parameters

    Returns:
        Configured high-order network HJB solver
    """
    return HighOrderNetworkHJBSolver(problem, scheme=scheme, **kwargs)
