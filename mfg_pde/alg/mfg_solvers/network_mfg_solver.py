"""
Network MFG Solvers.

This module implements complete Mean Field Games solvers for network structures,
combining network HJB and FP solvers to solve the coupled MFG system on graphs.

Key solver types:
- NetworkFixedPointIterator: Picard iteration for network MFG
- NetworkFlowMFGSolver: Flow-based network MFG solver
- NetworkPolicyMFGSolver: Policy iteration-based network MFG solver
"""

import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np

from ..base_mfg_solver import MFGSolver
from ..fp_solvers.fp_network import NetworkFPSolver, create_network_fp_solver
from ..hjb_solvers.hjb_network import NetworkHJBSolver, create_network_hjb_solver

if TYPE_CHECKING:
    from ...core.network_mfg_problem import NetworkMFGProblem


class NetworkFixedPointIterator(MFGSolver):
    """
    Fixed point iterator for Mean Field Games on networks.

    Solves the coupled network MFG system:

    HJB: ∂u/∂t + H_i(m, ∇_G u, t) = 0
    FP:  ∂m/∂t - div_G(m ∇_G H_p) - σ²/2 Δ_G m = 0

    using Picard iteration on the network structure.
    """

    def __init__(
        self,
        problem: "NetworkMFGProblem",
        hjb_solver_type: str = "explicit",
        fp_solver_type: str = "explicit",
        damping_factor: float = 0.5,
        hjb_kwargs: Optional[Dict[str, Any]] = None,
        fp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize network fixed point iterator.

        Args:
            problem: Network MFG problem instance
            hjb_solver_type: Type of HJB solver ("explicit", "implicit", "policy_iteration")
            fp_solver_type: Type of FP solver ("explicit", "implicit", "upwind", "flow")
            damping_factor: Picard iteration damping parameter
            hjb_kwargs: Additional arguments for HJB solver
            fp_kwargs: Additional arguments for FP solver
        """
        super().__init__(problem)

        self.network_problem = problem
        self.damping_factor = damping_factor

        # Initialize HJB solver
        hjb_kwargs = hjb_kwargs or {}
        self.hjb_solver = create_network_hjb_solver(problem, hjb_solver_type, **hjb_kwargs)

        # Initialize FP solver
        fp_kwargs = fp_kwargs or {}
        self.fp_solver = create_network_fp_solver(problem, fp_solver_type, **fp_kwargs)

        # Solver properties
        self.num_nodes = problem.num_nodes
        self.Nt = problem.Nt
        self.T = problem.T

        # Solution storage
        self.U: Optional[np.ndarray] = None
        self.M: Optional[np.ndarray] = None

        # Convergence tracking
        self.convergence_history = []
        self.iterations_run = 0

        # Solver identification
        self.name = f"NetworkMFG_HJB-{hjb_solver_type}_FP-{fp_solver_type}"

    def solve(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solve network MFG system using fixed point iteration.

        Args:
            max_iterations: Maximum Picard iterations
            tolerance: Convergence tolerance
            verbose: Print convergence information
            **kwargs: Additional parameters

        Returns:
            (U, M, convergence_info) tuple
            - U: (Nt+1, num_nodes) value function evolution
            - M: (Nt+1, num_nodes) density evolution
            - convergence_info: Dictionary with convergence data
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"NETWORK MFG SOLVER: {self.name}")
            print(f"{'='*80}")
            print(f"Network: {self.network_problem.network_data.network_type.value}")
            print(f"Nodes: {self.num_nodes}, Edges: {self.network_problem.num_edges}")
            print(f"Time: T={self.T}, Nt={self.Nt}")
            print(f"Max iterations: {max_iterations}, Tolerance: {tolerance:.2e}")
            print(f"Damping factor: {self.damping_factor}")
            print()

        solve_start_time = time.time()

        # Initialize solutions
        self.U = np.zeros((self.Nt + 1, self.num_nodes))
        self.M = np.zeros((self.Nt + 1, self.num_nodes))

        # Set initial and terminal conditions
        self.M[0, :] = self.network_problem.get_initial_density()
        self.U[self.Nt, :] = self.network_problem.get_terminal_value()

        # Initialize interior values
        for t in range(self.Nt):
            self.U[t, :] = self.U[self.Nt, :]  # Constant initial guess
            if t > 0:
                self.M[t, :] = self.M[0, :]  # Constant initial guess

        # Fixed point iteration
        convergence_achieved = False
        self.convergence_history = []

        for iteration in range(max_iterations):
            iter_start_time = time.time()

            if verbose:
                print(f"Iteration {iteration + 1}/{max_iterations}")

            # Store previous iteration
            U_old = self.U.copy()
            M_old = self.M.copy()

            # Solve HJB equation
            U_new = self.hjb_solver.solve_hjb_system(
                M_density_evolution=self.M,
                U_final_condition_at_T=self.network_problem.get_terminal_value(),
                U_from_prev_picard=U_old,
            )

            # Apply damping to U
            self.U = self.damping_factor * U_new + (1 - self.damping_factor) * U_old

            # Solve FP equation
            M_new = self.fp_solver.solve_fp_system(
                m_initial_condition=self.network_problem.get_initial_density(),
                U_solution_for_drift=self.U,
            )

            # Apply damping to M
            self.M = self.damping_factor * M_new + (1 - self.damping_factor) * M_old

            # Compute convergence metrics
            u_error = np.linalg.norm(self.U - U_old) / max(np.linalg.norm(self.U), 1e-12)
            m_error = np.linalg.norm(self.M - M_old) / max(np.linalg.norm(self.M), 1e-12)
            total_error = max(u_error, m_error)

            iter_time = time.time() - iter_start_time

            # Store convergence info
            conv_info = {
                "iteration": iteration + 1,
                "u_error": u_error,
                "m_error": m_error,
                "total_error": total_error,
                "iteration_time": iter_time,
                "mass_conservation": np.sum(self.M, axis=1),  # Mass at each time step
            }
            self.convergence_history.append(conv_info)

            if verbose:
                print(f"  U error: {u_error:.2e}, M error: {m_error:.2e}")
                print(f"  Total error: {total_error:.2e}, Time: {iter_time:.2f}s")

                # Check mass conservation
                mass_variation = np.std(conv_info["mass_conservation"])
                print(f"  Mass conservation (std): {mass_variation:.2e}")

            # Check convergence
            if total_error < tolerance:
                convergence_achieved = True
                if verbose:
                    print(f"\nSUCCESS: Convergence achieved after {iteration + 1} iterations!")
                break

            self.iterations_run = iteration + 1

            if verbose:
                print()

        if not convergence_achieved and verbose:
            print(f"\nWARNING:  Maximum iterations ({max_iterations}) reached without convergence")
            print(f"Final error: {total_error:.2e}")

        execution_time = time.time() - solve_start_time

        if verbose:
            print(f"\n🏁 Total execution time: {execution_time:.2f}s")
            print(f"Average time per iteration: {execution_time/self.iterations_run:.2f}s")

        # Prepare convergence info
        convergence_info = {
            "converged": convergence_achieved,
            "iterations": self.iterations_run,
            "final_error": total_error if "total_error" in locals() else float("inf"),
            "execution_time": execution_time,
            "tolerance": tolerance,
            "convergence_history": self.convergence_history,
            "solver_name": self.name,
            "network_stats": self.network_problem.get_network_statistics(),
        }

        # Mark solution as computed
        self._solution_computed = True

        return self.U, self.M, convergence_info

    def get_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get computed U and M solutions."""
        if self.U is None or self.M is None:
            raise ValueError("No solution available. Call solve() first.")
        return self.U, self.M

    def get_convergence_history(self) -> list:
        """Get convergence history."""
        return self.convergence_history

    def get_network_flows(self, time_index: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute network flows at specified time (or all times).

        Args:
            time_index: Time index (None for all times)

        Returns:
            Dictionary with flow information
        """
        if self.U is None or self.M is None:
            raise ValueError("No solution available. Call solve() first.")

        if time_index is not None:
            # Single time step
            u = self.U[time_index, :]
            m = self.M[time_index, :]

            flows = self._compute_flows_at_time(u, m)
            return {"time_index": time_index, "flows": flows}
        else:
            # All time steps
            all_flows = []
            for t in range(self.Nt + 1):
                u = self.U[t, :]
                m = self.M[t, :]
                flows = self._compute_flows_at_time(u, m)
                all_flows.append(flows)

            return {"all_times": all_flows}

    def _compute_flows_at_time(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Compute network flows at a single time step."""
        num_edges = self.network_problem.num_edges
        flows = np.zeros(num_edges)

        # This would need proper edge indexing implementation
        # Simplified version for now
        edge_idx = 0
        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)
            for j in neighbors:
                if i < j:  # Avoid double counting
                    # Flow from i to j
                    du = u[j] - u[i]
                    flow_ij = m[i] * du  # Simplified flow computation
                    flows[edge_idx] = flow_ij
                    edge_idx += 1
                    if edge_idx >= num_edges:
                        break
            if edge_idx >= num_edges:
                break

        return flows


class NetworkFlowMFGSolver(NetworkFixedPointIterator):
    """
    Flow-based MFG solver for networks with enhanced flow tracking.

    This solver uses flow-preserving schemes and provides detailed
    flow analysis capabilities.
    """

    def __init__(self, problem: "NetworkMFGProblem", **kwargs):
        """Initialize flow-based network MFG solver."""
        # Force flow-based FP solver
        fp_kwargs = kwargs.pop("fp_kwargs", {})
        super().__init__(problem, fp_solver_type="flow", fp_kwargs=fp_kwargs, **kwargs)

        self.name = f"NetworkFlowMFG_{self.hjb_solver.hjb_method_name}_{self.fp_solver.fp_method_name}"

    def analyze_network_flows(self) -> Dict[str, Any]:
        """Comprehensive flow analysis for the network solution."""
        if self.U is None or self.M is None:
            raise ValueError("No solution available. Call solve() first.")

        # Compute various flow metrics
        flow_analysis = {
            "total_flow_over_time": [],
            "max_flow_over_time": [],
            "flow_concentration": [],
            "flow_patterns": {},
        }

        for t in range(self.Nt + 1):
            flows = self._compute_flows_at_time(self.U[t, :], self.M[t, :])

            flow_analysis["total_flow_over_time"].append(np.sum(np.abs(flows)))
            flow_analysis["max_flow_over_time"].append(np.max(np.abs(flows)))

            # Flow concentration (Gini-like coefficient)
            sorted_flows = np.sort(np.abs(flows))
            n = len(sorted_flows)
            gini = (2 * np.sum((np.arange(n) + 1) * sorted_flows)) / (n * np.sum(sorted_flows)) - (n + 1) / n
            flow_analysis["flow_concentration"].append(gini)

        # Identify dominant flow patterns
        avg_flows = np.mean(
            [self._compute_flows_at_time(self.U[t, :], self.M[t, :]) for t in range(self.Nt + 1)],
            axis=0,
        )

        flow_analysis["flow_patterns"] = {
            "dominant_edges": np.argsort(np.abs(avg_flows))[-5:],  # Top 5 edges by flow
            "average_flows": avg_flows,
            "flow_statistics": {
                "mean": np.mean(avg_flows),
                "std": np.std(avg_flows),
                "max": np.max(avg_flows),
                "min": np.min(avg_flows),
            },
        }

        return flow_analysis


# Factory function for network MFG solvers
def create_network_mfg_solver(
    problem: "NetworkMFGProblem", solver_type: str = "fixed_point", **kwargs
) -> NetworkFixedPointIterator:
    """
    Create network MFG solver with specified type.

    Args:
        problem: Network MFG problem
        solver_type: Type of solver ("fixed_point", "flow")
        **kwargs: Additional solver parameters

    Returns:
        Configured network MFG solver
    """
    if solver_type == "flow":
        return NetworkFlowMFGSolver(problem, **kwargs)
    elif solver_type == "fixed_point":
        return NetworkFixedPointIterator(problem, **kwargs)
    else:
        raise ValueError(f"Unknown network MFG solver type: {solver_type}")
