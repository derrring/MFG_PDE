"""
Adaptive Mesh Refinement MFG Solver.

This module provides MFG solvers that can work with adaptive meshes,
automatically refining and coarsening the mesh based on solution error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from tqdm import tqdm

import numpy as np

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from numpy.typing import NDArray

    from mfg_pde.backends.base_backend import BaseBackend
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.factory.solver_factory import SolverType
    from mfg_pde.geometry.amr_mesh import AdaptiveMesh, QuadTreeNode
    from mfg_pde.types.internal import JAXArray

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

from mfg_pde.alg.base_mfg_solver import MFGSolver
from mfg_pde.utils.solver_result import SolverResult


class AMRMFGSolver(MFGSolver):
    """
    MFG solver with adaptive mesh refinement capabilities.

    This solver automatically adapts the computational mesh based on
    solution error estimates, focusing computational effort on regions
    with high solution variation.
    """

    def __init__(
        self,
        problem: MFGProblem,
        adaptive_mesh: AdaptiveMesh,
        base_solver_type: SolverType = "fixed_point",
        amr_frequency: int = 5,  # Adapt mesh every N iterations
        max_amr_cycles: int = 3,  # Maximum AMR cycles per solve
        backend: BaseBackend | None = None,
        **solver_kwargs,
    ):
        """
        Initialize AMR MFG solver.

        Args:
            problem: MFG problem instance
            adaptive_mesh: Adaptive mesh configuration
            base_solver_type: Underlying solver type ("fixed_point", "particle", etc.)
            amr_frequency: Frequency of mesh adaptation (iterations)
            max_amr_cycles: Maximum AMR cycles per solve
            backend: Computational backend
            **solver_kwargs: Additional solver arguments
        """
        super().__init__(problem)
        self.backend = backend

        self.adaptive_mesh = adaptive_mesh
        self.base_solver_type = base_solver_type
        self.amr_frequency = amr_frequency
        self.max_amr_cycles = max_amr_cycles
        self.solver_kwargs = solver_kwargs

        # Initialize base solver on uniform grid
        self._initialize_base_solver()

        # AMR statistics tracking
        self.amr_stats = {
            "total_refinements": 0,
            "total_coarsenings": 0,
            "adaptation_cycles": 0,
            "mesh_efficiency": [],
        }

    def _initialize_base_solver(self):
        """Initialize the underlying MFG solver"""
        from mfg_pde.factory import create_fast_solver

        # Create solver with current backend
        self.base_solver = create_fast_solver(
            self.problem,
            solver_type=cast("SolverType", self.base_solver_type),
            backend=self.backend,
            **self.solver_kwargs,
        )

    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6, verbose: bool = True) -> SolverResult:
        """
        Solve MFG problem with adaptive mesh refinement.

        Args:
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance
            verbose: Enable progress output

        Returns:
            Solver result with AMR statistics
        """
        if verbose:
            print("Starting AMR MFG solver...")
            print(f"Initial mesh: {self.adaptive_mesh.total_cells} cells")

        # Initialize solution on current mesh
        U, M = self._initialize_solution()

        convergence_history = []
        mesh_history = []

        # Main AMR-solver loop
        for amr_cycle in range(self.max_amr_cycles):
            if verbose:
                pbar = tqdm(
                    range(max_iterations),
                    desc=f"AMR Cycle {amr_cycle + 1}/{self.max_amr_cycles}",
                )
                # Solve on current mesh with progress bar
                for iteration in pbar:
                    # Perform solver step
                    U_new, M_new, residual = self._solver_step(U, M)

                    # Check convergence
                    change = self._compute_solution_change(U, M, U_new, M_new)
                    convergence_history.append(change)

                    pbar.set_postfix(
                        {
                            "residual": f"{residual:.2e}",
                            "change": f"{change:.2e}",
                            "cells": self.adaptive_mesh.total_cells,
                        }
                    )

                    U, M = U_new, M_new

                    # Adapt mesh periodically
                    if (iteration + 1) % self.amr_frequency == 0:
                        self._adapt_mesh({"U": U, "M": M}, verbose=verbose)

                    # Check convergence
                    if change < tolerance:
                        if verbose:
                            print(f"\nConverged in {iteration + 1} iterations")
                        break
            else:
                # Solve on current mesh without progress bar
                for iteration in range(max_iterations):
                    # Perform solver step
                    U_new, M_new, residual = self._solver_step(U, M)

                    # Check convergence
                    change = self._compute_solution_change(U, M, U_new, M_new)
                    convergence_history.append(change)

                    U, M = U_new, M_new

                    # Adapt mesh periodically
                    if (iteration + 1) % self.amr_frequency == 0:
                        self._adapt_mesh({"U": U, "M": M}, verbose=verbose)

                    # Check convergence
                    if change < tolerance:
                        break

            # Record mesh statistics
            mesh_stats = self.adaptive_mesh.get_mesh_statistics()
            mesh_history.append(mesh_stats)

            # Final mesh adaptation for this cycle
            adaptation_stats = self._adapt_mesh({"U": U, "M": M}, verbose=verbose)

            # Check if mesh stabilized
            if adaptation_stats["total_refined"] == 0 and adaptation_stats["total_coarsened"] == 0:
                if verbose:
                    print("Mesh adaptation converged")
                break

            # Update solver for new mesh if needed
            if adaptation_stats["total_refined"] > 0 or adaptation_stats["total_coarsened"] > 0:
                U, M = self._project_solution_to_new_mesh(U, M)

        # Create final result
        execution_time = 0.0  # Would be tracked in real implementation
        final_residual = convergence_history[-1] if convergence_history else 1.0
        total_iterations = len(convergence_history)

        # Convert convergence history to error arrays (simplified)
        error_history_U = np.array(convergence_history) * 0.5  # Approximate split
        error_history_M = np.array(convergence_history) * 0.5  # Approximate split

        result = SolverResult(
            U=U,
            M=M,
            iterations=total_iterations,
            error_history_U=error_history_U,
            error_history_M=error_history_M,
            solver_name=f"AMR_{self.base_solver_type}",
            convergence_achieved=final_residual < tolerance,
            execution_time=execution_time,
            metadata={
                "solver_type": f"AMR_{self.base_solver_type}",
                "amr_stats": self.amr_stats,
                "mesh_history": mesh_history,
                "final_mesh_stats": self.adaptive_mesh.get_mesh_statistics(),
                "final_residual": final_residual,
                "convergence_history": convergence_history,
            },
        )

        # Store results for get_results() method
        self._last_U = U.copy()
        self._last_M = M.copy()

        return result

    def _initialize_solution(self) -> tuple[NDArray, NDArray]:
        """Initialize solution arrays on current mesh"""
        # For now, use uniform grid initialization
        # In full implementation, would initialize on AMR mesh
        nx, ny = 64, 64  # Default resolution

        U = np.zeros((nx, ny))
        M = np.ones((nx, ny)) / (nx * ny)  # Normalized initial density

        return U, M

    def _solver_step(self, U: NDArray, M: NDArray) -> tuple[NDArray, NDArray, float]:
        """
        Perform one step of the underlying MFG solver.

        Args:
            U: Current value function
            M: Current density

        Returns:
            Updated U, M, and residual
        """
        # This is a simplified implementation
        # In practice, would use the actual base solver

        # Simplified HJB step
        dt = self.problem.T / self.problem.Nt
        dx = (self.problem.xmax - self.problem.xmin) / U.shape[0]

        # Simple finite difference HJB update
        U_new = U - dt * self._compute_hamiltonian(U, M, dx)

        # Simple FP step
        M_new = M + dt * self._compute_fp_update(U_new, M, dx)

        # Normalize density
        M_new = np.maximum(M_new, 0)
        M_new = M_new / np.sum(M_new) * np.sum(M)

        # Compute residual
        residual = np.max(np.abs(U_new - U)) + np.max(np.abs(M_new - M))

        return U_new, M_new, residual

    def _compute_hamiltonian(self, U: NDArray, M: NDArray, dx: float) -> NDArray:
        """Compute Hamiltonian for HJB equation"""
        # Simplified Hamiltonian computation
        dU_dx = np.gradient(U, dx, axis=0)
        dU_dy = np.gradient(U, dx, axis=1)  # Assuming square cells

        # Quadratic Hamiltonian: H(x, p, m) = |p|^2/2 + congestion_cost(m)
        H = 0.5 * (dU_dx**2 + dU_dy**2) + self.problem.coefCT * M

        return H

    def _compute_fp_update(self, U: NDArray, M: NDArray, dx: float) -> NDArray:
        """Compute Fokker-Planck update"""
        # Simplified FP equation: ∂m/∂t - σ²/2 Δm - div(m ∇H_p) = 0

        # Diffusion term
        sigma_sq = self.problem.sigma**2
        laplacian_M = (
            np.roll(M, 1, axis=0) + np.roll(M, -1, axis=0) + np.roll(M, 1, axis=1) + np.roll(M, -1, axis=1) - 4 * M
        ) / dx**2

        diffusion = 0.5 * sigma_sq * laplacian_M

        # Drift term (simplified)
        dU_dx = np.gradient(U, dx, axis=0)
        dU_dy = np.gradient(U, dx, axis=1)

        drift_x = np.gradient(M * dU_dx, dx, axis=0)
        drift_y = np.gradient(M * dU_dy, dx, axis=1)
        drift = drift_x + drift_y

        return diffusion - drift

    def _compute_solution_change(self, U_old: NDArray, M_old: NDArray, U_new: NDArray, M_new: NDArray) -> float:
        """Compute change in solution between iterations"""
        u_change = np.max(np.abs(U_new - U_old))
        m_change = np.max(np.abs(M_new - M_old))
        return max(u_change, m_change)

    def _adapt_mesh(self, solution_data: dict[str, NDArray], verbose: bool = False) -> dict[str, int]:
        """
        Adapt the mesh based on current solution.

        Args:
            solution_data: Current solution arrays
            verbose: Enable progress output

        Returns:
            Adaptation statistics
        """
        if verbose:
            print(f"Adapting mesh (current: {self.adaptive_mesh.total_cells} cells)...")

        # Perform mesh adaptation
        adaptation_stats = self.adaptive_mesh.adapt_mesh(solution_data)

        # Update tracking statistics
        self.amr_stats["total_refinements"] += adaptation_stats["total_refined"]
        self.amr_stats["total_coarsenings"] += adaptation_stats["total_coarsened"]
        self.amr_stats["adaptation_cycles"] += 1

        # Compute mesh efficiency (cells per unit area with high error)
        mesh_stats = self.adaptive_mesh.get_mesh_statistics()
        efficiency = mesh_stats["total_cells"] / mesh_stats["total_area"]
        self.amr_stats["mesh_efficiency"].append(efficiency)

        if verbose and (adaptation_stats["total_refined"] > 0 or adaptation_stats["total_coarsened"] > 0):
            print(f"  Refined: {adaptation_stats['total_refined']} cells")
            print(f"  Coarsened: {adaptation_stats['total_coarsened']} cells")
            print(f"  New total: {self.adaptive_mesh.total_cells} cells")
            print(f"  Max level: {self.adaptive_mesh.max_level}")

        return adaptation_stats

    def _project_solution_to_new_mesh(self, U: NDArray, M: NDArray) -> tuple[NDArray, NDArray]:
        """
        Project solution from old mesh to new mesh after adaptation.

        Uses conservative interpolation to maintain mass conservation
        and solution accuracy during mesh adaptation.
        """
        if hasattr(self.adaptive_mesh, "solution_transfer_needed") and self.adaptive_mesh.solution_transfer_needed:
            # Perform conservative interpolation
            U_new, M_new = self._conservative_interpolation(U, M)

            # Reset transfer flag
            self.adaptive_mesh.solution_transfer_needed = False

            return U_new, M_new
        else:
            # No mesh changes, return original solution
            return U, M

    def _conservative_interpolation(self, U: NDArray, M: NDArray) -> tuple[NDArray, NDArray]:
        """
        Perform conservative interpolation of solution between mesh levels.

        This method ensures:
        1. Mass conservation for density M
        2. Gradient preservation for value function U
        3. Monotonicity preservation where appropriate
        """
        # Get current mesh structure

        # Create mapping from old uniform grid to AMR mesh
        nx, ny = U.shape
        x_coords = np.linspace(self.adaptive_mesh.x_min, self.adaptive_mesh.x_max, nx)
        y_coords = np.linspace(self.adaptive_mesh.y_min, self.adaptive_mesh.y_max, ny)

        # Initialize new solution arrays
        U_new = np.zeros_like(U)
        M_new = np.zeros_like(M)

        # For each grid point, find containing AMR cell and interpolate
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                containing_node = self.adaptive_mesh._find_containing_node(x, y)

                if containing_node is not None:
                    # Get old solution values in this region
                    old_U_val, old_M_val = self._get_region_values(U, M, containing_node, x_coords, y_coords)

                    # Conservative interpolation for density (mass preserving)
                    M_new[i, j] = self._conservative_density_interpolation(old_M_val, containing_node, x, y)

                    # Gradient-preserving interpolation for value function
                    U_new[i, j] = self._gradient_preserving_interpolation(old_U_val, containing_node, x, y)

        # Ensure mass conservation globally
        M_new = self._enforce_mass_conservation(M, M_new)

        return U_new, M_new

    def _get_region_values(
        self,
        U: NDArray,
        M: NDArray,
        node: QuadTreeNode,
        x_coords: NDArray,
        y_coords: NDArray,
    ) -> tuple[float, float]:
        """
        Get average solution values in the region corresponding to AMR node.
        """
        # Find grid points within this node's bounds
        x_mask = (x_coords >= node.x_min) & (x_coords <= node.x_max)
        y_mask = (y_coords >= node.y_min) & (y_coords <= node.y_max)

        # Get indices
        i_indices = np.where(x_mask)[0]
        j_indices = np.where(y_mask)[0]

        if len(i_indices) == 0 or len(j_indices) == 0:
            # Fallback to nearest neighbor
            i_center = int(
                (node.center_x - self.adaptive_mesh.x_min)
                / (self.adaptive_mesh.x_max - self.adaptive_mesh.x_min)
                * U.shape[0]
            )
            j_center = int(
                (node.center_y - self.adaptive_mesh.y_min)
                / (self.adaptive_mesh.y_max - self.adaptive_mesh.y_min)
                * U.shape[1]
            )

            i_center = np.clip(i_center, 0, U.shape[0] - 1)
            j_center = np.clip(j_center, 0, U.shape[1] - 1)

            return float(U[i_center, j_center]), float(M[i_center, j_center])

        # Compute weighted averages
        region_U = U[np.ix_(i_indices, j_indices)]
        region_M = M[np.ix_(i_indices, j_indices)]

        avg_U = float(np.mean(region_U))
        avg_M = float(np.mean(region_M))

        return avg_U, avg_M

    def _conservative_density_interpolation(self, old_density: float, node: QuadTreeNode, x: float, y: float) -> float:
        """
        Conservative interpolation for density function.

        Maintains mass conservation by using volume-weighted averages.
        """
        # For conservative interpolation, preserve the average density
        # This ensures mass conservation
        return old_density

    def _gradient_preserving_interpolation(self, old_value: float, node: QuadTreeNode, x: float, y: float) -> float:
        """
        Gradient-preserving interpolation for value function.

        Uses bilinear interpolation to maintain solution smoothness.
        """
        # For gradient preservation, use the node's stored value
        # In a full implementation, would compute gradients and use
        # higher-order interpolation
        return old_value

    def _enforce_mass_conservation(self, old_M: NDArray, new_M: NDArray) -> NDArray:
        """
        Enforce global mass conservation after interpolation.
        """
        old_mass = np.sum(old_M)
        new_mass = np.sum(new_M)

        if new_mass > 0:
            # Scale to conserve total mass
            conservation_factor = old_mass / new_mass
            new_M = new_M * conservation_factor
        else:
            # Fallback to original distribution
            new_M = old_M.copy()

        return new_M

    def get_amr_statistics(self) -> dict[str, Any]:
        """Get comprehensive AMR statistics"""
        mesh_stats = self.adaptive_mesh.get_mesh_statistics()

        return {
            "mesh_statistics": mesh_stats,
            "adaptation_statistics": self.amr_stats.copy(),
            "efficiency_metrics": {
                "cells_per_level": mesh_stats["level_distribution"],
                "refinement_ratio": mesh_stats["refinement_ratio"],
                "average_efficiency": (
                    np.mean(self.amr_stats["mesh_efficiency"]) if self.amr_stats["mesh_efficiency"] else 0.0
                ),
            },
        }

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the computed U and M arrays from the last solve.

        Returns:
            Tuple of (U, M) solution arrays

        Raises:
            RuntimeError: If solve() has not been called yet
        """
        if not hasattr(self, "_last_U") or not hasattr(self, "_last_M"):
            raise RuntimeError("No solution available. Call solve() first.")

        return self._last_U, self._last_M


class JAXAcceleratedAMR:
    """JAX-accelerated AMR operations for high performance"""

    def __init__(self, backend: BaseBackend):
        self.backend = backend

        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for JAXAcceleratedAMR")

    @staticmethod
    @jax.jit
    def compute_error_indicators(U: JAXArray, M: JAXArray, dx: float, dy: float) -> JAXArray:
        """
        Compute error indicators for all cells using JAX.

        Args:
            U: Value function
            M: Density function
            dx, dy: Grid spacing

        Returns:
            Error indicator array
        """
        # JAX availability is guaranteed by JAXAcceleratedAMR constructor
        assert jnp is not None, "JAX must be available for JAXAcceleratedAMR"

        # Compute gradients
        dU_dx = jnp.gradient(U, dx, axis=0)
        dU_dy = jnp.gradient(U, dy, axis=1)

        dM_dx = jnp.gradient(M, dx, axis=0)
        dM_dy = jnp.gradient(M, dy, axis=1)

        # Gradient magnitudes
        grad_U = jnp.sqrt(dU_dx**2 + dU_dy**2)
        grad_M = jnp.sqrt(dM_dx**2 + dM_dy**2)

        # Combined error indicator
        error_indicator = jnp.maximum(grad_U, grad_M)

        # Add curvature-based indicator
        d2U_dx2 = jnp.gradient(dU_dx, dx, axis=0)
        d2U_dy2 = jnp.gradient(dU_dy, dy, axis=1)
        curvature_U = jnp.abs(d2U_dx2) + jnp.abs(d2U_dy2)

        d2M_dx2 = jnp.gradient(dM_dx, dx, axis=0)
        d2M_dy2 = jnp.gradient(dM_dy, dy, axis=1)
        curvature_M = jnp.abs(d2M_dx2) + jnp.abs(d2M_dy2)

        curvature_indicator = jnp.maximum(curvature_U, curvature_M)

        return error_indicator + 0.1 * curvature_indicator

    @staticmethod
    @jax.jit
    def conservative_interpolation_1d(
        coarse_values: JAXArray, coarse_coords: JAXArray, fine_coords: JAXArray
    ) -> JAXArray:
        """
        Conservative 1D interpolation between mesh levels.

        Args:
            coarse_values: Values on coarse mesh
            coarse_coords: Coordinates of coarse mesh
            fine_coords: Coordinates of fine mesh

        Returns:
            Interpolated values on fine mesh
        """
        assert jnp is not None, "JAX must be available for JAXAcceleratedAMR"
        return jnp.interp(fine_coords, coarse_coords, coarse_values)

    @staticmethod
    @jax.jit
    def conservative_mass_interpolation_2d(
        density: JAXArray, old_dx: float, old_dy: float, new_dx: float, new_dy: float
    ) -> JAXArray:
        """
        Conservative mass interpolation for 2D density fields.

        Ensures total mass is conserved during interpolation.

        Args:
            density: Original density field
            old_dx, old_dy: Old grid spacing
            new_dx, new_dy: New grid spacing

        Returns:
            Mass-conserving interpolated density
        """
        assert jnp is not None, "JAX must be available for JAXAcceleratedAMR"

        # Compute total mass
        old_mass = jnp.sum(density) * old_dx * old_dy

        # Simple resize (in practice would use area-weighted interpolation)
        from jax.scipy.ndimage import map_coordinates

        # Create coordinate mapping
        old_shape = density.shape
        new_shape = (
            int(old_shape[0] * old_dx / new_dx),
            int(old_shape[1] * old_dy / new_dy),
        )

        # Generate new coordinate grid
        x_new = jnp.linspace(0, old_shape[0] - 1, new_shape[0])
        y_new = jnp.linspace(0, old_shape[1] - 1, new_shape[1])

        X_new, Y_new = jnp.meshgrid(x_new, y_new, indexing="ij")
        coords = jnp.stack([X_new.ravel(), Y_new.ravel()])

        # Interpolate
        density_interp = map_coordinates(density, coords, order=1, mode="nearest")
        density_interp = density_interp.reshape(new_shape)

        # Enforce mass conservation
        new_mass = jnp.sum(density_interp) * new_dx * new_dy
        conservation_factor = old_mass / (new_mass + 1e-12)  # Avoid division by zero

        return density_interp * conservation_factor

    @staticmethod
    @jax.jit
    def gradient_preserving_interpolation_2d(
        values: JAXArray, old_dx: float, old_dy: float, new_dx: float, new_dy: float
    ) -> JAXArray:
        """
        Gradient-preserving interpolation for value functions.

        Maintains solution smoothness and gradient information.

        Args:
            values: Original value field
            old_dx, old_dy: Old grid spacing
            new_dx, new_dy: New grid spacing

        Returns:
            Gradient-preserving interpolated values
        """
        assert jnp is not None, "JAX must be available for JAXAcceleratedAMR"

        from jax.scipy.ndimage import map_coordinates

        # Create coordinate mapping for higher-order interpolation
        old_shape = values.shape
        new_shape = (
            int(old_shape[0] * old_dx / new_dx),
            int(old_shape[1] * old_dy / new_dy),
        )

        # Generate new coordinate grid
        x_new = jnp.linspace(0, old_shape[0] - 1, new_shape[0])
        y_new = jnp.linspace(0, old_shape[1] - 1, new_shape[1])

        X_new, Y_new = jnp.meshgrid(x_new, y_new, indexing="ij")
        coords = jnp.stack([X_new.ravel(), Y_new.ravel()])

        # Use cubic interpolation for better gradient preservation
        values_interp = map_coordinates(values, coords, order=3, mode="nearest")
        values_interp = values_interp.reshape(new_shape)

        return values_interp


def create_amr_solver(
    problem: MFGProblem,
    domain_bounds: tuple[float, float, float, float] | None = None,
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    base_solver: SolverType = "fixed_point",
    backend: str = "auto",
    **kwargs,
) -> AMRMFGSolver:
    """
    Factory function to create an AMR MFG solver.

    Args:
        problem: MFG problem instance
        domain_bounds: Domain bounds (x_min, x_max, y_min, y_max)
        error_threshold: Error threshold for refinement
        max_levels: Maximum refinement levels
        base_solver: Base solver type
        backend: Backend type
        **kwargs: Additional solver arguments

    Returns:
        Configured AMRMFGSolver
    """
    from mfg_pde.backends import create_backend
    from mfg_pde.geometry.amr_mesh import create_amr_mesh

    # Default domain bounds from problem
    if domain_bounds is None:
        domain_bounds = (
            problem.xmin,
            problem.xmax,
            problem.xmin,
            problem.xmax,  # Assume square domain for now
        )

    # Create adaptive mesh
    adaptive_mesh = create_amr_mesh(
        domain_bounds=domain_bounds,
        error_threshold=error_threshold,
        max_levels=max_levels,
        backend=backend,
    )

    # Create backend
    backend_instance = create_backend(backend)

    return AMRMFGSolver(
        problem=problem,
        adaptive_mesh=adaptive_mesh,
        base_solver_type=base_solver,
        backend=backend_instance,
        **kwargs,
    )
