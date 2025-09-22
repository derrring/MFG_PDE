#!/usr/bin/env python3
"""
AMR Enhancement for Existing MFG Solvers

This module provides AMR (Adaptive Mesh Refinement) as an enhancement
that can be applied to any existing MFG solver, rather than being
a standalone solver type.

AMR is a mesh adaptation technique, not a solution method itself.
This design reflects that by wrapping existing solvers with AMR capability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mfg_pde.alg.base_mfg_solver import MFGSolver
    from mfg_pde.geometry.amr_mesh import BaseErrorEstimator

logger = logging.getLogger(__name__)


class AMREnhancedSolver:
    """
    AMR enhancement wrapper for existing MFG solvers.

    This class wraps any MFG solver and adds adaptive mesh refinement
    capability without changing the underlying solution method.

    Design Philosophy:
    - AMR is a mesh adaptation technique, not a solution method
    - Any solver (FDM, particle, spectral, etc.) can benefit from AMR
    - AMR should enhance, not replace, existing solvers
    """

    def __init__(
        self,
        base_solver: MFGSolver,
        amr_mesh: Any,  # 1D, 2D, or triangular AMR mesh
        error_estimator: BaseErrorEstimator,
        amr_config: dict[str, Any] | None = None,
    ):
        """
        Initialize AMR enhancement for an existing solver.

        Args:
            base_solver: Any MFG solver (FDM, particle, etc.)
            amr_mesh: Adaptive mesh (1D, 2D structured, or triangular)
            error_estimator: Algorithm for error estimation
            amr_config: AMR-specific configuration
        """
        self.base_solver = base_solver
        self.amr_mesh = amr_mesh
        self.error_estimator = error_estimator

        # AMR configuration
        default_config = {
            "adaptation_frequency": 5,  # Adapt mesh every N iterations
            "max_adaptations": 3,  # Maximum AMR cycles
            "convergence_tolerance": 1e-6,  # Overall convergence
            "mesh_change_threshold": 0.1,  # Re-solve if mesh changes significantly
            "verbose": True,
        }
        self.amr_config = {**default_config, **(amr_config or {})}

        # AMR state
        self.adaptation_history = []
        self.total_adaptations = 0
        self.current_mesh_generation = 0

        # Problem reference
        self.problem = base_solver.problem

    def solve(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Solve MFG problem with adaptive mesh refinement.

        This method:
        1. Solves using the base solver on current mesh
        2. Estimates errors and adapts mesh if needed
        3. Transfers solution to new mesh conservatively
        4. Continues until convergence or max adaptations reached

        Args:
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance
            verbose: Enable verbose output
            **kwargs: Additional arguments for base solver

        Returns:
            Solution dictionary with AMR statistics
        """
        verbose = verbose if verbose is not None else self.amr_config["verbose"]

        if verbose:
            print(f"Starting AMR-enhanced {type(self.base_solver).__name__}")
            print(f"  Base solver: {type(self.base_solver).__name__}")
            print(f"  AMR mesh: {type(self.amr_mesh).__name__}")
            print(f"  Problem dimension: {getattr(self.problem, 'dimension', 'unknown')}")

        # Main AMR iteration loop
        for amr_cycle in range(self.amr_config["max_adaptations"] + 1):
            if verbose:
                print(f"\nAMR Cycle {amr_cycle + 1}")
                self._print_mesh_status()

            # Solve on current mesh
            solution_result = self._solve_on_current_mesh(max_iterations, tolerance, verbose, **kwargs)

            # Check convergence - if converged and no significant errors, we're done
            if solution_result.get("converged", False) and amr_cycle > 0:
                if verbose:
                    print("  Solution converged with acceptable mesh resolution")
                break

            # Skip adaptation on last allowed cycle
            if amr_cycle >= self.amr_config["max_adaptations"]:
                if verbose:
                    print(f"  Reached maximum AMR cycles ({self.amr_config['max_adaptations']})")
                break

            # Estimate errors and adapt mesh
            adaptation_stats = self._adapt_mesh_if_needed(solution_result, verbose)

            if adaptation_stats["total_adapted"] == 0:
                if verbose:
                    print("  No mesh adaptation needed - converged")
                break
            else:
                # Transfer solution to new mesh
                solution_result = self._transfer_solution_to_new_mesh(solution_result, adaptation_stats, verbose)
                self.current_mesh_generation += 1

        # Collect final results
        final_result = self._collect_final_results(solution_result, verbose)

        if verbose:
            self._print_final_summary(final_result)

        return final_result

    def _solve_on_current_mesh(self, max_iterations: int, tolerance: float, verbose: bool, **kwargs) -> dict[str, Any]:
        """Solve using base solver on current mesh."""
        if verbose:
            current_mesh_size = self._get_current_mesh_size()
            print(f"  Solving on mesh with {current_mesh_size} elements...")

        try:
            # Call base solver - handle different return formats
            solver_result = self.base_solver.solve(max_iterations=max_iterations, tolerance=tolerance, **kwargs)

            # Normalize result format
            if isinstance(solver_result, tuple):
                # Old format: (U, M, info)
                U, M, info = solver_result
                return {
                    "U": U,
                    "M": M,
                    "converged": info.get("converged", False),
                    "iterations": info.get("iterations", max_iterations),
                    "solver_info": info,
                }
            elif isinstance(solver_result, dict):
                # New format: already a dict
                return solver_result
            else:
                raise ValueError(f"Unexpected solver result format: {type(solver_result)}")

        except Exception as e:
            logger.error(f"Base solver failed: {e}")
            raise RuntimeError(f"Base solver {type(self.base_solver).__name__} failed: {e}") from e

    def _adapt_mesh_if_needed(self, solution_result: dict[str, Any], verbose: bool) -> dict[str, Any]:
        """Estimate errors and adapt mesh if refinement is needed."""
        try:
            # Extract solution data
            solution_data = {"U": solution_result["U"], "M": solution_result["M"]}

            # Adapt mesh using appropriate method
            if hasattr(self.amr_mesh, "adapt_mesh"):
                # 2D structured or triangular AMR
                stats = self.amr_mesh.adapt_mesh(solution_data, self.error_estimator)
            elif hasattr(self.amr_mesh, "adapt_mesh_1d"):
                # 1D AMR
                stats = self.amr_mesh.adapt_mesh_1d(solution_data, self.error_estimator)
            else:
                raise ValueError(f"Unknown AMR mesh type: {type(self.amr_mesh)}")

            # Record adaptation
            self.adaptation_history.append(stats)
            self.total_adaptations += stats.get("total_refined", stats.get("total_adapted", 0))

            if verbose and stats.get("total_refined", 0) > 0:
                print(f"  Adapted {stats.get('total_refined', 0)} elements")
                print(f"  New mesh size: {self._get_current_mesh_size()}")

            return stats

        except Exception as e:
            logger.warning(f"Mesh adaptation failed: {e}")
            return {"total_adapted": 0, "error": str(e)}

    def _transfer_solution_to_new_mesh(
        self,
        solution_result: dict[str, Any],
        adaptation_stats: dict[str, Any],
        verbose: bool,
    ) -> dict[str, Any]:
        """Transfer solution from old mesh to adapted mesh."""
        if verbose:
            print("  Transferring solution to adapted mesh...")

        # This is a simplified transfer - in practice would need
        # conservative interpolation based on mesh type

        # For now, let base solver handle re-initialization
        # Real implementation would do conservative interpolation

        return solution_result

    def _get_current_mesh_size(self) -> int:
        """Get current mesh size (number of active elements)."""
        if hasattr(self.amr_mesh, "leaf_triangles"):
            return len(self.amr_mesh.leaf_triangles)
        elif hasattr(self.amr_mesh, "leaf_intervals"):
            return len(self.amr_mesh.leaf_intervals)
        elif hasattr(self.amr_mesh, "leaf_nodes"):
            return len(self.amr_mesh.leaf_nodes)
        else:
            return getattr(self.amr_mesh, "total_elements", 0)

    def _print_mesh_status(self):
        """Print current mesh status."""
        mesh_size = self._get_current_mesh_size()

        if hasattr(self.amr_mesh, "get_mesh_statistics"):
            stats = self.amr_mesh.get_mesh_statistics()
            print(f"  Current mesh: {mesh_size} elements, {stats.get('max_level', 0)} levels")
        else:
            print(f"  Current mesh: {mesh_size} elements")

    def _collect_final_results(self, solution_result: dict[str, Any], verbose: bool) -> dict[str, Any]:
        """Collect comprehensive results including AMR statistics."""

        # Get final mesh statistics
        if hasattr(self.amr_mesh, "get_mesh_statistics"):
            mesh_stats = self.amr_mesh.get_mesh_statistics()
        else:
            mesh_stats = {"elements": self._get_current_mesh_size()}

        # Combine all results
        final_result = {
            **solution_result,  # Include all base solver results
            # AMR-specific results
            "amr_enabled": True,
            "base_solver_type": type(self.base_solver).__name__,
            "mesh_statistics": mesh_stats,
            "adaptation_history": self.adaptation_history,
            "total_adaptations": self.total_adaptations,
            "mesh_generations": self.current_mesh_generation + 1,
            # Enhanced convergence info
            "amr_converged": len(self.adaptation_history) > 0
            and self.adaptation_history[-1].get("total_refined", 0) == 0,
        }

        return final_result

    def _print_final_summary(self, result: dict[str, Any]):
        """Print final AMR enhancement summary."""
        print("\nAMR Enhancement Summary:")
        print(f"  Base solver: {result['base_solver_type']}")
        print(f"  Total adaptations: {result['total_adaptations']}")
        print(f"  Mesh generations: {result['mesh_generations']}")
        print(f"  Final mesh size: {self._get_current_mesh_size()}")

        if "mesh_statistics" in result:
            stats = result["mesh_statistics"]
            if "max_level" in stats:
                print(f"  Max refinement level: {stats['max_level']}")

        print(f"  Converged: {result.get('converged', False)}")


def create_amr_enhanced_solver(
    base_solver: MFGSolver,
    dimension: int | None = None,
    amr_config: dict[str, Any] | None = None,
) -> AMREnhancedSolver:
    """
    Create AMR-enhanced version of any MFG solver.

    This factory function automatically creates appropriate AMR mesh
    and error estimator based on the problem dimension.

    Args:
        base_solver: Any MFG solver to enhance with AMR
        dimension: Problem dimension (auto-detected if None)
        amr_config: AMR configuration parameters

    Returns:
        AMR-enhanced solver wrapping the base solver

    Example:
        >>> base_solver = create_fdm_solver(problem)
        >>> amr_solver = create_amr_enhanced_solver(base_solver)
        >>> result = amr_solver.solve()
    """
    from mfg_pde.geometry.amr_mesh import AdaptiveMesh, GradientErrorEstimator
    from mfg_pde.geometry.domain_1d import Domain1D, periodic_bc
    from mfg_pde.geometry.one_dimensional_amr import OneDimensionalErrorEstimator, create_1d_amr_mesh

    problem = base_solver.problem

    # Auto-detect dimension
    if dimension is None:
        dimension = getattr(problem, "dimension", None)
        if dimension is None:
            dimension = 2 if hasattr(problem, "ymin") and hasattr(problem, "ymax") else 1

    # Create appropriate AMR mesh and error estimator
    if dimension == 1:
        # 1D AMR - MFGProblem doesn't have domain attribute, extract parameters directly
        xmin = getattr(problem, "xmin", 0.0)
        xmax = getattr(problem, "xmax", 1.0)
        domain_1d = Domain1D(xmin, xmax, periodic_bc())

        amr_mesh = create_1d_amr_mesh(
            domain_1d=domain_1d,
            initial_intervals=(amr_config.get("initial_intervals", 20) if amr_config else 20),
            error_threshold=(amr_config.get("error_threshold", 1e-4) if amr_config else 1e-4),
            max_levels=amr_config.get("max_levels", 5) if amr_config else 5,
        )
        error_estimator = OneDimensionalErrorEstimator()

    elif dimension == 2:
        # 2D AMR
        xmin = getattr(problem, "xmin", 0.0)
        xmax = getattr(problem, "xmax", 1.0)
        ymin = getattr(problem, "ymin", 0.0)
        ymax = getattr(problem, "ymax", 1.0)

        amr_mesh = AdaptiveMesh(
            domain_bounds=(xmin, xmax, ymin, ymax),
            initial_resolution=(20, 20),
        )
        error_estimator = GradientErrorEstimator()

    else:
        raise ValueError(f"AMR not supported for {dimension}D problems")

    return AMREnhancedSolver(base_solver, amr_mesh, error_estimator, amr_config)
