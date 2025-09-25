"""
Adaptive Mesh Refinement (AMR) implementation for MFG_PDE.

This module provides quadtree-based adaptive mesh refinement capabilities
for Mean Field Games problems, with support for error-based refinement
and JAX acceleration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

# Apply TYPE_CHECKING isolation principle for JAX (same as OmegaConf pattern)
if TYPE_CHECKING:
    # Static typing world - simple definitions
    import numpy as np

    jnp = np  # Type alias for static analysis

    def jit(fun, **kwargs): ...
    def vmap(fun, **kwargs): ...

    JAX_AVAILABLE = True
else:
    # Runtime world - actual imports with fallbacks
    try:
        import jax.numpy as jnp
        from jax import jit, vmap

        JAX_AVAILABLE = True
    except ImportError:
        # Runtime fallback to numpy
        jnp = np  # type: ignore[misc]
        JAX_AVAILABLE = False

        # Fallback implementations
        def jit(fun, **kwargs):
            return fun

        def vmap(fun, **kwargs):
            return fun


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.backends.base_backend import BaseBackend
    from mfg_pde.types.internal import JAXArray


@dataclass
class AMRRefinementCriteria:
    """
    Criteria for adaptive mesh refinement decisions.

    Attributes:
        error_threshold: Maximum allowed solution error per cell
        gradient_threshold: Maximum allowed gradient magnitude per cell
        max_refinement_levels: Maximum number of refinement levels
        min_cell_size: Minimum allowed cell size
        coarsening_threshold: Threshold for mesh coarsening (fraction of error_threshold)
    """

    error_threshold: float = 1e-4
    gradient_threshold: float = 0.1
    max_refinement_levels: int = 5
    min_cell_size: float = 1e-6
    coarsening_threshold: float = 0.1

    # Advanced criteria
    solution_variance_threshold: float = 1e-5
    density_gradient_threshold: float = 0.05
    adaptive_error_scaling: bool = True


@dataclass
class QuadTreeNode:
    """
    Quadtree node for 2D adaptive mesh refinement.

    Attributes:
        level: Refinement level (0 = root)
        x_min, x_max, y_min, y_max: Cell boundaries
        parent: Parent node (None for root)
        children: Child nodes (None for leaf nodes)
        is_leaf: Whether this is a leaf node
        cell_id: Unique identifier for this cell
        solution_data: Solution values at this cell
        error_estimate: Local error estimate
    """

    level: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    parent: QuadTreeNode | None = None
    children: list[QuadTreeNode] | None = None
    is_leaf: bool = True
    cell_id: int | None = None
    solution_data: dict[str, NDArray] | None = None
    error_estimate: float = 0.0

    def __post_init__(self):
        if self.solution_data is None:
            self.solution_data = {}

    @property
    def center_x(self) -> float:
        """X-coordinate of cell center"""
        return 0.5 * (self.x_min + self.x_max)

    @property
    def center_y(self) -> float:
        """Y-coordinate of cell center"""
        return 0.5 * (self.y_min + self.y_max)

    @property
    def dx(self) -> float:
        """Cell width"""
        return self.x_max - self.x_min

    @property
    def dy(self) -> float:
        """Cell height"""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Cell area"""
        return self.dx * self.dy

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside this cell"""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def subdivide(self) -> list[QuadTreeNode]:
        """
        Subdivide this cell into 4 children.

        Returns:
            List of 4 child nodes
        """
        if not self.is_leaf:
            raise ValueError("Cannot subdivide non-leaf node")

        mid_x = 0.5 * (self.x_min + self.x_max)
        mid_y = 0.5 * (self.y_min + self.y_max)

        # Create 4 children: SW, SE, NW, NE
        children = [
            QuadTreeNode(
                level=self.level + 1,
                x_min=self.x_min,
                x_max=mid_x,
                y_min=self.y_min,
                y_max=mid_y,
                parent=self,
            ),  # SW
            QuadTreeNode(
                level=self.level + 1,
                x_min=mid_x,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=mid_y,
                parent=self,
            ),  # SE
            QuadTreeNode(
                level=self.level + 1,
                x_min=self.x_min,
                x_max=mid_x,
                y_min=mid_y,
                y_max=self.y_max,
                parent=self,
            ),  # NW
            QuadTreeNode(
                level=self.level + 1,
                x_min=mid_x,
                x_max=self.x_max,
                y_min=mid_y,
                y_max=self.y_max,
                parent=self,
            ),  # NE
        ]

        self.children = children
        self.is_leaf = False

        return children


class BaseErrorEstimator(ABC):
    """Base class for error estimation algorithms"""

    @abstractmethod
    def estimate_error(self, node: QuadTreeNode, solution_data: dict[str, NDArray]) -> float:
        """
        Estimate the local error in a mesh cell.

        Args:
            node: Quadtree node representing the cell
            solution_data: Current solution data

        Returns:
            Error estimate for this cell
        """


class GradientErrorEstimator(BaseErrorEstimator):
    """Error estimator based on solution gradients"""

    def __init__(self, backend: BaseBackend | None = None):
        self.backend = backend

    def estimate_error(self, node: QuadTreeNode, solution_data: dict[str, NDArray]) -> float:
        """
        Estimate error based on solution gradients.

        Uses finite difference approximation of gradients to estimate
        local solution variation.
        """
        if "U" not in solution_data or "M" not in solution_data:
            return 0.0

        U = solution_data["U"]
        M = solution_data["M"]

        # Get cell indices (simplified for now)
        # In practice, we'd need proper grid mapping
        i, j = self._get_cell_indices(node)

        if i <= 0 or i >= U.shape[0] - 1 or j <= 0 or j >= U.shape[1] - 1:
            return 0.0

        # Compute gradients using finite differences
        dU_dx = (U[i + 1, j] - U[i - 1, j]) / (2.0 * node.dx)
        dU_dy = (U[i, j + 1] - U[i, j - 1]) / (2.0 * node.dy)

        dM_dx = (M[i + 1, j] - M[i - 1, j]) / (2.0 * node.dx)
        dM_dy = (M[i, j + 1] - M[i, j - 1]) / (2.0 * node.dy)

        # Combined gradient magnitude
        grad_U = np.sqrt(dU_dx**2 + dU_dy**2)
        grad_M = np.sqrt(dM_dx**2 + dM_dy**2)

        return max(grad_U, grad_M)

    def _get_cell_indices(self, node: QuadTreeNode) -> tuple[int, int]:
        """
        Map cell center to grid indices.

        This is a simplified implementation - in practice would need
        proper coordinate transformation.
        """
        # Placeholder implementation
        return 10, 10


class AdaptiveMesh:
    """
    Adaptive mesh refinement system for MFG problems.

    This class manages a quadtree-based adaptive mesh with automatic
    refinement and coarsening based on solution error estimates.
    """

    def __init__(
        self,
        domain_bounds: tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
        initial_resolution: tuple[int, int] = (32, 32),
        refinement_criteria: AMRRefinementCriteria | None = None,
        error_estimator: BaseErrorEstimator | None = None,
        backend: BaseBackend | None = None,
    ):
        """
        Initialize adaptive mesh.

        Args:
            domain_bounds: (x_min, x_max, y_min, y_max)
            initial_resolution: Initial grid resolution
            refinement_criteria: Criteria for mesh adaptation
            error_estimator: Error estimation algorithm
            backend: Computational backend (NumPy or JAX)
        """
        self.x_min, self.x_max, self.y_min, self.y_max = domain_bounds
        self.initial_resolution = initial_resolution
        self.criteria = refinement_criteria or AMRRefinementCriteria()
        self.error_estimator = error_estimator or GradientErrorEstimator(backend)
        self.backend = backend

        # Initialize root node
        self.root = QuadTreeNode(
            level=0,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
        )

        # Track all leaf nodes for efficient access
        self.leaf_nodes: list[QuadTreeNode] = [self.root]
        self.total_cells = 1
        self.max_level = 0

        # Cell ID counter
        self._next_cell_id = 0
        self.root.cell_id = self._get_next_cell_id()

        # Solution transfer tracking
        self.solution_transfer_needed = False

    def _get_next_cell_id(self) -> int:
        """Get next unique cell ID"""
        cell_id = self._next_cell_id
        self._next_cell_id += 1
        return cell_id

    def refine_mesh(self, solution_data: dict[str, NDArray]) -> int:
        """
        Perform one step of mesh refinement based on current solution.

        Args:
            solution_data: Current solution arrays (U, M, etc.)

        Returns:
            Number of cells that were refined
        """
        cells_to_refine = []

        # Identify cells that need refinement
        for node in self.leaf_nodes:
            if node.level >= self.criteria.max_refinement_levels:
                continue

            if min(node.dx, node.dy) <= self.criteria.min_cell_size:
                continue

            # Estimate error in this cell
            error = self.error_estimator.estimate_error(node, solution_data)
            node.error_estimate = error

            if error > self.criteria.error_threshold:
                cells_to_refine.append(node)

        # Refine identified cells
        refined_count = 0
        for node in cells_to_refine:
            children = node.subdivide()

            # Assign cell IDs to children
            for child in children:
                child.cell_id = self._get_next_cell_id()

            # Remove parent from leaf list, add children
            self.leaf_nodes.remove(node)
            self.leaf_nodes.extend(children)

            self.total_cells += 4
            self.max_level = max(self.max_level, node.level + 1)
            refined_count += 1

        return refined_count

    def coarsen_mesh(self, solution_data: dict[str, NDArray]) -> int:
        """
        Perform mesh coarsening where appropriate.

        Args:
            solution_data: Current solution arrays

        Returns:
            Number of cells that were coarsened
        """
        coarsened_count = 0
        coarsening_threshold = self.criteria.error_threshold * self.criteria.coarsening_threshold

        # Group leaf nodes by parent
        parent_groups: dict[Any, list[Any]] = {}
        for node in self.leaf_nodes:
            if node.parent is not None:
                parent = node.parent
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(node)

        # Check if all children of a parent can be coarsened
        for parent, children in parent_groups.items():
            if len(children) != 4:  # All 4 children must be leaves
                continue

            # Check if all children have low error
            all_low_error = True
            for child in children:
                error = self.error_estimator.estimate_error(child, solution_data)
                if error > coarsening_threshold:
                    all_low_error = False
                    break

            if all_low_error:
                # Coarsen: remove children, make parent a leaf
                for child in children:
                    self.leaf_nodes.remove(child)

                parent.children = None
                parent.is_leaf = True
                self.leaf_nodes.append(parent)

                self.total_cells -= 3  # Net decrease of 3 cells (4 children -> 1 parent)
                coarsened_count += 1

        return coarsened_count

    def adapt_mesh(self, solution_data: dict[str, NDArray], max_iterations: int = 5) -> dict[str, int]:
        """
        Perform complete mesh adaptation (refinement + coarsening).

        Args:
            solution_data: Current solution arrays
            max_iterations: Maximum adaptation iterations

        Returns:
            Dictionary with adaptation statistics
        """
        stats = {
            "total_refined": 0,
            "total_coarsened": 0,
            "iterations": 0,
            "final_cells": 0,
            "max_level": 0,
        }

        for _iteration in range(max_iterations):
            refined = self.refine_mesh(solution_data)
            coarsened = self.coarsen_mesh(solution_data)

            stats["total_refined"] += refined
            stats["total_coarsened"] += coarsened
            stats["iterations"] += 1

            # Mark solution transfer needed if mesh changed
            if refined > 0 or coarsened > 0:
                self.solution_transfer_needed = True

            # Stop if no changes were made
            if refined == 0 and coarsened == 0:
                break

        stats["final_cells"] = self.total_cells
        stats["max_level"] = self.max_level

        return stats

    def get_mesh_statistics(self) -> dict[str, Any]:
        """Get current mesh statistics"""
        level_counts: dict[int, int] = {}
        total_area = 0.0
        min_cell_size = float("inf")
        max_cell_size = 0.0

        for node in self.leaf_nodes:
            level = node.level
            level_counts[level] = level_counts.get(level, 0) + 1

            total_area += node.area
            cell_size = min(node.dx, node.dy)
            min_cell_size = min(min_cell_size, cell_size)
            max_cell_size = max(max_cell_size, cell_size)

        return {
            "total_cells": self.total_cells,
            "leaf_cells": len(self.leaf_nodes),
            "max_level": self.max_level,
            "level_distribution": level_counts,
            "total_area": total_area,
            "min_cell_size": min_cell_size,
            "max_cell_size": max_cell_size,
            "refinement_ratio": (max_cell_size / min_cell_size if min_cell_size > 0 else 1.0),
        }

    def interpolate_solution(
        self, coarse_solution: dict[str, NDArray], target_grid: tuple[int, int]
    ) -> dict[str, NDArray]:
        """
        Interpolate solution from AMR mesh to uniform grid.

        Args:
            coarse_solution: Solution on AMR mesh
            target_grid: Target uniform grid size (nx, ny)

        Returns:
            Interpolated solution on uniform grid
        """
        nx, ny = target_grid
        x_coords = np.linspace(self.x_min, self.x_max, nx)
        y_coords = np.linspace(self.y_min, self.y_max, ny)

        # Initialize output arrays
        interpolated = {}
        for key in coarse_solution:
            interpolated[key] = np.zeros((nx, ny))

        # For each point in target grid, find containing cell and interpolate
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                containing_node = self._find_containing_node(x, y)

                if containing_node is not None and containing_node.solution_data is not None:
                    # Simple piecewise constant interpolation for now
                    for key in coarse_solution:
                        if key in containing_node.solution_data:
                            interpolated[key][i, j] = containing_node.solution_data[key]

        return interpolated  # type: ignore[return-value]

    def _find_containing_node(self, x: float, y: float) -> QuadTreeNode | None:
        """Find the leaf node containing point (x, y)"""
        current = self.root

        while not current.is_leaf:
            # Find which child contains the point
            mid_x = 0.5 * (current.x_min + current.x_max)
            mid_y = 0.5 * (current.y_min + current.y_max)

            # Determine quadrant: SW=0, SE=1, NW=2, NE=3
            if x < mid_x and y < mid_y:
                child_idx = 0  # SW
            elif x >= mid_x and y < mid_y:
                child_idx = 1  # SE
            elif x < mid_x and y >= mid_y:
                child_idx = 2  # NW
            else:
                child_idx = 3  # NE

            if current.children is None or child_idx >= len(current.children):
                break

            current = current.children[child_idx]

        return current if current.contains_point(x, y) else None


# JAX-accelerated functions for AMR operations
if JAX_AVAILABLE:

    @jit
    def compute_gradient_error_jax(U: JAXArray, M: JAXArray, dx: float, dy: float) -> JAXArray:
        """
        Compute gradient-based error estimate using JAX.

        Args:
            U: Value function array
            M: Density function array
            dx, dy: Grid spacing

        Returns:
            Error estimate array
        """
        # Compute gradients using finite differences
        dU_dx = (jnp.roll(U, -1, axis=0) - jnp.roll(U, 1, axis=0)) / (2.0 * dx)
        dU_dy = (jnp.roll(U, -1, axis=1) - jnp.roll(U, 1, axis=1)) / (2.0 * dy)

        dM_dx = (jnp.roll(M, -1, axis=0) - jnp.roll(M, 1, axis=0)) / (2.0 * dx)
        dM_dy = (jnp.roll(M, -1, axis=1) - jnp.roll(M, 1, axis=1)) / (2.0 * dy)

        # Gradient magnitudes
        grad_U = jnp.sqrt(dU_dx**2 + dU_dy**2)
        grad_M = jnp.sqrt(dM_dx**2 + dM_dy**2)

        return jnp.maximum(grad_U, grad_M)


def create_amr_mesh(
    domain_bounds: tuple[float, float, float, float],
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    backend: str = "auto",
) -> AdaptiveMesh:
    """
    Factory function to create an adaptive mesh.

    Args:
        domain_bounds: (x_min, x_max, y_min, y_max)
        error_threshold: Error threshold for refinement
        max_levels: Maximum refinement levels
        backend: Backend type ("numpy", "jax", or "auto")

    Returns:
        Configured AdaptiveMesh instance
    """
    from mfg_pde.backends import create_backend

    # Create backend
    backend_instance = create_backend(backend)

    # Create refinement criteria
    criteria = AMRRefinementCriteria(error_threshold=error_threshold, max_refinement_levels=max_levels)

    # Create error estimator
    error_estimator = GradientErrorEstimator(backend_instance)

    return AdaptiveMesh(
        domain_bounds=domain_bounds,
        refinement_criteria=criteria,
        error_estimator=error_estimator,
        backend=backend_instance,
    )
