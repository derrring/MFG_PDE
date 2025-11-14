#!/usr/bin/env python3
"""
1D Adaptive Mesh Refinement for MFG Problems

This module implements 1D AMR using interval-based hierarchical refinement,
completing the geometry module architecture by providing consistent AMR
support across all dimensions (1D, 2D structured, 2D triangular).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.geometry.base_geometry import MeshData
from mfg_pde.geometry.protocol import GeometryType

from .amr_quadtree_2d import AMRRefinementCriteria, BaseErrorEstimator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import jax.numpy as jnp
    from jax import jit

    from mfg_pde.backends.base_backend import BaseBackend
    from mfg_pde.geometry.grids.grid_1d import SimpleGrid1D
    from mfg_pde.types.solver_types import JAXArray

# Always define JAX_AVAILABLE at module level
try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import numba  # noqa: F401
    from numba import jit as numba_jit  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class Interval1D:
    """1D interval for adaptive refinement."""

    # Basic properties
    interval_id: int
    x_min: float
    x_max: float
    level: int = 0

    # AMR hierarchy
    parent_id: int | None = None
    children_ids: list[int] | None = None

    # Solution and error data
    solution_data: dict[str, float] | None = None
    error_estimate: float = 0.0

    def __post_init__(self):
        """Initialize solution data dictionary."""
        if self.solution_data is None:
            self.solution_data = {}

    @property
    def center(self) -> float:
        """Interval center point."""
        return 0.5 * (self.x_min + self.x_max)

    @property
    def width(self) -> float:
        """Interval width."""
        return self.x_max - self.x_min

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf interval."""
        return self.children_ids is None or len(self.children_ids) == 0

    def contains_point(self, x: float) -> bool:
        """Check if point is inside interval."""
        return self.x_min <= x <= self.x_max

    def subdivide(self) -> tuple[Interval1D, Interval1D]:
        """Split interval into 2 children."""
        mid = self.center

        left_child = Interval1D(
            interval_id=-1,  # Will be set by parent
            x_min=self.x_min,
            x_max=mid,
            level=self.level + 1,
            parent_id=self.interval_id,
        )

        right_child = Interval1D(
            interval_id=-1,  # Will be set by parent
            x_min=mid,
            x_max=self.x_max,
            level=self.level + 1,
            parent_id=self.interval_id,
        )

        return left_child, right_child


class OneDimensionalAMRMesh:
    """
    1D Adaptive Mesh Refinement for MFG problems.

    This class provides interval-based hierarchical refinement for 1D domains,
    maintaining consistency with 2D AMR interfaces while handling 1D-specific
    concerns like boundary conditions and conservative interpolation.
    """

    def __init__(
        self,
        domain_1d: SimpleGrid1D,
        initial_num_intervals: int = 10,
        refinement_criteria: AMRRefinementCriteria | None = None,
        backend: BaseBackend | None = None,
    ):
        """
        Initialize 1D AMR mesh from SimpleGrid1D.

        Args:
            domain_1d: 1D domain specification
            initial_num_intervals: Initial number of intervals
            refinement_criteria: AMR refinement parameters
            backend: Computational backend
        """
        self.domain = domain_1d
        self.initial_num_intervals = initial_num_intervals
        self.criteria = refinement_criteria or AMRRefinementCriteria()
        self.backend = backend

        # Interval hierarchy
        self.intervals: dict[int, Interval1D] = {}
        self.leaf_intervals: list[int] = []
        self._next_interval_id = 0

        self._build_initial_intervals()

        # AMR statistics
        self.total_intervals = len(self.intervals)
        self.max_level = 0
        self.refinement_history: list[dict[str, Any]] = []

        # JAX backend integration
        self.use_jax = JAX_AVAILABLE and (backend is None or getattr(backend, "name", "") == "jax")

        if self.use_jax:
            self._setup_jax_functions()

    def _build_initial_intervals(self):
        """Build initial uniform interval mesh."""
        dx = self.domain.length / self.initial_num_intervals

        for i in range(self.initial_num_intervals):
            x_min = self.domain.xmin + i * dx
            x_max = self.domain.xmin + (i + 1) * dx

            interval = Interval1D(interval_id=i, x_min=x_min, x_max=x_max, level=0)

            self.intervals[i] = interval
            self.leaf_intervals.append(i)
            self._next_interval_id = max(self._next_interval_id, i + 1)

    def _setup_jax_functions(self):
        """Setup JAX-accelerated functions for 1D operations."""
        if not self.use_jax:
            return

        @jit
        def compute_1d_gradient_error(u_vals: JAXArray, x_coords: JAXArray) -> JAXArray:
            """JAX-accelerated gradient-based error estimation for 1D."""
            # Ensure arrays are properly typed for JAX operations
            u_array = jnp.asarray(u_vals)
            x_array = jnp.asarray(x_coords)

            # First derivative using central differences
            du_dx = jnp.gradient(u_array, x_array)

            # Second derivative for curvature term
            d2u_dx2 = jnp.gradient(du_dx, x_array)  # type: ignore[arg-type]

            # Error indicator: |du/dx| * h + |d²u/dx²| * h²
            h = jnp.diff(x_array).mean()
            error_vals = jnp.abs(du_dx) * h + jnp.abs(d2u_dx2) * h**2  # type: ignore[arg-type]

            return error_vals

        @jit
        def conservative_interpolation_1d(
            parent_vals: JAXArray,
            child_coords: JAXArray,
            parent_coords: JAXArray,
        ) -> JAXArray:
            """JAX-accelerated conservative interpolation for 1D refinement."""
            return jnp.interp(child_coords, parent_coords, parent_vals)

        self._jax_gradient_error = compute_1d_gradient_error
        self._jax_conservative_interp = conservative_interpolation_1d

    # ==================== GeometryProtocol Implementation ====================
    # Added in v0.10.1 to enable AMR meshes to work with geometry-first API

    @property
    def dimension(self) -> int:
        """Spatial dimension (1D)."""
        return 1

    @property
    def geometry_type(self) -> GeometryType:
        """Geometry type (CARTESIAN_GRID for structured adaptive grids)."""
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        """Number of active (leaf) intervals in current AMR state."""
        return len(self.leaf_intervals)

    def get_spatial_grid(self) -> np.ndarray:
        """
        Get spatial grid as (N, 1) array of leaf interval centers.

        Returns current AMR state - updates dynamically as mesh is refined.

        Returns:
            NDArray with shape (num_leaf_intervals, 1)
        """
        centers = [self.intervals[i].center for i in self.leaf_intervals]
        return np.array(centers).reshape(-1, 1)

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        This polymorphic method provides AMR-specific configuration for MFGProblem.
        AMR meshes have dynamic grids, so configuration is based on current state.

        Returns:
            Dictionary with keys:
                - num_spatial_points: Number of leaf intervals
                - spatial_shape: (num_leaf_intervals,) - flattened for AMR
                - spatial_bounds: None - AMR has dynamic bounds
                - spatial_discretization: None - AMR has variable spacing
                - legacy_1d_attrs: None - AMR doesn't support legacy attrs

        Added in v0.10.1 for polymorphic geometry handling.
        """
        return {
            "num_spatial_points": len(self.leaf_intervals),
            "spatial_shape": (len(self.leaf_intervals),),
            "spatial_bounds": None,  # AMR has dynamic bounds
            "spatial_discretization": None,  # AMR has variable spacing
            "legacy_1d_attrs": None,  # AMR doesn't support legacy 1D attributes
        }

    # ==================== AMR Operations ====================

    def refine_interval(self, interval_id: int) -> list[int]:
        """
        Refine a 1D interval into 2 children.

        Args:
            interval_id: ID of interval to refine

        Returns:
            List of new interval IDs
        """
        if interval_id not in self.intervals:
            raise ValueError(f"Interval {interval_id} not found")

        interval = self.intervals[interval_id]

        if not interval.is_leaf:
            raise ValueError(f"Interval {interval_id} is not a leaf")

        # Create children
        left_child, right_child = interval.subdivide()

        # Assign IDs
        left_id = self._next_interval_id
        right_id = self._next_interval_id + 1
        self._next_interval_id += 2

        left_child.interval_id = left_id
        right_child.interval_id = right_id

        # Store children
        self.intervals[left_id] = left_child
        self.intervals[right_id] = right_child

        # Update hierarchy
        interval.children_ids = [left_id, right_id]
        self.leaf_intervals.remove(interval_id)
        self.leaf_intervals.extend([left_id, right_id])

        # Update statistics
        self.total_intervals += 1  # Net increase of 1 (2 children - 1 parent)
        self.max_level = max(self.max_level, interval.level + 1)

        return [left_id, right_id]

    def adapt_mesh_1d(
        self, solution_data: dict[str, np.ndarray], error_estimator: BaseErrorEstimator
    ) -> dict[str, int]:
        """
        Adapt 1D mesh based on error estimates.

        Args:
            solution_data: Solution fields (U, M) on interval centers
            error_estimator: Error estimation algorithm

        Returns:
            Adaptation statistics
        """
        refinements = 0
        coarsenings = 0

        # Collect leaf intervals that need refinement
        intervals_to_refine = []

        for interval_id in self.leaf_intervals.copy():  # Copy since we'll modify
            interval = self.intervals[interval_id]

            # Check refinement constraints
            if interval.level >= self.criteria.max_refinement_levels:
                continue
            if interval.width < self.criteria.min_cell_size:
                continue

            # Estimate error for this interval
            error = self._estimate_interval_error(interval, solution_data, error_estimator)
            interval.error_estimate = error

            # Refinement decision
            if error > self.criteria.error_threshold:
                intervals_to_refine.append(interval_id)

        # Execute refinements
        for interval_id in intervals_to_refine:
            self.refine_interval(interval_id)
            refinements += 1

        # Record adaptation statistics
        stats = {
            "total_refined": refinements,
            "total_coarsened": coarsenings,
            "final_intervals": len(self.leaf_intervals),
            "total_intervals": self.total_intervals,
            "max_level": self.max_level,
        }

        self.refinement_history.append(stats)
        return stats

    def _estimate_interval_error(
        self,
        interval: Interval1D,
        solution_data: dict[str, np.ndarray],
        error_estimator: BaseErrorEstimator,
    ) -> float:
        """
        Estimate error for a single interval.

        Integrates with existing error estimation framework using pseudo-node.
        """

        # Create pseudo-node for compatibility with existing error estimator
        class PseudoNode:
            def __init__(self, interval: Interval1D):
                self.dx = interval.width
                self.dy = interval.width  # For compatibility
                self.area = interval.width
                self.center_x = interval.center
                self.center_y = 0.0  # For compatibility

        pseudo_node = PseudoNode(interval)

        # Use existing error estimator - cast to expected type
        return error_estimator.estimate_error(pseudo_node, solution_data)  # type: ignore[arg-type]

    def get_mesh_statistics(self) -> dict[str, Any]:
        """Get comprehensive 1D mesh statistics."""
        # Level distribution
        level_counts: dict[int, int] = {}
        total_length = 0.0
        min_width = np.inf
        max_width = 0.0

        for interval_id in self.leaf_intervals:
            interval = self.intervals[interval_id]
            level = interval.level
            level_counts[level] = level_counts.get(level, 0) + 1
            total_length += interval.width

            if interval.width < min_width:
                min_width = interval.width
            if interval.width > max_width:
                max_width = interval.width

        return {
            "total_intervals": self.total_intervals,
            "leaf_intervals": len(self.leaf_intervals),
            "max_level": self.max_level,
            "level_distribution": level_counts,
            "total_length": total_length,
            "domain_length": self.domain.length,
            "min_interval_width": min_width,
            "max_interval_width": max_width,
            "refinement_ratio": max_width / min_width if min_width > 0 else 1.0,
            "initial_intervals": self.initial_num_intervals,
            "refinement_history": self.refinement_history,
        }

    def get_grid_points(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get 1D grid points and spacing for current adaptive mesh.

        Returns:
            Tuple of (grid_points, interval_widths)
        """
        # Sort leaf intervals by x_min
        sorted_intervals = sorted([self.intervals[iid] for iid in self.leaf_intervals], key=lambda i: i.x_min)

        # Extract grid points and spacings
        grid_points = []
        interval_widths = []

        for interval in sorted_intervals:
            grid_points.append(interval.center)
            interval_widths.append(interval.width)

        return np.array(grid_points), np.array(interval_widths)

    def export_to_mesh_data(self) -> MeshData:
        """
        Export 1D AMR mesh to MeshData format for integration.

        Returns:
            MeshData representation of 1D adaptive mesh
        """
        # Get sorted leaf intervals
        sorted_intervals = sorted([self.intervals[iid] for iid in self.leaf_intervals], key=lambda i: i.x_min)

        # Build vertices (interval endpoints and centers)
        vertices = []
        elements = []

        for i, interval in enumerate(sorted_intervals):
            # For 1D, "vertices" are interval endpoints
            vertices.extend([[interval.x_min], [interval.x_max]])

            # "Elements" are the intervals themselves (vertex pairs)
            elements.append([2 * i, 2 * i + 1])

        # Remove duplicate vertices
        unique_vertices = []
        vertex_map = {}
        next_vertex_id = 0

        for vertex in vertices:
            vertex_key = vertex[0]
            if vertex_key not in vertex_map:
                vertex_map[vertex_key] = next_vertex_id
                unique_vertices.append(vertex)
                next_vertex_id += 1

        # Update elements to use unique vertex indices
        unique_elements = []
        for _i, interval in enumerate(sorted_intervals):
            left_idx = vertex_map[interval.x_min]
            right_idx = vertex_map[interval.x_max]
            unique_elements.append([left_idx, right_idx])

        return MeshData(
            vertices=np.array(unique_vertices),
            elements=np.array(unique_elements),
            element_type="line",  # 1D elements are line segments
            boundary_tags=np.array([0, 1]),  # Left and right boundaries
            element_tags=np.array(list(self.leaf_intervals)),
            boundary_faces=np.array([[0], [len(unique_vertices) - 1]]),  # Endpoints
            dimension=1,
            metadata={
                "amr_adapted": True,
                "max_level": self.max_level,
                "total_refinements": len(self.refinement_history),
                "original_intervals": self.initial_num_intervals,
                "domain_bounds": [self.domain.xmin, self.domain.xmax],
                "boundary_conditions": str(self.domain.boundary_conditions),
            },
        )


class OneDimensionalErrorEstimator(BaseErrorEstimator):
    """Error estimator specifically designed for 1D intervals."""

    def __init__(self, backend: BaseBackend | None = None):
        self.backend = backend
        self.use_jax = JAX_AVAILABLE and (backend is None or getattr(backend, "name", "") == "jax")

    def estimate_error(self, node, solution_data: dict[str, np.ndarray]) -> float:
        """
        Estimate error for 1D interval using gradient-based indicators.

        Uses finite difference gradients and curvature estimates.
        """
        if "U" not in solution_data or "M" not in solution_data:
            return 0.0

        U = solution_data["U"]
        M = solution_data["M"]

        # Get interval properties
        dx = getattr(node, "dx", 1.0)
        getattr(node, "center_x", 0.0)

        # For 1D, estimate local gradients using neighboring information
        # In practice, this would use solution values from neighboring intervals

        if U.ndim == 1 and U.size > 2:
            # Find closest index to center
            # This is simplified - would need proper interpolation
            i = min(max(int(len(U) / 2), 1), len(U) - 2)

            # Central difference gradient
            du_dx = (U[i + 1] - U[i - 1]) / (2.0 * dx)
            dm_dx = (M[i + 1] - M[i - 1]) / (2.0 * dx)

            # Second derivative (curvature)
            d2u_dx2 = (U[i + 1] - 2 * U[i] + U[i - 1]) / (dx**2)
            d2m_dx2 = (M[i + 1] - 2 * M[i] + M[i - 1]) / (dx**2)

            # Combined error indicator
            gradient_term = dx * (abs(du_dx) + abs(dm_dx))
            curvature_term = dx**2 * (abs(d2u_dx2) + abs(d2m_dx2))

            return gradient_term + curvature_term

        else:
            # Fallback for insufficient data
            return float(dx * (np.mean(np.abs(U)) + np.mean(np.abs(M))))


# Factory function for 1D AMR
def create_1d_amr_mesh(
    domain_1d: SimpleGrid1D,
    initial_intervals: int = 10,
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    backend: BaseBackend | None = None,
) -> OneDimensionalAMRMesh:
    """
    Create 1D AMR mesh from SimpleGrid1D.

    Args:
        domain_1d: 1D domain specification
        initial_intervals: Initial number of intervals
        error_threshold: Error threshold for refinement
        max_levels: Maximum refinement levels
        backend: Computational backend

    Returns:
        OneDimensionalAMRMesh ready for adaptive refinement
    """
    criteria = AMRRefinementCriteria(
        error_threshold=error_threshold,
        max_refinement_levels=max_levels,
        min_cell_size=domain_1d.length / (initial_intervals * 2**max_levels),
    )

    return OneDimensionalAMRMesh(
        domain_1d=domain_1d,
        initial_num_intervals=initial_intervals,
        refinement_criteria=criteria,
        backend=backend,
    )
