#!/usr/bin/env python3
"""
1D Adaptive Mesh Refinement for MFG Problems

This module implements 1D AMR using interval-based hierarchical refinement,
completing the geometry module architecture by providing consistent AMR
support across all dimensions (1D, 2D structured, 2D triangular).

Updated: Issue #466 - Renamed to OneDimensionalAMRGrid.

Note: Cannot inherit from CartesianGrid ABC due to circular import:
    amr_1d -> base -> meshes/__init__ -> mesh_1d -> base
Uses duck typing for protocol compliance instead.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.geometry.meshes.mesh_data import MeshData
from mfg_pde.geometry.protocol import GeometryType

from .amr_quadtree_2d import AMRRefinementCriteria, BaseErrorEstimator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp
    from jax import jit

    from mfg_pde.backends.base_backend import BaseBackend
    from mfg_pde.geometry.grids.tensor_grid import TensorProductGrid
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


class OneDimensionalAMRGrid:
    """
    1D Adaptive Mesh Refinement Grid for MFG problems.

    This class provides interval-based hierarchical refinement for 1D domains,
    maintaining consistency with 2D AMR interfaces while handling 1D-specific
    concerns like boundary conditions and conservative interpolation.

    Design Note:
        Cannot inherit from CartesianGrid ABC due to circular import in the
        geometry module. Implements all CartesianGrid methods via duck typing
        for protocol compliance.

    Protocol Compliance:
        - GeometryProtocol: Full implementation (duck typing)
        - AdaptiveGeometry: Full implementation of AMR capability
        - CartesianGrid-compatible: All methods implemented

    Examples:
        >>> from mfg_pde.geometry import TensorProductGrid, OneDimensionalAMRGrid
        >>> from mfg_pde.geometry.protocol import AdaptiveGeometry, is_adaptive
        >>>
        >>> domain = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx_points=[11])
        >>> amr = OneDimensionalAMRGrid(domain, initial_num_intervals=10)
        >>>
        >>> # Protocol checks
        >>> isinstance(amr, AdaptiveGeometry)  # True
        >>> is_adaptive(amr)  # True
    """

    def __init__(
        self,
        domain_1d: TensorProductGrid,
        initial_num_intervals: int = 10,
        refinement_criteria: AMRRefinementCriteria | None = None,
        backend: BaseBackend | None = None,
    ):
        """
        Initialize 1D AMR mesh from TensorProductGrid (1D).

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
        self._max_level = 0
        self.refinement_history: list[dict[str, Any]] = []

        # JAX backend integration
        self.use_jax = JAX_AVAILABLE and (backend is None or getattr(backend, "name", "") == "jax")

        if self.use_jax:
            self._setup_jax_functions()

    def _build_initial_intervals(self):
        """Build initial uniform interval mesh."""
        # Get domain bounds
        min_coords, max_coords = self.domain.get_bounds()
        domain_xmin = float(min_coords[0])
        domain_xmax = float(max_coords[0])
        domain_length = domain_xmax - domain_xmin

        dx = domain_length / self.initial_num_intervals

        for i in range(self.initial_num_intervals):
            x_min = domain_xmin + i * dx
            x_max = domain_xmin + (i + 1) * dx

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

    # =========================================================================
    # Geometry ABC Implementation (from CartesianGrid inheritance)
    # =========================================================================
    # Added in v0.10.1 to enable AMR meshes to work with geometry-first API
    # Updated in v0.16.5 (Issue #460) for CartesianGrid inheritance

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

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of the adaptive mesh.

        Delegates to underlying domain.

        Returns:
            (min_coords, max_coords) tuple of arrays
        """
        return self.domain.get_bounds()

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        This polymorphic method provides AMR-specific configuration for MFGProblem.
        AMR meshes have dynamic grids, so configuration is based on current state.

        Returns:
            Dictionary with keys:
                - num_spatial_points: Number of leaf intervals
                - spatial_shape: (num_leaf_intervals,) - flattened for AMR
                - spatial_bounds: Domain bounds from underlying grid
                - spatial_discretization: None - AMR has variable spacing
                - legacy_1d_attrs: None - AMR doesn't support legacy attrs

        Added in v0.10.1 for polymorphic geometry handling.
        """
        min_bounds, max_bounds = self.get_bounds()
        return {
            "num_spatial_points": len(self.leaf_intervals),
            "spatial_shape": (len(self.leaf_intervals),),
            "spatial_bounds": tuple(zip(min_bounds, max_bounds, strict=True)),
            "spatial_discretization": None,  # AMR has variable spacing
            "legacy_1d_attrs": None,  # AMR doesn't support legacy 1D attributes
        }

    # =========================================================================
    # CartesianGrid ABC Implementation (required methods)
    # =========================================================================

    def get_grid_spacing(self) -> list[float]:
        """
        Get grid spacing for adaptive mesh.

        For AMR, returns the finest level spacing (smallest interval width).

        Returns:
            [dx] where dx is the minimum interval width
        """
        if not self.leaf_intervals:
            min_coords, max_coords = self.domain.get_bounds()
            domain_length = float(max_coords[0]) - float(min_coords[0])
            return [domain_length / self.initial_num_intervals]

        min_width = min(self.intervals[i].width for i in self.leaf_intervals)
        return [min_width]

    def get_grid_shape(self) -> tuple[int, ...]:
        """
        Get grid shape for adaptive mesh.

        For AMR, returns the number of leaf intervals (not a true tensor shape).

        Returns:
            (num_leaf_intervals,)
        """
        return (len(self.leaf_intervals),)

    # =========================================================================
    # Solver Operation Interface (CartesianGrid requirements)
    # =========================================================================

    def get_laplacian_operator(self) -> Callable:
        """
        Return discretized Laplacian operator for adaptive 1D mesh.

        Uses finite differences on adaptive intervals with variable spacing.

        Returns:
            Function with signature: (u: NDArray, idx: int) -> float

        Notes:
            The operator accounts for non-uniform spacing in the adaptive mesh.
            For interior points, uses central differences.
            For boundary points, uses one-sided differences.
        """
        # Pre-sort intervals for efficient lookup
        sorted_ids = sorted(self.leaf_intervals, key=lambda i: self.intervals[i].center)

        def laplacian_adaptive(u: np.ndarray, idx: int) -> float:
            """
            Compute Laplacian at leaf interval index.

            Args:
                u: Solution values at leaf interval centers, shape (num_leaf,)
                idx: Index in the sorted leaf intervals (0 to num_leaf-1)

            Returns:
                Laplacian value at that interval
            """
            n = len(sorted_ids)
            if n < 3:
                return 0.0  # Need at least 3 points for Laplacian

            if idx <= 0 or idx >= n - 1:
                # Boundary: use one-sided difference (simplified)
                return 0.0

            # Get neighboring interval widths for non-uniform FD
            h_left = self.intervals[sorted_ids[idx]].center - self.intervals[sorted_ids[idx - 1]].center
            h_right = self.intervals[sorted_ids[idx + 1]].center - self.intervals[sorted_ids[idx]].center

            # Non-uniform central difference for second derivative
            # d²u/dx² ≈ 2/(h_left + h_right) * [u[i+1]/h_right - u[i]*(1/h_left + 1/h_right) + u[i-1]/h_left]
            coeff = 2.0 / (h_left + h_right)
            laplacian = coeff * (u[idx + 1] / h_right - u[idx] * (1.0 / h_left + 1.0 / h_right) + u[idx - 1] / h_left)

            return float(laplacian)

        return laplacian_adaptive

    def get_gradient_operator(self) -> Callable:
        """
        Return discretized gradient operator for adaptive 1D mesh.

        Uses central differences with variable spacing.

        Returns:
            Function with signature: (u: NDArray, idx: int) -> NDArray
        """
        sorted_ids = sorted(self.leaf_intervals, key=lambda i: self.intervals[i].center)

        def gradient_adaptive(u: np.ndarray, idx: int) -> np.ndarray:
            """
            Compute gradient at leaf interval index.

            Args:
                u: Solution values at leaf interval centers, shape (num_leaf,)
                idx: Index in the sorted leaf intervals (0 to num_leaf-1)

            Returns:
                Gradient value as 1D array [du/dx]
            """
            n = len(sorted_ids)
            if n < 2:
                return np.array([0.0])

            if idx <= 0:
                # Left boundary: forward difference
                h = self.intervals[sorted_ids[1]].center - self.intervals[sorted_ids[0]].center
                return np.array([(u[1] - u[0]) / h])
            elif idx >= n - 1:
                # Right boundary: backward difference
                h = self.intervals[sorted_ids[-1]].center - self.intervals[sorted_ids[-2]].center
                return np.array([(u[-1] - u[-2]) / h])
            else:
                # Interior: central difference
                h_total = self.intervals[sorted_ids[idx + 1]].center - self.intervals[sorted_ids[idx - 1]].center
                return np.array([(u[idx + 1] - u[idx - 1]) / h_total])

        return gradient_adaptive

    def get_interpolator(self) -> Callable:
        """
        Return interpolation function for adaptive 1D mesh.

        Uses linear interpolation between neighboring interval centers.

        Returns:
            Function with signature: (u: NDArray, point: NDArray) -> float
        """
        sorted_ids = sorted(self.leaf_intervals, key=lambda i: self.intervals[i].center)

        def interpolate_adaptive(u: np.ndarray, point: np.ndarray) -> float:
            """
            Interpolate solution at arbitrary point.

            Args:
                u: Solution values at leaf interval centers, shape (num_leaf,)
                point: Physical coordinates, shape (1,) or scalar

            Returns:
                Interpolated value
            """
            x = float(point[0]) if hasattr(point, "__len__") else float(point)
            centers = np.array([self.intervals[i].center for i in sorted_ids])

            # Find bracketing intervals
            if x <= centers[0]:
                return float(u[0])
            if x >= centers[-1]:
                return float(u[-1])

            # Binary search for bracketing interval
            idx = np.searchsorted(centers, x) - 1
            idx = max(0, min(idx, len(centers) - 2))

            # Linear interpolation
            x0, x1 = centers[idx], centers[idx + 1]
            t = (x - x0) / (x1 - x0)
            return float((1 - t) * u[idx] + t * u[idx + 1])

        return interpolate_adaptive

    def get_boundary_handler(self, bc_type: str = "dirichlet") -> Any:
        """
        Return boundary condition handler for adaptive 1D mesh.

        Delegates to underlying domain's boundary handler.

        Args:
            bc_type: Type of boundary condition

        Returns:
            Boundary handler object
        """
        return self.domain.get_boundary_handler(bc_type)

    # =========================================================================
    # AdaptiveGeometry Protocol Implementation (Issue #459)
    # =========================================================================

    def refine(self, criteria: object) -> int:
        """
        Refine intervals meeting refinement criteria.

        This is the AdaptiveGeometry protocol method. For detailed control,
        use `refine_interval()` or `adapt_mesh_1d()`.

        Args:
            criteria: Refinement criteria - can be:
                - dict with 'interval_ids': list of interval IDs to refine
                - dict with 'error_threshold': refine intervals above threshold
                - BaseErrorEstimator with solution_data

        Returns:
            Number of intervals refined
        """
        if isinstance(criteria, dict):
            if "interval_ids" in criteria:
                # Explicit list of intervals to refine
                refined = 0
                for interval_id in criteria["interval_ids"]:
                    if interval_id in self.leaf_intervals:
                        self.refine_interval(interval_id)
                        refined += 1
                return refined
            elif "solution_data" in criteria and "error_estimator" in criteria:
                # Use error estimator
                stats = self.adapt_mesh_1d(criteria["solution_data"], criteria["error_estimator"])
                return stats["total_refined"]
        return 0

    def coarsen(self, criteria: object) -> int:
        """
        Coarsen intervals meeting coarsening criteria.

        Note: Coarsening is not yet implemented for 1D AMR.

        Args:
            criteria: Coarsening criteria (currently unused)

        Returns:
            Number of intervals coarsened (currently always 0)
        """
        # Coarsening not implemented for 1D AMR yet
        # Would require merging sibling intervals back to parent
        return 0

    def adapt(self, solution_data: dict[str, object]) -> dict[str, int]:
        """
        Perform full adaptation cycle (refine + coarsen).

        Args:
            solution_data: Dictionary with solution arrays for error estimation.
                          Typically contains 'U' (value function), 'M' (density),
                          and 'error_estimator' (BaseErrorEstimator instance).

        Returns:
            Dictionary with adaptation statistics:
            {
                'refined': int,   # Number of intervals refined
                'coarsened': int, # Number of intervals coarsened (0 for now)
                'total_cells': int,  # Final interval count
            }
        """
        if "error_estimator" in solution_data:
            error_estimator = solution_data["error_estimator"]
            # Filter solution_data to only arrays
            arrays_only = {k: v for k, v in solution_data.items() if isinstance(v, np.ndarray)}
            stats = self.adapt_mesh_1d(arrays_only, error_estimator)  # type: ignore[arg-type]
            return {
                "refined": stats["total_refined"],
                "coarsened": stats["total_coarsened"],
                "total_cells": len(self.leaf_intervals),
            }
        return {"refined": 0, "coarsened": 0, "total_cells": len(self.leaf_intervals)}

    @property
    def max_refinement_level(self) -> int:
        """
        Maximum refinement level in the current mesh.

        Returns:
            Maximum level (0 = coarsest, higher = finer)
        """
        return self._max_level

    @property
    def num_leaf_cells(self) -> int:
        """
        Number of active (leaf) intervals.

        Returns:
            Count of leaf intervals (cells at finest local resolution)
        """
        return len(self.leaf_intervals)

    # =========================================================================
    # AMR Operations (existing methods)
    # =========================================================================

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
        self._max_level = max(self._max_level, interval.level + 1)

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
            "max_level": self._max_level,
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
        # Get domain bounds
        min_coords, max_coords = self.domain.get_bounds()
        domain_length = float(max_coords[0]) - float(min_coords[0])

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
            "max_level": self._max_level,
            "level_distribution": level_counts,
            "total_length": total_length,
            "domain_length": domain_length,
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
        # Get domain bounds
        min_coords, max_coords = self.domain.get_bounds()

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
                "max_level": self._max_level,
                "total_refinements": len(self.refinement_history),
                "original_intervals": self.initial_num_intervals,
                "domain_bounds": [float(min_coords[0]), float(max_coords[0])],
                "boundary_conditions": "from_domain",
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
def create_1d_amr_grid(
    domain_1d: TensorProductGrid,
    initial_intervals: int = 10,
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    backend: BaseBackend | None = None,
) -> OneDimensionalAMRGrid:
    """
    Create 1D AMR grid from TensorProductGrid (1D).

    Args:
        domain_1d: 1D domain specification
        initial_intervals: Initial number of intervals
        error_threshold: Error threshold for refinement
        max_levels: Maximum refinement levels
        backend: Computational backend

    Returns:
        OneDimensionalAMRGrid ready for adaptive refinement
    """
    # Compute domain length from bounds
    min_coords, max_coords = domain_1d.get_bounds()
    domain_length = float(max_coords[0]) - float(min_coords[0])

    criteria = AMRRefinementCriteria(
        error_threshold=error_threshold,
        max_refinement_levels=max_levels,
        min_cell_size=domain_length / (initial_intervals * 2**max_levels),
    )

    return OneDimensionalAMRGrid(
        domain_1d=domain_1d,
        initial_num_intervals=initial_intervals,
        refinement_criteria=criteria,
        backend=backend,
    )


# =============================================================================
# Backward Compatibility Aliases (deprecated, will be removed in v1.0.0)
# =============================================================================


def _deprecated_alias(name: str, new_name: str):
    """Emit deprecation warning for old names."""
    warnings.warn(
        f"{name} is deprecated, use {new_name} instead. Will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=3,
    )


class OneDimensionalAMRMesh(OneDimensionalAMRGrid):
    """Deprecated alias for OneDimensionalAMRGrid."""

    def __init__(self, *args, **kwargs):
        _deprecated_alias("OneDimensionalAMRMesh", "OneDimensionalAMRGrid")
        super().__init__(*args, **kwargs)


def create_1d_amr_mesh(
    domain_1d: TensorProductGrid,
    initial_intervals: int = 10,
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    backend: BaseBackend | None = None,
) -> OneDimensionalAMRGrid:
    """Deprecated alias for create_1d_amr_grid."""
    _deprecated_alias("create_1d_amr_mesh", "create_1d_amr_grid")
    return create_1d_amr_grid(
        domain_1d=domain_1d,
        initial_intervals=initial_intervals,
        error_threshold=error_threshold,
        max_levels=max_levels,
        backend=backend,
    )
