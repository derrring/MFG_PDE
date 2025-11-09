# Geometry Consolidation - Quick Start Guide

**For**: Implementing the consolidation plan
**See**: GEOMETRY_CONSOLIDATION_PLAN.md for full details
**Status**: Ready to implement

---

## TL;DR

**What**: Consolidate `GeometryProtocol` + `BaseGeometry` → single `Geometry` ABC
**Why**: Eliminate fragmentation, add solver operations, enable type-safe solver initialization
**When**: v0.11.0 - v1.0.0 (6-8 weeks)

---

## Quick Decision Tree

**Q1: Should we do this now?**
- ✅ YES if: Ready to commit to 6-8 week gradual migration
- ❌ NO if: Need to ship other features urgently

**Q2: What's the minimal viable first step?**
- Create `Geometry` ABC in new file `mfg_pde/geometry/base.py`
- Migrate ONE class (`TensorProductGrid`) as proof of concept
- Update ONE solver (`HJBFDMSolver`) to use new pattern
- **Estimated time**: 4-8 hours

**Q3: What breaks immediately?**
- ❌ Nothing! Old code keeps working via import aliases
- ✅ New code gets better interface

---

## Phase 1: Proof of Concept (Do This First)

### Step 1: Create `Geometry` ABC (30 min)

Create `mfg_pde/geometry/base.py`:

```python
from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray

from .geometry_protocol import GeometryType  # Reuse enum


class Geometry(ABC):
    """Unified base class for all MFG geometries."""

    # Data interface
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def geometry_type(self) -> GeometryType: ...

    @property
    @abstractmethod
    def num_spatial_points(self) -> int: ...

    @abstractmethod
    def get_spatial_grid(self) -> NDArray: ...

    @abstractmethod
    def get_bounds(self) -> tuple[NDArray, NDArray] | None: ...

    @abstractmethod
    def get_problem_config(self) -> dict: ...

    # Solver operation interface (NEW)
    @abstractmethod
    def get_laplacian_operator(self) -> Callable:
        """Return: (u: NDArray, point_idx) -> float"""
        ...

    @abstractmethod
    def get_gradient_operator(self) -> Callable:
        """Return: (u: NDArray, point_idx) -> NDArray"""
        ...

    @abstractmethod
    def get_interpolator(self) -> Callable:
        """Return: (u: NDArray, point: NDArray) -> float"""
        ...

    @abstractmethod
    def get_boundary_handler(self):
        """Return boundary condition handler."""
        ...

    # Grid utilities (default None for non-Cartesian)
    def get_grid_spacing(self) -> list[float] | None:
        return None

    def get_grid_shape(self) -> tuple[int, ...] | None:
        return None


class CartesianGrid(Geometry):
    """ABC for regular Cartesian grids."""

    # Override with concrete implementations
    def get_grid_spacing(self) -> list[float]:
        raise NotImplementedError("Subclass must implement")

    def get_grid_shape(self) -> tuple[int, ...]:
        raise NotImplementedError("Subclass must implement")
```

### Step 2: Migrate `TensorProductGrid` (1-2 hours)

Edit `mfg_pde/geometry/tensor_product_grid.py`:

```python
# Add import
from .base import CartesianGrid

# Change class definition
class TensorProductGrid(CartesianGrid):  # Was: no inheritance
    """Tensor product grid for multi-dimensional structured domains."""

    # ... existing __init__ and data methods stay the same ...

    # NEW: Implement solver operations
    def get_laplacian_operator(self) -> Callable:
        """Return finite difference Laplacian."""
        def laplacian_fd(u: NDArray, idx: tuple[int, ...]) -> float:
            """
            Compute Laplacian at grid point idx using central differences.

            Args:
                u: Solution array of shape self.grid_shape
                idx: Grid index tuple (i, j, k, ...)

            Returns:
                Laplacian value at idx
            """
            laplacian = 0.0
            for dim in range(self.dimension):
                dx = self.spacing[dim]
                # Create index tuples for neighbors
                idx_plus = list(idx)
                idx_minus = list(idx)
                idx_plus[dim] = min(idx[dim] + 1, self.num_points[dim] - 1)
                idx_minus[dim] = max(idx[dim] - 1, 0)

                # Central difference: (u[i+1] - 2*u[i] + u[i-1]) / dx^2
                laplacian += (
                    u[tuple(idx_plus)] - 2 * u[idx] + u[tuple(idx_minus)]
                ) / (dx ** 2)

            return laplacian

        return laplacian_fd

    def get_gradient_operator(self) -> Callable:
        """Return finite difference gradient."""
        def gradient_fd(u: NDArray, idx: tuple[int, ...]) -> NDArray:
            """
            Compute gradient at grid point idx using central differences.

            Args:
                u: Solution array of shape self.grid_shape
                idx: Grid index tuple (i, j, k, ...)

            Returns:
                Gradient vector [du/dx1, du/dx2, ...]
            """
            gradient = np.zeros(self.dimension)
            for dim in range(self.dimension):
                dx = self.spacing[dim]
                idx_plus = list(idx)
                idx_minus = list(idx)
                idx_plus[dim] = min(idx[dim] + 1, self.num_points[dim] - 1)
                idx_minus[dim] = max(idx[dim] - 1, 0)

                # Central difference: (u[i+1] - u[i-1]) / (2*dx)
                gradient[dim] = (u[tuple(idx_plus)] - u[tuple(idx_minus)]) / (2 * dx)

            return gradient

        return gradient_fd

    def get_interpolator(self) -> Callable:
        """Return linear interpolator."""
        def interpolate_linear(u: NDArray, point: NDArray) -> float:
            """
            Linear interpolation at arbitrary point.

            Args:
                u: Solution array
                point: Physical coordinates [x, y, z, ...]

            Returns:
                Interpolated value
            """
            # Convert physical coords to grid indices
            indices = []
            for dim in range(self.dimension):
                x_coord = point[dim]
                x_min, x_max = self.bounds[dim]
                # Map to [0, num_points-1]
                idx_float = (x_coord - x_min) / (x_max - x_min) * (self.num_points[dim] - 1)
                indices.append(idx_float)

            # Use scipy's interpn for nD linear interpolation
            from scipy.interpolate import RegularGridInterpolator
            interpolator = RegularGridInterpolator(
                self.coordinates, u, method='linear', bounds_error=False, fill_value=0.0
            )
            return float(interpolator(point))

        return interpolate_linear

    def get_boundary_handler(self):
        """Return boundary condition handler."""
        # Placeholder - implement based on BC type
        from mfg_pde.geometry.boundary_conditions_1d import BoundaryConditions
        return BoundaryConditions(bc_type="periodic")  # Default

    def get_bounds(self) -> tuple[NDArray, NDArray]:
        """Return bounding box."""
        min_coords = np.array([b[0] for b in self.bounds])
        max_coords = np.array([b[1] for b in self.bounds])
        return min_coords, max_coords

    # Override grid utilities
    def get_grid_spacing(self) -> list[float]:
        return self.spacing

    def get_grid_shape(self) -> tuple[int, ...]:
        return tuple(self.num_points)
```

### Step 3: Update ONE Solver (1-2 hours)

Edit `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`:

```python
# Add imports
from mfg_pde.geometry.base import CartesianGrid

class HJBFDMSolver:
    def __init__(self, problem: MFGProblemProtocol):
        """
        Initialize HJB FDM solver (requires Cartesian grid).

        Args:
            problem: MFG problem with CartesianGrid geometry

        Raises:
            TypeError: If geometry is not CartesianGrid
        """
        # Type-safe geometry check (replaces hasattr patterns)
        geom = problem.geometry
        if not isinstance(geom, CartesianGrid):
            raise TypeError(
                f"HJB FDM solver requires CartesianGrid geometry, "
                f"got {type(geom).__name__}. "
                f"For non-Cartesian geometries, use particle-based solvers."
            )

        self.problem = problem
        self.geometry: CartesianGrid = geom

        # Get solver operations from geometry
        self.laplacian_op = geom.get_laplacian_operator()
        self.gradient_op = geom.get_gradient_operator()

        # Grid properties (guaranteed to exist for CartesianGrid)
        self.dimension = geom.dimension
        self.dx = geom.get_grid_spacing()
        self.grid_shape = geom.get_grid_shape()

        # No more hasattr() checks!
```

### Step 4: Test (30 min)

Create `tests/unit/test_geometry/test_consolidated_base.py`:

```python
import numpy as np
import pytest
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.base import Geometry, CartesianGrid


def test_tensorproductgrid_is_geometry():
    """Verify TensorProductGrid satisfies Geometry ABC."""
    grid = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        num_points=[10, 10]
    )

    assert isinstance(grid, Geometry)
    assert isinstance(grid, CartesianGrid)


def test_solver_operations_exist():
    """Verify solver operation methods are callable."""
    grid = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        num_points=[10, 10]
    )

    # Check all required methods exist
    assert callable(grid.get_laplacian_operator())
    assert callable(grid.get_gradient_operator())
    assert callable(grid.get_interpolator())
    assert grid.get_boundary_handler() is not None


def test_laplacian_operator():
    """Test finite difference Laplacian."""
    grid = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        num_points=[10, 10]
    )

    laplacian = grid.get_laplacian_operator()

    # Create test function: u(x,y) = x^2 + y^2
    # Laplacian should be 4 everywhere
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = X**2 + Y**2

    # Test at center point
    lap_value = laplacian(u, (5, 5))
    assert np.isclose(lap_value, 4.0, rtol=0.1)


def test_gradient_operator():
    """Test finite difference gradient."""
    grid = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        num_points=[10, 10]
    )

    gradient = grid.get_gradient_operator()

    # Test function: u(x,y) = 2*x + 3*y
    # Gradient should be [2, 3]
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = 2*X + 3*Y

    grad = gradient(u, (5, 5))
    assert np.allclose(grad, [2.0, 3.0], rtol=0.1)


def test_grid_utilities():
    """Test CartesianGrid-specific utilities."""
    grid = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 2.0)],
        num_points=[10, 20]
    )

    # Check grid spacing
    dx = grid.get_grid_spacing()
    assert len(dx) == 2
    assert np.isclose(dx[0], 1.0/9)  # (1-0) / (10-1)
    assert np.isclose(dx[1], 2.0/19)  # (2-0) / (20-1)

    # Check grid shape
    shape = grid.get_grid_shape()
    assert shape == (10, 20)
```

Run tests:
```bash
pytest tests/unit/test_geometry/test_consolidated_base.py -v
```

### Step 5: Add Backward Compatibility (15 min)

Edit `mfg_pde/geometry/__init__.py`:

```python
# New imports
from .base import Geometry, CartesianGrid, UnstructuredMesh, NetworkGraph

# Keep old imports for backward compatibility
from .geometry_protocol import GeometryProtocol as _GeometryProtocol
from .base_geometry import BaseGeometry as _BaseGeometry

# Create aliases (deprecated, remove in v1.0.0)
import warnings

class GeometryProtocol(_GeometryProtocol):
    """DEPRECATED: Use Geometry instead."""
    def __init_subclass__(cls):
        warnings.warn(
            f"{cls.__name__} uses deprecated GeometryProtocol. "
            f"Use mfg_pde.geometry.Geometry instead. "
            f"Will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2
        )

# Export everything
__all__ = [
    "Geometry",
    "CartesianGrid",
    # ... other exports ...
    "GeometryProtocol",  # Deprecated but still exported
]
```

---

## Stop Here and Evaluate

**After Phase 1 POC, ask**:

1. ✅ Does `TensorProductGrid` satisfy `Geometry` ABC?
2. ✅ Do solver operations work correctly?
3. ✅ Does `HJBFDMSolver` initialize without `hasattr()` checks?
4. ✅ Do tests pass?
5. ✅ Is backward compatibility maintained?

**If YES to all → Proceed to Phase 2 (migrate remaining Cartesian grids)**
**If NO → Debug and fix before continuing**

---

## Phase 2-6: Remaining Work

See `GEOMETRY_CONSOLIDATION_PLAN.md` for full migration plan.

**Summary**:
- Phase 2: Migrate `Domain1D`, `SimpleGrid2D/3D` (1 week)
- Phase 3: Migrate mesh classes (1 week)
- Phase 4: Migrate network classes (1 week)
- Phase 5: Update all solvers (1 week)
- Phase 6: Deprecation & cleanup (1 week)

**Total**: 5-6 weeks after POC

---

## Troubleshooting

**Q: Tests fail with "NotImplementedError"?**
A: Ensure all abstract methods in `Geometry` ABC are implemented.

**Q: Import errors?**
A: Check `__init__.py` exports. Use absolute imports.

**Q: Solver doesn't recognize geometry?**
A: Use `isinstance(geom, CartesianGrid)` not `type(geom) == TensorProductGrid`.

**Q: Laplacian values look wrong?**
A: Check boundary handling (index clamping) and grid spacing calculation.

---

## Next Actions

**Immediate** (do now):
1. Review `GEOMETRY_CONSOLIDATION_PLAN.md`
2. Create branch: `git checkout -b feature/geometry-consolidation`
3. Implement Phase 1 POC (~4-8 hours)
4. Evaluate results

**Short-term** (this week):
5. If POC succeeds, continue to Phase 2

**Long-term** (next 6 weeks):
6. Complete Phases 2-6
7. Release v1.0.0 with unified geometry system

---

**Ready to start?** Begin with Phase 1, Step 1: Create `Geometry` ABC.
