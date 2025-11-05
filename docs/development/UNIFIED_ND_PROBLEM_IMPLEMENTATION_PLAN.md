# Unified nD MFGProblem Implementation Plan

**Issue**: #245
**Branch**: `feature/unified-nd-mfg-problem`
**Target**: v1.0.0
**Status**: IN PROGRESS

## Executive Summary

Radical architecture renovation to create single dimension-agnostic `UnifiedMFGProblem` that works for 1D, 2D, 3D, nD problems.

## Current Architecture Problems

1. **MFGProblem is concrete 1D-only**, not abstract base
2. **GridBasedMFGProblem** is separate deprecated class for 2D+
3. **HighDimMFGProblem** doesn't extend MFGProblem (fragmented hierarchy)
4. **Inconsistent APIs**: `Nx` vs `spatial_shape`, `xmin` vs `domain_bounds`, etc.
5. **Solver duplication**: Separate implementations for each dimension

## Solution Architecture

### Core Hierarchy

```
MFGProblemProtocol (typing.Protocol)
    ↑
BaseMFGProblem (ABC)
    ↑
UnifiedMFGProblem (concrete, dimension-agnostic)
    ↑
├── StochasticMFGProblem
├── NetworkMFGProblem
└── VariationalMFGProblem
```

### Key Design Principles

1. **Protocol-based**: Use `MFGProblemProtocol` for type safety
2. **Dimension-agnostic**: All code works for 1D through nD
3. **Consistent API**: Same attributes/methods regardless of dimension
4. **Backward compatible**: Old API auto-converts with deprecation warnings

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-3)

#### Day 1: Protocol & Abstract Base

**File**: `mfg_pde/core/base_problem.py` (NEW)

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

@runtime_checkable
class MFGProblemProtocol(Protocol):
    """Protocol that ALL MFG problems must satisfy."""

    # Spatial properties
    dimension: int
    spatial_bounds: list[tuple[float, float]]
    spatial_discretization: list[int]
    spatial_grid: NDArray | list[NDArray]  # 1D array or list of nD arrays
    grid_shape: tuple[int, ...]
    grid_spacing: list[float]

    # Temporal properties
    T: float
    Nt: int
    dt: float
    time_grid: NDArray

    # Physical properties
    sigma: float  # Diffusion coefficient

    # MFG components
    def hamiltonian(self, x, m, p, t) -> float: ...
    def terminal_cost(self, x) -> float: ...
    def initial_density(self, x) -> float: ...
    def running_cost(self, x, m, t) -> float: ...


class BaseMFGProblem(ABC):
    """
    True abstract base for ALL MFG problems (1D, 2D, 3D, nD).

    This class defines the universal interface that all MFG problems
    must implement, regardless of dimension or domain type.
    """

    @abstractmethod
    def __init__(
        self,
        spatial_bounds: list[tuple[float, float]],
        spatial_discretization: list[int],
        time_domain: tuple[float, int],
        diffusion_coeff: float,
    ):
        """
        Initialize MFG problem (dimension-agnostic).

        Args:
            spatial_bounds: [(xmin_0, xmax_0), (xmin_1, xmax_1), ...]
            spatial_discretization: [Nx_0, Nx_1, ...] points per dimension
            time_domain: (T_final, Nt)
            diffusion_coeff: Diffusion coefficient σ
        """
        # Spatial setup
        self.spatial_bounds = spatial_bounds
        self.spatial_discretization = spatial_discretization
        self.dimension = len(spatial_bounds)

        # Temporal setup
        self.T, self.Nt = time_domain
        self.dt = self.T / self.Nt
        self.time_grid = np.linspace(0, self.T, self.Nt + 1)

        # Physical parameters
        self.sigma = diffusion_coeff

        # Grid properties (computed by subclass)
        self.grid_shape: tuple[int, ...] = tuple(spatial_discretization)
        self.grid_spacing: list[float] = []
        self.spatial_grid: NDArray | list[NDArray] = None

    @abstractmethod
    def _build_spatial_grid(self) -> None:
        """Build dimension-specific spatial grid."""
        pass

    @abstractmethod
    def hamiltonian(self, x, m, p, t) -> float:
        """Hamiltonian H(x, m, p, t)."""
        pass

    @abstractmethod
    def terminal_cost(self, x) -> float:
        """Terminal cost g(x)."""
        pass

    @abstractmethod
    def initial_density(self, x) -> float:
        """Initial density m_0(x)."""
        pass

    def running_cost(self, x, m, t) -> float:
        """Running cost f(x, m, t). Default: 0."""
        return 0.0
```

#### Day 2: Unified Problem Implementation

**File**: `mfg_pde/core/unified_problem.py` (NEW)

```python
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .base_problem import BaseMFGProblem
from .mfg_problem import MFGComponents

if TYPE_CHECKING:
    from numpy.typing import NDArray


class UnifiedMFGProblem(BaseMFGProblem):
    """
    Unified MFG problem class for 1D, 2D, 3D, nD Cartesian grids.

    This single class replaces:
    - Old MFGProblem (1D only)
    - GridBasedMFGProblem (2D+ deprecated)
    - ExampleMFGProblem (legacy)

    Examples:
        # 1D problem
        problem_1d = UnifiedMFGProblem(
            spatial_bounds=[(0.0, 1.0)],
            spatial_discretization=[100],
            time_domain=(1.0, 50),
            diffusion_coeff=0.1
        )

        # 2D problem
        problem_2d = UnifiedMFGProblem(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
            spatial_discretization=[50, 50],
            time_domain=(1.0, 100),
            diffusion_coeff=0.1
        )

        # 3D problem - same API!
        problem_3d = UnifiedMFGProblem(
            spatial_bounds=[(0, 1), (0, 1), (0, 1)],
            spatial_discretization=[20, 20, 20],
            time_domain=(1.0, 50),
            diffusion_coeff=0.1
        )
    """

    def __init__(
        self,
        spatial_bounds: list[tuple[float, float]],
        spatial_discretization: list[int],
        time_domain: tuple[float, int] = (1.0, 100),
        diffusion_coeff: float = 0.1,
        components: MFGComponents | None = None,
        domain_type: str = "grid",
    ):
        """
        Initialize unified MFG problem.

        Args:
            spatial_bounds: [(xmin_0, xmax_0), ...] for each dimension
            spatial_discretization: [Nx_0, ...] grid points per dimension
            time_domain: (T_final, Nt)
            diffusion_coeff: Diffusion coefficient σ
            components: Custom MFG components (Hamiltonian, costs, etc.)
            domain_type: 'grid' (default) or other domain types
        """
        super().__init__(spatial_bounds, spatial_discretization, time_domain, diffusion_coeff)

        self.domain_type = domain_type
        self.components = components or self._default_components()

        # Build spatial grid
        self._build_spatial_grid()

        # Setup initial/terminal conditions
        self._setup_conditions()

    def _build_spatial_grid(self) -> None:
        """Build nD tensor product grid."""
        # Create 1D grids for each dimension
        grids_1d = [
            np.linspace(bounds[0], bounds[1], n_points)
            for bounds, n_points in zip(self.spatial_bounds, self.spatial_discretization)
        ]

        # Compute grid spacing
        self.grid_spacing = [
            (bounds[1] - bounds[0]) / (n_points - 1)
            for bounds, n_points in zip(self.spatial_bounds, self.spatial_discretization)
        ]

        # Create nD meshgrid
        if self.dimension == 1:
            # For 1D, just use the 1D array
            self.spatial_grid = grids_1d[0]
        else:
            # For nD, create meshgrid with indexing='ij' (matrix indexing)
            self.spatial_grid = np.meshgrid(*grids_1d, indexing='ij')

        # Compute total number of spatial points
        self.num_spatial_points = np.prod(self.spatial_discretization)

    def _default_components(self) -> MFGComponents:
        """Create default LQ-type components."""
        from .mfg_problem import MFGComponents

        return MFGComponents(
            description="Default LQ-type MFG problem",
            problem_type="lq_mfg"
        )

    def _setup_conditions(self) -> None:
        """Setup initial density and terminal cost arrays."""
        # Evaluate initial density
        if self.dimension == 1:
            x_vals = self.spatial_grid
            self.m_init = self.initial_density(x_vals)
        else:
            # For nD, evaluate at each grid point
            points_shape = self.grid_shape
            self.m_init = np.zeros(points_shape)

            # TODO: Vectorize this for performance
            for idx in np.ndindex(points_shape):
                x_point = tuple(grid[idx] for grid in self.spatial_grid)
                self.m_init[idx] = self.initial_density(x_point)

        # Normalize initial density
        self.m_init = self.m_init / np.sum(self.m_init)

        # Similarly for terminal cost
        if self.dimension == 1:
            self.u_fin = self.terminal_cost(self.spatial_grid)
        else:
            self.u_fin = np.zeros(self.grid_shape)
            for idx in np.ndindex(self.grid_shape):
                x_point = tuple(grid[idx] for grid in self.spatial_grid)
                self.u_fin[idx] = self.terminal_cost(x_point)

    # MFG component methods (can be overridden or use components)

    def hamiltonian(self, x, m, p, t) -> float:
        """
        Hamiltonian H(x, m, p, t).

        Default: H = 0.5·|p|² (LQ-type)
        """
        if self.components.hamiltonian_func:
            return self.components.hamiltonian_func(x, m, p, t)

        # Default LQ Hamiltonian
        if isinstance(p, (list, tuple, np.ndarray)):
            return 0.5 * np.sum(np.array(p)**2)
        return 0.5 * p**2

    def terminal_cost(self, x) -> float:
        """
        Terminal cost g(x).

        Default: g(x) = 0.5·|x|²
        """
        if self.components.final_value_func:
            return self.components.final_value_func(x)

        # Default quadratic cost
        if isinstance(x, (list, tuple, np.ndarray)):
            return 0.5 * np.sum(np.array(x)**2)
        return 0.5 * x**2

    def initial_density(self, x) -> float:
        """
        Initial density m_0(x).

        Default: Uniform over domain
        """
        if self.components.initial_density_func:
            return self.components.initial_density_func(x)

        # Default: uniform (will be normalized)
        return 1.0
```

### Phase 2: Compatibility Layer (Days 3-4)

**File**: `mfg_pde/compat/legacy_api.py` (UPDATE)

Add auto-conversion functions for old APIs.

### Phase 3: Solver Refactoring (Days 5-10)

Update all solvers to use `MFGProblemProtocol` and work dimension-agnostically.

### Phase 4: Testing & Documentation (Days 11-14)

Comprehensive tests and migration guide.

## Migration Path for Users

### Old API (deprecated but works)

```python
# 1D
from mfg_pde import MFGProblem
problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, Nt=50, T=1.0, sigma=0.1)

# 2D
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
problem = GridBasedMFGProblem(domain_bounds=(0, 1, 0, 1), grid_resolution=20, ...)
```

### New API (recommended)

```python
from mfg_pde import UnifiedMFGProblem

# 1D
problem = UnifiedMFGProblem(
    spatial_bounds=[(0.0, 1.0)],
    spatial_discretization=[100],
    time_domain=(1.0, 50),
    diffusion_coeff=0.1
)

# 2D
problem = UnifiedMFGProblem(
    spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
    spatial_discretization=[20, 20],
    time_domain=(1.0, 100),
    diffusion_coeff=0.1
)
```

## Success Criteria

1. ✅ All existing tests pass with compatibility layer
2. ✅ New `UnifiedMFGProblem` works for 1D, 2D, 3D
3. ✅ Solvers work dimension-agnostically
4. ✅ Performance: No regression vs current implementation
5. ✅ Documentation: Complete migration guide
6. ✅ Examples: Updated to new API

## Timeline

- **Days 1-3**: Core infrastructure
- **Days 4-5**: Compatibility layer
- **Days 6-10**: Solver refactoring
- **Days 11-14**: Testing & documentation
- **Week 3**: v1.0.0-rc1
- **Week 4**: v1.0.0 release

## Next Steps

1. Create `base_problem.py` with protocol and ABC
2. Implement `unified_problem.py`
3. Update import structure in `__init__.py`
4. Begin solver refactoring

---

**Started**: 2025-11-05
**Last Updated**: 2025-11-05
