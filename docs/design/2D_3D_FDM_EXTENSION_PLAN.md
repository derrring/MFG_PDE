# N-Dimensional FDM Solver Extension Plan

**Date**: 2025-10-30
**Status**: Architecture Design Complete (Dimension-Agnostic)
**Timeline**: 6 weeks for full production integration

## Executive Summary

This document outlines the architecture design for extending MFG_PDE's Finite Difference Method (FDM) solvers from 1D to **arbitrary n-dimensional** spatial domains. The current implementation is tightly coupled to 1D geometry, requiring careful refactoring to maintain backward compatibility while enabling multi-dimensional MFG research.

**Key Design Principles**:
- **Single Implementation**: One codebase works for 1D, 2D, 3D, ..., nD (no separate classes)
- **Automatic Dimension Detection**: Problem dimension detected from spatial bounds specification
- **Computational Feasibility Warnings**: Automatic warnings for high-dimensional problems (grid-based methods become impractical beyond 3-4 dimensions due to curse of dimensionality)
- **Backward Compatibility**: All existing 1D code continues to work without changes

**Key Findings**:
- Current 1D FDM has clean separation in HJB (3 dimension-specific functions, rest reusable)
- FP FDM is entirely 1D-specific (requires complete rewrite for n-D)
- MFGProblem class is 1D-only, needs dimension-agnostic redesign
- Recommended: Two-phase approach (research prototype → production integration)

**Computational Scalability**:
| Dimension | Grid Size (per dim) | Total Points | Memory | Practical? |
|-----------|---------------------|--------------|--------|------------|
| 1D | 1000 | 1,000 | ~8 KB | ✅ Excellent |
| 2D | 100×100 | 10,000 | ~80 KB | ✅ Excellent |
| 3D | 50×50×50 | 125,000 | ~1 MB | ✅ Good |
| 4D | 30×30×30×30 | 810,000 | ~6 MB | ⚠️ Marginal |
| 5D | 20×20×20×20×20 | 3,200,000 | ~25 MB | ❌ Impractical |
| 6D+ | ... | ... | ... | ❌ Use particle/network methods |

---

## 1. Current 1D FDM Architecture Analysis

### 1.1 HJB FDM Solver

**File Structure**:
- `hjb_fdm.py` (107 lines): Thin wrapper, parameter management only
- `base_hjb.py` (768 lines): **Actual implementation**

**Dimension-Specific Components** (require n-D versions):

| Function | Lines | Purpose | 1D Implementation |
|----------|-------|---------|-------------------|
| `_calculate_derivatives()` | 77-156 | Gradient computation | Central difference: `(u[i+1] - u[i-1]) / (2*Dx)` |
| `compute_hjb_residual()` | 198-305 | PDE residual | Laplacian: `(u[i+1] - 2*u[i] + u[i-1]) / Dx^2` |
| `compute_hjb_jacobian()` | 307-488 | Newton Jacobian matrix | Tridiagonal sparse matrix (3 bands) |

**Dimension-Agnostic Components** (reuse as-is):

| Function | Lines | Purpose |
|----------|-------|---------|
| `newton_hjb_step()` | 490-557 | Single Newton iteration |
| `solve_hjb_timestep_newton()` | 559-655 | Newton convergence loop |
| `solve_hjb_system_backward()` | 657-768 | Backward time-stepping |

**Key Insight**: Only ~25% of HJB code is dimension-specific!

### 1.2 FP FDM Solver

**File Structure**:
- `fp_fdm.py` (294 lines): Complete 1D implementation

**Critical 1D Dependencies**:
```python
# Line 83-293: Forward time loop
for k_idx_fp in range(Nt - 1):  # Time (dimension-agnostic)
    for i in range(Nx):  # 1D SPATIAL LOOP (dimension-specific!)
        # Matrix assembly using:
        # - i-1 (left neighbor)
        # - i   (center)
        # - i+1 (right neighbor)
        # Result: Tridiagonal matrix
```

**Boundary Conditions** (lines 98-250):
- **Periodic**: `(i+1) % Nx`, `(i-1+Nx) % Nx`
- **Dirichlet**: `m[0] = left_value`, `m[-1] = right_value`
- **No-flux**: `∂m/∂x = 0` at endpoints

**Key Insight**: Entire FP FDM is 1D-specific, requires complete rewrite for n-D.

### 1.3 MFGProblem Class

**Current 1D-only geometry** (`mfg_problem.py:lines 103-114`):
```python
self.xmin, self.xmax = xmin, xmax
self.Nx = Nx
self.Dx = (xmax - xmin) / Nx
self.xSpace = np.linspace(xmin, xmax, Nx + 1)  # 1D grid

# Arrays: (Nx+1,) shaped
self.f_potential = np.zeros(self.Nx + 1)
self.u_fin = np.zeros(self.Nx + 1)
self.m_init = np.zeros(self.Nx + 1)
```

**No support for**:
- Multiple spatial dimensions
- Arbitrary dimension parameter
- Dynamic array shapes

---

## 2. Dimension-Agnostic Extension Architecture

### 2.1 Core Design Philosophy

**Single Unified Implementation**:
```
┌─────────────────────────────────────────────────────────────┐
│  OLD APPROACH (Rejected)        NEW APPROACH (Approved)     │
├─────────────────────────────────────────────────────────────┤
│  HJBFDMSolver                   HJBFDMSolver                 │
│  ├── HJBFDM1DSolver             └── Works for ANY dimension │
│  ├── HJBFDM2DSolver                  (1D, 2D, 3D, ..., nD) │
│  └── HJBFDM3DSolver                                          │
│                                                              │
│  (Separate implementations)     (Single implementation)     │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- No code duplication (single implementation for all dimensions)
- Automatic dimension detection (no manual class selection)
- Maintainability (fix bugs once, benefits all dimensions)
- Future-proof (works for any dimension without code changes)

### 2.2 Direct Unified Approach

**Goal**: Extend existing `MFGProblem` to support arbitrary dimensions while maintaining 100% backward compatibility with existing 1D code.

**Design Principle**: **Single unified class** that works for all dimensions (1D, 2D, 3D, ..., nD) through automatic dimension detection.

#### Unified MFGProblem API Design

```python
# mfg_pde/core/mfg_problem.py (extended)
class MFGProblem:
    """
    Unified MFG problem class supporting arbitrary spatial dimensions.

    Supports two initialization modes:
    1. Legacy 1D mode (backward compatible): Specify Nx, xmin, xmax
    2. N-dimensional mode (new): Specify spatial_bounds, spatial_discretization

    Dimension is automatically detected from the provided parameters.
    """

    def __init__(
        self,
        # Legacy 1D parameters (backward compatible)
        Nx: int | None = None,
        xmin: float = 0.0,
        xmax: float = 1.0,

        # New n-D parameters
        spatial_bounds: list[tuple[float, float]] | None = None,
        spatial_discretization: list[int] | None = None,

        # Common parameters
        T: float = 1.0,
        Nt: int = 51,
        sigma: float = 1.0,
        coefCT: float = 0.5,
        suppress_warnings: bool = False
    ):
        """
        Initialize MFG problem.

        Mode 1 - Legacy 1D (backward compatible):
            >>> problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, Nt=100)

        Mode 2 - N-dimensional (new):
            >>> # 2D
            >>> problem = MFGProblem(
            ...     spatial_bounds=[(0, 1), (0, 1)],
            ...     spatial_discretization=[50, 50],
            ...     Nt=50
            ... )
            >>>
            >>> # 3D
            >>> problem = MFGProblem(
            ...     spatial_bounds=[(0, 1), (0, 1), (0, 1)],
            ...     spatial_discretization=[30, 30, 30],
            ...     Nt=30
            ... )
        """

        # Detect initialization mode
        if Nx is not None and spatial_bounds is None:
            # Mode 1: Legacy 1D
            self._init_1d_legacy(Nx, xmin, xmax, T, Nt, sigma, coefCT)

        elif spatial_bounds is not None and Nx is None:
            # Mode 2: N-dimensional
            self._init_nd(spatial_bounds, spatial_discretization,
                         T, Nt, sigma, coefCT, suppress_warnings)

        elif Nx is not None and spatial_bounds is not None:
            raise ValueError(
                "Cannot specify both Nx (1D mode) and spatial_bounds (n-D mode). "
                "Use only one initialization mode."
            )

        else:
            raise ValueError(
                "Must specify either:\n"
                "  - Nx (for 1D problems, backward compatible)\n"
                "  - spatial_bounds (for n-D problems, new API)"
            )

    def _init_1d_legacy(self, Nx, xmin, xmax, T, Nt, sigma, coefCT):
        """Initialize in legacy 1D mode (backward compatible)."""
        self.dimension = 1
        self.Nx = Nx
        self.Nt = Nt
        self.Dx = (xmax - xmin) / Nx
        self.Dt = T / Nt
        self.sigma = sigma
        self.coefCT = coefCT

        # Use TensorProductGrid internally (but maintain legacy interface)
        from mfg_pde.geometry import TensorProductGrid
        self._grid = TensorProductGrid(
            dimension=1,
            bounds=[(xmin, xmax)],
            num_points=[Nx]
        )

        # Legacy 1D arrays (maintain backward compatibility)
        self.xSpace = self._grid.coordinates[0]
        self.f_potential = np.zeros(Nx + 1)
        self.u_fin = np.zeros(Nx + 1)
        self.m_init = np.zeros(Nx + 1)

    def _init_nd(self, spatial_bounds, spatial_discretization,
                 T, Nt, sigma, coefCT, suppress_warnings):
        """Initialize in n-dimensional mode."""
        from mfg_pde.geometry import TensorProductGrid

        self.dimension = len(spatial_bounds)
        self.spatial_bounds = spatial_bounds
        self.N_spatial = spatial_discretization

        # Temporal discretization
        self.Nt = Nt
        self.Dt = T / Nt

        # Physical parameters
        self.sigma = sigma
        self.coefCT = coefCT

        # Create n-dimensional grid using existing TensorProductGrid
        self._grid = TensorProductGrid(
            dimension=self.dimension,
            bounds=spatial_bounds,
            num_points=spatial_discretization
        )

        # Grid spacings
        self.D_spatial = self._grid.spacing

        # Spatial shape
        self.spatial_shape = tuple(N + 1 for N in spatial_discretization)

        # Check computational feasibility
        if not suppress_warnings:
            self._check_computational_feasibility()

        # N-dimensional arrays
        self.f_potential = np.zeros(self.spatial_shape)
        self.u_fin = np.zeros(self.spatial_shape)
        self.m_init = np.zeros(self.spatial_shape)

        # Initialize with defaults
        self._initialize_nd_defaults()

    def _check_computational_feasibility(self):
        """Warn about computational limits for high-dimensional problems."""
        MAX_PRACTICAL_DIMENSION = 4
        MAX_TOTAL_GRID_POINTS = 10_000_000

        total_spatial_points = self._grid.total_points()
        total_points = total_spatial_points * (self.Nt + 1)
        memory_mb = total_points * 8 / (1024**2)

        if self.dimension > MAX_PRACTICAL_DIMENSION:
            warnings.warn(
                f"\n{'='*80}\n"
                f"⚠️  HIGH DIMENSION WARNING ⚠️\n"
                f"{'='*80}\n"
                f"Problem dimension: {self.dimension}D\n"
                f"Practical limit for grid-based FDM: {MAX_PRACTICAL_DIMENSION}D\n"
                f"\n"
                f"Grid-based FDM methods scale as O(N^d) and become computationally\n"
                f"impractical for d > 4 due to the curse of dimensionality.\n"
                f"\n"
                f"Your problem:\n"
                f"  - Spatial points: {total_spatial_points:,}\n"
                f"  - Time steps: {self.Nt + 1}\n"
                f"  - Total points: {total_points:,}\n"
                f"  - Memory estimate: {memory_mb:.1f} MB per array\n"
                f"\n"
                f"Consider alternative methods for high-dimensional MFG:\n"
                f"  1. Particle-based collocation (ParticleHJBSolver)\n"
                f"  2. Network-based MFG (NetworkMFGSolver)\n"
                f"\n"
                f"To suppress this warning, set suppress_warnings=True.\n"
                f"{'='*80}\n",
                UserWarning,
                stacklevel=3
            )

        elif total_spatial_points > MAX_TOTAL_GRID_POINTS:
            warnings.warn(
                f"\n{'='*80}\n"
                f"⚠️  LARGE GRID WARNING ⚠️\n"
                f"{'='*80}\n"
                f"Total spatial grid points: {total_spatial_points:,}\n"
                f"Memory estimate: {memory_mb:.1f} MB per array\n"
                f"\n"
                f"Consider:\n"
                f"  - Reducing grid resolution\n"
                f"  - Using adaptive mesh refinement\n"
                f"  - Switching to particle-based methods\n"
                f"{'='*80}\n",
                UserWarning,
                stacklevel=3
            )

    def _initialize_nd_defaults(self):
        """Initialize with dimension-agnostic default functions."""
        # Default: Gaussian initial density at domain center
        center = np.array([
            (bounds[0] + bounds[1]) / 2
            for bounds in self.spatial_bounds
        ])

        # Compute squared distance from center
        grids = self._grid.meshgrid()
        squared_dist = np.zeros(self.spatial_shape)
        for dim in range(self.dimension):
            squared_dist += (grids[dim] - center[dim])**2

        # Initial density: Gaussian
        self.m_init = np.exp(-squared_dist / (2 * 0.1**2))
        self.m_init /= np.sum(self.m_init) * np.prod(self.D_spatial)

        # Terminal cost: Quadratic distance
        self.u_fin = squared_dist

        # Running cost: Zero (default)
        self.f_potential = np.zeros(self.spatial_shape)
```

#### Backward Compatibility Guarantee

**ALL existing 1D code continues to work unchanged**:

```python
# Existing 1D code (NO CHANGES REQUIRED)
from mfg_pde import MFGProblem

problem = MFGProblem(Nx=100, Nt=100)  # ✅ Works exactly as before
problem.Nx  # ✅ Still available
problem.Dx  # ✅ Still available
problem.xSpace  # ✅ Still available
problem.f_potential  # ✅ Still (Nx+1,) shaped array
```

**New n-D capabilities**:

```python
# New 2D code
problem_2d = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    Nt=50
)
problem_2d.dimension  # 2
problem_2d.spatial_shape  # (51, 51)
problem_2d.f_potential  # (51, 51) shaped array

# New 3D code
problem_3d = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1), (0, 1)],
    spatial_discretization=[30, 30, 30],
    Nt=30
)
problem_3d.dimension  # 3
problem_3d.spatial_shape  # (31, 31, 31)
```

#### Implementation Strategy

1. **Week 1**: Extend `MFGProblem` with dual initialization modes
2. **Week 2**: Update HJB/FP FDM solvers to detect dimension from problem
3. **Weeks 3-4**: Implement n-D gradient/Laplacian/Jacobian functions
4. **Weeks 5-6**: Comprehensive testing (backward compat + n-D validation)

### 2.3 Dimension-Agnostic Problem Class

```python
# experiments/nd_fdm_solvers/problem_nd.py
class MFGProblemND:
    """
    N-dimensional MFG problem (research prototype).

    Geometry: [x1_min, x1_max] × [x2_min, x2_max] × ... × [0, T]
    Grid: (N1+1) × (N2+1) × ... × (Nt+1) points

    Args:
        spatial_bounds: List of (min, max) tuples for each dimension
                       1D: [(0, 1)]
                       2D: [(0, 1), (0, 1)]
                       3D: [(0, 1), (0, 1), (0, 1)]
        spatial_discretization: Number of intervals per dimension [N1, N2, ...]
        T: Terminal time
        Nt: Number of time intervals
        suppress_warnings: If False, warns about high-dimensional problems
    """

    # Computational feasibility thresholds
    MAX_PRACTICAL_DIMENSION = 4
    MAX_TOTAL_GRID_POINTS = 10_000_000  # 10M points (warns if exceeded)

    def __init__(self,
                 spatial_bounds: list[tuple[float, float]],
                 spatial_discretization: list[int],
                 T: float = 1.0,
                 Nt: int = 51,
                 sigma: float = 1.0,
                 coefCT: float = 0.5,
                 suppress_warnings: bool = False):

        # Detect spatial dimension
        self.dimension = len(spatial_bounds)
        if self.dimension != len(spatial_discretization):
            raise ValueError("Mismatch: len(spatial_bounds) != len(spatial_discretization)")

        # Spatial discretization (dimension-agnostic)
        self.spatial_bounds = spatial_bounds
        self.N_spatial = spatial_discretization  # [N1, N2, ..., Nd]

        # Grid spacings for each dimension
        self.D_spatial = [
            (bounds[1] - bounds[0]) / N
            for bounds, N in zip(spatial_bounds, spatial_discretization)
        ]  # [D1, D2, ..., Dd]

        # Temporal discretization
        self.Nt = Nt
        self.Dt = T / Nt

        # Physical parameters
        self.sigma = sigma
        self.coefCT = coefCT

        # N-dimensional spatial shape (N1+1, N2+1, ..., Nd+1)
        spatial_shape = tuple(N + 1 for N in spatial_discretization)
        self.spatial_shape = spatial_shape

        # Total grid points
        self.total_spatial_points = np.prod(spatial_shape)
        self.total_points = self.total_spatial_points * (Nt + 1)

        # Check computational feasibility
        if not suppress_warnings:
            self._check_computational_feasibility()

        # N-dimensional grids (using meshgrid with dynamic dimensions)
        grid_1d = [
            np.linspace(bounds[0], bounds[1], N + 1)
            for bounds, N in zip(spatial_bounds, spatial_discretization)
        ]
        self.grids = np.meshgrid(*grid_1d, indexing='ij')

        # Arrays: (N1+1, N2+1, ..., Nd+1) shaped
        self.f_potential = np.zeros(spatial_shape)
        self.u_fin = np.zeros(spatial_shape)
        self.m_init = np.zeros(spatial_shape)

        # Initialize with dimension-agnostic functions
        self._initialize_nd()

    def _check_computational_feasibility(self):
        """
        Check if problem size is computationally feasible for grid-based FDM.

        Warns if:
        - Dimension > 4 (curse of dimensionality)
        - Total grid points > 10M (memory concerns)

        Grid-based methods scale as O(N^d), becoming impractical for high dimensions.
        """
        # Dimension warning
        if self.dimension > self.MAX_PRACTICAL_DIMENSION:
            memory_estimate_mb = self.total_points * 8 / (1024**2)  # float64
            warnings.warn(
                f"\n{'='*80}\n"
                f"⚠️  HIGH DIMENSION WARNING ⚠️\n"
                f"{'='*80}\n"
                f"Problem dimension: {self.dimension}D\n"
                f"Practical limit for grid-based FDM: {self.MAX_PRACTICAL_DIMENSION}D\n"
                f"\n"
                f"Grid-based FDM methods scale as O(N^d) and become computationally\n"
                f"impractical for d > 4 due to the curse of dimensionality.\n"
                f"\n"
                f"Your problem:\n"
                f"  - Spatial points: {self.total_spatial_points:,}\n"
                f"  - Time steps: {self.Nt + 1}\n"
                f"  - Total points: {self.total_points:,}\n"
                f"  - Memory estimate: {memory_estimate_mb:.1f} MB per array\n"
                f"\n"
                f"Consider alternative methods for high-dimensional MFG:\n"
                f"  1. Particle-based collocation (ParticleHJBSolver)\n"
                f"  2. Network-based MFG (NetworkMFGSolver)\n"
                f"  3. Curse-of-dimensionality-free methods (NeuralMFGSolver)\n"
                f"\n"
                f"To suppress this warning, set suppress_warnings=True.\n"
                f"{'='*80}\n",
                UserWarning,
                stacklevel=2
            )

        # Memory warning
        elif self.total_spatial_points > self.MAX_TOTAL_GRID_POINTS:
            memory_estimate_mb = self.total_points * 8 / (1024**2)
            warnings.warn(
                f"\n{'='*80}\n"
                f"⚠️  LARGE GRID WARNING ⚠️\n"
                f"{'='*80}\n"
                f"Total spatial grid points: {self.total_spatial_points:,}\n"
                f"Memory estimate: {memory_estimate_mb:.1f} MB per array\n"
                f"\n"
                f"This is a very large grid and may cause:\n"
                f"  - High memory usage\n"
                f"  - Long computation times\n"
                f"  - Numerical stability issues\n"
                f"\n"
                f"Consider:\n"
                f"  - Reducing grid resolution\n"
                f"  - Using adaptive mesh refinement\n"
                f"  - Switching to particle-based methods\n"
                f"{'='*80}\n",
                UserWarning,
                stacklevel=2
            )

    def get_computational_cost_estimate(self) -> dict:
        """
        Estimate computational cost for this problem.

        Returns:
            dict with keys:
                - dimension: Spatial dimension
                - total_points: Total grid points (space × time)
                - memory_mb: Estimated memory per array (MB)
                - is_practical: Boolean (True if dimension <= 4 and points < 10M)
                - warning_level: 'ok', 'high_dimension', or 'high_memory'
        """
        memory_mb = self.total_points * 8 / (1024**2)

        if self.dimension > self.MAX_PRACTICAL_DIMENSION:
            warning_level = 'high_dimension'
            is_practical = False
        elif self.total_spatial_points > self.MAX_TOTAL_GRID_POINTS:
            warning_level = 'high_memory'
            is_practical = False
        else:
            warning_level = 'ok'
            is_practical = True

        return {
            'dimension': self.dimension,
            'total_points': self.total_points,
            'memory_mb': memory_mb,
            'is_practical': is_practical,
            'warning_level': warning_level,
        }

    def _initialize_nd(self):
        """Initialize with dimension-agnostic default functions."""
        # Default: Gaussian initial density at domain center
        center = np.array([
            (bounds[0] + bounds[1]) / 2
            for bounds in self.spatial_bounds
        ])

        # Compute squared distance from center for all grid points
        squared_dist = np.zeros(self.spatial_shape)
        for dim in range(self.dimension):
            squared_dist += (self.grids[dim] - center[dim])**2

        # Initial density: Gaussian
        self.m_init = np.exp(-squared_dist / (2 * 0.1**2))
        self.m_init /= np.sum(self.m_init) * np.prod(self.D_spatial)  # Normalize

        # Terminal cost: Quadratic distance to center
        self.u_fin = squared_dist

        # Running cost: Zero (default)
        self.f_potential = np.zeros(self.spatial_shape)


# ================================================================================
# EXAMPLE USAGE: Demonstrates automatic warnings
# ================================================================================

# 1D: No warnings
problem_1d = MFGProblemND(
    spatial_bounds=[(0, 1)],
    spatial_discretization=[100],
    Nt=100
)
# Output: (none - practical problem)

# 2D: No warnings
problem_2d = MFGProblemND(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[100, 100],
    Nt=100
)
# Output: (none - practical problem)

# 3D: No warnings
problem_3d = MFGProblemND(
    spatial_bounds=[(0, 1), (0, 1), (0, 1)],
    spatial_discretization=[50, 50, 50],
    Nt=50
)
# Output: (none - practical problem)

# 4D: No warnings (at threshold)
problem_4d = MFGProblemND(
    spatial_bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    spatial_discretization=[30, 30, 30, 30],
    Nt=30
)
# Output: (none - practical problem at upper limit)

# 5D: HIGH DIMENSION WARNING
problem_5d = MFGProblemND(
    spatial_bounds=[(0, 1)] * 5,
    spatial_discretization=[20] * 5,
    Nt=20
)
# Output:
# ================================================================================
# ⚠️  HIGH DIMENSION WARNING ⚠️
# ================================================================================
# Problem dimension: 5D
# Practical limit for grid-based FDM: 4D
#
# Grid-based FDM methods scale as O(N^d) and become computationally
# impractical for d > 4 due to the curse of dimensionality.
#
# Your problem:
#   - Spatial points: 3,200,000
#   - Time steps: 21
#   - Total points: 67,200,000
#   - Memory estimate: 512.7 MB per array
#
# Consider alternative methods for high-dimensional MFG:
#   1. Particle-based collocation (ParticleHJBSolver)
#   2. Network-based MFG (NetworkMFGSolver)
#   3. Curse-of-dimensionality-free methods (NeuralMFGSolver)
#
# To suppress this warning, set suppress_warnings=True.
# ================================================================================

# 6D: HIGH DIMENSION WARNING (even worse)
problem_6d = MFGProblemND(
    spatial_bounds=[(0, 1)] * 6,
    spatial_discretization=[15] * 6,
    Nt=10,
    suppress_warnings=True  # Suppress to avoid spam
)
# Would warn about 11M+ spatial points and > 2 GB memory
```

### 2.4 Dimension-Agnostic Gradient Computation

```python
# experiments/nd_fdm_solvers/hjb_fdm_nd.py

def _calculate_gradient_nd(U, indices: tuple, D_spatial: list[float],
                           spatial_shape: tuple, periodic: bool = True):
    """
    Compute n-dimensional gradient using central differences.

    Args:
        U: N-dimensional array (N1+1, N2+1, ..., Nd+1)
        indices: Spatial indices (i1, i2, ..., id)
        D_spatial: Grid spacings [D1, D2, ..., Dd]
        spatial_shape: Shape of U (N1+1, N2+1, ..., Nd+1)
        periodic: Use periodic boundary conditions

    Returns:
        derivs = {
            1D: {(0,): u, (1,): p_x}
            2D: {(0,0): u, (1,0): p_x, (0,1): p_y}
            3D: {(0,0,0): u, (1,0,0): p_x, (0,1,0): p_y, (0,0,1): p_z}
            nD: {(0,...,0): u, (1,0,...,0): p_1, ..., (0,...,0,1): p_n}
        }
    """
    dimension = len(indices)

    # Function value at current point
    u = float(U[indices])

    # Initialize derivatives dictionary with zero multi-index
    derivs = {tuple([0] * dimension): u}

    # Compute gradient components for each dimension
    for dim in range(dimension):
        # Forward and backward neighbor indices
        idx_forward = list(indices)
        idx_backward = list(indices)

        if periodic:
            # Periodic boundaries
            idx_forward[dim] = (indices[dim] + 1) % spatial_shape[dim]
            idx_backward[dim] = (indices[dim] - 1) % spatial_shape[dim]
        else:
            # One-sided differences at boundaries
            if indices[dim] == 0:
                # Left boundary: Forward difference
                idx_forward[dim] = indices[dim] + 1
                idx_backward[dim] = indices[dim]
            elif indices[dim] == spatial_shape[dim] - 1:
                # Right boundary: Backward difference
                idx_forward[dim] = indices[dim]
                idx_backward[dim] = indices[dim] - 1
            else:
                # Interior: Central difference
                idx_forward[dim] = indices[dim] + 1
                idx_backward[dim] = indices[dim] - 1

        # Central difference gradient
        u_forward = float(U[tuple(idx_forward)])
        u_backward = float(U[tuple(idx_backward)])
        p_dim = (u_forward - u_backward) / (2 * D_spatial[dim])

        # Multi-index for this partial derivative
        multi_index = [0] * dimension
        multi_index[dim] = 1
        derivs[tuple(multi_index)] = p_dim

    return derivs


# Example usage:
# 1D: _calculate_gradient_nd(U, (50,), [0.01], (101,))
#     Returns: {(0,): u, (1,): p_x}
#
# 2D: _calculate_gradient_nd(U, (50, 50), [0.01, 0.01], (101, 101))
#     Returns: {(0,0): u, (1,0): p_x, (0,1): p_y}
#
# 3D: _calculate_gradient_nd(U, (25, 25, 25), [0.02, 0.02, 0.02], (51, 51, 51))
#     Returns: {(0,0,0): u, (1,0,0): p_x, (0,1,0): p_y, (0,0,1): p_z}
```

### 2.5 Dimension-Agnostic Laplacian and Sparse Matrix

```python
def _compute_laplacian_nd(U, indices: tuple, D_spatial: list[float],
                          spatial_shape: tuple, periodic: bool = True):
    """
    Compute n-dimensional Laplacian: ∇²u = Σ ∂²u/∂x_i²

    Uses (2n+1)-point stencil:
    - 1D: 3 points (left, center, right)
    - 2D: 5 points (center + 4 neighbors)
    - 3D: 7 points (center + 6 neighbors)
    - nD: (2n+1) points (center + 2n neighbors)

    Args:
        U: N-dimensional array
        indices: Spatial indices (i1, i2, ..., id)
        D_spatial: Grid spacings [D1, D2, ..., Dd]
        spatial_shape: Shape of U
        periodic: Use periodic boundary conditions

    Returns:
        float: Laplacian value ∇²u
    """
    dimension = len(indices)
    laplacian = 0.0

    # Sum second derivatives over all dimensions
    for dim in range(dimension):
        # Forward and backward neighbor indices
        idx_forward = list(indices)
        idx_backward = list(indices)

        if periodic:
            idx_forward[dim] = (indices[dim] + 1) % spatial_shape[dim]
            idx_backward[dim] = (indices[dim] - 1) % spatial_shape[dim]
        else:
            # Handle boundaries (use one-sided if needed)
            idx_forward[dim] = min(indices[dim] + 1, spatial_shape[dim] - 1)
            idx_backward[dim] = max(indices[dim] - 1, 0)

        # Second derivative in dimension `dim`
        u_center = U[indices]
        u_forward = U[tuple(idx_forward)]
        u_backward = U[tuple(idx_backward)]

        second_deriv = (u_forward - 2*u_center + u_backward) / D_spatial[dim]**2
        laplacian += second_deriv

    return laplacian


def _build_jacobian_nd(spatial_shape: tuple, D_spatial: list[float],
                       Dt: float, sigma: float, periodic: bool = True):
    """
    Build n-dimensional Jacobian sparse matrix.

    Matrix structure:
    - 1D: Tridiagonal (3 bands)
    - 2D: Pentadiagonal (5 bands: main, ±1, ±N2)
    - 3D: Heptadiagonal (7 bands: main, ±1, ±N2, ±N2*N3)
    - nD: (2n+1)-diagonal structure

    Matrix size: (N_total × N_total) where N_total = Π(Ni+1)

    Returns:
        scipy.sparse.csr_matrix: Sparse Jacobian matrix
    """
    dimension = len(spatial_shape)
    N_total = np.prod(spatial_shape)  # Total number of spatial points

    row_indices = []
    col_indices = []
    data_values = []

    # Helper: Convert multi-index to flat index
    def flat_index(multi_idx):
        """Convert (i1, i2, ..., id) to flat index."""
        flat = 0
        stride = 1
        for i in range(dimension - 1, -1, -1):
            flat += multi_idx[i] * stride
            stride *= spatial_shape[i]
        return flat

    # Helper: Convert flat index to multi-index
    def unravel_index(flat):
        """Convert flat index to (i1, i2, ..., id)."""
        multi_idx = []
        for i in range(dimension - 1, -1, -1):
            multi_idx.insert(0, flat % spatial_shape[i])
            flat //= spatial_shape[i]
        return tuple(multi_idx)

    # Iterate over all spatial grid points
    for flat_idx in range(N_total):
        indices = unravel_index(flat_idx)

        # Diagonal coefficient (center point)
        val_center = 1.0 / Dt
        for dim in range(dimension):
            val_center += 2 * sigma**2 / D_spatial[dim]**2

        row_indices.append(flat_idx)
        col_indices.append(flat_idx)
        data_values.append(val_center)

        # Off-diagonal coefficients (neighbors in each dimension)
        for dim in range(dimension):
            # Forward neighbor
            idx_forward = list(indices)
            if periodic:
                idx_forward[dim] = (indices[dim] + 1) % spatial_shape[dim]
            else:
                if indices[dim] < spatial_shape[dim] - 1:
                    idx_forward[dim] = indices[dim] + 1
                else:
                    continue  # Skip boundary

            flat_forward = flat_index(tuple(idx_forward))
            row_indices.append(flat_idx)
            col_indices.append(flat_forward)
            data_values.append(-sigma**2 / D_spatial[dim]**2)

            # Backward neighbor
            idx_backward = list(indices)
            if periodic:
                idx_backward[dim] = (indices[dim] - 1) % spatial_shape[dim]
            else:
                if indices[dim] > 0:
                    idx_backward[dim] = indices[dim] - 1
                else:
                    continue  # Skip boundary

            flat_backward = flat_index(tuple(idx_backward))
            row_indices.append(flat_idx)
            col_indices.append(flat_backward)
            data_values.append(-sigma**2 / D_spatial[dim]**2)

    # Build sparse matrix
    import scipy.sparse as sparse
    J = sparse.coo_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(N_total, N_total)
    ).tocsr()

    return J
```

### 2.6 Sparse Matrix Scaling

| Dimension | Stencil | Matrix Structure | Nonzeros per row | Total Nonzeros | Example Grid |
|-----------|---------|------------------|------------------|----------------|--------------|
| **1D** | 3-point | Tridiagonal | 3 | 3N | 100 → 300 |
| **2D** | 5-point | Pentadiagonal | 5 | 5N | 100×100 → 50K |
| **3D** | 7-point | Heptadiagonal | 7 | 7N | 50³ → 875K |
| **4D** | 9-point | 9-diagonal | 9 | 9N | 30⁴ → 7.3M |
| **nD** | (2n+1)-point | (2n+1)-diagonal | 2n+1 | (2n+1)N | ... |

**Memory Scaling**:
- 2D (100×100): ~400 KB sparse matrix (vs 80 MB dense)
- 3D (50³): ~7 MB sparse matrix (vs 1 GB dense)
- Sparse storage is **O(N)**, dense is **O(N²)** for matrix

---

## 3. Gradient Notation (Already Standardized)

**Phase 2/3 gradient notation is already complete** (merged 2025-10-30):

### 1D:
```python
derivs = {
    (0,): u,      # Function value
    (1,): p_x     # ∂u/∂x
}
```

### 2D:
```python
derivs = {
    (0, 0): u,      # Function value
    (1, 0): p_x,    # ∂u/∂x
    (0, 1): p_y     # ∂u/∂y
}
```

### 3D:
```python
derivs = {
    (0, 0, 0): u,      # Function value
    (1, 0, 0): p_x,    # ∂u/∂x
    (0, 1, 0): p_y,    # ∂u/∂y
    (0, 0, 1): p_z     # ∂u/∂z
}
```

### nD:
```python
derivs = {
    (0, ..., 0): u,           # Function value
    (1, 0, ..., 0): p_1,      # ∂u/∂x₁
    (0, 1, 0, ..., 0): p_2,   # ∂u/∂x₂
    ...
    (0, ..., 0, 1): p_n       # ∂u/∂xₙ
}
```

**No changes needed** - gradient notation is dimension-agnostic!

---

## 4. Computational Feasibility and Curse of Dimensionality

### 4.1 Theoretical Background

Grid-based FDM methods suffer from the **curse of dimensionality**:
- **Memory**: O(N^d) spatial grid points
- **Computation**: O(N^d) operations per timestep
- **Sparse matrix**: O(N^d) rows, (2d+1) nonzeros per row

**Why High Dimensions Become Impractical**:

| Dimension | Grid Resolution | Total Points | Memory (float64) | Practical? |
|-----------|----------------|--------------|------------------|------------|
| 1D | 1000 | 1,000 | ~8 KB | ✅ Excellent |
| 2D | 100×100 | 10,000 | ~80 KB | ✅ Excellent |
| 3D | 50×50×50 | 125,000 | ~1 MB | ✅ Good |
| 4D | 30×30×30×30 | 810,000 | ~6 MB | ⚠️ Marginal |
| 5D | 20×20×20×20×20 | 3,200,000 | ~25 MB | ❌ Impractical |
| 6D | 15×15×15×15×15×15 | 11,390,625 | ~87 MB | ❌ Impractical |
| 7D | 12×...×12 (7 times) | 35,831,808 | ~273 MB | ❌ Impractical |

**With Time Dimension** (add factor of Nt+1):
- 5D spatial + time (Nt=50): 3.2M × 51 = 163M points → 1.2 GB per array
- Multiple arrays (U, M, residuals): 3-5 GB total memory
- Newton iterations require Jacobian: (3.2M)² sparse matrix (challenging!)

### 4.2 Automatic Feasibility Checks

The dimension-agnostic problem class includes automatic warnings:

```python
def _check_computational_feasibility(self):
    """Warn if problem is computationally challenging."""

    # High dimension warning (d > 4)
    if self.dimension > MAX_PRACTICAL_DIMENSION:
        warnings.warn(
            f"⚠️  Grid-based FDM impractical for {self.dimension}D\n"
            f"Consider particle/network methods instead."
        )

    # Large grid warning (> 10M spatial points)
    elif self.total_spatial_points > MAX_TOTAL_GRID_POINTS:
        warnings.warn(
            f"⚠️  Very large grid: {self.total_spatial_points:,} points\n"
            f"Memory: {memory_estimate_mb:.1f} MB per array"
        )
```

### 4.3 Programmatic Cost Estimation

Users can query computational cost before solving:

```python
problem = MFGProblemND(spatial_bounds=[...], spatial_discretization=[...])

cost = problem.get_computational_cost_estimate()
# Returns:
# {
#     'dimension': 5,
#     'total_points': 67200000,
#     'memory_mb': 512.7,
#     'is_practical': False,
#     'warning_level': 'high_dimension'
# }

if not cost['is_practical']:
    print(f"WARNING: {cost['warning_level']}")
    # Switch to alternative solver
    solver = ParticleHJBSolver(problem)  # Instead of FDM
else:
    solver = HJBFDMSolver(problem)
```

### 4.4 Alternative Methods for High Dimensions

When d > 4, recommend:

1. **Particle-based Collocation**:
   - Samples N_particles << N^d points
   - Works for arbitrary dimension
   - Example: 1000 particles work for d=10

2. **Network-based MFG**:
   - Reduces spatial dimension to graph structure
   - Works for embedded network problems
   - Example: Traffic networks (high-dim state, low-dim network)

3. **Neural PDE Solvers**:
   - Physics-informed neural networks (PINNs)
   - Curse-of-dimensionality-free (in theory)
   - Active research area

---

## 5. Test Strategy

### 5.1 Dimension-Agnostic Validation Tests

**Test 1: N-D Gaussian Diffusion**
```python
def test_nd_gaussian_evolution(dimension, N_per_dim=50):
    """Test n-D FP solver with analytical Gaussian solution."""
    # Create problem
    spatial_bounds = [(0, 1)] * dimension
    spatial_discretization = [N_per_dim] * dimension

    problem = MFGProblemND(spatial_bounds, spatial_discretization)

    # Initial: n-D Gaussian at center
    center = np.array([0.5] * dimension)
    squared_dist = np.zeros(problem.spatial_shape)
    for dim in range(dimension):
        squared_dist += (problem.grids[dim] - center[dim])**2

    m0 = np.exp(-squared_dist / (2 * 0.1**2))
    m0 /= np.sum(m0) * np.prod(problem.D_spatial)

    # Solve with zero drift
    U = np.zeros((problem.Nt+1, *problem.spatial_shape))
    fp_solver = FPFDMNDSolver(problem)
    m = fp_solver.solve_fp_system(m0, U)

    # Check: Mass conservation
    for k in range(problem.Nt + 1):
        mass = np.sum(m[k]) * np.prod(problem.D_spatial)
        assert np.isclose(mass, 1.0, rtol=0.01)

    # Check: Symmetry (for simple domain)
    # Gaussian should diffuse symmetrically


# Run for multiple dimensions
for d in [1, 2, 3]:
    test_nd_gaussian_evolution(dimension=d)
```

**Test 2: Dimension Reduction (nD reduces to 1D when all but one dimension has 1 point)**
```python
def test_nd_reduces_to_1d():
    """Verify nD solver matches 1D when other dimensions degenerate."""
    # 1D problem
    problem_1d = MFGProblemND(
        spatial_bounds=[(0, 1)],
        spatial_discretization=[100],
        Nt=100
    )

    # "2D" problem with Ny=1 (should behave like 1D)
    problem_pseudo_2d = MFGProblemND(
        spatial_bounds=[(0, 1), (0, 0.01)],  # Very thin in y
        spatial_discretization=[100, 0],      # Ny=1
        Nt=100
    )

    # Solve both
    solver_1d = HJBFDMNDSolver(problem_1d)
    solver_2d = HJBFDMNDSolver(problem_pseudo_2d)

    # Results should match
    U_1d = solver_1d.solve_hjb_system(...)
    U_2d = solver_2d.solve_hjb_system(...)

    # Extract 1D slice from 2D result
    U_2d_slice = U_2d[:, :, 0]  # All time, all x, j=0

    assert np.allclose(U_1d, U_2d_slice, rtol=1e-10)
```

**Test 3: Computational Feasibility Warnings**
```python
def test_high_dimension_warnings():
    """Test that high-dimensional problems trigger warnings."""

    # 3D: No warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem_3d = MFGProblemND(
            spatial_bounds=[(0, 1)] * 3,
            spatial_discretization=[50] * 3
        )
        assert len(w) == 0  # No warnings

    # 5D: HIGH DIMENSION WARNING
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem_5d = MFGProblemND(
            spatial_bounds=[(0, 1)] * 5,
            spatial_discretization=[20] * 5
        )
        assert len(w) == 1
        assert "HIGH DIMENSION WARNING" in str(w[0].message)

    # 5D with suppression: No warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem_5d_suppressed = MFGProblemND(
            spatial_bounds=[(0, 1)] * 5,
            spatial_discretization=[20] * 5,
            suppress_warnings=True
        )
        assert len(w) == 0
```

**Test 4: Multi-Index Utilities**
```python
def test_multi_index_utilities():
    """Test flat_index and unravel_index conversions."""
    spatial_shape = (11, 21, 31)  # 3D grid

    # Test all points
    for i in range(spatial_shape[0]):
        for j in range(spatial_shape[1]):
            for k in range(spatial_shape[2]):
                multi_idx = (i, j, k)

                # Convert to flat
                flat = flat_index(multi_idx, spatial_shape)

                # Convert back
                recovered = unravel_index(flat, spatial_shape)

                assert recovered == multi_idx
```

### 5.2 Production Tests (Integration Phase)

**After merging to `mfg_pde/`**:

1. **Backward compatibility**: All existing 1D tests must pass
2. **2D/3D accuracy**: Compare vs analytical solutions (L2 error < 1e-3)
3. **Performance**: Solver should scale as O(N^d) (verify empirically)
4. **Memory**: Sparse matrices only (no dense O(N²) storage)
5. **High dimension**: Verify warnings trigger appropriately

---

## 6. Implementation Timeline (Direct Unified Approach)

| Phase | Week | Tasks | Deliverables |
|-------|------|-------|--------------|
| **1: Problem Class** | 1 | Extend `MFGProblem` with dual initialization modes | Unified problem class (1D + n-D) |
| **2: Solver Detection** | 2 | Update HJB/FP solvers to detect problem dimension | Solvers auto-adapt to dimension |
| **3: N-D Core Functions** | 3 | Implement n-D gradient/Laplacian computations | Working n-D numerical core |
| **4: Sparse Matrices** | 4 | Implement n-D Jacobian/FP matrix assembly | N-D sparse matrix builders |
| **5: Testing** | 5 | Unit tests, backward compat verification | 791+ existing tests pass |
| **6: Validation** | 6 | N-D analytical tests, convergence studies | 2D/3D validation complete |

**Total**: 6 weeks (1.5 months)

**Key Benefits of Direct Approach**:
- ✅ **No code duplication** (single implementation)
- ✅ **No migration phase** (no temporary research code)
- ✅ **Immediate production quality** (proper testing from start)
- ✅ **Clean user experience** (one problem class, one API)

---

## 7. Migration Path

### For Researchers (Phase 1)

```python
# Before (1D only)
from mfg_pde import MFGProblem
problem = MFGProblem(Nx=100, Nt=100)

# After (n-D research prototype)
import sys
sys.path.insert(0, 'experiments/nd_fdm_solvers')
from problem_nd import MFGProblemND

# 1D (backward compatible)
problem = MFGProblemND(
    spatial_bounds=[(0, 1)],
    spatial_discretization=[100],
    Nt=100
)

# 2D (new capability)
problem = MFGProblemND(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    Nt=50
)

# 3D (new capability)
problem = MFGProblemND(
    spatial_bounds=[(0, 1), (0, 1), (0, 1)],
    spatial_discretization=[30, 30, 30],
    Nt=30
)
```

### For Production Users (Phase 2)

```python
# After production integration (unified API)
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver

# 1D (backward compatible)
problem_1d = MFGProblem(
    spatial_bounds=[(0, 1)],
    spatial_discretization=[100],
    Nt=100
)
hjb_solver = HJBFDMSolver(problem_1d)

# 2D (new capability)
problem_2d = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    Nt=50
)
hjb_solver = HJBFDMSolver(problem_2d)

# 3D (new capability)
problem_3d = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1), (0, 1)],
    spatial_discretization=[30, 30, 30],
    Nt=30
)
hjb_solver = HJBFDMSolver(problem_3d)

# Dimension automatically detected from problem!
```

---

## 8. Next Steps

### Immediate (Week 1)

1. ✅ **Architecture design complete** (this document)
2. ⏳ Create `experiments/nd_fdm_solvers/` directory
3. ⏳ Implement `MFGProblemND` class with computational warnings
4. ⏳ Implement multi-index utilities (`flat_index`, `unravel_index`)
5. ⏳ Port 1D HJB solver to n-D (basic version)

### Short-term (Week 2)

1. ⏳ Implement n-D FP solver
2. ⏳ Add 6 validation tests (Gaussian, dimension reduction, warnings, etc.)
3. ⏳ Create examples: 1D (compat), 2D (crowd), 3D (visualization)

### Medium-term (Weeks 3-6)

1. ⏳ Production integration into `mfg_pde/`
2. ⏳ Comprehensive test suite
3. ⏳ Performance benchmarks (1D-4D)
4. ⏳ Documentation: API docs, computational cost guide

---

## 9. Related Work

### Completed Prerequisites

- ✅ **Bug #15 (QP Sigma Fix)**: Fixed 2025-10-30
- ✅ **Anderson Multi-dimensional**: Fixed 2025-10-30
- ✅ **Gradient Notation Standardization**: Completed 2025-10-30

### Blocked Features (Unblocked by This Work)

1. **2D/3D MFG Problems** (blocked by lack of n-D solvers)
2. **Multi-Agent Path Planning** (needs 2D obstacle avoidance)
3. **Crowd Dynamics** (requires 2D spatial modeling)
4. **Network MFG** (2D embedding of network graphs)

---

## 10. References

- **Architecture Audit**: `docs/architecture/README.md`
- **Current 1D Solvers**:
  - `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py` (lines 1-107)
  - `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py` (lines 77-768)
  - `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py` (lines 1-294)
- **Gradient Notation**: `docs/theory/foundations/NOTATION_STANDARDS.md`
- **Compatibility Layer**: `mfg_pde/compat/gradient_notation.py`
- **Test Files**:
  - `tests/unit/test_hjb_fdm_solver.py` (375 lines)
  - `tests/unit/test_fp_fdm_solver.py` (416 lines)
  - `tests/test_gradient_utils.py` (250 lines) - Gradient notation tests

---

## 11. Computational Cost Reference

### Memory Scaling Table

| Dimension | Grid Resolution | Spatial Points | Memory per Array | With Time (Nt=50) | Total (U+M+aux) |
|-----------|----------------|----------------|------------------|-------------------|-----------------|
| 1D | 1000 | 1,000 | ~8 KB | ~400 KB | ~1.2 MB |
| 2D | 100×100 | 10,000 | ~80 KB | ~4 MB | ~12 MB |
| 3D | 50×50×50 | 125,000 | ~1 MB | ~50 MB | ~150 MB |
| 4D | 30×30×30×30 | 810,000 | ~6 MB | ~310 MB | ~930 MB |
| 5D | 20×20×20×20×20 | 3,200,000 | ~25 MB | ~1.2 GB | ~3.6 GB |
| 6D | 15×15×15×15×15×15 | 11,390,625 | ~87 MB | ~4.3 GB | ~13 GB |

### Computation Time Scaling (Estimated)

Assuming 100 Newton iterations per timestep, 50 timesteps:

| Dimension | Operations per Timestep | Total Operations | Estimated Time (CPU) |
|-----------|-------------------------|------------------|----------------------|
| 1D | 3,000 | 15M | ~0.1 sec |
| 2D | 50,000 | 250M | ~2 sec |
| 3D | 875,000 | 4.4B | ~40 sec |
| 4D | 7.3M | 36B | ~6 min |
| 5D | 29M | 145B | ~24 min |
| 6D | 102M | 511B | ~1.5 hours |

**These are rough estimates.** Actual times depend on hardware, implementation, and problem-specific factors (convergence rate, sparsity patterns, etc.).

---

**Document Status**: Living document (dimension-agnostic design approved)
**Last Updated**: 2025-10-30
**Maintainer**: MFG_PDE Development Team
