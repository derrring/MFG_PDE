# Architecture Refactoring Implementation Plan

**Date**: 2025-11-02
**Based On**: Issue #200, Architecture Audit (2025-10-30)
**Timeline**: 3-9 months
**Status**: Planning Phase

---

## Executive Summary

Systematic refactoring to address 48 documented architectural issues discovered during 3 weeks of intensive research usage. Total projected cost without fixes: **780 hours/year** (19.5 work-weeks).

**Phase 1 Status**: ✅ COMPLETED (all 3 items)
**Phase 2.1 Status**: ✅ COMPLETED (already existed - discovered 2025-11-02)
**Current Phase**: Phase 2.2 Planning (Missing Utilities - 4 weeks estimated)

---

## Phase 1: Immediate Fixes ✅ COMPLETED (2 weeks)

### 1.1 Bug #15: QP Sigma API ✅
**Status**: COMPLETED (PR #214, merged 2025-11-02)

**Changes**:
- Added `_get_sigma_value()` helper method
- Handles: numeric sigma, callable sigma(x), legacy nu
- Fixed 4 code locations in `hjb_gfdm.py`
- Added 5 unit tests

**Impact**: Removed TypeError blocker for particle methods with QP constraints

### 1.2 Anderson Multi-Dimensional ✅
**Status**: COMPLETED (already in codebase)

**Implementation** (`anderson_acceleration.py:108-135`):
```python
# Store original shape on first call
if self._original_shape is None:
    self._original_shape = x_current.shape

# Flatten arrays for vector operations
x_flat = x_current.ravel()
f_flat = f_current.ravel()
# ... compute ...
# Reshape back
return x_next_flat.reshape(self._original_shape)
```

**Impact**: Transparently handles 1D, 2D, 3D arrays without manual flatten/reshape

### 1.3 Gradient Notation Standardization ✅
**Status**: COMPLETED (already standardized)

**Current Format** (consistent across codebase):
```python
derivs[(0,)]      # u (function value)
derivs[(1,)]      # ∂u/∂x (1D first derivative)
derivs[(2,)]      # ∂²u/∂x² (1D second derivative)
derivs[(1, 0)]    # ∂u/∂x (2D)
derivs[(0, 1)]    # ∂u/∂y (2D)
derivs[(2, 0)]    # ∂²u/∂x² (2D)
```

**Verification**:
- All HJB solvers use tuple format
- No string-based keys ('dpx', 'grad_u_x') found in codebase
- Prevents Bug #13 type mismatches

---

## Phase 2: High-Priority Items (3 months)

### 2.1 Implement 2D/3D FDM Solvers ✅ ALREADY COMPLETE

**Priority**: CRITICAL (was)
**Status**: ✅ COMPLETED (discovered 2025-11-02, existed before audit)
**Estimated Effort**: 4-6 weeks (was)
**Actual Status**: Already fully implemented and tested
**Blocker For**: Research papers requiring FDM baseline comparisons (UNBLOCKED)

#### Discovery Summary (2025-11-02)

**Architecture Audit Claim** (2025-10-30):
> "FDM solvers are hard-coded 1D-only with 23+ locations assuming 1D indexing"

**Actual Implementation Status**: ✅ **Fully implemented and working**

Upon starting implementation for Issue #215, discovered that 2D/3D FDM support is already complete:

**Evidence**:
- ✅ **HJB FDM**: Full nD support with dimension detection (hjb_fdm.py:152-325)
- ✅ **FP FDM**: Full nD support with dimension detection (fp_fdm.py:55-134)
- ✅ **Tests**: 26 passing tests validating 2D/3D functionality
  - test_hjb_fdm_2d_validation.py: 9/9 tests (convergence, physical properties)
  - test_multidim_workflow.py: 14/14 tests (2D/3D operators, visualization)
  - test_coupled_hjb_fp_2d.py: 3/3 tests (coupled system)
- ✅ **Examples**: 9+ working 2D examples, 1 working 3D example
- ✅ **Documentation**: HJB_SOLVER_SELECTION_GUIDE.md covers 2D/3D (545 lines)

**Working Code Example**:
```python
# This already works perfectly!
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver

problem_2d = GridBasedMFGProblem(
    domain_bounds=(-1, 1, -1, 1),  # 2D domain
    grid_resolution=50,
    time_domain=(1.0, 20),
    diffusion_coeff=0.1
)

solver = HJBFDMSolver(problem_2d)  # ✅ Works perfectly!
solver.dimension  # Returns: 2
result = solver.solve_hjb_system(...)  # ✅ Full 2D solve
```

**Reconciliation**: The architecture audit (Oct 30) missed this existing implementation. The maturity of tests (convergence studies, physical property validation) indicates this is not a recent addition but has been in the codebase for some time.

**Impact**:
- CRITICAL blocker is actually UNBLOCKED
- 4-6 weeks of work avoided
- Research papers can proceed with FDM baseline comparisons

#### Original Problem Statement (Obsolete)

**Claimed Limitation** (incorrect):
- `HJBFDMSolver` and `FPFDMSolver` only accept `MFGProblem` (1D)
- 23+ locations in code assume 1D indexing
- `GridBasedMFGProblem` (2D/3D) exists but incompatible with FDM solvers

#### Actual Implementation Found (Option A Already Implemented)

The codebase already implements **Option A** (unified solver with dimension dispatch):

**HJB FDM Solver** (`hjb_fdm.py`):
```python
# Dimension detection (lines 152-160)
def _detect_dimension(self, problem) -> int:
    if hasattr(problem, "geometry") and hasattr(problem.geometry, "grid"):
        return getattr(problem.geometry.grid, "dimension", 1)
    if hasattr(problem, "dimension"):
        return problem.dimension
    # Fallback to inspecting grid attributes

# Automatic routing (lines 173-186)
def solve_hjb_system(self, M, U_final, U_prev):
    if self.dimension == 1:
        return base_hjb.solve_hjb_system_backward(...)
    else:
        return self._solve_hjb_nd(...)  # Full nD implementation

# nD gradient computation (lines 267-300)
def _compute_gradients_nd(self, U):
    # Central differences for all dimensions
    # Handles arbitrary dimension d
```

**FP FDM Solver** (`fp_fdm.py`):
```python
# Dimension detection and routing (lines 55-91)
def solve_fp_system(self, m_initial, U_drift, show_progress=True):
    if self.dimension == 1:
        return self._solve_fp_1d(...)
    else:
        return _solve_fp_nd_full_system(...)  # Full nD implementation
```

#### Verification Status

**All planned success criteria already met**:
- ✅ `HJBFDMSolver` accepts `GridBasedMFGProblem` (2D/3D) - **WORKING**
- ✅ All existing 1D tests pass (backward compatibility) - **VERIFIED**
- ✅ New 2D/3D tests with O(h²) convergence - **26 TESTS PASSING**
- ✅ Examples demonstrate usage - **9+ EXAMPLES WORKING**
- ✅ Performance validated - **CONVERGENCE STUDIES PASS**

**Existing Test Coverage**:
- `tests/integration/test_hjb_fdm_2d_validation.py` - 9 tests (convergence, physical properties)
- `tests/integration/test_multidim_workflow.py` - 14 tests (2D/3D operators)
- `tests/integration/test_coupled_hjb_fp_2d.py` - 3 tests (coupled system)
- `tests/unit/test_geometry/test_boundary_conditions_{2d,3d}.py` - Boundary conditions
- `tests/unit/test_geometry/test_domain_{2d,3d}.py` - Domain setup

**Existing Examples**:
- `examples/basic/2d_crowd_motion_fdm.py` - Basic 2D FDM usage ✅
- `examples/advanced/traffic_flow_2d_demo.py` - Traffic flow in 2D ✅
- `examples/advanced/mfg_2d_geometry_example.py` - Complex 2D geometry ✅
- Plus 6 more 2D examples and 1 3D example ✅

**Documentation**:
- `docs/user_guide/HJB_SOLVER_SELECTION_GUIDE.md` - 545 lines covering 2D/3D FDM ✅

#### Optional Enhancements (Low Priority)

Since implementation is complete, these are optional polish items:

1. **Maze example** - Create `examples/advanced/fdm_2d_maze.py` (2-3 hours)
2. **Additional 3D examples** - More demonstrations (4-6 hours)
3. **Dedicated guide** - `docs/user_guide/FDM_MULTIDIMENSIONAL_GUIDE.md` (optional, coverage exists)

---

### 2.2 Add Missing Utilities (4 weeks)

**Priority**: HIGH
**Estimated Effort**: 4 weeks
**Impact**: Saves ~1,655 lines of duplicate code per research project

#### 2.2.1 Particle Interpolation Utilities (1 week)

**Current**: Every project writes ~220 lines of custom interpolation code

**Proposed** (`mfg_pde/utils/numerical/particle_interpolation.py`):
```python
def interpolate_grid_to_particles(
    grid_values: np.ndarray,
    grid_bounds: tuple,
    particle_positions: np.ndarray,
    method: str = "linear"
) -> np.ndarray:
    """
    Interpolate values from regular grid to particle positions.

    Args:
        grid_values: (Nt, Nx) or (Nt, Nx, Ny) grid values
        grid_bounds: ((xmin, xmax),) or ((xmin, xmax), (ymin, ymax))
        particle_positions: (N_particles, d) positions
        method: "linear", "cubic", or "rbf"

    Returns:
        Interpolated values at particle positions
    """

def interpolate_particles_to_grid(
    particle_values: np.ndarray,
    particle_positions: np.ndarray,
    grid_shape: tuple,
    grid_bounds: tuple,
    method: str = "rbf"
) -> np.ndarray:
    """
    Interpolate particle values to regular grid.

    Args:
        particle_values: (N_particles,) values at particles
        particle_positions: (N_particles, d) positions
        grid_shape: (Nx,) or (Nx, Ny) grid dimensions
        grid_bounds: ((xmin, xmax),) or ((xmin, xmax), (ymin, ymax))
        method: "rbf", "nearest", or "kde"

    Returns:
        Grid values (Nx,) or (Nx, Ny)
    """
```

#### 2.2.2 Signed Distance Function (SDF) Helpers (1 week)

**Current**: Each project implements obstacle distance calculations

**Proposed** (`mfg_pde/utils/geometry/sdf.py`):
```python
def compute_sdf_from_obstacles(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    obstacles: list[Obstacle],
    return_gradient: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute signed distance function for obstacles.

    Args:
        grid_x, grid_y: Meshgrid coordinates
        obstacles: List of Obstacle objects (rectangles, circles, polygons)
        return_gradient: If True, also return ∇φ for boundary handling

    Returns:
        φ: Signed distance (negative inside obstacles)
        ∇φ: (optional) Gradient of distance function
    """
```

#### 2.2.3 QP Result Caching (1 week)

**Current**: QP solves same problem multiple times

**Proposed** (`mfg_pde/utils/numerical/qp_cache.py`):
```python
class QPCache:
    """
    Cache QP solutions for identical neighborhood structures.

    For collocation methods solving 3000+ QPs per iteration,
    many QPs have identical structure (same A matrix, W weights).
    Caching can provide 2-5× speedup.
    """
    def __init__(self, max_cache_size: int = 1000):
        self._cache: dict[int, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    def get_cached_solution(self, key: int) -> np.ndarray | None:
        """Look up cached QP solution by hash key."""

    def cache_solution(self, key: int, solution: np.ndarray):
        """Store QP solution with LRU eviction."""
```

#### 2.2.4 Convergence Monitor Utility (1 week)

**Current**: Each script writes ~60 lines of monitoring code

**Proposed** (`mfg_pde/utils/logging/convergence_monitor.py`):
```python
class ConvergenceMonitor:
    """
    Track and visualize convergence of iterative solvers.

    Features:
    - Automatic plotting
    - Stagnation detection
    - Convergence rate estimation
    - Export to dataframe for analysis
    """
    def update(self, iteration: int, residual: float, extra_metrics: dict | None = None):
        """Record iteration data."""

    def plot(self, show: bool = True, save_path: str | None = None):
        """Generate convergence plot."""

    def estimate_convergence_rate(self) -> float:
        """Estimate r in ||x_k - x*|| ≈ r^k."""
```

---

### 2.3 Quick Wins (1 week)

**Priority**: MEDIUM
**Estimated Effort**: 1 week
**Impact**: Reduces boilerplate by ~90 lines per experiment

#### 2.3.1 Standardize Solver Return Format (2 days)

**Current**: Different solvers return different formats
```python
U = hjb_solver.solve(...)               # Some return just U
U, info = hjb_solver.solve(...)         # Others return tuple
result = hjb_solver.solve(...)          # Some return dict
```

**Proposed**: Consistent dataclass return
```python
@dataclass
class SolverResult:
    solution: np.ndarray         # Primary solution (U or M)
    info: SolverInfo             # Convergence info
    iterations: int
    residual: float
    solve_time: float
    extras: dict[str, Any]       # Solver-specific extras

# All solvers return SolverResult
result = hjb_solver.solve(...)
U = result.solution              # Easy access
print(f"Converged in {result.iterations} iterations")
```

#### 2.3.2 Solver Return Format Migration (1 day)

Update all solvers to return `SolverResult`:
- HJB solvers: `hjb_fdm.py`, `hjb_gfdm.py`, `hjb_semi_lagrangian.py`
- FP solvers: `fp_fdm.py`, `fp_particle.py`
- MFG solvers: `fixed_point_iterator.py`

**Backward compatibility**: Add `legacy_mode` parameter (default: False)

#### 2.3.3 High-Level solve_mfg() Function (2 days)

**Proposed** (`mfg_pde/solvers/solve_mfg.py`):
```python
def solve_mfg(
    problem: MFGProblem | GridBasedMFGProblem,
    method: str = "auto",
    hjb_method: str | None = None,
    fp_method: str | None = None,
    resolution: int | None = None,
    backend: str = "auto",
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    High-level MFG solver with automatic method selection.

    Args:
        problem: MFG problem instance
        method: "auto", "fdm", "gfdm", "semi-lagrangian", "particle"
        hjb_method: Override HJB solver (if not auto)
        fp_method: Override FP solver (if not auto)
        resolution: Grid resolution (auto if None)
        backend: "numpy", "jax", "torch", or "auto"
        verbose: Print progress

    Returns:
        U: Value function (Nt, Nx) or (Nt, Nx, Ny)
        M: Density (Nt, Nx) or (Nt, Nx, Ny)
        info: Convergence and timing information

    Example:
        U, M, info = solve_mfg(problem, method="auto", resolution=50)
    """
```

**Benefits**:
- Single entry point for users
- Automatic solver selection based on problem dimension
- Consistent return format
- Reduces setup from ~30 lines to ~1 line

---

## Phase 3: Long-Term Refactoring (6-9 months)

### 3.1 Unified Problem Class (8-10 weeks)

**Priority**: CRITICAL (long-term)
**Complexity**: Very High
**Impact**: Eliminates 1,080 lines of custom problem code per project

**Current State**:
- 5 different problem classes
- Only 4/25 solver combinations work natively
- Constant confusion and adapter code

**Proposed**:
```python
class MFGProblem:
    """
    Unified MFG problem class supporting all dimensions and solver types.

    Automatically detects:
    - Dimension (1D, 2D, 3D, network, variational)
    - Domain type (regular grid, particle cloud, graph, manifold)
    - Appropriate solver methods
    """
    def __init__(
        self,
        dimension: int | str,
        domain: tuple | list | NetworkGraph,
        time_domain: tuple[float, int],
        diffusion: float | Callable,
        hamiltonian: Callable,
        initial_density: Callable | np.ndarray,
        terminal_cost: Callable | np.ndarray,
        obstacles: list | None = None,
        **kwargs
    ):
        # Unified initialization
```

**Migration Path**: 6-phase gradual transition with deprecation warnings

### 3.2 Configuration Simplification (2-3 weeks)

**Current**: 3 competing config systems
**Proposed**: Single YAML-based configuration with schema validation

### 3.3 Backend Integration (2 weeks)

**Current**: Backends not accessible through `create_fast_solver()`
**Proposed**: Wire JAX/PyTorch backends through factory functions

---

## Testing Strategy

### 1. Solver Combination Matrix Tests
Test all HJB×FP×dimension combinations with clear errors for unsupported cases

### 2. Gradient Format Consistency Tests
Verify all solvers use tuple format `derivs[(α,β)]`

### 3. API Contract Tests
Verify problem classes provide expected attributes (prevent Bug #15 recurrence)

### 4. Real Problem Integration Tests
- Maze navigation (2D + obstacles)
- Crowd dynamics (anisotropic)
- Traffic flow (network)

### 5. Performance Regression Tests
- QP performance (<10ms for 100 variables)
- Memory usage (3D grids)
- Convergence rates

---

## Timeline & Milestones

### 2025-11: Phase 2.1 Start
- **Week 1-2**: HJB 2D/3D implementation
- **Week 3-4**: FP 2D/3D implementation

### 2025-12: Phase 2.1 Completion
- **Week 5-6**: Testing, validation, examples
- **Milestone**: 2D/3D FDM solvers working

### 2026-01: Phase 2.2-2.3
- **Week 1-4**: Missing utilities implementation
- **Week 5**: Quick wins
- **Milestone**: Phase 2 complete

### 2026-02-08: Phase 3 Planning
- Architecture design review
- Migration path finalization
- Community input period

### 2026-03-09: Phase 3 Execution
- Unified problem class implementation
- 8-10 weeks of systematic refactoring

---

## Success Metrics

### Phase 2 Success Criteria
1. ✅ 2D/3D FDM solvers functional and tested
2. ✅ Utilities reduce duplicate code by 1,655 lines
3. ✅ Quick wins reduce boilerplate by 90 lines per experiment
4. ✅ All existing tests pass (backward compatibility)

### Phase 3 Success Criteria
1. ✅ Single unified `MFGProblem` class
2. ✅ All 25 HJB×FP combinations clearly supported or documented as unsupported
3. ✅ Integration overhead < 2× (down from 7.6×)
4. ✅ Zero custom problem classes needed for standard use cases

### Annual Impact
- **Before**: 780 hours/year lost to workarounds
- **After Phase 2**: ~400 hours/year saved (50% reduction)
- **After Phase 3**: ~650 hours/year saved (83% reduction)

---

## Next Steps

1. **Create Issue #215**: "Implement 2D/3D FDM Solvers" with detailed specification
2. **Start Phase 2.1**: Begin HJB 2D/3D implementation
3. **Weekly Progress Updates**: Document progress in this file
4. **Community Engagement**: Share plan on GitHub Discussions for feedback

---

**Status**: ✅ Planning Complete, Ready to Start Phase 2.1
**Last Updated**: 2025-11-02
**Next Milestone**: 2D/3D FDM HJB solver (2 weeks)
