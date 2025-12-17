# Deprecation Modernization Guide

**Last Updated**: 2025-12-17
**Current Version**: v0.16.11
**Target**: v1.0.0 (deprecated patterns will be removed)
**Status**: Active migration in progress

---

## Overview

This guide documents deprecated usage patterns in MFG_PDE and provides migration paths to modern APIs. All deprecated patterns will continue to work until v1.0.0 but will emit deprecation warnings.

---

## Deprecation Timeline

| Version | Milestone |
|:--------|:----------|
| v0.16.x | Current - deprecation warnings active |
| **v0.17.0** | **Consolidate deprecations, stricter warnings** |
| v0.18.x | Final warning period |
| v1.0.0 | Remove all deprecated patterns |

---

## v0.16.11 Deprecation Summary (BC Architecture)

### New Deprecations in v0.16.11

#### 8. BC Calculator Renaming (PR #520, Issue #516)

Physics-based naming for boundary condition calculators:

| Old Name | New Name | Semantics |
|:---------|:---------|:----------|
| `NoFluxCalculator` | `ZeroGradientCalculator` | du/dn = 0 (edge extension) |
| `FPNoFluxCalculator` | `ZeroFluxCalculator` | J·n = 0 (mass conservation) |

**Why renamed**: The old names conflated two different physics:
- `ZeroGradientCalculator`: Neumann BC, extends edge value (suitable for HJB)
- `ZeroFluxCalculator`: Robin BC, ensures J·n = 0 (suitable for Fokker-Planck)

These are equivalent only when drift velocity μ = 0.

**Migration**:
```python
# Old (deprecated, will work until v0.19)
from mfg_pde.geometry.boundary import NoFluxCalculator, FPNoFluxCalculator

# New (preferred)
from mfg_pde.geometry.boundary import ZeroGradientCalculator, ZeroFluxCalculator
```

#### 9. Legacy 1D FDM BoundaryConditions (Issue #516)

| Old Name | New Name | Status |
|:---------|:---------|:-------|
| `BoundaryConditions1DFDM` | `BoundaryConditions` | Deprecated |
| `LegacyBoundaryConditions1D` | `BoundaryConditions` | Deprecated |

**Migration**:
```python
# Old (deprecated)
from mfg_pde.geometry.boundary import BoundaryConditions1DFDM

# New (preferred)
from mfg_pde.geometry.boundary import BoundaryConditions, neumann_bc
bc = neumann_bc(dimension=1)
```

---

## v0.17.0 Deprecation Summary

### New Deprecations in v0.17

#### 7. Convergence Module Renaming (PR #457)

| Old Name | New Name | Status |
|:---------|:---------|:-------|
| `StochasticConvergenceMonitor` | `RollingConvergenceMonitor` | Deprecated |
| `AdvancedConvergenceMonitor` | `DistributionConvergenceMonitor` | Deprecated |
| `ParticleMethodDetector` | `SolverTypeDetector` | Deprecated |
| `AdaptiveConvergenceWrapper` | `ConvergenceWrapper` | Deprecated |
| `OscillationDetector` | `_ErrorHistoryTracker` (internal) | Deprecated |
| `create_default_monitor()` | `create_distribution_monitor()` | Deprecated |
| `create_stochastic_monitor()` | `create_rolling_monitor()` | Deprecated |

**Migration**:
```python
# Old (deprecated)
from mfg_pde.utils.numerical.convergence import (
    StochasticConvergenceMonitor,
    AdvancedConvergenceMonitor,
    create_default_monitor,
)

# New (preferred)
from mfg_pde.utils.numerical.convergence import (
    RollingConvergenceMonitor,
    DistributionConvergenceMonitor,
    create_distribution_monitor,
)
```

### Existing Deprecations (Carried Forward)

| Category | Deprecated | Modern | Priority |
|:---------|:-----------|:-------|:---------|
| Problem Construction | `ExampleMFGProblem(Nx=50, ...)` | `MFGProblem(geometry=domain, ...)` | High |
| Legacy Attributes | `problem.xmin`, `problem.Nx`, etc. | `problem.geometry.get_bounds()` | High |
| Scalar Parameters | `Nx=50` (scalar) | `Nx=[50]` (array) | Medium |
| solve_mfg method | `solve_mfg(problem, method='accurate')` | `create_accurate_solver(problem, ...)` | Medium |
| Profiling API | `enable_profiling=True` | `profiling_mode=ProfilingMode.SILENT` | Low |
| Legacy Problems | `ExampleMFGProblem`, etc. | `MFGProblem` + geometry | Low |
| Geometry Aliases | `Domain1D`, `Domain2D`, `Domain3D` | `TensorProductGrid`, `Mesh2D`, `Mesh3D` | Low |
| p_values parameter | `hamiltonian(t, x, m, p_values=...)` | `hamiltonian(t, x, m, derivs=...)` | Medium |
| Lowercase params | `nx=50`, `nt=25` | `Nx=50`, `Nt=25` | Low |

---

## 1. Manual Grid Construction (High Priority)

### Deprecated Pattern

```python
from mfg_pde import ExampleMFGProblem

problem = ExampleMFGProblem(Nx=50, xmin=0.0, xmax=1.0, Nt=25, T=1.0, sigma=0.1)
```

**Issues**:
- Manual grid construction bypasses geometry system
- Scalar notation (`Nx=50`) instead of array notation (`Nx=[50]`)
- Limited to 1D problems
- Cannot leverage unified geometry infrastructure

### Modern Pattern

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Step 1: Create geometry with boundary conditions
domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
# Note: TensorProductGrid replaces SimpleGrid1D/2D/3D (removed in v0.15.3)

# Step 2: Create problem with geometry
problem = MFGProblem(geometry=domain, T=1.0, Nt=25, sigma=0.1)
```

### Benefits
- ✅ Explicit boundary condition specification
- ✅ Works with dimension-agnostic solvers
- ✅ Integrates with boundary handler system (Issue #272)
- ✅ Extensible to 2D/3D via `TensorProductGrid`
- ✅ Supports advanced geometries (AMR, FEM meshes)

### Migration Impact
- **Files affected**: 22 test files in `tests/unit/`
- **Breaking change**: v1.0.0 (manual construction will be restricted)
- **Effort**: Low (mechanical replacement)

---

## 2. Scalar Parameter Notation (Medium Priority)

### Deprecated Pattern

```python
problem = ExampleMFGProblem(Nx=50, xmin=0.0, xmax=1.0)  # Scalars
```

**Deprecation warning**: `mfg_pde/core/mfg_problem.py:113`

### Modern Pattern

```python
problem = MFGProblem(Nx=[50], xmin=[0.0], xmax=[1.0])  # Arrays
```

**Rationale**: Array notation is dimension-agnostic and consistent with nD problems.

### Migration
- Replace `Nx=50` → `Nx=[50]`
- Replace `xmin=0.0` → `xmin=[0.0]`
- Replace `xmax=1.0` → `xmax=[1.0]`

**Note**: If using geometry-first API (pattern #1), this is automatically handled.

---

## 3. Legacy Attribute Access (High Priority) - NEW

### Deprecated Pattern

```python
problem = MFGProblem(Nx=[50], xmin=[0.0], xmax=[1.0], T=1.0, Nt=10)

# Accessing legacy attributes directly
x_min = problem.xmin      # DeprecationWarning
x_max = problem.xmax      # DeprecationWarning
grid_size = problem.Nx    # DeprecationWarning
spacing = problem.dx      # DeprecationWarning
```

**Deprecation**: These attributes now emit `DeprecationWarning` when accessed from external code. They will be removed in v1.0.0.

### Modern Pattern

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Create problem with geometry-first API
domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
problem = MFGProblem(geometry=domain, T=1.0, Nt=10)

# Access spatial information via geometry
bounds = problem.geometry.get_bounds()        # (min_array, max_array)
grid = problem.geometry.get_spatial_grid()    # Spatial grid points
num_points = problem.geometry.num_spatial_points  # Total grid points
dim = problem.geometry.dimension              # Spatial dimension

# Use helper properties
if problem.is_cartesian:
    # Cartesian grid-specific logic
    pass
elif problem.is_network:
    # Network-specific logic
    pass
```

### What Changed (Issue #435, PRs #436-#443)

1. **`problem.geometry` is never None** - Always set after initialization
2. **Legacy attributes are computed properties** - Derive values from `self.geometry`
3. **Legacy attributes hidden from autocomplete** - `__dir__()` excludes them
4. **Write protection** - Setting `problem.xmin = value` emits warning (but works for backward compatibility)
5. **Type safety** - `geometry` is typed as `GeometryProtocol`
6. **Helper properties** - `is_cartesian`, `is_network`, `is_implicit` for type dispatch

### Migration

| Old | New |
|:----|:----|
| `problem.xmin` | `problem.geometry.get_bounds()[0][0]` |
| `problem.xmax` | `problem.geometry.get_bounds()[1][0]` |
| `problem.Nx` | `problem.geometry.num_spatial_points` |
| `problem.dx` | Compute from bounds and num_points |
| `problem.xSpace` | `problem.geometry.get_spatial_grid()` |

---

## 4. StrategySelector Parameters (Low Priority)

### Deprecated Pattern

```python
from mfg_pde.backends.strategies.strategy_selector import StrategySelector

selector = StrategySelector(enable_profiling=True, verbose=False)
```

**Deprecation location**: `mfg_pde/backends/strategies/strategy_selector.py`

### Modern Pattern

```python
from mfg_pde.backends.strategies.strategy_selector import StrategySelector, ProfilingMode

selector = StrategySelector(profiling_mode=ProfilingMode.SILENT)
```

**Profiling modes**:
- `ProfilingMode.DISABLED`: No profiling
- `ProfilingMode.SILENT`: Profile but don't print
- `ProfilingMode.VERBOSE`: Profile and print stats

### Migration Impact
- **Files affected**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py` (already fixed)
- **Breaking change**: v1.0.0
- **Effort**: Minimal (enum-based API)

---

## 5. solve_mfg() Method Parameter (Medium Priority)

### Deprecated Pattern

```python
from mfg_pde import solve_mfg

# Ambiguous "method" parameter
result = solve_mfg(problem, method='accurate')
```

**Issues**:
- Ambiguous name: "method" could mean algorithm, coupling scheme, or preset
- Hidden complexity: Single parameter controls multiple settings
- Redundant: Users can already set `max_iterations` and `tolerance` directly
- Opaque: Doesn't show what's actually being configured

### Modern Pattern

```python
from mfg_pde.factory import create_accurate_solver

# Explicit factory function + direct parameter control
solver = create_accurate_solver(
    problem,
    max_iterations=500,  # Clear and explicit
    tolerance=1e-6
)
result = solver.solve()
```

**What the presets actually do**:
- `method='fast'`: Uses `create_standard_solver()`, max_iterations=100
- `method='accurate'`: Uses `create_accurate_solver()`, max_iterations=500
- `method='research'`: Uses `create_research_solver()`, max_iterations=1000, auto-creates HJB/FP solvers

### Benefits of Modern Pattern

- ✅ **Explicit**: Clear which factory function is used
- ✅ **Transparent**: All parameters visible at call site
- ✅ **Flexible**: Direct control over all solver settings
- ✅ **Discoverable**: IDE autocomplete shows available factories
- ✅ **No magic**: User controls solver selection, not hidden preset

### Migration Strategy

**Phase 1** (v0.x.x): Keep `method` parameter with deprecation warning

**Phase 2** (v1.0.0): Remove `method` parameter, force explicit factory usage

**For now**: Document modern pattern, update examples to use factory functions directly

---

## 6. Legacy Problem Classes (Low Priority)

### Deprecated Classes

All classes in `mfg_pde/compat/legacy_problems.py`:
- `ExampleMFGProblem` → Use `MFGProblem` with geometry
- `CrowdDynamicsProblem` → Use `MFGProblem` with factory
- `TrafficFlowProblem` → Use `MFGProblem` with factory
- `PortfolioOptimizationProblem` → Use `MFGProblem` with factory
- `CustomMFGProblem` → Inherit from `MFGProblem` directly

### Modern Approach

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Example: Crowd dynamics problem
domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[101])
# Note: TensorProductGrid replaces SimpleGrid1D/2D/3D (removed in v0.15.3)

problem = MFGProblem(
    geometry=domain,
    T=1.0,
    Nt=50,
    sigma=0.1,
    coupling_coefficient=1.0
)
```

**For custom problems**: Inherit from `MFGProblem` and override methods:

```python
from mfg_pde import MFGProblem

class MyCustomProblem(MFGProblem):
    def potential(self, t, x, m):
        # Custom potential implementation
        return 0.5 * x**2

    def terminal_condition(self, x, m):
        # Custom terminal condition
        return 0.0
```

---

## Migration Strategy

### Phase 1: Documentation (✅ Current)
- Document modern patterns (this guide)
- Create migration examples in `examples/tutorials/`
- Update README with modern usage

### Phase 2: Controlled Test Conversion (Next)
- Convert 2-3 test files as reference examples
- Verify no regressions
- Document conversion pattern

### Phase 3: Systematic Migration (v0.x.x series)
- Batch convert remaining test files
- Update all `examples/` to use modern API
- Issue deprecation warnings for all legacy patterns

### Phase 4: Enforcement (v1.0.0)
- Remove deprecated constructors
- Restrict manual grid construction
- Legacy problem classes raise errors with migration hints

---

## Quick Reference

| Category | Deprecated | Modern | Priority | Since |
|:---------|:-----------|:-------|:---------|:------|
| Problem | `ExampleMFGProblem(Nx=50, ...)` | `MFGProblem(geometry=domain, ...)` | High | v0.15 |
| Problem | `problem.xmin`, `problem.Nx`, etc. | `problem.geometry.get_bounds()` | High | v0.16 |
| Problem | `Nx=50` (scalar) | `Nx=[50]` (array) | Medium | v0.15 |
| Problem | `hamiltonian(..., p_values=...)` | `hamiltonian(..., derivs=...)` | Medium | v0.16 |
| Problem | `nx=50`, `nt=25` (lowercase) | `Nx=50`, `Nt=25` | Low | v0.16 |
| Factory | `solve_mfg(problem, method='accurate')` | `create_accurate_solver(problem, ...)` | Medium | v0.15 |
| Strategy | `enable_profiling=True` | `profiling_mode=ProfilingMode.SILENT` | Low | v0.15 |
| Legacy | `ExampleMFGProblem`, etc. | `MFGProblem` + geometry | Low | v0.15 |
| Geometry | `Domain1D`, `Domain2D`, `Domain3D` | `TensorProductGrid`, `Mesh2D`, `Mesh3D` | Low | v0.15.3 |
| Convergence | `StochasticConvergenceMonitor` | `RollingConvergenceMonitor` | Low | v0.17 |
| Convergence | `AdvancedConvergenceMonitor` | `DistributionConvergenceMonitor` | Low | v0.17 |
| Convergence | `ParticleMethodDetector` | `SolverTypeDetector` | Low | v0.17 |
| Convergence | `AdaptiveConvergenceWrapper` | `ConvergenceWrapper` | Low | v0.17 |
| Convergence | `create_default_monitor()` | `create_distribution_monitor()` | Low | v0.17 |
| Convergence | `create_stochastic_monitor()` | `create_rolling_monitor()` | Low | v0.17 |
| BC | `NoFluxCalculator` | `ZeroGradientCalculator` | Low | v0.16.11 |
| BC | `FPNoFluxCalculator` | `ZeroFluxCalculator` | Low | v0.16.11 |
| BC | `BoundaryConditions1DFDM` | `BoundaryConditions` | Low | v0.16.11 |
| BC | `LegacyBoundaryConditions1D` | `BoundaryConditions` | Low | v0.16.11 |

---

## Testing Modernized Code

After migrating to modern API:

```python
# Verify geometry integration
assert hasattr(problem, 'geometry')
assert problem.geometry.dimension == 1

# Verify boundary handler works
bc_handler = problem.geometry.get_boundary_handler(bc_type='periodic')
assert bc_handler is not None

# Run existing tests to ensure backward compatibility
pytest tests/unit/your_test_file.py
```

---

## Getting Help

- **Migration questions**: Open GitHub discussion
- **Bugs in modern API**: Report via GitHub issues with label `area: geometry`
- **Examples**: See `examples/tutorials/` for modern usage patterns

---

## Tracking Progress

**Current Status** (as of 2025-12-17):
- ✅ Modern API fully implemented
- ✅ Dimension-agnostic boundary handler (PR #305, #306)
- ✅ Geometry-first unification complete (Issue #435, PRs #436-#443)
- ✅ Legacy attributes converted to computed properties (PR #443)
- ✅ Convergence module reorganized with renamed classes (PR #457)
- ✅ BC Topology/Calculator composition (PR #520, Issue #516)
- ⏳ Test file migration: 0/22 files converted
- ⏳ Example migration: 0/10 files converted
- ❌ v1.0.0 enforcement: Not yet implemented

**Deprecation Counts by Version**:
| Version | New Deprecations | Total Active |
|:--------|:-----------------|:-------------|
| v0.15.x | 6 patterns | 6 |
| v0.16.x | 4 patterns | 10 |
| **v0.16.11** | **4 patterns (BC)** | **14** |
| v0.17.x | 6 patterns | 20 |

**Deprecation Timeline** (quick deprecation strategy - remove in 2-3 minor versions):
- v0.16.11 deprecations → Remove in v0.19
- v0.17.x deprecations → Remove in v0.20 or v1.0.0

**Next Milestone**: v0.17.0 release with consolidated deprecation warnings

---

## See Also

- `docs/development/GEOMETRY_PARAMETER_MIGRATION.md` - Detailed geometry API docs
- `docs/development/CONSISTENCY_GUIDE.md` - Code style and naming conventions
- `mfg_pde/geometry/README.md` - Geometry system overview
