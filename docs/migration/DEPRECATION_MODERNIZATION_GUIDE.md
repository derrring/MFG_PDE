# Deprecation Modernization Guide

**Last Updated**: 2025-11-13
**Target**: v1.0.0 (deprecated patterns will be restricted)
**Status**: Active migration in progress

---

## Overview

This guide documents deprecated usage patterns in MFG_PDE and provides migration paths to modern APIs. All deprecated patterns will continue to work until v1.0.0 but will emit deprecation warnings.

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
from mfg_pde.geometry import SimpleGrid1D

# Step 1: Create geometry with boundary conditions
domain = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions='periodic')
domain.create_grid(num_points=51)  # Note: Nx+1 points

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

## 3. StrategySelector Parameters (Low Priority)

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

## 4. Legacy Problem Classes (Low Priority)

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
from mfg_pde.geometry import SimpleGrid1D

# Example: Crowd dynamics problem
domain = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions='periodic')
domain.create_grid(num_points=101)

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

| Deprecated | Modern | Priority |
|:-----------|:-------|:---------|
| `ExampleMFGProblem(Nx=50, ...)` | `MFGProblem(geometry=domain, ...)` | High |
| `Nx=50` (scalar) | `Nx=[50]` (array) | Medium |
| `enable_profiling=True` | `profiling_mode=ProfilingMode.SILENT` | Low |
| Legacy problem classes | `MFGProblem` + geometry | Low |

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

**Current Status** (as of 2025-11-13):
- ✅ Modern API fully implemented
- ✅ Dimension-agnostic boundary handler (PR #305, #306)
- ⏳ Test file migration: 0/22 files converted
- ⏳ Example migration: 0/10 files converted
- ❌ v1.0.0 enforcement: Not yet implemented

**Next Milestone**: Convert 2 test files to modern API as reference examples

---

## See Also

- `docs/development/GEOMETRY_PARAMETER_MIGRATION.md` - Detailed geometry API docs
- `docs/development/CONSISTENCY_GUIDE.md` - Code style and naming conventions
- `mfg_pde/geometry/README.md` - Geometry system overview
