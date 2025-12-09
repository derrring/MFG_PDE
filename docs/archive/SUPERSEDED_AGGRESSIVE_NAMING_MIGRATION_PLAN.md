# Aggressive Naming Convention Migration Plan

**Created**: 2025-11-10
**Status**: Planning Phase
**Goal**: Complete migration to modern protocol-compliant naming conventions

---

## Executive Summary

Aggressive refactoring to replace ALL legacy naming conventions with modern MFGProblemProtocol-compliant names throughout the entire codebase.

**Rationale**:
- Eliminate technical debt accumulated from 1D-centric origins
- Full protocol compliance enables dimension-agnostic solvers
- Cleaner, more maintainable codebase
- Foundation for v1.0 release

---

## Scope Analysis

### Deprecated Names → Standard Names

| Old Name | New Name | Occurrences | Files Affected |
|:---------|:---------|:------------|:---------------|
| `Dt` | `dt` | 52 | 20 |
| `tSpace` | `time_grid` | 37 | 15 |
| `xSpace` | `spatial_grid` | 82 | 20 |
| `Dx` | `grid_spacing[0]` | 94 | 20 |
| `Nx` (scalar) | `spatial_discretization[0]` | 65+ | 20+ |
| `xmin`, `xmax` (scalars) | `spatial_bounds[0]` | 65+ | 20+ |

**Total Estimated Changes**: ~400+ lines across ~50+ files

---

## Migration Strategy

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Update `MFGProblem` Class
**File**: `mfg_pde/core/mfg_problem.py`

**Changes**:
```python
# OLD (deprecated)
self.Dt = (self.T - self.t0) / self.Nt
self.tSpace = np.linspace(self.t0, self.T, self.Nt + 1)
self.xSpace = np.linspace(xmin, xmax, Nx + 1)
self.Dx = (xmax - xmin) / Nx

# NEW (standard)
self.dt = (self.T - self.t0) / self.Nt
self.time_grid = np.linspace(self.t0, self.T, self.Nt + 1)
self.spatial_grid = self._create_spatial_grid()  # Unified for all modes
self.grid_spacing = self._compute_grid_spacing()  # Always returns list
self.grid_shape = self._compute_grid_shape()  # Always returns tuple
```

**Backward Compatibility** (TEMPORARY):
```python
# Add deprecation warnings for old names
@property
def Dt(self) -> float:
    warnings.warn("Dt is deprecated. Use dt instead.", DeprecationWarning, stacklevel=2)
    return self.dt

@property
def tSpace(self) -> np.ndarray:
    warnings.warn("tSpace is deprecated. Use time_grid instead.", DeprecationWarning, stacklevel=2)
    return self.time_grid

# Similar for xSpace, Dx, etc.
```

#### 1.2 Update `BaseProblem` Class
**File**: `mfg_pde/core/base_problem.py`

Propagate changes to base class.

#### 1.3 Update `MFGProblemProtocol`
**File**: `mfg_pde/types/problem_protocols.py`

Ensure protocol specifies only NEW names (no legacy attributes).

---

### Phase 2: Solver Infrastructure (Week 1)

#### 2.1 HJB Solvers
**Files**: `mfg_pde/alg/numerical/hjb_solvers/*.py`

**Changes**:
```python
# OLD
dt = problem.Dt
x_grid = problem.xSpace
dx = problem.Dx

# NEW
dt = problem.dt
x_grid = problem.spatial_grid
dx = problem.grid_spacing[0]  # For 1D
```

**Affected files** (10 occurrences in `base_hjb.py` alone):
- `base_hjb.py`
- `hjb_semi_lagrangian.py`
- `hjb_weno.py`
- Others...

#### 2.2 FP Solvers
**Files**: `mfg_pde/alg/numerical/fp_solvers/*.py`

Similar pattern to HJB solvers.

**High impact file**: `fp_particle.py` (11 occurrences of `xSpace`)

#### 2.3 Coupling Methods
**Files**: `mfg_pde/alg/numerical/coupling/*.py`

Update `fixed_point_iterator.py` (2 `Dt` occurrences).

#### 2.4 Optimization Solvers
**Files**: `mfg_pde/alg/optimization/variational_solvers/*.py`

Update `base_variational.py`.

---

### Phase 3: Examples (Week 2)

#### 3.1 Basic Examples
**Directory**: `examples/basic/`

**Strategy**:
1. Update all problem instantiations to use new names
2. Update any direct attribute access
3. Test each example runs successfully

**High-impact files**:
- `acceleration_comparison.py` (2 `Nx` occurrences)
- `santa_fe_bar_demo.py` (2 `Nx` occurrences)
- `policy_iteration_lq_demo.py` (3 `Nx` occurrences)

#### 3.2 Advanced Examples
**Directory**: `examples/advanced/`

**High-impact files**:
- `weno_family_comparison_demo.py`
- `anisotropic_crowd_dynamics_2d/two_door_production_solver.py`
- `lagrangian_constrained_optimization.py`

---

### Phase 4: Tests (Week 2)

#### 4.1 Integration Tests
**Directory**: `tests/integration/`

**High-impact files** (mass conservation tests):
- `test_mass_conservation_1d.py` (8 `xSpace`, 3 `Nx`)
- `test_mass_conservation_1d_simple.py`
- `test_mass_conservation_fast.py`
- `test_stochastic_mass_conservation.py`
- `test_fdm_solvers_mfg_complete.py` (12 bound occurrences)

#### 4.2 Unit Tests
**Directory**: `tests/unit/`

**High-impact files**:
- `test_convergence.py` (8 `Dt` occurrences)
- `test_core/test_mfg_problem.py` (6 `tSpace`, 3 `Dt` occurrences)
- `test_fp_fdm_solver.py` (3 `Dt` occurrences)

#### 4.3 New Protocol Compliance Test
**Create**: `tests/unit/test_core/test_protocol_compliance.py`

Verify all new attributes exist and work correctly.

---

### Phase 5: Utilities & Visualization (Week 2)

#### 5.1 Utilities
**Files**:
- `mfg_pde/utils/experiment_manager.py` (7 `xSpace`, 7 `Dx`, 2 `tSpace`)
- `mfg_pde/utils/numerical/convergence.py` (5 `Dx`)
- `mfg_pde/utils/numerical/hjb_policy_iteration.py`
- `mfg_pde/utils/aux_func.py` (2 `Dx`)

#### 5.2 Visualization
**Files**:
- `mfg_pde/visualization/legacy_plotting.py` (2 `tSpace`, 2 `xSpace`)

---

### Phase 6: Benchmarks & Investigations (Week 3)

#### 6.1 Benchmarks
**Directory**: `benchmarks/`

Update all benchmark scripts:
- `solver_comparisons/` (multiple files)
- `parallel_computation_benchmark.py`
- `particle_gpu_speedup_analysis.py`
- `test_solver_performance.py`

#### 6.2 Investigations
**Directory**: `investigations/`

**Decision**: Archive or update?
- `bug_investigations/` - Update if still relevant
- `property_based/` - Update
- `mathematical/` - Update `test_mass_conservation.py` (13 `Dx` occurrences)

---

## Implementation Details

### Approach: Systematic Replacement

**Step 1**: Create helper script for automated replacement:
```python
# scripts/migration/rename_attributes.py
import re
from pathlib import Path

REPLACEMENTS = {
    r'\bDt\b': 'dt',
    r'\btSpace\b': 'time_grid',
    r'\bxSpace\b': 'spatial_grid',
    # Add special handling for Dx, Nx, xmin, xmax (context-dependent)
}

def migrate_file(filepath):
    """Apply naming migrations to a single file."""
    # Implementation...
```

**Step 2**: Manual review of each change (don't blindly replace).

**Step 3**: Test after each logical group of files.

---

## Special Cases

### 1. Array Notation Migration (Issue #243)

**Combined with this effort**:
```python
# OLD (scalar for 1D)
problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, T=1.0, Nt=50, sigma=0.1)

# NEW (array notation + new names)
problem = MFGProblem(
    spatial_discretization=[100],
    spatial_bounds=[(0.0, 1.0)],
    T=1.0, Nt=50, sigma=0.1
)

# Access
Nx = problem.spatial_discretization[0]  # Not problem.Nx
dx = problem.grid_spacing[0]             # Not problem.Dx
x_grid = problem.spatial_grid            # Not problem.xSpace
dt = problem.dt                          # Not problem.Dt
t_grid = problem.time_grid               # Not problem.tSpace
```

### 2. Multi-Dimensional Problems

**2D Example**:
```python
# OLD
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    ...
)
# Access: problem.spatial_shape[0], problem.spatial_shape[1] (inconsistent!)

# NEW
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    ...
)
# Access:
Nx, Ny = problem.grid_shape              # Tuple (50, 50)
dx, dy = problem.grid_spacing            # List [dx, dy]
grid = problem.spatial_grid              # Unified interface
```

### 3. Geometry-Based Problems

**Geometry mode** (new in v0.11.0):
```python
from mfg_pde.geometry import SimpleGrid2D

grid = SimpleGrid2D(bounds=(0, 1, 0, 1), resolution=(50, 50))
problem = MFGProblem(geometry=grid, T=1.0, Nt=50, sigma=0.1)

# Access (delegated to geometry):
grid_shape = problem.grid_shape          # (50, 50) from geometry
grid_spacing = problem.grid_spacing      # [dx, dy] from geometry
spatial_grid = problem.spatial_grid      # From geometry.get_spatial_grid()
```

---

## Testing Strategy

### 1. Protocol Compliance Test
**Create**: `tests/unit/test_core/test_protocol_compliance.py`

```python
def test_protocol_compliance_1d():
    """Test 1D problem has all protocol attributes."""
    problem = MFGProblem(
        spatial_discretization=[100],
        spatial_bounds=[(0.0, 1.0)],
        T=1.0, Nt=50, sigma=0.1
    )

    # Test new attributes exist
    assert hasattr(problem, 'dt')
    assert hasattr(problem, 'time_grid')
    assert hasattr(problem, 'spatial_grid')
    assert hasattr(problem, 'grid_shape')
    assert hasattr(problem, 'grid_spacing')

    # Test types
    assert isinstance(problem.dt, float)
    assert isinstance(problem.time_grid, np.ndarray)
    assert isinstance(problem.grid_shape, tuple)
    assert isinstance(problem.grid_spacing, list)

    # Test values
    assert problem.grid_shape == (100,)
    assert len(problem.grid_spacing) == 1
    assert problem.time_grid.shape == (51,)

def test_protocol_compliance_2d():
    """Test 2D problem has all protocol attributes."""
    # Similar for 2D...

def test_protocol_compliance_geometry():
    """Test geometry-based problem has all protocol attributes."""
    # Similar for geometry mode...
```

### 2. Backward Compatibility Test (Temporary)
**Test deprecated names still work with warnings**:

```python
def test_deprecated_names_still_work():
    """Verify backward compatibility during migration."""
    problem = MFGProblem(...)

    # Old names should work but emit warnings
    with pytest.warns(DeprecationWarning, match="Dt is deprecated"):
        dt_old = problem.Dt

    with pytest.warns(DeprecationWarning, match="tSpace is deprecated"):
        tspace_old = problem.tSpace

    # Should match new names
    assert dt_old == problem.dt
    assert np.array_equal(tspace_old, problem.time_grid)
```

### 3. Full Integration Tests
**Run existing test suite**:
```bash
pytest tests/integration/ -xvs
pytest tests/unit/ -xvs
```

---

## Timeline & Milestones

### Week 1: Core Infrastructure
- **Day 1-2**: Update `MFGProblem` and `BaseProblem` classes
- **Day 3-4**: Update all solvers (HJB, FP, coupling, optimization)
- **Day 5**: Create protocol compliance test, run unit tests

### Week 2: Examples & Tests
- **Day 1-2**: Migrate all examples (basic + advanced)
- **Day 3-4**: Migrate all tests (integration + unit)
- **Day 5**: Update utilities and visualization

### Week 3: Final Cleanup
- **Day 1**: Update benchmarks
- **Day 2**: Update/archive investigations
- **Day 3**: Documentation update (docstrings, guides)
- **Day 4**: Full test suite run, fix any issues
- **Day 5**: Code review, final polish

### Week 4: Deprecation Removal (Optional for v1.0)
- Remove backward compatibility layer
- Remove deprecation warnings
- Final testing
- Release as v1.0.0 or v0.12.0

---

## Breaking Changes

### If Removing Backward Compatibility (v1.0)

**Breaking changes**:
1. `problem.Dt` → `problem.dt` (lowercase)
2. `problem.tSpace` → `problem.time_grid`
3. `problem.xSpace` → `problem.spatial_grid`
4. `problem.Dx` → `problem.grid_spacing[0]`
5. `problem.Nx` (scalar) → `problem.spatial_discretization[0]`

**Migration guide** for users:
```python
# Old code (v0.11.x and earlier)
problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, ...)
dt = problem.Dt
x = problem.xSpace
dx = problem.Dx

# New code (v1.0+)
problem = MFGProblem(spatial_discretization=[100], spatial_bounds=[(0.0, 1.0)], ...)
dt = problem.dt
x = problem.spatial_grid
dx = problem.grid_spacing[0]
```

---

## Risk Mitigation

### 1. Staged Rollout
- **Phase 1**: Add new names alongside old (both work)
- **Phase 2**: Add deprecation warnings for old names
- **Phase 3**: Remove old names (breaking change for v1.0)

### 2. Comprehensive Testing
- Protocol compliance tests
- Backward compatibility tests (during transition)
- Full integration test suite
- Manual testing of examples

### 3. Documentation
- Update all docstrings immediately
- Create migration guide for users
- Update README and tutorials
- Changelog with clear breaking changes section

---

## Decision Points

### A. Versioning Strategy

**Option 1: Release as v0.12.0** (Recommended)
- Keep backward compatibility layer
- Deprecation warnings guide users
- Allow gradual user migration
- Remove compatibility in v1.0.0

**Option 2: Release as v1.0.0** (Aggressive)
- Remove old names completely
- Breaking changes acceptable for v1.0
- Forces immediate user migration
- Cleaner codebase from start

**Recommendation**: Option 1 - Two-phase approach is safer.

### B. Array Notation Integration

**Should we combine with Issue #243 Phase 2?**

**YES** - Benefits:
- Single migration effort instead of two
- Users update code once, not twice
- Consistent modern API from start

**Combined changes**:
```python
# Current (v0.11.x)
MFGProblem(Nx=100, xmin=0, xmax=1, T=1, Nt=50)
problem.Dt, problem.xSpace, problem.Dx

# Target (v0.12.0 or v1.0.0)
MFGProblem(spatial_discretization=[100], spatial_bounds=[(0,1)], T=1, Nt=50)
problem.dt, problem.spatial_grid, problem.grid_spacing
```

---

## Success Criteria

✅ All 400+ occurrences of deprecated names updated
✅ Protocol compliance test passes for 1D/2D/3D/geometry modes
✅ Full test suite passes (172 unit tests + integration tests)
✅ All examples run successfully
✅ Documentation updated
✅ Migration guide created for users
✅ Zero breaking changes during v0.12.0 (if using two-phase approach)

---

## Next Steps

1. **Get approval for aggressive migration plan**
2. **Choose versioning strategy** (v0.12.0 vs v1.0.0)
3. **Decide on Issue #243 integration** (array notation)
4. **Create feature branch**: `chore/aggressive-naming-standardization`
5. **Begin Week 1 implementation**

---

**Status**: ⏸️ **AWAITING APPROVAL**

**Estimated Total Effort**: 2-3 weeks full-time
**Risk Level**: MEDIUM (comprehensive testing mitigates)
**Impact**: HIGH (foundation for all future development)
