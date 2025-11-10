# Capitalization Migration Plan: Dt → dt, Dx → dx

**Created**: 2025-11-10
**Status**: Ready for Implementation
**Goal**: Fix capitalization to match official naming conventions

---

## Executive Summary

**Scope**: Change `Dt` → `dt` and `Dx` → `dx` throughout entire codebase

**Rationale**: Official `NAMING_CONVENTIONS.md` specifies lowercase:
- Line 24: `dt: Time step size Δt`
- Line 262: `dx` (lowercase in table)

**Impact**: ~150 occurrences across ~40 files

**Strategy**: Two-phase with deprecation warnings (v0.12.0 → v1.0.0)

---

## Migration Scope

### Files by Priority

**Priority 1: Core (Week 1)**
- `mfg_pde/core/mfg_problem.py` - Primary attribute definitions
- `mfg_pde/core/base_problem.py` - Base class
- `mfg_pde/types/problem_protocols.py` - Protocol definitions

**Priority 2: Solvers (Week 1)**
- `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py` (10 occurrences)
- `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py`
- `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`
- `mfg_pde/alg/numerical/fp_solvers/fp_particle.py` (11 xSpace occurrences - NO CHANGE)
- `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py` (2 Dt)
- `mfg_pde/alg/optimization/variational_solvers/base_variational.py`

**Priority 3: Utilities (Week 2)**
- `mfg_pde/utils/experiment_manager.py` (7 Dx, 2 tSpace - tSpace NO CHANGE)
- `mfg_pde/utils/numerical/convergence.py` (5 Dx)
- `mfg_pde/utils/numerical/hjb_policy_iteration.py`
- `mfg_pde/utils/aux_func.py` (2 Dx)
- `mfg_pde/visualization/legacy_plotting.py` (2 tSpace - NO CHANGE)

**Priority 4: Tests (Week 2)**
- `tests/unit/test_convergence.py` (8 Dt)
- `tests/unit/test_core/test_mfg_problem.py` (3 Dt, 6 tSpace - tSpace NO CHANGE)
- `tests/unit/test_fp_fdm_solver.py` (3 Dt)
- `tests/integration/test_mass_conservation_*.py` (multiple files)

**Priority 5: Examples (Week 2)**
- `examples/basic/*.py`
- `examples/advanced/*.py`

**Priority 6: Benchmarks (Week 3)**
- `benchmarks/solver_comparisons/*.py`
- `benchmarks/particle_gpu_speedup_analysis.py`

---

## Changes NOT Being Made

**Keep as-is** (already correct per conventions):
- ✅ `xSpace` - Universal spatial grid (all geometries)
- ✅ `tSpace` - Temporal grid
- ✅ `Nx` - Grid discretization (migrate to array in Issue #243, separate)

**Why**: These are already correct according to official naming conventions.

---

## Implementation Strategy

### Phase 1: Core Infrastructure (Days 1-3)

#### Step 1: Update `mfg_pde/core/mfg_problem.py`

**Current**:
```python
class MFGProblem:
    def __init__(self, ...):
        self.Dt = (self.T - self.t0) / self.Nt
        self.Dx = (self.xmax - self.xmin) / self.Nx  # 1D mode only
```

**New (v0.12.0 - with backward compatibility)**:
```python
class MFGProblem:
    def __init__(self, ...):
        # NEW: Lowercase is primary
        self.dt = (self.T - self.t0) / self.Nt
        if self.dimension == 1 and hasattr(self, 'Nx'):
            self.dx = (self.xmax - self.xmin) / self.Nx

    # Deprecated properties for backward compatibility
    @property
    def Dt(self) -> float:
        """Deprecated: Use dt (lowercase) instead."""
        import warnings
        warnings.warn(
            "Dt is deprecated. Use dt (lowercase) instead. "
            "Backward compatibility will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.dt

    @property
    def Dx(self) -> float:
        """Deprecated: Use dx (lowercase) instead."""
        import warnings
        warnings.warn(
            "Dx is deprecated. Use dx (lowercase) instead. "
            "Backward compatibility will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.dx
```

#### Step 2: Update `mfg_pde/types/problem_protocols.py`

**Current**:
```python
@runtime_checkable
class GridProblem(Protocol):
    Dt: float
    Dx: float
```

**New (v0.12.0)**:
```python
@runtime_checkable
class GridProblem(Protocol):
    """Protocol for grid-based problems."""

    # NEW standard (lowercase)
    dt: float
    dx: float  # 1D mode only

    # Deprecated (will be removed in v1.0.0)
    # Note: These are kept as properties for backward compatibility
    @property
    def Dt(self) -> float:
        """Deprecated: Use dt instead."""
        return self.dt

    @property
    def Dx(self) -> float:
        """Deprecated: Use dx instead."""
        return self.dx
```

**Note**: Protocol properties are tricky - may need to keep both attributes in v0.12.0.

#### Step 3: Update All Internal Code

Systematic replacement in all files:
```python
# OLD
dt = problem.Dt
dx = problem.Dx

# NEW
dt = problem.dt
dx = problem.dx
```

---

### Phase 2: Systematic File Migration (Days 4-10)

**Script-assisted replacement** (with manual review):

```python
# scripts/migration/fix_capitalization.py
import re
from pathlib import Path

def migrate_file(filepath: Path) -> None:
    """Migrate Dt → dt and Dx → dx in a single file."""
    content = filepath.read_text()

    # Replace attribute access
    content = re.sub(r'\bproblem\.Dt\b', 'problem.dt', content)
    content = re.sub(r'\bself\.Dt\b', 'self.dt', content)
    content = re.sub(r'\bproblem\.Dx\b', 'problem.dx', content)
    content = re.sub(r'\bself\.Dx\b', 'self.dx', content)

    # Replace in calculations
    content = re.sub(r'\bDt\s*=\s*', 'dt = ', content)
    content = re.sub(r'\bDx\s*=\s*', 'dx = ', content)

    # Don't replace in docstrings or comments (manual review needed)

    filepath.write_text(content)
    print(f"✓ Migrated: {filepath}")

# Usage
for pyfile in Path("mfg_pde").rglob("*.py"):
    migrate_file(pyfile)
```

**Process**:
1. Run script on each priority group
2. Manual review each file
3. Run tests after each group
4. Fix any issues
5. Move to next priority group

---

### Phase 3: Testing & Documentation (Days 11-15)

#### Testing Strategy

**Unit tests**:
```bash
pytest tests/unit/test_core/test_mfg_problem.py -xvs
pytest tests/unit/test_convergence.py -xvs
pytest tests/unit/test_fp_fdm_solver.py -xvs
```

**Integration tests**:
```bash
pytest tests/integration/ -xvs
```

**Examples** (smoke tests):
```bash
for example in examples/basic/*.py; do
    echo "Testing $example"
    python "$example" || exit 1
done
```

#### Documentation Updates

**Files to update**:
1. All docstrings mentioning `Dt` or `Dx`
2. `docs/NAMING_CONVENTIONS.md` - Verify consistency
3. `CHANGELOG.md` - Document deprecation
4. Migration guide for users

---

## Version Strategy

### v0.12.0 (Transition Release)

**Both work with deprecation warnings**:
```python
# New way (recommended)
dt = problem.dt
dx = problem.dx

# Old way (deprecated but works)
dt = problem.Dt  # ⚠️ DeprecationWarning
dx = problem.Dx  # ⚠️ DeprecationWarning
```

**Changelog entry**:
```markdown
### Deprecated
- `Dt` attribute: Use lowercase `dt` instead. Backward compatibility maintained via deprecated property.
- `Dx` attribute: Use lowercase `dx` instead. Backward compatibility maintained via deprecated property.

### Changed
- Primary time step attribute changed from `Dt` to `dt` (follows official naming conventions)
- Primary spacing attribute changed from `Dx` to `dx` (1D mode, follows official naming conventions)
```

### v1.0.0 (Breaking Change Release)

**Remove deprecated properties**:
```python
class MFGProblem:
    def __init__(self, ...):
        self.dt = (self.T - self.t0) / self.Nt
        # Dt property removed completely
```

**Changelog entry**:
```markdown
### Breaking Changes
- **Removed `Dt` property**: Use `dt` (lowercase) instead.
- **Removed `Dx` property**: Use `dx` (lowercase) instead.

### Migration Guide
Replace all instances:
- `problem.Dt` → `problem.dt`
- `problem.Dx` → `problem.dx`
```

---

## Timeline

### Week 1: Core + Solvers
- **Day 1**: Update core classes (`mfg_problem.py`, `base_problem.py`)
- **Day 2**: Update protocols (`problem_protocols.py`)
- **Day 3**: Update HJB solvers (5 files)
- **Day 4**: Update FP solvers + coupling (3 files)
- **Day 5**: Run unit tests, fix issues

### Week 2: Tests + Examples + Utils
- **Day 6**: Update utilities (4 files)
- **Day 7**: Update unit tests (10+ files)
- **Day 8**: Update integration tests (10+ files)
- **Day 9**: Update examples basic (10+ files)
- **Day 10**: Update examples advanced (5+ files)

### Week 3: Benchmarks + Documentation + Final Testing
- **Day 11**: Update benchmarks (5+ files)
- **Day 12**: Update all docstrings
- **Day 13**: Update user documentation + migration guide
- **Day 14**: Full test suite run
- **Day 15**: Code review, final polish

---

## Migration Checklist

### Core Changes
- [ ] `mfg_pde/core/mfg_problem.py` - Add `dt`, `dx` attributes + deprecated `Dt`, `Dx` properties
- [ ] `mfg_pde/core/base_problem.py` - Update base class
- [ ] `mfg_pde/types/problem_protocols.py` - Update protocols

### Solvers (20 files)
- [ ] `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py`
- [ ] `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py`
- [ ] `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`
- [ ] `mfg_pde/alg/numerical/fp_solvers/*`
- [ ] `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py`
- [ ] `mfg_pde/alg/optimization/variational_solvers/base_variational.py`

### Utilities (5 files)
- [ ] `mfg_pde/utils/experiment_manager.py`
- [ ] `mfg_pde/utils/numerical/convergence.py`
- [ ] `mfg_pde/utils/numerical/hjb_policy_iteration.py`
- [ ] `mfg_pde/utils/aux_func.py`

### Tests (20+ files)
- [ ] `tests/unit/test_convergence.py`
- [ ] `tests/unit/test_core/test_mfg_problem.py`
- [ ] `tests/unit/test_fp_fdm_solver.py`
- [ ] `tests/integration/test_mass_conservation_*.py` (multiple)

### Examples (15+ files)
- [ ] `examples/basic/*.py`
- [ ] `examples/advanced/*.py`

### Benchmarks (5+ files)
- [ ] `benchmarks/solver_comparisons/*.py`
- [ ] `benchmarks/particle_gpu_speedup_analysis.py`

### Documentation
- [ ] Update all docstrings referencing `Dt` or `Dx`
- [ ] Verify `docs/NAMING_CONVENTIONS.md` consistency
- [ ] Create migration guide for users
- [ ] Update `CHANGELOG.md`

### Testing
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All examples run successfully
- [ ] No deprecation warnings in internal code
- [ ] Deprecation warnings work correctly for external use

---

## Success Criteria

✅ All ~150 internal usages migrated to lowercase `dt`, `dx`
✅ Deprecated properties work with warnings
✅ Full test suite passes (172 unit tests + integration)
✅ All examples run without errors
✅ Documentation updated
✅ Migration guide created for users
✅ Ready for v0.12.0 release

---

## Next Steps

1. **Get approval** for 2-3 week timeline
2. **Create branch**: `chore/lowercase-dt-dx-capitalization`
3. **Begin Week 1**: Core + Solvers
4. **Daily commits**: Systematic progress tracking
5. **Final PR**: Review and merge for v0.12.0

---

**Status**: ⏸️ **READY TO START**
**Estimated Effort**: 2-3 weeks
**Risk Level**: LOW (systematic replacement, comprehensive testing)
**Breaking Changes**: None in v0.12.0 (deprecation only), breaking in v1.0.0
