# Final Naming Migration Summary

**Created**: 2025-11-10
**Status**: Ready for Implementation
**Based on**: Official `docs/NAMING_CONVENTIONS.md` + actual protocol definitions

---

## Key Finding: Capitalization Inconsistency

### Current Situation

**Official Naming Conventions** (`NAMING_CONVENTIONS.md`):
```python
dt = T / Nt          # ✅ Lowercase (line 24)
dx = [...]           # ✅ Lowercase list (line 262)
xSpace = np.linspace(...)  # ✅ Correct
tSpace = np.linspace(...)  # ✅ Correct
```

**Protocol Definitions** (`problem_protocols.py`):
```python
class GridProblem(Protocol):
    Dt: float        # ⚠️ Capital D (backward compat)
    Dx: float        # ⚠️ Capital D (backward compat)
    xSpace: NDArray  # ✅ Correct
    tSpace: NDArray  # ✅ Correct
```

**Conclusion**:
- ✅ `xSpace` and `tSpace` are CORRECT in both - **keep them**
- ⚠️ `Dt` vs `dt` - Protocols use capital for backward compat, conventions say lowercase
- ⚠️ `Dx` vs `dx` - Same issue

---

## What Needs to Change

### ✅ Keep (Already Correct)

| Name | Status | Rationale |
|:-----|:-------|:----------|
| `xSpace` | ✅ Keep | Official standard, used in protocols, clear meaning |
| `tSpace` | ✅ Keep | Official standard, used in protocols, clear meaning |
| `Nx` | ✅ Keep (but migrate to array) | Official standard as **list**, not scalar |

### 🔄 Migrate (Capitalization)

| Old Name | New Name | Scope | Priority |
|:---------|:---------|:------|:---------|
| `Dt` | `dt` | ~52 occurrences, 20 files | HIGH |
| `Dx` | `dx` | ~94 occurrences, 20+ files | HIGH |

**Rationale**: Official conventions use lowercase `dt` and `dx`

### 🆕 Add (Missing Properties)

| Property | Type | Purpose |
|:---------|:-----|:--------|
| `grid_shape` | `tuple[int, ...]` | Unified shape access: `(N₀, N₁, ...)` |
| `grid_spacing` | `list[float]` | Unified spacing access: `[dx₀, dx₁, ...]` |

### 📦 Array Notation (Issue #243)

| Old | New | Status |
|:----|:----|:-------|
| `Nx=100` (scalar) | `Nx=[100]` or `spatial_discretization=[100]` | Planned v0.12.0 |
| `xmin=0.0` (scalar) | `xmin=[0.0]` or `spatial_bounds=[(0.0, 1.0)]` | Planned v0.12.0 |

---

## Migration Plan: 3-Phase Approach

### Phase 1: Lowercase Migration (`Dt` → `dt`, `Dx` → `dx`) ⭐ **IMMEDIATE**

**Goal**: Align with official naming conventions

**Scope**: ~150 occurrences across ~40 files
- Core: `mfg_pde/core/*.py`
- Solvers: `mfg_pde/alg/**/*.py`
- Utils: `mfg_pde/utils/*.py`
- Tests: `tests/**/*.py`
- Examples: `examples/**/*.py`

**Strategy**:
1. Update core `MFGProblem` class:
   ```python
   # NEW: Use lowercase
   self.dt = (self.T - self.t0) / self.Nt
   # OLD: Keep as deprecated property
   @property
   def Dt(self) -> float:
       warnings.warn("Dt is deprecated. Use dt (lowercase).", DeprecationWarning)
       return self.dt
   ```

2. Update protocols to match:
   ```python
   class GridProblem(Protocol):
       dt: float        # NEW standard
       dx: float        # NEW standard (1D mode)
       # Deprecated aliases for v0.12.x
       @property
       def Dt(self) -> float: return self.dt
       @property
       def Dx(self) -> float: return self.dx
   ```

3. Systematic file-by-file migration:
   - Replace `problem.Dt` → `problem.dt`
   - Replace `self.Dt` → `self.dt`
   - Similar for `Dx` → `dx`

**Effort**: 1-2 weeks
**Risk**: MEDIUM (many files, but straightforward search-replace)

---

### Phase 2: Add New Properties ⭐ **IMMEDIATE** (Can do in parallel)

**Goal**: Add missing dimension-agnostic properties

**Implementation**:
```python
# mfg_pde/core/mfg_problem.py

@property
def grid_shape(self) -> tuple[int, ...]:
    """
    Grid shape (N₀, N₁, ..., N_{d-1}) for all dimensions.

    Returns tuple of grid sizes per dimension.

    Examples:
        1D: (100,)
        2D: (100, 80)
        3D: (100, 80, 60)
    """
    if self.dimension == 1 and hasattr(self, 'Nx') and isinstance(self.Nx, int):
        return (self.Nx,)
    elif hasattr(self, 'spatial_discretization'):
        return tuple(self.spatial_discretization)
    elif hasattr(self, 'geometry'):
        return self.geometry.get_grid_shape()
    else:
        raise AttributeError("Cannot determine grid_shape")

@property
def grid_spacing(self) -> list[float]:
    """
    Grid spacing [dx₀, dx₁, ..., dx_{d-1}] for each dimension.

    Returns list for dimension-agnostic indexing: grid_spacing[i]

    Examples:
        1D: [0.01]
        2D: [0.01, 0.0125]
        3D: [0.01, 0.0125, 0.0167]
    """
    if self.dimension == 1 and hasattr(self, 'dx') and isinstance(self.dx, float):
        return [self.dx]
    elif hasattr(self, 'Nx') and hasattr(self, 'xmin') and hasattr(self, 'xmax'):
        # Compute from bounds
        if isinstance(self.Nx, list):
            return [(xmax - xmin) / N for (xmin, xmax), N in
                    zip(zip(self.xmin, self.xmax), self.Nx)]
        else:  # Legacy 1D scalar
            return [(self.xmax - self.xmin) / self.Nx]
    elif hasattr(self, 'geometry'):
        return self.geometry.get_grid_spacing()
    else:
        raise AttributeError("Cannot determine grid_spacing")
```

**Testing**:
```python
# tests/unit/test_core/test_protocol_compliance.py

def test_grid_properties_1d():
    """Test grid properties for 1D problem."""
    problem = MFGProblem(Nx=[100], xmin=[0.0], xmax=[1.0], T=1.0, Nt=50)

    assert problem.grid_shape == (100,)
    assert problem.grid_spacing == [0.01]
    assert len(problem.grid_spacing) == 1

def test_grid_properties_2d():
    """Test grid properties for 2D problem."""
    problem = MFGProblem(
        spatial_discretization=[50, 50],
        spatial_bounds=[(0, 1), (0, 1)],
        T=1.0, Nt=50
    )

    assert problem.grid_shape == (50, 50)
    assert len(problem.grid_spacing) == 2
    assert problem.grid_spacing[0] == pytest.approx(0.02)
    assert problem.grid_spacing[1] == pytest.approx(0.02)
```

**Effort**: 4-6 hours
**Risk**: LOW (purely additive, no breaking changes)

---

### Phase 3: Array Notation Migration (Issue #243) ⏸️ **DEFER to v0.12.0**

Already planned. Keep separate from this effort to avoid scope creep.

---

## Implementation Timeline

### Week 1: Properties + Planning
- **Day 1**: Add `grid_shape` and `grid_spacing` properties (Phase 2)
- **Day 2**: Write protocol compliance tests
- **Day 3**: Create `Dt` → `dt` migration script
- **Day 4**: Test migration script on subset of files
- **Day 5**: Plan file-by-file migration order

### Week 2-3: Systematic Migration
- **Core first**: `mfg_pde/core/*.py` (highest priority)
- **Solvers**: `mfg_pde/alg/**/*.py`
- **Utilities**: `mfg_pde/utils/*.py`
- **Tests**: `tests/**/*.py`
- **Examples**: `examples/**/*.py`
- **Benchmarks**: `benchmarks/**/*.py`

### Week 3: Testing & Documentation
- Full test suite runs
- Update all docstrings
- Update user documentation
- Create migration guide

---

## Backward Compatibility Strategy

### Option A: Two-Phase with Deprecation (v0.12.0 + v1.0.0) ⭐ **RECOMMENDED**

**v0.12.0** (Transition):
```python
# New lowercase is primary
self.dt = (self.T - self.t0) / self.Nt

# Old uppercase is deprecated property
@property
def Dt(self) -> float:
    warnings.warn("Dt is deprecated. Use dt (lowercase).", DeprecationWarning)
    return self.dt
```

**v1.0.0** (Breaking change):
```python
# Only lowercase exists
self.dt = (self.T - self.t0) / self.Nt
# Dt property removed completely
```

**Benefits**:
- Users get 1-2 release cycles to migrate
- Clear deprecation path
- Safe for production code

### Option B: Immediate Breaking Change (v1.0.0 only)

Break immediately, no transition period.

**Benefits**: Clean codebase faster
**Risks**: Breaks all user code immediately

---

## Summary of Changes

### What Changes (Lowercase Migration)

| Component | Old → New | Count |
|:----------|:----------|:------|
| Time step | `Dt` → `dt` | ~52 occurrences |
| Spacing 1D | `Dx` → `dx` | ~94 occurrences |
| **Total** | | **~150 occurrences, ~40 files** |

### What Stays the Same

| Component | Name | Rationale |
|:----------|:-----|:----------|
| Spatial grid | `xSpace` | ✅ Official standard |
| Temporal grid | `tSpace` | ✅ Official standard |
| Grid points | `Nx` | ✅ Official (but migrate to array) |
| Bounds | `xmin`, `xmax` | ✅ Official (but migrate to array) |

### What Gets Added

| Component | Type | Purpose |
|:----------|:-----|:--------|
| `grid_shape` | `tuple[int, ...]` | Dimension-agnostic shape access |
| `grid_spacing` | `list[float]` | Dimension-agnostic spacing access |

---

## Next Steps: Your Decision

**A. Aggressive Approach** (Recommended if targeting v1.0)
- Migrate `Dt` → `dt` and `Dx` → `dx` immediately
- Add `grid_shape` and `grid_spacing` properties
- Remove backward compatibility in v1.0.0
- **Timeline**: 2-3 weeks
- **Version**: v1.0.0 (breaking changes acceptable)

**B. Conservative Approach** (Recommended if staying pre-v1.0)
- Add `grid_shape` and `grid_spacing` properties (Phase 2 only)
- Keep `Dt` and `Dx` as-is for now
- Defer lowercase migration to later
- **Timeline**: 1 week
- **Version**: v0.12.0 (no breaking changes)

**C. Hybrid Approach** (Balanced)
- Add new properties immediately (Phase 2)
- Migrate `Dt` → `dt` with deprecation warnings (Phase 1)
- Remove deprecated names in v1.0.0
- **Timeline**: 2-3 weeks
- **Version**: v0.12.0 (transition) → v1.0.0 (breaking)

---

## My Recommendation

**Approach C (Hybrid)** for v0.12.0:
1. ✅ Add `grid_shape` and `grid_spacing` properties (no breaking changes)
2. ✅ Migrate `Dt` → `dt`, `Dx` → `dx` with deprecation warnings
3. ✅ Keep backward compatibility through v0.12.x
4. ✅ Remove deprecated names in v1.0.0

**Benefits**:
- Aligns with official naming conventions
- Safe migration path
- Users have time to update
- Clean break at v1.0.0

**What do you want to do?**
- Option A: Aggressive (immediate breaking changes for v1.0)
- Option B: Conservative (just add properties, no migration)
- Option C: Hybrid (migrate with deprecation warnings)
