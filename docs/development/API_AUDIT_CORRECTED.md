# API Consistency Audit (Corrected)

**Status**: Phase 1 - Violation Inventory
**Date**: 2025-11-11
**Issue**: #277

This document catalogs actual API consistency violations based on NAMING_CONVENTIONS.md standards.

---

## Executive Summary

**Correct Standard** (per NAMING_CONVENTIONS.md):
- ✅ `Nx`, `dx` **ARE the standard notation** for mathematical algorithms
- ✅ Should be **arrays**: `Nx=[100, 80]` for 2D (not separate `Ny`, `Nz`)
- ✅ Array-based notation enables dimension-agnostic code
- ❌ Separate `ny`, `nz` variables violate the standard
- ❌ Scalar `Nx=100` is deprecated (should be `Nx=[100]`)

**Violation Counts**:
- **Array-Based Notation**: 2 geometry classes (SimpleGrid2D, SimpleGrid3D)
- **Boolean Pairs**: 6 high-priority conversions needed
- **Tuple Returns**: 2 critical dataclass conversions
- **Deprecated Test API**: 5 integration test files

---

## Category 1: Array-Based Notation Violations

### Standard (NAMING_CONVENTIONS.md lines 50-77, 259-264)

**Correct Pattern** (`TensorProductGrid`):
```python
class TensorProductGrid:
    def __init__(self, dimension: int, bounds: Sequence[tuple[float, float]], num_points: Sequence[int]):
        self.dimension = dimension
        self.num_points = list(num_points)  # [N1, N2, ..., Nd]
        self.bounds = list(bounds)  # [(min1, max1), (min2, max2), ...]

        # Compute spacing array
        self.spacing = [
            (bounds[i][1] - bounds[i][0]) / num_points[i]
            for i in range(dimension)
        ]

    # Access via indexing
    # self.num_points[0]  # Points in dimension 0
    # self.spacing[1]     # Spacing in dimension 1
```

### Violation 1: SimpleGrid2D (mfg_pde/geometry/simple_grid.py)

**Lines 39, 44-45**:
```python
# ❌ WRONG: Separate variables
self.nx, self.ny = resolution
self.dx = (self.xmax - self.xmin) / self.nx
self.dy = (self.ymax - self.ymin) / self.ny

# ✅ CORRECT: Array-based
self.num_points = list(resolution)  # [nx, ny]
self.spacing = [
    (self.xmax - self.xmin) / self.num_points[0],
    (self.ymax - self.ymin) / self.num_points[1]
]

# Backward compatibility properties
@property
def nx(self) -> int:
    return self.num_points[0]

@property
def ny(self) -> int:
    return self.num_points[1]
```

**Impact**: 45+ references to `self.ny` throughout simple_grid.py (lines 39, 45, 63, 89, 100-147, 216, 232, 234, 248, 333)

**Migration Effort**: Medium (2-3 hours)
- Update constructor to use arrays
- Add compatibility properties for `nx`, `ny`, `dx`, `dy`
- Update all internal references to use array indexing
- Test 2D grid generation

### Violation 2: SimpleGrid3D (mfg_pde/geometry/simple_grid.py)

**Lines 369, 374-375**:
```python
# ❌ WRONG: Three separate variables
self.nx, self.ny, self.nz = resolution
self.dy = (self.ymax - self.ymin) / self.ny
self.dz = (self.zmax - self.zmin) / self.nz

# ✅ CORRECT: Array-based
self.num_points = list(resolution)  # [nx, ny, nz]
self.spacing = [
    (self.xmax - self.xmin) / self.num_points[0],
    (self.ymax - self.ymin) / self.num_points[1],
    (self.zmax - self.zmin) / self.num_points[2]
]
```

**Impact**: 60+ references to `self.ny`, `self.nz` in 3D mesh generation (lines 369-450)

**Migration Effort**: Medium (2-3 hours)
- Similar to 2D, but with additional complexity
- Update 3D tetrahedron element generation
- Add compatibility properties
- Test 3D grid generation

### Other Files Using Array Unpacking

**utils/sparse_operations.py** (lines 168, 207, 283, 313):
```python
# Current (may be acceptable if grid.spacing is already an array)
dx, dy = self.grid.spacing  # Line 168
dx, dy, dz = self.grid.spacing  # Line 207

# This is actually CORRECT if grid.spacing returns an array
# The issue is that SimpleGrid2D doesn't provide spacing as an array property
```

**Verdict**: sparse_operations.py is correct - it expects grid.spacing to be an array. The violation is that SimpleGrid2D/3D don't provide it.

---

## Category 2: Boolean Pairs → Enum Conversion

### 2.1 PINN Training Strategy (HIGH PRIORITY)

**Location**: `mfg_pde/alg/neural/pinn_solvers/adaptive_training.py:64-82`

**Current**:
```python
enable_curriculum: bool = True
enable_multiscale: bool = True
enable_refinement: bool = True
```

**Proposed**:
```python
class AdaptiveTrainingMode(str, Enum):
    """Adaptive training mode for PINN solvers."""
    BASIC = "basic"  # No adaptive features
    CURRICULUM = "curriculum"  # Curriculum learning only
    MULTISCALE = "multiscale"  # Multiscale training only
    REFINEMENT = "refinement"  # Adaptive refinement only
    FULL = "full"  # All adaptive features (default)

training_mode: AdaptiveTrainingMode | str = AdaptiveTrainingMode.FULL
```

**Deprecation**:
```python
# Handle deprecated boolean flags
if enable_curriculum is not None or enable_multiscale is not None or enable_refinement is not None:
    warnings.warn(
        "Boolean flags enable_curriculum/enable_multiscale/enable_refinement are deprecated. "
        "Use training_mode parameter with AdaptiveTrainingMode enum instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Map to enum based on combination
    if all([enable_curriculum, enable_multiscale, enable_refinement]):
        training_mode = AdaptiveTrainingMode.FULL
    elif enable_curriculum and not enable_multiscale and not enable_refinement:
        training_mode = AdaptiveTrainingMode.CURRICULUM
    # ... etc
```

**Rationale**: Three mutually-configurable booleans create 8 possible states (2³). Enum clarifies which combinations are supported and makes common configurations explicit.

### 2.2 Normalization Type (MEDIUM PRIORITY)

**Location**: `mfg_pde/alg/neural/pinn_solvers/base_pinn.py:83-84`

**Current**:
```python
use_batch_norm: bool = False
use_layer_norm: bool = False
```

**Proposed**:
```python
class NormalizationType(str, Enum):
    """Normalization strategy for neural networks."""
    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"
    INSTANCE = "instance"  # Future extensibility
    GROUP = "group"  # Future extensibility

normalization: NormalizationType | str = NormalizationType.NONE
```

**Rationale**: Mutually exclusive options. Also allows future normalization types without adding more booleans.

### 2.3 DGM Variance Reduction (MEDIUM PRIORITY)

**Location**: `mfg_pde/alg/neural/dgm/base_dgm.py:64-65`

**Current**:
```python
use_control_variates: bool = True
use_importance_sampling: bool = False
```

**Proposed**:
```python
class VarianceReductionMethod(str, Enum):
    """Variance reduction strategy for Monte Carlo sampling."""
    NONE = "none"
    CONTROL_VARIATES = "control_variates"
    IMPORTANCE_SAMPLING = "importance_sampling"
    COMBINED = "combined"  # Both methods

variance_reduction: VarianceReductionMethod | str = VarianceReductionMethod.CONTROL_VARIATES
```

### 2.4 Already Converted ✅ (Reference Pattern)

**FPParticleSolver** (mfg_pde/alg/numerical/fp_solvers/fp_particle.py:26-125):
```python
class KDENormalization(str, Enum):
    NONE = "none"
    INITIAL_ONLY = "initial_only"
    ALL = "all"

class ParticleMode(str, Enum):
    HYBRID = "hybrid"
    COLLOCATION = "collocation"

# Deprecation handling (lines 103-119)
if normalize_kde_output is not None or normalize_only_initial is not None:
    warnings.warn(..., DeprecationWarning, stacklevel=2)
    # Map old booleans to new enum
    if normalize_kde_output is False:
        kde_normalization = KDENormalization.NONE
    elif normalize_only_initial is True:
        kde_normalization = KDENormalization.INITIAL_ONLY
    else:
        kde_normalization = KDENormalization.ALL
```

**This is the gold standard pattern to follow.**

---

## Category 3: Tuple Returns → Dataclass Conversion

### 3.1 Time Domain (HIGH PRIORITY)

**Current Usage**:
```python
time_domain: tuple[float, int] = (1.0, 100)
self.T, self.Nt = time_domain  # Unpacking everywhere
```

**Locations**:
- `mfg_pde/core/highdim_mfg_problem.py:71, 85`
- `mfg_pde/core/mfg_problem.py` (time_domain parameter)
- All MFG problem subclasses

**Proposed**:
```python
@dataclass
class TimeDomain:
    """Time discretization parameters for MFG problems."""
    T_final: float  # Terminal time
    num_steps: int  # Number of time intervals (Nt)

    def __post_init__(self):
        if self.T_final <= 0:
            raise ValueError("T_final must be positive")
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1")

    @property
    def dt(self) -> float:
        """Time step size: dt = T / Nt"""
        return self.T_final / self.num_steps

    @property
    def Nt(self) -> int:
        """Alias for num_steps (backward compatibility)."""
        return self.num_steps

    @property
    def T(self) -> float:
        """Alias for T_final (backward compatibility)."""
        return self.T_final

    @property
    def time_grid(self) -> np.ndarray:
        """Time grid points: [0, dt, 2*dt, ..., T]"""
        return np.linspace(0, self.T_final, self.num_steps + 1)

    @classmethod
    def from_tuple(cls, t: tuple[float, int]) -> TimeDomain:
        """Support legacy tuple format: (T, Nt)"""
        return cls(T_final=t[0], num_steps=t[1])
```

**Migration**:
```python
def __init__(self, time_domain: TimeDomain | tuple[float, int] = (1.0, 100), ...):
    # Normalize to dataclass
    if isinstance(time_domain, tuple):
        warnings.warn(
            "Passing time_domain as tuple (T, Nt) is deprecated. "
            "Use TimeDomain(T_final=..., num_steps=...) dataclass instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        time_domain = TimeDomain.from_tuple(time_domain)

    self.time_domain = time_domain
    self.T = time_domain.T  # Backward compatibility
    self.Nt = time_domain.Nt  # Backward compatibility
```

**Benefits**:
- Named access: `time_domain.T_final` vs `time_domain[0]`
- Computed properties: `time_domain.dt`, `time_domain.time_grid`
- Validation in `__post_init__`
- Type-safe

**Migration Effort**: Medium (1-2 days)
- Create TimeDomain dataclass
- Update all MFGProblem constructors
- Add backward compatibility
- Test all problem types

### 3.2 Domain Bounds (MEDIUM PRIORITY)

**Current**: Various tuple formats
- 1D: `bounds=(xmin, xmax)`
- 2D: `bounds=(xmin, xmax, ymin, ymax)`
- 3D: `bounds=(xmin, xmax, ymin, ymax, zmin, zmax)`

**Issue**: Non-uniform format makes dimension-agnostic code difficult

**Proposed**:
```python
@dataclass
class DomainBounds:
    """Domain bounds for nD problems."""
    bounds_per_dim: list[tuple[float, float]]  # [(min1, max1), (min2, max2), ...]

    @property
    def dimension(self) -> int:
        return len(self.bounds_per_dim)

    @property
    def min_coords(self) -> np.ndarray:
        return np.array([b[0] for b in self.bounds_per_dim])

    @property
    def max_coords(self) -> np.ndarray:
        return np.array([b[1] for b in self.bounds_per_dim])

    @property
    def lengths(self) -> np.ndarray:
        """Domain length in each dimension."""
        return self.max_coords - self.min_coords

    @classmethod
    def from_flat_tuple(cls, bounds: tuple[float, ...]) -> DomainBounds:
        """Convert (xmin, xmax, ymin, ymax, ...) to DomainBounds."""
        if len(bounds) % 2 != 0:
            raise ValueError("Bounds tuple must have even length")
        dim = len(bounds) // 2
        bounds_per_dim = [(bounds[2*i], bounds[2*i+1]) for i in range(dim)]
        return cls(bounds_per_dim=bounds_per_dim)
```

**Note**: TensorProductGrid already uses `bounds: Sequence[tuple[float, float]]` format ✅

**Migration Effort**: Low (this is mostly documentation - TensorProductGrid already correct)

---

## Category 4: Deprecated Test API Migration

### 4.1 Skipped Integration Tests

**Files requiring migration**:
1. `tests/integration/test_coupled_hjb_fp_2d.py` (3 tests)
   - `test_coupled_hjb_fp_2d_basic()`
   - `test_coupled_hjb_fp_2d_weak_coupling()`
   - `test_coupled_hjb_fp_dimension_detection()`

2. `tests/integration/test_hjb_fdm_2d_validation.py` (all tests)

**Current Status**: All skipped with:
```python
@pytest.mark.skip(
    reason="Requires comprehensive API migration. Tracked in Issue #277."
)
```

**Old API** (deprecated):
```python
problem = SimpleCoupledMFGProblem(
    domain_bounds=(0.0, 1.0, 0.0, 1.0),  # Flat tuple
    grid_resolution=10,  # Scalar
    time_domain=(0.2, 10),  # Tuple
)
```

**New API** (correct):
```python
from mfg_pde.geometry.tensor_product_grid import TensorProductGrid

grid = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],  # List of tuples
    num_points=[10, 10]  # Array
)
time_domain = TimeDomain(T_final=0.2, num_steps=10)  # Dataclass

problem = SimpleCoupledMFGProblem(
    geometry=grid,
    time_domain=time_domain,
)
```

**Migration Effort**: Medium (1 week)
- Migrate 5 test files to new API
- Verify all assertions still valid
- Update test utilities

---

## Implementation Priority

| Category | Priority | Effort | Breaking? | Timeline |
|:---------|:---------|:-------|:----------|:---------|
| **TimeDomain dataclass** | HIGH | Medium (1-2 days) | No (accept both) | Week 1 |
| **PINN Training enum** | HIGH | Small (4 hours) | No (deprecate) | Week 1 |
| **SimpleGrid array notation** | HIGH | Medium (2-3 days) | No (properties) | Week 2 |
| **Normalization enum** | MEDIUM | Small (2 hours) | No (deprecate) | Week 2 |
| **DGM variance enum** | MEDIUM | Small (2 hours) | No (deprecate) | Week 2 |
| **Test API migration** | HIGH | Medium (1 week) | No (tests only) | Week 3 |

---

## Phase Plan

### Phase 1: Documentation ✅ (Completed)
- Audit violations
- Document correct standards
- Create migration patterns

### Phase 2: Non-Breaking Enum Conversions (Week 1)
- PINN training mode enum
- Add deprecation warnings
- Tests for new enums
- Tests for deprecated booleans

### Phase 3: TimeDomain Dataclass (Week 1-2)
- Create TimeDomain dataclass
- Add tuple compatibility
- Update all problem classes
- Comprehensive testing

### Phase 4: SimpleGrid Migration (Week 2)
- Convert to array-based notation
- Add compatibility properties
- Update internal references
- Test 2D/3D grid generation

### Phase 5: Test Migration (Week 3)
- Migrate 5 integration test files
- Remove skip markers
- Verify all tests pass

### Phase 6: Additional Enums (Week 2-3)
- Normalization type enum
- DGM variance reduction enum
- Documentation updates

---

## Success Criteria

- [ ] All boolean pairs converted to enums with deprecation warnings
- [ ] TimeDomain dataclass implemented with tuple compatibility
- [ ] SimpleGrid2D/3D use array-based notation
- [ ] All 5 skipped integration tests migrated and passing
- [ ] Zero breaking changes (all backward compatible)
- [ ] Comprehensive documentation updated
- [ ] Deprecation timeline established

---

## References

- **NAMING_CONVENTIONS.md**: Authoritative source for naming standards
- **Issue #277**: API Consistency Audit tracking
- **PR #275**: Examples redesign (blocked by test migration)
- **fp_particle.py:26-125**: Gold standard enum conversion pattern

---

**Audit Completed**: 2025-11-11
**Next**: Phase 2 implementation (enum conversions)
