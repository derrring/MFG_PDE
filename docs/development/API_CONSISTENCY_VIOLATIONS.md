# API Consistency Violations Inventory

**Status**: Phase 1 Audit Complete
**Date**: 2025-11-11
**Issue**: #277

This document catalogs API consistency violations across MFG_PDE requiring standardization.

---

## Executive Summary

**Violation Counts**:
- **Naming Conventions**: ~30 files with `Nx`, `dx` patterns
- **Boolean Pairs**: 15+ candidates for enum conversion
- **Tuple Returns**: 10+ functions needing dataclass conversion
- **Deprecated APIs**: 5 test files using old `GridBasedMFGProblem` API

**Priority Areas**:
1. Core problem classes (MFGProblem, GridBasedMFGProblem)
2. Solver parameters (PINN, DeepONet, variational solvers)
3. Geometry classes (SimpleGrid2D, TensorProductGrid)
4. Test migration (5 files with skipped tests)

---

## Category 1: Naming Convention Violations

### 1.1 Grid Dimension Naming: `Nx`, `Ny`, `Nz` → `num_points`

**Standard**: Use `num_points[d]` or `num_points` for grid resolution

**Violations** (30 files):

#### Core Problems
- **mfg_pde/core/variational_mfg_problem.py:154**
  ```python
  self.dx = self.Lx / Nx  # Should use: spacing[0]
  ```

- **mfg_pde/core/mfg_problem.py:97-99**
  ```python
  >>> MFGProblem._normalize_to_array(100, "Nx")  # Warns
  [100]
  ```

#### Variational Solvers (HIGH PRIORITY)
- **mfg_pde/alg/optimization/variational_solvers/base_variational.py:91, 98**
  ```python
  self.Nx = problem.Nx
  logger.info(f"  Grid: {self.Nx + 1} × {self.Nt} (space × time)")
  ```

- **mfg_pde/alg/optimization/variational_solvers/variational_mfg_solver.py**: 10 occurrences
  - Lines: 117, 140, 141, 180, 253, 306

- **mfg_pde/alg/optimization/variational_solvers/primal_dual_solver.py**: 5 occurrences
  - Lines: 255, 262, 266, 318, 338

#### Geometry Classes
- **mfg_pde/geometry/simple_grid.py:39, 44-45**
  ```python
  self.nx, self.ny = resolution  # Should use: num_points
  self.dx = (self.xmax - self.xmin) / self.nx  # Should use: spacing
  self.dy = (self.ymax - self.ymin) / self.ny
  ```

- **mfg_pde/geometry/simple_grid_1d.py**: Uses `Nx`, `Dx`
- **mfg_pde/geometry/projection.py**: Uses `Nx`, `Ny`, `dx`, `dy`
- **mfg_pde/geometry/tensor_product_grid.py**: Uses `num_points` ✅ (CORRECT)

#### Visualization and Documentation
- **mfg_pde/visualization/legacy_plotting.py**: Uses `Nx`, `Ny`
- **mfg_pde/solve_mfg.py:70-71**
  ```python
  # Comments use old notation:
  - U: Value function array (Nt+1, Nx+1) or (Nt+1, Nx+1, Ny+1)
  - M: Density array (Nt+1, Nx+1) or (Nt+1, Nx+1, Ny+1)
  ```

#### Network Problems
- **mfg_pde/core/network_mfg_problem.py:504-509**
  ```python
  @property
  def Dx(self) -> float:  # Should be: spacing
      """Spatial discretization for continuous interpolation."""
  ```

### 1.2 Spacing/Grid Size: `dx`, `dy`, `dz` → `spacing`

**Standard**: Use `spacing[d]` for grid spacing in dimension d

**Violations** (20+ files):
- All files using `Nx` patterns also use corresponding `dx` patterns
- **mfg_pde/core/variational_mfg_problem.py:154, 158**
  ```python
  self.dx = self.Lx / Nx
  self.dt = T / (Nt - 1)  # Time spacing should also be spacing[0] in time domain
  ```

- **mfg_pde/alg/numerical/fp_solvers/fp_particle.py:149, 159**
  ```python
  def _compute_gradient(self, U_array, Dx: float, use_backend: bool = False):
  def _normalize_density(self, M_array, Dx: float, use_backend: bool = False):
  ```

### 1.3 Variable Naming: Single-Letter vs Descriptive

**Current**: Mix of `M`, `U`, `m`, `u` for density/value function
**Issue**: Inconsistent capitalization and clarity

**Examples**:
- `M` vs `m_density` vs `density`
- `U` vs `u_value` vs `value_function`

**Recommendation**: Establish consistent convention in style guide (likely keep `M`, `U` for mathematical clarity, but document clearly)

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
class TrainingStrategy(str, Enum):
    BASIC = "basic"
    CURRICULUM = "curriculum"
    MULTISCALE = "multiscale"
    REFINEMENT = "refinement"
    ADAPTIVE = "adaptive"  # All strategies combined

training_strategy: TrainingStrategy = TrainingStrategy.ADAPTIVE
```

**Migration**: Add deprecation warnings for old boolean parameters

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
    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"
    GROUP = "group"  # Future extensibility

normalization: NormalizationType = NormalizationType.NONE
```

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
    NONE = "none"
    CONTROL_VARIATES = "control_variates"
    IMPORTANCE_SAMPLING = "importance_sampling"
    COMBINED = "combined"  # Both methods

variance_reduction: VarianceReductionMethod = VarianceReductionMethod.CONTROL_VARIATES
```

### 2.4 Already Converted (REFERENCE) ✅

**Location**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py:26-38`

**Good Pattern**:
```python
class KDENormalization(str, Enum):
    """KDE normalization strategy for particle-based FP solvers."""
    NONE = "none"
    INITIAL_ONLY = "initial_only"
    ALL = "all"

class ParticleMode(str, Enum):
    """Particle solver operating mode for FP-based solvers."""
    HYBRID = "hybrid"
    COLLOCATION = "collocation"
```

**Deprecation Handling** (lines 103-119):
```python
if normalize_kde_output is not None or normalize_only_initial is not None:
    warnings.warn(
        "Parameters 'normalize_kde_output' and 'normalize_only_initial' are deprecated. "
        "Use 'kde_normalization' instead...",
        DeprecationWarning,
        stacklevel=2,
    )
    # Map old parameters to new enum
    if normalize_kde_output is False:
        kde_normalization = KDENormalization.NONE
    elif normalize_only_initial is True:
        kde_normalization = KDENormalization.INITIAL_ONLY
    else:
        kde_normalization = KDENormalization.ALL
```

### 2.5 Low-Priority Boolean Pairs

**Keep as separate booleans** (independent features, not mutually exclusive):
- **DeepONet**: `use_bias_net`, `use_attention` (independent architectural choices)
- **DGM Residual**: `use_residual_connections` (single boolean is appropriate)
- **Factory**: `enable_amr` (single boolean for AMR toggle)

---

## Category 3: Tuple Returns → Dataclass Conversion

### 3.1 Time Domain Specification (HIGH PRIORITY)

**Current**: `time_domain: tuple[float, int] = (1.0, 100)`

**Usage**:
```python
# mfg_pde/core/highdim_mfg_problem.py:85-86
self.T, self.Nt = time_domain
```

**Proposed**:
```python
@dataclass
class TimeDomain:
    """Time discretization parameters."""
    T_final: float  # Terminal time
    num_steps: int  # Number of time steps

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T_final / self.num_steps

    @property
    def time_grid(self) -> np.ndarray:
        """Time grid points."""
        return np.linspace(0, self.T_final, self.num_steps + 1)
```

**Migration**: Accept both tuple and dataclass with deprecation warning

### 3.2 Domain Bounds (MEDIUM PRIORITY)

**Current**: Various tuple formats
- 1D: `(xmin, xmax)`
- 2D: `(xmin, xmax, ymin, ymax)`
- 3D: `(xmin, xmax, ymin, ymax, zmin, zmax)`

**Proposed**:
```python
@dataclass
class DomainBounds:
    """Domain bounds for nD problems."""
    min_coords: np.ndarray  # Shape (dimension,)
    max_coords: np.ndarray  # Shape (dimension,)

    @classmethod
    def from_tuple(cls, bounds: tuple[float, ...]) -> DomainBounds:
        """Convert tuple to DomainBounds."""
        if len(bounds) % 2 != 0:
            raise ValueError("Bounds must have even length")
        dim = len(bounds) // 2
        min_coords = np.array([bounds[2*i] for i in range(dim)])
        max_coords = np.array([bounds[2*i+1] for i in range(dim)])
        return cls(min_coords, max_coords)
```

### 3.3 Grid Resolution (MEDIUM PRIORITY)

**Current**: Various formats
- Single int: `resolution=32`
- Tuple: `resolution=(32, 32)`
- List: `[32, 32, 32]`

**Proposed**: Accept all formats, normalize internally
```python
def _normalize_resolution(
    resolution: int | tuple[int, ...] | list[int],
    dimension: int
) -> list[int]:
    """Normalize resolution to list format."""
    if isinstance(resolution, int):
        return [resolution] * dimension
    return list(resolution)
```

### 3.4 Tuple Return Examples (Search Results)

**Found 12 files with tuple returns**:
- `mfg_pde/geometry/simple_grid.py`: `bounds` property returns tuple
- `mfg_pde/utils/performance/monitoring.py`: Performance metrics as tuples
- `mfg_pde/geometry/implicit/hyperrectangle.py`: Bounds as tuples
- `mfg_pde/backends/numba_backend.py`: Multiple return values

**Action**: Review each case individually (not all need dataclass conversion)

---

## Category 4: Deprecated Test API Migration

### 4.1 Skipped Tests Requiring Migration

**Issue**: 5 integration tests skipped due to deprecated `GridBasedMFGProblem` API

**Files**:
1. **tests/integration/test_coupled_hjb_fp_2d.py**
   - `test_coupled_hjb_fp_2d_basic()`
   - `test_coupled_hjb_fp_2d_weak_coupling()`
   - `test_coupled_hjb_fp_dimension_detection()`

2. **tests/integration/test_hjb_fdm_2d_validation.py** (module-level skip)
   - All tests in module

**Current Status**: Skipped with reason:
```python
@pytest.mark.skip(
    reason="Requires comprehensive API migration. Tracked in Issue #277 (API Consistency Audit)."
)
```

**Migration Required**:
```python
# OLD (deprecated)
problem = SimpleCoupledMFGProblem(
    domain_bounds=(0.0, 1.0, 0.0, 1.0),
    grid_resolution=10,
    time_domain=(0.2, 10),
    diffusion_coeff=0.1,
)

# NEW (modern geometry-first API)
from mfg_pde.geometry.tensor_product_grid import TensorProductGrid

grid = TensorProductGrid(
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    num_points=[10, 10]
)
problem = SimpleCoupledMFGProblem(
    geometry=grid,
    time_domain=(0.2, 10),  # Will migrate to TimeDomain dataclass
    diffusion_coeff=0.1,
)
```

---

## Category 5: Documentation Cleanup

### 5.1 Docstring Updates

**Search for outdated references**:
- "Nx+1" → "num_points + 1"
- "dx" → "spacing"
- Update mathematical notation to match new naming

**Example** (mfg_pde/solve_mfg.py:70-71):
```python
# Current:
- U: Value function array (Nt+1, Nx+1) or (Nt+1, Nx+1, Ny+1)

# Updated:
- U: Value function array (num_time_steps+1, num_spatial_points+1)
     For 2D: (num_time_steps+1, num_points[0]+1, num_points[1]+1)
```

### 5.2 Example Code Updates

**Update all examples** to use modern API patterns:
- `examples/basic/`
- `examples/advanced/`
- `examples/tutorials/`
- `docs/` inline examples

---

## Priority Matrix

| Category | Priority | Effort | Files | Breaking? |
|:---------|:---------|:-------|:------|:----------|
| Naming: Nx/dx → num_points/spacing | HIGH | Large | ~30 | Yes |
| PINN Training Strategy enum | HIGH | Medium | 2 | No (deprecate) |
| Time domain dataclass | HIGH | Small | 10 | No (accept both) |
| Normalization enum | MEDIUM | Small | 3 | No (deprecate) |
| DGM variance reduction enum | MEDIUM | Small | 2 | No (deprecate) |
| Domain bounds dataclass | MEDIUM | Medium | 15 | No (accept both) |
| Test API migration | HIGH | Medium | 2 | No (tests only) |
| Documentation cleanup | HIGH | Large | All | No |

---

## Implementation Strategy

### Phase 1: Documentation and Style Guide (1 week)
- ✅ Create violation inventory (this document)
- ⏳ Create comprehensive style guide
- ⏳ Document migration patterns
- ⏳ Plan deprecation timeline

### Phase 2: Non-Breaking Changes (1 week)
- Add dataclasses (accept both tuple and dataclass)
- Add enums with deprecation warnings
- Update documentation strings
- Add migration examples

### Phase 3: Core Naming Migration (2-3 weeks)
- Migrate `Nx`, `dx` → `num_points`, `spacing`
- Update all internal usage
- Update all examples
- Comprehensive testing

### Phase 4: Test Migration (1 week)
- Migrate 5 skipped integration tests
- Update test utilities
- Verify all tests pass

### Phase 5: Deprecation Removal (v2.0.0)
- Remove deprecated parameters
- Clean up compatibility code
- Final documentation pass

---

## Migration Patterns

### Pattern 1: Enum with Deprecation

```python
class MyEnum(str, Enum):
    OPTION_A = "a"
    OPTION_B = "b"

def __init__(
    self,
    mode: MyEnum | str = MyEnum.OPTION_A,
    # Deprecated parameters
    use_option_a: bool | None = None,
    use_option_b: bool | None = None,
):
    # Handle deprecation
    if use_option_a is not None or use_option_b is not None:
        warnings.warn(
            "Parameters 'use_option_a' and 'use_option_b' are deprecated. "
            "Use 'mode' parameter with MyEnum instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_option_a:
            mode = MyEnum.OPTION_A
        elif use_option_b:
            mode = MyEnum.OPTION_B

    # Convert string to enum
    if isinstance(mode, str):
        mode = MyEnum(mode)

    self.mode = mode
```

### Pattern 2: Dataclass with Tuple Compatibility

```python
@dataclass
class MyConfig:
    param1: float
    param2: int

    @classmethod
    def from_tuple(cls, t: tuple[float, int]) -> MyConfig:
        """Support legacy tuple format."""
        return cls(param1=t[0], param2=t[1])

def __init__(self, config: MyConfig | tuple[float, int]):
    if isinstance(config, tuple):
        warnings.warn(
            "Passing config as tuple is deprecated. Use MyConfig dataclass.",
            DeprecationWarning,
            stacklevel=2,
        )
        config = MyConfig.from_tuple(config)

    self.config = config
```

### Pattern 3: Naming with Property Aliases

```python
class MyClass:
    def __init__(self, num_points: int):
        self._num_points = num_points

    @property
    def num_points(self) -> int:
        """Modern API: number of grid points."""
        return self._num_points

    @property
    def Nx(self) -> int:
        """Deprecated: Use num_points instead."""
        warnings.warn(
            "Property 'Nx' is deprecated. Use 'num_points' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._num_points
```

---

## Testing Requirements

For each migration:
1. Add tests for new API
2. Add tests for deprecated API (verify warnings)
3. Add tests for mixed usage
4. Verify backward compatibility
5. Update integration tests

---

## Success Criteria

- [ ] All naming conventions consistent with style guide
- [ ] All boolean pairs converted to enums (where applicable)
- [ ] All critical tuple returns converted to dataclasses
- [ ] All 5 skipped tests migrated and passing
- [ ] Comprehensive documentation updated
- [ ] Deprecation warnings in place
- [ ] No breaking changes until v2.0.0
- [ ] All existing tests pass

---

## References

- **Issue #277**: API Consistency Audit
- **PR #275**: Examples redesign (blocked by test API migration)
- **NAMING_CONVENTIONS.md**: Existing naming conventions documentation
- **CONSISTENCY_GUIDE.md**: Development consistency guide

---

**Audit Completed**: 2025-11-11
**Next Steps**: Create API style guide, begin Phase 2 implementation
