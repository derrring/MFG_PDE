# API Consistency Violations

**Date**: 2025-11-11
**Issue**: #277
**Standard**: NAMING_CONVENTIONS.md

---

## Standard (NAMING_CONVENTIONS.md)

**Spatial Discretization** (lines 259-264):
```python
# Array notation for dimension-agnostic code
Nx: list[int]  # [Nx1, Nx2, ..., Nxd]
dx: list[float]  # [dx1, dx2, ..., dxd]

# Example 2D
Nx = [100, 80]
dx = [0.04, 0.025]
```

**Key Points**:
- Use `Nx` (capital N, lowercase x) - NOT `num_points`, `num_intervals`, etc.
- `Nx` must be an array even for 1D: `Nx = [100]`
- Access dimensions via `Nx[0]`, `Nx[1]`, etc.
- Same pattern for `dx`, `xmin`, `xmax` arrays

---

## Category 1: Nx/dx Array Naming Violations

### 1.1 SimpleGrid2D (mfg_pde/geometry/simple_grid.py)

**Violation** (lines 39, 44-45):
```python
# ❌ WRONG: Separate scalar attributes
self.nx, self.ny = resolution
self.dx = (self.xmax - self.xmin) / self.nx
self.dy = (self.ymax - self.ymin) / self.ny
```

**Correct**:
```python
# ✅ Use Nx array
self.Nx = list(resolution)  # [nx, ny]
self.dx = [(self.xmax - self.xmin) / self.Nx[0],
           (self.ymax - self.ymin) / self.Nx[1]]
```

**Impact**: 45+ references to `self.ny`, `self.dy` throughout file

### 1.2 SimpleGrid3D (mfg_pde/geometry/simple_grid.py)

**Violation** (lines 369, 374-375):
```python
# ❌ WRONG: Three separate variables
self.nx, self.ny, self.nz = resolution
self.dy = (self.ymax - self.ymin) / self.ny
self.dz = (self.zmax - self.zmin) / self.nz
```

**Correct**:
```python
# ✅ Use Nx array
self.Nx = list(resolution)  # [nx, ny, nz]
self.dx = [dx1, dx2, dx3]  # Array
```

**Impact**: 60+ references to `self.ny`, `self.nz` in 3D mesh generation

### 1.3 TensorProductGrid (mfg_pde/geometry/tensor_product_grid.py)

**Violation** (lines 75, 110+):
```python
# ❌ WRONG: Uses num_points instead of Nx
def __init__(self, dimension: int, bounds: Sequence[tuple[float, float]],
             num_points: Sequence[int], ...):
    self.num_points = list(num_points)
```

**Correct**:
```python
# ✅ Use Nx naming
def __init__(self, dimension: int, bounds: Sequence[tuple[float, float]],
             Nx: Sequence[int], ...):
    self.Nx = list(Nx)
```

**Impact**: Used throughout codebase as "the correct pattern" - but violates naming standard

**Migration Note**: This is the most widely used geometry class. Need careful deprecation strategy.

---

## Category 2: Boolean Pairs → Enum Conversion

### 2.1 PINN Training Strategy

**Location**: `mfg_pde/alg/neural/pinn_solvers/adaptive_training.py:64-82`

**Violation**:
```python
enable_curriculum: bool = True
enable_multiscale: bool = True
enable_refinement: bool = True
```

**Correct**: Use enum with deprecation warnings (see FPParticleSolver pattern at fp_particle.py:26-125)

### 2.2 Normalization Type

**Location**: `mfg_pde/alg/neural/pinn_solvers/base_pinn.py:83-84`

**Violation**:
```python
use_batch_norm: bool = False
use_layer_norm: bool = False
```

**Correct**: Use `NormalizationType` enum

### 2.3 DGM Variance Reduction

**Location**: `mfg_pde/alg/neural/dgm/base_dgm.py:64-65`

**Violation**:
```python
use_control_variates: bool = True
use_importance_sampling: bool = False
```

**Correct**: Use `VarianceReductionMethod` enum

---

## Category 3: Tuple Returns → Dataclass

### 3.1 Time Domain

**Current**: `time_domain: tuple[float, int] = (1.0, 100)`

**Issue**: Unpacked everywhere as `self.T, self.Nt = time_domain`

**Correct**: Create `TimeDomain` dataclass with properties

### 3.2 Domain Bounds

**Current**: Various tuple formats (flat tuples for 1D/2D/3D)

**Correct**: Structured format (TensorProductGrid uses correct format with list of tuples)

---

## Category 4: Deprecated Test API

**Files**: 5 integration test files using old `GridBasedMFGProblem` API (currently skipped)

Need migration to geometry-first API.

---

## Priority Summary

| Violation | Files | Priority | Effort |
|:----------|:------|:---------|:-------|
| SimpleGrid Nx/dx | 2 classes | HIGH | Medium (2-3 days) |
| TensorProductGrid Nx | 1 class | HIGH | Large (4-5 days, widely used) |
| Boolean → Enum | 3 files | MEDIUM | Small (1-2 days) |
| TimeDomain dataclass | Core | HIGH | Medium (2-3 days) |
| Test API migration | 5 files | MEDIUM | Medium (1 week) |

**Total Estimated Effort**: 3-4 weeks

---

**Reference**: FPParticleSolver (fp_particle.py:26-125) shows gold standard enum conversion with deprecation warnings
