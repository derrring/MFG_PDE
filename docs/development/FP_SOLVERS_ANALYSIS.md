# FP Solvers Analysis

**Date**: 2026-01-23
**Status**: [ANALYSIS]
**Context**: Comprehensive review of Fokker-Planck solver implementations
**Tracking Issues**: #633 (DRY `_detect_dimension`), #635 (FPParticleSolver split)

---

## 1. Overview

The `mfg_pde/alg/numerical/fp_solvers/` directory contains 17 files implementing various Fokker-Planck equation solvers:

| Solver | File | Lines | Status |
|--------|------|-------|--------|
| BaseFPSolver | `base_fp.py` | 291 | Clean |
| FPFDMSolver | `fp_fdm.py` | 1463 | Complex |
| FPGFDMSolver | `fp_gfdm.py` | 609 | Good |
| FPSLSolver | `fp_semi_lagrangian.py` | 648 | Clean |
| FPSLAdjointSolver | `fp_semi_lagrangian_adjoint.py` | 446 | Clean |
| FPParticleSolver | `fp_particle.py` | 2800+ | Too large |

**Supporting Modules**:
- `fp_fdm_advection.py` - Advection term discretization
- `fp_fdm_operators.py` - Common utilities
- `fp_fdm_bc.py` - BC handling
- `fp_fdm_alg_*.py` (4 files) - Individual advection scheme implementations
- `particle_result.py`, `particle_density_query.py` - Particle solver utilities

---

## 2. Individual Solver Analysis

### 2.1 BaseFPSolver (`base_fp.py`)

**Assessment**: Clean, well-designed abstract base class.

**Strengths**:
- Simple abstract interface with `solve_fp_system()` method
- Supports both `drift_field` and `diffusion_field` parameters
- Good documentation with docstrings
- Inherits from `BaseNumericalSolver` for BC resolution

**Code Structure** (291 lines):
```python
class BaseFPSolver(BaseNumericalSolver, ABC):
    @abstractmethod
    def solve_fp_system(
        self,
        m_initial_condition: np.ndarray,
        drift_field: np.ndarray | Callable | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,
        show_progress: bool = True,
    ) -> np.ndarray: ...
```

---

### 2.2 FPFDMSolver (`fp_fdm.py`)

**Assessment**: Complex with multiple code paths, but functional.

**Strengths**:
- 4 advection schemes with clear classification:
  - `gradient_centered` - NOT conservative, Pe < 2 stability
  - `gradient_upwind` - Conservative via row sums, always stable
  - `divergence_centered` - Conservative via flux, Pe < 2 stability
  - `divergence_upwind` - Conservative via telescoping, always stable
- Supports callable drift/diffusion with bootstrap strategy
- Well-documented scheme comparison table

**Issues**:

1. **Complex BC Resolution Hierarchy** (lines 184-229):
   ```python
   # 5-step BC resolution:
   # 1. Explicit boundary_conditions parameter
   # 2. problem.geometry.boundary_conditions
   # 3. problem.geometry.get_boundary_conditions()
   # 4. boundary_type string (legacy, deprecated)
   # 5. None (default to periodic_or_no_flux)
   ```
   This duplicates logic from `BaseNumericalSolver`.

2. **Legacy 1D vs Unified nD Routing** (lines 260-290):
   - Separate code paths for 1D and nD
   - 1D path exists "for backward compatibility"
   - Creates maintenance burden

3. **File Size**: 1463 lines is borderline too large.

4. **Parameter Count**: 12 constructor parameters:
   ```python
   def __init__(
       self, problem, boundary_conditions=None, boundary_type=None,
       conservative=True, upwind=True, advection_scheme=None,
       time_stepping_scheme="implicit_euler", solver_type="sparse",
       precompute_operators=False, enable_limiting=False,
       flux_limiter=None, verbose=True
   ): ...
   ```

---

### 2.3 FPGFDMSolver (`fp_gfdm.py`)

**Assessment**: Good implementation with honest documentation of limitations.

**Strengths**:
- Uses `GFDMOperator` for spatial derivatives (clean abstraction)
- Adaptive delta computation from point spacing
- Upwind stabilization options (exponential/linear)
- **Excellent documentation** of what it does NOT do:
  ```python
  """
  ACTUAL Implementation (径向有限差分 + 流线加权):
  This is NOT full GFDM Taylor expansion - it's a simplified approach...

  ✗ Lower order accuracy than full GFDM (O(h) radial FD vs O(h²) Taylor)
  ✗ Not using precomputed GFDM operator (rebuilds each call)
  """
  ```

**Issues**:

1. **BC Resolution Duplicates Base Class** (lines 152-197):
   - `_resolve_boundary_type()` does its own BC parsing
   - Should delegate to inherited `get_boundary_conditions()`

2. **No Callable Drift Support**:
   ```python
   if callable(drift_field):
       raise NotImplementedError("Callable drift_field not yet supported for GFDM")
   ```

---

### 2.4 FPSLSolver (`fp_semi_lagrangian.py`)

**Assessment**: Clean implementation with proper mathematical foundation.

**Strengths**:
- Clear operator splitting: advection with Jacobian + Crank-Nicolson diffusion
- Reuses BC infrastructure from HJB-SL via `apply_boundary_conditions_1d`
- Uses `tensor_calculus.laplacian` for proper BC handling (Issue #579 fix)
- Good documentation of the Jacobian correction formula

**Issues**:

1. **Duplicated `_detect_dimension()`** (lines 137-153):
   ```python
   def _detect_dimension(self, problem: Any) -> int:
       try:
           return problem.geometry.dimension
       except AttributeError:
           pass
       try:
           return problem.dimension
       except AttributeError:
           pass
       # ... legacy detection
   ```
   Same code exists in HJB-SL, FP-SL-Adjoint, and GFDM solvers.

2. **BC Resolution Duplicates Base Class** (lines 155-181):
   - `_get_boundary_conditions_from_problem()` reimplements base class logic

3. **`drift_field` Takes U, Converts Internally**:
   - Input is value function U from HJB
   - Velocity computed internally: `alpha = -grad(U)` (optimal control for H = ½|p|²)
   - Design choice: SL needs U for characteristic tracing anyway

---

### 2.5 FPSLAdjointSolver (`fp_semi_lagrangian_adjoint.py`)

**Assessment**: Clean implementation of the adjoint (forward) SL method.

**Strengths**:
- Correct adjoint interpretation: forward splatting vs backward interpolation
- Mass-conservative by construction (no Jacobian needed)
- Finite-volume interpretation for trapezoidal quadrature consistency
- Proper documentation of duality with HJB backward SL

**Issues**:
- Same as FPSLSolver: duplicated `_detect_dimension()`, BC resolution

---

### 2.6 FPParticleSolver (`fp_particle.py`)

**Assessment**: Too large, needs refactoring.

**Strengths**:
- Rich functionality: KDE normalization, multiple BC handling modes
- Well-integrated with `ParticleApplicator` for BC enforcement
- Supports implicit domains

**Issues**:

1. **File Size**: 2800+ lines is far too large
   - Should be split into:
     - Core particle evolution
     - KDE/density estimation
     - Boundary handling
     - Visualization/diagnostics

2. **Complex Enum**: `KDENormalization` has 5 modes with subtle differences

3. **Many Edge Cases**: Extensive special-casing for different scenarios

---

## 3. Cross-Cutting Issues

### 3.1 DRY Violation: `_detect_dimension()`

**Severity**: Medium
**Locations**:
- `fp_semi_lagrangian.py:137-153`
- `fp_semi_lagrangian_adjoint.py:110-120`
- `hjb_sl_characteristics.py:149-165`
- `hjb_gfdm.py:152-171`
- `hjb_semi_lagrangian.py:89-110`

**Pattern**:
```python
def _detect_dimension(self, problem: Any) -> int:
    try:
        return problem.geometry.dimension
    except AttributeError:
        pass
    try:
        return problem.dimension
    except AttributeError:
        pass
    # Legacy fallback...
```

**Recommendation**: Move to `BaseNumericalSolver` as a protected method.

---

### 3.2 `drift_field` Interface Design

**Semantics**: The `drift_field` in FP solvers represents the **optimal control** from HJB.

In MFG theory:
- HJB produces value function U(t,x)
- Optimal control: α* = -∇_p H(x, ∇U, m)
- For quadratic Hamiltonian H = ½|p|²: α* = -∇U
- FP uses α* as the drift field

| Solver | Input | Conversion | Notes |
|--------|-------|------------|-------|
| FPFDMSolver | U or velocity | Internal if U | Flexible |
| FPGFDMSolver | velocity α* | None | Direct use |
| FPSLSolver | U | `alpha = -grad(U)` | Internal conversion |
| FPSLAdjointSolver | U | `alpha = -grad(U)` | Internal conversion |

**Design Rationale**:
- SL solvers take U directly because they need it for characteristic tracing
- GFDM takes velocity directly because it's more general (works with any Hamiltonian)
- FDM is flexible for backward compatibility

This is intentional design, not inconsistency.

---

### 3.3 BC Resolution Hierarchy Duplication

**Severity**: Medium

Multiple solvers implement their own BC resolution:
- `FPFDMSolver._resolve_boundary_conditions()`
- `FPGFDMSolver._resolve_boundary_type()`
- `FPSLSolver._get_boundary_conditions_from_problem()`

**Recommendation**: Use inherited `get_boundary_conditions()` from `BaseNumericalSolver`.

---

### 3.4 Scheme Family Trait

**Status**: Good - consistently implemented

All FP solvers have `_scheme_family` trait for duality validation (Issue #580):
```python
from mfg_pde.alg.base_solver import SchemeFamily
_scheme_family = SchemeFamily.FDM  # or SL, GFDM
```

---

## 4. Advection Scheme Architecture

### 4.1 Scheme Classification (2x2 Matrix)

| Form | Spatial | Scheme Name | Conservative | Stable |
|------|---------|-------------|--------------|--------|
| Gradient (v·∇m) | Centered | `gradient_centered` | NO | Pe < 2 |
| Gradient (v·∇m) | Upwind | `gradient_upwind` | YES (rows) | Always |
| Divergence (∇·(vm)) | Centered | `divergence_centered` | YES (flux) | Pe < 2 |
| Divergence (∇·(vm)) | Upwind | `divergence_upwind` | YES (flux) | Always |

### 4.2 File Organization

Good separation of concerns:
```
fp_fdm_alg_gradient_centered.py   - Non-conservative gradient form
fp_fdm_alg_gradient_upwind.py     - Conservative gradient upwind
fp_fdm_alg_divergence_centered.py - Conservative divergence centered
fp_fdm_alg_divergence_upwind.py   - Conservative divergence upwind
```

### 4.3 Integration with AdvectionOperator (Issue #597)

The `fp_fdm_advection.py` module now uses `AdvectionOperator` internally:
```python
def compute_advection_term_nd(...):
    from mfg_pde.geometry.operators.advection import AdvectionOperator
    adv_op = AdvectionOperator(velocity_field, spacings, shape,
                               scheme=scheme, form="divergence")
    return adv_op(M)
```

This is a good modernization that leverages the unified operator infrastructure.

---

## 5. Comparison with HJB Solvers

| Aspect | FP Solvers | HJB Solvers |
|--------|------------|-------------|
| Parameter explosion | Lower (12 max) | Higher (19-30) |
| Code duplication | Medium | High |
| File sizes | 1 oversized | 2 oversized |
| Scheme family trait | Consistent | Consistent |
| Documentation | Better | Variable |

FP solvers are generally in better shape than HJB solvers, with:
- Better documentation
- Fewer parameters
- Cleaner advection scheme organization

---

## 6. Recommendations

### Immediate (Priority 1)

1. **Extract `_detect_dimension()` to base class**
   - Single implementation in `BaseNumericalSolver`
   - All solvers inherit it
   - Eliminates 5+ duplicate implementations

2. **Document `drift_field` semantics clearly**
   - Add docstring note: "drift_field = optimal control α* from HJB"
   - For SL solvers: note that U is converted to α* = -∇U internally
   - For GFDM: note that caller provides α* directly for Hamiltonian flexibility

### Medium Term (Priority 2)

3. **Split FPParticleSolver**
   - Extract KDE utilities to separate module
   - Extract BC handling to use ParticleApplicator
   - Target: 600-800 lines per file

4. **Consolidate BC resolution**
   - Remove solver-specific BC resolution
   - Use inherited `get_boundary_conditions()` from base class

### Long Term (Priority 3)

5. **Remove legacy 1D paths**
   - FPFDMSolver has separate 1D code path
   - Should be unified under nD implementation

6. **Document drift_field semantics**
   - Add note in docstrings: "drift_field = optimal control α* from HJB"
   - Current design is intentional (SL takes U, GFDM takes velocity)

---

## 7. File Summary Table

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `__init__.py` | 43 | Exports | Clean |
| `base_fp.py` | 291 | Abstract base | Clean |
| `fp_fdm.py` | 1463 | FDM solver | Complex, borderline size |
| `fp_fdm_advection.py` | 275 | Advection terms | Clean |
| `fp_fdm_bc.py` | ~200 | BC handling | Not reviewed |
| `fp_fdm_operators.py` | 92 | Utilities | Clean |
| `fp_fdm_alg_*.py` | ~200 ea | Advection schemes | Good modular design |
| `fp_gfdm.py` | 609 | GFDM solver | Good, honest docs |
| `fp_semi_lagrangian.py` | 648 | SL solver | Clean |
| `fp_semi_lagrangian_adjoint.py` | 446 | Adjoint SL | Clean |
| `fp_particle.py` | 2800+ | Particle solver | Too large, needs split |
| `particle_*.py` | ~300 | Particle utilities | Not reviewed |

---

## 8. Conclusion

FP solvers are in better overall shape than HJB solvers:

**Positive**:
- Cleaner abstraction with advection scheme separation
- Consistent scheme family traits for duality validation
- Good documentation in GFDM and SL solvers
- Modern integration with AdvectionOperator

**Needs Work**:
- `_detect_dimension()` duplication across 5+ files
- `fp_particle.py` is too large (2800+ lines)
- BC resolution reimplemented in each solver

Priority should be:
1. Extract shared utilities (dimension detection, BC resolution)
2. Split fp_particle.py
3. Document drift_field semantics (optimal control from HJB)
