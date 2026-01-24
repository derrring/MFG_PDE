# HJB Solvers Architecture Analysis

**Date**: 2026-01-23
**Status**: [ANALYSIS]
**Scope**: `mfg_pde/alg/numerical/hjb_solvers/`
**Tracking Issues**: #633 (DRY `_detect_dimension`), #634 (HJB parameter explosion)

---

## 1. Overview

The HJB solver family implements solutions to the Hamilton-Jacobi-Bellman equation:

$$-\frac{\partial u}{\partial t} + H(x, \nabla u, m) - \frac{\sigma^2}{2} \Delta u = 0$$

| Solver | File | Lines | Method | Dimension | Maturity |
|:-------|:-----|:------|:-------|:----------|:---------|
| Base | `base_hjb.py` | 1393 | Newton iteration | 1D | Production |
| FDM | `hjb_fdm.py` | 1051 | Finite Differences | 1D/nD | Production |
| Semi-Lagrangian | `hjb_semi_lagrangian.py` | 1000+ | Characteristic tracing | 1D/nD | Production |
| GFDM | `hjb_gfdm.py` | 1500+ | Meshfree collocation | 1D/nD | Production |
| WENO | `hjb_weno.py` | ~800 | High-order reconstruction | 1D/nD | Production |

---

## 2. Base HJB Solver (`base_hjb.py`)

### 2.1 Architecture

The base class provides:
1. **Newton iteration framework** for nonlinear HJB equations
2. **BC-aware gradient/Laplacian** computation (Issue #542)
3. **Residual and Jacobian** computation
4. **Backend compatibility** (NumPy/PyTorch)

### 2.2 Key Functions

| Function | Lines | Purpose |
|:---------|:------|:--------|
| `_compute_gradient_array_1d()` | 33-68 | BC-aware gradient computation |
| `_compute_laplacian_1d()` | 71-127 | BC-aware Laplacian with ghost cells |
| `_compute_laplacian_ghost_values_1d()` | 130-202 | Ghost cell computation for BC |
| `_get_bc_type_and_value_1d()` | 205-283 | BC type extraction (unified interface) |
| `compute_hjb_residual()` | 607-746 | HJB equation residual |
| `compute_hjb_jacobian()` | 749-956 | Jacobian for Newton iteration |
| `newton_hjb_step()` | 959-1052 | Single Newton iteration |
| `solve_hjb_timestep_newton()` | 1055-1219 | Complete Newton solve for one timestep |
| `solve_hjb_system_backward()` | 1222-1361 | Backward time integration |

### 2.3 Design Quality

**Strengths**:
- Comprehensive BC handling via unified interface
- Issue references throughout (#542, #543, #574, #527)
- Analytical Jacobian with numerical fallback
- Backend abstraction (PyTorch/NumPy)

**Issues**:

| Issue | Location | Severity | Description |
|:------|:---------|:---------|:------------|
| Deprecation accumulation | Lines 1063-1066, 1092-1108, 1230-1232, 1258-1275 | Medium | 4 separate deprecation blocks for same parameters |
| hasattr for backend | Lines 495, 519, 524, 662, 703, 722 | Low | Issue #543 acceptable for external library compat |
| Long file | 1393 lines | Medium | Consider extracting gradient/Laplacian utilities |

---

## 3. HJB FDM Solver (`hjb_fdm.py`)

### 3.1 Architecture

Standard finite difference method with:
- Newton solver for 1D (from base_hjb)
- FixedPointSolver or NewtonSolver for nD
- Multiple advection schemes (centered, upwind)
- Tensor diffusion support (diagonal only)

### 3.2 Key Components

| Component | Lines | Purpose |
|:----------|:------|:--------|
| `is_diagonal_tensor()` | 35-62 | Check if tensor is diagonal |
| `HJBFDMSolver.__init__()` | 108-300 | Initialization with 12+ parameters |
| `solve_hjb_system()` | 324-542 | Main entry (routes 1D vs nD) |
| `_solve_hjb_nd()` | 544-628 | nD backward time loop |
| `_solve_single_timestep()` | 630-704 | Single timestep nonlinear solve |
| `_evaluate_hamiltonian_vectorized()` | 895-1010 | 10-50x faster Hamiltonian |

### 3.3 Design Quality

**Strengths**:
- Dimension-agnostic core logic
- Vectorized Hamiltonian evaluation
- Variational constraint support (Issue #591)
- Trait validation (SupportsGradient)

**Issues**:

| Issue | Location | Severity | Description |
|:------|:---------|:---------|:------------|
| TYPE_CHECKING placement | Lines 35 vs 64-68 | Low | `NDArray` used before import (works due to PEP 563) |
| Deprecated `bc_mode` | Lines 116, 136-140, 181-204 | Medium | Deprecated but still functional |
| Many deprecated params | Lines 119-121, 362-419 | Medium | `NiterNewton`, `l2errBoundNewton`, etc. |
| Partial tensor diffusion | Lines 436-456, 797-857 | Low | Diagonal tensors only |

### 3.4 Advection Schemes

```python
# Available schemes
advection_scheme: Literal["gradient_centered", "gradient_upwind"]

# Gradient-centered: Second-order, may oscillate near discontinuities
# Gradient-upwind: First-order, monotone, better stability
```

---

## 4. HJB Semi-Lagrangian Solver (`hjb_semi_lagrangian.py`)

### 4.1 Architecture

Characteristic-based method:
1. Trace characteristics backward
2. Interpolate at departure points
3. Solve diffusion (ADI for nD)
4. Optimal control via Brent's method

### 4.2 Key Components

| Component | Lines | Purpose |
|:----------|:------|:--------|
| `__init__()` | 101-256 | 19 parameters (too many!) |
| `_setup_jax_functions()` | 258-274 | Optional JAX acceleration |
| `_clip_gradient_with_monitoring()` | 374-499 | Gradient clipping (Issue #583) |
| `_compute_gradient()` | 501-589 | Trait-based gradient |
| `_compute_cfl_and_substeps()` | 591-652 | Adaptive substepping |
| `_get_boundary_conditions()` | 654-699 | BC detection (DUPLICATES base) |

### 4.3 Supporting Modules

| Module | Purpose |
|:-------|:--------|
| `hjb_sl_characteristics.py` | Characteristic tracing (1D/nD) |
| `hjb_sl_interpolation.py` | Value interpolation methods |
| `hjb_sl_adi.py` | ADI diffusion splitting |

### 4.4 Design Quality

**Strengths**:
- Complete nD support
- Adaptive substepping for CFL control
- Gradient clipping with monitoring
- JAX acceleration option

**Issues**:

| Issue | Location | Severity | Description |
|:------|:---------|:---------|:------------|
| **Parameter explosion** | Lines 101-119 | **High** | 19 parameters in `__init__` |
| Embedded gradient clipping | Lines 374-499 | Medium | 125 lines - should be separate |
| BC detection duplication | Lines 654-699 | Medium | Duplicates base class logic |
| JAX divergence | Lines 56-62, 254-274 | Low | Parallel code paths |
| CFL duplication | Lines 591-652 | Low | Partial overlap with gradient |

### 4.5 Parameter Inventory

```python
def __init__(
    self,
    problem: MFGProblem,
    interpolation_method: str = "linear",      # 1. Interpolation
    optimization_method: str = "brent",        # 2. Optimization
    characteristic_solver: str = "explicit_euler",  # 3. Characteristic
    diffusion_method: str = "adi",             # 4. Diffusion
    use_rbf_fallback: bool = True,             # 5. RBF
    rbf_kernel: str = "thin_plate_spline",     # 6. RBF kernel
    use_jax: bool | None = None,               # 7. JAX
    tolerance: float = 1e-8,                   # 8. Tolerance
    max_char_iterations: int = 100,            # 9. Char iterations
    check_cfl: bool = True,                    # 10. CFL check
    enable_adaptive_substepping: bool = True,  # 11. Substepping
    max_substeps: int = 100,                   # 12. Max substeps
    cfl_target: float = 0.9,                   # 13. CFL target
    gradient_clip_threshold: float | None = None,  # 14. Gradient clip
    enable_gradient_monitoring: bool = True,   # 15. Monitoring
    # + 4 more inherited from base...
):
```

**Recommendation**: Group into configuration dataclasses:
- `InterpolationConfig`
- `CharacteristicConfig`
- `StabilityConfig`
- `GradientClipConfig`

---

## 5. HJB GFDM Solver (`hjb_gfdm.py`)

### 5.1 Architecture

Meshfree collocation using Generalized Finite Difference Method:
1. δ-neighborhood for local support
2. Taylor expansion with weighted least squares
3. Newton iteration for nonlinear solve
4. Optional QP constraints for monotonicity

### 5.2 Key Components

| Component | Lines | Purpose |
|:----------|:------|:--------|
| `__init__()` | 164-400+ | **30+ parameters** |
| `neighborhoods` property | 107-162 | Delegate to NeighborhoodBuilder |
| QP optimization | Throughout | Monotonicity via quadratic programming |
| Ghost nodes | 279-288 | Neumann BC enforcement |
| Wind-dependent BC | 281-288 | Viscosity solution compatibility |

### 5.3 External Components

| Component | Module | Purpose |
|:----------|:-------|:--------|
| `NeighborhoodBuilder` | `gfdm_components.py` | Build point neighborhoods |
| `BoundaryHandler` | `gfdm_components.py` | BC enforcement |
| `MonotonicityEnforcer` | `gfdm_components.py` | QP constraints |
| `GridCollocationMapper` | `gfdm_components.py` | Grid-collocation mapping |
| `TaylorOperator` | `gfdm_strategies.py` | Taylor expansion |
| `GFDMOperator` | `gfdm_operators.py` | Legacy operator |

### 5.4 Design Quality

**Strengths**:
- Component extraction started (NeighborhoodBuilder, etc.)
- Strategy pattern for derivative computation
- Comprehensive QP support
- Wind-dependent BC for viscosity solutions

**Issues**:

| Issue | Location | Severity | Description |
|:------|:---------|:---------|:------------|
| **Extreme complexity** | Lines 164-295 | **Critical** | 30+ parameters in `__init__` |
| **Multiple responsibilities** | Throughout | **Critical** | 12+ distinct concerns |
| Legacy + new infrastructure | Lines 22-30 | Medium | Both old and new imports |
| Property accessor overuse | Lines 107-162 | Low | 6 properties just delegate |
| Deprecation accumulation | Lines 175-177, 298-341 | Medium | Multiple deprecated params |

### 5.5 Parameter Categories (30+ total)

```python
# Core (6)
problem, collocation_points, delta, taylor_order, weight_function, weight_scale

# Newton (4, 2 deprecated)
max_newton_iterations, newton_tolerance, NiterNewton, l2errBoundNewton

# Boundary (2)
boundary_indices, boundary_conditions

# QP Optimization (5)
qp_optimization_level, qp_usage_target, qp_solver, qp_warm_start, qp_constraint_mode

# Neighborhoods (5)
adaptive_neighborhoods, k_min, max_delta_multiplier, k_neighbors, neighborhood_mode

# Derivatives (4)
derivative_method, rbf_kernel, rbf_poly_degree, use_new_infrastructure

# Advanced BC (3)
use_local_coordinate_rotation, use_ghost_nodes, use_wind_dependent_bc

# Hamiltonian (1)
congestion_mode
```

**Recommendation**: This class needs decomposition:
1. `GFDMCollocationSolver` (core Newton solve)
2. `GFDMConfiguration` (dataclass for all params)
3. Extract QP, neighborhood, BC handling to existing components

---

## 6. HJB WENO Solver (`hjb_weno.py`)

### 6.1 Architecture

High-order WENO reconstruction:
1. Fifth-order spatial accuracy
2. Non-oscillatory near discontinuities
3. TVD-RK3 time integration
4. Dimensional splitting for nD

### 6.2 Variants

| Variant | Description | Best For |
|:--------|:------------|:---------|
| `weno5` | Standard fifth-order | General problems |
| `weno-z` | Enhanced resolution | High-resolution needs |
| `weno-m` | Mapped weights | Critical points |
| `weno-js` | Original Jiang-Shu | Maximum stability |

### 6.3 Design Quality

**Strengths**:
- Clean variant selection
- Well-documented isotropy assumptions
- Reasonable parameter count (9)
- Ghost buffer composition

**Issues**:

| Issue | Location | Severity | Description |
|:------|:---------|:---------|:------------|
| Dimension detection | `_detect_problem_dimension()` | Low | Duplicated pattern |
| Limited to structured grids | Requirement | Low | By design (WENO needs structure) |

---

## 7. Cross-Cutting Issues

### 7.1 Duplicated Patterns

| Pattern | Occurrences | Recommendation |
|:--------|:------------|:---------------|
| `_detect_dimension()` | All 4 solvers | Extract to `BaseHJBSolver` |
| BC detection | 4 implementations | Use base class `get_boundary_conditions()` |
| Deprecation handling | 10+ blocks | Create `handle_deprecated_params()` utility |
| Dimension branching (1D vs nD) | FDM, SL, GFDM | Consider strategy pattern |

### 7.2 Lex Parsimoniae Violations

| Solver | Parameters | Recommendation |
|:-------|:-----------|:---------------|
| HJB-SL | 19 | Group into 4 config dataclasses |
| HJB-GFDM | 30+ | Decompose class, use config dataclass |
| HJB-FDM | 12 | Acceptable, minor grouping possible |
| HJB-WENO | 9 | Good |

### 7.3 hasattr Usage

Per CLAUDE.md guidelines, `hasattr` is:
- **Prohibited** for duck typing within codebase
- **Acceptable** for external library compatibility (Issue #543)

| Location | Usage | Status |
|:---------|:------|:-------|
| `base_hjb.py:495,519,524` | PyTorch tensor `.item()` | Acceptable |
| `base_hjb.py:662` | PyTorch `.roll()` | Acceptable |
| Other internal uses | Duck typing | Should use Protocol |

---

## 8. Recommendations

### 8.1 Immediate (No Breaking Changes)

1. **Extract dimension detection** to `BaseHJBSolver`:
   ```python
   # In BaseHJBSolver
   def _detect_dimension(self) -> int:
       try:
           return self.problem.geometry.dimension
       except AttributeError:
           return self.problem.dimension
   ```

2. **Consolidate deprecation handling**:
   ```python
   def handle_deprecated_params(new_name, old_name, new_val, old_val, stacklevel=3):
       if old_val is not None:
           warnings.warn(f"'{old_name}' deprecated, use '{new_name}'", DeprecationWarning, stacklevel)
           return old_val if new_val is None else new_val
       return new_val
   ```

3. **Use base class BC resolution** in all solvers instead of custom `_get_boundary_conditions()`.

### 8.2 Medium-Term (Minor Refactoring)

4. **Extract gradient clipping** from HJB-SL to `GradientClipper`:
   ```python
   class GradientClipper:
       def __init__(self, threshold: float, enable_monitoring: bool):
           ...
       def clip(self, grad, t_idx=None, m_density=None):
           ...
       def get_stats(self) -> dict:
           ...
   ```

5. **Create configuration dataclasses** for HJB-SL:
   ```python
   @dataclass
   class SLInterpolationConfig:
       method: str = "linear"
       use_rbf_fallback: bool = True
       rbf_kernel: str = "thin_plate_spline"

   @dataclass
   class SLStabilityConfig:
       check_cfl: bool = True
       enable_adaptive_substepping: bool = True
       max_substeps: int = 100
       cfl_target: float = 0.9
   ```

### 8.3 Long-Term (Major Refactoring)

6. **Decompose HJB-GFDM** into:
   - `GFDMCollocationSolver` (core)
   - `GFDMConfiguration` (dataclass)
   - Fully utilize existing components

7. **Create solver family hierarchy**:
   ```
   BaseHJBSolver
   ├── GridBasedHJBSolver (shared grid logic)
   │   ├── HJBFDMSolver
   │   └── HJBWENOSolver
   ├── CharacteristicHJBSolver (shared SL logic)
   │   └── HJBSemiLagrangianSolver
   └── MeshfreeHJBSolver
       └── HJBGFDMSolver
   ```

---

## 9. File-by-File Summary

| File | Lines | Issues | Priority |
|:-----|:------|:-------|:---------|
| `base_hjb.py` | 1393 | 3 medium | Low |
| `hjb_fdm.py` | 1051 | 4 medium | Low |
| `hjb_semi_lagrangian.py` | 1000+ | 1 high, 4 medium | Medium |
| `hjb_gfdm.py` | 1500+ | 2 critical, 3 medium | **High** |
| `hjb_weno.py` | ~800 | 2 low | Low |
| `hjb_sl_*.py` | ~600 total | None critical | Low |

---

## 10. Related Issues

- Issue #542: BC-aware gradient/Laplacian
- Issue #543: hasattr cleanup
- Issue #545: try/except pattern
- Issue #574: Adjoint-consistent BC
- Issue #580: Scheme family validation
- Issue #583: Gradient clipping
- Issue #591: Variational constraints
- Issue #596: Geometry traits

---

**Last Updated**: 2026-01-23
