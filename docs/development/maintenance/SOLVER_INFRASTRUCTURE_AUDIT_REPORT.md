# Technical Report: Solver Infrastructure Audit

**Issue**: #487
**Date**: 2025-12-16
**Revision**: 1.0
**Status**: Assessment Complete
**Author**: MFG_PDE Development Team

---

## Executive Summary

This report documents a comprehensive audit of the MFG_PDE solver infrastructure, identifying critical gaps, incomplete implementations, and architectural defects that limit solver capabilities. The audit covers HJB solvers (FDM, WENO, GFDM, Semi-Lagrangian), FP solvers (FDM, Particle, GFDM), and supporting utilities.

**Key Findings**:
- 3 critical blockers preventing 3D+ support
- 5 Phase 2 features incomplete (state-dependent coefficients)
- 4 architectural defects requiring refactoring
- Estimated 40+ hours of implementation work across all items

---

## 1. Audit Methodology

### 1.1 Scope

The audit examined:
- All HJB solvers in `mfg_pde/alg/numerical/hjb_solvers/`
- All FP solvers in `mfg_pde/alg/numerical/fp_solvers/`
- Supporting utilities in `mfg_pde/utils/numerical/`
- Geometry and boundary modules in `mfg_pde/geometry/`

### 1.2 Criteria

Each component was evaluated for:
1. **Completeness**: Are all advertised features implemented?
2. **Dimensionality**: Does it work for 1D, 2D, 3D, nD?
3. **Integration**: Does it properly use problem/geometry APIs?
4. **Robustness**: Are edge cases handled?

---

## 2. Critical Blockers (NotImplementedError)

These issues completely block certain use cases.

### 2.1 GFDM 3D+ Support Blocked

**Location**: `hjb_solvers/hjb_gfdm.py:1172`

```python
else:
    msg = f"Dimension {d} not implemented"
    raise NotImplementedError(msg)
```

**Context**: DerivativeTensors computation for Hessians only handles d=1 and d=2.

**Impact**:
- GFDM is a meshfree method designed for complex geometries
- 3D problems (crowd dynamics in buildings, traffic in 3D networks) completely blocked
- Users forced to use FDM/WENO with structured grids for 3D

**Root Cause**: Hessian matrix construction hardcoded for 2x2:
```python
hess = np.array([
    [derivs.get((2, 0), 0.0), derivs.get((1, 1), 0.0)],
    [derivs.get((1, 1), 0.0), derivs.get((0, 2), 0.0)],
])
```

**Fix Complexity**: Medium - Need to generalize to NxN Hessian construction.

### 2.2 nD Tensor Diffusion Blocked

**Location**: `utils/numerical/tensor_operators.py:660`

```python
raise NotImplementedError("General nD tensor diffusion (d>2) not yet implemented...")
```

**Impact**:
- Anisotropic diffusion in 3D not possible
- Limits modeling of directional crowd flow, stratified populations

**Current Support**:
| Dimension | Scalar σ | Diagonal Σ | Full Tensor Σ |
|-----------|----------|------------|---------------|
| 1D | ✅ | N/A | N/A |
| 2D | ✅ | ✅ | ✅ |
| 3D+ | ✅ | ✅ | ❌ Blocked |

**Fix Complexity**: High - Requires tensor calculus generalization.

### 2.3 Flux Diagnostics Limited to 2D

**Location**: `utils/numerical/flux_diagnostics.py:193, 230`

```python
# Only _compute_flux_1d and _compute_flux_2d implemented
```

**Impact**:
- Cannot validate mass conservation in 3D FP simulations
- Debugging 3D solver issues significantly harder
- Quality assurance gap for 3D problems

**Fix Complexity**: Medium - Pattern exists for 1D/2D, needs 3D extension.

---

## 3. Phase 2 Incomplete Features (State-Dependent Coefficients)

These features are documented in the roadmap but not implemented.

### 3.1 Callable Drift Field in FP FDM

**Location**: `fp_solvers/fp_fdm.py:306`

```python
# TODO: Phase 2 - Support callable drift_field
if callable(drift_field):
    raise NotImplementedError("Callable drift_field not yet supported")
```

**Expected API**:
```python
def drift(t, x, m):
    """State-dependent drift α(t, x, m)."""
    return -lambda_coeff * grad_u(x) + external_force(x, m)

solver = FPFDMSolver(problem, drift_field=drift)
```

**Impact**: Cannot model:
- Density-dependent crowd avoidance
- Time-varying external forces
- Feedback control systems

### 3.2 Callable Drift Field in FP Particle

**Location**: `fp_solvers/fp_particle.py:789`

```python
if callable(drift_field):
    raise NotImplementedError("Callable drift_field for particle method not yet supported")
```

**Impact**: Same as 3.1, but for particle-based methods.

### 3.3 Spatially-Varying Diffusion in FP Particle

**Location**: `fp_solvers/fp_particle.py:806`

```python
if callable(diffusion_field) or isinstance(diffusion_field, np.ndarray):
    raise NotImplementedError("Spatially-varying diffusion_field not yet supported")
```

**Expected API**:
```python
def sigma(x):
    """Position-dependent noise intensity."""
    return base_sigma * (1 + 0.5 * np.sin(x[0]))  # Varying diffusion

solver = FPParticleSolver(problem, diffusion_field=sigma)
```

**Impact**: Cannot model:
- Heterogeneous environments (indoor vs outdoor)
- Boundary layers with reduced diffusion
- Multi-scale turbulence

### 3.4 Tensor Diffusion in FP GFDM

**Location**: `fp_solvers/fp_gfdm.py:169`

```python
raise NotImplementedError("Only scalar diffusion currently supported")
```

**Impact**: GFDM FP limited to isotropic diffusion despite meshfree flexibility.

### 3.5 Tensor Diffusion in FP FDM (1D)

**Location**: `fp_solvers/fp_fdm.py:325`

```python
# 1D tensor diffusion not supported (Phase 3.0)
```

**Impact**: Minor - 1D tensor diffusion rarely needed.

---

## 4. Architectural Defects

### 4.1 HJB FDM Missing BC Integration (nD Path)

**Location**: `hjb_solvers/hjb_fdm.py:150-185`

**Issue**: The nD solver path completely ignores boundary conditions from the problem.

```python
# Current code - no BC retrieval
def __init__(self, problem, ...):
    if not isinstance(problem.geometry, CartesianGrid):
        raise ValueError("nD FDM requires problem with CartesianGrid geometry")
    self.grid = problem.geometry
    # NO: self.boundary_conditions = problem.get_boundary_conditions()
```

**Impact**:
- Users cannot specify Dirichlet/Neumann/Periodic BCs for nD HJB
- Implicit "natural" BC assumed (one-sided stencils)
- Inconsistent with FP solvers which do retrieve BCs

**Related**: Issue #486 addresses this specifically.

### 4.2 Tensor Diffusion Silent Data Loss

**Location**: `hjb_solvers/hjb_fdm.py:580-630`

**Issue**: Off-diagonal tensor elements silently discarded.

```python
# User provides full tensor
Sigma = np.array([[1.0, 0.5],
                  [0.5, 2.0]])  # Correlated diffusion

# Solver silently uses only diagonal
sigma_effective = np.array([Sigma[0,0], Sigma[1,1]])  # [1.0, 2.0]
# Off-diagonal 0.5 LOST without warning at runtime
```

**Warning exists at initialization** (lines 314-320) but not at evaluation time.

**Impact**: Users may unknowingly solve incorrect anisotropic problems.

**Fix**: Either:
1. Raise error for non-diagonal tensors, OR
2. Implement full tensor support

### 4.3 Inconsistent BC Retrieval Patterns

**Issue**: Different solvers use different patterns to access BCs.

| Pattern | Solvers Using It | Correct? |
|---------|-----------------|----------|
| `problem.boundary_conditions` | (Legacy) | ❌ May not exist |
| `problem.get_boundary_conditions()` | GFDM, Semi-Lagrangian | ✅ Preferred |
| `problem.components.boundary_conditions` | FP FDM | ⚠️ OK but verbose |
| `problem.geometry.get_boundary_handler()` | FP Particle | ⚠️ OK but different |
| None (ignored) | HJB FDM, HJB WENO | ❌ Missing |

**Fix**: Standardize on `problem.get_boundary_conditions()` across all solvers.

### 4.4 DerivativeTensors Migration Incomplete

**Issue**: Dual interface support creates confusion.

**Old Interface** (dict-based):
```python
derivs = {(1, 0): du_dx, (0, 1): du_dy, (2, 0): d2u_dx2, ...}
H = problem.H(idx, m, derivs=derivs)
```

**New Interface** (DerivativeTensors):
```python
derivs = DerivativeTensors(grad=grad, hess=hess)
H = problem.hamiltonian(x, m, derivs.grad, t)
```

**Current State**: Solvers try new interface, fall back to old:
```python
try:
    H = self.problem.H(idx, m, derivs=derivs)  # New
except TypeError:
    legacy = to_multi_index_dict(derivs)
    H = self.problem.H(idx, m, derivs=legacy)  # Old fallback
```

**Impact**:
- Maintenance burden of dual support
- Unclear which interface to use for new problems
- Performance overhead of try/except in hot loop

---

## 5. Dimension-Specific Limitations

### 5.1 WENO Dimensional Splitting

**Location**: `hjb_solvers/hjb_weno.py:75-97`

**Issue**: 2D/3D WENO uses dimensional splitting (Strang/Godunov) rather than true multi-dimensional reconstruction.

```python
# Current: Solve 1D WENO in each direction sequentially
for dim in range(ndim):
    u = weno_step_1d(u, dim, dt)
```

**Limitation**:
- Works well for isotropic Hamiltonians
- May fail for anisotropic Hamiltonians (traffic flow with preferred directions)
- Not truly high-order in multi-D

**Documentation states this limitation but no fix planned.**

### 5.2 Semi-Lagrangian nD Interpolation

**Location**: `hjb_solvers/hjb_semi_lagrangian.py`

**Issue**: 1D uses optimized cubic interpolation; nD uses generic `RegularGridInterpolator`.

| Dimension | Interpolation | Performance |
|-----------|--------------|-------------|
| 1D | Custom cubic spline | Fast |
| nD | scipy.RegularGridInterpolator | Slower, generic |

**Impact**: nD semi-Lagrangian ~3-5x slower than optimal.

### 5.3 FDM Solver Selection by Dimension

**Location**: `hjb_solvers/hjb_fdm.py:324-334`

**Issue**: Different solver paths for 1D vs nD.

| Dimension | Solver | Optimization Level |
|-----------|--------|-------------------|
| 1D | Custom Newton with Thomas algorithm | Highly optimized |
| nD | Generic FixedPointSolver/NewtonSolver | Generic sparse |

**Impact**: 1D problems run ~10x faster than equivalent 2D problems (beyond expected N² scaling).

---

## 6. Missing Infrastructure Components

### 6.1 Meshfree Neumann BC

**Location**: `geometry/boundary/applicator_meshfree.py:261`

```python
raise NotImplementedError(
    "Neumann BC for collocation requires solver-specific implementation..."
)
```

**Impact**: GFDM solvers cannot directly use Neumann BCs from geometry module.

**Current Workaround**: Ghost particles in `gfdm_operators.py` (partial support).

### 6.2 3D BC Applicator Optimization

**Location**: `geometry/boundary/applicator_fdm.py:1234`

```python
# TODO: Add optimized 3D implementation with face-specific handling
return apply_boundary_conditions_nd(field, boundary_conditions, domain_bounds, time, config)
```

**Impact**: Performance only - 3D BC application uses generic nD code.

### 6.3 CoefficientField Tensor Support

**Location**: `utils/pde_coefficients.py`

**Issue**: `CoefficientField` abstraction handles scalar and callable coefficients but tensor support is incomplete.

```python
# Works
diffusion = CoefficientField(sigma, default=0.1)

# Partially works
diffusion = CoefficientField(sigma_array, default=0.1)

# Not fully supported
diffusion = CoefficientField(sigma_tensor_callable, default=np.eye(2))
```

---

## 7. Summary Tables

### 7.1 Severity Classification

| Severity | Count | Description |
|----------|-------|-------------|
| **Critical** | 3 | Completely blocks functionality |
| **High** | 5 | Major feature missing or broken |
| **Medium** | 6 | Limitation with workaround |
| **Low** | 4 | Performance or documentation issue |

### 7.2 By Component

| Component | Critical | High | Medium | Low |
|-----------|----------|------|--------|-----|
| HJB GFDM | 1 | 0 | 1 | 0 |
| HJB FDM | 0 | 2 | 1 | 1 |
| HJB WENO | 0 | 0 | 2 | 0 |
| HJB Semi-Lagrangian | 0 | 0 | 1 | 0 |
| FP FDM | 0 | 2 | 1 | 0 |
| FP Particle | 0 | 2 | 0 | 0 |
| FP GFDM | 0 | 1 | 0 | 0 |
| Utilities | 2 | 0 | 0 | 1 |
| Geometry/BC | 0 | 0 | 0 | 2 |

### 7.3 By Effort

| Effort | Items | Examples |
|--------|-------|----------|
| Low (< 4h) | 6 | BC retrieval standardization, documentation |
| Medium (4-16h) | 8 | 3D flux diagnostics, callable drift |
| High (16-40h) | 4 | GFDM 3D, full tensor diffusion |

---

## 8. Recommended Prioritization

### 8.1 Immediate (Enables 3D Support)

| Item | Effort | Impact |
|------|--------|--------|
| GFDM 3D Hessian | High | Unblocks 3D meshfree |
| 3D Flux diagnostics | Medium | Enables 3D validation |
| nD tensor operators | High | Enables 3D anisotropic |

### 8.2 Short-Term (Phase 2 Completion)

| Item | Effort | Impact |
|------|--------|--------|
| Callable drift (FP FDM) | Medium | State-dependent dynamics |
| Callable drift (FP Particle) | Medium | Same for particle method |
| Spatially-varying σ | Medium | Heterogeneous environments |

### 8.3 Medium-Term (Architecture)

| Item | Effort | Impact |
|------|--------|--------|
| BC retrieval standardization | Low | Code consistency |
| DerivativeTensors migration | Medium | Remove legacy support |
| Tensor diffusion (full) | High | Correlated noise |

### 8.4 Long-Term (Optimization)

| Item | Effort | Impact |
|------|--------|--------|
| True multi-D WENO | Very High | Research-level |
| nD interpolation optimization | Medium | Performance |
| 3D BC applicator | Low | Performance |

---

## 9. Dependencies and Ordering

```
                    ┌─────────────────────┐
                    │  GFDM 3D Hessian    │
                    │  (Critical Blocker) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ nD Tensor Ops   │ │ 3D Flux Diag│ │ Callable Drift  │
    │ (Enables 3D     │ │ (Validates  │ │ (Phase 2)       │
    │  anisotropic)   │ │  3D solvers)│ │                 │
    └────────┬────────┘ └──────┬──────┘ └────────┬────────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ BC Unification      │
                    │ (Issue #486)        │
                    └─────────────────────┘
```

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 3D GFDM breaks existing 2D | Medium | High | Comprehensive test suite |
| Tensor ops performance regression | Medium | Medium | Benchmark before/after |
| Phase 2 API changes | Low | Medium | Deprecation warnings |
| DerivativeTensors migration breaks user code | Medium | High | Dual support period |

---

## 11. References

1. **GFDM Theory**: Benito, J.J. et al. "Generalized finite difference method" (2001)
2. **WENO Schemes**: Shu, C.W. "Essentially Non-Oscillatory and WENO Schemes" (2009)
3. **MFG Numerics**: Achdou, Y. "Mean Field Games: Numerical Methods" (2020)
4. **Tensor Diffusion**: Weickert, J. "Anisotropic Diffusion in Image Processing" (1998)

---

## Appendix A: File Index

### Critical Files (Blockers)
- `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:1172` - 3D Hessian
- `mfg_pde/utils/numerical/tensor_operators.py:660` - nD tensor
- `mfg_pde/utils/numerical/flux_diagnostics.py:193` - 3D flux

### Phase 2 Files
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py:306` - Callable drift
- `mfg_pde/alg/numerical/fp_solvers/fp_particle.py:789, 806` - Callable drift/σ
- `mfg_pde/alg/numerical/fp_solvers/fp_gfdm.py:169` - Tensor diffusion

### Architecture Files
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py:150-185` - BC integration
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py:580-630` - Tensor silent loss

---

## Appendix B: Code Examples

### B.1 GFDM 3D Hessian Fix (Sketch)

```python
def _compute_hessian_nd(self, derivs: dict, dimension: int) -> np.ndarray:
    """Compute NxN Hessian matrix for arbitrary dimension."""
    hess = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            # Multi-index for ∂²u/∂xᵢ∂xⱼ
            idx = [0] * dimension
            idx[i] += 1
            idx[j] += 1
            hess[i, j] = derivs.get(tuple(idx), 0.0)
    return hess
```

### B.2 Callable Drift Implementation (Sketch)

```python
def _evaluate_drift(self, t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Evaluate drift field at given state."""
    if callable(self.drift_field):
        return self.drift_field(t, x, m)
    elif isinstance(self.drift_field, np.ndarray):
        return self.drift_field  # Static array
    else:
        # Default: gradient of value function
        return -self.coupling_coefficient * self._compute_gradient(self.U_current)
```

### B.3 Tensor Diffusion Validation

```python
def _validate_tensor_diffusion(self, Sigma: np.ndarray) -> None:
    """Validate tensor diffusion matrix."""
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Diffusion tensor must be square, got {Sigma.shape}")

    # Check symmetry
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Diffusion tensor must be symmetric")

    # Check positive semi-definiteness
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.any(eigvals < -1e-10):
        raise ValueError(f"Diffusion tensor must be positive semi-definite, got eigenvalues {eigvals}")

    # Warn about off-diagonal if not supported
    if not self._supports_full_tensor():
        off_diag = np.abs(Sigma - np.diag(np.diag(Sigma))).max()
        if off_diag > 1e-10:
            warnings.warn(
                f"Off-diagonal tensor elements (max={off_diag:.2e}) will be ignored. "
                "Only diagonal diffusion supported in this solver."
            )
```
