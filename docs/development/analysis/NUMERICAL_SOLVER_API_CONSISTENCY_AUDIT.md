# Numerical Solver API Consistency Audit

**Date**: 2025-10-04
**Scope**: All numerical solvers in `mfg_pde/alg/numerical/`
**Purpose**: Identify and document API inconsistencies across solver hierarchy

---

## Executive Summary

**Status**: 🟡 **MODERATE INCONSISTENCY** - Several naming and parameter convention issues identified

**Critical Issues**:
1. ❌ **Inconsistent problem parameter naming**: `problem` vs `mfg_problem`
2. ❌ **Inconsistent Newton parameter naming**: `max_newton_iterations` vs `hjb_newton_iterations`
3. ⚠️ **Deprecated parameters still present**: `NiterNewton`, `l2errBoundNewton`
4. ⚠️ **Missing backend parameter**: Not all solvers support `backend` parameter

**Overall Score**: 6/10 (needs improvement)

---

## API Inventory by Solver Type

### HJB Solvers (5 solvers)

| Solver | problem | max_newton_iterations | newton_tolerance | backend | Notes |
|:-------|:--------|:---------------------|:----------------|:--------|:------|
| **HJBFDMSolver** | ✅ problem | ✅ | ✅ | ✅ | ✅ Modern API |
| **HJBGFDMSolver** | ✅ problem | ✅ | ✅ | ❌ | Missing backend |
| **HJBSemiLagrangianSolver** | ✅ problem | ❌ max_char_iterations | ✅ tolerance | ❌ | Different param names |
| **HJBWenoSolver** | ✅ problem | ❌ | ❌ | ❌ | Scheme-specific params |
| **NetworkHJBSolver** | ✅ problem | ✅ max_iterations | ✅ tolerance | ❌ | Network-specific |

**Consistency Score**: 7/10

### FP Solvers (3 solvers)

| Solver | problem | boundary_conditions | backend | Notes |
|:-------|:--------|:-------------------|:--------|:------|
| **FPFDMSolver** | ✅ problem | ✅ | ❌ | Minimal API |
| **FPParticleSolver** | ✅ problem | ✅ | ✅ | ✅ Modern API |
| **FPNetworkSolver** | ✅ problem | ❌ | ❌ | Network-specific |

**Consistency Score**: 6/10

### MFG Solvers (6 solvers)

| Solver | problem param | hjb_solver | fp_solver | Notes |
|:-------|:-------------|:-----------|:----------|:------|
| **FixedPointIterator** | ✅ problem | ✅ | ✅ | ✅ Standard composition |
| **ConfigAwareFixedPointIterator** | ✅ problem | ✅ | ✅ | Config-driven |
| **HybridFPParticleHJBFDM** | ❌ **mfg_problem** | internal | internal | ❌ **INCONSISTENT** |
| **ParticleCollocationSolver** | ✅ problem | internal | internal | Monolithic |
| **AdaptiveParticleCollocationSolver** | ✅ problem | internal | internal | Adaptive variant |
| **MonitoredParticleCollocationSolver** | ✅ problem | internal | internal | Monitoring variant |

**Consistency Score**: 4/10 (HybridFPParticleHJBFDM breaks convention)

---

## Detailed Inconsistencies

### 1. ❌ **CRITICAL: Inconsistent Problem Parameter Name**

**Issue**: `HybridFPParticleHJBFDM` uses `mfg_problem` instead of `problem`

**Evidence**:
```python
# 99% of solvers
def __init__(self, problem: MFGProblem, ...):

# HybridFPParticleHJBFDM (INCONSISTENT)
def __init__(self, mfg_problem: MFGProblem, ...):
```

**Impact**:
- Breaks API consistency
- Confuses users who expect `problem` parameter
- Incompatible with generic factory patterns

**Recommendation**: ✅ **FIX IMMEDIATELY** - Rename to `problem` with deprecation warning

---

### 2. ❌ **CRITICAL: Inconsistent Newton Iteration Parameters**

**Issue**: `HybridFPParticleHJBFDM` uses `hjb_newton_iterations` instead of `max_newton_iterations`

**Evidence**:
```python
# Standard naming (HJBFDMSolver, HJBGFDMSolver, ParticleCollocationSolver)
def __init__(self, ..., max_newton_iterations: int = 30, newton_tolerance: float = 1e-6):

# HybridFPParticleHJBFDM (INCONSISTENT)
def __init__(self, ..., hjb_newton_iterations: int = 30, hjb_newton_tolerance: float = 1e-7):
```

**Impact**:
- Parameter name conflict
- Not compatible with standard solver factories
- Unnecessarily verbose

**Recommendation**: ✅ **FIX** - Align with standard naming

---

### 3. ⚠️ **Deprecated Parameters Still Present**

**Issue**: Several solvers still accept deprecated parameters with fallback

**Evidence**:
```python
# HJBFDMSolver, HJBGFDMSolver
def __init__(
    self,
    problem: MFGProblem,
    max_newton_iterations: int | None = None,  # New
    newton_tolerance: float | None = None,      # New
    NiterNewton: int | None = None,             # DEPRECATED
    l2errBoundNewton: float | None = None,      # DEPRECATED
    ...
):
```

**Status**: ✅ **ACCEPTABLE** - Proper deprecation warnings in place

**Recommendation**: Keep for backward compatibility, remove in v2.0.0

---

### 4. ⚠️ **Inconsistent Backend Support**

**Issue**: Not all solvers support `backend` parameter

| Solver Type | Backend Support |
|:------------|:---------------|
| HJBFDMSolver | ✅ Yes |
| HJBGFDMSolver | ❌ No |
| HJBSemiLagrangianSolver | ❌ No (has `use_jax`) |
| HJBWenoSolver | ❌ No |
| FPFDMSolver | ❌ No |
| FPParticleSolver | ✅ Yes |
| FixedPointIterator | ✅ Yes |

**Recommendation**: ⚠️ **MEDIUM PRIORITY** - Add optional `backend` to all solvers

---

### 5. ⚠️ **Inconsistent Tolerance Parameter Names**

**Issue**: Different solvers use different names for tolerance

| Solver | Parameter Name |
|:-------|:--------------|
| HJBFDMSolver | `newton_tolerance` |
| HJBSemiLagrangianSolver | `tolerance` (generic) |
| NetworkHJBSolver | `tolerance` (generic) |
| FPNetworkSolver | `tolerance` (generic) |

**Impact**: Low - context makes it clear, but could be more consistent

**Recommendation**: ⚠️ **LOW PRIORITY** - Document in style guide

---

### 6. ⚠️ **Parameter Name Prefix Inconsistency**

**Issue**: `HybridFPParticleHJBFDM` prefixes parameters with solver type

```python
# HybridFPParticleHJBFDM
hjb_newton_iterations   # Prefixed
hjb_newton_tolerance    # Prefixed
hjb_fd_scheme          # Prefixed

# Better approach (MonitoredParticleCollocationSolver)
max_newton_iterations  # Generic (applies to internal HJB solver)
newton_tolerance       # Generic
```

**Recommendation**: ⚠️ **MEDIUM PRIORITY** - Only prefix when necessary to avoid ambiguity

---

## Recommended Fixes

### Priority 1: Critical API Consistency

**Fix 1: Rename `mfg_problem` → `problem` in HybridFPParticleHJBFDM**

```python
class HybridFPParticleHJBFDM(BaseMFGSolver):
    def __init__(
        self,
        problem: MFGProblem,  # RENAMED from mfg_problem
        num_particles: int = 5000,
        # ... rest of parameters

        # Deprecated parameter for backward compatibility
        mfg_problem: MFGProblem | None = None,
    ):
        import warnings

        # Handle deprecation
        if mfg_problem is not None:
            warnings.warn(
                "Parameter 'mfg_problem' is deprecated. Use 'problem' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if problem is None:
                problem = mfg_problem

        if problem is None:
            raise ValueError("'problem' parameter is required")

        super().__init__(problem)
```

**Fix 2: Rename Newton parameters in HybridFPParticleHJBFDM**

```python
def __init__(
    self,
    problem: MFGProblem,
    num_particles: int = 5000,
    kde_bandwidth: str | float = "scott",
    max_newton_iterations: int = 30,  # RENAMED from hjb_newton_iterations
    newton_tolerance: float = 1e-7,   # RENAMED from hjb_newton_tolerance
    damping_parameter: float = 0.5,
    config: MFGSolverConfig | None = None,

    # Deprecated parameters
    hjb_newton_iterations: int | None = None,
    hjb_newton_tolerance: float | None = None,
    hjb_fd_scheme: str | None = None,  # Remove entirely (not used in HJBFDMSolver)
    **kwargs: Any,
):
```

### Priority 2: Optional Backend Support

**Add `backend` parameter to all solvers that don't have it**:

```python
# Template for solvers missing backend
def __init__(
    self,
    problem: MFGProblem,
    # ... existing parameters ...
    backend: str | None = None,  # ADD THIS
):
    super().__init__(problem)

    # Initialize backend
    from mfg_pde.backends import create_backend
    self.backend = create_backend(backend) if backend else create_backend("numpy")
```

**Applies to**:
- HJBGFDMSolver
- HJBWenoSolver
- NetworkHJBSolver
- FPFDMSolver
- FPNetworkSolver

---

## API Consistency Checklist

**Standard Parameters** (all solvers should follow this pattern):

### Base Parameters
- ✅ **`problem`** - The MFG problem instance (NOT `mfg_problem`)
- ✅ **`backend`** - Optional backend selection ("numpy", "jax", "numba")

### Solver-Specific Parameters
- ✅ **`max_iterations`** or **`max_newton_iterations`** - Never solver-prefixed
- ✅ **`tolerance`** or **`newton_tolerance`** - Never solver-prefixed
- ✅ **`boundary_conditions`** - For solvers that support it

### Composition Parameters
- ✅ **`hjb_solver`** - For MFG solvers that compose sub-solvers
- ✅ **`fp_solver`** - For MFG solvers that compose sub-solvers

### Avoid
- ❌ Solver-specific prefixes on generic parameters (`hjb_newton_iterations`)
- ❌ Non-standard problem naming (`mfg_problem`)
- ❌ Inconsistent tolerance names across similar solvers

---

## Summary Statistics

| Metric | Count | Percentage |
|:-------|:------|:-----------|
| **Total Solvers Audited** | 14 | 100% |
| **Solvers with Deprecated Params** | 2 | 14% |
| **Solvers with `backend` Support** | 4 | 29% |
| **Solvers with Standard `problem` Param** | 13 | 93% |
| **Solvers with API Issues** | 1 | 7% |

**Critical Fix Required**: 1 solver (`HybridFPParticleHJBFDM`)

---

## Implementation Plan

### Phase 1: Critical Fixes (Immediate)
1. Fix `HybridFPParticleHJBFDM` parameter naming
   - `mfg_problem` → `problem` (with deprecation)
   - `hjb_newton_iterations` → `max_newton_iterations` (with deprecation)
   - Remove `hjb_fd_scheme` (not used in modern HJBFDMSolver)

2. Add tests for deprecated parameter warnings

### Phase 2: Optional Backend Support (Medium Term)
1. Add `backend` parameter to remaining 10 solvers
2. Update backend initialization pattern
3. Test backend switching for all solvers

### Phase 3: Documentation (Low Priority)
1. Create API style guide
2. Update all docstrings to show standard parameter names
3. Add examples showing consistent API usage

---

## Conclusion

The numerical solver API is **mostly consistent** with one critical outlier (`HybridFPParticleHJBFDM`).

**Immediate Action Required**:
- Fix HybridFPParticleHJBFDM to align with standard API conventions
- This will improve factory compatibility and user experience

**Overall Consistency Score**: 6.5/10 → Target: 9/10 after fixes
