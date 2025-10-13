# Grid Convention Analysis: Nx/Nt Uniformity Study

**Date**: 2025-10-13
**Status**: 🔍 ANALYSIS COMPLETE
**Priority**: CRITICAL
**Issue**: Inconsistent grid convention between solvers

## Executive Summary

The MFG_PDE codebase has **two conflicting grid conventions**:

1. **Standard Convention** (98% of codebase): `Nx` = number of intervals, `Nx+1` = number of grid points
2. **WENO Exception**: `Nx` = number of grid points directly (no `+1`)

**Recommendation**: **Standardize on Convention 1** (current standard) and fix WENO solver to match.

## Core Problem Statement

When a user creates `MFGProblem(Nx=50)`, they specify:
- **50 intervals** in space
- **51 grid points** (including endpoints)

However, the WENO solver interprets this as:
- **50 grid points** (not 51)

This causes **shape mismatches** and **inconsistent discretizations** between solvers.

## Detailed Analysis

### Convention 1: Standard (Majority of Codebase)

**Definition**: `Nx` = number of intervals, yielding `Nx+1` knots/grid points

**Core Implementation** (`mfg_pde/core/mfg_problem.py:103-114`):
```python
self.Nx: int = Nx                    # Number of intervals
self.Dx: float = (xmax - xmin) / Nx  # Interval spacing
self.Nt: int = Nt                    # Number of time intervals
self.Dt: float = T / Nt              # Time step

# Grid points: Nx+1 points (including endpoints)
self.xSpace: np.ndarray = np.linspace(xmin, xmax, Nx + 1, endpoint=True)
self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

# All arrays initialized with Nx+1 points
self.f_potential = np.zeros(self.Nx + 1)  # Line 161
self.u_fin = np.zeros(self.Nx + 1)        # Line 169
self.m_init = np.zeros(self.Nx + 1)       # Line 177
```

**Rationale**:
- Natural discretization: domain [xmin, xmax] divided into Nx equal intervals
- Standard finite difference notation: h = (b-a)/N
- Consistent with numerical analysis textbooks
- Matches spatial discretization Dx = L/Nx

### Convention 2: WENO Exception

**Definition**: `Nx` = number of grid points directly

**Implementation** (`mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py:196-197`):
```python
self.Nx = self.problem.Nx  # Directly copies Nx from problem
# But then uses it as number of grid points:
U_solved = np.zeros((Nt + 1, Nx))  # Line 575 - Nx not Nx+1
```

**Problem**: When `MFGProblem` has `Nx=50` (50 intervals → 51 points), WENO only creates 50 points.

### Solvers Using Convention 1 (Standard)

| Solver | File | Shape Convention |
|:-------|:-----|:----------------|
| **HJBFDMSolver** | `hjb_fdm.py` | Arrays: `(Nt, Nx)` means `(Nt, Nx)` in docstring but code uses `Nx+1` |
| **FPFDMSolver** | `fp_fdm.py:26-27` | `Nx = self.problem.Nx + 1` explicitly |
| **FPParticleSolver** | `fp_particle.py` | Uses `problem.Nx + 1` for array sizes |
| **NetworkHJBSolver** | `hjb_network.py` | `num_nodes` (analogous to `Nx+1`) |
| **NetworkFPSolver** | `fp_network.py` | `num_nodes` (analogous to `Nx+1`) |
| **HJBSemiLagrangian** | `hjb_semi_lagrangian.py` | Uses `Nx+1` convention |

### Solver Using Convention 2 (Exception)

| Solver | File | Shape Convention |
|:-------|:-----|:----------------|
| **HJBWenoSolver** | `hjb_weno.py:197, 575` | `self.Nx = problem.Nx` (no `+1`) |

## Impact Analysis

### Shape Mismatches

When `MFGProblem(Nx=50, Nt=50)` is created:

**Standard Solvers**:
- Create arrays of shape `(51, 51)` (time × space)
- 51 grid points in space, 51 time points

**WENO Solver**:
- Creates arrays of shape `(51, 50)` (time × space)
- Only 50 grid points in space, 51 time points

### Test Implications

**Phase 2.2a Tests** (recently added):
```python
# test_weno_family.py - Had to be corrected during Phase 2.2a
def test_solve_hjb_system_shape(self, integration_problem):
    solver = HJBWenoSolver(integration_problem, weno_variant="weno5")

    Nt = integration_problem.Nt + 1
    Nx = integration_problem.Nx  # WENO uses Nx, not Nx+1 ⚠️

    M_density = np.ones((Nt, Nx))
    U_final = np.zeros(Nx)

    U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

    assert U_solution.shape == (Nt, Nx)  # (51, 50) not (51, 51)
```

**All Other Solver Tests**:
```python
# Standard pattern used in all other tests
def test_solve_hjb_system_shape(self):
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30)
    solver = HJBFDMSolver(problem)

    Nt = problem.Nt + 1  # 31 time points
    Nx = problem.Nx + 1  # 31 spatial points

    M_density = np.ones((Nt, Nx))  # (31, 31)
    U_final = np.zeros(Nx)          # (31,)

    U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

    assert U_solution.shape == (Nt, Nx)  # (31, 31)
```

## Mathematical Consistency

### Finite Difference Discretization

**Standard Approach** (textbook):
- Domain: $[a, b]$
- Intervals: $N$ equal intervals
- Grid points: $N+1$ points $\{x_0, x_1, \ldots, x_N\}$
- Spacing: $h = (b-a)/N$
- Points: $x_i = a + ih$ for $i = 0, 1, \ldots, N$

**MFG_PDE Implementation**:
```python
Nx = 50                               # N intervals
Dx = (xmax - xmin) / Nx               # h = (b-a)/N
xSpace = np.linspace(xmin, xmax, Nx + 1)  # N+1 points
```

✅ **Mathematically consistent with standard discretization**

### WENO Solver Interpretation

WENO appears to use `Nx` as "number of cells" but doesn't account for endpoints:

```python
self.Nx = problem.Nx  # Copies Nx (50 intervals)
# But interprets as 50 grid points, not 51
```

❌ **Not consistent with standard FD discretization**

## Recommendation: Unify on Convention 1

### Why Convention 1 (Standard)?

1. **98% of codebase already uses it**
   - All FDM, particle, network, semi-Lagrangian solvers
   - All problem definitions
   - All examples and tests

2. **Mathematically standard**
   - Matches finite difference textbooks
   - Consistent with `Dx = (xmax - xmin) / Nx`
   - Natural interpretation: Nx intervals → Nx+1 points

3. **Minimal disruption**
   - Only WENO solver needs modification
   - All other solvers continue working
   - Most tests already use correct convention

4. **User intuition**
   - When user says `Nx=50`, they specify domain resolution (50 intervals)
   - Getting 51 grid points (including endpoints) is natural

### Implementation Plan

#### Step 1: Fix WENO Solver (Critical)

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`

**Changes Required**:

```python
# Line 197: BEFORE
self.Nx = self.problem.Nx

# Line 197: AFTER
self.Nx = self.problem.Nx + 1  # Grid points (intervals + 1)

# Line 575: BEFORE (implicit)
U_solved = np.zeros((Nt + 1, Nx))  # Nx is now Nx+1 from problem

# Line 575: AFTER (stays same, but Nx now means Nx+1)
U_solved = np.zeros((Nt + 1, Nx))  # Now correctly (Nt+1, Nx+1)
```

**Additional locations to verify** (all occurrences of `self.Nx` in WENO):
- Line 571: `Nx = self.Nx` - will now be correct
- Line 212, 219: 2D/3D cases - also need `+ 1`
- Anywhere `self.Nx` is used for array indexing

#### Step 2: Update WENO Tests

**File**: `tests/unit/test_weno_family.py`

**Changes Required**:

```python
# BEFORE (11 integration tests added in Phase 2.2a)
Nx = integration_problem.Nx  # WENO uses Nx, not Nx+1

# AFTER
Nx = integration_problem.Nx + 1  # Now consistent with standard
```

**Impact**: All 11 integration tests need this one-line change

#### Step 3: Documentation Updates

Update docstrings in `hjb_weno.py` to clarify:
```python
def solve_hjb_system(self, M_density, U_final, U_prev):
    """
    Args:
        M_density (np.ndarray): Shape (Nt+1, Nx+1) density evolution
        U_final (np.ndarray): Shape (Nx+1,) final condition
        U_prev (np.ndarray): Shape (Nt+1, Nx+1) previous iteration

    Returns:
        np.ndarray: Shape (Nt+1, Nx+1) value function solution
    """
```

#### Step 4: Add Consistency Tests

Create `tests/unit/test_grid_convention.py`:

```python
def test_grid_convention_consistency():
    """Test that all solvers use consistent Nx convention."""
    problem = MFGProblem(Nx=50, Nt=50)

    # All solvers should create (51, 51) shaped arrays
    expected_space_points = 51
    expected_time_points = 51

    solvers = [
        HJBFDMSolver(problem),
        HJBWenoSolver(problem),
        HJBSemiLagrangianSolver(problem),
        # ... etc
    ]

    for solver in solvers:
        # Verify spatial dimension interpretation
        if hasattr(solver, 'Nx'):
            assert solver.Nx == expected_space_points
```

### Migration Timeline

**Immediate (This Session)**:
- Document the issue ✅ (this file)
- Create GitHub issue for tracking

**Next Session**:
- Fix WENO solver implementation
- Update WENO tests
- Run full test suite to verify

**Future**:
- Add grid convention consistency tests
- Update documentation

## Related Issues

### GitHub Issues

**To Create**:
- Issue #XXX: "Unify grid convention: WENO solver uses Nx instead of Nx+1"
  - Labels: `bug`, `solver`, `architecture`, `priority: high`
  - Milestone: Phase 2.3 (before geometry tests)

### Documentation Updates

**Files to Update**:
1. `docs/development/CONSISTENCY_GUIDE.md` - Add grid convention section
2. `CLAUDE.md` - Add note about Nx convention
3. Solver docstrings - Clarify array shapes

## Testing Strategy

### Validation Tests

After implementing the fix:

1. **Unit Tests**:
   - Verify WENO arrays are (Nt+1, Nx+1)
   - Test with various Nx values (10, 50, 100)
   - Ensure no off-by-one errors

2. **Integration Tests**:
   - Run existing integration tests
   - Verify mass conservation still works
   - Check solution accuracy vs analytical

3. **Cross-Solver Consistency**:
   - Compare WENO vs FDM on same problem
   - Verify identical array shapes
   - Check numerical convergence rates

### Regression Prevention

Add to CI/CD:
```python
@pytest.mark.parametrize("Nx", [10, 50, 100])
def test_solver_grid_consistency(Nx):
    """Ensure all solvers interpret Nx identically."""
    problem = MFGProblem(Nx=Nx, Nt=20)
    expected_points = Nx + 1

    # Test each solver
    assert HJBFDMSolver(problem).problem.xSpace.shape[0] == expected_points
    assert HJBWenoSolver(problem).Nx == expected_points
    # ... etc
```

## Conclusion

**Current State**: Inconsistent grid convention with WENO as exception

**Target State**: Unified convention where `Nx` = number of intervals, yielding `Nx+1` grid points

**Action Required**: Fix WENO solver to match standard convention

**Priority**: HIGH - Must be done before Phase 2.3 geometry tests to avoid propagating incorrect patterns

**Estimated Effort**: 1-2 hours
- 30 min: WENO solver fix
- 30 min: Test updates
- 30 min: Verification and testing

---

**Next Steps**:
1. Create GitHub issue
2. Implement WENO fix
3. Update tests
4. Verify with full test suite
5. Document in CLAUDE.md
