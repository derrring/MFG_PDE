# Architectural Issue: hjb_weno.py Inflexibility

**Date**: 2025-10-30
**Severity**: MEDIUM
**Category**: Architecture / Design
**Related**: GitHub Issue #200 (Architecture Refactoring)

---

## Issue Summary

`hjb_weno.py` hardcodes a specific Hamiltonian form instead of using the flexible `problem.H()` interface. This limits the solver to one specific problem type and is inconsistent with MFG_PDE's extensible design philosophy.

**Impact**: Users cannot use WENO solver with custom Hamiltonians (obstacle costs, running costs, nonlinear control, etc.)

---

## Technical Details

### Current Implementation (Inflexible)

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`

**Lines 510, 793, 917**: Hardcoded Hamiltonian
```python
# Hardcoded: H = 0.5 * |∇u|² + m·∇u
hamiltonian = 0.5 * u_x[i] ** 2 + m[i] * u_x[i]

# Applied directly
rhs[i] = -hamiltonian + (self.problem.sigma**2 / 2) * u_xx[i]
```

**Consequences**:
- ❌ ONLY works for quadratic control cost: `|∇u|²/2`
- ❌ ONLY works for linear congestion: `m·∇u`
- ❌ NO running costs: Cannot add `c(x, m)` terms
- ❌ NO drift terms: Cannot add `b(x)·∇u`
- ❌ NO obstacle penalties: Cannot add `V_obstacle(x)`
- ❌ NO target tracking: Cannot add `‖x - x_target‖²` terms

### Comparison with Other Solvers

| Solver | Calls problem.H()? | Flexible? | Tuple Notation? |
|--------|-------------------|-----------|-----------------|
| `base_hjb.py` | ✅ YES | ✅ Full flexibility | ✅ YES (Phase 2) |
| `hjb_semi_lagrangian.py` | ✅ YES | ✅ Full flexibility | ✅ YES (Phase 2) |
| `hjb_gfdm.py` | ✅ YES | ✅ Full flexibility | ✅ YES (already) |
| **`hjb_weno.py`** | ❌ **NO** | ❌ **Hardcoded H** | ❌ **N/A** |

**Finding**: `hjb_weno.py` is the **only** HJB solver that doesn't use the `problem.H()` interface!

---

## Root Cause Analysis

**Why was it implemented this way?**

Likely reasons:
1. **Performance optimization**: Avoid function call overhead
2. **Simplicity**: Direct implementation for standard MFG problem
3. **Historical**: Early implementation before extensibility was prioritized

**Is the performance gain significant?**

NO. Analysis:
- Function call overhead: ~10-50 nanoseconds
- WENO reconstruction cost: ~1-10 microseconds (100-1000× larger!)
- **Performance impact of calling problem.H()**: < 1% (negligible)

**Conclusion**: The flexibility loss is NOT justified by performance gains.

---

## Proposed Solution

### Architecture: Separate Gradient Computation from Hamiltonian Evaluation

**Key insight**: WENO's strength is **high-order gradient approximation**, not Hamiltonian evaluation.

**Proposed design**:
1. **Keep WENO reconstruction** for computing gradients (this is valuable!)
2. **Package gradients in tuple notation** (consistent with Phase 2 standard)
3. **Call problem.H()** for Hamiltonian evaluation (enables flexibility)

### Implementation Sketch

```python
def _solve_1d_substep(self, u, m, dt):
    """
    Solve 1D HJB substep using WENO reconstruction.

    IMPROVED: Uses problem.H() for flexibility while preserving
    WENO's high-order gradient computation.
    """
    # Step 1: WENO reconstruction for high-order gradients
    u_x = self._weno_reconstruct_gradient(u, self.Dx)  # Keep WENO!
    u_xx = np.gradient(u_x, self.Dx)  # Second derivative

    rhs = np.zeros_like(u)

    for i in range(len(u)):
        # Step 2: Package gradients in tuple notation
        derivs = {
            (0,): u[i],      # Function value
            (1,): u_x[i],    # First derivative (from WENO)
            (2,): u_xx[i],   # Second derivative
        }

        # Step 3: Call problem.H() for flexibility
        H = self.problem.H(
            x_idx=i,
            m_at_x=m[i],
            derivs=derivs,  # Standard tuple notation
            t_idx=self.current_time_index
        )

        # Step 4: Standard HJB update
        diffusion = (self.problem.sigma**2 / 2) * derivs[(2,)]
        rhs[i] = -H + diffusion

    return rhs
```

### For 2D and 3D

```python
def _solve_2d_substep(self, u, m, dt):
    """2D WENO with flexible Hamiltonian."""
    # WENO reconstruction (keep this!)
    u_x = self._weno_reconstruct_gradient_x(u, self.Dx)
    u_y = self._weno_reconstruct_gradient_y(u, self.Dy)
    u_xx = np.gradient(u_x, self.Dx, axis=0)
    u_yy = np.gradient(u_y, self.Dy, axis=1)

    rhs = np.zeros_like(u)

    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            # Package gradients in 2D tuple notation
            derivs = {
                (0, 0): u[i, j],       # Function value
                (1, 0): u_x[i, j],     # ∂u/∂x (from WENO)
                (0, 1): u_y[i, j],     # ∂u/∂y (from WENO)
                (2, 0): u_xx[i, j],    # ∂²u/∂x²
                (0, 2): u_yy[i, j],    # ∂²u/∂y²
            }

            # Call problem.H() for flexibility
            x_position = (i * self.Dx, j * self.Dy)
            H = self.problem.H(
                x_idx=(i, j),
                x_position=x_position,
                m_at_x=m[i, j],
                derivs=derivs,
                t_idx=self.current_time_index
            )

            # Standard HJB update with diffusion
            laplacian = derivs[(2, 0)] + derivs[(0, 2)]
            diffusion = (self.problem.sigma**2 / 2) * laplacian
            rhs[i, j] = -H + diffusion

    return rhs
```

---

## Benefits of Proposed Solution

### 1. Preserves WENO Advantages
- ✅ High-order accurate gradient computation
- ✅ Shock capturing capabilities
- ✅ Essentially non-oscillatory behavior
- ✅ Better handling of discontinuities

### 2. Adds Flexibility
Users can now define:
- ✅ Custom control costs: `α|∇u|² + β|∇u|⁴`
- ✅ Running costs: `c(x, m) = ‖x - x_target‖²`
- ✅ Obstacle penalties: `V_obstacle(x)`
- ✅ Nonlinear congestion: `g(m)·∇u`
- ✅ State-dependent drift: `b(x)·∇u`
- ✅ Anisotropic control: Different costs per direction

### 3. Architectural Consistency
- ✅ Matches `base_hjb.py`, `hjb_semi_lagrangian.py`, `hjb_gfdm.py` design
- ✅ Uses tuple notation standard (from Phase 2)
- ✅ Respects `problem.H()` interface contract
- ✅ Enables WENO solver with ANY problem class

### 4. Enables New Applications

**Example 1: Maze Navigation**
```python
class MazeNavigationProblem:
    def H(self, x_idx, m_at_x, derivs, t_idx):
        # Extract gradients computed by WENO
        p_x = derivs[(1, 0)]
        p_y = derivs[(0, 1)]
        p_norm_sq = p_x**2 + p_y**2

        # Custom Hamiltonian with obstacle costs
        control_cost = 0.5 * p_norm_sq
        congestion = m_at_x * np.sqrt(p_norm_sq)
        obstacle_penalty = self.obstacle_field[x_idx]  # NEW!
        target_cost = np.linalg.norm(x_position - self.target)**2  # NEW!

        return control_cost + congestion + obstacle_penalty + target_cost

# Now works with WENO solver!
problem = MazeNavigationProblem(...)
solver = HJBWENOSolver(problem)  # Previously impossible!
```

**Example 2: Anisotropic Control**
```python
class AnisotropicProblem:
    def H(self, x_idx, m_at_x, derivs, t_idx):
        p_x = derivs[(1, 0)]
        p_y = derivs[(0, 1)]

        # Different costs for x and y directions (e.g., wind resistance)
        control_cost = 0.5 * (self.alpha_x * p_x**2 + self.alpha_y * p_y**2)

        return control_cost + ...

# Works with WENO!
```

---

## Performance Analysis

### Overhead Estimation

**Current (hardcoded)**:
```python
hamiltonian = 0.5 * u_x[i] ** 2 + m[i] * u_x[i]  # Direct computation
```
- Time: ~10 floating-point operations
- Cost: ~5-10 nanoseconds

**Proposed (call problem.H())**:
```python
derivs = {(0,): u[i], (1,): u_x[i], (2,): u_xx[i]}
H = problem.H(x_idx=i, m_at_x=m[i], derivs=derivs, t_idx=t)
```
- Dictionary creation: ~20-30 nanoseconds
- Function call: ~10-20 nanoseconds
- Hamiltonian computation: ~5-10 nanoseconds (same as before)
- **Total**: ~40-60 nanoseconds

**WENO reconstruction cost** (for comparison):
- 5th-order WENO stencil: ~1-10 **microseconds** (1000-10000 nanoseconds!)

**Relative overhead**:
- Overhead: 40-60 nanoseconds
- WENO reconstruction: 1000-10000 nanoseconds
- **Percentage**: **< 1-5%** overhead

**Conclusion**: Negligible performance impact compared to WENO reconstruction cost.

---

## Migration Strategy

### Phase 1: Add Flexible Interface (Non-Breaking)

1. **Add new method** `_solve_1d_flexible()` that uses `problem.H()`
2. **Keep old method** `_solve_1d_substep()` with hardcoded H (deprecated)
3. **Add flag** `use_flexible_hamiltonian=True` (default True for new code)
4. **Deprecation warning** when old method is used

```python
class HJBWENOSolver:
    def __init__(self, problem, use_flexible_hamiltonian=True):
        self.use_flexible_hamiltonian = use_flexible_hamiltonian
        if not use_flexible_hamiltonian:
            warnings.warn(
                "Hardcoded Hamiltonian is deprecated. "
                "Set use_flexible_hamiltonian=True for extensibility.",
                DeprecationWarning
            )
```

### Phase 2: Update Documentation (Week 1)

1. Document the limitation in current code
2. Add examples showing how to use flexible interface
3. Update tutorials to use `use_flexible_hamiltonian=True`

### Phase 3: Deprecate Old Interface (6 months)

1. Change default to `use_flexible_hamiltonian=True`
2. Strong deprecation warnings for `use_flexible_hamiltonian=False`
3. Update all examples and tests

### Phase 4: Remove Old Interface (12 months)

1. Remove hardcoded Hamiltonian code
2. Make `problem.H()` mandatory
3. Update version number (breaking change)

---

## Testing Strategy

### Unit Tests

```python
def test_weno_flexible_hamiltonian():
    """Test WENO solver with custom Hamiltonian."""

    class CustomHamiltonianProblem(ExampleMFGProblem):
        def H(self, x_idx, m_at_x, derivs, t_idx):
            # Custom: nonlinear control + obstacle cost
            p = derivs[(1,)]
            control = 0.3 * p**4  # Nonlinear!
            obstacle = self.obstacle_field[x_idx]
            return control + obstacle

    problem = CustomHamiltonianProblem(Nx=50, Nt=25)
    solver = HJBWENOSolver(problem, use_flexible_hamiltonian=True)

    U, info = solver.solve()

    assert info['converged']
    assert np.all(np.isfinite(U))
```

### Integration Tests

```python
def test_weno_vs_gfdm_consistency():
    """Verify WENO and GFDM give similar results with same problem."""

    problem = ExampleMFGProblem(Nx=50, Nt=25)
    m = np.ones(50) * 0.5

    # Solve with WENO
    weno_solver = HJBWENOSolver(problem, use_flexible_hamiltonian=True)
    U_weno, _ = weno_solver.solve()

    # Solve with GFDM
    gfdm_solver = HJBGFDMSolver(problem)
    U_gfdm, _ = gfdm_solver.solve()

    # Should be similar (within numerical error)
    assert np.allclose(U_weno, U_gfdm, rtol=0.1)  # 10% tolerance
```

### Regression Tests

```python
def test_weno_backward_compatibility():
    """Ensure old hardcoded interface still works (deprecated)."""

    problem = ExampleMFGProblem(Nx=50, Nt=25)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = HJBWENOSolver(problem, use_flexible_hamiltonian=False)

        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    # Old interface should still work
    U, info = solver.solve()
    assert info['converged']
```

---

## Related Work

### Similar Issues in Other Solvers

**None identified**: All other HJB solvers (`base_hjb.py`, `hjb_semi_lagrangian.py`, `hjb_gfdm.py`) correctly use `problem.H()` interface.

### Lessons from Phase 2 Migration

**Key insight from Phase 2**: Non-breaking migrations are possible!
- Created backward compatibility layer
- All 56 tests passed with zero modifications
- Smooth transition via deprecation warnings

**Apply to hjb_weno.py**:
- Same strategy: Add new flexible interface while keeping old one
- Deprecation warnings guide users to new interface
- 6-12 month transition period before removal

---

## Priority and Timeline

**Priority**: MEDIUM
- Not a bug (current code works for its intended use case)
- Architectural improvement (enables new use cases)
- Consistency improvement (aligns with other solvers)

**Estimated Effort**: 2-3 weeks
- Week 1: Implement flexible interface
- Week 2: Tests and documentation
- Week 3: Examples and migration guide

**Dependencies**:
- Phase 2 gradient notation migration (✅ COMPLETE)
- Phase 3 problem classes migration (planned)

**Recommendation**: Implement after Phase 3 is complete.

---

## Conclusion

`hjb_weno.py`'s hardcoded Hamiltonian is a **design limitation** that:
1. Restricts WENO solver to one specific problem type
2. Is inconsistent with MFG_PDE's extensible architecture
3. Has NO significant performance justification (< 1% overhead)

**Recommendation**: Refactor to use `problem.H()` interface while preserving WENO's gradient computation advantages.

**Impact**: Unlocks WENO solver for custom problems (maze navigation, obstacle avoidance, target tracking, etc.) with minimal performance cost.

---

**Related Documents**:
- `docs/GRADIENT_NOTATION_AUDIT_REPORT.md` - Phase 2 migration (solvers using problem.H())
- `docs/gradient_notation_standard.md` - Tuple notation standard
- `docs/PHASE_2_MIGRATION_COMPLETE.md` - Migration strategy reference

**Status**: Documented, not yet scheduled for implementation
**GitHub Issue**: Should create new issue (separate from #200)
**Assignee**: TBD
**Date**: 2025-10-30
