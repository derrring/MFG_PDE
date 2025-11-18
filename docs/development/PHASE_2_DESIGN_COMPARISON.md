# Phase 2 Design Document Comparison & Reconciliation

**Date**: 2025-11-19
**Documents Compared**:
- `PDE_COEFFICIENT_IMPLEMENTATION_ROADMAP.md` (Updated: 2025-11-18)
- `PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md` (Created: 2025-11-19)

---

## Executive Summary

Both documents describe Phase 2 implementation for state-dependent PDE coefficients. This comparison identifies:
- **Alignment**: Where documents agree
- **Gaps**: What one document has that the other lacks
- **Conflicts**: Where documents disagree
- **Recommendations**: How to reconcile and proceed

---

## 1. Document Scope Comparison

### Roadmap (`PDE_COEFFICIENT_IMPLEMENTATION_ROADMAP.md`)
**Scope**: Multi-phase strategic plan (Phase 1 → Phase 3+)
- ✅ High-level objectives and milestones
- ✅ Status tracking (Phase 1 complete)
- ✅ Brief implementation sketches
- ✅ Task checklists
- ❌ Lacks detailed algorithms
- ❌ Lacks comprehensive testing strategy
- ❌ Lacks performance analysis

### Design Doc (`PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md`)
**Scope**: Detailed Phase 2 technical specification
- ✅ Comprehensive implementation details
- ✅ Algorithm pseudocode with line references
- ✅ Testing strategy with specific test cases
- ✅ Performance analysis and optimization
- ✅ API design principles
- ✅ Mathematical formulations
- ✅ Example use cases
- ❌ Doesn't track Phase 1 status
- ❌ Less context on overall roadmap

**Verdict**: **Complementary, not conflicting**. Roadmap provides strategic context, Design Doc provides tactical details.

---

## 2. Phase 2.1: Array Diffusion in FP Solvers

### Alignment ✅

Both documents agree on:
- **Goal**: Support spatially/temporally varying diffusion in FP-FDM
- **Priority**: High
- **Effort**: 1 day
- **File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`
- **Approach**: Index into diffusion array per point `sigma[k,i]`
- **Key change**: Remove `NotImplementedError` for array `diffusion_field`

### Differences

| Aspect | Roadmap | Design Doc |
|:-------|:--------|:-----------|
| **Code examples** | 27 lines | 60+ lines (more detailed) |
| **Validation** | Mentioned | Explicit shape checking, NaN/Inf handling |
| **Edge cases** | Not covered | Table of 5 edge cases with handling |
| **nD extension** | Not mentioned | Covered with defer to Phase 2.4 |
| **Test cases** | 1 example | 5 comprehensive test cases |

**Example Gap**: Roadmap doesn't mention validation strategy.

**Design Doc Addition**:
```python
# Validate shape
expected_shape = (Nt, Nx)
if sigma.shape != expected_shape:
    raise ValueError(
        f"diffusion_field array shape {sigma.shape} doesn't match "
        f"expected shape {expected_shape} (Nt={Nt}, Nx={Nx})"
    )
```

### Recommendation
✅ **Use Design Doc implementation** (more complete)
✅ **Update Roadmap** to mark Phase 2.1 tasks with Design Doc reference

---

## 3. Phase 2.2: Callable Coefficient Evaluation

### Alignment ✅

Both documents agree on:
- **Goal**: Support state-dependent coefficients `D(t,x,m)` and `α(t,x,m)`
- **Priority**: High
- **Effort**: 2-3 days
- **Challenge**: Circular dependency (evaluating `D(m)` when solving for `m`)

### Major Difference: Evaluation Strategy

#### Roadmap Approach (Section 2.2, lines 226-261)
```python
def _evaluate_drift_callable(
    self,
    drift_func: DriftCallable,
    m_evolution: np.ndarray,  # (Nt+1, Nx)
) -> np.ndarray:
    """Evaluate ALL timesteps upfront using m_evolution."""
    drift_array = np.zeros((Nt, Nx))
    for t_idx in range(Nt):
        t = t_idx * self.problem.dt
        m_current = m_evolution[t_idx, :]
        drift_array[t_idx, :] = drift_func(t, x_grid, m_current)
    return drift_array
```

**Characteristics**:
- Evaluates callable **once** before timestepping
- Requires full `m_evolution` array as input
- Problem: Where does `m_evolution` come from? We're solving for it!

#### Design Doc Approach (Section 3.2, Strategy 1)
```python
def _solve_fp_1d_with_callable_diffusion(
    self,
    m_initial: np.ndarray,
    U_drift: np.ndarray,
    diffusion_func: DiffusionCallable,
) -> np.ndarray:
    """Evaluate callable ONCE PER TIMESTEP using computed m."""
    m_solution = np.zeros((Nt, Nx))
    m_solution[0, :] = m_initial

    for k in range(Nt - 1):
        # Evaluate diffusion at CURRENT density state
        t_current = k * self.problem.dt
        m_current = m_solution[k, :]  # ← Use computed m[k]

        sigma_array = diffusion_func(t_current, x_grid, m_current)

        # Solve one timestep
        m_solution[k + 1, :] = self._solve_single_timestep_fp(
            m_current, U_drift[k, :], sigma_array, ...
        )

    return m_solution
```

**Characteristics**:
- Evaluates callable **per timestep** during solve loop
- Uses `m[k]` to compute `m[k+1]` (bootstrap/causal)
- No circular dependency issue

### Analysis: Which Approach is Correct?

**Roadmap Approach Issues**:
1. ❌ **Circular Logic**: Needs `m_evolution` as input, but we're solving for `m`
2. ❌ **Won't work standalone**: Where does initial `m_evolution` come from?
3. ⚠️ **Only makes sense in MFG context**: Could use `M_old` from previous Picard iteration
4. ❌ **Not explained clearly**: Doesn't document where `m_evolution` comes from

**Design Doc Approach Benefits**:
1. ✅ **Causal**: Uses only past information (`m[k]` to get `m[k+1]`)
2. ✅ **Works standalone**: Doesn't require external `m` estimate
3. ✅ **Stable**: Forward-in-time evaluation (no backward dependency)
4. ✅ **Well-documented**: Clear explanation of bootstrap strategy

**Verdict**: **Design Doc approach is correct**. Roadmap approach has a conceptual error.

### Recommendation
❌ **Do NOT use Roadmap's `_evaluate_drift_callable()` as written**
✅ **Use Design Doc's bootstrap strategy** (evaluate per timestep)
✅ **Update Roadmap** to fix circular dependency issue

---

## 4. MFG Coupling Integration

### Roadmap Coverage
- ✅ Mentions coupling in "Challenges" (line 265-267)
- ❌ No dedicated section for coupling integration
- ❌ No code examples for `FixedPointIterator`

### Design Doc Coverage (Section 4, 40+ lines)
- ✅ Dedicated Phase 2.3 section
- ✅ Detailed `FixedPointIterator` modifications
- ✅ Shows how to re-evaluate callables each Picard iteration
- ✅ Usage examples
- ✅ Test strategy for MFG with callables

**Example from Design Doc**:
```python
class FixedPointIterator:
    def __init__(self, ..., diffusion_field=None):
        self.diffusion_field_override = diffusion_field

    def solve(self):
        for picard_iter in range(max_iterations):
            # Re-evaluate callable with current M_old
            if callable(self.diffusion_field_override):
                diffusion_for_hjb = self._evaluate_diffusion_for_hjb(
                    self.diffusion_field_override, M_old
                )
            else:
                diffusion_for_hjb = self.diffusion_field_override

            U_new = self.hjb_solver.solve_hjb_system(..., diffusion_field=diffusion_for_hjb)
            M_new = self.fp_solver.solve_fp_system(..., diffusion_field=self.diffusion_field_override)
```

### Recommendation
✅ **Design Doc is more complete** - has detailed MFG coupling strategy
✅ **Add to Roadmap**: New section 2.4 referencing Design Doc

---

## 5. Testing Strategy

### Roadmap (lines 279-296)
- Basic test case examples (3 functions)
- No test structure
- No validation tests
- No performance benchmarks

### Design Doc (Section 5, 100+ lines)
- Test hierarchy (unit, integration, benchmarks)
- Test matrix (scalar/array/callable × multiple aspects)
- 15+ specific test cases with code
- Validation strategy (shape, NaN/Inf, mass conservation)
- Performance benchmarks with analytical solutions

**Example Gap**:

Roadmap test:
```python
def porous_medium_diffusion(t, x, m):
    return 0.1 * m
```

Design Doc test:
```python
def test_porous_medium():
    """Test nonlinear diffusion D(m) = σ² m."""
    def porous(t, x, m):
        return 0.1 * m

    M = solver.solve_fp_system(m0, U, diffusion_field=porous)

    # Verify: High-density regions diffuse faster
    # Verify: Compact support (characteristic of porous medium)
    # Verify: Self-similar solution (compare to Barenblatt)
    assert ...  # Specific checks
```

### Recommendation
✅ **Use Design Doc testing strategy** (comprehensive)
✅ **Update Roadmap** to reference Design Doc Section 5 for test details

---

## 6. Key Gaps & Additions

### Design Doc Has (Roadmap Lacks)

| Section | Content | Value |
|:--------|:--------|:------|
| **Section 3.3** | Refactor `_solve_single_timestep_fp()` | Essential for code reuse |
| **Section 3.4** | HJB callable implementation | Complete Phase 2.2 |
| **Section 4** | MFG coupling modifications | Integration strategy |
| **Section 6** | Performance analysis | 10-20% overhead quantified |
| **Section 7** | API design principles | Progressive disclosure, fail fast |
| **Appendix A** | Mathematical formulations | Discretization details |
| **Appendix B** | 3 complete example use cases | Porous medium, crowd, temperature |
| **Appendix C** | Known limitations & Phase 3 | Clear scope boundaries |

### Roadmap Has (Design Doc Lacks)

| Section | Content | Value |
|:--------|:--------|:------|
| **Phase 1 Summary** | Completion status | Strategic context |
| **Phase 2.3** | Complete nD Support | Future work |
| **Phase 3** | Advanced features | Long-term vision |
| **Commit tracking** | Git history | Implementation progress |

### Recommendation
✅ **Keep both documents** - they serve different purposes:
- **Roadmap**: Strategic planning, status tracking, high-level
- **Design Doc**: Tactical implementation, detailed algorithms, testing

✅ **Cross-reference**: Add links between documents

---

## 7. Conflicts & Inconsistencies

### Conflict 1: Callable Evaluation Strategy ⚠️

**Issue**: Roadmap's `_evaluate_drift_callable()` has circular dependency.

**Resolution**:
```diff
# Roadmap (INCORRECT)
- def _evaluate_drift_callable(drift_func, m_evolution):  # Where does m_evolution come from?
-     for t_idx in range(Nt):
-         drift_array[t_idx, :] = drift_func(t, x_grid, m_evolution[t_idx, :])

# Design Doc (CORRECT)
+ def _solve_fp_1d_with_callable(m_initial, drift_func):
+     for k in range(Nt - 1):
+         m_current = m_solution[k, :]  # Use computed m[k]
+         drift_at_k = drift_func(t, x_grid, m_current)
+         m_solution[k + 1, :] = solve_timestep(m_current, drift_at_k, ...)
```

**Action**: Update Roadmap Section 2.2 to use Design Doc's bootstrap approach.

### Conflict 2: Section Numbering

**Roadmap**:
- 2.1: Array Diffusion FP
- 2.2: Callable Evaluation FP
- 2.3: Complete nD Support (different topic!)

**Design Doc**:
- 2.1: Array Diffusion FP
- 2.2: Callable Evaluation (FP + HJB)
- 2.3: MFG Coupling Integration
- 2.4: Complete nD Support (deferred)

**Resolution**: Design Doc numbering is more logical (couples callable FP+HJB+MFG together).

**Action**: Renumber Roadmap to match Design Doc structure.

---

## 8. Reconciliation Plan

### Option A: Merge into Single Document (❌ Not Recommended)

**Pros**: Single source of truth
**Cons**:
- Loses strategic context (Phase 1 complete, Phase 3 future)
- Design Doc is already 1,380 lines (too long for roadmap)
- Mixing high-level planning with detailed implementation

### Option B: Keep Both, Add Cross-References (✅ Recommended)

**Structure**:
```
PDE_COEFFICIENT_IMPLEMENTATION_ROADMAP.md  (Strategic)
├── Phase 1: Complete ✅
├── Phase 2: See PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md
│   ├── 2.1: Array Diffusion (→ Design Doc Section 2)
│   ├── 2.2: Callable Evaluation (→ Design Doc Section 3)
│   ├── 2.3: MFG Coupling (→ Design Doc Section 4)
│   └── 2.4: nD Support (deferred)
└── Phase 3: Future Work

PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md  (Tactical)
├── Comprehensive implementation details
├── Testing strategy
├── Performance analysis
└── See Roadmap for Phase 1 context and Phase 3 future work
```

### Recommended Updates

#### Update 1: Fix Roadmap Section 2.2 (Callable Evaluation)

```diff
### 2.2: Callable Evaluation in FP Solvers

**Priority**: High
**Estimated Effort**: Medium (2-3 days)

+ **See**: `PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md` Section 3 for detailed design.

#### Implementation Plan

- **File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

- Add callable evaluation in `_solve_fp_1d()`:
+ **Strategy**: Bootstrap evaluation (evaluate callable per timestep using m[k] to compute m[k+1])

```python
- def _evaluate_drift_callable(
-     self,
-     drift_func: DriftCallable,
-     m_evolution: np.ndarray,  # (Nt+1, Nx) ← Where does this come from?
- ) -> np.ndarray:

+ def _solve_fp_1d_with_callable(
+     self,
+     m_initial: np.ndarray,
+     U_drift: np.ndarray,
+     diffusion_func: DiffusionCallable,
+ ) -> np.ndarray:
+     """Solve FP with callable diffusion (bootstrap strategy)."""
+     m_solution = np.zeros((Nt, Nx))
+     m_solution[0, :] = m_initial
+
+     for k in range(Nt - 1):
+         # Evaluate diffusion at CURRENT state
+         t_current = k * self.problem.dt
+         m_current = m_solution[k, :]
+         sigma_array = diffusion_func(t_current, x_grid, m_current)
+
+         # Solve single timestep
+         m_solution[k + 1, :] = self._solve_single_timestep_fp(...)
```

#### Update 2: Add MFG Coupling Section to Roadmap

```diff
+ ### 2.3: MFG Coupling Integration
+
+ **Priority**: High
+ **Estimated Effort**: Small (1 day)
+
+ **See**: `PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md` Section 4 for detailed design.
+
+ #### Overview
+
+ Integrate callable coefficients into `FixedPointIterator`:
+ - Pass `diffusion_field` parameter through to HJB/FP solvers
+ - Re-evaluate callables each Picard iteration (necessary for state-dependence)
+ - Support both drift and diffusion overrides
+
+ #### Tasks
+
+ - [ ] Add `diffusion_field` parameter to `FixedPointIterator.__init__()`
+ - [ ] Implement `_evaluate_diffusion_for_hjb()` helper
+ - [ ] Re-evaluate callable with `M_old` each Picard iteration
+ - [ ] Pass evaluated diffusion to HJB solver
+ - [ ] Pass callable diffusion to FP solver (FP evaluates per timestep)
+ - [ ] Add integration tests for MFG with callable coefficients

- ### 2.3: Complete nD Support
+ ### 2.4: Complete nD Support
```

#### Update 3: Add Cross-References

**In Roadmap** (top of Phase 2):
```markdown
## Phase 2: State-Dependent & nD (🔄 NEXT)

**Detailed Design**: See `PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md` for:
- Comprehensive implementation algorithms
- Testing strategy and test cases
- Performance analysis
- API design principles
- Mathematical formulations
- Example use cases
```

**In Design Doc** (Section 1.1):
```markdown
### 1.1 Current State (Phase 1)

**See**: `PDE_COEFFICIENT_IMPLEMENTATION_ROADMAP.md` for:
- Phase 1 completion status and achievements
- Overall strategic roadmap (Phase 1 → Phase 3)
- Future work and Phase 3 objectives
```

---

## 9. Action Items

### Immediate (Before Implementation)

- [ ] **Update Roadmap Section 2.2**: Fix callable evaluation strategy (use bootstrap)
- [ ] **Add Roadmap Section 2.3**: MFG coupling integration
- [ ] **Renumber Roadmap 2.3 → 2.4**: nD support
- [ ] **Add cross-references**: Link between documents
- [ ] **Review updated Roadmap**: Verify consistency

### During Implementation

- [ ] **Use Design Doc as primary reference**: More detailed and correct
- [ ] **Update Roadmap task checkboxes**: Track completion
- [ ] **Document deviations**: If implementation differs from design

### After Phase 2 Complete

- [ ] **Update Roadmap Phase 2 status**: Mark completed sections
- [ ] **Create Phase 2 retrospective**: Lessons learned
- [ ] **Archive Design Doc**: Mark as [IMPLEMENTED] in Phase 3

---

## 10. Summary & Recommendation

### Key Findings

1. ✅ **Documents are complementary**, not conflicting (strategic vs tactical)
2. ⚠️ **Roadmap has conceptual error** in Section 2.2 (circular dependency)
3. ✅ **Design Doc is more complete** (testing, performance, MFG coupling)
4. ✅ **Both are valuable** for different purposes

### Recommended Approach

**DO**:
- ✅ Keep both documents
- ✅ Use Design Doc as implementation reference (more detailed)
- ✅ Use Roadmap for status tracking and strategic context
- ✅ Add cross-references between documents
- ✅ Fix Roadmap's callable evaluation strategy

**DON'T**:
- ❌ Merge into single document (loses clarity)
- ❌ Discard either document (both serve purpose)
- ❌ Implement Roadmap's `_evaluate_drift_callable()` as written (has bug)

### Confidence Assessment

| Aspect | Confidence | Notes |
|:-------|:-----------|:------|
| **Phase 2.1** | ✅ High | Both documents align, Design Doc more complete |
| **Phase 2.2** | ⚠️ Medium | Roadmap needs fix, Design Doc correct |
| **Phase 2.3** | ✅ High | Design Doc comprehensive, Roadmap missing |
| **Overall Strategy** | ✅ High | Clear path forward with reconciliation |

---

## Conclusion

The two documents work well together:
- **Roadmap**: "What" and "Why" (strategic planning)
- **Design Doc**: "How" and "When" (tactical implementation)

**Next Step**: Update Roadmap (fix Section 2.2, add Section 2.3, cross-reference), then proceed with Phase 2.1 implementation using Design Doc as primary reference.

**Estimated Time**: 30 minutes to reconcile documents, then ready for implementation.
