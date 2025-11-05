# Phase 3 Known Issues and Follow-Up Work

**Date**: 2025-11-03
**Status**: Post-Merge Documentation

---

## Overview

Phase 3 (3.1 + 3.2 + 3.3) architectural refactoring is merged and version 0.9.0 is released. However, several integration issues were discovered during testing that require follow-up work.

---

## Critical Issues

### 1. Solver-Config Integration Incomplete

**Issue**: The solver (`FixedPointIterator`) still expects old config fields that were removed in Phase 3.2.

**Error**:
```python
AttributeError: 'PicardConfig' object has no attribute 'damping_factor'
```

**Location**: `mfg_pde/alg/numerical/mfg_solvers/fixed_point_iterator.py:129`

**Cause**: Phase 3.2 removed `damping_factor` from `PicardConfig` but solver code wasn't updated.

**Impact**: HIGH - `solve_mfg()` cannot run with new config system

**Fix Required**:
1. Audit all solver code for old config field access
2. Update to use new Phase 3.2 config structure
3. Add compatibility layer if needed
4. Update all examples to use new API

**Estimated Effort**: 1-2 days

---

### 2. Factory Function Signatures Don't Match MFGProblem

**Issue**: Factory convenience functions use simplified signatures that don't match MFGProblem validation requirements.

**Expected (MFGProblem)**:
```python
def hamiltonian_func(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
    du_dx = derivs.get((1,), 0.0)
    return 0.5 * du_dx**2 + m_at_x
```

**Provided (factories)**:
```python
def hamiltonian(x, p, m, t):
    return 0.5 * p**2 + m
```

**Impact**: MEDIUM - Factory functions can't be used directly, limiting Phase 3.3 value

**Fix Required**:
1. Either remove simplified factories (current approach)
2. Or create signature adapter/wrapper layer
3. Update documentation to show correct usage
4. Create working factory examples

**Estimated Effort**: 2-3 days for option 2 (adapters)

**Current Workaround**: Use `ExampleMFGProblem` for demos, document complex signature

---

### 3. Domain API Changes Break Tests

**Issue**: Test fixtures use old `RectangularDomain` API that no longer exists.

**Error**:
```python
ImportError: cannot import name 'RectangularDomain' from 'mfg_pde.geometry'
```

**Current Domain API**: `Domain1D`, `Domain2D`, `Domain3D` with different signatures

**Impact**: LOW - Tests fail but core functionality works

**Fix Required**:
1. Update all test fixtures to use new Domain API
2. Fix `tests/unit/test_problem_factories.py`
3. Update any examples using old domain creation

**Estimated Effort**: 1 day

---

## Design Issues

### 4. Config Field Naming Inconsistency

**Issue**: Old config used `damping_factor`, new config removed it but solver still expects it.

**Root Cause**: Phase 3.2 config redesign didn't maintain all fields for backward compat

**Options**:
1. **Add back missing fields** to Phase 3.2 config (breaks clean design)
2. **Update all solvers** to use new config structure (more work, cleaner)
3. **Add compatibility layer** in solve_mfg() to bridge old/new (compromise)

**Recommendation**: Option 2 - Update solvers to use new config properly

---

### 5. Factory Pattern Complexity

**Issue**: MFGProblem signature requirements are too complex for simple factory usage.

**Root Cause**: MFGProblem designed for maximum flexibility, not ease of use

**Options**:
1. **Keep current complexity** and document well
2. **Add high-level builder** that handles signature wrapping
3. **Separate simple/advanced APIs** explicitly

**Recommendation**: Option 3 - Create `SimpleMFGProblem` for basic use cases

---

## Documentation Issues

### 6. Examples Don't Run

**Issue**: Both `solve_mfg_demo.py` and `factory_demo.py` fail due to config/signature mismatches.

**Impact**: HIGH - Users can't learn from examples

**Fix Required**:
1. Remove or fix `factory_demo.py` (removed in current commit)
2. Fix `solve_mfg_demo.py` to work with current API
3. Test all examples before release

**Status**: Partially fixed - factory_demo removed, solve_mfg_demo needs config fix

---

### 7. Migration Guide Incomplete

**Issue**: Migration guides show API changes but don't account for signature requirements.

**Fix Required**:
1. Update migration guides with full working examples
2. Show signature adaptation pattern
3. Document common pitfalls

**Estimated Effort**: 1 day

---

## Follow-Up Work Plan

### Phase 3.4: Integration Fixes (Priority: CRITICAL)

**Timeline**: 1 week

**Goals**:
1. Fix solver-config integration
2. Make solve_mfg() work with Phase 3.2 config
3. Fix or remove broken examples
4. Update tests for new Domain API

**Deliverables**:
- Working solve_mfg() with new config
- All examples run successfully
- Test suite passes
- Updated documentation

### Phase 3.5: Factory Improvements (Priority: HIGH)

**Timeline**: 1-2 weeks

**Goals**:
1. Add signature adapter layer
2. Create `SimpleMFGProblem` class
3. Restore working factory examples
4. Comprehensive factory documentation

**Deliverables**:
- Simple factory API for common cases
- Advanced factory API for complex cases
- Working factory_demo.py
- Factory best practices guide

---

## Temporary Workarounds

Until fixes are complete:

### For Users

**Use ExampleMFGProblem**:
```python
from mfg_pde import ExampleMFGProblem, solve_mfg

problem = ExampleMFGProblem()
# solve_mfg() still broken, use factory directly
from mfg_pde.factory import create_standard_solver
solver = create_standard_solver(problem)
result = solver.solve()
```

**For Custom Problems**:
See `examples/basic/custom_hamiltonian_derivs_demo.py` for correct signature pattern.

### For Developers

**Testing Config Changes**:
```python
from mfg_pde.config import presets
config = presets.fast_solver()
print(config.model_dump())  # Inspect actual fields
```

**Finding Old Config Usage**:
```bash
grep -r "damping_factor" mfg_pde/alg/
grep -r "\.picard\." mfg_pde/alg/
```

---

## Lessons Learned

1. **Test Integration Early**: Phase 3.2 and 3.3 should have been tested together before merge
2. **Update Dependents First**: Solvers should have been updated when config changed
3. **Example-Driven Development**: Examples should be part of definition of done
4. **Signature Complexity**: Consider ease of use vs flexibility trade-offs earlier

---

## Conclusion

Phase 3 architectural work is **structurally complete** but **functionally incomplete**. The new APIs exist but don't work end-to-end due to integration gaps.

**Recommendation**: Create Phase 3.4 (Integration Fixes) as highest priority follow-up work.

**Timeline**:
- Phase 3.4 (Fixes): 1 week
- Phase 3.5 (Improvements): 1-2 weeks
- Total: 2-3 weeks to fully functional

After Phase 3.5, MFG_PDE will have a complete, working, modern architecture.

---

**Status**: DOCUMENTED
**Next**: Create Phase 3.4 branch and begin integration fixes
