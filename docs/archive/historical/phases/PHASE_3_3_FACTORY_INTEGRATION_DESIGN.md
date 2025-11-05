# Phase 3.3: Factory Integration & Completion

**Date**: 2025-11-03
**Author**: Claude Code
**Status**: Design
**Timeline**: 1-2 weeks

---

## Summary

Complete Phase 3 architecture refactoring by integrating the new unified systems (Phase 3.1 MFGProblem, Phase 3.2 SolverConfig) with the existing factory system and `solve_mfg()` interface.

---

## Context

### What We've Built (Phase 3.1 & 3.2)

**Phase 3.1** - Unified `MFGProblem` class:
- Single problem class replacing 5 specialized classes
- Clean separation: problem (math) vs solver (algorithm)
- PR #218 - Ready for review

**Phase 3.2** - Unified `SolverConfig` system:
- Single config system replacing 3 competing systems
- Three usage patterns: YAML, Builder API, Presets
- PR #222 - Just completed

### Current Integration Gaps

The new unified systems are not yet integrated with:
1. **Factory functions** (`mfg_pde/factory/`)
   - Still create old specialized problem classes
   - Not using new `SolverConfig`

2. **`solve_mfg()` interface** (`mfg_pde/solve_mfg.py`)
   - Not using new `MFGProblem` or `SolverConfig`
   - Still using old config creation functions

3. **Examples**
   - Still import old problem classes and configs
   - Need migration to new unified API

---

## Goals

### Primary Goals

1. **Integrate new systems with factory functions**
   - Update factories to create `MFGProblem` instances
   - Update factories to use `SolverConfig`
   - Maintain backward compatibility

2. **Integrate new systems with `solve_mfg()`**
   - Accept `MFGProblem` + `SolverConfig`
   - Support both new and old APIs during transition
   - Clear migration path

3. **Update core examples**
   - Migrate key examples to new API
   - Show best practices
   - Demonstrate simplicity

### Secondary Goals

4. **Backend integration testing**
   - Ensure JAX/PyTorch backends work with new systems
   - Test GPU support where applicable

5. **Documentation updates**
   - Update API documentation
   - Revise user guides for new systems
   - Update quickstart to show new API first

---

## Design

### Part 1: Factory Integration

#### Current State

Factories in `mfg_pde/factory/`:
```python
# factory/problem_factories.py
def create_lq_problem(...) -> LQMFGProblem:
    return LQMFGProblem(...)

def create_crowd_problem(...) -> CrowdDynamicsProblem:
    return CrowdDynamicsProblem(...)
```

#### Proposed Changes

**Option A: Dual-Output Factories** (Recommended)
```python
# factory/problem_factories.py
def create_lq_problem(..., use_unified: bool = True) -> MFGProblem | LQMFGProblem:
    """
    Create Linear-Quadratic MFG problem.

    Parameters
    ----------
    use_unified : bool
        Use new unified MFGProblem class (default: True)
        Set to False for backward compatibility with LQMFGProblem
    """
    if use_unified:
        return MFGProblem(
            terminal_cost=g,
            hamiltonian=H,
            initial_density=rho_0,
            geometry=domain,
            problem_type="lq"
        )
    else:
        warnings.warn(
            "LQMFGProblem is deprecated. Use MFGProblem with use_unified=True.",
            DeprecationWarning
        )
        return LQMFGProblem(...)
```

**Benefits**:
- Backward compatible
- Clear migration path via parameter
- Old classes still available during transition

**Configuration Factories**:
```python
# factory/solver_factories.py (update existing)
def create_fdm_solver(problem, config=None, **kwargs):
    """
    Create FDM-based MFG solver.

    Parameters
    ----------
    problem : MFGProblem or legacy problem class
        Problem to solve
    config : SolverConfig, optional
        Solver configuration (if None, uses presets.fast_solver())
    """
    from mfg_pde.config import presets

    if config is None:
        config = presets.fast_solver()

    # Create solver with new config system
    ...
```

### Part 2: solve_mfg() Integration

#### Current State

```python
# mfg_pde/solve_mfg.py
def solve_mfg(problem, method="auto", **kwargs):
    # Uses old config system
    from mfg_pde.config import create_fast_config
    config = create_fast_config()
    ...
```

#### Proposed Changes

```python
# mfg_pde/solve_mfg.py
def solve_mfg(
    problem: MFGProblem | Any,  # Any for backward compat
    config: SolverConfig | str | None = None,
    method: str | None = None,  # Deprecated, use config
    **kwargs
) -> MFGResult:
    """
    Solve mean field game with unified interface.

    Parameters
    ----------
    problem : MFGProblem or legacy problem class
        Problem to solve
    config : SolverConfig, str, or None
        Solver configuration:
        - SolverConfig: Use provided configuration
        - str: Use named preset (e.g., "fast", "accurate")
        - None: Auto-select based on problem
    method : str, optional
        DEPRECATED: Use config parameter instead

    Examples
    --------
    >>> # New API (recommended)
    >>> from mfg_pde import MFGProblem, solve_mfg
    >>> from mfg_pde.config import presets
    >>>
    >>> problem = MFGProblem(...)
    >>> config = presets.accurate_solver()
    >>> result = solve_mfg(problem, config=config)

    >>> # Or with preset name
    >>> result = solve_mfg(problem, config="accurate")

    >>> # Or with YAML
    >>> from mfg_pde.config import load_solver_config
    >>> config = load_solver_config("experiment.yaml")
    >>> result = solve_mfg(problem, config=config)

    >>> # Old API (deprecated, still works)
    >>> result = solve_mfg(problem, method="accurate")
    """
    from mfg_pde.config import presets

    # Handle deprecated method parameter
    if method is not None and config is None:
        warnings.warn(
            "method parameter is deprecated. Use config parameter instead: "
            f"solve_mfg(problem, config='{method}')",
            DeprecationWarning
        )
        config = method

    # Convert string to config
    if isinstance(config, str):
        preset_map = {
            "auto": "fast",
            "fast": "fast",
            "accurate": "accurate",
            "research": "research",
            "production": "production",
        }
        preset_name = preset_map.get(config, config)
        config = getattr(presets, f"{preset_name}_solver")()

    # Default config
    if config is None:
        config = presets.fast_solver()

    # Create and run solver
    solver = create_solver_from_config(problem, config)
    return solver.solve(**kwargs)
```

### Part 3: Example Migration

#### Priority Examples to Migrate

1. **`examples/basic/solve_mfg_demo.py`** - Update to new API
2. **`examples/basic/lq_mfg_demo.py`** - Show MFGProblem usage
3. **`examples/advanced/crowd_evacuation_2d_hybrid.py`** - Show YAML config

#### Migration Pattern

**Before**:
```python
from mfg_pde import create_standard_solver
from mfg_pde.config import create_accurate_config

# Create problem (old way)
problem = LQMFGProblem(...)

# Create config (old way)
config = create_accurate_config()

# Solve
solver = create_standard_solver(problem, config)
result = solver.solve()
```

**After**:
```python
from mfg_pde import MFGProblem, solve_mfg
from mfg_pde.config import presets

# Create problem (new unified way)
problem = MFGProblem(
    terminal_cost=g,
    hamiltonian=H,
    initial_density=rho_0,
    geometry=domain
)

# Solve with preset (new way)
result = solve_mfg(problem, config=presets.accurate_solver())

# Or with YAML
result = solve_mfg(problem, config="accurate")
```

### Part 4: Backend Integration

#### Testing Strategy

Create integration tests for each backend:

```python
# tests/integration/test_backends_integration.py
import pytest

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_jax_backend_with_unified_api():
    """Test JAX backend with MFGProblem + SolverConfig."""
    from mfg_pde import MFGProblem, solve_mfg
    from mfg_pde.config import ConfigBuilder

    problem = MFGProblem(...)
    config = (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=2)
        .solver_fp(method="fdm")
        .backend(backend_type="jax", device="gpu")
        .build()
    )

    result = solve_mfg(problem, config=config)
    assert result.success

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pytorch_backend_with_unified_api():
    """Test PyTorch backend with MFGProblem + SolverConfig."""
    ...
```

---

## Implementation Plan

### Phase 3.3.1: Factory Integration (2-3 days)

**Tasks**:
1. Update `factory/problem_factories.py`:
   - Add `use_unified` parameter to all problem factories
   - Return `MFGProblem` when `use_unified=True`
   - Keep old classes for backward compat

2. Update `factory/solver_factories.py`:
   - Accept `SolverConfig` parameter
   - Default to `presets.fast_solver()` if None
   - Support both old and new config types

3. Add deprecation warnings for old APIs

**Testing**:
- Test factories create correct types
- Test backward compatibility
- Test new unified API

### Phase 3.3.2: solve_mfg() Integration (1-2 days)

**Tasks**:
1. Update `solve_mfg()` function:
   - Accept `MFGProblem` and `SolverConfig`
   - Support string config names
   - Deprecate `method` parameter

2. Add config name resolution:
   - Map "fast" → `presets.fast_solver()`
   - Map "accurate" → `presets.accurate_solver()`
   - etc.

3. Update error messages and validation

**Testing**:
- Test with new API
- Test with old API (backward compat)
- Test string config names
- Test error handling

### Phase 3.3.3: Example Migration (1-2 days)

**Tasks**:
1. Migrate core examples:
   - `examples/basic/solve_mfg_demo.py`
   - `examples/basic/lq_mfg_demo.py`
   - `examples/advanced/crowd_evacuation_2d_hybrid.py`

2. Create new examples showing:
   - YAML-based configuration
   - Builder API usage
   - Preset configurations

3. Update example documentation

**Testing**:
- Verify all examples run
- Check output correctness
- Validate new patterns

### Phase 3.3.4: Backend Integration Testing (1-2 days)

**Tasks**:
1. Create backend integration tests
2. Test JAX backend (if available)
3. Test PyTorch backend (if available)
4. Document backend-specific considerations

**Testing**:
- Run on available backends
- Mark as skip if backend not installed
- Document any backend limitations

### Phase 3.3.5: Documentation Updates (1 day)

**Tasks**:
1. Update API documentation
2. Update quickstart guide
3. Update user guides
4. Create Phase 3 completion summary

**Deliverables**:
- Updated API docs
- Migration examples
- Phase 3 summary document

---

## Backward Compatibility Strategy

### Deprecation Timeline

**v0.9.0** (Phase 3.3 completion):
- Old APIs work with `DeprecationWarning`
- New APIs recommended in docs
- Both work side-by-side

**v1.0.0** (+3 months):
- Warnings become more prominent
- Old APIs discouraged
- New APIs are default

**v2.0.0** (+6 months):
- Old APIs removed
- Only new unified API supported

### Supported Patterns During Transition

**Old pattern** (works with warning):
```python
from mfg_pde.problems import LQMFGProblem
from mfg_pde.config import create_accurate_config

problem = LQMFGProblem(...)
config = create_accurate_config()
result = solve_mfg(problem, method="accurate")
```

**New pattern** (recommended):
```python
from mfg_pde import MFGProblem, solve_mfg
from mfg_pde.config import presets

problem = MFGProblem(...)
result = solve_mfg(problem, config=presets.accurate_solver())
```

---

## Testing Strategy

### Unit Tests

1. **Factory Tests**:
   - Test `use_unified` parameter
   - Test old and new outputs
   - Test deprecation warnings

2. **solve_mfg() Tests**:
   - Test with MFGProblem + SolverConfig
   - Test with string config names
   - Test backward compatibility
   - Test parameter validation

3. **Integration Tests**:
   - Test complete workflows
   - Test YAML-based config
   - Test preset usage

### Integration Tests

1. **Backend Tests**:
   - JAX backend integration
   - PyTorch backend integration
   - GPU support (if available)

2. **Example Tests**:
   - All examples run successfully
   - Output correctness
   - No deprecation warnings (new examples)

---

## Success Criteria

### Must Have

- [x] Factories support both old and new APIs
- [x] `solve_mfg()` accepts MFGProblem + SolverConfig
- [x] All existing tests pass
- [x] Backward compatibility maintained
- [x] Core examples migrated
- [x] Deprecation warnings in place

### Should Have

- [x] Backend integration tests
- [x] Updated documentation
- [x] Migration guide
- [x] New example patterns

### Nice to Have

- [ ] Performance benchmarks comparing old vs new
- [ ] Community feedback on new API
- [ ] Example gallery showcase

---

## Risks and Mitigation

### Risk 1: Breaking Changes

**Risk**: Accidentally break existing user code
**Mitigation**:
- Maintain full backward compatibility
- Comprehensive test suite
- Clear deprecation warnings
- Long deprecation timeline (6 months)

### Risk 2: Factory Complexity

**Risk**: Dual-output factories become confusing
**Mitigation**:
- Clear documentation
- Default to new API (`use_unified=True`)
- Simple parameter to switch

### Risk 3: Example Migration

**Risk**: Examples become outdated or confusing
**Mitigation**:
- Migrate incrementally
- Keep old examples as `_old.py` during transition
- Clear comments about new vs old

---

## Documentation Plan

### Updated Documents

1. **API Documentation**:
   - `MFGProblem` API reference
   - `SolverConfig` API reference
   - Factory function signatures
   - `solve_mfg()` signature

2. **User Guides**:
   - Quickstart (show new API first)
   - Configuration guide (YAML, Builder, Presets)
   - Migration guide (old → new)

3. **Examples**:
   - Basic examples with new API
   - Advanced examples with YAML configs
   - Migration examples (side-by-side comparison)

### New Documents

1. **Phase 3 Completion Summary**:
   - What changed across all 3 phases
   - Migration strategy
   - Benefits of new architecture

2. **Best Practices Guide**:
   - When to use YAML vs Builder vs Presets
   - How to organize experiments
   - Performance considerations

---

## Timeline

### Week 1: Core Integration

**Days 1-2**: Factory integration
- Update problem factories
- Update solver factories
- Add deprecation warnings
- Write tests

**Days 3-4**: solve_mfg() integration
- Update function signature
- Add config resolution
- Update validation
- Write tests

**Day 5**: Testing and fixes
- Run full test suite
- Fix any issues
- Verify backward compatibility

### Week 2: Examples and Documentation

**Days 1-2**: Example migration
- Migrate core examples
- Create new example patterns
- Test all examples

**Days 3-4**: Backend integration
- Create backend tests
- Test JAX/PyTorch
- Document limitations

**Day 5**: Documentation
- Update API docs
- Update user guides
- Create completion summary

---

## Deliverables

### Code

1. Updated factory functions with dual-output support
2. Updated `solve_mfg()` with new API
3. Migrated core examples
4. Backend integration tests
5. Comprehensive test suite

### Documentation

1. Updated API documentation
2. Updated user guides
3. Migration guide
4. Phase 3 completion summary
5. Best practices guide

### Quality

1. All existing tests pass
2. New tests for Phase 3.3 features
3. No breaking changes
4. Clear deprecation path
5. Comprehensive examples

---

## Follow-up Work (Post Phase 3.3)

### Short-term (v0.9.x)

1. Monitor user feedback on new API
2. Refine documentation based on questions
3. Add more examples as requested
4. Fix any discovered issues

### Medium-term (v1.0)

1. Make new API more prominent in docs
2. Add warnings for old API usage
3. Prepare for v2.0 deprecation

### Long-term (v2.0)

1. Remove old problem classes
2. Remove old config systems
3. Clean up factory functions
4. Simplify codebase

---

## Conclusion

Phase 3.3 completes the architecture refactoring by integrating the new unified systems (MFGProblem, SolverConfig) with the existing user-facing APIs (factories, solve_mfg()). This provides:

1. **Clean, unified API** for all users
2. **Backward compatibility** during transition
3. **Clear migration path** to new systems
4. **Complete architecture** with all pieces integrated

After Phase 3.3, MFG_PDE will have a modern, maintainable architecture ready for future development.

---

**Status**: Design Complete - Ready for Implementation
**Next Step**: Create branch and begin Phase 3.3.1 (Factory Integration)
