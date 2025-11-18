# Legacy 1D API Deprecation Plan

**Status**: Phase 1 In Progress
**Target Removal**: v1.0.0
**Current Version**: v0.12.6
**Created**: 2025-11-19

## Executive Summary

Gradually deprecate the legacy 1D API (`xmin`, `xmax`, `Nx`) in favor of the unified geometry-based API. This migration improves code maintainability, eliminates API duplication, and provides a consistent interface across 1D, 2D, 3D, and graph-based problems.

---

## Background

### Legacy API (Deprecated)
```python
from mfg_pde import MFGProblem

# Old way: manual grid construction
problem = MFGProblem(
    xmin=0.0, xmax=1.0, Nx=50,  # 1D-specific parameters
    T=1.0, Nt=100,
    sigma=0.1
)
```

### Modern API (Recommended)
```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import SimpleGrid1D

# New way: geometry-first
domain = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions='periodic')
domain.create_grid(num_points=51)  # Nx+1 for consistency

problem = MFGProblem(
    geometry=domain,
    T=1.0, Nt=100,
    sigma=0.1
)
```

---

## Motivation

### Problems with Legacy API

1. **API Duplication**: Two ways to specify grids (legacy 1D vs geometry)
2. **Code Complexity**: Every solver must handle both APIs via `get_spatial_grid()`
3. **Inconsistent Semantics**: `Nx` means different things (points vs intervals)
4. **Extension Difficulty**: Adding new geometry types requires dual API support
5. **Maintenance Burden**: ~100+ locations check for `hasattr(problem, 'xmin')`

### Benefits of Geometry-First API

1. **Unified Interface**: Same API for 1D, 2D, 3D, graphs, AMR
2. **Explicit Semantics**: `num_points` is unambiguous
3. **Rich Features**: Geometry objects provide coordinates, operators, BC handling
4. **Better Separation**: Problem = math, Geometry = discretization
5. **Extensibility**: Easy to add new geometry types without problem class changes

---

## Deprecation Timeline

### Phase 1: Soft Deprecation (v0.12.x - v0.13.x) ✅ **CURRENT**

**Status**: In progress (v0.12.6)
**Goal**: Warn users but maintain full backward compatibility

**Completed**:
- [x] Add `DeprecationWarning` for scalar `xmin`, `xmax`, `Nx` parameters
- [x] Add `DeprecationWarning` for manual grid construction in `MFGProblem`
- [x] Provide clear migration path in warning messages
- [x] Update `get_spatial_grid()` to handle both APIs

**Remaining**:
- [ ] Mark legacy parameters with `deprecated=True` in docstrings
- [ ] Add migration guide to documentation
- [ ] Update all examples to use geometry-based API
- [ ] Add "See Also" links from legacy API docs to modern API

### Phase 2: Hard Deprecation (v0.14.x - v0.15.x)

**Status**: Planned (target: Q2 2026)
**Goal**: Restrict legacy API usage, provide automatic migration tools

**Tasks**:
- [ ] Raise `FutureWarning` instead of `DeprecationWarning`
- [ ] Add `strict_mode` flag to disable legacy API (opt-in for testing)
- [ ] Create automated migration script: `python -m mfg_pde.migrate legacy_to_geometry`
- [ ] Update CI to test with `strict_mode=True` for new code
- [ ] Deprecate `get_spatial_grid()` fallback to legacy API

### Phase 3: Removal (v1.0.0)

**Status**: Target (Q4 2026 - Q1 2027)
**Goal**: Remove legacy API entirely

**Tasks**:
- [ ] Remove `xmin`, `xmax`, `Nx` parameters from `MFGProblem.__init__()`
- [ ] Remove legacy API handling from `get_spatial_grid()`
- [ ] Simplify `CoefficientField` to only use `problem.geometry.coordinates`
- [ ] Remove backward compatibility shims in all solvers
- [ ] Update all documentation to geometry-first API only

---

## Migration Strategy

### 1. Automated Detection

Add linter rule to detect legacy API usage:

```python
# scripts/detect_legacy_api.py
import ast

class LegacyAPIDetector(ast.NodeVisitor):
    """Detect MFGProblem(..., xmin=..., xmax=..., Nx=...) patterns."""

    def visit_Call(self, node):
        if self._is_mfg_problem_call(node):
            if self._has_legacy_params(node):
                self.report_legacy_usage(node)
```

### 2. Automated Migration Tool

```bash
# Auto-migrate files
python -m mfg_pde.migrate legacy_to_geometry examples/

# Preview changes without modifying
python -m mfg_pde.migrate legacy_to_geometry examples/ --dry-run
```

Migration script transforms:
```python
# Before
problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=100)

# After
from mfg_pde.geometry import SimpleGrid1D
domain = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions='periodic')
domain.create_grid(num_points=51)
problem = MFGProblem(geometry=domain, T=1.0, Nt=100)
```

### 3. Documentation Updates

**Migration Guide** (`docs/migration/LEGACY_TO_GEOMETRY_API.md`):
- Side-by-side examples (old vs new)
- Common pitfalls (Nx vs num_points)
- FAQ: "Why change?", "When is deadline?", "How to migrate?"

**Deprecation Notices**:
- Prominent banner in documentation homepage
- Warning boxes in all legacy API references
- Link to migration guide from warnings

### 4. Test Suite Strategy

**Dual Testing (Phase 1-2)**:
```python
@pytest.mark.parametrize("api_style", ["legacy", "geometry"])
def test_solver_with_both_apis(api_style):
    if api_style == "legacy":
        problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=50)
    else:
        domain = SimpleGrid1D(xmin=0.0, xmax=1.0)
        domain.create_grid(num_points=31)
        problem = MFGProblem(geometry=domain, T=1.0, Nt=50)

    solver = MySolver(problem)
    result = solver.solve()
    assert result.converged
```

**Geometry-Only Testing (Phase 3)**:
- Remove `api_style` parameterization
- All tests use geometry-based API only

---

## Code Locations Requiring Changes

### Core Problem Class

**File**: `mfg_pde/core/mfg_problem.py`

**Changes**:
```python
# Phase 1: Add deprecation warnings (✅ Done)
if xmin is not None and xmax is not None:
    warnings.warn(
        "Manual grid construction is deprecated. Use geometry-first API.",
        DeprecationWarning, stacklevel=2
    )

# Phase 2: Add strict mode
if strict_mode and (xmin is not None or xmax is not None):
    raise ValueError("Legacy API disabled. Use geometry=... parameter.")

# Phase 3: Remove parameters entirely
def __init__(self, geometry, T, Nt, sigma, ...):
    # No xmin, xmax, Nx parameters
```

### Utility Functions

**File**: `mfg_pde/utils/pde_coefficients.py`

**Current** (dual API support):
```python
def get_spatial_grid(problem):
    # Modern geometry-based API
    if hasattr(problem, "geometry") and hasattr(problem.geometry, "coordinates"):
        return problem.geometry.coordinates

    # Legacy 1D API (DEPRECATED)
    elif hasattr(problem, "xmin") and hasattr(problem, "xmax"):
        Nx = problem.Nx + 1
        return np.linspace(problem.xmin, problem.xmax, Nx)

    else:
        raise AttributeError("Problem must have geometry or (xmin, xmax, Nx)")
```

**Phase 3** (geometry-only):
```python
def get_spatial_grid(problem):
    """Get spatial grid coordinates from problem geometry."""
    if not hasattr(problem, "geometry"):
        raise AttributeError(
            "Problem must have 'geometry' attribute. "
            "Legacy xmin/xmax API was removed in v1.0.0. "
            "See docs/migration/LEGACY_TO_GEOMETRY_API.md"
        )
    return problem.geometry.coordinates
```

### Solver Updates

**Files**: All solvers in `mfg_pde/alg/numerical/{hjb,fp}_solvers/`

**Phase 1-2**: Use `get_spatial_grid()` helper (✅ Done)
**Phase 3**: Direct access to `problem.geometry.coordinates`

### Factory Functions

**File**: `mfg_pde/factory/general_mfg_factory.py`

**Current**:
```python
def create_mfg_problem(domain_spec, ...):
    if isinstance(domain_spec, dict) and 'xmin' in domain_spec:
        # Legacy API support
        return MFGProblem(**domain_spec)
    else:
        # Geometry-based API
        return MFGProblem(geometry=domain_spec, ...)
```

**Phase 3**: Remove legacy dict handling

---

## Examples Migration

### Priority Order

1. **High Priority** (User-facing tutorials):
   - `examples/basic/*.py`
   - `examples/tutorials/*.ipynb`
   - README examples

2. **Medium Priority** (Advanced examples):
   - `examples/advanced/*.py`
   - Documentation code snippets

3. **Low Priority** (Internal tests):
   - `tests/unit/*.py` (dual testing during Phase 1-2)
   - `tests/integration/*.py`

### Migration Script Usage

```bash
# Step 1: Detect legacy usage
python scripts/detect_legacy_api.py examples/

# Step 2: Preview migration
python -m mfg_pde.migrate legacy_to_geometry examples/ --dry-run

# Step 3: Apply migration
python -m mfg_pde.migrate legacy_to_geometry examples/

# Step 4: Verify tests still pass
pytest tests/
```

---

## Communication Plan

### Version 0.13.0 Release Notes (Target)

```markdown
## Deprecations

### Legacy 1D API (`xmin`, `xmax`, `Nx`)

The manual grid construction API is deprecated in favor of the unified
geometry-based API. The legacy API will be removed in v1.0.0.

**Old (deprecated)**:
```python
problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=100)
```

**New (recommended)**:
```python
from mfg_pde.geometry import SimpleGrid1D
domain = SimpleGrid1D(xmin=0.0, xmax=1.0)
domain.create_grid(num_points=51)
problem = MFGProblem(geometry=domain, T=1.0, Nt=100)
```

See `docs/migration/LEGACY_TO_GEOMETRY_API.md` for full migration guide.
```

### GitHub Discussions

- Create pinned issue: "Legacy API Deprecation Timeline"
- Tag with `breaking-change`, `deprecation`, `v1.0.0`
- Link to migration guide
- Provide space for user feedback

### Documentation Updates

- Add prominent banner to homepage
- Update all code examples
- Create dedicated migration guide
- Update API reference with deprecation notices

---

## Success Metrics

### Phase 1: Soft Deprecation (v0.12.x - v0.13.x)
- [x] All new examples use geometry-based API
- [ ] Documentation shows geometry-first as primary
- [ ] <10% of examples use legacy API
- [ ] Migration guide published

### Phase 2: Hard Deprecation (v0.14.x - v0.15.x)
- [ ] Automated migration tool available
- [ ] CI tests pass with `strict_mode=True`
- [ ] All maintained examples migrated
- [ ] <5% of user code uses legacy API (based on issue reports)

### Phase 3: Removal (v1.0.0)
- [ ] Legacy API code removed from codebase
- [ ] All tests use geometry-based API
- [ ] No backward compatibility shims remain
- [ ] Clean, unified API surface

---

## Risk Mitigation

### Breaking User Code

**Risk**: Users' existing code breaks when legacy API is removed

**Mitigation**:
1. Multi-version deprecation period (0.12 → 0.14 → 1.0)
2. Clear warnings with migration instructions
3. Automated migration tool
4. Comprehensive documentation
5. GitHub issue for user questions

### Migration Tool Bugs

**Risk**: Automated migration produces incorrect code

**Mitigation**:
1. Extensive test suite for migration script
2. Always run with `--dry-run` first
3. Preserve original files (backup or git)
4. Manual review of generated code
5. Report bugs via GitHub issues

### Documentation Drift

**Risk**: Documentation shows inconsistent API usage

**Mitigation**:
1. Linter to detect legacy API in docs
2. CI check for deprecated patterns
3. Quarterly documentation audit
4. Automated testing of documentation examples

---

## Open Questions

1. **Timing**: Should removal be delayed to v2.0.0 for more conservative timeline?
2. **Opt-in vs Opt-out**: Should Phase 2 require opt-in to strict mode, or opt-out?
3. **Factory Functions**: Keep factory functions supporting both APIs longer?
4. **Type Hints**: Should type hints explicitly exclude legacy parameters during Phase 2?

---

## References

- **Migration Guide**: `docs/migration/LEGACY_TO_GEOMETRY_API.md` (TODO)
- **Geometry API Docs**: `docs/user_guide/geometry.md`
- **Related Issues**:
  - Geometry consolidation: `docs/development/planning/GEOMETRY_CONSOLIDATION_QUICK_START.md`
  - Parameter migration: `docs/migration/GEOMETRY_PARAMETER_MIGRATION.md`

---

**Last Updated**: 2025-11-19
**Responsible**: Core maintainers
**Review Cycle**: Quarterly until v1.0.0 release
