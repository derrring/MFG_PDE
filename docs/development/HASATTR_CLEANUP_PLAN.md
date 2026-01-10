# hasattr() Cleanup Plan

**Status**: Partial Completion
**Created**: 2025-01-06
**Last Updated**: 2026-01-11
**Total Violations**: 420 (original)
**Completed**: 79 protocol duck typing violations (Issue #543, 96% reduction)
**Remaining**: ~341 violations (caching, legacy compatibility, other patterns)

## Problem

The codebase has 420 `hasattr()` usages which violate the fail-fast principle by:
- Enabling duck typing that hides bugs
- Allowing silent fallbacks when attributes are missing
- Making code flow difficult to trace

## Top Offenders

| File | Count | Pattern |
|------|-------|---------|
| hjb_gfdm.py | 26 | Caching, interface detection |
| backends/compat.py | 21 | Legacy compatibility |
| hjb_semi_lagrangian.py | 16 | Interface detection |
| collocation.py | 13 | Feature detection |
| fp_particle.py | 13 | Caching |
| fixed_point_iterator.py | 13 | Interface detection |

## Patterns and Fixes

### Pattern 1: Caching (Easy)
```python
# BAD
if hasattr(self, "_cache"):
    return self._cache

# GOOD
if self._cache is not None:  # Initialize _cache = None in __init__
    return self._cache
```

### Pattern 2: Interface Detection (Medium)
```python
# BAD
if hasattr(obj, "get_bounds"):
    bounds = obj.get_bounds()

# GOOD
from mfg_pde.geometry.protocol import GeometryProtocol
if isinstance(obj, GeometryProtocol):
    bounds = obj.get_bounds()
```

### Pattern 3: Legacy Overrides (Remove with v0.17.0)
```python
# BAD - remove entirely
if hasattr(self, "_Lx_override"):
    return self._Lx_override
```

## Phase Plan

1. **Phase 1 (v0.16.15)**: Fix caching patterns in HJB/FP solvers
2. **Phase 2 (v0.17.0)**: Update interface checks to use Protocols
3. **Phase 3 (v0.17.0)**: Remove legacy override attributes

## Completed Work

**Issue #543** (Protocol Duck Typing Elimination) - ✅ **COMPLETED 2026-01-10**
- Scope: Core + Geometry modules
- Violations: 79 → 3 (96% reduction)
- Patterns: GeometryProtocol isinstance checks, try/except for optional attributes
- Documentation: `docs/archive/issue_543_hasattr_elimination_2026-01/`
- Pull Requests: #551, #552, #553, #554

## Remaining Work

The remaining ~341 violations fall into categories not addressed by Issue #543:
1. **Caching patterns** (~50 violations) - Should use `self._cache is not None` pattern
2. **Legacy compatibility** (~30 violations in backends/compat.py) - Remove with v0.17.0
3. **Solver interface detection** (~40 violations) - Need solver protocols
4. **Other patterns** (~221 violations) - Case-by-case evaluation needed

## Verification

Run: `python scripts/check_fail_fast.py --path mfg_pde`
