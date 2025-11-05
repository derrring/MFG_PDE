# Solver Tier Elimination Summary

**Date**: 2025-11-03
**Status**: Design docs fully updated, user docs need updating
**Decision**: Eliminate "Solver Tier" terminology and prioritize modular approach over generic factories

---

## Rationale

"Solver Tiers" (Tier 1/2/3) create confusion by conflating:
1. **Method choice** (FDM, hybrid, WENO) with **quality levels**
2. **Factory convenience functions** with **algorithmic quality**
3. **Configuration** (resolution, tolerance) with **method type**

**Reality**: Quality depends on configuration (grid resolution, tolerance, damping), not on arbitrary "tier" labels.

---

## Changes Made

### ✅ Updated Design Documents

1. **`docs/development/design/PROGRESSIVE_DISCLOSURE_API_DESIGN.md`** - **FULLY UPDATED**
   - Removed all "Tier 1/2/3" references
   - Replaced all "Factory Mode" language with "Modular Approach" (recommended) and "Domain Templates" (future)
   - Updated Executive Summary to prioritize modular approach
   - Replaced factory examples with modular composition examples
   - Added domain template examples (crowd motion, epidemic, finance, traffic)
   - Added "Legacy: Generic Factory Functions (Not Recommended)" section
   - Updated all tables and quick references
   - Updated implementation status and next steps

2. **`docs/development/design/TWO_LEVEL_API_INTEGRITY_CHECK.md`**
   - Removed "Solver Tiers" section
   - Replaced with "Factory Functions" section
   - Emphasized quality depends on configuration, not function name

---

## User Documentation Needs Updating

The following user-facing docs still reference "Tier 1/2/3" and need updating:

### Critical (User-Facing)

1. **`docs/user/SOLVER_SELECTION_GUIDE.md`** - Entire structure based on tiers
2. **`docs/user/quickstart.md`** - Has "Three Solver Tiers" section
3. **`docs/user/core_objects.md`** - References "Tier 2" examples
4. **`docs/user/README.md`** - May have tier references

### Lower Priority (Archive/Historical)

- `docs/archive/**` - Keep as historical records
- `docs/development/planning/**` - Context-specific, may be okay

---

## Recommended Replacements

### OLD (Confusing)
```python
# Tier 1: Basic (poor quality, benchmark only)
solver = create_basic_solver(problem)

# Tier 2: Standard (good quality, DEFAULT)
solver = create_fast_solver(problem)

# Tier 3: Advanced (specialized)
solver = create_accurate_solver(problem, solver_type="weno")
```

### NEW (Clear)
```python
# FDM-only (simple, baseline for comparison)
solver = create_basic_solver(problem)

# Hybrid FDM+Particle (balanced speed/accuracy, DEFAULT)
solver = create_fast_solver(problem)

# High-order methods (WENO, Semi-Lagrangian, etc.)
solver = create_accurate_solver(problem, solver_type="weno")
```

---

## Key Messages for User Docs

1. **Factory functions are method shortcuts**, not quality levels
2. **Quality depends on configuration** (Nx, Nt, tolerance, damping_factor)
3. **Default is hybrid FDM+Particle** (`create_fast_solver`) - balanced
4. **For research**, use modular approach with explicit composition
5. **For production**, factory mode provides sensible defaults

---

## Next Steps

1. ✅ Design docs updated
2. ❌ **TODO**: Update `docs/user/SOLVER_SELECTION_GUIDE.md`
   - Restructure around "method choice" not "quality tiers"
   - Decision tree: use case → method, not tier
3. ❌ **TODO**: Update `docs/user/quickstart.md`
   - Remove "Three Solver Tiers" section
   - Replace with "Factory Functions" or "Method Selection"
4. ❌ **TODO**: Update `docs/user/core_objects.md`
   - Replace "Tier 2" with "hybrid" or "default configuration"
5. ❌ **TODO**: Review `docs/user/README.md`

---

## Implementation Notes

**Code changes NOT required** - Factory function names stay the same:
- `create_basic_solver()` - keeps name, just documented differently
- `create_fast_solver()` - keeps name, just documented differently
- `create_accurate_solver()` - keeps name, just documented differently

Only **documentation terminology** changes.

---

**Last Updated**: 2025-11-03
**Completed By**: Updated design docs, identified user docs needing work
**Status**: ⚠️ PARTIAL - Design docs done, user docs pending
