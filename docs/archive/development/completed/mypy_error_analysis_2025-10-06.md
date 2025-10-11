# MyPy Error Analysis - Post Cleanup

**Date**: October 6, 2025
**MyPy Version**: Checked with mypy 1.5+
**Total Errors**: 375 (down from 423, 11.3% reduction)
**Files Affected**: 74 files

---

## Error Distribution by Type

| Error Code | Count | Description | Difficulty |
|:-----------|------:|:------------|:-----------|
| `assignment` | 73 | Type mismatches in assignments | Medium-Hard |
| `arg-type` | 67 | Argument type mismatches | Medium |
| `no-untyped-def` | 66 | Functions missing type annotations | **Easy** ⭐ |
| `int` | 40 | Integer type issues | Medium |
| `var-annotated` | 28 | Variables needing type annotations | **Easy** ⭐ |
| `attr-defined` | 28 | Attribute access on undefined types | Hard |
| `import-not-found` | 26 | Missing module imports | Config issue |
| `complex` | 23 | Complex number type issues | Medium |
| `no-redef` | 15 | Redefinition errors | Easy |
| `misc` | 14 | Miscellaneous type errors | Varies |
| `operator` | 10 | Operator type mismatches | Medium |
| `call-overload` | 9 | Function overload issues | Hard |
| `return-value` | 8 | Return type mismatches | Medium |
| `index` | 8 | Index type issues | Medium |
| `dict-item` | 8 | Dictionary item type issues | Medium |
| Other | <5 each | Various | Varies |

**Total**: 375 errors

---

## Quick Win Opportunities ⭐

### 1. **Add Type Annotations** (66 + 28 = 94 errors)
**Effort**: Medium (2-4 hours)
**Impact**: 25% error reduction

**Breakdown**:
- `no-untyped-def`: 66 functions missing annotations
- `var-annotated`: 28 variables needing annotations

**Example fixes**:
```python
# Before
def create_default_config(rows, cols, algorithm):
    return MazeConfig(rows=rows, cols=cols, algorithm=algorithm)

# After
def create_default_config(
    rows: int,
    cols: int,
    algorithm: Literal["recursive_backtracking", "wilsons"]
) -> MazeConfig:
    return MazeConfig(rows=rows, cols=cols, algorithm=algorithm)
```

**Files likely affected**:
- `mfg_pde/alg/reinforcement/environments/maze_config.py`
- Various utility modules
- Helper functions throughout codebase

### 2. **Fix Redefinition Errors** (15 errors)
**Effort**: Low (30 minutes)
**Impact**: 4% error reduction

**Example**:
```python
# Before
from typing import Any
Any = 5  # Redefinition!

# After
from typing import Any as AnyType
value: AnyType = 5
```

### 3. **Configure Import Exclusions** (26 errors)
**Effort**: Low (15 minutes)
**Impact**: 7% error reduction (if valid exclusions)

**Action**: Review `import-not-found` errors and add to mypy config exclusions if intentional.

---

## Complex Issues (Defer to Later)

### 1. **Assignment Type Mismatches** (73 errors)
**Problem**: Incompatible types in variable assignments
**Root cause**: Often `Union` types or polymorphic return values
**Solution**: Requires architectural refactoring (SolverResult standardization)

### 2. **Argument Type Mismatches** (67 errors)
**Problem**: Function calls with wrong argument types
**Root cause**: Overloaded functions, optional dependencies, generic types
**Solution**: Use `@overload` decorators or stricter type bounds

### 3. **Numeric Type Issues** (40 + 23 + 3 = 66 errors)
**Problem**: `int` vs `float` vs `complex` type confusion
**Root cause**: NumPy array operations returning different numeric types
**Solution**: Explicit type casts or using `Number` union types

### 4. **Attribute Errors** (28 errors)
**Problem**: Accessing attributes not defined on types
**Root cause**: Duck typing, optional dependencies, incomplete stubs
**Solution**: Use `hasattr()` checks or `Protocol` classes

---

## Recommended Action Plan

### **Phase 1: Low-Hanging Fruit** (This Week)
**Target**: Reduce errors by ~120 (32% reduction)

1. ✅ ~~Fix unused type ignores~~ (COMPLETED: 48 errors reduced)
2. **Add missing type annotations**: 94 errors
   - Functions: 66 errors (`no-untyped-def`)
   - Variables: 28 errors (`var-annotated`)
3. **Fix redefinition errors**: 15 errors
4. **Configure import exclusions**: 26 errors (if appropriate)

**Expected result**: 375 → ~255 errors

### **Phase 2: Architectural Improvements** (Next 2 Weeks)
**Target**: Reduce errors by ~140 (37% remaining)

1. **Standardize SolverResult type**: ~80 assignment errors
2. **Fix argument type mismatches**: ~60 arg-type errors
3. **Add Protocol classes for duck typing**: ~28 attr-defined errors

**Expected result**: 255 → ~115 errors

### **Phase 3: Refinement** (Next Month)
**Target**: Get below 50 errors

1. **Numeric type standardization**: 66 errors
2. **Overload decorators for polymorphic functions**: 9 errors
3. **Return value type fixes**: 8 errors
4. **Remaining miscellaneous fixes**: ~32 errors

**Expected result**: 115 → <50 errors

---

## Current Status

✅ **Completed** (October 6, 2025):
- Baseline established: 423 errors
- Unused ignores removed: 48 errors reduced
- Current state: **375 errors in 74 files**

🎯 **Next Target**:
- Add type annotations to reduce errors to ~255
- Estimated effort: 2-4 hours
- Estimated completion: This week

📊 **Long-term Goal**:
- Target: <50 mypy errors by end of month
- Strategy: Phased approach (low-hanging fruit → architecture → refinement)
- Compliance: Eventually aim for `--strict` mode compatibility

---

## Files Requiring Most Attention

Based on error distribution, these modules likely have the most type issues:

1. **Reinforcement learning environments** - Many untyped helper functions
2. **Numerical solvers** - Complex type relationships
3. **Factory modules** - Polymorphic return types
4. **Utility modules** - Generic helper functions
5. **Neural network modules** - PyTorch/JAX type stubs incomplete

---

## Tools and Automation

**Current approach**:
```bash
# Full analysis
make type-check

# Specific file
mypy mfg_pde/path/to/file.py --show-error-codes

# Error distribution
mypy mfg_pde/ --show-error-codes 2>&1 | grep -o '\[[a-z-]*\]' | sort | uniq -c | sort -rn
```

**Possible automation**:
- Create script to automatically add basic type annotations (mypy reveal_type hints)
- Use `monkeytype` or `pyannotate` for runtime type inference
- Set up incremental mypy checking in CI

---

**Last Updated**: October 6, 2025
**Next Review**: After Phase 1 completion
