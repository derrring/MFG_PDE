# API Consistency Audit Report - 2025-10-10

**Issue**: #125 - API Consistency Audit: Standardize naming, parameters, and return types
**Status**: Phase 1-2 Complete (Discovery & Classification)
**Date**: 2025-10-10

---

## Executive Summary

Conducted systematic API audit using automated searches to identify inconsistencies in:
- Attribute naming (uppercase vs lowercase)
- Boolean parameter proliferation
- Tuple returns vs dataclasses
- Mixed naming conventions

### Key Finding: **1 Critical Bug Discovered and Fixed**

**Critical Bug**: `mfg_pde/hooks/visualization.py:183-195`
- **Issue**: Using lowercase `.u` and `.m` attributes when `SolverResult` uses uppercase `.U` and `.M`
- **Impact**: Runtime AttributeError when visualization hook is triggered
- **Status**: ✅ **FIXED** (commit pending)

---

## Phase 1: Discovery Results

### 1. Attribute Access Patterns
**Search**: `grep -r "result\.[a-zA-Z_]" tests/ mfg_pde/`
**Results**: 100+ matches (truncated at 100)

**Analysis**:
- Most code correctly uses `SolverResult` uppercase attributes (`.U`, `.M`, `.iterations`)
- Found 1 critical bug using lowercase (visualization hook)
- Pattern is generally consistent across codebase

### 2. Tuple Return Statements
**Search**: `grep -r "return.*,.*,.*," mfg_pde/`
**Results**: 100+ matches (truncated at 100)

**Analysis**:
- Most tuple returns are internal helper functions (acceptable)
- Visualization modules use tuple returns for internal methods (backend-specific implementations)
- Public API functions mostly use `SolverResult` dataclass

**Key Patterns**:
- ✅ **Good**: Visualization internal methods returning tuples (not user-facing)
- ✅ **Good**: Factory methods returning single object/tuple of objects
- ✅ **Good**: State accessors returning `(u, m)` tuples (small, clear semantics)

### 3. Boolean Parameter Pairs
**Search**: `grep -r ".*: bool.*# .*or" mfg_pde/`
**Results**: 9 matches

**Findings**:

#### **High Priority** (Candidates for Enum):
1. **Functional Calculus** (`utils/numerical/functional_calculus.py`, `utils/functional_calculus.py`):
   ```python
   use_jax: bool = False  # Use JAX for automatic differentiation
   use_pytorch: bool = False  # Use PyTorch for automatic differentiation
   ```
   **Recommendation**: Create `AutoDiffBackend` enum:
   ```python
   class AutoDiffBackend(str, Enum):
       NUMPY = "numpy"  # Default, no autodiff
       JAX = "jax"
       PYTORCH = "pytorch"
   ```

2. **DeepONet Normalization** (`alg/neural/operator_learning/deeponet.py`):
   ```python
   use_batch_norm: bool = False  # Use batch normalization
   use_layer_norm: bool = True  # Use layer normalization
   use_bias_net: bool = True  # Use bias network
   ```
   **Recommendation**: Create `NormalizationType` enum:
   ```python
   class NormalizationType(str, Enum):
       NONE = "none"
       BATCH = "batch"
       LAYER = "layer"
   ```
   Note: `use_bias_net` is independent, can stay boolean

#### **Medium Priority**:
3. **Sinkhorn Solver** (`alg/optimization/optimal_transport/sinkhorn_solver.py`):
   ```python
   log_domain: bool = True  # Use log-domain Sinkhorn for numerical stability
   ```
   **Recommendation**: Keep as boolean (single flag, clear semantics)

4. **MCMC** (`utils/numerical/mcmc.py`):
   ```python
   step_size_adaptation: bool = True  # Dual averaging for step size
   ```
   **Recommendation**: Keep as boolean (single flag)

### 4. Naming Inconsistencies (Nx/nx)
**Search**: `grep -rE "def.*\(.*[nN]x" mfg_pde/`
**Results**: 7 matches

**Findings**:

#### **Inconsistent Patterns**:
1. **MFGProblemBuilder** (`core/mfg_problem.py`):
   - Uses `Nx` (uppercase) - ✅ Follows standard

2. **Mathematical Notation** (`core/mathematical_notation.py`):
   - Uses `Nx, Nt` (uppercase) - ✅ Follows standard

3. **Modern Config** (`config/modern_config.py`):
   - Uses `nx, nt` (lowercase) - ❌ **Inconsistent**
   ```python
   def with_grid_size(self, nx: int, nt: int | None = None) -> SolverConfig:
   ```

4. **Mathematical DSL** (`meta/mathematical_dsl.py`):
   - Uses `nx, nt` (lowercase) - ❌ **Inconsistent**
   ```python
   def domain(self, xmin: float, xmax: float, tmax: float, nx: int = 100, nt: int = 50):
   ```

5. **CLI Utils** (`utils/cli.py`):
   - Uses `Nx` (uppercase) - ✅ Follows standard

6. **Performance Utils** (`utils/performance/optimization.py`):
   - Uses `nx, ny, nz` (lowercase) - ❌ **Inconsistent**

7. **Workflow Manager** (`workflow/workflow_manager.py`):
   - Uses `Nx, Nt` (uppercase) - ✅ Follows standard

**Recommendation**: Standardize on **uppercase `Nx, Nt`** (mathematical notation standard)
- Affected files: `config/modern_config.py`, `meta/mathematical_dsl.py`, `utils/performance/optimization.py`
- Fix with deprecation warnings for backward compatibility

---

## Phase 2: Classification by Priority

### **CRITICAL** (Runtime Errors)
✅ **[FIXED]** `mfg_pde/hooks/visualization.py` - SolverResult attribute access
- **Issue**: Using `.u`/`.m` instead of `.U`/`.M`
- **Impact**: AttributeError at runtime
- **Fix**: Changed `hasattr(result, "u")` → `hasattr(result, "U")`
- **Lines**: 183, 189-190

### **HIGH** (User-Facing API Consistency)

1. **Naming Inconsistency: `Nx` vs `nx`**
   - **Files**: 3 files (`config/modern_config.py`, `meta/mathematical_dsl.py`, `utils/performance/optimization.py`)
   - **Impact**: Confusing for users, breaks consistency
   - **Effort**: 2-3 hours (add deprecation warnings)

2. **Boolean Proliferation: AutoDiff Backend Selection**
   - **Files**: 2 files (`utils/numerical/functional_calculus.py`, `utils/functional_calculus.py`)
   - **Impact**: Unclear API (`use_jax` and `use_pytorch` both True = error?)
   - **Effort**: 3-4 hours (create enum, add deprecation)

3. **Boolean Proliferation: Normalization Type**
   - **File**: `alg/neural/operator_learning/deeponet.py`
   - **Impact**: Mutually exclusive booleans confusing
   - **Effort**: 2-3 hours (create enum, update usages)

### **MEDIUM** (Code Quality)

4. **Tuple Returns in Internal Functions**
   - **Files**: Visualization modules (100+ instances)
   - **Impact**: Low (internal implementation detail)
   - **Recommendation**: **No action needed** - These are internal backend-specific methods

5. **Grid Size Parameter Names**
   - **Issue**: Some functions use lowercase `nx, nt` inconsistently
   - **Impact**: Medium (internal utilities mostly)
   - **Effort**: 4-6 hours (systematic rename with deprecation)

### **LOW** (Deferred)

6. **Legacy Code Modernization**
   - Many tuple returns are acceptable (state accessors, internal helpers)
   - **Recommendation**: Touch only when refactoring those modules

---

## Immediate Action Items

### ✅ **Action 1: Fix Critical Bug** - COMPLETED
- **File**: `mfg_pde/hooks/visualization.py`
- **Change**: `.u`/`.m` → `.U`/`.M`
- **Test**: Run visualization hook tests

### **Action 2: Standardize Grid Size Parameters** (3 hours)
**Files to fix**:
1. `config/modern_config.py:119` - `with_grid_size(nx, nt)` → `with_grid_size(Nx, Nt)`
2. `meta/mathematical_dsl.py:95` - `domain(..., nx=100, nt=50)` → `domain(..., Nx=100, Nt=50)`
3. `utils/performance/optimization.py:156` - `create_laplacian_3d(nx, ny, nz, ...)` → `create_laplacian_3d(Nx, Ny, Nz, ...)`

**Implementation**:
- Add backward compatibility with deprecation warnings
- Update all call sites
- Test thoroughly

### **Action 3: Create AutoDiffBackend Enum** (3 hours)
**Location**: `mfg_pde/utils/numerical/autodiff.py` (new file)

```python
from enum import Enum

class AutoDiffBackend(str, Enum):
    """Backend selection for automatic differentiation."""
    NUMPY = "numpy"  # Default finite differences
    JAX = "jax"      # JAX autodiff
    PYTORCH = "pytorch"  # PyTorch autograd
```

**Update**:
- `utils/numerical/functional_calculus.py`
- `utils/functional_calculus.py`
- Add deprecation warnings for boolean parameters

### **Action 4: Create NormalizationType Enum** (2 hours)
**Location**: `mfg_pde/alg/neural/operator_learning/normalization.py` (new file)

```python
from enum import Enum

class NormalizationType(str, Enum):
    """Normalization layer type."""
    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"
```

**Update**:
- `alg/neural/operator_learning/deeponet.py`
- Replace `use_batch_norm`, `use_layer_norm` with single `normalization` enum

---

## Statistics

**Automated Search Results**:
- Attribute patterns: 100+ matches (mostly correct)
- Tuple returns: 100+ matches (mostly acceptable internal usage)
- Boolean pairs: 9 matches (4 candidates for enum conversion)
- Naming inconsistencies: 7 matches (3 needing fixes)

**Critical Issues Found**: 1 (fixed)
**High Priority Issues**: 3 (to be fixed)
**Medium Priority Issues**: 2 (optional improvements)

**Estimated Effort**:
- Critical fix: ✅ 0.5 hours (completed)
- High priority fixes: 8-10 hours
- Total: ~10 hours for complete remediation

---

## Recommendations

### **Immediate (This Week)**
1. ✅ Fix critical bug (completed)
2. Commit critical bug fix
3. Standardize grid parameter names
4. Create AutoDiffBackend enum

### **Short Term (Next Week)**
5. Create NormalizationType enum
6. Update documentation with naming standards

### **Long Term (Future Refactor)**
7. Create comprehensive API style guide
8. Add linting rules to enforce consistency
9. Update examples to follow standards

---

## Success Metrics

**Phase 1-2 Complete**:
- ✅ Automated discovery complete
- ✅ Critical bug found and fixed
- ✅ Findings classified by priority
- ✅ Action items identified

**Remaining Phases**:
- [ ] Phase 3: Execute high-priority fixes
- [ ] Phase 4: Document API style guide

---

**Next Steps**: Commit critical bug fix, then proceed with high-priority standardization
