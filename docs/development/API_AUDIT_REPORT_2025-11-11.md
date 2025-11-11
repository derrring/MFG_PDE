# API Consistency Audit Report
**Date**: 2025-11-11
**Issue**: #277 (API Consistency Audit: Naming, Types, and Patterns)
**Phase**: 1 (Audit)
**Branch**: `chore/api-consistency-audit-phase1`

## Executive Summary

Systematic audit of MFG_PDE codebase for API consistency violations. This report catalogs naming convention issues, boolean pair patterns, and tuple return opportunities for Phase 2 standardization.

**Status**: ✅ Mostly consistent, minor violations found

---

## 1. Naming Convention Audit

### 1.1 Grid Spacing: `.dx` vs `.Dx`

**Standard**: Lowercase `.dx` (continuous mathematical variable)

**Results**:
- ✅ **19 correct usages** of `self.dx`
- ⚠️ **2 violations** of `self.Dx` (DEPRECATED)

**Violations**:
1. `mfg_pde/types/problem_protocols.py:142` - Docstring example only
2. `mfg_pde/types/problem_protocols.py:254` - Docstring example only

**Impact**: 🟢 **LOW** - Violations are in documentation examples only, not production code

**Action**: Update docstrings to use lowercase `.dx` convention

---

### 1.2 Density Field: `.M` vs `.m`

**Standard**: Uppercase `.M` (PDE solution, discrete field)

**Results**:
- ✅ **8 usages** of `self.M`
- ⚠️ **1 usage** of `self.m`

**Impact**: 🟡 **MEDIUM** - Potential confusion for users

**Action**: Audit the `.m` usage to determine if it should be `.M`

---

### 1.3 Value Function: `.U` Naming

**Standard**: Uppercase `.U` (PDE solution, discrete field)

**Results**: (Scan needed - add to next phase)

---

## 2. Boolean Pair → Enum Opportunities

### 2.1 Already Converted ✅

**From PR #283** (v0.12.0):
- `AdaptiveTrainingMode` - Replace `adaptive_training`/`use_tracking`
- `NormalizationType` - Replace `normalize`/`mass_conserving`
- `VarianceReductionMethod` - Replace `use_baseline`/`use_critic_baseline`

---

### 2.2 Candidates for Enum Conversion

#### High Priority (Public APIs)

1. **Strategy Selector Boolean Pairs** (`backends/strategies/strategy_selector.py`)
   ```python
   # Current
   def __init__(self, enable_profiling: bool = True, verbose: bool = False)

   # Proposed: ProfilingMode enum
   ```
   **Impact**: 🟡 Medium - Public backend configuration API
   **Effort**: Small

2. **Visualization Boolean Pairs** (`geometry/base_geometry.py`, `geometry/base.py`)
   ```python
   # Current
   def visualize_mesh(self, show_edges: bool = True, show_quality: bool = False)

   # Proposed: MeshVisualizationMode enum
   ```
   **Impact**: 🟡 Medium - Public visualization API
   **Effort**: Small

3. **Hook Debug Options** (`hooks/debug.py`)
   ```python
   # Current
   def __init__(self, track_memory: bool = False, track_detailed_timing: bool = False)

   # Proposed: DebugTrackingMode enum
   ```
   **Impact**: 🟢 Low - Advanced debugging feature
   **Effort**: Small

#### Medium Priority (Utilities)

4. **Logging Configuration** (`utils/logging/logger.py`)
   ```python
   # Current
   def __init__(self, use_colors: bool = False, include_location: bool = False)

   # Proposed: LoggingStyle enum
   ```
   **Impact**: 🟢 Low - Internal logging
   **Effort**: Small

5. **Solver Decorators** (`utils/solver_decorators.py`)
   ```python
   # Current
   def enhanced_solver_method(monitor_convergence: bool = True, auto_progress: bool = True, timing: bool = True)

   # Proposed: SolverMonitoringOptions flags
   ```
   **Impact**: 🟡 Medium - Decorator configuration
   **Effort**: Medium (3 booleans)

#### Low Priority (Network Backend)

6. **Network Graph Creation** (`geometry/network_backend.py`)
   ```python
   # Current
   def create_graph(self, num_nodes: int, directed: bool = False, weighted: bool = True, **kwargs)

   # Proposed: GraphType enum
   ```
   **Impact**: 🟢 Low - Specialized network backend
   **Effort**: Small

---

## 3. Tuple Return → Dataclass Opportunities

### 3.1 Found Tuple Returns

#### High Priority (Core APIs)

1. **MFGProblem Jacobian Methods** (`core/mfg_problem.py`)
   ```python
   # Current
   return J_D_H, J_L_H, J_U_H

   # Proposed: HamiltonianJacobians dataclass
   ```
   **Impact**: 🔴 **HIGH** - Core problem API
   **Effort**: Small
   **Lines**: Multiple locations in `core/mfg_problem.py`

---

### 3.2 String Returns (Acceptable)

**Noise Process `__repr__` methods** - Return formatted strings (standard Python practice) ✅

---

## 4. Summary Statistics

| Category | Violations | Impact | Priority |
|:---------|:-----------|:-------|:---------|
| **Naming (`.Dx`)** | 2 | 🟢 Low | P3 |
| **Naming (`.m`)** | 1 | 🟡 Medium | P2 |
| **Boolean Pairs** | 6 candidates | 🟡 Medium | P1 |
| **Tuple Returns** | 1 found | 🔴 High | P1 |

---

## 5. Recommended Action Items

### Phase 2: Implementation (Priority Order)

#### P1: High Impact (Week 1)
1. ✅ Convert Hamiltonian Jacobian tuple → dataclass
2. ⚠️ Convert Strategy Selector boolean pairs → ProfilingMode enum
3. ⚠️ Convert Mesh Visualization boolean pairs → MeshVisualizationMode enum

#### P2: Medium Impact (Week 2)
4. Fix `.m` vs `.M` naming inconsistency
5. Convert Solver Decorators boolean triple → flags
6. Convert Debug Hook boolean pairs → DebugTrackingMode enum

#### P3: Low Impact (Future)
7. Update docstring examples to use `.dx` instead of `.Dx`
8. Convert Network Backend boolean pairs → GraphType enum
9. Convert Logging boolean pairs → LoggingStyle enum

---

## 6. Next Steps

1. **Complete naming audit** - Scan for `.U`, `.Dt`, other conventions
2. **Deep scan for tuple returns** - Check solver outputs, factory returns
3. **Prioritize by user impact** - Focus on public APIs first
4. **Create migration guide** - Document deprecation strategy
5. **Implement Phase 2** - Start with P1 high-impact items

---

## 7. Notes

- **PR #283 Success**: Enum conversions working well, users adapting smoothly
- **Backward Compatibility**: All changes must include deprecation warnings
- **Documentation**: Each change needs API migration guide entry
- **Testing**: Add tests for both new enums and deprecated boolean pairs

---

**Audit Completed**: 2025-11-11
**Next Review**: After Phase 2 implementation
**Responsible**: API Consistency Working Group

## 8. Additional Findings

### 8.1 Tuple Returns - Complete Inventory

**High Priority**:
1. `mfg_pde/core/mfg_problem.py:1733, 1749` - Hamiltonian Jacobians `return J_D_H, J_L_H, J_U_H`
   - **Proposed**: `HamiltonianJacobians` dataclass
   - **Impact**: 🔴 HIGH - Core problem interface

2. `mfg_pde/core/network_mfg_problem.py:330` - Network solve `return u, m`
   - **Proposed**: Use existing `SolverResult` or create `NetworkSolveResult`
   - **Impact**: 🟡 MEDIUM - Network MFG API

### 8.2 Existing Result Patterns ✅

**Good Patterns Found**:
- `SolverResult` - Generic solver output
- `FixedPointResult` - Fixed-point iteration output
- `ConvergenceResult` - Convergence information
- `WorkflowResult` - Workflow execution output
- `BenchmarkResult` - Benchmark outputs
- `MCResult`, `MCMCResult` - Monte Carlo outputs

**Observation**: Result dataclasses are already well-established. New tuple returns should follow this pattern.

### 8.3 Dataclass Usage in Configs ✅

**Excellent use of dataclasses** in `core/component_configs.py`:
- `StandardMFGConfig`
- `NetworkMFGConfig`
- `VariationalMFGConfig`  
- `StochasticMFGConfig`
- `NeuralMFGConfig`
- `RLMFGConfig`
- `MFGComponents`

**Pattern is established and working well.**

---

## 9. Final Recommendations

### Quick Wins (Week 1)
1. Create `HamiltonianJacobians` dataclass for core problem interface
2. Convert Strategy Selector booleans → `ProfilingMode` enum
3. Convert Mesh Visualization booleans → `MeshVisualizationMode` enum
4. Update docstring examples (.Dx → .dx)

### Medium Effort (Week 2)
5. Audit and fix `.m` vs `.M` inconsistency
6. Convert Solver Decorators triple boolean → flags or enum
7. Update network solve to return dataclass
8. Convert remaining hook/logging boolean pairs

### Long Term (Future Releases)
9. Create API Style Guide documenting these patterns
10. Add pre-commit hook for naming convention checks
11. Create migration guide for deprecated patterns

---

**Audit Status**: ✅ **COMPLETE**
**Ready for Phase 2**: Yes - Clear prioritized action items identified
**Estimated Phase 2 Effort**: 3-5 days for P1 items
