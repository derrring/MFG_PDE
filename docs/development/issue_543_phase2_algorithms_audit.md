# Issue #543 Phase 2: hasattr Elimination in Algorithms

**Date**: 2026-01-11
**Status**: IN PROGRESS
**Scope**: Eliminate 149 hasattr violations in `mfg_pde/alg/` directory

## Background

Issue #543 Phase 1 successfully eliminated hasattr from core and geometry modules (96% reduction, 79 → 3 violations). Phase 2 extends this work to the algorithms directory, which contains 149 hasattr checks.

## Current State

**Total hasattr count by module**:
```
alg/: 149 ← FOCUS OF THIS PHASE
backends/: 41
config/: 3 ✅
core/: 5 ✅ (mostly cleaned)
geometry/: 27
utils/: 49
Other modules: 26
```

## Pattern Analysis

### Category 1: Problem API Variations (40+ violations)

**Pattern**: Code checking which version of problem API is present

**Top violations**:
- `hasattr(self.problem, "geometry")` - 10 occurrences
- `hasattr(self.problem, "initial_density")` - 5
- `hasattr(self.problem, "T")` - 4
- `hasattr(self.problem, "get_initial_m")` - 3
- `hasattr(self.problem, "domain")` - 3
- Plus: `get_u_fin`, `m_init`, `u_fin`, `dimension`, `f_potential`, `components`, etc.

**Refactoring Strategy**:
```python
# OLD: hasattr duck typing
if hasattr(self.problem, "geometry"):
    geometry = self.problem.geometry
else:
    geometry = None  # Silent fallback

# NEW: Try/except with clear error
try:
    geometry = self.problem.geometry
except AttributeError as e:
    raise TypeError(f"Problem must provide 'geometry' attribute") from e
```

**Alternative for optional attributes**:
```python
# Use getattr with None default when attribute is truly optional
initial_density = getattr(self.problem, "initial_density", None)
if initial_density is not None:
    # Use it
else:
    # Create default
```

### Category 2: Boundary Conditions API (11 violations)

**Pattern**: Checking BC interface compatibility

**Violations**:
- `hasattr(boundary_conditions, "is_uniform")` - 6
- `hasattr(boundary_conditions, "type")` - 5

**Refactoring Strategy**:
```python
# OLD: Check for legacy API
if hasattr(boundary_conditions, "is_uniform"):
    if boundary_conditions.is_uniform:
        bc_type = boundary_conditions.type

# NEW: Use modern BoundaryConditions API directly
from mfg_pde.geometry.boundary import BoundaryConditions
if isinstance(boundary_conditions, BoundaryConditions):
    # Use modern API
    bc_type = boundary_conditions.get_uniform_type()
else:
    # Legacy fallback with warning
    logger.warning("Using legacy BC interface - please upgrade")
    bc_type = boundary_conditions.type if hasattr(...) else "periodic"
```

### Category 3: Environment API (6 violations - RL code)

**Pattern**: Checking if RL environment has certain methods

**Violations**:
- `hasattr(self.env, "get_population_state")` - 6

**Refactoring Strategy**:
Define `MFGEnvironmentProtocol` in `mfg_pde/alg/reinforcement/types.py`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MFGEnvironmentProtocol(Protocol):
    """Protocol for MFG reinforcement learning environments."""
    def get_population_state(self) -> np.ndarray: ...
    def reset(self) -> np.ndarray: ...
    def step(self, action) -> tuple: ...

# In RL code
if isinstance(self.env, MFGEnvironmentProtocol):
    population = self.env.get_population_state()
else:
    raise TypeError("Environment must implement MFGEnvironmentProtocol")
```

### Category 4: Solver Method Variations (8 violations)

**Pattern**: Checking which solve method exists

**Violations**:
- `hasattr(self.hjb_solver, "solve_hjb_system")` - 4
- `hasattr(self.fp_solver, "solve_fp_system")` - 4

**Refactoring Strategy**:
Define `HJBSolverProtocol` and `FPSolverProtocol`:

```python
from typing import Protocol

@runtime_checkable
class HJBSolverProtocol(Protocol):
    """Protocol for HJB solvers."""
    def solve(self, ...) -> SolverResult: ...

@runtime_checkable
class FPSolverProtocol(Protocol):
    """Protocol for FP solvers."""
    def solve(self, ...) -> SolverResult: ...

# In coupling code
try:
    result = self.hjb_solver.solve(...)
except AttributeError:
    # Fallback to legacy solve_hjb_system()
    logger.warning("Using legacy HJB solve method")
    result = self.hjb_solver.solve_hjb_system(...)
```

### Category 5: Backend/Tensor Compatibility (~84 violations)

**Pattern**: Checking if backend/tensor supports certain operations

**Examples**:
- `hasattr(xp, "clip")` - NumPy/CuPy compatibility
- `hasattr(xp, "manual_seed")` - PyTorch compatibility
- `hasattr(v, "item")` - Tensor to scalar conversion
- `hasattr(particles_tensor, "reshape")` - Tensor operations
- `hasattr(torch.backends, "mps")` - MPS backend availability

**Refactoring Strategy**:
**Keep hasattr for external library compatibility checks** but add comments:

```python
# Backend feature detection - hasattr acceptable for external libraries
if hasattr(xp, "clip"):
    clipped = xp.clip(values, min_val, max_val)
else:
    # Fallback for older NumPy
    clipped = xp.minimum(xp.maximum(values, min_val), max_val)
```

**Exception**: For our own backend abstraction layer, use protocols instead:

```python
# mfg_pde/backends/protocol.py
from typing import Protocol

class BackendProtocol(Protocol):
    """Protocol for numerical backends."""
    def clip(self, a, min_val, max_val) -> NDArray: ...
    def rand(self, *shape) -> NDArray: ...
    # etc.

# In algorithm code
if isinstance(backend, BackendProtocol):
    # Use protocol methods
    result = backend.clip(values, 0, 1)
```

## Prioritization

### High Priority (Days 1-3)

**Target**: Coupling and base solver modules (core infrastructure)

1. **mfg_pde/alg/numerical/coupling/** - 30+ violations
   - `fictitious_play.py` - 15 violations
   - `fixed_point_iterator.py` - 10 violations
   - `hybrid_fp_particle_hjb_fdm.py` - 3 violations
   - `base_mfg.py` - 2 violations

2. **mfg_pde/alg/numerical/fp_solvers/base_fp.py** - 1 violation

### Medium Priority (Days 4-5)

**Target**: FP and HJB solver implementations

3. **mfg_pde/alg/numerical/fp_solvers/** - 20+ violations
   - `fp_fdm.py` - 6 violations
   - `fp_fdm_time_stepping.py` - 4 violations
   - Various algorithm files - 10+ violations

4. **mfg_pde/alg/numerical/network_solvers/** - 1 violation
   - `fp_network.py`

5. **mfg_pde/alg/numerical/stochastic/** - 1 violation
   - `common_noise_solver.py`

### Lower Priority (Day 6+)

**Target**: RL algorithms (separate ecosystem)

6. **mfg_pde/alg/reinforcement/** - 80+ violations
   - Define `MFGEnvironmentProtocol` first
   - Then refactor RL algorithms systematically

**Note**: RL code can be deferred since it's a separate subsystem with its own conventions.

### Defer (Keep hasattr for now)

**Target**: Backend compatibility checks

7. **mfg_pde/alg/numerical/particle_utils.py** - Backend feature detection
   - `hasattr(xp, "clip")`, `hasattr(xp, "manual_seed")`, etc.
   - Keep these with explanatory comments

## Implementation Plan

### Phase 2A: Coupling Modules (Days 1-2)

**Files**:
- `mfg_pde/alg/numerical/coupling/fictitious_play.py` (15 violations)
- `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py` (10 violations)
- `mfg_pde/alg/numerical/coupling/hybrid_fp_particle_hjb_fdm.py` (3 violations)

**Strategy**:
1. Create `HJBSolverProtocol` and `FPSolverProtocol` in `mfg_pde/types/protocols.py`
2. Replace solver method checks with try/except
3. Replace problem API checks with try/except or getattr
4. Add logging for fallbacks
5. Run tests after each file

**Estimated Effort**: 2 days

### Phase 2B: FP Solver Modules (Days 3-4)

**Files**:
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py` (6 violations)
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm_time_stepping.py` (4 violations)
- `mfg_pde/alg/numerical/fp_solvers/base_fp.py` (1 violation)
- Algorithm files: `fp_fdm_alg_*.py` (10+ violations)

**Strategy**:
1. Centralize BC retrieval (similar to HJB solvers)
2. Replace BC API checks with isinstance(BoundaryConditions)
3. Replace problem API checks with try/except
4. Run tests after each file

**Estimated Effort**: 2 days

### Phase 2C: RL Algorithms (Days 5-6, Optional)

**Files**:
- `mfg_pde/alg/reinforcement/` (80+ violations)

**Strategy**:
1. Define `MFGEnvironmentProtocol`
2. Replace env method checks with isinstance
3. Document expected env interface
4. Run RL tests

**Estimated Effort**: 2 days (can be deferred)

## Acceptance Criteria

- [ ] All hasattr in coupling modules replaced (30 violations → 0)
- [ ] All hasattr in FP solvers replaced (20 violations → 0)
- [ ] HJBSolverProtocol and FPSolverProtocol defined
- [ ] Tests passing for modified modules
- [ ] Documentation updated for protocol usage
- [ ] (Optional) RL hasattr eliminated (80 violations → 0)

**Target**: Reduce algorithms hasattr from 149 to <70 (53% reduction) in Phase 2A+2B.

## Progress Tracking

**Phase 2A**: ✅ COMPLETE (2026-01-11)
**Phase 2B**: ❌ Not started
**Phase 2C**: ❌ Not started (deferred)

**Total Eliminated**: 24/149 (16% - Phase 2A only)

### Phase 2A Completion Summary (2026-01-11)

**Commits**:
- `ae12ce2` - fictitious_play.py refactoring
- `62da835` - fixed_point_iterator.py refactoring
- `a025e9c` - hybrid_fp_particle_hjb_fdm.py refactoring

**Results**:
- **fictitious_play.py**: 15 → 2 documented (87% reduction)
  - Centralized `_get_initial_and_terminal_conditions()` helper (8 hasattr → 1 method)
  - Cached solver signatures in `__init__` (4 hasattr → signature cache)
  - Documented 2 progress bar hasattr as acceptable (tqdm/rich interface)

- **fixed_point_iterator.py**: 10 → 2 documented (80% reduction)
  - Applied same patterns as fictitious_play.py
  - Centralized initial/terminal condition retrieval
  - Cached solver signatures
  - Documented 2 progress bar hasattr as acceptable

- **hybrid_fp_particle_hjb_fdm.py**: 3 → 0 (100% reduction)
  - Replaced problem method hasattr with try/except
  - Initialized U_solution/M_solution to None (eliminated hasattr in get_results())

**Infrastructure Created**:
- **HJBSolverProtocol** and **FPSolverProtocol** in `mfg_pde/types/protocols.py`
- Reusable `_get_initial_and_terminal_conditions()` pattern
- Signature caching pattern for solver method detection

**Metrics**:
```
Before Phase 2A:  149 hasattr in alg/
After Phase 2A:   124 hasattr in alg/
Reduction:        25 violations (17%)
  - Eliminated:   24 violations
  - Documented:   4 violations (progress bar interface checks)
  - Remaining:    ~120 violations (FP solvers, RL algorithms)
```

## Testing Strategy

**Unit Tests**:
- Run existing test suite after each module refactoring
- No new tests needed (behavior unchanged)

**Integration Tests**:
- Run coupling solver tests (fictitious play, fixed point iteration)
- Run FP solver tests (FDM, particle)

**Smoke Tests**:
- Verify examples still execute

## Related Issues

- Issue #543 Phase 1 (completed): Core and geometry hasattr elimination
- Issue #545 (completed): Solver BC refactoring (eliminates some hasattr)
- CLAUDE.md § Fail Fast & Surface Problems

## Notes

**Backend Compatibility Exception**: Keep hasattr checks for external library feature detection (NumPy, PyTorch, CuPy). Add explanatory comments to clarify these are for third-party library compatibility, not our own code.

**Logging**: Add `logger.warning()` for all legacy API fallbacks to encourage migration.

---

**Last Updated**: 2026-01-11
**Status**: Audit complete, ready to start Phase 2A
