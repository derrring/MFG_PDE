# Development Priority List (2026-01-18)

## Current Version: v0.17.2 (released 2026-01-18)

This document outlines the prioritized roadmap for infrastructure improvements following PR #548.

**Recent Completions**:
- Issue #591 Phase 2 (Variational Inequality Constraints) merged to main on 2026-01-18
- Issue #496 (Time-Varying BC Fix) fixed on 2026-01-18
- Issue #544 (Legacy Parameter Deprecation) completed on 2026-01-18 (v0.17.2)
- Issue #545 (Solver Mixin Cleanup) completed on 2026-01-17 (v0.17.2)

---

## ✅ Priority 1: Fix FDM Periodic BC Bug (#542) - **COMPLETED**

**Issue**: [#542](https://github.com/derrring/MFG_PDE/issues/542)
**Status**: ✅ CLOSED (2026-01-10)
**Priority**: High (correctness bug)
**Size**: Medium
**Actual Effort**: 2 days (PR #548 + PR #550)

### Problem
FDM 1D HJB solver used `np.roll()` for Laplacian computation, implementing periodic BC regardless of geometry settings.

**Impact**: 1D corridor evacuation validation showed 50% error (vs 0.5% for GFDM).

### Solution Implemented

**PR #548**: BC-aware gradient and Laplacian computation using ghost cells
**PR #550**: Explicit BC enforcement for Dirichlet and Neumann boundary values

**Key Insight**: BC-aware derivatives ≠ BC enforcement
- **Dirichlet BC**: Explicitly set `u(boundary) = g`
- **Neumann BC**: Set boundary value to satisfy `∂u/∂n = g`

### Result
- ✅ BC-aware stencils replace `np.roll()`
- ✅ Explicit Dirichlet/Neumann enforcement added
- ✅ 1D corridor evacuation validation error reduced to < 2%
- ✅ Mixed BC handling implemented and tested

---

## ✅ Priority 2: Eliminate Silent Fallbacks (#547) - **COMPLETED**

**Issue**: [#547](https://github.com/derrring/MFG_PDE/issues/547)
**Status**: ✅ CLOSED (2026-01-11)
**Priority**: Medium
**Size**: Medium
**Actual Effort**: 2 days (PR #555 + PR #556)

### Problem
Code catches broad exceptions and silently falls back to lower-fidelity methods without warning users. This masks configuration errors and performance degradation.

### Solution Implemented

**Two-Phase Implementation**:
1. **PR #555** - High/Medium priority (9/13 fixes)
   - High: Newton solver failures, spectral analysis failures
   - Medium: Backend detection, GPU memory stats, volume computation, vmap fallback
2. **PR #556** - Low priority cosmetic (4/13 fixes)
   - Backend info retrieval, LaTeX setup, Quasi-MC fallback, performance monitoring

**Patterns Established**:
- Critical user-facing warnings: Specific exceptions + `logger.warning()` with fallback implications
- Diagnostic debug logging: Specific exceptions + `logger.debug()` for non-critical info
- Initialization warnings: Specific exceptions + `warnings.warn()` for module setup
- Re-raise with context: Broad exception OK when immediately re-raising with added logging

### Result
- ✅ 100% completion (13/13 fixes)
- ✅ All broad `except Exception:` replaced with specific exception types
- ✅ Comprehensive logging with context throughout
- ✅ Consistent MFG_PDE logging infrastructure
- ✅ All fallback behavior preserved for robustness
- ✅ Critical bugs (Newton solver) now surface instead of silently failing

**Documentation**: `docs/development/SILENT_FALLBACK_AUDIT_547.md`, `docs/development/SILENT_FALLBACK_COMPLETION_547.md`

### Why Second?
- **Aligns with Fail Fast principle** (CLAUDE.md core value)
- **Improves debugging experience** (users know what's happening)
- **Medium scope** (codebase-wide but straightforward)
- **Independent of other work** (can proceed in parallel with other tasks)

---

## ✅ Priority 3: hasattr() Elimination - Protocol Duck Typing (#543) - **COMPLETED**

**Issue**: [#543](https://github.com/derrring/MFG_PDE/issues/543)
**Status**: ✅ COMPLETED (2026-01-10)
**Priority**: High
**Size**: Large (but incremental)
**Actual Effort**: 4 days (4 PRs merged)

### Result
Eliminated 96% of protocol duck typing violations (79 → 3) in core and geometry modules.

### Problem
Protocol duck typing with `hasattr()` violates Fail Fast principle and creates unclear contracts.

### Solution Implemented

**Pull Requests Merged**:
1. ✅ PR #551 - Core module cleanup (27 → 5 violations)
2. ✅ PR #552 - Geometry Phase 1 (47 → 38 violations)
3. ✅ PR #553 - Core protocol checks (5 → 3 violations)
4. ✅ PR #554 - Geometry protocol checks (38 → 0 violations)

**Patterns Established**:
- `isinstance(geometry, GeometryProtocol)` for required methods
- `try/except AttributeError` for optional attributes
- Explicit error messages guide users to proper implementations

**Documentation**: `docs/archive/issue_543_hasattr_elimination_2026-01/`

**Remaining Work**: ~341 hasattr violations in other patterns (caching, legacy compatibility) - see `HASATTR_CLEANUP_PLAN.md`

**Phase 3: Utils & Workflow** (Priority 7)
- `mfg_pde/utils/`
- `mfg_pde/workflow/`

### Acceptance Criteria (Phase 1)
- [ ] Define additional protocols (SolverProtocol, ConfigProtocol)
- [ ] Replace all hasattr in `mfg_pde/core/` with isinstance checks
- [ ] Add CI check to reject new hasattr in core/
- [ ] Document protocol usage in `docs/development/CONSISTENCY_GUIDE.md`

### Why Third?
- **Builds on GeometryProtocol** work from PR #548
- **High priority** but can proceed incrementally
- **Core first** (highest leverage for downstream code)
- **Enables better type checking** and fail-fast behavior

---

## ✅ Priority 3.5: Adjoint-Aware Solver Pairing (#580) - **COMPLETED**

**Issue**: [#580](https://github.com/derrring/MFG_PDE/issues/580)
**PR**: [#585](https://github.com/derrring/MFG_PDE/pull/585)
**Status**: ✅ COMPLETED (2026-01-17)
**Priority**: High (scientific correctness)
**Size**: Large
**Actual Effort**: ~10 hours (5 implementation phases + comprehensive review)

### Problem
Users could accidentally mix incompatible discretization schemes (e.g., FDM HJB + GFDM FP), breaking the adjoint duality relationship L_FP = L_HJB^T required for Nash equilibrium convergence.

**Impact**: Silent failure - Nash gap doesn't converge even as mesh size h→0.

### Solution Implemented

**Three-Mode Solving API** (PR #585):

1. **Safe Mode**: `problem.solve(scheme=NumericalScheme.FDM_UPWIND)`
   - Factory creates guaranteed dual pairs
   - Impossible to create invalid pairings

2. **Expert Mode**: `problem.solve(hjb_solver=hjb, fp_solver=fp)`
   - Manual control with automatic validation
   - Educational warnings if mismatched

3. **Auto Mode**: `problem.solve()`
   - Intelligent defaults (backward compatible)
   - Currently uses FDM_UPWIND

**Key Components**:
- `NumericalScheme` enum: User-facing scheme selection
- `SchemeFamily` enum: Internal classification (FDM, SL, GFDM, etc.)
- Trait system: Refactoring-safe `_scheme_family` attributes
- Duality validation: `check_solver_duality()` utility
- Paired solver factory: `create_paired_solvers()` with config threading

### Result
- ✅ 12 files created (3 core, 6 tests, 3 docs)
- ✅ 14 files modified (12 solver traits, 2 facade)
- ✅ 121 tests passing (~98% coverage)
- ✅ 2,400+ lines of documentation
- ✅ 100% backward compatible
- ✅ All reviews passed (5-star ratings across all categories)
- ✅ Zero performance regression (<1% overhead)

**Documentation**:
- User guide: `docs/user/three_mode_api_migration_guide.md`
- Technical guide: `docs/development/issue_580_adjoint_pairing_implementation.md`
- Demo: `examples/basic/three_mode_api_demo.py`
- Reviews: `.github/*_REVIEW_580.md` (7 documents)

### Why 3.5?
- **Critical for scientific correctness** but not infrastructure refactoring
- **Independent feature development** (not part of infrastructure priority sequence)
- **High value** but **different scope** than infrastructure improvements
- **Inserted after completion** to maintain chronological record

---

## ✅ Priority 3.6: Unified Ghost Node Architecture (#576) - **COMPLETED**

**Issue**: [#576](https://github.com/derrring/MFG_PDE/issues/576)
**PR**: [#586](https://github.com/derrring/MFG_PDE/pull/586)
**Status**: ✅ COMPLETED (2026-01-17)
**Priority**: Medium (infrastructure improvement)
**Size**: Large
**Actual Effort**: ~6 hours (4 implementation phases + comprehensive testing)

### Problem
Ghost node handling was fragmented across solvers, leading to code duplication and inconsistent boundary accuracy. WENO5 had 5th-order interior accuracy but only 2nd-order boundary accuracy.

**Impact**: Boundary errors dominate in problems with critical boundary regions.

### Solution Implemented

**Request Pattern Architecture** (PR #586):

Decoupled **Physics** (BC type: Neumann, Dirichlet) from **Numerics** (reconstruction order).

**Key Components**:
- `order` parameter in `PreallocatedGhostBuffer` (default: 2)
- Order-based dispatch: linear (order ≤ 2) vs polynomial (order > 2)
- Vandermonde-based extrapolation for O(h^order) accuracy
- WENO5 integration with `order=5`

**Mathematical Foundation**:
```
u_{-k} = Σ w_{k,j} · u_j + β_k · g_bc
```

For Neumann BC (∂u/∂n = 0):
- Construct polynomial p(x) satisfying: p(x_j) = u_j, p'(0) = 0
- Solve Vandermonde system with n interior + 1 BC constraint
- Evaluate at ghost locations

### Result
- ✅ 6 files changed, 1,206 insertions
- ✅ 14 new tests, all passing
- ✅ 376 existing tests pass (backward compatibility verified)
- ✅ Machine precision accuracy (~1e-16) for polynomial solutions
- ✅ WENO5 achieves true O(h⁵) boundary accuracy
- ✅ Zero breaking changes (default `order=2` unchanged)

**Implementation**:
- `mfg_pde/geometry/boundary/applicator_fdm.py`: Core architecture (~245 lines)
- `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`: WENO integration (~17 lines)
- 3 test files: Parameter validation, accuracy verification, integration tests

**Documentation**:
- Implementation guide: `docs/development/issue_576_ghost_node_architecture.md`

**Deferred Optimizations**:
- Weight table pre-computation (Phase 5): Can be added if profiling shows need
- Semi-Lagrangian refactoring (Phase 7): Already uses `order=2` by default
- Performance benchmarks (Phase 8): Deferred until Phase 5

### Why 3.6?
- **Independent feature development** (not part of main infrastructure sequence)
- **Improves boundary accuracy** for high-order schemes (WENO5)
- **Inserted after completion** to maintain chronological record
- **Related to Priority 4** (boundary handling) but different scope

---

## ✅ Priority 4: Mixin Refactoring - FPParticle Template (#545) - **COMPLETED**

**Issue**: [#545](https://github.com/derrring/MFG_PDE/issues/545)
**PR**: [#548](https://github.com/derrring/MFG_PDE/pull/548)
**Status**: ✅ CLOSED (2026-01-11)
**Priority**: High
**Size**: Large
**Actual Effort**: Completed as part of infrastructure improvements

### Problem
Deep mixin hierarchies with implicit state sharing made solvers hard to understand and maintain. Each solver implemented BC handling differently.

### Solution Implemented

**Completed in PR #548** (Infrastructure improvements):
- ✅ Defined common BC interface patterns
- ✅ Refactored solver BC handling
- ✅ Geometry provides boundary detection methods
- ✅ Removed duplicate BC detection logic
- ✅ GeometryProtocol completion

### Result
- ✅ Unified BC handling across FDM, GFDM, Particle solvers
- ✅ Explicit composition patterns established
- ✅ Mixin hierarchies simplified
- ✅ Common interface for boundary operations

---

## ✅ Priority 5: hasattr() Elimination - Phase 2 (Algorithms) (#543) - **COMPLETED**

**Issue**: [#543](https://github.com/derrring/MFG_PDE/issues/543)
**Status**: ✅ COMPLETED (2026-01-17)
**Priority**: High
**Size**: Medium
**Actual Effort**: 1 day (5 commits)

### Problem
Remaining hasattr violations in numerical solvers and optimization algorithms (23 active patterns in core solver code).

### Solution Implemented

**Categories Addressed**:
1. ✅ **Category A - Backend Compatibility** (29 patterns documented)
   - JAX/NumPy/PyTorch/scipy feature detection
   - Added #543 tags to all backend compatibility patterns
   - 100% documentation coverage achieved

2. ✅ **Category B - Internal Cache** (4 patterns fixed)
   - Replaced hasattr with explicit None initialization in `__init__`
   - Pattern: `self._cached_attr: Type | None = None`
   - Fixed in: `hjb_gfdm.py` (3 violations)

3. ✅ **Category C - Problem API** (9 patterns fixed)
   - Replaced hasattr with `getattr(problem, "attr", None)`
   - More efficient (single lookup vs two)
   - Fixed in: `common_noise_solver.py`, `primal_dual_solver.py`, `wasserstein_solver.py`, `sinkhorn_solver.py`

4. ✅ **Category D - Interface Checks** (0 violations)
   - Verified no magic method checks (`__getitem__`, `__len__`, etc.)
   - All previous violations resolved in Phase 2A/2B

**Commits**:
1. `924ad3e0` - 4 violations (1 Category A, 3 Category B)
2. `39f7d9db` - 3 violations (2 Category A, 1 Category C)
3. `23baf770` - 4 violations (1 Category A, 3 Category C)
4. `22fd37fc` - 6 violations (1 Category A, 5 Category C)
5. `f0176dbb` - 6 inline tags (complete documentation coverage)

### Result
- ✅ 100% completion (23/23 active patterns addressed)
- ✅ All tests passing
- ✅ Issue #543 marked CLOSED
- ✅ Code quality improvements (object shape stability, type safety)

**Documentation**: `/tmp/issue_543_phase2_FINAL_SESSION_SUMMARY.md`

**Remaining Work**: Category E (RL code - 10 violations, deferred to future RL refactoring)

---

## ✅ Priority 5.5: Progress Bar Protocol Pattern (#587) - **COMPLETED**

**Issue**: [#587](https://github.com/derrring/MFG_PDE/issues/587)
**Status**: ✅ COMPLETED (2026-01-17)
**Priority**: Medium
**Size**: Medium
**Actual Effort**: 4 hours (3 phases)

### Problem
7 progress bar patterns using hasattr duck typing with tqdm/RichProgressBar, categorized as "acceptable backend compatibility" but identified by expert review as architectural issue requiring Protocol pattern.

### Solution Implemented

**Phase 1: Extend API (Non-Breaking)**
- Added `ProgressTracker` Protocol with `__iter__()`, `update_metrics()`, `log()`
- Implemented `NoOpProgressBar` (Null Object pattern) for zero-overhead silent operation
- Created `create_progress_bar()` factory for type-safe polymorphism
- Extended `RichProgressBar` with Protocol methods
- Deprecated `set_postfix()` → calls `update_metrics()` internally

**Phase 2: Migrate Solvers (7 Patterns Eliminated)**
- `fixed_point_iterator.py` - 2 hasattr checks removed
- `fictitious_play.py` - 2 hasattr checks removed
- `block_iterators.py` - 2 hasattr checks removed
- `hjb_gfdm.py` - 1 hasattr check removed
- All integration tests passing (10/10)

**Phase 3: Documentation & Cleanup**
- Updated `CLAUDE.md` with new progress bar pattern
- Added v0.17.0 section to `DEPRECATION_MODERNIZATION_GUIDE.md`
- Documented migration path with before/after examples
- Removed all #543 tags from migrated code (no longer hasattr violations)

**Commits**:
1. `2d7ae71c` - Protocol infrastructure + solver migration
2. `7d2e1e4e` - Documentation updates

### Result
- ✅ Zero hasattr checks in progress bar code
- ✅ Type safety: Mypy can verify all ProgressTracker calls
- ✅ Performance: NoOpProgressBar is zero-overhead pass-through
- ✅ Testability: NoOpProgressBar is built-in test double
- ✅ Extensibility: Easy to add WebSocket/Jupyter progress bars

**Architecture Documentation**: `docs/development/progress_bar_protocol_design.md`

**Behavior Change**: `verbose=False` is now completely silent (no print statements)

---

## ✅ Priority 6: hasattr() Elimination - Phase 3 (RL Code) (#543) - **COMPLETED**

**Issue**: [#543](https://github.com/derrring/MFG_PDE/issues/543)
**Status**: ✅ COMPLETED (2026-01-17)
**Priority**: High
**Size**: Small
**Actual Effort**: 1 hour (1 commit)

### Problem
Remaining 11 hasattr violations in RL subsystem (`mfg_pde/alg/reinforcement/`).

### Solution Implemented

**Category A - Backend Compatibility** (10 patterns documented):
- RL Agent API: Optional `reset_noise()` method (1)
- Gym Environment API: Optional `get_population_state()` method (6)
- Gym Action Space API: `action_space`, `n`, `nvec` attributes (3)

**Category C - Problem API** (1 pattern fixed):
- Validation code: Replaced `hasattr` with `getattr`

**Files Modified**: 6 RL algorithm files

### Result
- ✅ 100% coverage of active solver code (Core + Geometry + Numerical + Optimization + RL)
- ✅ Total: 61 patterns addressed across 3 phases (8 + 42 + 11)
- ✅ All backend compatibility patterns properly documented
- ✅ Issue #543 fully complete for core solver infrastructure

**Out of Scope**: Meta-programming infrastructure (22 patterns in `mfg_pde/meta/`) - deferred as non-critical

**Documentation**: `/tmp/issue_543_phase3_summary.md`

---

## ✅ Priority 6.5: Adjoint-Consistent Boundary Conditions (#574) - **COMPLETED**

**Issue**: [#574](https://github.com/derrring/MFG_PDE/issues/574)
**PR**: [#588](https://github.com/derrring/MFG_PDE/pull/588)
**Status**: ✅ COMPLETED (2026-01-17)
**Priority**: Medium (scientific correctness)
**Size**: Medium
**Actual Effort**: ~8 hours (4 phases)

### Problem
Standard Neumann BC (∂U/∂n = 0) for HJB at reflecting boundaries is mathematically inconsistent with the equilibrium solution when stall points occur at domain boundaries.

**Mathematical Analysis**: At Boltzmann-Gibbs equilibrium, the drift is α* = -∇U* = (σ²/2T_eff) · ∇V. When the stall point is at the boundary, ∇V ≠ 0 there, so the equilibrium requires ∇U ≠ 0, violating standard Neumann BC.

**Impact**: 2.65x error increase observed in mfg-research exp14b validation when stall point at boundary vs centered.

### Solution Implemented

**Adjoint-Consistent Robin BC** (PR #588):

At reflecting boundaries with zero total flux J·n = 0 where J = -σ²/2·∇m + m·α, the correct HJB BC for quadratic Hamiltonians is:
```
∂U/∂n = -σ²/2 · ∂ln(m)/∂n
```

**Four-Phase Implementation**:
1. **Phase 1: Utilities** - BC coupling computation functions
2. **Phase 2: Solver Integration** - `bc_mode` parameter + critical bug fix
3. **Phase 3: Validation** - 2.13x convergence improvement demonstrated
4. **Phase 4: Documentation** - Protocol, conventions, tutorial

**Key Components**:
- `mfg_pde/geometry/boundary/bc_coupling.py` - BC computation utilities (230 lines)
- `bc_mode` parameter in `HJBFDMSolver` ("standard" | "adjoint_consistent")
- Automatic BC computation from density gradient each Picard iteration
- Parameter threading through 6-level solver call chain
- BC value overrides in ghost cell computations

**Critical Bug Fix**:
- Fixed BC type recognition for `'no_flux'` string in `base_hjb.py`
- Previously, `neumann_bc()` objects were misinterpreted as periodic
- **Impact**: Affects ALL Neumann BC usage throughout codebase (scope beyond #574)

### Result
- ✅ 8 files changed, 1,034 additions, 12 deletions
- ✅ 11 new tests (smoke, integration, validation) - all passing
- ✅ All 40 existing HJB FDM tests pass (backward compatibility verified)
- ✅ **2.13x convergence improvement** validated (703 → 330 max error)
- ✅ Negligible computational overhead (<0.1%)
- ✅ 100% backward compatible (default `bc_mode="standard"`)
- ✅ Tutorial validated with expected results

**Implementation**:
- Core: `bc_coupling.py`, `hjb_fdm.py` (+42 lines), `base_hjb.py` (+52 lines + bug fix)
- Documentation: Protocol, CLAUDE.md patterns, design doc, tutorial (703 total lines)
- Testing: 4 smoke tests, 3 integration tests, 1 validation test, 1 tutorial validation

**Documentation**:
- Protocol: `docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md` § BC Consistency
- Conventions: `CLAUDE.md` § Boundary Condition Coupling Patterns
- Design: `docs/development/issue_574_robin_bc_design.md`
- Tutorial: `examples/tutorials/06_boundary_condition_coupling.py`

**Known Limitations** (documented):
1. 1D only (2D/nD extension planned)
2. Quadratic Hamiltonian (non-quadratic requires Issue #573 integration)
3. Scalar diffusion only (tensor diffusion not yet supported)
4. 1st-order gradient accuracy (higher-order possible)

### Why 6.5?
- **Scientific correctness** feature (not infrastructure refactoring)
- **Independent development** (parallel to infrastructure sequence)
- **Medium priority** (important but not blocking other work)
- **Inserted after completion** to maintain chronological record
- **Discovered during validation** in mfg-research experiments

---

## ✅ Priority 6.6: LinearOperator Architecture Completion (#595) - **COMPLETED**

**Issue**: [#595](https://github.com/derrring/MFG_PDE/issues/595) Phase 2
**Status**: ✅ COMPLETED (2026-01-17)
**Priority**: Medium (infrastructure improvement)
**Size**: Small
**Actual Effort**: < 1 day

### Problem
Incomplete migration to scipy.sparse.linalg.LinearOperator interface. Several differential operators still used raw matrix representations instead of LinearOperator classes.

### Solution Implemented

**Created Operator Classes**:
1. **DivergenceOperator** (`mfg_pde/geometry/operators/divergence.py`)
   - Computes ∇·F for vector fields
   - Supports 1D, 2D, nD
   - Central differences with ghost cell BC handling

2. **AdvectionOperator** (`mfg_pde/geometry/operators/advection.py`)
   - Computes b·∇u for scalar fields with drift b
   - Supports variable drift coefficients
   - 2nd-order upwind biasing

3. **InterpolationOperator** (`mfg_pde/geometry/operators/interpolation.py`)
   - Maps between staggered and collocated grids
   - Linear interpolation
   - Preserves boundary information

**Pattern**: All follow `LinearOperator` protocol:
```python
class MyOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, grid, ...):
        self.shape = (grid.size, grid.size)
        self.dtype = np.float64

    def _matvec(self, v):
        # Implement action on vector
        return result
```

### Integration

All operators integrated into:
- HJB FDM solver
- FP FDM solver
- Coupling solvers

**Benefit**: Operators work with scipy's sparse linear algebra ecosystem (iterative solvers, eigenvalue solvers, etc.)

### Result
- ✅ 100% LinearOperator coverage for geometry operators
- ✅ 39/40 HJB tests passing (1 pre-existing failure)
- ✅ Consistent scipy.sparse.linalg interface
- ✅ ~220 lines of new operator code

**Files Created**:
- `mfg_pde/geometry/operators/divergence.py` (~75 lines)
- `mfg_pde/geometry/operators/advection.py` (~80 lines)
- `mfg_pde/geometry/operators/interpolation.py` (~65 lines)

**Files Modified**:
- `mfg_pde/geometry/operators/__init__.py` - Export new operators
- `mfg_pde/geometry/operators/laplacian.py` - Ruff linting fixes

### Why 6.6?
- **Infrastructure improvement** (not user-facing)
- **Independent development** (parallel to other work)
- **Low priority** (optimization, not correctness)
- **Inserted after completion** to maintain chronological record
- **Builds on existing operator infrastructure** from earlier work

---

## ✅ Priority 6.7: Variational Inequality Constraints (#591) - **COMPLETED**

**Issue**: [#591](https://github.com/derrring/MFG_PDE/issues/591) Phase 2
**Status**: ✅ COMPLETED (2026-01-17)
**Priority**: Medium (feature addition)
**Size**: Medium
**Actual Effort**: 1 day

### Problem
No infrastructure for variational inequality constraints (obstacle problems, box constraints) in MFG solvers. Users cannot enforce state constraints like u(t,x) ≥ ψ(x).

### Solution Implemented

**Core Infrastructure**:

1. **ConstraintProtocol** (`mfg_pde/geometry/boundary/constraint_protocol.py`)
   - Protocol-based interface for duck typing
   - Three required methods: `project()`, `is_feasible()`, `get_active_set()`

2. **ObstacleConstraint** (`mfg_pde/geometry/boundary/constraints.py`)
   - Unilateral constraints: u ≥ ψ (lower) or u ≤ ψ (upper)
   - Regional support: Constraints active only in spatial subdomains
   - Projection: P_K(u) = max(u, ψ) or min(u, ψ)

3. **BilateralConstraint** (`mfg_pde/geometry/boundary/constraints.py`)
   - Box constraints: ψ_lower ≤ u ≤ ψ_upper
   - Regional support
   - Projection: P_K(u) = clip(u, ψ_lower, ψ_upper)

**Mathematical Foundation**:
```
Projection operator:  P_K(u) = argmin_{v ∈ K} ||v - u||²

Properties:
- Idempotent: P(P(u)) = P(u)
- Non-expansive: ||P(u) - P(v)|| ≤ ||u - v||
- Feasibility: P(u) ∈ K
```

### Integration

**HJB FDM Solver** (`mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`):
- Added `constraint` parameter to `__init__()`
- Projection applied after each timestep
- Both 1D and nD paths supported

**Usage**:
```python
from mfg_pde.geometry.boundary import ObstacleConstraint
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver

# Define obstacle
psi = -0.5 * (x - 0.5) ** 2

# Create constraint
constraint = ObstacleConstraint(psi, constraint_type="lower")

# Solve with constraint
solver = HJBFDMSolver(problem, constraint=constraint)
result = solver.solve(...)
```

### Validation

**Three Physics-Based Examples**:

1. **Heat Equation with Obstacle** (`examples/advanced/obstacle_problem_1d_heat.py`)
   - Cooling rod with thermostat
   - Active set growth: 0% → 68.6%
   - PDE: ∂u/∂t = σ²/2 ∂²u/∂x² - λu
   - Constraint: u ≥ ψ (parabolic, 0.3 at center, 0.1 at edges)

2. **Bilateral Constraint** (`examples/advanced/obstacle_problem_1d_bilateral.py`)
   - Temperature control with heating and cooling
   - Box constraints: ψ_lower ≤ u ≤ ψ_upper
   - Lower constraint active: 17.6% → 0.0% (releases as system warms)

3. **Regional Constraint** (`examples/advanced/obstacle_problem_1d_regional.py`)
   - Protected zone x ∈ [0.3, 0.7]
   - Constraint enforced only in protected region
   - Outside zone: Temperature drops to 0.17 < ψ = 0.4 freely
   - Protected zone: 100% active (maintained at ψ = 0.4)

All examples demonstrate:
- Perfect constraint satisfaction (zero violations)
- Numerical stability (CFL condition satisfied)
- Physical correctness (active set evolution matches physics)
- Comprehensive visualization (6-panel figures)

### Testing

**Test Suite** (`tests/unit/geometry/boundary/test_constraints.py`):
- 34 tests, all passing ✅
- Protocol compliance verification
- Projection properties (idempotence, non-expansiveness, feasibility)
- Active set detection
- Regional constraints
- Error handling

**Integration Tests**:
- 39/40 HJB FDM tests passing (1 pre-existing failure)
- No regressions from constraint addition

### Result
- ✅ Protocol-based constraint infrastructure
- ✅ Three constraint types (obstacle, bilateral, regional)
- ✅ Integration with HJB FDM solver
- ✅ 34 passing unit tests
- ✅ 3 physics-based validation examples
- ✅ Complete documentation (~2400 lines total)

**Files Created**:
- `mfg_pde/geometry/boundary/constraint_protocol.py` (~60 lines)
- `mfg_pde/geometry/boundary/constraints.py` (~640 lines)
- `tests/unit/geometry/boundary/test_constraints.py` (~370 lines)
- `examples/advanced/obstacle_problem_1d_heat.py` (~400 lines)
- `examples/advanced/obstacle_problem_1d_bilateral.py` (~420 lines)
- `examples/advanced/obstacle_problem_1d_regional.py` (~470 lines)
- `docs/development/VARIATIONAL_INEQUALITY_CONSTRAINTS_SUMMARY.md` (~440 lines)

**Files Modified**:
- `mfg_pde/geometry/boundary/__init__.py` - Export constraint classes
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py` - Constraint integration

**Known Limitations**:
1. Pre-existing HJB running cost bug prevents proper obstacle problems with Hamiltonians
2. Validation examples use heat equation to bypass HJB bug
3. 1D examples only (2D extension planned)

**Impact**: Enables research into:
- Constrained mean field games
- Optimal control with state constraints
- Free boundary problems
- Complementarity formulations

### Why 6.7?
- **Feature addition** (not infrastructure refactoring)
- **Independent development** (parallel to operator work)
- **Medium priority** (important but not blocking)
- **Inserted after completion** to maintain chronological record
- **Discovered need during research experiments** (Issue #591 Phase 1)

---

## ✅ Priority 7: Mixin Refactoring - Remaining Solvers (#545) - **COMPLETED**

**Continuation of Priority 4**
**Status**: ✅ COMPLETED (2026-01-17)
**Actual Effort**: Cleanup only (< 1 hour)

### Problem
Priority list suggested additional mixin refactoring work remained for GFDM, FDM, FEM, DGM solvers.

### Solution Implemented

**Audit Findings** (2026-01-17):
All solver mixin refactoring was already complete from Issue #545 (closed 2026-01-11):
- ✅ HJBFDMSolver: Inherits from BaseHJBSolver only (no mixins)
- ✅ HJBGFDMSolver: Uses composition with 4 components (BoundaryHandler, MonotonicityEnforcer, GridCollocationMapper, NeighborhoodBuilder)
- ✅ FPFDMSolver: Inherits from BaseFPSolver only (no mixins)
- ✅ FPGFDMSolver: Inherits from BaseFPSolver only (no mixins)
- ✅ FPParticleSolver: Uses composition with ParticleApplicator component
- ✅ All other solvers (Semi-Lagrangian, WENO, etc.): Inherit from base classes only

**Cleanup Work**:
1. Deleted `hjb_gfdm_monotonicity.py` (unused MonotonicityMixin - 28KB dead code)
2. Deleted 3 compiled mixin .pyc files (gfdm_boundary_mixin, gfdm_interpolation_mixin, gfdm_stencil_mixin)
3. Updated 5 outdated comments in hjb_gfdm.py referencing "MonotonicityMixin" → "MonotonicityEnforcer component"

### Result
- ✅ No active mixin usage in solver code
- ✅ All solvers use explicit composition or simple inheritance
- ✅ 39/40 HJB FDM tests passing (1 pre-existing failure)
- ✅ Zero regressions from cleanup

**Note**: FEM/DGM mentioned in original priority do not exist as traditional PDE solvers. FEM BC applicators exist in geometry layer (already using proper architecture). Neural DGM inherits from BaseNeuralSolver only (no mixins).

---

## ✅ Priority 8: Legacy Parameter Deprecation (#544) - **COMPLETED**

**Issue**: [#544](https://github.com/derrring/MFG_PDE/issues/544)
**Priority**: High
**Size**: Large
**Status**: ✅ **PHASE 1 & 2 COMPLETE** (2026-01-18)
**Actual Effort**: 2 days

### Problem
MFGProblem supports both legacy (`Nx`, `xmin`) and modern (Geometry) APIs, creating bloat.

### Solution Implemented (Phased)

**Phase 1: Add DeprecationWarning (v0.17.1)** ✅ **COMPLETED**
- ✅ Add DeprecationWarning to MFGProblem.__init__
- ✅ Migrate representative tests to Geometry API (4 files)
- ✅ Document migration in `docs/migration/LEGACY_PARAMETERS.md`
- ✅ Verify examples already use Geometry API
- **Commit**: `a6e54d21` (2026-01-18)

**Phase 2: Migrate Test Suite (v0.17.1)** ✅ **COMPLETED**
- ✅ Migrate all remaining test files to Geometry API (7 files, 23 calls)
- ✅ Fix mock objects for Geometry API compatibility
- ✅ Verify zero test regressions (79 + 23 + 12 passing)
- **Commit**: `83f2031c` (2026-01-18)

**Phase 3: Remove Legacy Parameters (v1.0.0)** ⏳ **PLANNED**
- [ ] Remove Nx, xmin, xmax, spatial_bounds parameters
- [ ] Remove _override attributes and legacy grid construction
- [ ] Simplify MFGProblem.__init__ (target < 200 lines from ~600)
- [ ] Remove TestLegacy1DMode tests
- [ ] Archive migration guide
- **Timeline**: 6-12 months (v1.0.0 release)

### Summary
Phase 1 & 2 complete. All internal code uses modern Geometry API. Users have 6-12 month deprecation period before v1.0.0 removal.

---

## Summary Timeline

| Week | Priority | Issue | Deliverable |
|:-----|:---------|:------|:------------|
| **Week 1** | P1 | #542 | FDM BC fix + validation |
| **Week 2** | P2 | #547 | Silent fallback elimination |
| **Week 3** | P3 | #543 Phase 1 | Core hasattr → isinstance |
| **Week 4** | P4a | #545 Phase 1 | BoundaryHandler protocol |
| **Week 5** | P4b | #545 Phase 2 | FPParticle composition |
| **Week 6-7** | P5 | #543 Phase 2 | Algorithm hasattr cleanup |
| **Week 8-9** | P6 | #545 Phase 3 | Other solver refactoring |
| **Week 10-11** | P7 | #544 Phase 2 | Legacy deprecation |
| **Week 12** | P8 | #543 Phase 3 | Utils hasattr cleanup |

**Total Estimated Duration**: ~12 weeks for all priorities

---

## Dependencies Graph

```
P1 (#542 FDM BC) ────────────────────┐
    └─ Independent                   │
                                     │
P2 (#547 Silent Fallbacks) ──────────┤
    └─ Independent                   │
                                     │
P3 (#543 Phase 1: Core) ─────────────┤
    └─ Builds on GeometryProtocol    │
           │                         │
           ├──> P4a (BoundaryHandler)│
           │       │                 │
           │       └──> P4b (FPParticle Template)
           │              │          │
           └──> P5 (#543 Phase 2)   │
                   │                 │
                   └──> P6 (Other Solvers)
                           │         │
                           └──> P7 (#544 Deprecation)
                                  │  │
                                  │  └──> P8 (#543 Phase 3)
                                  │
                                  └─ All infrastructure stable
```

---

## Principles

### 1. **Correctness First**
Fix numerical bugs (#542) before architectural refactoring.

### 2. **Fail Fast & Surface Problems**
Eliminate silent fallbacks (#547) early to improve debugging.

### 3. **Foundation Before Facades**
Complete protocol work (#543 core) before large refactoring (#545, #544).

### 4. **Incremental Progress**
Break large issues into phases to maintain momentum.

### 5. **Validate Continuously**
Run tests after each priority, validate with research experiments.

---

## Notes

- **Issue #546 (sys.exit removal)**: ✅ CLOSED in v0.16.16
- **GeometryProtocol**: ✅ Foundation complete in v0.16.16
- **Parallel Work**: P1 and P2 can proceed in parallel if needed
- **Flexibility**: Timeline is estimate; adjust based on discoveries during work

---

**Last Updated**: 2026-01-18
**Completed**: P1 (#542), P2 (#547), P3 (#543 Phase 1), P3.5 (#580), P3.6 (#576), P4 (#545), P5 (#543 Phase 2), P5.5 (#587), P6 (#543 Phase 3), P6.5 (#574), P6.6 (#595), P6.7 (#591), P7 (#545), P8 (#544 Phases 1-2), **#573**
**Current Focus**: ✅ Infrastructure complete through Priority 8! Starting geometry trait system (#590) for operator abstraction framework.

## Recently Completed (2026-01-18)

### ✅ Issue #573: Non-Quadratic Hamiltonian Support - **COMPLETED**

**Issue**: [#573](https://github.com/derrring/MFG_PDE/issues/573)
**Status**: ✅ CLOSED (2026-01-18)
**Priority**: Medium
**Size**: Medium
**Actual Effort**: 1 day

### Problem
FP solvers assumed quadratic Hamiltonians (α* = -∇U), preventing use with L1, quartic, or constrained control problems.

### Solution Implemented
Clarified that `drift_field` parameter accepts drift velocity α* for ANY Hamiltonian, not just quadratic.

**Key Insight**: The API was already correct - only documentation needed clarification!

**Changes**:
- Updated FP FDM/GFDM docstrings with non-quadratic examples
- Added test suite: `test_fp_nonquadratic.py` (8/8 passing)
- Created demonstration: `examples/advanced/mfg_l1_control.py`

**Commits**:
- `f5cb1039` - Documentation clarification + tests
- `1c13a450` - L1 control example

**Usage Pattern**:
```python
# For ANY Hamiltonian: caller computes α* = -∂_p H(∇U)
alpha_L1 = -np.sign(grad_U)  # L1 control
M = fp_solver.solve_fp_system(m0, drift_field=alpha_L1)
```

---

## Remaining Open Issues (by priority)

| Priority | Issue | Description | Size | Status |
|:---------|:------|:------------|:-----|:-------|
| HIGH | #590 | Phase 1: Geometry Trait System | Medium | 🎯 **IN PROGRESS** |
| HIGH | #596 | Phase 2: Solver Integration with Traits | Large | Blocked by #590 |
| HIGH | #589 | Geometry/BC Architecture (Master Tracking) | Large | In Progress |
| MEDIUM | #598 | BCApplicatorProtocol → ABC refactoring | Medium | Open |
| MEDIUM | #597 | FP Operator Refactoring | Large | Open |
| MEDIUM | #549 | BC framework for non-tensor geometries | Large | Open |
| MEDIUM | #535 | BC framework enhancement | Large | Open |
| MEDIUM | #489 | Direct particle query for coupling | Large | Open |
| LOW | #577 | Neumann BC ghost cell consolidation (Phase 3) | Small | Phases 1-2 complete |
| LOW | #523 | MMS validation suite for BC | Medium | Open |
| LOW | #521 | 3D corner handling | Large | Open |
| LOW | #517 | Semantic dispatch factory | Medium | Open |
| ~~MEDIUM~~ | ~~#573~~ | ~~Non-quadratic Hamiltonian support~~ | ~~Medium~~ | ✅ Closed (f5cb1039, 1c13a450) |
| ~~LOW~~ | ~~#571~~ | ~~test_geometry_benchmarks: Missing fixture~~ | ~~Small~~ | ✅ Fixed (8997589b) |
| ~~LOW~~ | ~~#570~~ | ~~test_particle_gpu_pipeline: Shape mismatch~~ | ~~Small~~ | ✅ Fixed (8997589b) |
| LOW | Others | Various infrastructure/features | Large | Open |
