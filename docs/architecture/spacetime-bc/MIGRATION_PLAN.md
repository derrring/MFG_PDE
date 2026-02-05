# Migration Plan: Space-Time & BC Architecture

**Date**: 2026-02-05
**Principle**: Cherry-pick what serves MFG. Defer general PDE framework scope.

---

## Phase 0: Documentation Consolidation ✅ COMPLETED (2026-02-05)

**Effort**: Small | **Blocks**: Nothing

Consolidated BC docs per Issue #729. All moved to `docs/archive/bc_completed_2026-02/`.

### Archived Files (9 total)

**From `docs/architecture/`** (4 files — superseded by this project):
- `BC_AUDIT_RESPONSE.md`, `BC_COMPLETE_WORKFLOW.md`, `BC_FLOW_CLARIFICATION.md`,
  `BC_SPECIFICATION_VS_APPLICATOR.md`

**From `docs/development/`** (3 completed + 2 superseded):
- `issue_574_robin_bc_design.md` (implemented v0.17.1)
- `issue_597_milestone2_bc_fix_completion.md` (complete)
- `[SUPERSEDED]_ic_bc_geometry_compatibility.md` (superseded by #681)
- `bc_architecture_analysis.md` → consolidated into `boundary_condition_handling_summary.md`
- `matrix_assembly_bc_protocol.md` → consolidated into `boundary_condition_handling_summary.md`
- `terminal_bc_compatibility.md` → consolidated into `boundary_condition_handling_summary.md`
- `[SUPERSEDED]_spacetime_boundary_unification.md` → replaced by this project
- `[SUPERSEDED]_spacetime_operator_architecture_proposal.md` → replaced by this project

**Kept**: `GEOMETRY_AND_TOPOLOGY.md` (separate topic)

---

## Phase 1: BC Formalization & Applicator Cleanup (v0.17.x)

**Effort**: Medium | **Issues**: #712, #517

### 1a. Merge ImplicitApplicator + MeshfreeApplicator

**Why**: User confusion ("implicit IS meshfree"). Duplicated field BC logic.

**Approach**: Option C from Issue #712 — single `GeometryApplicator`:

```python
class GeometryApplicator(BaseBCApplicator):
    """BC applicator for geometry-based (non-grid) methods."""

    def apply_particle_bc(self, particles, bc_type) -> NDArray:
        """Position transforms for Lagrangian methods."""
        ...

    def apply_field_bc(self, u, points, bc_type, bc_value) -> NDArray:
        """Field value enforcement at boundary points."""
        ...
```

**Migration**:
1. Create `GeometryApplicator` combining both classes
2. Deprecate `ImplicitApplicator` and `MeshfreeApplicator` with redirects
3. Update GFDM solver and particle solver imports
4. Remove deprecated classes after 3 minor versions

### 1b. Formalize 3-Axis BC Naming

**Why**: Current BCSegment already implements the 3-axis model but without
explicit naming. Making it explicit improves documentation and validation.

**Approach**: Add docstring-level documentation and type alias clarity.
No new classes needed — BCSegment already IS the `BCSpec`:

```python
# Document the implicit axes in BCSegment docstring:
# Axis 1 (Region):    boundary / region / sdf_region / region_name
# Axis 2 (MathType):  bc_type: BCType
# Axis 3 (ValueSource): value: float | Callable | BCValueProvider
# (Enforcement is solver-side, not spec-side)
```

---

## Phase 2: SpacetimeBoundaryData (v0.18.x)

**Effort**: Medium | **Issues**: #679, #682

### 2a. Create the Container

Unify the three boundary components of $\partial\mathcal{Q}$ into one object:

```python
@dataclass
class SpacetimeBoundaryData:
    """Boundary data on the space-time cylinder dQ = [0,T] x Omega."""

    # Lateral: [0,T] x dOmega
    spatial_bc: BoundaryConditions

    # Bottom cap: {0} x Omega
    initial_condition: NDArray | Callable | None = None

    # Top cap: {T} x Omega
    terminal_condition: NDArray | Callable | None = None

    def validate_corner_consistency(self, geometry, tol=1e-6) -> ValidationResult:
        """Check IC/BC and TC/BC compatibility at cylinder corners."""
        ...
```

### 2b. Wire into MFGProblem

Replace the scattered attributes:

```python
# Before:
problem.boundary_conditions  # spatial BC
problem.m_initial            # IC (via MFGComponents)
problem.u_final              # TC (via MFGComponents)

# After:
problem.hjb_boundary_data    # SpacetimeBoundaryData for HJB
problem.fp_boundary_data     # SpacetimeBoundaryData for FP
```

Note: HJB and FP have different temporal directions:
- HJB: TC is the "initial" data (backward equation), IC is "terminal"
- FP: IC is the "initial" data (forward equation), TC is "terminal"

The SpacetimeBoundaryData container uses mathematical convention (IC = t=0, TC = t=T)
and lets solvers interpret direction.

### 2c. Corner Consistency Validation

Check that boundary data is compatible at cylinder corners:
- $\lim_{t \to 0^+} \text{BC}(t,x) = \text{IC}(x)$ for $x \in \partial\Omega$
- $\lim_{t \to T^-} \text{BC}(t,x) = \text{TC}(x)$ for $x \in \partial\Omega$

Violations cause Gibbs phenomena in high-order solvers. Emit `ValidationResult`
with severity WARNING (not ERROR — many practical problems have corner
discontinuities that work fine with low-order FDM).

---

## Phase 3: TrajectorySolver Protocol (v0.19.x / post-v1.0)

**Effort**: Large | **Issues**: #476, #634

### 3a. Define the Protocol

```python
class TrajectorySolver(Protocol):
    """Solves a PDE over the full time horizon."""

    def solve_trajectory(
        self,
        boundary_data: SpacetimeBoundaryData,
        coupling_field: SpacetimeField | None = None,
    ) -> SpacetimeField:
        ...
```

### 3b. Wrap Existing Solvers

Create `SequentialMarchingSolver` as a thin wrapper around existing solvers:

```python
class SequentialMarchingSolver(TrajectorySolver):
    """Wraps existing time-stepping solvers into TrajectorySolver interface."""

    def __init__(self, solver: BaseHJBSolver | BaseFPSolver, direction: str):
        self.solver = solver
        self.direction = direction  # 'forward' or 'backward'

    def solve_trajectory(self, boundary_data, coupling_field=None):
        # Delegate to existing solver.solve_hjb_system() or solve_fp_system()
        # The time loop still lives inside the concrete solver.
        ...
```

This provides the new interface without refactoring solver internals.

### 3c. Extract StepOperator (optional, if needed)

Only if concrete use case demands pluggable time integration:

```python
class StepOperator(Protocol):
    """Single time-step operator: u^{n+1} = Step(u^n)."""

    def step(self, u_current: NDArray, t: float, dt: float,
             coupling: NDArray | None = None) -> NDArray:
        ...
```

This requires separating spatial operator, BC enforcement, and time loop
in every solver. High effort, uncertain payoff for MFG.

---

## Phase 4: Operator Cleanup (v0.19.x)

**Effort**: Medium | **Issues**: #658

Unify the operator interface. This is independent of the space-time work
and can proceed in parallel.

### Goals

1. All operators implement `scipy.sparse.linalg.LinearOperator` interface
2. Operator algebra: `L1 + L2`, `alpha * L`, `L1 @ L2`
3. Geometry queries capabilities via existing `Supports*` protocols
4. BC-aware operator construction (operator encapsulates BC enforcement)

### Non-Goals (for MFG)

- Lazy computation graphs
- Geometry-agnostic equation code ("same code on cube and Mobius strip")
- JIT compilation of operator kernels

---

## Explicitly Deferred

| Item | Reason | Revisit When |
|:-----|:-------|:-------------|
| GlobalSpacetimeSolver | Research contribution, requires space-time matrix assembly | After published paper demonstrates value |
| StoragePolicy | No scaling problem; all current problems fit in RAM | When 3D MFG problems become standard |
| GKS / SBP-SAT formal infrastructure | First-order MFG is hyperbolic but already handled via Godunov/WENO | When formal energy-stability proofs needed beyond upwinding |
| Parareal | Parallel-in-time is extremely ambitious | Never (different project scope) |
| ALE (moving mesh) | No moving domain use case in MFG | If crowd models on deforming domains emerge |
| Time Integrator Traits | MFG has fixed time structure (HJB backward, FP forward) | If multi-physics coupling introduced |
| Linear Solver Traits | scipy.sparse.spsolve is not the bottleneck | When problems exceed 10^5 spatial DOFs |
| TPMS/Sphere periodicity | Materials science, not MFG | Never (wrong project) |

---

## Dependency Graph

```
Phase 0 (docs consolidation)
    │
    ▼
Phase 1a (applicator merge)     Phase 1b (BC formalization)
    │                                │
    ▼                                ▼
Phase 2a (SpacetimeBoundaryData container)
    │
    ▼
Phase 2b (wire into MFGProblem)
    │
    ├── Phase 2c (corner validation)
    │
    ▼
Phase 3a (TrajectorySolver protocol)
    │
    ▼
Phase 3b (SequentialMarchingSolver wrapper)
    │
    ▼
Phase 3c (StepOperator extraction — optional)

Phase 4 (operator cleanup) — independent, parallel track
```

---

**Last Updated**: 2026-02-05
