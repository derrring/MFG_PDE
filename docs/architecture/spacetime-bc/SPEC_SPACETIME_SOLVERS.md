# [SPEC] Space-Time Solvers & Boundaries Architecture

**Document ID**: MFG-SPEC-ST-0.8
**Status**: DRAFT INTEGRATION
**Date**: 2026-01-30
**Dependencies**: MFG-SPEC-GEO-1.0, MFG-SPEC-BC-0.2

---

## 1. Executive Summary

This specification defines the architecture for **Time Integration** and
**Global Solution** modules of MFG_PDE. It marks a paradigm shift from
imperative time-stepping loops to declarative **Space-Time Operators**.

**Core Philosophy**:
1. **The Cylinder Manifold**: Time $t$ is a dimension parallel to space $x$.
   Domain is the space-time cylinder $\mathcal{Q} = [0,T] \times \Omega$.
2. **Boundary Unification**: IC, TC, and spatial BCs are boundary data on $\partial\mathcal{Q}$.
3. **Solver Agnosticism**: Both **Sequential Marching** and **Global Assembly**
   under a unified `TrajectorySolver` protocol.
4. **Memory Safety**: Explicit `StoragePolicy` traits manage in-core vs out-of-core.

---

## 2. Mathematical Foundation

### 2.1 The Space-Time Cylinder

Computational domain: $\mathcal{Q} = [0,T] \times \Omega$.

Boundary $\partial\mathcal{Q}$ decomposes into:
1. **Lateral Surface** ($\Gamma_{\text{lat}}$): $[0,T] \times \partial\Omega$ — Spatial BCs.
2. **Bottom Cap** ($\Gamma_{\text{bot}}$): $\{0\} \times \Omega$ — Initial Condition.
3. **Top Cap** ($\Gamma_{\text{top}}$): $\{T\} \times \Omega$ — Terminal Condition.

### 2.2 The Generalized Operator

MFG system as coupled BVP on $\mathcal{Q}$:

$$
\begin{cases}
\mathcal{L}_{\text{HJB}}(u, m) = 0 & \text{in } \mathcal{Q} \\
\mathcal{L}_{\text{FP}}(u, m) = 0 & \text{in } \mathcal{Q} \\
\mathcal{B}(u, m) = g & \text{on } \partial\mathcal{Q}
\end{cases}
$$

**Note on equation type**: The character of $\mathcal{L}$ depends on viscosity:
- **Second-order MFG** ($\sigma > 0$): Both HJB and FP are **parabolic**.
- **First-order MFG** ($\sigma = 0$): HJB becomes Hamilton-Jacobi (hyperbolic characteristics);
  FP becomes a transport/conservation law (hyperbolic). This regime requires
  characteristic-aware BC treatment (Lopatinskii-Shapiro, upwinding).

---

## 3. Type System: Solver Traits

### 3.1 StoragePolicy (Memory Management)

> **Status**: DEFERRED for MFG_PDE v1.0. Documented for future reference.

- **`InCore`**: Full trajectory $(N_t, N_x)$ in RAM/VRAM. Enables global optimization and AD.
- **`Streaming`**: Only current time-slab stored. Standard time-stepping.
- **`Checkpointed`**: Keyframes stored; intermediate steps recomputed. Optimal for adjoint methods.
- **`OutCore`**: Memory-mapped to disk (HDF5/Zarr).

### 3.2 SolverStrategy (Algorithm Selection)

- **`Sequential`**: Classical marching ($t \to t+\Delta t$). $O(N_t)$ serial complexity.
- **`GlobalDirect`**: Monolithic sparse solve. $O((N_t N_x)^{1.5})$. Memory intensive.
- **`GlobalIterative`**: GMRES/BiCGSTAB on space-time matrix.
- **`Parareal`**: Parallel-in-time decomposition. Research-grade.

---

## 4. Core Abstractions

### 4.1 SpacetimeField (Data Container)

Unified field representation over $\mathcal{Q}$, aware of storage policy.

```python
@dataclass
class SpacetimeField:
    domain: Geometry
    storage: StoragePolicy
    _data_backend: np.ndarray | zarr.Array | CheckpointBuffer

    def get_slice(self, t_idx: int) -> np.ndarray:
        """Spatial field at t_idx. May raise AccessError if Streaming."""
        ...
```

### 4.2 SpacetimeBoundaryData (Unified BC Container)

> **Status**: ADOPTED for Phase 2 of migration plan.

```python
@dataclass
class SpacetimeBoundaryData:
    # Lateral: [0,T] x dOmega (spatial BCs, possibly time-varying)
    spatial_bc: BoundaryConditions

    # Bottom cap: {0} x Omega
    initial_condition: NDArray | Callable | None = None

    # Top cap: {T} x Omega
    terminal_condition: NDArray | Callable | None = None

    def validate_corner_consistency(self, geometry, tol=1e-6):
        """Check compatibility at cylinder corners (t=0 and t=T on dOmega)."""
        ...
```

### 4.3 TrajectorySolver (Protocol)

> **Status**: ADOPTED for Phase 3 of migration plan.

```python
class TrajectorySolver(Protocol):
    def solve_trajectory(
        self,
        boundary_data: SpacetimeBoundaryData,
        coupling_field: SpacetimeField | None = None,
    ) -> SpacetimeField:
        ...
```

---

## 5. Implementation Strategies

### 5.1 Strategy A: SequentialMarchingSolver (Backward Compatible)

Wraps existing solver logic. First implementation target.

```python
class SequentialMarchingSolver(TrajectorySolver):
    def __init__(self, step_op: StepOperator, direction: str):
        self.step_op = step_op
        self.direction = direction  # 'forward' or 'backward'

    def solve_trajectory(self, boundary_data, coupling_field=None):
        # The time loop lives here, encapsulated.
        # Respects boundary_data.initial/terminal conditions.
        ...
```

### 5.2 Strategy B: GlobalSpacetimeSolver (Advanced)

> **Status**: DEFERRED. Research contribution, not infrastructure.

Assembles full space-time matrix: $\mathcal{A} = D_t \otimes M_x + I_t \otimes L_x$.

Key features:
- ALE support for moving meshes (mesh velocity advection $\mathbf{w} \cdot \nabla$)
- Mandatory preconditioning (block circulant)
- Enables "one-shot" LQ-MFG solving without Picard iterations

---

## 6. Interaction Protocols

### 6.1 Corner Consistency Protocol

> **Status**: ADOPTED for Phase 2c.

Ensures well-posedness at cylinder corners:
1. **Validator**: Check $\lim_{t\to 0} \text{BC}(t,x) = \text{IC}(x)$ before solving.
2. **Smoothing**: If inconsistent, apply transition layer (optional) or emit warning.
3. **Severity**: WARNING for standard FDM (tolerates discontinuity), ERROR for high-order.

### 6.2 ALE Protocol (Moving Mesh)

> **Status**: DEFERRED. No current MFG use case.

When geometry is dynamically deforming:
- Sequential: Update mesh coordinates $x(t)$ at each step.
- Global: Store 4D tensor $(t, x, y, z)$; assembler adds skew term $-\mathbf{w} \cdot \nabla u$.

---

## 7. Migration Path

- **Phase 1 (Wrappers)**: Implement `TrajectorySolver` protocol. Wrap existing solvers.
  Introduce `SpacetimeBoundaryData` as optional container.
- **Phase 2 (Unification)**: Deprecate raw loops. Force all solvers to consume
  `SpacetimeBoundaryData`.
- **Phase 3 (Global)**: Release `GlobalSpacetimeSolver` for LQ-MFG. Enables one-shot
  solving without Picard iterations. Research-grade.

---

## 8. Extension Specs

Each of the following subsystems has been expanded into a dedicated design
document with current-state analysis, proposed trait system, and migration path.

### 8.1 Time Integration System → `SPEC_TIME_INTEGRATION.md`

**Document ID**: MFG-SPEC-TI-0.1 | **Status**: Deferred

Covers: `StepOperator` protocol, `TimeIntegrator` driver, scheme traits
(SchemeType, TemporalOrder, StorageClass, AdaptivityMode). Maps current
HJB backward Euler + Newton and FP implicit/explicit schemes to the
proposed trait system.

> Deferred: MFG has fixed time structure (HJB backward, FP forward).
> Phase A (annotate) is low effort; Phase B (extract StepOperator) is medium.

### 8.2 Operator System → `SPEC_OPERATOR_SYSTEM.md`

**Document ID**: MFG-SPEC-OP-0.1 | **Status**: Phase 4 (parallel track)

Covers: `PDEOperator` base class with operator algebra (`+`, `*`, `@`),
`OperatorTraits` metadata (DifferentialType, StencilWidth, Conservation,
UpwindScheme), `CompositeOperator` for lazy composition, and `to_sparse()`
export. Maps all 8 current operators to the trait system.

> Partially implemented via `SupportsLaplacian`, `SupportsGradient` protocols.
> Issue #658 Phases 0-2 complete; Phases 3+ deferred.

### 8.3 Linear Solver System → `SPEC_LINEAR_SOLVER.md`

**Document ID**: MFG-SPEC-LS-0.1 | **Status**: Deferred

Covers: `LinearSolverTraits` (MatrixStructure, MatrixCoupling, MatrixOrigin),
`select_solver()` auto-selection, operator-to-traits inference. Analyzes
all 5 linear solve sites in the codebase and their matrix properties.

> Deferred: `scipy.sparse.spsolve` is not the bottleneck for current
> problem sizes (1D-2D, $N < 10^4$). Becomes relevant for 3D ($N > 10^5$).

---

**Last Updated**: 2026-02-05
