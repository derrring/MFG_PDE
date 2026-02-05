# [SPEC] Time Integration System

**Document ID**: MFG-SPEC-TI-0.1
**Status**: DRAFT (Deferred — documented for future reference)
**Date**: 2026-02-05
**Dependencies**: MFG-SPEC-ST-0.8 (TrajectorySolver protocol)

---

## 1. Motivation

MFG_PDE's time integration is currently embedded inside individual solvers.
Each solver owns its time loop, scheme selection, and step-size logic. This
works for the current solver count (7 HJB + 4 FP) but creates issues:

1. **Code duplication**: Each solver reimplements backward/forward marching.
2. **Inflexibility**: Changing from implicit Euler to Crank-Nicolson requires
   modifying solver internals, not swapping a component.
3. **Testing**: Time integration correctness is tested only through full solver
   tests, never in isolation.

This spec defines a **trait-based time integration system** that could
decouple time stepping from spatial discretization.

---

## 2. Current State

### 2.1 HJB Time Integration

**Location**: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py`

| Aspect | Implementation |
|:-------|:---------------|
| Direction | Backward ($n = N_t-2$ down to $0$) |
| Scheme | Implicit (Newton linearization at each step) |
| Step size | Fixed: $\Delta t = T / N_t$ |
| Nonlinear solve | Newton iteration with sparse Jacobian |
| Linear solve | `scipy.sparse.linalg.spsolve(J, -F)` |
| BC enforcement | After Newton convergence, per-step |
| Adaptivity | None |

**Time discretization**:
$$\frac{U^n - U^{n+1}}{\Delta t} + H(\nabla U^n, x, t_n, m^n) - \frac{\sigma^2}{2}\Delta U^n = 0$$

Newton solves the nonlinear system $F(U^n) = 0$ at each time step.

### 2.2 FP Time Integration

**Location**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm_time_stepping.py`

Three variants coexist, selected by problem characteristics:

| Variant | Scheme | When Used | Stability |
|:--------|:-------|:----------|:----------|
| `solve_timestep_full_nd()` | Fully implicit | MFG-coupled mode (default) | Unconditional |
| `solve_timestep_tensor_explicit()` | Forward Euler | Tensor diffusion $\nabla\cdot(\Sigma\nabla m)$ | CFL-limited |
| `solve_timestep_explicit_with_drift()` | Lie splitting (implicit diffusion + explicit advection) | Callable drift | Conditional |

**Implicit full-system** (default):
$$(I/\Delta t + A_{\text{adv}} + D_{\text{diff}}) \, m^{k+1} = m^k / \Delta t$$

Assembled as sparse matrix, solved via `spsolve()`.

**Forward Euler** (tensor diffusion):
$$m^{k+1} = m^k + \Delta t \bigl(\nabla\cdot(\Sigma\nabla m^k) - \nabla\cdot(\alpha m^k)\bigr)$$

### 2.3 Coupling Layer

**Location**: `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py`

The Picard iteration drives both time loops:
1. HJB solves backward: $U^{(k)}(t)$ given $M^{(k-1)}(t)$
2. FP solves forward: $M^{(k)}(t)$ given $U^{(k)}(t)$
3. Damping: $U \leftarrow (1-\theta)U_{\text{old}} + \theta U_{\text{new}}$
4. Convergence check on $\|U^{(k)} - U^{(k-1)}\|$ and $\|M^{(k)} - M^{(k-1)}\|$

Time synchronization: both solvers use identical $\Delta t = T/N_t$.

### 2.4 What's Missing

| Gap | Impact | Severity |
|:----|:-------|:---------|
| No `StepOperator` abstraction | Cannot swap time schemes without modifying solvers | Low (MFG has fixed structure) |
| No adaptive $\Delta t$ | Efficiency loss on problems with varying time scales | Low (uniform $\Delta t$ sufficient) |
| No CFL enforcement | Explicit FP schemes can silently blow up | Medium |
| No scheme metadata | Cannot query order, stability region, storage cost | Low |
| Implicit/explicit selection hardcoded | Cannot experiment with IMEX schemes | Low |

---

## 3. Proposed Trait System

### 3.1 Atomic Traits

```python
class SchemeType(Enum):
    """How the scheme treats the spatial operator."""
    EXPLICIT = "explicit"       # u^{n+1} = f(u^n) — CFL-limited
    IMPLICIT = "implicit"       # A u^{n+1} = f(u^n) — requires linear solve
    IMEX = "imex"              # Split: implicit diffusion + explicit advection

class TemporalOrder(Enum):
    """Formal order of accuracy in time."""
    FIRST = 1                   # Euler, backward Euler
    SECOND = 2                  # Crank-Nicolson, Heun, SSP-RK2
    THIRD = 3                   # SSP-RK3
    FOURTH = 4                  # RK4, LSERK4

class StorageClass(Enum):
    """Memory model for intermediate stages."""
    STANDARD = "standard"       # Store all stages (K stages = K arrays)
    LOW_STORAGE = "low_storage" # 2N or 3N storage (Williamson-type)

class AdaptivityMode(Enum):
    """Time step adaptation strategy."""
    FIXED = "fixed"             # Constant dt
    PID = "pid"                 # PID controller on local error estimate
    CFL = "cfl"                 # CFL-based for explicit schemes
```

### 3.2 StepOperator Protocol

```python
class StepOperator(Protocol):
    """Single time-step operator: u^{n+1} = Step(u^n, t, dt)."""

    @property
    def scheme_type(self) -> SchemeType: ...

    @property
    def temporal_order(self) -> TemporalOrder: ...

    def step(
        self,
        u_current: NDArray,
        t: float,
        dt: float,
        coupling: NDArray | None = None,
    ) -> NDArray:
        """Advance one time step."""
        ...
```

### 3.3 Concrete Schemes (Generative Matrix)

The trait axes generate all relevant schemes:

| Scheme | SchemeType | Order | Storage | MFG Relevance |
|:-------|:-----------|:------|:--------|:--------------|
| Forward Euler | Explicit | 1 | Standard | FP with tensor diffusion |
| Backward Euler | Implicit | 1 | Standard | **HJB (current)**, FP implicit |
| Crank-Nicolson | Implicit | 2 | Standard | Higher-order HJB/FP |
| SSP-RK2 | Explicit | 2 | Standard | Hyperbolic FP ($\sigma=0$) |
| SSP-RK3 | Explicit | 3 | Standard | WENO time integration |
| RK4 | Explicit | 4 | Standard | Semi-Lagrangian HJB |
| LSERK4 | Explicit | 4 | Low-Storage | Memory-constrained 3D |
| IMEX-Euler | IMEX | 1 | Standard | Split diffusion/advection |
| IMEX-CN/AB2 | IMEX | 2 | Standard | Higher-order splitting |

**Bold** = currently implemented (implicitly, inside solvers).

### 3.4 TimeIntegrator (High-Level)

```python
class TimeIntegrator:
    """Drives a StepOperator over a time interval."""

    def __init__(
        self,
        step_op: StepOperator,
        direction: Literal["forward", "backward"],
        adaptivity: AdaptivityMode = AdaptivityMode.FIXED,
    ):
        self.step_op = step_op
        self.direction = direction
        self.adaptivity = adaptivity

    def integrate(
        self,
        u_initial: NDArray,
        t_span: tuple[float, float],
        dt: float,
        coupling_trajectory: NDArray | None = None,
    ) -> NDArray:
        """Integrate over [t0, T] (forward) or [T, t0] (backward).

        Returns:
            Solution trajectory of shape (Nt+1, *spatial_shape).
        """
        ...

    def cfl_limit(self, spatial_operator, dx: float) -> float:
        """Estimate maximum stable dt for explicit schemes."""
        if self.step_op.scheme_type == SchemeType.IMPLICIT:
            return float("inf")
        # Scheme-dependent CFL number
        ...
```

---

## 4. Interaction with Existing Architecture

### 4.1 Relationship to TrajectorySolver

`TrajectorySolver` (MFG-SPEC-ST-0.8 §4.3) consumes `SpacetimeBoundaryData` and returns
`SpacetimeField`. The `TimeIntegrator` would live **inside** a `SequentialMarchingSolver`:

```
TrajectorySolver (protocol)
└── SequentialMarchingSolver
    ├── TimeIntegrator          ← this spec
    │   └── StepOperator        ← spatial + temporal in one step
    └── SpacetimeBoundaryData   ← IC/TC/spatial BC
```

### 4.2 Relationship to Coupling

The Picard iterator would not change. It continues to call
`hjb_solver.solve_hjb_system()` and `fp_solver.solve_fp_system()`.
The `StepOperator` extraction is internal to each solver.

### 4.3 Newton as a StepOperator

The current HJB Newton solve IS a `StepOperator` — it takes $U^{n+1}$
and produces $U^n$ (backward). Extracting it requires separating:
1. **Residual computation**: $F(U^n) = (U^n - U^{n+1})/\Delta t + H(\nabla U^n) - (\sigma^2/2)\Delta U^n$
2. **Jacobian assembly**: $J = \partial F / \partial U^n$
3. **Newton loop**: Iterate $\delta U = -J^{-1}F$ until convergence

Currently all three are interleaved in `solve_hjb_timestep_newton()`.

---

## 5. Migration Path

### Phase A: Document & Annotate (v0.18.x) — Low effort

Add docstring annotations to existing solvers identifying the implicit
time integration scheme:

```python
class HJBFDMSolver:
    """HJB solver using FDM with implicit backward Euler + Newton.

    Time integration: Backward Euler (SchemeType.IMPLICIT, Order 1)
    Direction: backward (t = T → 0)
    Nonlinear solve: Newton iteration
    """
```

### Phase B: Extract StepOperator (v0.19.x) — Medium effort

For **one solver** (FP FDM, simplest case), extract the spatial operator
from the time loop:

```python
class FPImplicitStep(StepOperator):
    """One implicit Euler step for Fokker-Planck."""

    scheme_type = SchemeType.IMPLICIT
    temporal_order = TemporalOrder.FIRST

    def __init__(self, diffusion_op, advection_op, bc):
        self.A = diffusion_op
        self.B = advection_op
        self.bc = bc

    def step(self, m_current, t, dt, coupling=None):
        # (I/dt + A + B) m_next = m_current / dt
        lhs = eye(self.A.shape[0]) / dt + self.A + self.B
        rhs = m_current.ravel() / dt
        return spsolve(lhs, rhs).reshape(m_current.shape)
```

### Phase C: Generalize (post-v1.0) — Large effort

- Extract `StepOperator` from HJB Newton solver
- Implement SSP-RK3 for WENO time integration
- Add CFL estimation utility
- Wire `TimeIntegrator` into `SequentialMarchingSolver`

---

## 6. Explicitly Deferred

| Item | Reason | Revisit When |
|:-----|:-------|:-------------|
| Adaptive $\Delta t$ (PID controller) | MFG uses fixed $\Delta t = T/N_t$; no multi-scale problems yet | When stiff coupling or multi-scale problems arise |
| Low-storage RK (LSERK4) | Only relevant for 3D problems exceeding RAM | When 3D MFG becomes standard |
| Parareal (parallel-in-time) | Research-grade, extremely ambitious | Never (different project scope) |
| Deferred correction | Higher-order temporal accuracy via SDC | When formal convergence proofs needed |
| Exponential integrators | $e^{tL}$ for stiff linear part | When spectral spatial operators added |

---

## 7. Key Design Decisions

### Decision 1: StepOperator includes both spatial and temporal

The `StepOperator.step()` method encapsulates the **full advancement**:
spatial operator application, BC enforcement, and temporal scheme. This
is simpler than separating spatial and temporal operators (which would
require the `TimeIntegrator` to know about matrix assembly).

### Decision 2: Newton stays inside the StepOperator

For implicit schemes with nonlinear spatial operators (HJB), the Newton
iteration is part of `step()`. The `TimeIntegrator` does not need to
know about Newton convergence — it just calls `step()` and gets $u^{n+1}$.

### Decision 3: Direction is a TimeIntegrator concern, not StepOperator

A `StepOperator` advances **forward** by one $\Delta t$. The
`TimeIntegrator` handles direction by iterating backward through time
indices and negating $\Delta t$ semantics for backward equations.

---

**Last Updated**: 2026-02-05
