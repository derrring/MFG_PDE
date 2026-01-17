# Space-Time Operator Architecture Proposal

**Date**: 2026-01-17
**Status**: PROPOSAL - Architectural Design
**Scope**: Refactoring `mfg_pde.alg.numerical` time integration

## Executive Summary

**Proposal**: Refactor time integration from imperative "time-stepping loops" to declarative "space-time operator" paradigm, treating time $t$ as a parallel dimension to space $x$.

**Rationale**: MFG problems are boundary value problems (BVP) in time, not initial value problems (IVP). The forward-backward coupling (HJB terminal condition + FP initial condition) naturally suggests global space-time formulations.

**Benefits**:
1. ✅ Enables parallel-in-time methods (Parareal, Multigrid-in-Time)
2. ✅ Unified interface for sequential vs global solvers
3. ✅ Hardware acceleration friendly (JAX, GPU-friendly tensors)
4. ✅ Mathematical clarity (operator composition vs loop logic)

**Migration Path**: Introduce new abstraction layer while preserving backward compatibility.

---

## 1. Mathematical Foundation

### 1.1 The Cylinder Manifold View

Classical MFG system:

**HJB (backward in time)**:
```
-∂u/∂t + H(∇u, x, m) - (σ²/2)Δu = 0    on [0,T] × Ω
u(T, x) = g(x)                          (terminal condition)
```

**FP (forward in time)**:
```
∂m/∂t + div(m·α(∇u)) - (σ²/2)Δm = 0    on [0,T] × Ω
m(0, x) = m₀(x)                          (initial condition)
```

### 1.2 Space-Time Formulation

View the solution as fields on the **cylinder** $\mathcal{Q} = [0,T] \times \Omega$:

```
u : [0,T] × Ω → ℝ    (Value function)
m : [0,T] × Ω → ℝ    (Density)
```

The coupled system becomes a **boundary value problem** on $\mathcal{Q}$:

```
L_HJB(u, m) = 0    with u|_{t=T} = g
L_FP(u, m) = 0     with m|_{t=0} = m₀
```

where $L_{\text{HJB}}$ and $L_{\text{FP}}$ are differential operators on the cylinder.

**Key Insight**: Time-stepping (sequential) is just **Gaussian elimination** on the block-triangular structure of the discretized space-time operator.

---

## 2. Current Architecture Analysis

### 2.1 Existing Design Pattern

**Current Interface** (`mfg_pde/alg/numerical/`):

```python
# HJB Solver
class BaseHJBSolver:
    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,  # Shape: (Nt+1, Nx, ...)
        U_final_condition_at_T: np.ndarray,       # Shape: (Nx, ...)
        U_from_prev_picard: np.ndarray,           # Shape: (Nt+1, Nx, ...)
        diffusion_field: float | np.ndarray | None = None,
    ) -> np.ndarray:  # Returns: (Nt+1, Nx, ...)
        """Solve HJB backward from T to 0"""

        # IMPERATIVE LOOP (hidden in implementation)
        U_solution = np.zeros((Nt+1, *shape))
        U_solution[-1] = U_final_condition_at_T

        for n in range(Nt-1, -1, -1):  # Backward iteration
            U_solution[n] = step_backward(U_solution[n+1], M_density[n], ...)

        return U_solution
```

```python
# FP Solver
def solve_fp_nd_full_system(
    m_initial_condition: np.ndarray,           # Shape: (Nx, ...)
    U_solution_for_drift: np.ndarray | None,  # Shape: (Nt+1, Nx, ...)
    problem: MFGProblem,
    ...
) -> np.ndarray:  # Returns: (Nt, Nx, ...)
    """Solve FP forward from 0 to T"""

    # IMPERATIVE LOOP (explicit in code)
    M_solution = np.zeros((Nt, *shape))
    M_solution[0] = m_initial_condition

    for k in range(Nt - 1):  # Forward iteration
        M_solution[k+1] = step_forward(M_solution[k], U_solution[k], ...)

    return M_solution
```

### 2.2 Observations

**Strengths**:
- ✅ Simple, intuitive interface
- ✅ Memory efficient (can stream snapshots)
- ✅ Easy to debug (step-by-step execution)

**Limitations**:
- ❌ Time iteration hardcoded in implementation (cannot parallelize)
- ❌ Forward/backward distinction handled manually
- ❌ Difficult to implement parallel-in-time methods
- ❌ Tight coupling between "loop logic" and "step operator"

**Current Data Flow**:
```
Initial/Terminal Condition
    ↓
[Time Loop] ← Imperative control flow
    ↓
Sequence of Snapshots (Nt+1, Nx, ...)
```

---

## 3. Proposed Space-Time Architecture

### 3.1 Core Abstraction Layers

**Layer 1: Space-Time Field (Data)**

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from mfg_pde.geometry.base import Geometry

@dataclass
class SpacetimeField:
    """
    Unified representation of fields over [0,T] × Ω.

    This is the fundamental data structure for solutions in the space-time view.
    """
    domain: Geometry  # Contains BOTH spatial grid AND time grid
    data: np.ndarray  # Shape: (Nt, Nx, Ny, ...) - full space-time tensor

    def get_spatial_slice(self, t_index: int) -> np.ndarray:
        """Extract spatial field at time index t_index"""
        return self.data[t_index]

    def get_temporal_slice(self, *spatial_indices) -> np.ndarray:
        """Extract time series at spatial location"""
        return self.data[:, *spatial_indices]

    def flatten(self) -> np.ndarray:
        """Flatten to vector for global matrix solvers"""
        return self.data.ravel()

    @classmethod
    def from_snapshots(cls, snapshots: list[np.ndarray], domain: Geometry) -> "SpacetimeField":
        """Construct from sequence of spatial snapshots"""
        return cls(domain=domain, data=np.array(snapshots))
```

**Layer 2: Trajectory Solver Protocol (The Interface)**

```python
class TrajectorySolver(Protocol):
    """
    Abstract interface for solvers that generate full time trajectories.

    This protocol unifies both sequential (time-stepping) and global (space-time)
    solution methods under a common API.
    """

    def solve_trajectory(
        self,
        boundary_condition: np.ndarray,  # Initial (FP) or Terminal (HJB)
        coupling_field: SpacetimeField | None = None,  # m(t,x) for HJB, u(t,x) for FP
        **kwargs
    ) -> SpacetimeField:
        """
        Solve PDE over entire time horizon [0, T].

        Args:
            boundary_condition: Boundary in time (IC for FP, TC for HJB)
            coupling_field: Coupling term from other equation

        Returns:
            SpacetimeField containing full solution trajectory
        """
        ...
```

**Layer 3: Implementation Strategies**

Two complementary implementations of `TrajectorySolver`:

#### Strategy A: Sequential Marching Solver (Current Logic)

```python
class SequentialMarchingSolver(TrajectorySolver):
    """
    Classical time-stepping: solves slice-by-slice.

    Best for:
    - Nonlinear Hamiltonians (HJB with general H)
    - Memory-constrained environments
    - Explicit time-stepping schemes
    - Debugging (step-by-step inspection)

    Implementation: Wraps current loop-based logic.
    """

    def __init__(
        self,
        step_kernel: StepOperator,  # Single timestep operator
        direction: Literal["forward", "backward"] = "forward"
    ):
        self.step_kernel = step_kernel
        self.direction = direction

    def solve_trajectory(
        self,
        boundary_condition: np.ndarray,
        coupling_field: SpacetimeField | None = None,
        **kwargs
    ) -> SpacetimeField:
        """
        Solve via sequential time-stepping (the loop is HERE).
        """
        Nt = self.step_kernel.domain.time_grid.size
        shape = boundary_condition.shape

        # Allocate storage
        trajectory_data = np.zeros((Nt, *shape))

        # Set boundary condition in time
        if self.direction == "forward":
            trajectory_data[0] = boundary_condition  # Initial condition
            time_indices = range(Nt - 1)
        else:  # backward
            trajectory_data[-1] = boundary_condition  # Terminal condition
            time_indices = range(Nt - 2, -1, -1)

        # THE LOOP (encapsulated here, not in application code)
        current_state = boundary_condition
        for t_idx in time_indices:
            # Extract coupling at current time if provided
            coupling_t = coupling_field.get_spatial_slice(t_idx) if coupling_field else None

            # Advance one timestep
            current_state = self.step_kernel.step(
                current_state,
                coupling=coupling_t,
                **kwargs
            )

            # Store result
            if self.direction == "forward":
                trajectory_data[t_idx + 1] = current_state
            else:
                trajectory_data[t_idx] = current_state

        return SpacetimeField(domain=self.step_kernel.domain, data=trajectory_data)
```

#### Strategy B: Global Space-Time Solver (Future)

```python
class GlobalSpacetimeSolver(TrajectorySolver):
    """
    All-at-once solver: Assembles full space-time linear system.

    Best for:
    - Linear/Quadratic MFG (LQ-MFG)
    - Parallel-in-time methods (Parareal, Multigrid-in-Time)
    - Global error control (minimize ||u - u_exact||_{L²(Q)})
    - Hardware acceleration (GPU/TPU with large tensors)

    Mathematical Structure:
        L @ u_flat = b
        where L is (Nt*Nx) × (Nt*Nx) sparse matrix
              L = Dt ⊗ I_x + I_t ⊗ L_x

    For HJB: L is upper-triangular (backward time derivative)
    For FP:  L is lower-triangular (forward time derivative)

    Sequential solving (current) = Forward/Backward substitution on L
    Global solving (this class) = Direct/Iterative sparse solve on L
    """

    def __init__(
        self,
        assembler: SpacetimeMatrixAssembler,  # Builds L and b
        solver_type: Literal["direct", "iterative", "parareal"] = "direct"
    ):
        self.assembler = assembler
        self.solver_type = solver_type

    def solve_trajectory(
        self,
        boundary_condition: np.ndarray,
        coupling_field: SpacetimeField | None = None,
        **kwargs
    ) -> SpacetimeField:
        """
        Solve via global space-time matrix assembly.

        This treats the entire PDE as a single linear system:
            (Dt ⊗ I + I ⊗ Laplacian) @ u_flat = b
        """
        # 1. Assemble global space-time matrix A
        #    This is a tensor product of:
        #    - Time derivative operator Dt (tridiagonal in time)
        #    - Spatial operators (Laplacian, Advection)
        #
        #    For N time points, Nx spatial points:
        #    A is (N*Nx) × (N*Nx) sparse matrix
        A = self.assembler.build_spacetime_matrix(coupling_field)

        # 2. Build RHS vector b (includes boundary conditions in time)
        b = self.assembler.build_rhs(boundary_condition)

        # 3. Solve global system
        if self.solver_type == "direct":
            # Sparse LU factorization (good for moderate sizes)
            u_flat = sparse.linalg.spsolve(A, b)
        elif self.solver_type == "iterative":
            # GMRES/BiCGSTAB (good for large systems)
            u_flat, info = sparse.linalg.gmres(A, b)
        elif self.solver_type == "parareal":
            # Parareal: Coarse sequential + Fine parallel corrections
            u_flat = self._solve_parareal(A, b, **kwargs)

        # 4. Reshape back to (Nt, Nx, Ny, ...)
        Nt, *spatial_shape = self.assembler.domain.get_full_shape()
        trajectory_data = u_flat.reshape((Nt, *spatial_shape))

        return SpacetimeField(domain=self.assembler.domain, data=trajectory_data)

    def _solve_parareal(self, A, b, **kwargs):
        """
        Parareal algorithm: Parallel-in-time iterative solver.

        Algorithm:
        1. Solve with coarse sequential solver (cheap, inaccurate)
        2. Parallelize: Solve fine problems on [t_n, t_{n+1}] simultaneously
        3. Iteratively correct coarse solution using fine results

        This achieves parallelism in the time direction!
        """
        # (Implementation details omitted for brevity)
        pass
```

### 3.2 Step Operator Protocol

To decouple "single timestep logic" from "iteration strategy":

```python
class StepOperator(Protocol):
    """
    Operator for advancing/retreating a single time step.

    This is the primitive that SequentialMarchingSolver uses.
    """

    domain: Geometry  # Contains time grid (for dt) and spatial grid

    def step(
        self,
        current_state: np.ndarray,  # u^n or m^n
        coupling: np.ndarray | None = None,  # Coupling term at time n
        **kwargs
    ) -> np.ndarray:  # u^{n+1} or m^{n+1}
        """
        Advance one timestep.

        For HJB: backward step (n+1 → n)
        For FP:  forward step (n → n+1)
        """
        ...
```

Example implementation:

```python
class ImplicitDiffusionStep(StepOperator):
    """
    Single implicit diffusion timestep.

    Solves: (I/dt - D*Δ) u^{n+1} = u^n / dt
    """

    def __init__(self, domain: Geometry, diffusion_coeff: float, bc: BoundaryConditions):
        self.domain = domain
        self.D = diffusion_coeff
        self.bc = bc

        # Pre-assemble Laplacian matrix (reusable across steps)
        from mfg_pde.geometry.operators.laplacian import LaplacianOperator
        L_op = LaplacianOperator(
            spacings=domain.get_grid_spacing(),
            field_shape=domain.get_grid_shape(),
            bc=bc
        )
        self.L_matrix = L_op.as_scipy_sparse()

    def step(self, current_state: np.ndarray, coupling=None, **kwargs) -> np.ndarray:
        """Single backward implicit diffusion step"""
        dt = self.domain.time_grid.dt
        N_total = int(np.prod(current_state.shape))

        # Build system matrix (reuses pre-assembled Laplacian)
        I = sparse.eye(N_total)
        A = I / dt - self.D * self.L_matrix

        # RHS
        b = current_state.ravel() / dt

        # Solve
        next_state = sparse.linalg.spsolve(A, b).reshape(current_state.shape)
        return next_state
```

---

## 4. Migration Path

### 4.1 Phase 1: Introduce Abstractions (Backward Compatible)

**Objective**: Add new layer without breaking existing code

**Steps**:
1. Create `mfg_pde/alg/propagation/` module
2. Define protocols: `TrajectorySolver`, `StepOperator`, `SpacetimeField`
3. Implement `SequentialMarchingSolver` as wrapper around current time-stepping logic

**Example Wrapper**:

```python
# mfg_pde/alg/propagation/sequential.py

class SequentialHJBSolver(TrajectorySolver):
    """
    Wrapper around existing HJB time-stepping logic.
    """

    def __init__(self, hjb_solver: BaseHJBSolver):
        self.hjb_solver = hjb_solver

    def solve_trajectory(
        self,
        boundary_condition: np.ndarray,  # Terminal condition u(T, x)
        coupling_field: SpacetimeField | None = None,  # m(t,x)
        **kwargs
    ) -> SpacetimeField:
        """Solve HJB via existing solve_hjb_system()"""

        # Convert SpacetimeField to old API format
        if coupling_field is not None:
            M_density = coupling_field.data
        else:
            M_density = np.ones((self.hjb_solver.problem.Nt + 1, *boundary_condition.shape))

        # Call existing method
        U_solution = self.hjb_solver.solve_hjb_system(
            M_density_evolution_from_FP=M_density,
            U_final_condition_at_T=boundary_condition,
            U_from_prev_picard=kwargs.get("U_from_prev_picard", None),
            **kwargs
        )

        # Convert to SpacetimeField
        return SpacetimeField(
            domain=self.hjb_solver.problem.geometry,
            data=U_solution
        )
```

**Benefits**:
- ✅ Zero breaking changes
- ✅ New API co-exists with old API
- ✅ Gradual adoption path

### 4.2 Phase 2: Refactor Solvers to Use Step Operators

**Objective**: Decouple loop logic from step logic

**Example**: Refactor HJB FDM solver

```python
# Before (current)
class HJBFDMSolver:
    def solve_hjb_system(self, M_density, U_final, ...):
        # Loop is baked into implementation
        for n in range(Nt-1, -1, -1):
            U_solution[n] = self._step_backward(U_solution[n+1], M_density[n], ...)
        return U_solution

# After (refactored)
class HJBFDMStepOperator(StepOperator):
    """Single backward HJB timestep"""
    def step(self, current_state, coupling, **kwargs):
        # Original _step_backward logic here
        return next_state

class HJBFDMSolver:
    def __init__(self, ...):
        self.step_operator = HJBFDMStepOperator(...)
        self.trajectory_solver = SequentialMarchingSolver(
            step_kernel=self.step_operator,
            direction="backward"
        )

    def solve_hjb_system(self, M_density, U_final, ...):
        # Delegate to trajectory solver
        coupling_field = SpacetimeField(domain=self.geometry, data=M_density)
        result = self.trajectory_solver.solve_trajectory(
            boundary_condition=U_final,
            coupling_field=coupling_field
        )
        return result.data  # For backward compatibility
```

**Benefits**:
- ✅ Step operator reusable in other contexts
- ✅ Easier to test (test step operator independently)
- ✅ Opens path for global solvers

### 4.3 Phase 3: Implement Global Space-Time Solvers

**Objective**: Add parallel-in-time capabilities

**New Features**:
1. `GlobalSpacetimeSolver` for LQ-MFG problems
2. `PararealSolver` for parallel-in-time
3. JAX/GPU acceleration support

**Example Use Case**: LQ-MFG with global solve

```python
# Linear-Quadratic MFG: Both HJB and FP are linear!
problem = create_lq_mfg_problem(...)

# Assemble full space-time system for both equations simultaneously
assembler = MonolithicMFGAssembler(problem)
solver = GlobalSpacetimeSolver(assembler, solver_type="direct")

# Solve HJB and FP SIMULTANEOUSLY (not iteratively!)
u_field, m_field = solver.solve_coupled_trajectory(
    u_terminal=problem.terminal_cost,
    m_initial=problem.initial_density
)
```

This is **impossible** with current time-stepping architecture!

---

## 5. Design Recommendations

### 5.1 Proposed Directory Structure

```
mfg_pde/alg/
├── propagation/              # NEW MODULE (Space-Time Operators)
│   ├── __init__.py
│   ├── base.py              # Protocols: TrajectorySolver, StepOperator
│   ├── fields.py            # SpacetimeField class
│   ├── sequential.py        # SequentialMarchingSolver
│   ├── global_solver.py     # GlobalSpacetimeSolver (future)
│   ├── parareal.py          # PararealSolver (future)
│   └── assemblers.py        # SpacetimeMatrixAssembler (future)
│
├── numerical/               # EXISTING (gradual refactoring)
│   ├── hjb_solvers/
│   │   ├── base_hjb.py      # Add: implements StepOperator protocol
│   │   ├── hjb_fdm.py       # Refactor: extract step logic
│   │   └── ...
│   ├── fp_solvers/
│   │   ├── fp_fdm_time_stepping.py  # Refactor: extract step logic
│   │   └── ...
│   └── coupling/
│       ├── fixed_point_iterator.py  # Update: use TrajectorySolver protocol
│       └── ...
```

### 5.2 Implementation Priorities

**High Priority** (v0.17-v0.18):
1. ✅ Define protocols in `propagation/base.py`
2. ✅ Implement `SpacetimeField` in `propagation/fields.py`
3. ✅ Implement `SequentialMarchingSolver` in `propagation/sequential.py`
4. ✅ Create wrappers for existing HJB/FP solvers

**Medium Priority** (v0.19-v1.0):
5. ⏸️ Refactor HJB/FP solvers to use `StepOperator` protocol
6. ⏸️ Update coupling solvers to use `TrajectorySolver` interface
7. ⏸️ Deprecate direct loop-based APIs

**Low Priority** (v1.1+):
8. ⏸️ Implement `GlobalSpacetimeSolver` for LQ-MFG
9. ⏸️ Implement `PararealSolver` for parallel-in-time
10. ⏸️ JAX/GPU backend integration with space-time tensors

### 5.3 Backward Compatibility Strategy

**Principle**: **Adapters, not rewrites**

```python
# Old API (keep working)
hjb_solver = HJBFDMSolver(problem)
U_solution = hjb_solver.solve_hjb_system(M_density, U_final, ...)

# New API (coexists)
from mfg_pde.alg.propagation import SequentialHJBSolver

trajectory_solver = SequentialHJBSolver(hjb_solver)
result_field = trajectory_solver.solve_trajectory(
    boundary_condition=U_final,
    coupling_field=SpacetimeField(domain=geometry, data=M_density)
)
U_solution = result_field.data  # Same result!
```

**Deprecation Timeline**:
- v0.17: Introduce new APIs
- v0.18-v0.19: Soft deprecation warnings on old loop-based methods
- v1.0: Hard deprecation (but keep adapters)
- v2.0: Remove old APIs entirely

---

## 6. MFG-Specific Considerations

### 6.1 Forward-Backward Coupling

**Current Approach** (Picard iteration):

```python
for k in range(max_iterations):
    # HJB: backward in time, uses m^k
    U_next = hjb_solver.solve_hjb_system(M_current, ...)

    # FP: forward in time, uses u^{k+1}
    M_next = fp_solver.solve_fp_system(U_next, ...)

    # Check convergence
    if converged(U_next, U_current, M_next, M_current):
        break
```

**Space-Time View**:

The Picard iteration is solving:
```
F(u, m) = 0
```
where $F: \mathcal{H}([0,T] \times \Omega)^2 \to \mathcal{H}([0,T] \times \Omega)^2$ is the **coupled space-time operator**:

```
F(u, m) = [ L_HJB(u, m) ]
          [ L_FP(u, m)  ]
```

**Proposed Coupling Solver**:

```python
class SpacetimePicardIterator:
    """
    Picard iteration using TrajectorySolver protocol.
    """

    def __init__(
        self,
        hjb_trajectory_solver: TrajectorySolver,
        fp_trajectory_solver: TrajectorySolver
    ):
        self.hjb_solver = hjb_trajectory_solver
        self.fp_solver = fp_trajectory_solver

    def solve_mfg(
        self,
        u_terminal: np.ndarray,
        m_initial: np.ndarray,
        max_iterations: int = 100
    ) -> tuple[SpacetimeField, SpacetimeField]:
        """Solve coupled MFG system"""

        # Initialize
        m_field = self._initialize_density(m_initial)

        for k in range(max_iterations):
            # HJB: u^{k+1} = solve HJB given m^k
            u_field = self.hjb_solver.solve_trajectory(
                boundary_condition=u_terminal,
                coupling_field=m_field
            )

            # FP: m^{k+1} = solve FP given u^{k+1}
            m_field_next = self.fp_solver.solve_trajectory(
                boundary_condition=m_initial,
                coupling_field=u_field
            )

            # Convergence check
            if self._converged(u_field, m_field_next, m_field):
                return u_field, m_field_next

            m_field = m_field_next

        return u_field, m_field
```

**Key Advantage**: The coupling logic is **independent** of whether solvers use sequential stepping or global assembly!

### 6.2 Monolithic vs Sequential Coupling

**Sequential Coupling** (current): Solve HJB → FP → iterate

**Monolithic Coupling** (future with global solvers): Solve both simultaneously

For LQ-MFG, the coupled system is **linear**:
```
[ L_HJB    C_HJB ] [ u ]   [ f_HJB ]
[ C_FP     L_FP  ] [ m ] = [ f_FP  ]
```

With `GlobalSpacetimeSolver`, this becomes:

```python
# Assemble coupled space-time system
assembler = MonolithicLQMFGAssembler(problem)
A_coupled = assembler.build_coupled_matrix()  # Full (2*Nt*Nx) × (2*Nt*Nx)
b_coupled = assembler.build_coupled_rhs(u_terminal, m_initial)

# Solve ONCE (no iterations needed!)
solution = sparse.linalg.spsolve(A_coupled, b_coupled)

# Extract u and m
u_flat, m_flat = np.split(solution, 2)
u_field = SpacetimeField(domain, u_flat.reshape((Nt, *shape)))
m_field = SpacetimeField(domain, m_flat.reshape((Nt, *shape)))
```

This is **vastly more efficient** for LQ-MFG than Picard iteration!

---

## 7. Performance Considerations

### 7.1 Memory Footprint

**Sequential (current)**:
- Storage: 2 × (Nt × Nx) for U and M
- Can stream to disk if Nt very large

**Global**:
- Storage: Sparse matrix (Nt×Nx) × (Nt×Nx) ≈ 5 × Nt × Nx nonzeros
- RHS/solution vectors: Nt × Nx

**Trade-off**: Global uses more memory but enables parallelism

**Recommendation**:
- Sequential for Nt > 10,000 or Nx > 100,000
- Global for moderate sizes with parallel hardware

### 7.2 Computational Complexity

**Sequential (time-stepping)**:
- Time: $O(N_t \times C_{\text{step}})$ where $C_{\text{step}}$ = cost of one timestep
- For implicit step with sparse solve: $C_{\text{step}} = O(N_x^{1.5})$ (sparse LU)
- Total: $O(N_t \times N_x^{1.5})$
- **Serial in time** (cannot parallelize time)

**Global (space-time matrix)**:
- Time: $O((N_t \times N_x)^{1.5})$ for sparse direct solve
- Or: $O(k \times N_t \times N_x)$ for $k$ iterations of GMRES
- **Parallel in time** (can use parallel preconditioners)

**When Global Wins**:
- When parallel hardware available (multi-GPU)
- When Nt/Nx ratio is moderate
- When linear problems (no outer iterations)

### 7.3 JAX/GPU Acceleration

**Sequential with JAX**:

```python
import jax.numpy as jnp
from jax import lax

def sequential_solve_jax(u_terminal, m_evolution, step_fn):
    """JAX-accelerated sequential solver using lax.scan"""

    def backward_step(u_next, m_t):
        """Single backward step (runs on GPU)"""
        u_current = step_fn(u_next, m_t)
        return u_current, u_current  # (carry, output)

    # lax.scan compiles the loop (much faster than Python for-loop)
    u_init = u_terminal
    m_reversed = jnp.flip(m_evolution, axis=0)

    _, u_trajectory_reversed = lax.scan(backward_step, u_init, m_reversed)
    u_trajectory = jnp.flip(u_trajectory_reversed, axis=0)

    return u_trajectory
```

**Global with JAX**: Entire space-time tensor on GPU

```python
# Full (Nt, Nx) tensor on GPU
u_flat = jnp.linalg.solve(A_jax, b_jax)  # Single GPU solve
u_field = u_flat.reshape((Nt, Nx))
```

**Benefit**: JAX can JIT-compile and fuse operations, massive speedup

---

## 8. Comparison with Reference

The user referenced a Gemini discussion: https://gemini.google.com/share/6d83a4f9a475

**Alignment with Gemini Discussion**:
- ✅ Time as parallel dimension (tensor product spaces)
- ✅ Declarative operator composition
- ✅ Support for both sequential and global methods
- ✅ Parareal algorithm as a TrajectorySolver implementation

**Extensions Beyond Gemini**:
- Concrete protocols and data structures (SpacetimeField, TrajectorySolver)
- Migration path with backward compatibility
- MFG-specific coupling considerations
- Performance analysis and hardware acceleration

---

## 9. Decision Criteria

### When to Use Sequential Marching:
- ✅ Nonlinear Hamiltonians (general H)
- ✅ Explicit time-stepping schemes
- ✅ Very long time horizons (Nt > 10,000)
- ✅ Memory-constrained environments
- ✅ Debugging/validation

### When to Use Global Space-Time:
- ✅ Linear/Quadratic Hamiltonians (LQ-MFG)
- ✅ Implicit schemes with small Nt
- ✅ Parallel hardware available (multi-GPU/TPU)
- ✅ Global error minimization needed
- ✅ Parallel-in-time desired (Parareal)

### Recommended Default:
**Start with Sequential (existing logic), migrate to Global when beneficial**

The proposed architecture supports **both** under unified interface!

---

## 10. Action Items

### Immediate (This Week):
1. Create `docs/development/spacetime_operator_architecture_proposal.md` (this document)
2. Gather feedback from maintainers
3. Decide on adoption timeline

### Short-Term (v0.17):
1. Create `mfg_pde/alg/propagation/` module
2. Define protocols (TrajectorySolver, StepOperator, SpacetimeField)
3. Implement SequentialMarchingSolver
4. Add wrappers for existing solvers (backward compatible)
5. Write examples comparing old vs new API

### Medium-Term (v0.18-v1.0):
1. Refactor HJB/FP solvers to extract StepOperator
2. Update coupling solvers to use TrajectorySolver
3. Deprecate loop-based APIs
4. Documentation and migration guide

### Long-Term (v1.1+):
1. Implement GlobalSpacetimeSolver for LQ-MFG
2. Implement PararealSolver
3. JAX/GPU backend integration
4. Benchmarking study (sequential vs global)

---

## 11. Risks and Mitigation

### Risk 1: Complexity Overhead
**Concern**: New abstractions add complexity for simple cases

**Mitigation**:
- Provide high-level wrappers for common use cases
- Maintain backward compatibility
- Document migration path clearly

### Risk 2: Performance Regression
**Concern**: Abstraction layers may slow down existing code

**Mitigation**:
- Implement wrappers as zero-cost (delegate directly)
- Benchmark before/after refactoring
- Use profiling to identify bottlenecks

### Risk 3: Adoption Friction
**Concern**: Users may resist learning new API

**Mitigation**:
- Gradual deprecation (v0.17 → v2.0)
- Comprehensive examples
- Clear benefits documentation (parallel-in-time, GPU)

---

## 12. Conclusion

The Space-Time Operator Architecture offers a mathematically principled and computationally flexible framework for MFG solvers. By treating time as a dimension and defining declarative interfaces (TrajectorySolver, StepOperator), we enable:

1. **Backward Compatibility**: Existing time-stepping logic wraps cleanly
2. **Future Extensibility**: Parallel-in-time, GPU acceleration, monolithic coupling
3. **Mathematical Clarity**: Operators on cylinder manifold vs imperative loops
4. **Practical Benefits**: Same API for sequential and global solvers

**Recommendation**: **Proceed with phased adoption** starting with protocol definitions and wrappers (v0.17), followed by gradual solver refactoring (v0.18-v1.0), culminating in advanced features (v1.1+).

This positions MFG_PDE as a cutting-edge research platform while preserving stability for existing users.

---

**Contributors**: Claude Opus 4.5
**Date**: 2026-01-17
**Status**: Proposal - Awaiting Feedback
**Related**: Issue #597 (Operator Refactoring), Gemini Discussion 6d83a4f9a475
