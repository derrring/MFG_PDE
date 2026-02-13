# Conditions vs Constraints: Architectural Separation

**Date**: 2026-01-19
**Context**: ConditionsMixin handles classical PDE conditions; ConstraintProtocol handles variational inequalities

---

## Conceptual Distinction

### Conditions (Classical PDE Theory)

**Definition**: Values or derivatives specified at boundaries or time endpoints

**Examples**:
- **Initial condition (IC)**: $m(0, x) = m_0(x)$ - density at t=0
- **Terminal condition (TC)**: $u(T, x) = u_T(x)$ - value at final time
- **Boundary condition (BC)**:
  - Dirichlet: $u|_{\partial\Omega} = g$
  - Neumann: $\frac{\partial u}{\partial n}\bigg|_{\partial\Omega} = h$
  - Robin: $\alpha u + \beta \frac{\partial u}{\partial n}\bigg|_{\partial\Omega} = f$

**Mathematical nature**: Specified data on lower-dimensional manifolds (boundaries, initial/final time)

**Implementation**: Direct substitution into discretization
- Dirichlet: Set DOF values directly
- Neumann: Modify stencil coefficients
- Robin: Linear combination

**Handled by**: `ConditionsMixin` in `mfg_components.py`

---

### Constraints (Variational Inequality Theory)

**Definition**: Inequality restrictions on solution values throughout domain

**Examples**:
- **Obstacle constraint**: $u(t,x) \geq \psi(x)$ everywhere (lower bound)
- **Capacity constraint**: $m(t,x) \leq m_{max}$ everywhere (upper bound)
- **Bilateral constraint**: $\psi_{lower}(x) \leq u(t,x) \leq \psi_{upper}(x)$
- **Congestion pricing**: $c(m) = \begin{cases} 0 & m < m_0 \\ \infty & m \geq m_0 \end{cases}$

**Mathematical nature**: Active set inequalities, complementarity conditions

**Implementation**: Projection onto convex sets
- Obstacle: $\text{proj}_K(u) = \max(u, \psi)$ or $\min(u, \psi)$
- QP solver: Enforce constraints via KKT conditions
- Active set methods: Iteratively solve on inactive set

**Handled by**: `ConstraintProtocol` in `geometry/boundary/constraint_protocol.py`

---

## Architectural Separation

### Why Different Systems?

**Conditions** (ConditionsMixin):
- Apply to **boundaries/endpoints** only
- Deterministic (known functions)
- Built into PDE discretization
- Part of problem definition

**Constraints** (ConstraintProtocol):
- Apply **everywhere in domain**
- May change during solve (active set evolves)
- Applied via projection/optimization
- Part of solver algorithm

**Key insight**: Conditions are **problem data**, constraints are **algorithmic enforcement**.

---

## Current Implementation

### ConditionsMixin (mfg_components.py)

```python
class ConditionsMixin:
    """Handles classical PDE conditions."""

    def _setup_custom_initial_density(self) -> None:
        """IC: m(0, x) = m_0(x)"""
        for i in range(num_points):
            self.m_init[i] = initial_func(x_i)

    def _setup_custom_final_value(self) -> None:
        """TC: u(T, x) = u_T(x)"""
        for i in range(num_points):
            self.u_fin[i] = final_func(x_i)

    def get_boundary_conditions(self) -> BoundaryConditions:
        """BC: Dirichlet/Neumann/Robin at ∂Ω"""
        return self.geometry.get_boundary_conditions()
```

**Scope**: Setup time (problem construction)

**Usage**: Called once during `MFGProblem.__init__()`

---

### ConstraintProtocol (constraint_protocol.py)

```python
@runtime_checkable
class ConstraintProtocol(Protocol):
    """Handles variational inequality constraints."""

    def project(self, u: NDArray) -> NDArray:
        """Project onto constraint set K: P_K(u)"""
        ...

    def is_feasible(self, u: NDArray, tol: float) -> bool:
        """Check if u ∈ K"""
        ...

    def get_active_set(self, u: NDArray, tol: float) -> NDArray:
        """Identify where constraints bind"""
        ...
```

**Scope**: Solve time (iteration loop)

**Usage**: Called every iteration in VI solvers

---

## Example: HJB with Obstacle

**Problem**: Solve HJB equation with obstacle $u \geq \psi$

```python
# Setup (problem definition)
problem = MFGProblem(
    geometry=geometry,
    T=1.0,
    Nt=50,
    components=MFGComponents(
        final_value_func=lambda x: 0.0,  # TC: u(T,x) = 0
        # BC: Neumann ∂u/∂n = 0 at ∂Ω (from geometry default)
    )
)

# Obstacle constraint (solver-level)
obstacle = ObstacleConstraint(psi, constraint_type='lower')

# Solve with projection
solver = HJBGFDMSolver(problem, constraint=obstacle)
for iteration in range(max_iterations):
    u_unconstrained = solve_hjb_step()
    u = obstacle.project(u_unconstrained)  # ← Constraint enforcement
```

**Key distinction**:
- **Terminal condition** (u(T,x)=0): Handled by ConditionsMixin → sets `problem.u_fin`
- **Obstacle constraint** (u≥ψ): Handled by ConstraintProtocol → solver projects each iteration

---

## Interaction Between Systems

### No Direct Coupling

Conditions and constraints are **independent**:
- ConditionsMixin doesn't know about constraints
- ConstraintProtocol doesn't know about conditions

**Example**: HJB with obstacle AND Dirichlet BC
```python
# Condition: u = 0 at boundary (exact)
bc = dirichlet_bc(value=0.0)

# Constraint: u ≥ -1 in interior (projected)
constraint = ObstacleConstraint(lower=-1.0)

# Application order:
1. Solve unconstrained HJB step
2. Project onto constraint: u = max(u, -1)
3. Apply BC: u[boundary] = 0  # ← Overwrites constraint if needed
```

**Priority**: Boundary conditions **override** constraints at boundary points.

---

## Time-Parallel Construction (Future)

### Current Sequential Model

```python
# ConditionsMixin: Sequential time-stepping
m_init = setup_initial_density()  # t=0 only

# Solver loop
for t in range(Nt):
    m[t+1] = solve_fp_step(m[t])  # Sequential dependency
```

### Proposed Time-Parallel Model

**Goal**: Construct all timesteps simultaneously (Parareal, MGRIT)

```python
# Proposed: Time-parallel condition setup
def setup_conditions_all_times(self) -> None:
    """Setup IC/TC for all time slices simultaneously."""
    # IC at t=0
    self.m_all_times[0, :] = initial_func(x)

    # TC at t=T
    self.u_all_times[-1, :] = final_func(x)

    # Interior times: Initialize with interpolation
    for t_idx in range(1, Nt):
        self.m_all_times[t_idx, :] = interpolate(...)

# Space-time coupled system
A @ u_all_times = b  # Single linear system, not time-stepping loop
```

**Impact on ConditionsMixin**:
- ✅ Current methods stay unchanged (backward compatible)
- ✅ Add new methods: `_setup_conditions_all_times_parallel()`
- ✅ Protocol pattern makes extension easy

**No breaking changes**: Sequential and parallel modes coexist

---

## Design Principles

### 1. Separation of Concerns

**ConditionsMixin**: Problem definition (what)
**ConstraintProtocol**: Algorithmic enforcement (how)

**Benefit**: Can change constraint enforcement algorithm without touching problem setup

### 2. Protocol-Based Composition

Both use Protocol pattern:
```python
class MFGContextProtocol(Protocol):      # ← Proposed (Issue proposal)
    """Required by ConditionsMixin"""
    ...

class ConstraintProtocol(Protocol):      # ← Existing (Issue #591)
    """Required by VI solvers"""
    ...
```

**Benefit**: Explicit contracts, easy to extend

### 3. Time Evolution Strategy Independence

**Current**: Sequential (explicit in time)
**Future**: Parallel (implicit space-time)

**Design**: ConditionsMixin supports both via separate methods

**Benefit**: Incremental adoption, no breaking changes

---

## Implementation Checklist

### Current Status (v0.17.x)

- [x] ConditionsMixin handles IC/TC/BC
- [x] ConstraintProtocol defines VI interface (Issue #591)
- [x] ObstacleConstraint implements protocol
- [x] HJBGFDMSolver accepts optional constraint
- [x] Systems are decoupled (no cross-dependencies)

### Future Enhancements

**For Protocol Documentation (Issue proposal)**:
- [ ] Define `MFGContextProtocol` for ConditionsMixin requirements
- [ ] Document in `MFGPROBLEM_ARCHITECTURE.md`

**For Time-Parallel Support (Separate issue)**:
- [ ] Add `_setup_conditions_all_times_parallel()` to ConditionsMixin
- [ ] Implement space-time coupled solver (new solver class)
- [ ] Add time-parallel backend (Parareal/MGRIT)

---

## FAQ

### Q: Why not unify conditions and constraints?

**A**: Mathematically distinct concepts:
- Conditions: Exact values on manifolds (boundaries, t=0, t=T)
- Constraints: Inequality restrictions in domain interior

Unified system would conflate problem data with algorithmic choices.

### Q: Can constraints apply to boundary conditions?

**A**: Yes, but they're applied differently:
```python
# Dirichlet BC: u|_∂Ω = g (exact, overrides everything)
# Constraint: u ≥ ψ (projected, may be violated at boundary by BC)

# Application order:
1. Apply constraint projection in interior
2. Overwrite boundary DOFs with BC values
```

Boundary conditions take **precedence** over constraints.

### Q: What about inequality boundary conditions?

**A**: These are **constraints**, not conditions:
```python
# This is a constraint (handled by ConstraintProtocol):
u|_∂Ω ≥ ψ

# This is a condition (handled by ConditionsMixin):
u|_∂Ω = g
```

Inequality BCs should use the constraint system.

### Q: How do time-parallel methods affect this?

**A**: Orthogonal concerns:
- **Sequential vs parallel**: Time evolution strategy
- **Conditions vs constraints**: Problem data vs algorithmic enforcement

Time-parallel extends ConditionsMixin methods but doesn't change the conditions/constraints separation.

---

## Related Documentation

- `mfg_pde/core/mfg_components.py` - ConditionsMixin implementation
- `mfg_pde/geometry/boundary/constraint_protocol.py` - ConstraintProtocol definition
- `mfg_pde/geometry/boundary/constraints.py` - ObstacleConstraint implementation
- Issue #591: Constraint Protocol & Obstacle Problems
- Issue #589: Boundary Condition Tier System
- Issue proposal: MFGProblem Mixin Documentation (this effort)

---

**Summary**: ConditionsMixin and ConstraintProtocol are **separate, complementary systems**. Conditions handle classical PDE boundary/initial data; constraints handle variational inequalities. Both use Protocol pattern for extensibility. Time-parallel construction is a future enhancement orthogonal to this separation.
