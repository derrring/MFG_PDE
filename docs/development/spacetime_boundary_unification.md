# Space-Time Boundary Unification

**Date**: 2026-01-17
**Related**: Space-Time Operator Architecture, BC Design
**Status**: Conceptual Design

## Executive Summary

**Observation**: Time-varying boundary conditions and space-time operators share a common mathematical foundation: **the boundary of the space-time cylinder**.

In the classical time-stepping view, spatial boundaries and temporal boundaries (initial/terminal conditions) are treated separately. In the space-time view, they are **unified as boundary data on the cylinder manifold** $\partial Q$.

This unification has profound implications for BC specification, consistency checking, and global solvers.

---

## 1. The Cylinder Boundary Decomposition

### 1.1 Mathematical Setup

**Space-Time Domain**: The cylinder $Q = [0, T] \times \Omega$

**Boundary Components**: $\partial Q$ decomposes into three parts:

1. **Lateral Surface** (spatial boundaries over time):
   $$\Gamma_{\text{lateral}} = [0, T] \times \partial\Omega$$

2. **Bottom Cap** (initial time):
   $$\Gamma_{\text{bottom}} = \{t = 0\} \times \Omega$$

3. **Top Cap** (terminal time):
   $$\Gamma_{\text{top}} = \{t = T\} \times \Omega$$

**Complete Boundary**:
$$\partial Q = \Gamma_{\text{lateral}} \cup \Gamma_{\text{bottom}} \cup \Gamma_{\text{top}}$$

`★ Insight ─────────────────────────────────────`
**Geometric Intuition**: Think of a soup can:
- Lateral surface = the label wrapped around the side
- Bottom cap = the bottom lid
- Top cap = the top lid

In PDE terms:
- Lateral = spatial BCs (Dirichlet/Neumann/Robin on $\partial\Omega$)
- Bottom = initial condition (for FP)
- Top = terminal condition (for HJB)
`─────────────────────────────────────────────────`

### 1.2 Classical vs Space-Time Views

**Classical Time-Stepping View**:

Spatial BCs and temporal BCs are **separate concerns**:

```python
# Spatial BC (applied at each timestep)
for t in time_steps:
    apply_boundary_conditions(u[t], bc_spatial)

# Temporal BC (applied once)
u[0] = initial_condition  # For FP
u[T] = terminal_condition  # For HJB
```

Problems:
- ❌ Spatial and temporal BCs treated differently
- ❌ Consistency between them not enforced
- ❌ Time-varying spatial BCs require special handling

**Space-Time View**:

All BCs are **boundary data on $\partial Q$**:

```python
# Single unified BC specification on ∂Q
boundary_data = {
    "lateral": bc_spatial(x, t),    # Function on [0,T] × ∂Ω
    "bottom": ic(x),                 # Function on {0} × Ω
    "top": tc(x),                    # Function on {T} × Ω
}

# Apply all BCs simultaneously in global solver
solve_on_cylinder(Q, boundary_data)
```

Benefits:
- ✅ Unified treatment of all BCs
- ✅ Consistency automatically enforced
- ✅ Time-varying spatial BCs are natural

---

## 2. Time-Varying Spatial Boundaries

### 2.1 Current MFG_PDE Design

MFG_PDE **already supports** time-dependent BCs via `Callable(point, time)`:

```python
from mfg_pde.geometry.boundary import dirichlet_bc

# Time-varying Dirichlet BC: u(t, x_boundary) = g(t, x)
def time_varying_value(point, time):
    """
    BC value as function of space and time.

    Args:
        point: Spatial location (x, y, ...) on boundary
        time: Current time t

    Returns:
        BC value at (t, x)
    """
    x = point[0] if len(point) > 0 else 0
    return np.sin(2 * np.pi * time) * x  # Example: sinusoidal in time, linear in space

bc = dirichlet_bc(value=time_varying_value, dimension=1)
```

**Interface** (`mfg_pde/geometry/boundary/types.py`):

```python
@dataclass
class BCSegment:
    """
    Boundary condition specification.

    Attributes:
        value: BC value (constant or function of (x, t))
               - float: constant value
               - Callable[[point, time], float]: spatially/temporally varying
    """
    name: str
    bc_type: BCType
    value: float | Callable[[np.ndarray, float], float]  # ← Key feature!
```

**Application in Time-Stepping** (`mfg_pde/geometry/boundary/applicator_fdm.py`):

```python
def apply_boundary_conditions_nd(
    field: np.ndarray,
    bc: BoundaryConditions,
    time: float = 0.0,  # ← Current time parameter
    ...
):
    """
    Apply BCs to field at given time.

    For time-dependent BCs (callable values), evaluates at specified time.
    """
    if callable(bc_value):
        # Evaluate time-varying BC at current time
        boundary_value = bc_value(boundary_point, time)
    else:
        # Constant BC
        boundary_value = bc_value
```

### 2.2 Space-Time Interpretation

**What's happening mathematically**:

When you specify `value=callable(x, t)`, you're defining **boundary data on the lateral surface** $\Gamma_{\text{lateral}} = [0, T] \times \partial\Omega$.

**Classical view**: "BC that changes at each timestep"
**Space-Time view**: "Function defined on lateral boundary of cylinder"

These are **the same thing**, just different perspectives!

**Example**: Heat equation with time-varying Dirichlet BC

```python
# PDE: ∂u/∂t = Δu on [0,T] × Ω
# BC:  u(t, x=0) = sin(πt) for t ∈ [0,T]  (left boundary varies in time)
#      u(t, x=1) = 0       for t ∈ [0,T]  (right boundary constant)

def left_bc(point, time):
    return np.sin(np.pi * time)

bc_left = BCSegment(name="left", bc_type=BCType.DIRICHLET,
                    value=left_bc, boundary="left")
bc_right = BCSegment(name="right", bc_type=BCType.DIRICHLET,
                     value=0.0, boundary="right")
```

**Space-time perspective**: This defines boundary data:
$$
g: \Gamma_{\text{lateral}} \to \mathbb{R}, \quad g(t, x) = \begin{cases}
\sin(\pi t) & \text{if } x = 0 \\
0 & \text{if } x = 1
\end{cases}
$$

---

## 3. Initial and Terminal Conditions as Boundary Data

### 3.1 The Temporal Caps

**For Fokker-Planck** (forward evolution):

Initial condition $m(0, x) = m_0(x)$ is **boundary data on the bottom cap**:
$$m|_{\Gamma_{\text{bottom}}} = m_0(x)$$

**For HJB** (backward evolution):

Terminal condition $u(T, x) = g(x)$ is **boundary data on the top cap**:
$$u|_{\Gamma_{\text{top}}} = g(x)$$

**Traditional view**: These are "different" from spatial BCs (applied once vs every timestep)

**Space-Time view**: They're all just **Dirichlet data on different parts of $\partial Q$**!

### 3.2 Unified BC Specification

In the space-time paradigm, we can specify **all boundary data** uniformly:

```python
@dataclass
class SpacetimeBoundaryData:
    """
    Complete boundary specification for space-time cylinder Q = [0,T] × Ω.

    Unifies spatial BCs, initial conditions, and terminal conditions.
    """
    # Lateral surface: [0,T] × ∂Ω
    spatial_bc: BoundaryConditions  # Can be time-varying via callable values

    # Bottom cap: {0} × Ω (initial condition)
    initial_condition: np.ndarray | Callable[[np.ndarray], float] | None = None

    # Top cap: {T} × Ω (terminal condition)
    terminal_condition: np.ndarray | Callable[[np.ndarray], float] | None = None

    def evaluate_on_boundary(self, t: float, x: np.ndarray, boundary_part: str) -> float:
        """
        Evaluate boundary data at (t, x) ∈ ∂Q.

        Args:
            t: Time coordinate
            x: Spatial coordinate
            boundary_part: "lateral", "bottom", or "top"

        Returns:
            Boundary value at (t, x)
        """
        if boundary_part == "lateral":
            # Spatial boundary at time t
            return self.spatial_bc.evaluate(x, time=t)
        elif boundary_part == "bottom":
            # Initial condition (t = 0)
            return self.initial_condition(x) if callable(self.initial_condition) else self.initial_condition
        elif boundary_part == "top":
            # Terminal condition (t = T)
            return self.terminal_condition(x) if callable(self.terminal_condition) else self.terminal_condition
```

**Example Usage**:

```python
# Define boundary data for entire space-time cylinder
boundary_data = SpacetimeBoundaryData(
    # Lateral: u(t, x ∈ ∂Ω) = sin(2πt) * x
    spatial_bc=dirichlet_bc(value=lambda x, t: np.sin(2*np.pi*t) * x[0]),

    # Bottom: m(0, x) = exp(-x²)
    initial_condition=lambda x: np.exp(-x[0]**2),

    # Top: u(T, x) = x²
    terminal_condition=lambda x: x[0]**2
)

# Use in global space-time solver
solver = GlobalSpacetimeSolver(...)
u_field = solver.solve(boundary_data=boundary_data)
```

---

## 4. Consistency and Compatibility

### 4.1 The Corner Problem

**Issue**: What happens at corners where different boundary parts meet?

**Corners of the Cylinder**:
- **Bottom-Lateral**: $(t=0, x \in \partial\Omega)$ - where initial condition meets spatial BC
- **Top-Lateral**: $(t=T, x \in \partial\Omega)$ - where terminal condition meets spatial BC

**Consistency Requirement**: Boundary data must be **continuous** at corners!

**Example (1D domain $[0, 1]$)**:

```python
# Spatial BC: u(t, x=0) = sin(πt)
spatial_bc = dirichlet_bc(value=lambda x, t: np.sin(np.pi * t), dimension=1)

# Terminal condition: u(T, x) = x²
terminal_condition = lambda x: x[0]**2

# POTENTIAL INCONSISTENCY at corner (t=T, x=0):
# - Spatial BC says: u(T, 0) = sin(πT)
# - Terminal condition says: u(T, 0) = 0²= 0
#
# If sin(πT) ≠ 0, we have a CONTRADICTION!
```

**Solution**: Enforce consistency

```python
def check_boundary_consistency(boundary_data: SpacetimeBoundaryData, T: float, domain_bounds):
    """
    Check that boundary data is consistent at cylinder corners.
    """
    # Check bottom-lateral corners
    for x_corner in get_spatial_boundary_points(domain_bounds):
        spatial_at_t0 = boundary_data.spatial_bc.evaluate(x_corner, time=0.0)
        initial_at_x = boundary_data.initial_condition(x_corner)

        if not np.isclose(spatial_at_t0, initial_at_x):
            raise ValueError(
                f"Inconsistent BC at (t=0, x={x_corner}): "
                f"spatial_bc gives {spatial_at_t0}, initial_condition gives {initial_at_x}"
            )

    # Check top-lateral corners
    for x_corner in get_spatial_boundary_points(domain_bounds):
        spatial_at_tT = boundary_data.spatial_bc.evaluate(x_corner, time=T)
        terminal_at_x = boundary_data.terminal_condition(x_corner)

        if not np.isclose(spatial_at_tT, terminal_at_x):
            raise ValueError(
                f"Inconsistent BC at (t={T}, x={x_corner}): "
                f"spatial_bc gives {spatial_at_tT}, terminal_condition gives {terminal_at_x}"
            )
```

### 4.2 MFG-Specific Considerations

**For Mean Field Games**, boundary consistency is particularly important because:

1. **HJB**: Has terminal condition $u(T, x) = g(x)$ and spatial BCs $u(t, x \in \partial\Omega)$
2. **FP**: Has initial condition $m(0, x) = m_0(x)$ and spatial BCs $m(t, x \in \partial\Omega)$

**Common scenario**: No-flux spatial BCs for FP

```python
# Fokker-Planck with no-flux lateral BCs
fp_boundary_data = SpacetimeBoundaryData(
    spatial_bc=no_flux_bc(dimension=2),  # ∂m/∂n = 0 on [0,T] × ∂Ω
    initial_condition=lambda x: gaussian(x, mu=0, sigma=1),  # m(0, x)
    terminal_condition=None  # Not used for forward evolution
)

# No-flux means: ∂m/∂n = 0 for all t ∈ [0,T]
# This is automatically consistent with any initial condition!
```

**Dirichlet spatial BCs** require more care:

```python
# If spatial BC is Dirichlet with time-varying value
spatial_bc = dirichlet_bc(value=lambda x, t: boundary_function(x, t))

# Then initial condition MUST satisfy:
# m₀(x) = boundary_function(x, 0) for x ∈ ∂Ω

# Otherwise, we have a discontinuity at t=0!
```

---

## 5. Implications for Global Solvers

### 5.1 Weak Formulation on Space-Time

**Variational Formulation**: Instead of solving timestep-by-timestep, we can formulate the PDE **globally on $Q$**.

**Example: Heat equation**

**Strong Form** (classical):
$$
\begin{cases}
\partial_t u - \Delta u = f & \text{in } Q = (0,T) \times \Omega \\
u = g & \text{on } \Gamma_{\text{lateral}} \\
u(0, \cdot) = u_0 & \text{on } \Gamma_{\text{bottom}}
\end{cases}
$$

**Weak Form** (space-time):

Find $u \in H^1(Q)$ such that $u|_{\partial Q} = g_{\text{boundary}}$ and
$$
\int_Q (\partial_t u \cdot v + \nabla u \cdot \nabla v) \, dx \, dt = \int_Q f \cdot v \, dx \, dt \quad \forall v \in H^1_0(Q)
$$

where the boundary data $g_{\text{boundary}}$ is the unified specification on $\partial Q$.

### 5.2 Finite Element Discretization

**Space-Time Finite Elements**: Use basis functions $\phi_{i,j}(t, x)$ that span both time and space.

```python
# Basis: Tensor product of temporal and spatial basis
φ(t, x) = ψ_temporal(t) ⊗ ψ_spatial(x)

# Example: Linear in time, linear in space
φ_{i,j}(t, x) = ψ_i(t) * ψ_j(x)
```

**Boundary enforcement**: All BCs (lateral, bottom, top) are enforced **simultaneously** in the global system.

```python
class SpacetimeFEMSolver:
    """
    Finite element solver on space-time cylinder.
    """

    def assemble_system(self, boundary_data: SpacetimeBoundaryData):
        """
        Assemble global stiffness matrix with BCs.
        """
        K = self.build_stiffness_matrix()  # (Nt*Nx) × (Nt*Nx)
        f = self.build_load_vector()

        # Apply boundary conditions on entire ∂Q
        K_bc, f_bc = self.apply_boundary_data(K, f, boundary_data)

        return K_bc, f_bc

    def apply_boundary_data(self, K, f, boundary_data):
        """
        Enforce BCs on lateral, bottom, and top simultaneously.
        """
        # Lateral boundary (spatial BCs over time)
        for t_idx in range(self.Nt):
            t = self.time_grid[t_idx]
            for x_boundary_node in self.get_spatial_boundary_nodes():
                value = boundary_data.spatial_bc.evaluate(x_boundary_node, time=t)
                # Set row/col in global matrix
                global_idx = self.get_global_index(t_idx, x_boundary_node)
                K, f = apply_dirichlet_bc(K, f, global_idx, value)

        # Bottom boundary (initial condition)
        for x_node in self.get_all_spatial_nodes():
            value = boundary_data.initial_condition(x_node)
            global_idx = self.get_global_index(t_idx=0, x_node)
            K, f = apply_dirichlet_bc(K, f, global_idx, value)

        # Top boundary (terminal condition, if present)
        if boundary_data.terminal_condition is not None:
            for x_node in self.get_all_spatial_nodes():
                value = boundary_data.terminal_condition(x_node)
                global_idx = self.get_global_index(t_idx=Nt-1, x_node)
                K, f = apply_dirichlet_bc(K, f, global_idx, value)

        return K, f
```

**Advantage**: This naturally handles time-varying spatial BCs - they're just part of the boundary data on $\partial Q$!

---

## 6. Connection to Current MFG_PDE Architecture

### 6.1 What Already Exists

MFG_PDE **already has** the foundation for space-time BCs:

1. ✅ **Time-varying spatial BCs**: `value=callable(x, t)` in `BCSegment`
2. ✅ **Initial/terminal conditions**: Passed separately to solvers
3. ✅ **BC evaluation at time `t`**: `apply_boundary_conditions_nd(..., time=t)`

**What's missing**:

1. ❌ Unified `SpacetimeBoundaryData` class
2. ❌ Consistency checking at cylinder corners
3. ❌ Global space-time solvers that use this unified specification

### 6.2 Proposed Enhancement

**Add Space-Time BC Unification** (optional, for global solvers):

```python
# mfg_pde/geometry/boundary/spacetime.py

from dataclasses import dataclass
from typing import Callable
import numpy as np
from .conditions import BoundaryConditions

@dataclass
class SpacetimeBoundaryData:
    """
    Unified boundary data specification for space-time cylinder [0,T] × Ω.

    This class unifies:
    - Spatial BCs on lateral surface [0,T] × ∂Ω
    - Initial condition on bottom cap {0} × Ω (for forward PDEs)
    - Terminal condition on top cap {T} × Ω (for backward PDEs)

    Usage:
        # For Fokker-Planck (forward)
        bc_data = SpacetimeBoundaryData(
            spatial_bc=no_flux_bc(dimension=2),
            initial_condition=lambda x: exp(-||x||²),
            terminal_condition=None  # Not used for forward evolution
        )

        # For HJB (backward)
        bc_data = SpacetimeBoundaryData(
            spatial_bc=neumann_bc(dimension=2),
            initial_condition=None,  # Not used for backward evolution
            terminal_condition=lambda x: g(x)
        )
    """

    # Lateral surface: [0,T] × ∂Ω
    spatial_bc: BoundaryConditions

    # Bottom cap: {0} × Ω (initial condition for forward PDEs)
    initial_condition: np.ndarray | Callable[[np.ndarray], float] | None = None

    # Top cap: {T} × Ω (terminal condition for backward PDEs)
    terminal_condition: np.ndarray | Callable[[np.ndarray], float] | None = None

    def check_consistency(self, T: float, domain_bounds: np.ndarray, tol: float = 1e-10):
        """
        Check boundary data consistency at cylinder corners.

        Raises:
            ValueError if boundary data is discontinuous at corners
        """
        # (Implementation as shown in Section 4.1)
        pass
```

**Backward Compatibility**: This is **optional** - existing code continues to work!

```python
# Old way (still works)
hjb_solver.solve_hjb_system(
    M_density=...,
    U_final_condition_at_T=terminal_condition,
    ...
)

# New way (for global solvers)
boundary_data = SpacetimeBoundaryData(
    spatial_bc=bc,
    terminal_condition=terminal_condition
)
global_solver.solve(boundary_data=boundary_data)
```

---

## 7. Practical Benefits

### 7.1 For Sequential Solvers (Current)

**Benefit 1**: **Consistency Validation**

Even if using time-stepping, can check BC consistency:

```python
# Before solving
boundary_data = SpacetimeBoundaryData(
    spatial_bc=my_time_varying_bc,
    initial_condition=m0,
    terminal_condition=None
)

# Validate (catches errors before expensive solve!)
boundary_data.check_consistency(T=problem.T, domain_bounds=problem.domain_bounds)

# Then solve as usual
fp_solver.solve_fp_system(m0, U_solution, ...)
```

**Benefit 2**: **Documentation and Clarity**

Makes the complete boundary specification explicit:

```python
# Instead of scattered BC definitions
bc1 = ...  # Somewhere
ic = ...   # Somewhere else
tc = ...   # Another place

# Single coherent specification
boundary_data = SpacetimeBoundaryData(
    spatial_bc=bc1,
    initial_condition=ic,
    terminal_condition=tc
)
```

### 7.2 For Global Solvers (Future)

**Benefit 1**: **Natural Time-Varying BCs**

Global solvers treat time-varying spatial BCs **naturally**:

```python
# Time-varying Dirichlet: u(t, x=0) = sin(ωt)
bc = dirichlet_bc(value=lambda x, t: np.sin(omega * t))

# Global solver assembles full space-time matrix
# Time variation is just part of the RHS vector!
```

**Benefit 2**: **Weak Enforcement**

Can use Lagrange multipliers or penalty methods to enforce BCs weakly:

```python
# Nitsche's method for weakly enforcing Dirichlet BCs
# Automatically handles time-varying BCs
```

---

## 8. Summary

### The Unification

| Concept | Classical View | Space-Time View |
|:--------|:--------------|:----------------|
| **Spatial BC** | BC at each timestep | Function on $\Gamma_{\text{lateral}} = [0,T] \times \partial\Omega$ |
| **Initial Condition** | Special initial data | Dirichlet BC on $\Gamma_{\text{bottom}} = \{0\} \times \Omega$ |
| **Terminal Condition** | Special final data | Dirichlet BC on $\Gamma_{\text{top}} = \{T\} \times \Omega$ |
| **Time-Varying BC** | BC function evaluated at each step | Natural: just a function on lateral surface |

### Key Insights

1. **Geometric Unity**: All BCs are boundary data on $\partial Q = \Gamma_{\text{lateral}} \cup \Gamma_{\text{bottom}} \cup \Gamma_{\text{top}}$

2. **Time-Varying BCs**: Are just functions on the lateral surface - no special treatment needed in space-time formulation

3. **Consistency**: Can check corner compatibility when all boundary data is unified

4. **Global Solvers**: Naturally handle all BC types uniformly

### Implementation Status

**MFG_PDE Current**:
- ✅ Supports time-varying BCs via `callable(x, t)`
- ✅ Evaluates at each timestep
- ⏸️ No unified space-time BC class
- ⏸️ No consistency checking

**Proposed Enhancement**:
- Add `SpacetimeBoundaryData` class (optional)
- Consistency validation
- Use in future global solvers

**Backward Compatible**: All existing code continues to work!

---

**Contributors**: Claude Opus 4.5
**Date**: 2026-01-17
**Related**: Space-Time Operator Architecture Proposal
