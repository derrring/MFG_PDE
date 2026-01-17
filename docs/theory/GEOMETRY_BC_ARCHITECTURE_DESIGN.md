# Geometry and Boundary Condition Architecture Design

**Document Type**: Theoretical Design Specification
**Status**: Design Phase (v0.17.1 baseline, v1.0.0 vision)
**Date**: 2026-01-17
**Authors**: MFG_PDE Core Team, Expert Consultation

---

## Executive Summary

This document defines the **comprehensive architectural design** for geometry and boundary condition handling in MFG_PDE. The design synthesizes:

1. **Modern PDE solver practices** (FEniCS, PETSc, Dedalus patterns)
2. **MFG-specific requirements** (coupled systems, equilibrium consistency)
3. **Multi-backend support** (structured grids, implicit domains, graphs, meshes)
4. **Advanced mathematical structures** (variational inequalities, free boundaries, manifolds)

**Core Principles**:
- **Operator-based abstraction**: Solvers request operations, not raw geometry data
- **Three-tier BC hierarchy**: Standard BCs, Variational Constraints, Dynamic Interfaces
- **Geometry-physics separation**: Mark regions geometrically, apply physics semantically
- **Framework-first**: Use existing infrastructure, never bypass it

---

## Related Documentation

This theoretical design document is part of a comprehensive BC/geometry architecture redesign. Related documentation:

### Architecture & Workflow
- **`docs/architecture/BC_COMPLETE_WORKFLOW.md`**: Complete BC workflow from user specification to solver application
  - §1: BC Classification (temporal vs spatial, grid vs particle paradigms)
  - §3: Target architecture (SSOT pattern, geometry owns spatial BC)
  - §4: Method-specific BC application patterns (FDM, FEM, Particle)
  - §6: BC sources and consumers (geometry as single source of truth)

- **`docs/architecture/BC_SPECIFICATION_VS_APPLICATOR.md`**: Two-layer architecture design (specification vs applicator)
  - §3: Specification Layer (`BoundaryConditions`, `BCSegment`, factory functions)
  - §4: Applicator Layer (`BaseBCApplicator` hierarchy, method-specific implementations)
  - §5: How specification and applicator connect (current vs ideal state)

### Integration & Capabilities
- **`docs/development/BC_SOLVER_INTEGRATION_DESIGN.md`**: Paradigm-specific BC helpers (Issue #527)
  - §2: Four paradigm helpers (Numerical field ops, Neural BC loss, RL env bounds, Optimization constraints)
  - §3: Geometry protocol extensions for BC-related methods
  - Status: Phase 2-3 complete (unified BC access via `BaseMFGSolver.get_boundary_conditions()`)

- **`docs/development/BC_CAPABILITY_MATRIX.md`**: Current solver BC support matrix (v0.17.1 baseline)
  - §2: BC type support by solver (HJB FDM/GFDM/SemiLag/WENO, FP FDM/Particle/GFDM)
  - §3: Infrastructure integration level (which solvers use unified BC framework)
  - §4: Detailed solver analysis (BC source, applicator mechanism, gaps)

### Implementation Details
- **`docs/development/issue_574_robin_bc_design.md`**: State-dependent BC (adjoint-consistent Robin BC)
  - Implemented in v0.17.1 using Robin BC framework (`BCType.ROBIN` with α=0, β=1)
  - Example of Tier 1 BC with state-dependent values (coupling FP density to HJB BC)
  - Demonstrates framework extensibility

### Implementation Plans
- **`docs/development/GEOMETRY_BC_IMPLEMENTATION_PLANS.md`**: Implementation roadmap for this design
  - 4 strategic options (Conservative Sequential, Parallel Tracks, Minimal Viable, Aggressive)
  - Detailed phase breakdown for Plan A (recommended, 12-16 weeks)
  - Risk analysis, resource requirements, success metrics

### Current Implementation (v0.17.1)
- **Code**: `mfg_pde/geometry/boundary/`
  - `conditions.py`: `BoundaryConditions` unified class (supports uniform and mixed BCs)
  - `types.py`: `BCType` enum, `BCSegment` dataclass
  - `applicator_fdm.py`: FDM ghost cell BC application (1D/2D/3D/nD)
  - `applicator_fem.py`: FEM mesh-based BC application (2D/3D)
  - `applicator_meshfree.py`: Meshfree/collocation BC application (GFDM, RBF)
  - `applicator_graph.py`: Graph/network BC application
  - `applicator_particle.py`: Particle/SDE BC application (reflection, absorption)
  - `bc_coupling.py`: State-dependent BC utilities (Issue #574)
  - `dispatch.py`: Unified BC application dispatcher (Issue #527)

**Reading Order for New Contributors**:
1. Start: `BC_COMPLETE_WORKFLOW.md` (big picture)
2. Deep dive: `BC_SPECIFICATION_VS_APPLICATOR.md` (architecture layers)
3. Current state: `BC_CAPABILITY_MATRIX.md` (what works today)
4. Theory: This document (comprehensive design vision)
5. Practice: `GEOMETRY_BC_IMPLEMENTATION_PLANS.md` (how to build it)

---

## Part I: Philosophical Foundations

### 1.1 The Operator Abstraction Paradigm

**Central Tenet**: *Solvers should be geometry-agnostic.*

A solver for the MFG system:
$$\begin{aligned}
-\partial_t U + \mathcal{L} U + H(\nabla U) &= f(m) \\
\partial_t m + \mathcal{L}^* m + \text{div}(m \alpha) &= 0
\end{aligned}$$

should **not** know whether $\mathcal{L}$ is:
- A finite difference stencil (TensorProductGrid)
- A FEM stiffness matrix (UnstructuredMesh)
- A graph Laplacian (GraphGeometry)
- A meshless RBF operator (PointCloud)

**Implementation**: Geometry provides operators as callables or matrices.

```python
# Solver code (geometry-agnostic)
laplacian = geometry.get_laplacian_operator()
gradient = geometry.get_gradient_operator()

# Usage in HJB residual
residual = -laplacian(U) + hamiltonian(gradient(U)) - f(m)
```

**Benefits**:
1. **Extensibility**: New geometries require no solver changes
2. **Testability**: Mock operators for unit tests
3. **Mathematical clarity**: Code mirrors equations

---

### 1.2 Boundary Conditions Are Not Geometry

**Historical Mistake**: Coupling BC specification to grid structure.

**Modern Approach**: Two-layer abstraction.

#### Layer 1: Geometric Marking (Where)

```python
# Mark boundaries by geometry
domain.mark_boundary("inlet", lambda x: x[0] < tolerance)
domain.mark_boundary("exit", sdf_region=lambda x: norm(x - x_exit) - r_exit)
```

**Properties**:
- Dimension-agnostic
- Physics-neutral
- Supports complex regions (SDF-based)

#### Layer 2: Physical Semantics (What)

```python
# Apply physics to marked regions
bc_inlet = DirichletBC(value=1.0, sub_domain="inlet")
bc_exit = NeumannBC(value=0.0, sub_domain="exit")
bc = mixed_bc([bc_inlet, bc_exit], ...)
```

**Properties**:
- Type-safe (BCType enum)
- Time-dependent (callables)
- Priority-based (for overlaps)

**Separation Benefits**:
- Same geometry, multiple physics scenarios
- Geometry reusable across problems
- BC specification portable

---

### 1.3 The Three-Tier Boundary Condition Hierarchy

Based on expert recommendations, BCs are **not monolithic**. They form a hierarchy by mathematical complexity.

#### Tier 1: Classical Boundary Conditions (Linear)

**Mathematical Structure**: Linear algebraic constraints on solution.

$$\alpha u + \beta \frac{\partial u}{\partial n} = g(x,t) \quad \text{on } \partial\Omega$$

**Types**:
- Dirichlet: $u = g$ (α=1, β=0)
- Neumann: $\partial u/\partial n = g$ (α=0, β=1)
- Robin: $\alpha u + \beta \partial u/\partial n = g$ (mixed)
- Periodic: $u(x_{\min}) = u(x_{\max})$ (topology-based)

**Implementation**: Standard ghost cell, matrix row modification, or weak form assembly.

**Status in MFG_PDE**: ✅ Production-ready (v0.17.1)

---

#### Tier 2: Variational Constraints (Nonlinear)

**Mathematical Structure**: Inequality constraints, complementarity conditions.

$$u \geq \psi \quad \text{(obstacle)} \quad \text{or} \quad \min(F(u), u - \psi) = 0 \quad \text{(VI)}$$

**Use Cases in MFG**:
1. **Obstacle problems**: Value cannot fall below a threshold
   - Physical: Minimum payoff in game
   - Mathematical: Variational inequality

2. **Optimal stopping**: Agents can exit at stopping regions
   - Exit set $E$: Where $u(x) = g(x)$ (terminal payoff)
   - Continuation set $C$: Where PDE holds

3. **Congestion constraints**: Density cannot exceed capacity
   - $m(x,t) \leq m_{\max}(x)$
   - Pressure term emerges: $p(m) = \lambda$ if $m = m_{\max}$

**Key Distinction**: These are **NOT** boundary conditions in the classical sense. They are:
- Domain-wide constraints
- Complementarity conditions
- Part of the nonlinear solver, not the BC applicator

**Implementation Strategy**: Solver middleware (projections, penalty methods).

```python
# NOT a BoundaryCondition, but a Constraint
problem.solve(
    scheme=NumericalScheme.FDM_UPWIND,
    constraints=[
        ObstacleConstraint(lower_bound=psi),
        CapacityConstraint(max_density=m_max)
    ]
)
```

**Status in MFG_PDE**: ❌ Not implemented (planned v0.18.0)

---

#### Tier 3: Dynamic Interfaces (PDE-Coupled)

**Mathematical Structure**: Boundary $\partial\Omega(t)$ evolves according to a PDE.

$$\begin{aligned}
F(u, \nabla u) &= 0 \quad \text{in } \Omega(t) \\
V_{\text{interface}} &= \mathcal{F}(\nabla u, m, ...) \quad \text{on } \partial\Omega(t)
\end{aligned}$$

**Use Cases in MFG**:
1. **Stefan problems**: Phase boundaries (congested/free flow)
2. **Free boundaries**: Optimal exercise boundary (American options)
3. **Front propagation**: Epidemic spread, information diffusion

**Implementation Strategy**: Level Set Method (Eulerian, not Lagrangian).

**Level Set Formulation**:
- Interface $\Gamma(t)$ represented as zero level set: $\phi(x,t) = 0$
- Evolution PDE: $\partial \phi/\partial t + F(\phi, \nabla\phi) = 0$
- Reinitialization: Maintain signed distance property

**Advantages for MFG**:
- Handles topology changes (crowd splitting)
- Works on fixed grids (no remeshing)
- Natural for TensorProductGrid

**Status in MFG_PDE**: ❌ Not implemented (research-level, v0.19.0+)

---

## Part II: Geometry Abstraction Architecture

### 2.1 The Geometry Protocol (ABC)

**Design Philosophy**: Trait-based composition, not monolithic inheritance.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class GeometryProtocol(Protocol):
    """Minimal contract all geometries must satisfy."""

    @property
    def dimension(self) -> int:
        """Spatial dimension."""

    @property
    def num_points(self) -> int:
        """Total degrees of freedom."""

    def get_coordinates(self) -> NDArray:
        """Return spatial coordinates of all points."""
```

**Rationale**: Not all geometries support all operations. Use traits to advertise capabilities.

---

### 2.2 Trait Hierarchy: Solver Capabilities

#### Trait 1: Supports Laplacian (Diffusion)

```python
@runtime_checkable
class SupportsLaplacian(Protocol):
    """Geometry can provide diffusion operator."""

    def get_laplacian_operator(self) -> Callable[[NDArray], NDArray]:
        """
        Return operator L such that L[u] ≈ ∇²u.

        Returns:
            Callable taking u (shape: num_points) and returning L[u].

        Note:
            - TensorProductGrid: Returns finite difference stencil
            - UnstructuredMesh: Returns FEM stiffness matrix application
            - GraphGeometry: Returns graph Laplacian matrix
        """
```

**Usage in Solver**:
```python
def solve_heat_equation(u0, geometry: SupportsLaplacian, dt, steps):
    laplacian = geometry.get_laplacian_operator()
    u = u0.copy()
    for _ in range(steps):
        u = u + dt * laplacian(u)  # Forward Euler
    return u
```

---

#### Trait 2: Supports Gradient (Advection)

```python
@runtime_checkable
class SupportsGradient(Protocol):
    """Geometry can compute spatial gradients."""

    def get_gradient_operator(self) -> Callable[[NDArray], NDArray]:
        """
        Return operator ∇ computing spatial gradients.

        Returns:
            Callable taking u (shape: num_points) and returning
            ∇u (shape: (dimension, num_points)).

        Note:
            Sign convention: ∇u[d, i] is ∂u/∂x_d at point i.
        """
```

**Dimension-Specific**:
- 1D: Returns ∂u/∂x (shape: (1, Nx))
- 2D: Returns [∂u/∂x, ∂u/∂y] (shape: (2, Nx*Ny))
- Graph: **Undefined** (no continuous gradient)

**Implication**: Solvers requiring gradients cannot use GraphGeometry.

---

#### Trait 3: Supports Upwinding (Hyperbolic)

```python
@runtime_checkable
class SupportsUpwinding(Protocol):
    """Geometry can provide upwind stencils."""

    def get_upwind_stencil(
        self,
        velocity_field: NDArray,
        direction: str = "positive"
    ) -> Callable[[NDArray], NDArray]:
        """
        Return upwind derivative operator based on flow direction.

        Args:
            velocity_field: Advection velocity at each point
            direction: "positive" or "negative" (flow direction)

        Returns:
            Upwind derivative operator for ∂u/∂x

        Note:
            - Grid: One-sided finite difference
            - Graph: Edge-based upwinding (flow from high to low potential)
        """
```

**Application**: Godunov upwinding in HJB, flux-limited advection in FP.

---

#### Trait 4: Supports Boundary Detection

```python
@runtime_checkable
class SupportsBoundary(Protocol):
    """Geometry can identify and mark boundaries."""

    def is_on_boundary(
        self,
        points: NDArray,
        tolerance: float = 1e-8
    ) -> NDArray[bool]:
        """Check if points are on boundary."""

    def get_boundary_normal(
        self,
        points: NDArray
    ) -> NDArray:
        """Return outward unit normal at boundary points."""

    def mark_boundary(
        self,
        name: str,
        predicate: Callable[[NDArray], bool]
    ) -> None:
        """Mark boundary region by name for BC application."""
```

**Implementation Variants**:
- **TensorProductGrid**: Axis-aligned detection ($x_i = x_{\min}$ or $x_i = x_{\max}$)
- **ImplicitDomain**: SDF-based ($|\phi(x)| < \epsilon$)
- **UnstructuredMesh**: Face tags from mesh generator
- **GraphGeometry**: Node classification (boundary nodes manually specified)

---

### 2.3 Multi-Backend Support: The Geometry Zoo

MFG_PDE supports **four geometry families**, each with different strengths:

#### Family 1: Structured Grids (TensorProductGrid)

**Characteristics**:
- Cartesian product: $\Omega = [a_1, b_1] \times \cdots \times [a_d, b_d]$
- Uniform or non-uniform spacing
- $O(1)$ neighbor lookup
- Cache-friendly memory layout

**Supported Operations**:
- ✅ Laplacian: FD stencils (2nd/4th/6th order)
- ✅ Gradient: Central/upwind differences
- ✅ Upwinding: Godunov, WENO-3/5
- ✅ Boundary: Axis-aligned ghost cells

**Best For**:
- High-performance simulations
- Rectangular domains
- Separable PDEs (ADI methods)

**Limitation**: Cannot conform to curved boundaries.

**Workaround**: Embedded domain methods (see Tier 3).

---

#### Family 2: Implicit Domains (SDF-Based)

**Characteristics**:
- Domain $\Omega$ defined by signed distance function $\phi(x)$
- $\Omega = \{x : \phi(x) < 0\}$
- Boundary $\partial\Omega = \{x : \phi(x) = 0\}$
- Background grid (Cartesian) + SDF defines complex shapes

**Supported Operations**:
- ✅ Laplacian: Via penalization or ghost fluid method
- ✅ Gradient: SDF gradient gives normals
- ✅ Boundary: SDF evaluation ($|\phi| < \epsilon$)
- 🟡 Upwinding: Requires special treatment near interface

**Best For**:
- Complex geometries on simple grids
- Moving boundaries (Level Set Method)
- No mesh generation required

**Example Use Cases**:
- Crowd evacuation (non-rectangular rooms)
- Obstacle problems (irregular obstacles)
- Free boundary problems (interface tracking)

**Implementation Strategy**:
```python
# Define domain implicitly
sdf = lambda x: np.linalg.norm(x - center) - radius  # Circle/sphere
domain = ImplicitDomain(dimension=2, sdf=sdf, bounds=[(-2, 2), (-2, 2)])

# Boundary automatically detected via SDF
domain.is_on_boundary(points)  # |φ(x)| < ε
domain.get_boundary_normal(points)  # ∇φ/|∇φ|
```

---

#### Family 3: Unstructured Meshes (FEM)

**Characteristics**:
- Triangular/tetrahedral elements
- Node connectivity graph
- Boundary face tagging (from Gmsh, etc.)

**Supported Operations**:
- ✅ Laplacian: FEM stiffness matrix assembly
- ✅ Gradient: Element-wise interpolation
- ✅ Boundary: Face tags from mesh file
- 🟡 Upwinding: SUPG/DG methods

**Best For**:
- Complex geometries with exact boundary conforming
- Adaptive refinement (h-adaptivity)
- Variational formulations

**Limitation**: Mesh generation overhead, harder to handle topology changes.

**Current Status**: Infrastructure present, weak form assembly incomplete.

---

#### Family 4: Graphs (Network MFG)

**Characteristics**:
- Discrete state space: $\Omega = V$ (vertex set)
- Edges $E$ define connectivity
- No continuous space ($dx$ undefined)

**Supported Operations**:
- ✅ Laplacian: Graph Laplacian $L = D - A$
- ❌ Gradient: **Undefined** (no continuous derivatives)
- ✅ Upwinding: Edge-based (flow along edges)
- ✅ Boundary: Designated boundary nodes

**Best For**:
- Social networks
- Transportation networks
- Epidemic models on contact graphs

**Mathematical Differences**:
- FP becomes master equation: $\dot{m}_i = \sum_j (T_{ij} m_j - T_{ji} m_i)$
- HJB becomes discrete DP: $u_i = \min_a [c(i,a) + \gamma \sum_j P_{ij}^a u_j]$
- No Hamiltonian in continuous sense

---

### 2.4 Geometry Selection Logic (Factory Pattern)

**Problem**: How should `problem.solve()` choose geometry?

**Solution**: Explicit geometry construction (no magic).

```python
# User constructs geometry first
from mfg_pde.geometry import TensorProductGrid

geometry = TensorProductGrid(
    dimension=2,
    bounds=[(0, 10), (0, 10)],
    Nx_points=[101, 101]
)

# Then creates problem
problem = MFGProblem(
    geometry=geometry,
    T=1.0,
    Nt=100,
    ...
)
```

**Scheme-Geometry Compatibility**:

| Scheme | Compatible Geometries |
|:-------|:---------------------|
| `FDM_UPWIND` | TensorProductGrid |
| `FDM_CENTRAL` | TensorProductGrid |
| `WENO3`, `WENO5` | TensorProductGrid |
| `GFDM` | TensorProductGrid, ImplicitDomain, PointCloud |
| `FEM_P1`, `FEM_P2` | UnstructuredMesh |
| `GRAPH_MFG` | GraphGeometry |
| `PARTICLE_LAGRANGIAN` | Any (uses point cloud sampling) |

**Runtime Check**:
```python
# In solver factory
if scheme == NumericalScheme.FDM_UPWIND:
    if not isinstance(geometry, TensorProductGrid):
        raise TypeError(
            f"{scheme} requires TensorProductGrid, got {type(geometry).__name__}"
        )
```

---

## Part III: Boundary Condition Framework Design

### 3.1 The Unified BoundaryConditions Class

**Design Goal**: One class for all BC types (uniform, mixed, time-dependent, callable).

```python
@dataclass
class BoundaryConditions:
    """
    Unified BC specification (dimension-agnostic).

    Supports:
    - Uniform BCs: Same type everywhere
    - Mixed BCs: Different types on different boundaries
    - Time-dependent: Callable values
    - SDF regions: Boundary segments defined by SDF
    """
    dimension: int | None = None  # Lazy binding
    segments: list[BCSegment] = field(default_factory=list)
    default_bc: BCType = BCType.PERIODIC
    default_value: float = 0.0

    # Domain specification
    domain_bounds: NDArray | None = None  # Rectangular
    domain_sdf: Callable | None = None    # General
```

**Key Features**:

1. **Lazy Dimension Binding**:
   ```python
   bc = dirichlet_bc(value=0.0)  # dimension=None
   bc_2d = bc.bind_dimension(2)  # dimension=2 (via Geometry)
   ```

2. **Priority-Based Resolution**:
   - Multiple segments can overlap
   - Highest priority wins
   - Enables corner/edge special handling

3. **SDF-Based Regions**:
   ```python
   # Exit region: small circle at top
   exit_sdf = lambda x: np.linalg.norm(x - np.array([5, 9])) - 0.5
   exit_seg = BCSegment(
       name="exit",
       bc_type=BCType.DIRICHLET,
       value=0.0,
       sdf_region=exit_sdf,
       priority=2  # Higher than walls
   )
   ```

---

### 3.2 BCSegment Specification

**Design**: Each segment is a boundary region + BC type + value.

```python
@dataclass
class BCSegment:
    """
    BC specification for a geometric boundary segment.

    Matching modes:
    1. Rectangular: boundary="left", region={"y": (2, 8)}
    2. SDF: sdf_region=lambda x: ...
    3. Normal: normal_direction=np.array([0, 1])
    """
    name: str
    bc_type: BCType
    value: float | Callable = 0.0

    # Robin coefficients: α*u + β*∂u/∂n = g
    alpha: float = 1.0  # Dirichlet weight
    beta: float = 0.0   # Neumann weight

    # Matching specifications
    boundary: str | None = None  # "left", "right", "x_min", etc.
    region: dict | None = None   # {"y": (2, 8)} for partial boundaries
    sdf_region: Callable | None = None  # SDF defining segment
    normal_direction: NDArray | None = None  # Target outward normal

    priority: int = 0  # For overlap resolution

    # Tier 2 feature: Flux-limited absorption
    flux_capacity: float | None = None  # Max mass/time absorption
```

**Example: Mixed BC with Exit**:
```python
# Walls: Neumann (no-flux)
wall = BCSegment(
    name="walls",
    bc_type=BCType.NEUMANN,
    value=0.0,
    priority=0  # Low priority (default)
)

# Exit: Dirichlet (absorbing)
exit = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=0.0,
    boundary="x_max",
    region={"y": (4, 6)},  # Only middle section
    priority=1  # Higher priority
)

bc = mixed_bc([wall, exit], dimension=2, ...)
```

---

### 3.3 BC Application: The Applicator Pattern

**Design Philosophy**: Separate BC **specification** from **application**.

```python
class BCApplicatorProtocol(Protocol):
    """Protocol for BC application strategies."""

    def apply_bc(
        self,
        u: NDArray,
        bc: BoundaryConditions,
        time: float = 0.0
    ) -> NDArray:
        """
        Apply boundary conditions to solution array.

        Implementation depends on discretization:
        - FDM: Ghost cells (padding)
        - FEM: Modify matrix rows / penalty method
        - Particle: Reflection / absorption
        """
```

**Implementations**:

| Applicator | Method | Use Case |
|:-----------|:-------|:---------|
| `FDMApplicator` | Ghost cells | Structured grids |
| `FEMApplicator` | Weak form assembly | Unstructured meshes |
| `ParticleApplicator` | Reflection/absorption | Lagrangian methods |
| `GraphApplicator` | Node classification | Network MFG |

**Factory Pattern**:
```python
def get_applicator(geometry: Geometry) -> BCApplicator:
    if isinstance(geometry, TensorProductGrid):
        return FDMApplicator(geometry)
    elif isinstance(geometry, UnstructuredMesh):
        return FEMApplicator(geometry)
    elif isinstance(geometry, GraphGeometry):
        return GraphApplicator(geometry)
    else:
        raise TypeError(f"No applicator for {type(geometry)}")
```

---

### 3.4 State-Dependent Boundary Conditions

**Use Case**: Adjoint-consistent BC in MFG (Issue #574).

**Mathematical Form**:
$$\frac{\partial U}{\partial n} = -\frac{\sigma^2}{2} \frac{\partial \ln(m)}{\partial n} \quad \text{(depends on current density } m \text{)}$$

**Design Pattern**: Create new BC object each iteration.

```python
# In Picard iteration
for iteration in range(max_iterations):
    # Solve FP
    m_new = fp_solver.solve(U_prev)

    # Create density-dependent BC for HJB
    hjb_bc = create_adjoint_consistent_bc(
        m_current=m_new,
        geometry=problem.geometry,
        sigma=problem.sigma
    )

    # Solve HJB with updated BC
    U_new = hjb_solver.solve(m_new, bc=hjb_bc)
```

**Implementation** (v0.17.1 architecture):
```python
def create_adjoint_consistent_bc_1d(
    m_current: NDArray,
    dx: float,
    sigma: float,
    domain_bounds: NDArray | None = None,
) -> BoundaryConditions:
    """
    Create Robin BC coupling to FP density.

    Returns:
        BoundaryConditions with Robin segments where
        α=0, β=1, g=-σ²/2 · ∂ln(m)/∂n
    """
    # Compute density gradients
    grad_left = compute_boundary_log_density_gradient_1d(m_current, dx, "left")
    grad_right = compute_boundary_log_density_gradient_1d(m_current, dx, "right")

    # Create Robin BC segments
    segments = [
        BCSegment(
            name="left_adjoint_consistent",
            bc_type=BCType.ROBIN,
            alpha=0.0,  # No U term
            beta=1.0,   # ∂U/∂n coefficient
            value=-sigma**2 / 2 * grad_left,
            boundary="x_min",
        ),
        BCSegment(
            name="right_adjoint_consistent",
            bc_type=BCType.ROBIN,
            alpha=0.0,
            beta=1.0,
            value=-sigma**2 / 2 * grad_right,
            boundary="x_max",
        ),
    ]

    return mixed_bc(segments, dimension=1, domain_bounds=domain_bounds)
```

**Key Insight**: State-dependent BCs are **not callable values** but **regenerated BC objects**.

---

## Part IV: Advanced Mathematical Structures

### 4.1 Variational Inequalities (Tier 2)

**Mathematical Formulation**:

Find $u \in K$ such that:
$$\langle F(u), v - u \rangle \geq 0 \quad \forall v \in K$$

where $K = \{u : u \geq \psi\}$ (convex constraint set).

**Equivalent Formulation** (complementarity):
$$\min(F(u), u - \psi) = 0$$

**Physical Interpretation**:
- $u$: Value function
- $\psi$: Obstacle (minimum payoff)
- Contact set: $\{x : u(x) = \psi(x)\}$ (where constraint is active)

**Projection Method** (simplest solver):

```python
# Iterative projection
for k in range(max_iterations):
    # Solve unconstrained PDE
    u_star = solve_pde(u_k)

    # Project onto constraint set
    u_{k+1} = np.maximum(u_star, obstacle)

    # Check convergence
    if np.linalg.norm(u_{k+1} - u_k) < tolerance:
        break
```

**Design as Solver Middleware**:

```python
class ObstacleConstraint:
    """Tier 2: Variational inequality constraint."""

    def __init__(self, lower_bound: NDArray | Callable):
        self.lower_bound = lower_bound

    def project(self, u: NDArray, x: NDArray = None, t: float = 0.0) -> NDArray:
        """Project solution onto constraint set."""
        if callable(self.lower_bound):
            psi = self.lower_bound(x, t)
        else:
            psi = self.lower_bound
        return np.maximum(u, psi)

# User API
problem.solve(
    scheme=NumericalScheme.FDM_UPWIND,
    constraints=[ObstacleConstraint(lower_bound=psi)]
)
```

**Advanced Solvers** (future):
- **Semismooth Newton**: Quadratic convergence for VIs
- **Active Set Methods**: Track contact set explicitly
- **Primal-Dual**: Augmented Lagrangian formulation

---

### 4.2 Free Boundary Problems (Tier 3)

**Stefan Problem** (classical example):

$$\begin{aligned}
\text{Heat equation:} \quad \partial_t u &= \nabla^2 u \quad \text{in } \Omega(t) \\
\text{Interface:} \quad V_n &= [\nabla u \cdot n] \quad \text{on } \partial\Omega(t) \\
\text{BC on interface:} \quad u &= 0 \quad \text{on } \partial\Omega(t)
\end{aligned}$$

where $\Omega(t)$ evolves and $V_n$ is normal velocity, $[\cdot]$ is jump across interface.

**MFG Application**: Congestion phase transition.
- Free flow region: $m < m_{\text{crit}}$
- Congested region: $m = m_{\text{crit}}$
- Interface: Propagating congestion front

**Level Set Method** (recommended approach):

1. **Interface Representation**:
   $$\Gamma(t) = \{x : \phi(x,t) = 0\}$$

2. **Evolution PDE**:
   $$\frac{\partial \phi}{\partial t} + V_n |\nabla \phi| = 0$$

   where $V_n = F(\nabla u, m, ...)$ depends on PDE solution.

3. **Reinitialization** (maintain SDF property):
   $$\frac{\partial \phi}{\partial \tau} = \text{sign}(\phi_0)(1 - |\nabla \phi|)$$

**Advantages for MFG**:
- Works on fixed Cartesian grid (TensorProductGrid)
- Handles topology changes (crowd splitting/merging)
- Natural coupling to HJB/FP (both on same grid)

**Design Pattern**:

```python
class LevelSetEvolver:
    """Tier 3: Dynamic interface tracking."""

    def __init__(self, phi0: NDArray, geometry: TensorProductGrid):
        self.phi = phi0  # Initial interface
        self.geometry = geometry

    def evolve_step(self, velocity_field: NDArray, dt: float) -> NDArray:
        """
        Evolve interface by one timestep.

        Solves: ∂φ/∂t + V·∇φ = 0
        """
        # Hamilton-Jacobi solver for φ
        grad_phi = self.geometry.get_gradient_operator()(self.phi)
        phi_new = self.phi - dt * velocity_field * np.linalg.norm(grad_phi, axis=0)
        return phi_new

    def reinitialize(self, phi: NDArray, iterations: int = 5) -> NDArray:
        """Restore signed distance property."""
        # Solve: ∂φ/∂τ = sign(φ_0)(1 - |∇φ|)
        phi_sdf = phi.copy()
        for _ in range(iterations):
            grad = self.geometry.get_gradient_operator()(phi_sdf)
            grad_norm = np.linalg.norm(grad, axis=0)
            phi_sdf = phi_sdf - 0.1 * np.sign(phi) * (grad_norm - 1.0)
        return phi_sdf

# Usage in MFG
level_set = LevelSetEvolver(phi0, geometry)

for t in time_steps:
    # Solve MFG system
    U, m = solve_mfg_step(...)

    # Compute interface velocity (depends on solution)
    velocity = compute_interface_velocity(U, m)

    # Evolve interface
    phi_new = level_set.evolve_step(velocity, dt)

    # Reinitialize periodically
    if t % 10 == 0:
        phi_new = level_set.reinitialize(phi_new)

    level_set.phi = phi_new
```

---

### 4.3 Weak Boundary Imposition: Nitsche's Method

**Motivation**: Strong imposition (modify matrix rows) is inflexible for:
- Unfitted boundaries (ImplicitDomain)
- Time-dependent geometries
- High-order methods

**Nitsche's Method** (for Dirichlet BC):

**Weak form with Nitsche penalty**:
$$a(u,v) = \int_\Omega \nabla u \cdot \nabla v - \int_{\partial\Omega} \frac{\partial u}{\partial n} v - \int_{\partial\Omega} u \frac{\partial v}{\partial n} + \frac{\gamma}{h} \int_{\partial\Omega} u v = \int_\Omega fv + \frac{\gamma}{h} \int_{\partial\Omega} g v$$

**Parameters**:
- $\gamma$: Penalty parameter (typically 10-100)
- $h$: Mesh size
- $g$: Prescribed boundary value

**Benefits**:
1. ✅ No matrix row modification (symmetric, well-conditioned)
2. ✅ Works with unfitted boundaries
3. ✅ Optimal convergence rate
4. ✅ Extends to Robin/Neumann naturally

**Implementation** (FEM context):

```python
class NitscheBC:
    """Weak BC imposition via Nitsche's method."""

    def __init__(self, gamma: float = 10.0):
        self.gamma = gamma  # Penalty parameter

    def assemble_bc_terms(
        self,
        basis: Callable,
        boundary_nodes: NDArray,
        g: float | Callable,
        h: float  # Mesh size
    ) -> tuple[SparseMatrix, NDArray]:
        """
        Assemble Nitsche BC terms.

        Returns:
            (A_bc, b_bc) where:
            A_bc: Penalty matrix (γ/h ∫ uv on boundary)
            b_bc: RHS contribution (γ/h ∫ gv on boundary)
        """
        # Penalty term: γ/h * ∫_∂Ω u v
        penalty_matrix = self.gamma / h * assemble_mass_matrix(
            basis, nodes=boundary_nodes
        )

        # Consistency term: - ∫_∂Ω ∂u/∂n v - ∫_∂Ω u ∂v/∂n
        # (Requires normal derivatives - complex)

        # RHS: γ/h * ∫_∂Ω g v
        rhs_bc = self.gamma / h * assemble_rhs(g, basis, boundary_nodes)

        return penalty_matrix, rhs_bc

# Usage in FEM solver
A = assemble_stiffness_matrix(...)
b = assemble_rhs(...)

A_bc, b_bc = nitsche.assemble_bc_terms(basis, boundary_nodes, g=0.0, h=dx)

A_total = A + A_bc
b_total = b + b_bc

u = solve(A_total, b_total)
```

**Status in MFG_PDE**: ❌ Not implemented (FEM infrastructure incomplete)

---

### 4.4 Penalty Methods for Embedded Boundaries

**Idea**: Enforce BC by adding penalty term to PDE.

**Mathematical Formulation**:
$$\mathcal{L}u = f \quad \Rightarrow \quad \mathcal{L}u + \frac{1}{\epsilon} \chi_{\partial\Omega}(u - g) = f$$

where:
- $\chi_{\partial\Omega}$: Characteristic function of boundary region (or narrow band)
- $\epsilon \to 0$: Penalty parameter (small → strong enforcement)
- $g$: Boundary value

**Volume Penalization** (for obstacles):
$$\mathcal{L}u = f \quad \Rightarrow \quad \mathcal{L}u + \frac{1}{\epsilon} \chi_{\Omega^c} (u - g_{\text{obstacle}}) = f$$

where $\Omega^c$ is obstacle interior.

**Advantages**:
1. ✅ No grid conforming needed
2. ✅ Works on Cartesian grids
3. ✅ Simple implementation
4. ✅ Good for moving boundaries

**Disadvantages**:
1. ❌ Stiff system ($\epsilon$ small)
2. ❌ Lower accuracy near boundary
3. ❌ Requires implicit time stepping

**Implementation**:

```python
def penalty_bc(
    u: NDArray,
    sdf: Callable,
    g: float,
    epsilon: float = 1e-3
) -> NDArray:
    """
    Apply penalty BC via volume penalization.

    Returns:
        Penalty term: (1/ε) χ(x) (u - g)
    """
    # Identify boundary region (narrow band)
    x = geometry.get_coordinates()
    phi = sdf(x)

    # Characteristic function (smoothed)
    chi = smooth_heaviside(-phi, width=2*dx)

    # Penalty term
    penalty = (1.0 / epsilon) * chi * (u - g)
    return penalty

# In time stepper
def time_step_with_penalty(u_n, dt, laplacian, penalty_bc):
    # Implicit Euler with penalty
    # (I - dt*L - dt*P)[u_{n+1}] = u_n

    A = identity - dt * laplacian - dt * (1/epsilon) * diag(chi)
    b = u_n - dt * (1/epsilon) * chi * g

    u_new = solve(A, b)
    return u_new
```

**Application**: Obstacles on Cartesian grids (no remeshing).

---

## Part V: Stability and Well-Posedness

### 5.1 GKS Stability Condition (Internal Quality Standard)

**Gustafsson-Kreiss-Sundström Condition**: Ensures BC discretization doesn't destabilize scheme.

**Mathematical Statement**:

For a stable PDE scheme (Lax-Richtmyer stable), adding a BC discretization maintains stability if:
$$\text{Eigenvalues of BC-modified scheme satisfy } |\lambda| \leq 1 + C\Delta t$$

**Practical Check**:

1. **Periodic problem is stable**: Verify base scheme
2. **BC doesn't introduce growing modes**: Eigenvalue analysis of BC stencil
3. **Numerical group velocity**: Waves propagate correctly near boundary

**Implementation Strategy**:

```python
# In test suite (NOT user API)
def test_gks_neumann_bc_fdm():
    """
    Verify Neumann BC satisfies GKS for FDM Laplacian.

    Method:
    1. Construct FD matrix with Neumann BC
    2. Compute eigenvalues
    3. Check: max(|λ|) ≤ 1 + O(dx)
    """
    # Build Laplacian matrix with Neumann BC
    L = build_laplacian_matrix_1d(Nx=100, bc_type="neumann")

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(L.toarray())
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # GKS check: For heat equation, L eigenvalues should be non-positive
    assert max_eigenvalue <= 1e-10, f"Unstable BC: max |λ| = {max_eigenvalue}"
```

**Expert Recommendation**: This is **internal validation**, not user-facing API.

---

### 5.2 Lopatinskii-Shapiro Condition (PDE Well-Posedness)

**Purpose**: Ensures IBVP (Initial-Boundary Value Problem) is **well-posed** at PDE level.

**Heuristic** (for hyperbolic systems):
- **Inflow boundary**: Can specify BC (characteristics enter domain)
- **Outflow boundary**: Cannot specify BC (characteristics leave domain)

**Example**: 1D advection $\partial_t u + c \partial_x u = 0$ with $c > 0$
- **Left boundary** ($x=0$): Inflow → Dirichlet OK ✅
- **Right boundary** ($x=L$): Outflow → Dirichlet **VIOLATES** L-S ❌

**Correct BC**: Only specify $u(0,t) = g(t)$. At $x=L$, use extrapolation or do-nothing BC.

**Implementation as Runtime Check**:

```python
def verify_lopatinskii_shapiro(
    pde_type: str,
    velocity_field: NDArray,
    bc: BoundaryConditions
) -> None:
    """
    Check L-S condition for hyperbolic PDEs.

    Raises:
        BCStabilityError if BC violates well-posedness.
    """
    if pde_type == "hyperbolic":
        # Check flow direction at boundaries
        for seg in bc.segments:
            if seg.bc_type == BCType.DIRICHLET:
                # Get normal velocity
                if seg.boundary == "x_min":
                    v_normal = velocity_field[0]  # v at left
                elif seg.boundary == "x_max":
                    v_normal = velocity_field[-1]  # v at right

                # Check: Dirichlet only allowed on inflow
                if seg.boundary == "x_min" and v_normal < 0:
                    raise BCStabilityError("Outflow Dirichlet at x_min")
                if seg.boundary == "x_max" and v_normal > 0:
                    raise BCStabilityError("Outflow Dirichlet at x_max")
```

**Expert Recommendation**: Internal validation during solver initialization.

---

## Part VI: Design Patterns and Best Practices

### 6.1 Framework-First Principle

**Rule**: Never bypass existing infrastructure.

**Anti-Pattern** (Issue #574 flawed implementation):
```python
# BAD: Manual dictionary threading
bc_values = {"x_min": compute_value_left(), "x_max": compute_value_right()}
solver.solve(..., bc_values=bc_values)  # Bypasses BC framework
```

**Correct Pattern** (Issue #574 refactored):
```python
# GOOD: Use BC framework
bc = create_adjoint_consistent_bc(...)  # Returns BoundaryConditions object
solver.solve(..., bc=bc)  # Framework applies it
```

**Rationale**:
1. ✅ Reuses existing applicator infrastructure
2. ✅ Works with all backends (FDM, GFDM, FEM, particles)
3. ✅ Type-safe, testable
4. ✅ Extends to nD naturally

---

### 6.2 Single Source of Truth

**Rule**: Geometry owns its boundary conditions.

```python
class Geometry(ABC):
    def __init__(self, boundary_conditions: BoundaryConditions):
        self._bc = boundary_conditions

    def get_boundary_conditions(self) -> BoundaryConditions:
        """Single source of truth for BCs."""
        return self._bc

# Solvers retrieve BCs from geometry
bc = problem.geometry.get_boundary_conditions()
```

**Benefits**:
- No BC duplication
- Consistent across solvers
- Easier to modify globally

---

### 6.3 Lazy Binding for Reusability

**Pattern**: Defer dimension specification until needed.

```python
# Create reusable BC template
bc_template = dirichlet_bc(value=0.0)  # dimension=None

# Bind to specific geometry
bc_1d = bc_template.bind_dimension(1)
bc_2d = bc_template.bind_dimension(2)
```

**Use Case**: Factory functions that work for any dimension.

---

## Part VII: Migration and Evolution Strategy

### 7.1 Current Status (v0.17.1)

✅ **Production-Ready**:
- Tier 1 BCs (DNR + Mixed + Periodic)
- TensorProductGrid geometry
- ImplicitDomain (SDF) geometry
- GraphGeometry
- FDM applicators (ghost cells, 2-layer architecture)
- Robin BC framework (Issue #574 refactored)

🟡 **Infrastructure Present, Needs Implementation**:
- UnstructuredMesh (FEM weak form assembly)
- Particle applicators (basic reflection)

❌ **Not Implemented**:
- Tier 2 (Variational constraints)
- Tier 3 (Level set evolution)
- GKS/L-S validation
- Nitsche's method
- Penalty methods

---

### 7.2 Backward Compatibility Strategy

**Principle**: Evolution, not revolution.

1. **Deprecation Path**:
   - Mark old API as deprecated (warnings)
   - Keep for 2-3 minor versions
   - Remove in v1.0.0

2. **Parallel APIs**:
   ```python
   # Old (deprecated but working)
   problem = MFGProblem(domain=[0,1], Nx=100, ...)

   # New (recommended)
   geometry = TensorProductGrid(dimension=1, bounds=[(0,1)], Nx_points=[101])
   problem = MFGProblem(geometry=geometry, ...)
   ```

3. **Feature Flags**:
   - New features opt-in initially
   - Become default after stabilization
   - Old behavior removed in major version

---

## Part VIII: Open Research Questions

### 8.1 Lipschitz Boundaries on Manifolds

**Challenge**: MFG on curved spaces (e.g., sphere, torus).

**Approach**: Embedding + Penalty.
- Embed manifold in $\mathbb{R}^{d+1}$
- Use Cartesian grid
- Penalize off-manifold region

**Alternative**: Intrinsic coordinates (requires differential geometry).

---

### 8.2 Adaptive Mesh Refinement (AMR)

**Challenge**: Balance accuracy and performance.

**Strategy**: Quadtree/Octree for TensorProductGrid.
- Refine near shocks, interfaces
- Coarsen in smooth regions
- Maintain neighbor connectivity

**Implementation**: Complex (planned v0.20.0+).

---

### 8.3 High-Order Boundary Treatment

**Challenge**: WENO-5 requires 3 ghost cells.

**Current**: Linear extrapolation (accuracy loss).

**Better**: Polynomial extrapolation matching interior order.

**Research**: Compact stencils at boundaries.

---

## Appendix A: Glossary

### A.1 Acronyms and Abbreviations

| Term | Definition |
|:-----|:-----------|
| **BC** | Boundary Condition |
| **DNR** | Dirichlet-Neumann-Robin (classical BC types) |
| **SDF** | Signed Distance Function (φ(x) < 0 inside, > 0 outside) |
| **VI** | Variational Inequality |
| **GKS** | Gustafsson-Kreiss-Sundström stability condition |
| **L-S** | Lopatinskii-Shapiro well-posedness condition |
| **IBVP** | Initial-Boundary Value Problem |
| **FEM** | Finite Element Method |
| **FDM** | Finite Difference Method |
| **GFDM** | Generalized Finite Difference Method (meshless) |

### A.2 Boundary Condition Terminology: Physics vs Numerics

**CRITICAL DISTINCTION**: Some BC terms have different meanings in physics (PDE semantics) vs numerics (implementation).

#### No-Flux vs Zero-Gradient

**BCType.NO_FLUX** - Physics interpretation depends on PDE:

**For Fokker-Planck Equation** (mass conservation):
- **Physics**: Zero **total flux** J·n = 0, where J = -σ²/2∇m + m·α
- **Meaning**: No mass crosses boundary (reflecting wall for agents)
- **Implementation**: `FPNoFluxCalculator` (physics-aware, requires drift field α)
- **Ghost cell**: Computed to enforce J·n = 0 using both diffusion and advection
- **Equation**: Coupled to density m and drift α

**For HJB/Poisson Equation** (edge extension):
- **Physics**: Zero **gradient** ∂u/∂n = 0
- **Meaning**: Solution extends smoothly at boundary (no information from boundary)
- **Implementation**: `ZeroGradientCalculator` (simple Neumann BC)
- **Ghost cell**: u_ghost = u_interior (2nd-order symmetry)
- **Equation**: Pure Neumann boundary condition

**For Advection-Diffusion** (general case):
- **Physics**: Zero **net flux** -κ∇u·n + u·v·n = 0
- **Implementation**: `AdvectionDiffusionNoFluxCalculator`
- **Requires**: Both diffusion κ and velocity v

**Summary**:
```
NO_FLUX ≠ ZERO_GRADIENT in general!

No-flux is physics (J·n = 0, depends on PDE type)
Zero-gradient is numerics (∂u/∂n = 0, simple Neumann)
```

#### Reflecting vs Absorbing vs No-Flux

**Three related but distinct concepts**:

**Reflecting BC** (particle methods):
- **Application**: Particle/SDE methods (FP equation)
- **Meaning**: Particles bounce elastically at boundary
- **Implementation**: Velocity reversal v → -v when particle hits wall
- **Grid equivalent**: No-flux BC (J·n = 0)
- **BCType**: `BCType.REFLECTING` (particle-specific)

**Absorbing BC** (particle methods / FP Dirichlet):
- **Application**: Particle/SDE methods, exits
- **Meaning**: Particles removed when reaching boundary
- **Implementation**: Delete particle from system
- **Grid equivalent**: Dirichlet BC m = 0
- **BCType**: `BCType.DIRICHLET` with value=0 for FP
- **Physical meaning**: Agents leave domain (exit door)

**No-Flux BC** (grid methods):
- **Application**: Grid-based FP equation
- **Meaning**: Zero total flux through boundary
- **Implementation**: Ghost cells enforcing J·n = 0
- **Particle equivalent**: Reflecting BC
- **BCType**: `BCType.NO_FLUX`

**Relationship**:
```
Grid-based FP:  NO_FLUX   ←→  Particle-based FP: REFLECTING
Grid-based FP:  DIRICHLET(0) ←→  Particle-based FP: ABSORBING
```

#### Extrapolation BC (Unbounded Domains)

**When**: Domain is effectively unbounded (far-field decay expected).

**Types**:
- **Linear Extrapolation**: `BCType.EXTRAPOLATION_LINEAR`
  - Assumes: ∂²u/∂n² = 0 (linear profile at boundary)
  - Ghost: u_g = 2u₁ - u₂
  - Order: 2nd-order for smooth solutions

- **Quadratic Extrapolation**: `BCType.EXTRAPOLATION_QUADRATIC`
  - Assumes: ∂³u/∂n³ = 0 (quadratic profile)
  - Ghost: u_g = 3u₁ - 3u₂ + u₃
  - Order: 3rd-order

- **Sommerfeld BC** (wave radiation):
  - Assumes: ∂u/∂t + c·∂u/∂n = 0 (outgoing wave)
  - Application: Hyperbolic problems
  - Not yet implemented in MFG_PDE

**Use Case**: MFG on large domains where agents can "leave to infinity".

### A.3 Geometry Trait System vs GeometryProtocol

**CLARIFICATION**: The trait system **augments** the existing `GeometryProtocol`, not replaces it.

#### GeometryProtocol (Base Requirements)

**Purpose**: Minimal contract all geometries must satisfy.

**Core Properties**:
- `dimension`: Spatial dimension
- `num_spatial_points`: Total number of discrete points
- `geometry_type`: Type classification (grid, mesh, graph, etc.)

**Spatial Data Access**:
- `get_spatial_grid()`: Returns grid/mesh representation
- `get_bounds()`: Returns bounding box
- `get_grid_shape()`: Returns discretization shape

**Boundary Queries** (mandatory - every domain has boundary):
- `is_on_boundary(points)`: Check if points are on boundary
- `get_boundary_normal(points)`: Get outward normal at boundary
- `project_to_boundary(points)`: Project points onto boundary
- `get_boundary_regions()`: Named boundary regions for mixed BCs

#### Trait Protocols (Optional Capabilities)

**Purpose**: Advertise which operations a geometry supports.

**Operator Traits**:
- `SupportsLaplacian`: Can compute Laplacian operator
- `SupportsGradient`: Can compute gradient operator
- `SupportsDivergence`: Can compute divergence operator
- `SupportsAdvection`: Can compute advection operator

**Boundary Traits**:
- `SupportsBoundaryNormal`: Can compute outward normals
- `SupportsBoundaryProjection`: Can project onto boundary
- `SupportsBoundaryDistance`: Can compute distance to boundary (SDF)

**Topological Traits**:
- `SupportsManifold`: Is a smooth manifold
- `SupportsLipschitz`: Has Lipschitz-continuous boundary

#### Usage Pattern (Runtime Trait Checking)

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mfg_pde.geometry import GeometryProtocol, SupportsLaplacian

def solve_poisson(geometry: GeometryProtocol):
    """Solve Poisson equation on given geometry."""

    # All geometries have dimension (base protocol)
    d = geometry.dimension

    # Check if geometry supports operation we need
    if isinstance(geometry, SupportsLaplacian):
        laplacian = geometry.get_laplacian_operator()
        # ... solve using operator
    else:
        raise TypeError(
            f"{type(geometry).__name__} doesn't support Laplacian operator. "
            f"Required traits: SupportsLaplacian"
        )
```

#### Migration Strategy (Backward Compatible)

**Existing Code**: No changes required
```python
# Old code continues to work
geometry = TensorProductGrid(...)
problem = MFGProblem(geometry=geometry, ...)
```

**New Trait Implementation** (multiple inheritance):
```python
class TensorProductGrid(
    GeometryProtocol,          # Base protocol (required)
    SupportsLaplacian,          # Trait (optional capability)
    SupportsGradient,           # Trait
    SupportsDivergence,         # Trait
    SupportsAdvection,          # Trait
    SupportsBoundaryNormal,     # Trait
):
    """Production-ready structured grid with full operator support."""

    # GeometryProtocol methods (required)
    @property
    def dimension(self) -> int:
        return self._dimension

    # SupportsLaplacian methods (trait)
    def get_laplacian_operator(self, order=2, bc=None) -> LinearOperator:
        from mfg_pde.backends.fdm import build_laplacian_matrix
        return build_laplacian_matrix(self.grid_shape, self.dx, order, bc)

    # ... other trait methods
```

**Benefit**: Existing geometries gain traits incrementally without breaking API.

---

## Appendix B: References

**Theoretical Foundations**:
1. Gustafsson, Kreiss, Oliger: "Time Dependent Problems and Difference Methods" (1995)
2. Quarteroni, Valli: "Numerical Approximation of Partial Differential Equations" (2008)
3. Osher, Fedkiw: "Level Set Methods and Dynamic Implicit Surfaces" (2003)

**Software Design**:
1. FEniCSx Documentation (operator-based abstraction)
2. PETSc Manual (SNESVI for variational inequalities)
3. Dedalus Documentation (spectral BC handling)

**MFG-Specific**:
1. Achdou, Capuzzo-Dolcetta: "Mean Field Games: Numerical Methods" (2010)
2. Lasry, Lions: "Mean Field Games" (2007)
3. Issue #574 (MFG_PDE): Adjoint-consistent boundary conditions

---

**Document Version**: 1.0
**Last Updated**: 2026-01-17
**Next Review**: After v0.18.0 implementation
