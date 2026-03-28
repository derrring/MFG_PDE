# Boundary Condition Enforcement Architecture

**Status**: Architectural design document (updated v0.17.16)
**Version**: v0.17.0+ roadmap
**Related**: Issue #542 (fixed), Issue #549 (planned), Issue #625 (adjoint-consistent BC)
**Last Revised**: 2026-03-28

---

## Current State (v0.17.16)

### What Works

**Unified specification layer** (v0.17.0+):
- ✅ `BCSegment` + `BoundaryConditions`: Dimension-agnostic, solver-agnostic specification
- ✅ 8 `BCType` values: DIRICHLET, NEUMANN, ROBIN, PERIODIC, NO_FLUX, REFLECTING, EXTRAPOLATION_*
- ✅ 5 boundary matching modes: coordinate, region, SDF, normal, region_name
- ✅ Dynamic BC values: `BCValueProvider` protocol, `AdjointConsistentProvider`
- ✅ Mixed BC: Priority-based segment resolution
- ✅ Corner handling: Priority, average, or mollify strategies

**Applicator hierarchy** (v0.17.0+):
- ✅ 6 concrete applicators: FDM, FEM, Meshfree, Implicit, Particle, Graph
- ✅ Dispatch layer: `apply_bc()` auto-selects applicator from geometry
- ✅ Topology/Calculator decomposition: Reusable physics across geometries

**Partially implemented**:
- ⚠️ FEM: Dirichlet via `condense()`, Neumann as natural BC. Robin/Periodic planned.
- ⚠️ Neural: No unified BC integration. Each PINN/DGM solver has ad-hoc loss terms.
- ⚠️ RL: No BC framework integration.

### Remaining Limitations

- ❌ Free boundaries (moving frontiers)
- ❌ Equation-dependent BC resolution (see Section: The Resolution Problem)
- ❌ Neural solver BC integration (loss-based enforcement)
- ❌ Robin/Periodic FEM enforcement via FacetBasis

---

## Architectural Problems

### 1. The Resolution Problem: Physical Intent vs Mathematical BC ⚠️ CRITICAL

**The same physical intent maps to different mathematical BCs depending on the PDE.**

Consider "no-flux" (impermeable wall) across the MFG system:

| PDE | "No-flux" means | Mathematical BC | Implementation |
|-----|----------------|-----------------|----------------|
| **Pure diffusion** ($\partial m/\partial t = D\Delta m$) | Zero gradient | Neumann: $\partial m/\partial n = 0$ | `ZeroGradientCalculator` |
| **Advection-diffusion** (Fokker-Planck: $\partial m/\partial t + \nabla\cdot(\alpha m) = D\Delta m$) | Zero total flux | Robin: $\alpha m \cdot n - D \partial m/\partial n = 0$ | `ZeroFluxCalculator` (needs drift $\alpha$, diffusion $D$) |
| **HJB** ($-\partial u/\partial t + H(\nabla u) = 0$) | Optimal control points inward | Variational inequality: $\nabla u \cdot n \leq 0$ | `AdjointConsistentProvider` or constraint |
| **Coupled MFG** (HJB boundary, density-dependent) | Adjoint-consistent reflecting | Dynamic Robin: $g = -\sigma^2/2 \cdot \partial(\ln m)/\partial n$ | `AdjointConsistentProvider` |
| **Particle** (Lagrangian FP) | Elastic reflection | $v \to v - 2(v \cdot n)n$ | `ParticleReflector` |
| **PINN** (neural) | Boundary loss | $L_{bc} = \|J \cdot n\|^2$ on $\partial\Omega$ | Custom loss term |

**Current state**: `BCType.NO_FLUX` dispatches to `ZeroGradientCalculator` by default
(pure Neumann), which is **incorrect for advection-diffusion**. The correct Robin-type
formula (`ZeroFluxCalculator`) requires `use_zero_flux=True` plus drift and diffusion
coefficients — but these live in the solver, not the BC specification.

**Root cause**: The architecture has 3 layers (specification → matching → enforcement)
but is missing a 4th layer: **equation-dependent resolution**.

**Why this matters beyond MFG**: Any multi-physics framework that supports multiple PDE
types will face this. "Impermeable wall" is universal physical language, but its
mathematical realization depends on the physics being modeled.

### 2. `BCType` Conflates Three Distinct Concepts

Current `BCType` mixes:

| Concept | Examples | What it describes |
|---------|----------|-------------------|
| **Mathematical BC** | DIRICHLET, NEUMANN, ROBIN, PERIODIC | PDE boundary operator |
| **Physical intent** | NO_FLUX, REFLECTING | What the boundary does physically |
| **Numerical technique** | EXTRAPOLATION_LINEAR, EXTRAPOLATION_QUADRATIC | How to handle truncated domains |

This conflation forces every applicator to understand all 8 types. `NO_FLUX` is not a
mathematical BC type — it's a physical intent that resolves to Neumann (pure diffusion),
Robin (advection-diffusion), or VI (HJB) depending on the equation. `REFLECTING` is the
particle interpretation of the same physical intent.

### 3. Monolithic Enforcement Logic (original problem, partially resolved)

Originally BC enforcement was embedded in Newton solver. Now extracted into applicators,
but each solver still wires up BCs differently:
- FDM: `pad_array_with_ghosts()` directly
- FEM: `apply_bc_to_fem_system(A, rhs, basis, bc)`
- Particle: 3 different code paths in `fp_particle.py`
- Neural: ad-hoc loss terms, no framework integration

---

## Proposed Architecture: Four-Layer BC Framework

### The Missing Layer: Equation-Dependent Resolution

The existing 3-layer model (WHAT/WHERE/HOW) must become 4 layers:

```
Layer 1: SPECIFICATION (user-facing)
    BCSegment: Physical intent + boundary location
    "This wall is impermeable"  "This exit absorbs"  "This wraps around"
         │
         ▼
Layer 2: RESOLUTION (solver-facing)  ← THE MISSING LAYER
    Physical intent × PDE type → Mathematical BC
    "No-flux for FP = Robin(α,D)"  "No-flux for HJB = Neumann or VI"
         │
         ▼
Layer 3: ENFORCEMENT (discretization-facing)
    Mathematical BC × Discretization → Discrete operations
    "Robin on FDM = ghost cell formula"  "Dirichlet on FEM = condense()"
         │
         ▼
Layer 4: APPLICATION (geometry-facing)
    Discrete operations × Geometry → Modified field/matrix/particles/loss
    "Ghost cells on TensorProductGrid"  "DOF elimination on TriangularMesh"
```

### Layer 1: Physical Intent (User Specification)

Users specify **what the boundary does**, not the mathematics:

```python
# Physical intent taxonomy
class BoundaryIntent:
    IMPERMEABLE = "impermeable"     # Nothing crosses (reflecting wall)
    ABSORBING = "absorbing"        # Things disappear (exit, drain)
    PERIODIC = "periodic"          # Wraps around (torus)
    FIXED_VALUE = "fixed_value"    # Field = prescribed value
    FIXED_FLUX = "fixed_flux"      # Flux = prescribed value
    FREE = "free"                  # Boundary location evolves
```

**Backward compatibility**: `BCType.NO_FLUX` → `BoundaryIntent.IMPERMEABLE`,
`BCType.REFLECTING` → `BoundaryIntent.IMPERMEABLE`,
`BCType.DIRICHLET` → `BoundaryIntent.FIXED_VALUE`, etc.

### Layer 2: Resolution (Solver Resolves Intent → Math)

Each solver family resolves physical intent into a mathematical BC using its
knowledge of the PDE being solved:

```python
class BCResolver(Protocol):
    """Resolves physical boundary intent into mathematical BC for a specific PDE."""

    def resolve(
        self,
        segment: BCSegment,
        solver_state: dict[str, Any],
    ) -> ResolvedBC:
        """
        Resolve physical intent into mathematical BC.

        Args:
            segment: Physical boundary specification
            solver_state: PDE coefficients and current solution state
                Keys may include: drift, diffusion, density, value_function, time

        Returns:
            ResolvedBC with concrete mathematical type and values
        """
        ...
```

**Solver-specific resolvers**:

```python
class FPResolver:
    """Resolves BC intent for Fokker-Planck equation."""

    def resolve(self, segment, solver_state):
        if segment.bc_type in (BCType.NO_FLUX, BCType.REFLECTING):
            drift = solver_state["drift"]           # α at boundary
            diffusion = solver_state["diffusion"]    # D = σ²/2
            # FP no-flux: J·n = αm·n - D(∂m/∂n) = 0  →  Robin BC
            return ResolvedBC(
                math_type=MathBCType.ROBIN,
                alpha=drift,        # coefficient of m
                beta=-diffusion,    # coefficient of ∂m/∂n
                value=0.0,          # J·n = 0
            )
        elif segment.bc_type == BCType.DIRICHLET:
            return ResolvedBC(
                math_type=MathBCType.DIRICHLET,
                value=segment.value,
            )
        ...


class HJBResolver:
    """Resolves BC intent for Hamilton-Jacobi-Bellman equation."""

    def resolve(self, segment, solver_state):
        if segment.bc_type in (BCType.NO_FLUX, BCType.REFLECTING):
            # HJB reflecting: ∇u·n ≤ 0 (variational inequality)
            # Simple approximation: Neumann ∂u/∂n = 0
            # Adjoint-consistent: Robin g = -σ²/2 · ∂(ln m)/∂n
            if segment.value is not None and isinstance(segment.value, BCValueProvider):
                # Dynamic provider (adjoint-consistent case)
                resolved_value = segment.value.compute(solver_state)
                return ResolvedBC(
                    math_type=MathBCType.ROBIN,
                    alpha=0.0,
                    beta=1.0,
                    value=resolved_value,
                )
            else:
                return ResolvedBC(
                    math_type=MathBCType.NEUMANN,
                    value=0.0,
                )
        ...


class ParticleResolver:
    """Resolves BC intent for particle/Lagrangian methods."""

    def resolve(self, segment, solver_state):
        if segment.bc_type in (BCType.NO_FLUX, BCType.REFLECTING):
            return ResolvedBC(
                math_type=MathBCType.REFLECT,  # Particle-specific
                normal=solver_state.get("boundary_normal"),
            )
        elif segment.bc_type == BCType.DIRICHLET:
            return ResolvedBC(
                math_type=MathBCType.ABSORB,   # Particle-specific
            )
        ...


class NeuralResolver:
    """Resolves BC intent for PINN/DGM methods."""

    def resolve(self, segment, solver_state):
        if segment.bc_type in (BCType.NO_FLUX, BCType.REFLECTING):
            pde_type = solver_state["pde_type"]  # "hjb" or "fp"
            if pde_type == "fp":
                # Loss: ||J·n||² where J = αm - D∇m
                return ResolvedBC(
                    math_type=MathBCType.FLUX_LOSS,
                    drift=solver_state["drift"],
                    diffusion=solver_state["diffusion"],
                )
            else:
                # Loss: ||max(∇u·n, 0)||² (penalize outward gradient)
                return ResolvedBC(
                    math_type=MathBCType.GRADIENT_INEQUALITY_LOSS,
                )
        ...
```

**Key insight**: `AdjointConsistentProvider` (Issue #625) is already a resolver —
it resolves "reflecting wall" into a concrete Robin value using the current density.
The resolution layer generalizes this pattern from "advanced feature" to "standard
mechanism."

### Layer 3: Enforcement (Discretization-Specific)

Once resolved to a mathematical BC, enforcement is purely discretization-dependent:

```python
class BCEnforcer(Protocol):
    """Enforces a resolved mathematical BC on a discrete representation."""

    def enforce(
        self,
        resolved_bc: ResolvedBC,
        field: Any,           # NDArray, sparse matrix, particle array, loss tensor
        geometry: Any,
        boundary_info: Any,   # Indices, face tags, node sets, sample points
    ) -> Any:
        ...
```

| Discretization | Enforcer | What it does |
|---------------|----------|-------------|
| FDM/GFDM | `GhostCellEnforcer` | Computes ghost values from `ResolvedBC` |
| FEM | `FEMEnforcer` | `condense()` for Dirichlet, `FacetBasis` integral for Robin |
| Particle | `TrajectoryEnforcer` | Reflect/absorb/wrap based on `ResolvedBC` |
| Neural | `LossEnforcer` | Constructs boundary loss from `ResolvedBC` |
| Graph | `NodeEnforcer` | Sets/constrains node values |

### Layer 4: Application (Geometry-Specific)

Maps abstract enforcement to concrete geometry operations (boundary identification,
index lookup, normal computation). This is what the current applicator dispatch does.

### Design Principles

1. **Separation of Concerns** (4-way):
   - `BCSegment`: **What** the boundary does physically (intent)
   - `BCResolver`: **Which** mathematical BC this implies (equation-dependent)
   - `BCEnforcer`: **How** to enforce it discretely (discretization-dependent)
   - `Geometry`: **Where** to apply it (geometry-dependent)

2. **Protocol-Based Extension**:
   - Define protocols for each component
   - Allow custom implementations via dependency injection
   - Preserve backward compatibility with default implementations

3. **Geometry-Agnostic BC Specification**:
   - Use abstract boundary identifiers, not coordinate names
   - Geometry provides mapping from IDs to discrete locations
   - Support both legacy strings ("x_min") and new IDs ("inlet")

4. **Resolution is Solver Responsibility**:
   - The solver knows the PDE. The BC spec knows the physics. Resolution bridges them.
   - `BCValueProvider` / `AdjointConsistentProvider` are the existing pattern for this.
   - Generalizing: every solver has a `BCResolver` that turns intent into math.

---

## Component Redesign

### 1. `ResolvedBC` — Mathematical BC After Resolution

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any

class MathBCType(Enum):
    """Pure mathematical BC types (no physical interpretation)."""
    DIRICHLET = "dirichlet"                 # u = g
    NEUMANN = "neumann"                     # du/dn = g
    ROBIN = "robin"                         # alpha*u + beta*du/dn = g
    PERIODIC = "periodic"                   # u(x_min) = u(x_max)
    VARIATIONAL_INEQUALITY = "vi"           # du/dn <= 0 (constrained)
    # Particle-specific (not PDE BC but trajectory operations)
    REFLECT = "reflect"                     # v -> v - 2(v.n)n
    ABSORB = "absorb"                       # Remove particle
    WRAP = "wrap"                           # Periodic repositioning
    # Neural-specific (loss terms)
    FLUX_LOSS = "flux_loss"                 # ||J.n||^2 penalty
    GRADIENT_INEQUALITY_LOSS = "grad_ineq"  # ||max(du/dn, 0)||^2

@dataclass
class ResolvedBC:
    """A fully resolved mathematical BC ready for enforcement."""
    math_type: MathBCType
    value: float | None = None
    alpha: float | None = None          # Robin: alpha*u
    beta: float | None = None           # Robin: beta*du/dn
    normal: Any = None                  # Boundary normal vector
    drift: Any = None                   # For flux-based BCs
    diffusion: float | None = None      # For flux-based BCs
    metadata: dict | None = None        # Solver-specific extras
```

### 2. `BCResolver` Protocol — Equation-Dependent Resolution

```python
class BCResolver(Protocol):
    """Resolves physical boundary intent into mathematical BC."""

    def resolve(
        self,
        segment: BCSegment,
        solver_state: dict[str, Any],
    ) -> ResolvedBC:
        """
        Resolve physical intent into mathematical BC.

        The resolver bridges physical language ("impermeable wall") and
        mathematical language ("Robin BC with alpha=drift, beta=-diffusion").

        Args:
            segment: Physical boundary specification (intent + location)
            solver_state: PDE coefficients at/near boundary:
                - drift: Advection velocity alpha(x)
                - diffusion: Diffusion coefficient D = sigma^2/2
                - density: Current density m(x) (for coupled BCs)
                - time: Current time t

        Returns:
            ResolvedBC with concrete mathematical type and parameters
        """
        ...
```

### 3. `BCEnforcer` Protocol — Discretization-Specific Enforcement

```python
class BCEnforcer(Protocol):
    """Enforces a resolved mathematical BC on a discrete representation."""

    def enforce(
        self,
        resolved_bc: ResolvedBC,
        target: Any,
        geometry: Any,
        boundary_indices: Any,
    ) -> Any:
        """
        Enforce a resolved BC on a discrete field/matrix/particle set.

        Args:
            resolved_bc: Mathematical BC from resolver
            target: What to modify (NDArray, sparse matrix, particle array, ...)
            geometry: Geometry object for spatial queries
            boundary_indices: Where to enforce (indices, face tags, ...)

        Returns:
            Modified target (or new target for immutable types)
        """
        ...
```

### Existing Code Mapped to New Layers

| Current code | New layer | New role |
|-------------|-----------|---------|
| `BCSegment`, `BoundaryConditions` | Layer 1: Specification | Unchanged |
| `BCValueProvider`, `AdjointConsistentProvider` | Layer 2: Resolution | Already resolvers (generalize) |
| `use_zero_flux` flag in `bc_to_topology_calculator()` | Layer 2: Resolution | Move to `FPResolver` |
| `ZeroGradientCalculator`, `ZeroFluxCalculator` | Layer 3: Enforcement | Ghost cell enforcers |
| `FDMApplicator`, `FEMApplicator`, etc. | Layer 3+4: Enforcement + Application | Refactor to consume `ResolvedBC` |
| `dispatch.apply_bc()` | Layer 4: Application | Orchestrates all layers |

### Existing Sections Below Preserved for Reference

The following sections (Geometry Boundary Mapping, Enhanced BCSegment, Integration,
Migration, Examples, Testing, etc.) remain from the v0.16.16 design and are still
relevant. They will be updated as the resolver layer is implemented.

---

### 2. Geometry Boundary Mapping (v0.16.16 original, still valid)

**Purpose**: Geometry provides mapping from abstract boundary IDs to discrete locations.

```python
class TensorProductGrid:
    """Current implementation - add boundary mapping method."""

    def map_boundary_to_indices(self, boundary_spec: str | int) -> np.ndarray:
        """
        Map boundary specification to grid indices.

        Args:
            boundary_spec: Either legacy string ("x_min") or boundary ID

        Returns:
            Array of indices where BC should be enforced
        """
        # Legacy string matching (backward compat)
        if boundary_spec == "x_min":
            return np.array([0])
        elif boundary_spec == "x_max":
            return np.array([self.Nx[0] - 1])
        elif boundary_spec == "y_min":
            return self._get_y_min_indices()  # 2D case
        elif boundary_spec == "y_max":
            return self._get_y_max_indices()

        # New boundary ID system
        elif isinstance(boundary_spec, int):
            return self._boundary_id_map[boundary_spec]

        else:
            raise ValueError(f"Unknown boundary: {boundary_spec}")


class TriangularMesh:
    """Future implementation for FEM/DGM."""

    def __init__(self, vertices, faces, boundary_markers):
        self.vertices = vertices
        self.faces = faces
        self.boundary_markers = boundary_markers  # Dict[str, array of face IDs]

    def map_boundary_to_indices(self, boundary_spec):
        """Return node indices on specified boundary."""
        if isinstance(boundary_spec, str):
            # String → boundary marker → face IDs → node IDs
            boundary_faces = self.boundary_markers[boundary_spec]
            return self._faces_to_nodes(boundary_faces)
        else:
            return self.boundary_markers[boundary_spec]
```

### 3. Enhanced BCSegment

**Purpose**: Support both legacy coordinate strings and abstract boundary IDs.

```python
@dataclass
class BCSegment:
    """Boundary condition specification (enhanced for v0.17.0+)."""

    name: str
    bc_type: BCType
    value: float | Callable

    # Legacy field (backward compat)
    boundary: str | None = None  # "x_min", "x_max", etc.

    # New field (v0.17.0+)
    boundary_id: str | int | None = None  # "inlet", "wall", 42, etc.

    # Mesh-based specification (FEM/DGM)
    boundary_faces: list[int] | None = None
    boundary_nodes: list[int] | None = None

    # Implicit boundary specification
    boundary_geometry: dict | None = None  # {"type": "plane", "normal": [...]}

    def get_boundary_spec(self):
        """Return boundary specification (prioritize new over legacy)."""
        if self.boundary_id is not None:
            return self.boundary_id
        elif self.boundary_faces is not None:
            return self.boundary_faces
        elif self.boundary_nodes is not None:
            return self.boundary_nodes
        elif self.boundary is not None:
            return self.boundary  # Legacy
        else:
            raise ValueError("No boundary specification provided")
```

---

## Integration with Solvers

### Current (v0.16.16)

BC enforcement is **embedded** in `solve_hjb_timestep_newton()`:

```python
def solve_hjb_timestep_newton(..., bc: BoundaryConditions | None = None):
    # Newton iteration
    for newton_iter in range(max_newton_iterations):
        # ... solve ...
        pass

    # Hardcoded enforcement
    if bc is not None:
        left_type, left_value = _get_bc_type_and_value_1d(bc, "left", current_time)
        if left_type == BCType.DIRICHLET:
            U[0] = left_value
        # ...
```

### Proposed (v0.17.0+)

BC enforcement is **injected** as strategy:

```python
def solve_hjb_timestep_newton(
    ...,
    bc: BoundaryConditions | None = None,
    bc_enforcer: BoundaryEnforcer | None = None  # ← New parameter
):
    # Default to fixed boundary enforcer
    if bc_enforcer is None:
        bc_enforcer = FixedBoundaryEnforcer()

    # Newton iteration
    for newton_iter in range(max_newton_iterations):
        # ... solve ...
        pass

    # Strategy-based enforcement
    if bc is not None:
        for segment in bc.segments:
            bc_enforcer.apply(
                U=U_n_current_newton_iterate,
                bc_segment=segment,
                geometry=geometry,
                time=current_time,
                dx=dx  # Pass solver state
            )
```

**Benefits**:
- ✅ Decoupled: Can swap enforcement strategy without modifying solver
- ✅ Testable: Can test enforcement independently
- ✅ Extensible: Add new enforcer types without touching core solver
- ✅ Backward compatible: Default enforcer provides current behavior

---

## Migration Path

### Phase 1: Add Protocol and Default Implementation (v0.17.0)

**Changes**:
1. Define `BoundaryEnforcer` protocol in `mfgarchon/geometry/boundary/enforcement.py`
2. Extract current logic into `FixedBoundaryEnforcer`
3. Add `bc_enforcer` parameter to solvers (default = `FixedBoundaryEnforcer()`)
4. Add `map_boundary_to_indices()` to `TensorProductGrid`
5. Add `boundary_id` field to `BCSegment` (optional)

**Backward compatibility**: 100% - existing code uses default enforcer

### Phase 2: Mesh-Based Enforcement (v0.18.0)

**Changes**:
1. Implement `MeshBoundaryEnforcer` for FEM/DGM
2. Add `TriangularMesh` geometry class with boundary markers
3. Add `boundary_faces`/`boundary_nodes` to `BCSegment`

**Use case**: Enable FEM/DGM solvers on triangulated domains

### Phase 3: Free Boundary Support (v0.19.0)

**Changes**:
1. Implement `FreeBoundaryEnforcer`
2. Add frontier tracking utilities (level set, moving mesh)
3. Add examples: Stefan problem, American options

**Use case**: Enable free boundary MFG problems

### Phase 4: Deprecate Legacy Strings (v1.0.0)

**Changes**:
1. Deprecation warnings for `boundary="x_min"` syntax
2. Require `boundary_id` for new code
3. Keep legacy support until v2.0.0

---

## Examples

### Current API (v0.16.16)

```python
from mfgarchon.geometry.boundary import BCSegment, BoundaryConditions
from mfgarchon.geometry.boundary.types import BCType

bc = BoundaryConditions(segments=[
    BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
    BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max"),
])

grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
problem = MFGProblem(geometry=grid, T=1.0, Nt=10)
solver = HJBFDMSolver(problem)
U = solver.solve()  # Uses default FixedBoundaryEnforcer
```

### Future API - Abstract Boundary IDs (v0.17.0+)

```python
bc = BoundaryConditions(segments=[
    BCSegment(name="inlet", bc_type=BCType.DIRICHLET, value=1.0, boundary_id="west"),
    BCSegment(name="outlet", bc_type=BCType.NEUMANN, value=0.0, boundary_id="east"),
])

# Geometry maps IDs to indices
grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
grid.register_boundary_id("west", "x_min")  # Map "west" → x_min
grid.register_boundary_id("east", "x_max")  # Map "east" → x_max

problem = MFGProblem(geometry=grid, T=1.0, Nt=10)
solver = HJBFDMSolver(problem)
U = solver.solve()  # Still uses FixedBoundaryEnforcer
```

### Future API - Free Boundary (v0.19.0+)

```python
from mfgarchon.geometry.boundary.enforcement import FreeBoundaryEnforcer
from mfgarchon.geometry.boundary.frontier import LevelSetTracker

# Define BC on free boundary
bc = BoundaryConditions(segments=[
    BCSegment(name="free_surface", bc_type=BCType.DIRICHLET, value=0.0, boundary_id="frontier"),
])

# Free boundary enforcer with level set tracking
frontier_tracker = LevelSetTracker(initial_location=0.5)
bc_enforcer = FreeBoundaryEnforcer(frontier_tracker)

grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
problem = MFGProblem(geometry=grid, T=1.0, Nt=10)
solver = HJBFDMSolver(problem)

# Inject free boundary enforcer
U = solver.solve(bc_enforcer=bc_enforcer)

# Query final frontier location
final_frontier = frontier_tracker.get_location(t=1.0)
```

### Future API - Mesh-Based (v0.18.0+)

```python
from mfgarchon.geometry import TriangularMesh
from mfgarchon.geometry.boundary.enforcement import MeshBoundaryEnforcer

# Load mesh with boundary markers
mesh = TriangularMesh.from_gmsh("domain.msh")
# Mesh has boundary markers: {"inlet": [1,2,3], "wall": [4,5,6,7], "outlet": [8,9]}

bc = BoundaryConditions(segments=[
    BCSegment(name="inlet", bc_type=BCType.DIRICHLET, value=1.0, boundary_id="inlet"),
    BCSegment(name="wall", bc_type=BCType.NEUMANN, value=0.0, boundary_id="wall"),
])

bc_enforcer = MeshBoundaryEnforcer()

problem = MFGProblem(geometry=mesh, T=1.0, Nt=10)
solver = HJBDGMSolver(problem)  # DGM solver for triangular mesh
U = solver.solve(bc_enforcer=bc_enforcer)
```

---

## Testing Strategy

### Unit Tests

```python
def test_fixed_boundary_enforcer_dirichlet():
    """Test Dirichlet BC enforcement on fixed boundaries."""
    enforcer = FixedBoundaryEnforcer()

    U = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    bc = BCSegment(name="left", bc_type=BCType.DIRICHLET, value=5.0, boundary="x_min")
    grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[5])

    enforcer.apply(U, bc, grid, time=0.0, dx=0.25)

    assert U[0] == 5.0, "Left boundary should be set to 5.0"
    assert U[1:].sum() == 6.0, "Interior should be unchanged"


def test_free_boundary_enforcer():
    """Test free boundary enforcement with moving frontier."""
    frontier_tracker = MockFrontierTracker(initial_location=2)
    enforcer = FreeBoundaryEnforcer(frontier_tracker)

    U = np.zeros(10)
    bc = BCSegment(name="frontier", bc_type=BCType.DIRICHLET, value=1.0, boundary_id="free")
    grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[10])

    enforcer.apply(U, bc, grid, time=0.0)
    assert U[2] == 1.0, "BC should be enforced at frontier location (index 2)"

    # Frontier moves to index 5
    frontier_tracker.set_location(5)
    enforcer.apply(U, bc, grid, time=0.5)
    assert U[5] == 1.0, "BC should be enforced at new frontier location"
```

### Integration Tests

Test scripts in `scripts/`:
- `validate_issue_542_fix.py`: Fixed boundary validation (current)
- `validate_free_boundary_enforcement.py`: Free boundary validation (v0.19.0+)
- `validate_mesh_boundary_enforcement.py`: Mesh-based validation (v0.18.0+)

---

## Performance Considerations

### Overhead

**Additional cost per time step**:
1. Boundary index lookup: O(1) for fixed, O(N) for free boundary
2. BC enforcement: O(B) where B = number of boundary points

**Total**: Negligible compared to Newton iteration (O(N²) for FDM, O(N³) for direct solve)

### Optimization

For performance-critical applications:
- Cache boundary indices for fixed boundaries
- Use JIT compilation for enforcement loops
- Vectorize multi-segment enforcement

---

## Open Questions

1. **Should BC enforcement be before or after Newton iteration?**
   - Current: After (allows Newton to converge freely, then enforce)
   - Alternative: During (enforce at each Newton step)
   - **Decision**: After is correct - BC is part of the problem, not the solver

2. **How to handle time-dependent BC values?**
   - Current: `value` can be `float` or `Callable[[float], float]`
   - **Decision**: Keep callable support, evaluate at current_time

3. **Should Neumann BC modify boundary value or ghost cell?**
   - Current: Modifies boundary value to satisfy gradient
   - Alternative: Use ghost cell (already done in Laplacian)
   - **Decision**: Current approach is correct - ghost cells are for derivatives, boundary values must be set explicitly

4. **How to handle mixed Dirichlet-Neumann at same boundary?**
   - Example: Dirichlet on part of x_min, Neumann on another part
   - **Solution**: Use multiple BCSegments with refined boundary_id specification

---

## References

- Issue #542: FDM BC handling bug (fixed in v0.16.16)
- Issue #549: BC generalization for non-tensor-product geometries
- Issue #625: Adjoint-consistent BC (BCValueProvider — the first resolver)
- PR #548: BC-aware Laplacian implementation
- `CONDITIONS_VS_CONSTRAINTS_ARCHITECTURE.md`: Conditions vs variational inequalities
- `BC_SOLVER_INTEGRATION_DESIGN.md`: Paradigm-specific BC helpers
- Achdou & Lauriere (2020): MFG Applications, FP no-flux BC derivation

---

**Last Updated**: 2026-03-28
**Status**: Design document, updated with 4-layer resolution architecture
**Authors**: Claude Sonnet 4.5 (original v0.16.16), updated v0.17.16 with resolution layer
