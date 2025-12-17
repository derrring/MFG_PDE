# Boundary Condition Architecture Analysis

**Status**: Technical Analysis
**Date**: 2024-12
**Related**: Issue #486 (BC Unification)

---

## Executive Summary

This document analyzes the boundary condition (BC) architecture in MFG_PDE, synthesizing lessons learned from implementing FP solver BC integration. The key insight is that **boundary conditions involve three distinct concerns that are often conflated**:

1. **Topology**: How space connects (periodic vs bounded)
2. **Discretization**: Where values are stored (cell-centered vs vertex-centered)
3. **Physics**: What values are prescribed (Dirichlet, Neumann, Robin, etc.)

The current architecture handles these reasonably well through the applicator hierarchy, but lacks explicit conceptual separation. This document proposes a mental model and identifies edge cases for future development.

---

## 1. The Four-Tier Constraint Taxonomy

Before diving into implementation, we need a classification that transcends specific equations. The traditional Dirichlet/Neumann/Robin taxonomy is **mathematical**, but a **physical** taxonomy is more useful for architecture:

### Tier 1: State Constraints — "Lock the value"
Direct constraints on the primary variable.

| Sub-Type | Physical Meaning | Examples |
|----------|-----------------|----------|
| Fixed State | Constant value | Ground voltage, fixed temperature |
| Profile/Inlet | Prescribed function | Inlet velocity u(x,t), time-varying control |
| No-Slip Wall | Vector field locked | Viscous wall (u=0) |

### Tier 2: Gradient Constraints — "Lock the shape"
Constraints on derivatives, expressing symmetry or smoothness.

| Sub-Type | Physical Meaning | Examples |
|----------|-----------------|----------|
| Symmetry | Mirror plane | Slip wall, axis-symmetry |
| Clamped | Angle locked | Fixed beam end |
| Constant Slope | Prescribed gradient | Specified shear |

### Tier 3: Flux/Conservation Constraints — "Lock the flow"
Constraints on conserved quantities. **Most context-dependent** — requires knowing the physical flux J(u, ∇u).

| Sub-Type | Physical Meaning | Examples |
|----------|-----------------|----------|
| No-Flux | Impermeable wall | FP reflecting, adiabatic |
| Specified Flux | Prescribed inflow | Heat injection, current source |
| Stress Balance | Force equilibrium | Free surface, pressure BC |

### Tier 4: Artificial/Open Constraints — "Fake the infinity"
Truncation of unbounded domains. Goal: let information flow out, don't reflect back.

| Sub-Type | Physical Meaning | Examples |
|----------|-----------------|----------|
| Outflow | Free exit | Developed flow, HJB far-field |
| Absorbing | Wave sink | Sommerfeld, PML |
| Far-Field | Asymptotic match | u → u∞ |

### The Two-Level Mapping Architecture

This taxonomy suggests a **two-level mapping** design:

```
User Config (Physical Intent)     Solver Resolution (Mathematical Implementation)
─────────────────────────────     ──────────────────────────────────────────────
bc_type = "wall"            ──►   HJBSolver: NeumannBC(0)      [gradient lock]
                                  FPSolver:  ZeroFluxRobinBC() [flux lock]

bc_type = "outflow"         ──►   HJBSolver: LinearExtrapolationBC()
                                  FPSolver:  DirichletBC(0)    [density→0]
```

The solver knows how to interpret physical intent based on its equation physics.

### Physical Intent → Mathematical Implementation Table

| Physical Intent | HJB (Value Function) | FP (Density) | Particles |
|-----------------|---------------------|--------------|-----------|
| **wall** | Neumann ∂V/∂n=0 | Robin (zero total flux) | Reflect |
| **exit** | Dirichlet V=0 | Dirichlet ρ=0 | Absorb |
| **symmetry** | Neumann ∂V/∂n=0 | Neumann ∂ρ/∂n=0 | Reflect |
| **outflow** | Linear extrapolation | Dirichlet ρ→0 | Pass through |
| **periodic** | Wrap (topology) | Wrap (topology) | Wrap |
| **inlet** | Dirichlet V=g(x,t) | Dirichlet ρ=g(x,t) | Inject |

This table shows why a single `BCType` enum is insufficient — the same physical word maps to different mathematics.

---

## 2. The Two-Layer Processing Model

### Layer 1: Topology — "Where does data come from?"

Topology determines **connectivity** — whether space wraps around or has edges.

| Topology | Characteristic | Action | Ghost Source |
|----------|---------------|--------|--------------|
| **Periodic** | Space wraps (torus) | Wrap indexing | Copy from opposite boundary |
| **Bounded** | Space has edges | Allocate ghost memory | Compute from physics |

**Key constraint**: Periodic topology requires BOTH boundaries of an axis to be periodic. You cannot have periodic on left and Dirichlet on right — that's topologically incoherent.

**Implementation note**: For periodic topology, no physics calculation is needed — it's pure data movement. This is why particle solvers only need topology, not physics.

### Layer 2: Physics — "What value goes in the ghost cell?"

Physics determines **numerical values** at boundaries. Only activated when topology is **Bounded**.

| Category | Common Names | Mathematical Form | Ghost Calculation |
|----------|-------------|-------------------|-------------------|
| **Hard Value** | Dirichlet, Absorbing, Exit | u = g | `u_ghost = 2g - u_inner` (cell-centered) |
| **Slope** | Neumann, HJB-Reflective, Symmetric | ∂u/∂n = g | `u_ghost = u_inner ± dx·g` |
| **Mixed** | Robin, FP-Reflective, FP-No-Flux | αu + β∂u/∂n = g | Solve algebraic equation |

**Critical insight**: The same physical name (e.g., "no-flux") requires different discrete formulas depending on the equation being solved:

- **Heat equation no-flux**: Neumann (∂T/∂n = 0)
- **Fokker-Planck no-flux**: Robin (zeroes total flux J = vρ - D∇ρ)
- **HJB reflective**: Neumann (∂V/∂n = 0)

---

## 2. Current Architecture Mapping

### What We Have

```
mfg_pde/geometry/boundary/
├── types.py              # BCType enum (mixes topology + physics)
├── conditions.py         # BoundaryConditions class
├── applicator_base.py    # Base classes + ghost cell helpers
├── applicator_fdm.py     # FDM ghost cell application
├── applicator_fem.py     # FEM matrix modification
├── applicator_meshfree.py # Particle reflection
└── applicator_graph.py   # Graph node constraints
```

### BCType Enum Analysis

```python
class BCType(Enum):
    # TOPOLOGY (affects connectivity)
    PERIODIC = "periodic"      # Wraps around

    # PHYSICS (affects values)
    DIRICHLET = "dirichlet"    # u = g
    NEUMANN = "neumann"        # ∂u/∂n = g
    ROBIN = "robin"            # αu + β∂u/∂n = g
    NO_FLUX = "no_flux"        # J·n = 0 (total flux)
    REFLECTING = "reflecting"  # Particle reflection
```

**Observation**: `PERIODIC` is fundamentally different from the others — it's about topology, not physics. The others are all bounded-topology physics types.

### Applicator Hierarchy

The applicator pattern correctly separates concerns by discretization method:

| Applicator | Discretization | Topology Handling | Physics Handling |
|------------|---------------|-------------------|------------------|
| `FDMApplicator` | Structured grid | Periodic wrap / ghost allocation | Ghost cell formulas |
| `FEMApplicator` | Unstructured mesh | DOF identification | Matrix row modification |
| `MeshfreeApplicator` | Particles | Wrap / reflect | Position clamping |
| `GraphApplicator` | Graph nodes | Node identification | Value constraints |

### Physics-Aware Ghost Functions

The `applicator_base.py` correctly provides equation-specific ghost formulas:

```python
ghost_cell_dirichlet()      # Standard Dirichlet
ghost_cell_neumann()        # Standard Neumann
ghost_cell_robin()          # General Robin
ghost_cell_fp_no_flux()     # FP-specific: zeroes total flux
```

This is the right approach — encoding equation-specific physics in dedicated functions.

---

## 3. Edge Cases and Hidden Complexity

### 3.1 Corner Problem (2D/3D)

**Issue**: In 2D+, ghost cells exist at corners, not just faces. Corner ghosts depend on both x and y boundary rules.

**Current handling**: `np.pad` handles this implicitly via dimensional ordering.

```
     y_ghost
        │
   ┌────┼────┬────────┐
   │ ?? │ Ny │        │  <- Corner ghost: filled by Y-pass
   ├────┼────┼────────┤      using X-ghost as "interior"
   │ Gx │ G  │        │
x_ghost─┼────┼────────┤
   │    │    │        │
   └────┴────┴────────┘
```

**Commutativity Warning**: For mixed BCs, the application order affects corner values:
- Example: X-boundary has Dirichlet(u=1), Y-boundary has Dirichlet(u=0)
- X→Y order: corner first gets value from X-rule, then Y-rule may overwrite
- Y→X order: opposite result

For most physical problems (continuous fields), corner values should be continuous and order doesn't matter. However, if BCs are discontinuous at corners (singularity), order becomes critical.

**Standard**: Implementation uses **X→Y→Z order**. Document this explicitly in code comments.

### 3.2 Grid Alignment (Vertex vs Cell-Centered)

**Issue**: The same physical BC has different discrete formulas for different grid types.

| BC Type | Vertex-Centered | Cell-Centered |
|---------|-----------------|---------------|
| Dirichlet u=g | `u[0] = g` | `u_ghost = 2g - u[0]` |
| Neumann ∂u/∂n=0 | `u_ghost = u[1]` | `u_ghost = u[0]` |

**Conceptual clarification**: Discretization (Vertex vs Cell) is a **Grid property**, not a BC property. The BC calculator *consumes* this information from the grid context. When writing a new solver, the grid type must be passed to the BC applicator.

**Current handling**: `GridType` enum exists but isn't always used consistently.

**Recommendation**: All ghost cell functions should accept `grid_type` parameter. The `ghost_cell_fp_no_flux()` already does this correctly. Solvers must propagate grid context to BC applicators.

### 3.3 Time-Dependent BCs

**Issue**: Current interface `compute_ghost(u_inner, dx)` doesn't support time-varying BCs like `u(boundary, t) = g(t)`.

**Current handling**: Not explicitly supported.

**Recommendation**: For future extension, consider:
```python
def compute_ghost(u_inner, dx, *, context: dict = None):
    # context may contain: t, iteration, dt, etc.
```

### 3.4 Implicit Solver Matrix Assembly

**Issue**: Implicit schemes (Au^{n+1} = b) don't use ghost cells — they modify matrix rows.

| BC Type | Explicit (Ghost) | Implicit (Matrix) |
|---------|-----------------|-------------------|
| Dirichlet | `u_ghost = 2g - u_inner` | Set row to `[0,...,1,...,0]`, b to `g` |
| Neumann | `u_ghost = u_inner` | Modify diagonal and off-diagonal |

**Two approaches for implicit BCs**:
1. **Matrix Reduction**: Remove boundary DOFs from unknown vector (matrix shrinks)
2. **Identity Row (Penalty)**: Keep boundary rows, set diagonal=1, RHS=g (matrix same size)

**Recommendation**: Use **Identity Row approach** (option 2). It maintains index consistency between explicit and implicit solvers, reducing mapping complexity. The slight computational overhead is negligible.

**Current handling**: `FEMApplicator` handles matrix modification. FDM implicit would need similar.

**Interface**: If implicit FDM solvers are needed, add `apply_to_matrix(A, b)` interface to BC calculators.

### 3.5 Unbounded Domains

**Issue**: Many MFG problems are defined on R^d but computed on truncated domains.

**Classification**: Unbounded = Bounded topology + special physics (extrapolation)

| Equation | Far-field Behavior | Strategy |
|----------|-------------------|----------|
| Fokker-Planck | ρ → 0 | Dirichlet(0) at large L |
| HJB value | V → ∞ (often linear) | Linear extrapolation |
| Waves | Outgoing | Absorbing BC / PML |

**Linear extrapolation formula**:
```python
u_ghost = 2*u[0] - u[1]  # Extrapolate from two interior points
```

**Mathematical interpretation**: This is equivalent to the **Zero Second Derivative Condition** (d²u/dx² = 0 at boundary). The function is assumed to continue linearly beyond the grid.

**Caveat for quadratic growth**: For HJB problems where the value function has quadratic growth (e.g., LQG control), linear extrapolation forces the second derivative to zero at the boundary, creating an artificial "kink". For such cases, **quadratic extrapolation** (d³u/dx³ = 0, using three interior points) may be more appropriate:
```python
u_ghost = 3*u[0] - 3*u[1] + u[2]  # Quadratic extrapolation
```

**Current handling**: Not explicitly implemented.

**Recommendation**:
- Short-term: Add `LinearExtrapolationBC` for general HJB unbounded domains
- Long-term: Add `QuadraticExtrapolationBC` for LQG-type problems with quadratic value functions

### 3.6 Mixed BCs Per Dimension

**Issue**: Different axes may have different BCs (e.g., periodic in x, Dirichlet in y).

**Current handling**:
- `BoundaryConditions` supports mixed BCs via segments
- `get_bc_type_at_boundary()` enables per-boundary queries
- Particle solver uses `_get_topology_per_dimension()` for per-axis topology

**Processing order**: For each axis, always check topology FIRST, then physics:
1. Is this axis periodic? (both min and max must be periodic)
2. If bounded, what physics applies to min boundary? To max boundary?

This ensures topology-level decisions (wrap vs allocate ghost) are made before physics-level calculations.

**Status**: Well-handled after Issue #486 Phase 2.

---

## 4. Recommended Mental Model

### For Developers

When implementing BC handling, ask these questions in order:

1. **Topology**: Is this axis periodic or bounded?
   - Periodic → Copy from opposite boundary (no physics needed, done)
   - Bounded → Proceed to step 2

2. **Grid Context**: What is the grid type? (This is a Grid property, not BC property)
   - Vertex-centered → Boundary point is at grid node
   - Cell-centered → Boundary is at face between ghost and first interior cell
   - **Action**: Pass grid type to BC calculator

3. **Physics**: What physical constraint applies?
   - Map physical name to mathematical form (Dirichlet/Neumann/Robin)
   - Consider equation-specific variants (FP no-flux ≠ HJB reflective)
   - **Action**: Select appropriate ghost cell formula

4. **Special Context**: Any additional considerations?
   - Time dependence → Pass time to BC calculator
   - Corners in 2D+ → Ensure X→Y→Z order
   - Matrix assembly for implicit → Use `apply_to_matrix(A, b)` instead

### For Users

When specifying BCs in config:

```python
# Uniform BC (same everywhere)
bc = periodic_bc(dimension=2)  # Topology: periodic
bc = no_flux_bc(dimension=2)   # Topology: bounded, Physics: no-flux

# Mixed BC (different per boundary)
bc = mixed_bc([
    BCSegment(bc_type=BCType.PERIODIC, boundary="x_min"),
    BCSegment(bc_type=BCType.PERIODIC, boundary="x_max"),
    BCSegment(bc_type=BCType.DIRICHLET, boundary="y_min", value=0),
    BCSegment(bc_type=BCType.NEUMANN, boundary="y_max", value=0),
], dimension=2)
```

---

## 5. Architecture Recommendations

### Short-term (No Breaking Changes)

1. **Documentation**: Add taxonomy comments to `BCType` enum explaining topology vs physics distinction

2. **Validation**: Add warning when periodic BC is only on one side of an axis

3. **Consistency**: Ensure all ghost functions accept `grid_type` parameter

### Medium-term (Minor API Extension)

4. **Unbounded support**: Add `LinearExtrapolationBC` for HJB on truncated domains

5. **Context passing**: Extend ghost function signatures to support time-dependent BCs

6. **Corner validation**: Add debug mode that validates corner ghost values in 2D+

### Long-term (Architectural Evolution)

7. **Two-Level Mapping**: Introduce physical intent layer (wall/exit/outflow) that solvers interpret:
   ```python
   # User specifies physical intent
   bc = BoundaryIntent(type="wall")

   # Solver resolves to mathematical implementation
   class HJBSolver:
       def resolve_bc(self, intent):
           if intent.type == "wall":
               return NeumannBC(0)  # Gradient lock for HJB

   class FPSolver:
       def resolve_bc(self, intent):
           if intent.type == "wall":
               return ZeroFluxRobinBC(self.drift, self.sigma)  # Flux lock for FP
   ```

8. **Matrix interface**: Add `apply_to_matrix(A, b)` to `BoundaryCalculator` protocol (if implicit FDM needed)

---

## 6. Summary Table

| Aspect | Current Status | Risk Level | Recommendation |
|--------|---------------|------------|----------------|
| Topology/Physics separation | Implicit in code | Low | Document explicitly |
| Grid alignment (V/C) | Partial support | **Medium** | Enforce in all ghost functions |
| Corner handling | Auto via np.pad | Low | Validate in debug mode |
| Time-dependent BCs | Not supported | Low | Extend when needed |
| Unbounded domains | Not supported | **Medium** | Add extrapolation BC |
| Mixed BCs | Fully supported | Low | Done in #486 |
| Implicit matrix assembly | FEM only | Low | Add if FDM implicit needed |

---

## References

1. LeVeque (2007). *Finite Difference Methods for ODEs and PDEs*. SIAM.
2. Risken (1996). *The Fokker-Planck Equation*. Springer.
3. Achdou & Capuzzo-Dolcetta (2010). Mean Field Games: Numerical Methods. *SIAM J. Numer. Anal.*
4. Patankar (1980). *Numerical Heat Transfer and Fluid Flow*. CRC Press.

---

**Document History**:
- 2024-12: Initial analysis based on Issue #486 Phase 2 implementation
