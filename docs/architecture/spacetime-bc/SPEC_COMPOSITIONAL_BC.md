# [SPEC] Compositional Boundary Condition Framework

**Document ID**: MFG-SPEC-BC-0.2
**Status**: DRAFT PROPOSAL
**Date**: 2026-01-30
**Theme**: Mathematical Rigor (GKS & Lopatinskii-Shapiro)

---

## Part I: The Compositional Framework

### 1. Core Insight: BC as Operator Modifier

A boundary condition is not an object — it is a **modification** to the
discrete operator. A complete BC is defined by 4 orthogonal axes.

> **Implementation note**: MFG_PDE already implements 3 of these 4 axes in
> `BCSegment`. The 4th axis (Enforcement) is correctly handled as a solver-side
> concern via the applicator hierarchy. See `CURRENT_STATE_ANALYSIS.md` for mapping.

---

### 2. The 4 Atomic Traits of BCs

#### 2.1 Region (where?)

Defines which part of the geometry boundary the BC applies to.

| Trait | Description | MFG_PDE Mapping |
|:------|:------------|:----------------|
| `GlobalBoundary` | Entire boundary | `BCSegment(boundary=None)` (no restriction) |
| `TaggedRegion(ID)` | Marked boundary segment | `BCSegment(boundary="left")`, `.region_name` |
| `ImplicitInterface` | SDF zero level-set | `BCSegment(sdf_region=...)` |

**Current implementation (5 matching modes)**: boundary name, axis ranges,
SDF region, normal direction, marked region name. Exceeds the spec.

#### 2.2 Mathematical Type (what equation?)

| Trait | Equation | MFG_PDE Mapping |
|:------|:---------|:----------------|
| `Dirichlet` | $u = g$ | `BCType.DIRICHLET` |
| `Neumann` | $\partial u / \partial n = g$ | `BCType.NEUMANN` |
| `Robin` | $\alpha u + \beta \partial u / \partial n = g$ | `BCType.ROBIN` |
| `Cauchy` | Both value and gradient | Not implemented (needed for high-order) |

**Additional types in MFG_PDE**: `PERIODIC`, `REFLECTING`, `NO_FLUX`,
`EXTRAPOLATION_LINEAR`, `EXTRAPOLATION_QUADRATIC`.

#### 2.3 Value Source (equals what?)

| Trait | Description | Optimization | MFG_PDE Mapping |
|:------|:------------|:-------------|:----------------|
| `Zero` | $g = 0$ (homogeneous) | Eliminate additions | `value=0.0` |
| `Constant(c)` | $g = c$ | Scalar broadcast | `value=c` |
| `Functional(func)` | $g = f(x, t)$ | JIT-compilable | `value=callable` |
| `DataField(array)` | $g$ from discrete data | Direct indexing | `value=np.ndarray` |

**MFG_PDE extension**: `BCValueProvider` protocol for state-dependent values
(e.g., `AdjointConsistentProvider` computes $g = -\sigma^2/2 \cdot \partial \ln(m)/\partial n$
from current density). More powerful than static `DataField`.

#### 2.4 Enforcement Method (how to implement?)

| Trait | Mechanism | Suitable For |
|:------|:----------|:-------------|
| `Strong` | Modify matrix rows or ghost cells | FDM, FVM |
| `Weak` | Variational boundary integral / Nitsche | FEM, IGA |
| `GhostFluid` | Interpolation across SDF interface | Level-Set methods |

> **Design decision**: Enforcement is a **solver concern**, not a **problem
> specification concern**. A user declares "Dirichlet u=0"; the solver decides
> ghost cells (FDM) vs penalty (FEM). This is implemented via the applicator
> hierarchy: `FDMApplicator` (Strong), `FEMApplicator` (Weak — partial),
> `ImplicitApplicator` (projection-based). See README.md Decision 1.

---

### 3. The Generative Matrix: Combinations

| Physics Case | Region | MathType | Value | Enforcement | Notes |
|:-------------|:-------|:---------|:------|:------------|:------|
| Adiabatic wall | `Tagged("Wall")` | Neumann | Zero | Weak (FEM) / Strong (FVM) | Natural BC |
| Pipe inlet | `Tagged("Inlet")` | Dirichlet | Functional (parabolic) | Strong | Velocity profile |
| Immersed body | `ImplicitInterface` | Dirichlet | Zero (no-slip) | GhostFluid | Level-set required |
| Far field | `GlobalBoundary` | Robin | Constant | Strong | Radiation BC |
| MFG reflecting | `Tagged("left")` | Robin | Provider (adjoint) | Strong | Issue #625 |

---

### 4. Geometry-BC Interaction Protocols

#### Protocol A: Explicit Mesh

- **Geometry side**: `get_boundary_nodes(tag_id)` via `FacetMarkers`.
- **BC side**: Solver loops over node indices, applies Strong or Weak.
- **MFG_PDE**: `GeometryProtocol.get_boundary_indices()`, `get_boundary_regions()`.

#### Protocol B: Structured Grid

- **Geometry side**: `TensorProductGrid` — no explicit boundary node list.
- **BC side**: JIT-compiled kernel loops on boundary slices
  (`field[0, :, :]`, `field[-1, :, :]`, etc.).
- **Ghost Cell mode**: Geometry allocates halo layer; BC fills halo data.
- **MFG_PDE**: `FDMApplicator.apply_2d()` implements exactly this pattern.

#### Protocol C: Level Set (Implicit Boundary)

- **Geometry side**: Provides SDF $\phi(x)$.
- **BC side**: Solver checks `sign(phi)` changes; triggers GhostFluid or penalty.
- **MFG_PDE**: `ImplicitApplicator` uses SDF but projection-based, not ghost fluid.

---

### 5. Implementation Prototype

```python
@dataclass(frozen=True)
class BCSpec:
    """Compositional BC specification. Maps to existing BCSegment."""
    region: RegionTrait          # e.g., Tagged("Inlet")
    math_type: MathTypeTrait     # e.g., Dirichlet
    value_source: ValueTrait     # e.g., Functional
    enforcement: EnforcementTrait  # Solver-side (informational only)

def apply_bcs(domain, field, bcs: list[BoundaryCondition]):
    """Dispatcher: selects strategy based on domain traits and enforcement."""

    if isinstance(domain, CartesianGrid) and all(bc.enforcement == Strong for bc in bcs):
        return fill_ghost_cells(field, bcs)

    elif isinstance(domain, UnstructuredMesh) and all(bc.enforcement == Weak for bc in bcs):
        return assemble_boundary_forms(domain, field, bcs)

    # ... other combinations
```

> **Implementation note**: This dispatcher pattern is already implemented via
> the applicator hierarchy. `FDMApplicator`, `MeshfreeApplicator`, etc. are
> selected by the solver based on geometry type.

---

## Part II: Stability & Well-posedness (GKS & Lopatinskii-Shapiro)

### 6. Motivation

Naive BC implementations can cause:
- **Ill-posedness**: Violating the number of incoming characteristics
  (Lopatinskii-Shapiro condition).
- **Numerical instability**: Exponentially growing boundary modes
  (GKS stability violation).

> **Applicability note**: This theory is relevant for **first-order MFG**
> ($\sigma = 0$) where HJB is Hamilton-Jacobi (hyperbolic) and FP is a
> transport equation (hyperbolic conservation law). For **second-order MFG**
> ($\sigma > 0$), both equations are parabolic and well-posedness follows
> from standard energy estimates.

### 7. Lopatinskii-Shapiro Validation Layer

For hyperbolic systems $\partial_t u + A \partial_x u = 0$, the number of
BCs at a boundary must equal the number of incoming characteristics.

```python
class WellPosednessValidator:
    @staticmethod
    def validate(pde: PDESystem, bc_config: BCConfiguration, state_vector):
        eigenvalues = pde.compute_eigenvalues(state_vector, normal_vector)
        num_incoming = sum(1 for e in eigenvalues if e < 0)  # Outward normal
        num_constraints = bc_config.count_dofs()

        if num_incoming != num_constraints:
            raise IllPosedError(
                f"Lopatinskii-Shapiro violation: PDE expects {num_incoming} BCs, "
                f"but {num_constraints} provided."
            )
```

**When this matters for MFG**:
- First-order HJB ($\sigma = 0$): Characteristics propagate along optimal trajectories.
  BC count must match incoming characteristic count at each boundary.
- Vanishing viscosity limit ($\sigma \to 0$): Boundary layers form; viscous BCs may
  become over-determined as diffusion vanishes.
- Inviscid HJB with shocks: Entropy conditions needed (Lax-Oleinik, viscosity solution).

### 8. GKS Stability via SBP-SAT

Correct-by-construction approach using **Summation-By-Parts** operators
with **Simultaneous Approximation Term** enforcement.

#### 8.1 Enforcement Method Expansion

- **Strong_Injection**: Direct value overwrite. Fast but GKS-risky for hyperbolic.
- **Weak_SAT**: Penalty source term. Requires SBP property: $D = H^{-1}(Q + B)$.
  Guarantees energy decay if penalty parameter $\tau$ chosen correctly.

#### 8.2 SBP-SAT Solver Factory

```python
def create_stable_solver(spec):
    sbp_op = SBP_Operators.get(order=4, type='Classic')
    bc_enforcement = SAT_Penalty(parameter_heuristic='Eigenvalue')
    return Solver(operator=sbp_op, bc_strategy=bc_enforcement)
```

### 9. Augmented BC Periodic Table

| Physics Case | Math Type | Enforcement | L-S Check | GKS Strategy |
|:-------------|:----------|:------------|:----------|:-------------|
| Subsonic inflow | Characteristic | Weak_SAT | **Required** (2/3 chars) | SBP-SAT |
| Supersonic outflow | None (extrapolation) | N/A | **Required** (0 BCs needed) | SBP upwind |
| Viscous wall | NoSlip (Dirichlet) | Strong_Ghost | Optional (parabolic) | Tuning needed |
| Inviscid wall | Slip (tangency) | Weak_SAT | **Required** | SBP-SAT |
| **1st-order HJB** | Characteristic | Upwind | **Required** | Godunov/WENO |
| **2nd-order HJB** | Neumann/Robin | Strong_Ghost | Optional | Standard FDM |

---

### 10. Completeness Check

1. **Orthogonality**: BC definition (math type) decoupled from geometry implementation
   (grid vs mesh). Same Dirichlet works on structured grid (ghost cells) and
   unstructured mesh (matrix row modification). ✅
2. **Performance**: Zero values enable compile-time elimination. Structured geometry
   BCs use slice operations, not expensive `where`/`gather`. ✅
3. **Extensibility**: New physics (e.g., PEC boundary for Maxwell) = new MathType only.
   Region and Enforcement unchanged. ✅

---

**Last Updated**: 2026-02-05
