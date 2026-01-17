# Implementation Plans: Geometry & BC Architecture

**Date**: 2026-01-17
**Status**: PLANNING
**Related**: `docs/theory/GEOMETRY_BC_ARCHITECTURE_DESIGN.md` (Theoretical Design)

---

## Executive Summary

This document presents **multiple strategic approaches** for implementing the comprehensive geometry and BC architecture specified in the theoretical design document. Each plan balances risk, effort, and feature completeness differently.

**Current State (v0.17.1 Baseline)**:
- ✅ **Tier 1 BCs**: Production-ready (DNR, Robin, Periodic, Mixed)
- ✅ **TensorProductGrid**: Mature with operator abstraction
- ✅ **ImplicitDomain**: Production-ready SDF-based geometry
- ✅ **GraphGeometry**: Production-ready network MFG
- ⚠️ **Tier 2 BCs**: Not implemented (VIs, obstacle problems)
- ⚠️ **Tier 3 BCs**: Not implemented (free boundaries, Level Set)
- ⚠️ **UnstructuredMesh**: Infrastructure present, needs trait implementation
- ⚠️ **GKS/L-S Validation**: Not implemented

---

## §0: Relationship to Existing Work

**IMPORTANT**: This implementation plan builds upon and coordinates with existing BC infrastructure work.

### Completed Foundation (Issues #493, #527 Phase 1-3, #574)

**What's Already Done** (v0.17.1):

1. **Issue #493** (Geometry owns BC - SSOT Pattern): ✅ **Complete**
   - Geometry stores spatial BC as single source of truth
   - `problem.get_boundary_conditions()` delegates to geometry
   - HJB and FP solvers guaranteed consistent BC

2. **Issue #527** (BC Solver Integration) - **Phase 2-3 Complete**:
   - ✅ Unified BC access: `BaseMFGSolver.get_boundary_conditions()` (5-level resolution)
   - ✅ Paradigm-specific helpers in base classes:
     - `BaseNumericalSolver.apply_boundary_conditions()` (delegates to `dispatch.apply_bc()`)
     - `BaseNeuralSolver.sample_boundary_points()` + `get_boundary_target_values()`
     - `BaseRLSolver.get_environment_boundary_config()`
     - `BaseOptimizationSolver.get_domain_constraints()`
   - ✅ All numerical solvers migrated to unified pattern

3. **Issue #574** (Adjoint-Consistent BC): ✅ **Complete** (v0.17.1)
   - Robin BC framework for state-dependent BCs
   - `create_adjoint_consistent_bc_1d()` creates proper `BoundaryConditions` objects
   - Example of Tier 1 extensibility

**Impact on This Plan**:
- **Plan A Phase 1** builds on completed SSOT pattern and solver integration
- **Operator abstraction** (Phase 1.1-1.2) complements existing `dispatch.apply_bc()`
- **Region registry** (Phase 1.3) extends existing boundary marking infrastructure

### Planned Work with Overlap

**Issue #535** (BC Framework Enhancement): **Partial Overlap**

**From Issue #535 Roadmap**:
- L-S well-posedness validation (theory)
- Boundary matrix construction (FEM)
- Neural BC loss interface (PINN/DGM)

**Overlap with Plan A**:
- **Phase 4.2 (GKS/L-S Validation)**: Implements stability checking mentioned in #535
  - **Coordination**: Use same eigenvalue analysis framework
  - **Scope difference**: Plan A focuses on GKS (discrete stability), #535 on L-S (PDE well-posedness)
  - **Recommendation**: Implement GKS in Phase 4, defer L-S to #535 (requires PDE theory)

- **Neural BC Loss**: Already addressed by Issue #527 Phase 2
  - `BaseNeuralSolver.get_boundary_target_values()` provides interface
  - Plan A does not duplicate this work

**Issue #536** (Particle Absorbing BC): **Minor Overlap**

**From Issue #536**:
- Add proper absorbing BC for particle methods (beyond simple deletion)
- Track absorption statistics (exit times, locations)

**Overlap with Plan A**:
- **No direct conflict**: Plan A focuses on grid-based Tier 2/3 BCs
- **Complementary**: Particle absorbing BC is Tier 1 enhancement, not Tier 2/3

**Recommendation**: Proceed with both independently

### Coordination Strategy

**To Avoid Duplication**:

1. **GKS vs L-S Validation** (Plan A Phase 4.2 vs Issue #535):
   - **Plan A implements**: `check_gks_condition()` for discrete operator stability
   - **Issue #535 implements**: L-S condition checker for PDE well-posedness
   - **Shared infrastructure**: `geometry/boundary/validation/` module
   - **Action**: Create validation module in Phase 4, allow #535 to extend

2. **Neural BC Interface** (Issue #527 vs Issue #535):
   - **Already resolved**: Issue #527 Phase 2 complete
   - **Plan A**: No neural-specific work (focuses on grid-based Tier 2/3)
   - **Action**: None needed

3. **Particle BC** (Issue #536):
   - **Independent**: Tier 1 particle enhancement vs Tier 2/3 grid-based
   - **Action**: Proceed in parallel

### Features Unique to Plan A (No Overlap)

The following Plan A features are **new** and do not overlap with existing issues:

- ✅ **Geometry Trait System** (Phase 1): `SupportsLaplacian`, `SupportsGradient`, etc.
- ✅ **Region Registry** (Phase 1.3): Named spatial regions for mixed BC
- ✅ **Tier 2 BCs** (Phase 2): Variational inequalities, obstacle problems, capacity constraints
- ✅ **Tier 3 BCs** (Phase 3): Level Set evolution, Stefan problems, free boundaries
- ✅ **Nitsche's Method** (Phase 4.1): Weak BC imposition for FEM

### Summary Table

| Feature | Existing Issue | Plan A Phase | Relationship |
|:--------|:---------------|:-------------|:-------------|
| **SSOT Pattern** | #493 (complete) | Foundation | ✅ Builds upon |
| **Unified BC Access** | #527 Phase 2-3 (complete) | Foundation | ✅ Builds upon |
| **Paradigm Helpers** | #527 Phase 2 (complete) | Foundation | ✅ Uses existing |
| **Robin BC Framework** | #574 (complete) | Foundation | ✅ Builds upon |
| **Geometry Traits** | (none) | Phase 1 | ✅ **New** |
| **Region Registry** | (none) | Phase 1.3 | ✅ **New** |
| **Tier 2 BCs (VIs)** | (none) | Phase 2 | ✅ **New** |
| **Tier 3 BCs (Level Set)** | (none) | Phase 3 | ✅ **New** |
| **Nitsche's Method** | (none) | Phase 4.1 | ✅ **New** |
| **GKS Validation** | #535 (planned) | Phase 4.2 | ⚠️ **Overlap** (coordinate) |
| **L-S Validation** | #535 (planned) | (not in plan) | ✅ Defer to #535 |
| **Neural BC Loss** | #527 Phase 2 (complete) | (not in plan) | ✅ Already done |
| **Particle Absorbing** | #536 (open) | (not in plan) | ✅ Independent |

---

## Strategic Options

### Plan A: Conservative Sequential (Recommended)
**Philosophy**: Minimize risk, deliver incrementally, validate thoroughly
**Timeline**: 12-16 weeks
**Risk**: Low
**Best For**: Production stability, research validation requirements

### Plan B: Parallel Tracks
**Philosophy**: Maximize velocity, independent teams
**Timeline**: 8-10 weeks (with 2-3 developers)
**Risk**: Medium
**Best For**: Multi-developer teams, time-critical deliverables

### Plan C: Minimal Viable Extension
**Philosophy**: Deliver only what's needed for immediate research
**Timeline**: 4-6 weeks
**Risk**: Low (limited scope)
**Best For**: Specific research needs (e.g., obstacle problems only)

### Plan D: Aggressive Transformation
**Philosophy**: Complete architecture overhaul, breaking changes acceptable
**Timeline**: 16-20 weeks
**Risk**: High
**Best For**: Major version bump (v1.0.0), long-term vision

---

## Plan A: Conservative Sequential (RECOMMENDED)

### Phase 1: Geometry Trait System (2-3 weeks)

**Objective**: Formalize trait protocols, retrofit existing geometries

#### 1.1 Protocol Definition (3-5 days)
**Files to Create**:
- `mfg_pde/geometry/protocols/operators.py` - Operator protocols
- `mfg_pde/geometry/protocols/topology.py` - Topological trait protocols
- `mfg_pde/geometry/protocols/regions.py` - Region marking protocols

**Key Protocols**:
```python
# mfg_pde/geometry/protocols/operators.py

class SupportsLaplacian(Protocol):
    """Geometry can compute Laplacian operator."""
    def get_laplacian_operator(
        self,
        order: int = 2,
        bc: BoundaryConditions | None = None,
    ) -> LinearOperator:
        """Return discrete Laplacian matrix/operator."""
        ...

class SupportsGradient(Protocol):
    """Geometry can compute gradient operator."""
    def get_gradient_operator(
        self,
        direction: int | None = None,
        order: int = 2,
    ) -> LinearOperator | tuple[LinearOperator, ...]:
        """Return gradient operator(s)."""
        ...

class SupportsDivergence(Protocol):
    """Geometry can compute divergence operator."""
    def get_divergence_operator(self, order: int = 2) -> LinearOperator:
        ...

class SupportsAdvection(Protocol):
    """Geometry can compute advection operator."""
    def get_advection_operator(
        self,
        velocity_field: NDArray,
        scheme: Literal["upwind", "centered", "weno"] = "upwind",
    ) -> LinearOperator:
        ...
```

**Testing**:
- Protocol compliance checks for all existing geometries
- Operator composition tests (Laplacian = div(grad))
- BC integration tests

**Effort**: 3-5 days (protocol design + existing geometry retrofit)

---

#### 1.2 Retrofit Existing Geometries (5-7 days)

**Geometries to Update**:
1. **TensorProductGrid** (already ~80% compliant)
   - Add missing protocols: `SupportsAdvection`, `SupportsInterpolation`
   - Formalize region marking (currently ad-hoc)
   - Add trait introspection methods

2. **ImplicitDomain**
   - Implement `SupportsGradient` (via SDF gradient)
   - Add `SupportsBoundaryQuery` (SDF < ε)
   - Integrate with `RegionRegistry`

3. **GraphGeometry**
   - Implement graph Laplacian operator
   - Add `SupportsTopology` (manifold vs non-manifold)

**Example Retrofit** (TensorProductGrid):
```python
# mfg_pde/geometry/domain/tensor_product_grid.py

class TensorProductGrid(
    BaseGeometry,
    SupportsLaplacian,
    SupportsGradient,
    SupportsDivergence,
    SupportsAdvection,
    SupportsInterpolation,
    SupportsBoundaryNormal,
):
    """Production-ready structured grid with full operator support."""

    def get_laplacian_operator(
        self,
        order: int = 2,
        bc: BoundaryConditions | None = None,
    ) -> LinearOperator:
        """Laplacian via finite differences."""
        # Delegate to FDM backend
        from mfg_pde.backends.fdm import build_laplacian_matrix
        return build_laplacian_matrix(self.grid_shape, self.dx, order, bc)

    def get_advection_operator(
        self,
        velocity_field: NDArray,
        scheme: Literal["upwind", "centered", "weno"] = "upwind",
    ) -> LinearOperator:
        """Build advection operator for given velocity field."""
        from mfg_pde.backends.fdm import build_advection_matrix
        return build_advection_matrix(velocity_field, self.dx, scheme)

    def mark_region(
        self,
        name: str,
        predicate: Callable[[NDArray], NDArray[np.bool_]],
    ) -> None:
        """Register named region for BC/constraint application."""
        mask = predicate(self.grid_points)
        self._region_registry[name] = mask

    def get_supported_traits(self) -> set[type]:
        """Runtime trait introspection."""
        return {
            SupportsLaplacian,
            SupportsGradient,
            SupportsDivergence,
            SupportsAdvection,
            SupportsInterpolation,
            SupportsBoundaryNormal,
        }
```

**Testing**:
- Operator accuracy tests (convergence order verification)
- BC integration tests (all Tier 1 BCs)
- Region marking tests

**Effort**: 5-7 days

---

#### 1.3 Region Registry System (3-4 days)

**Objective**: Unified region marking and query system

**Files to Create**:
- `mfg_pde/geometry/regions/registry.py` - Region storage and query
- `mfg_pde/geometry/regions/predicates.py` - Common region predicates

**Key Components**:
```python
# mfg_pde/geometry/regions/registry.py

@dataclass
class Region:
    """Named spatial region with boolean mask."""
    name: str
    mask: NDArray[np.bool_]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def indices(self) -> NDArray[np.intp]:
        """Get linear indices of region points."""
        return np.flatnonzero(self.mask)

class RegionRegistry:
    """Central registry for spatial regions."""

    def __init__(self):
        self._regions: dict[str, Region] = {}

    def register(
        self,
        name: str,
        mask: NDArray[np.bool_] | None = None,
        predicate: Callable[[NDArray], NDArray[np.bool_]] | None = None,
        grid_points: NDArray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Region:
        """Register a named region."""
        if mask is None and predicate is not None:
            if grid_points is None:
                raise ValueError("Need grid_points when using predicate")
            mask = predicate(grid_points)
        elif mask is None:
            raise ValueError("Need either mask or (predicate, grid_points)")

        region = Region(name, mask, metadata or {})
        self._regions[name] = region
        return region

    def query(self, name: str) -> Region:
        """Get region by name."""
        return self._regions[name]

    def get_boundary_mask(
        self,
        boundary_name: str,  # e.g., "x_min", "x_max", "wall"
    ) -> NDArray[np.bool_]:
        """Get mask for named boundary."""
        return self._regions[boundary_name].mask

    def intersect(self, *names: str) -> NDArray[np.bool_]:
        """Boolean intersection of multiple regions."""
        masks = [self._regions[name].mask for name in names]
        return np.logical_and.reduce(masks)

    def union(self, *names: str) -> NDArray[np.bool_]:
        """Boolean union of multiple regions."""
        masks = [self._regions[name].mask for name in names]
        return np.logical_or.reduce(masks)
```

**Common Predicates**:
```python
# mfg_pde/geometry/regions/predicates.py

def box_region(
    lower: NDArray,
    upper: NDArray,
) -> Callable[[NDArray], NDArray[np.bool_]]:
    """Predicate for axis-aligned box."""
    def predicate(points: NDArray) -> NDArray[np.bool_]:
        return np.all((points >= lower) & (points <= upper), axis=-1)
    return predicate

def sphere_region(
    center: NDArray,
    radius: float,
) -> Callable[[NDArray], NDArray[np.bool_]]:
    """Predicate for spherical region."""
    def predicate(points: NDArray) -> NDArray[np.bool_]:
        return np.linalg.norm(points - center, axis=-1) <= radius
    return predicate

def sdf_region(
    sdf: Callable[[NDArray], NDArray],
    tolerance: float = 0.0,
) -> Callable[[NDArray], NDArray[np.bool_]]:
    """Predicate for SDF-based region (φ <= tol)."""
    def predicate(points: NDArray) -> NDArray[np.bool_]:
        return sdf(points) <= tolerance
    return predicate
```

**Integration with BC Framework**:
```python
# Example: Apply Dirichlet BC to named region
geometry.mark_region("inlet", box_region([0, 0], [0, 1]))
bc_segment = BCSegment(
    name="inlet_bc",
    bc_type=BCType.DIRICHLET,
    value=1.0,
    boundary="inlet",  # References region registry
)
bc = mixed_bc([bc_segment], dimension=2, geometry=geometry)
```

**Testing**:
- Region registration and query
- Boolean operations (intersect, union, complement)
- BC integration via named regions

**Effort**: 3-4 days

---

### Phase 2: Tier 2 BCs - Variational Constraints (3-4 weeks)

**Objective**: Implement obstacle problems and variational inequalities

#### 2.1 Constraint Protocol (1 week)

**Files to Create**:
- `mfg_pde/geometry/boundary/constraints.py` - Constraint base classes
- `mfg_pde/geometry/boundary/constraint_protocol.py` - Protocol definitions

**Key Abstractions**:
```python
# mfg_pde/geometry/boundary/constraint_protocol.py

class ConstraintProtocol(Protocol):
    """Protocol for solution constraints."""

    def project(self, u: NDArray) -> NDArray:
        """Project solution onto constraint set."""
        ...

    def is_feasible(self, u: NDArray, tol: float = 1e-10) -> bool:
        """Check if solution satisfies constraint."""
        ...

    def get_active_set(self, u: NDArray, tol: float = 1e-10) -> NDArray[np.bool_]:
        """Return mask of active constraint points."""
        ...

class ObstacleConstraint:
    """Unilateral constraint: u >= ψ or u <= ψ."""

    def __init__(
        self,
        obstacle: NDArray | Callable,
        constraint_type: Literal["lower", "upper"] = "lower",
        region: str | None = None,
    ):
        self.obstacle = obstacle
        self.constraint_type = constraint_type
        self.region = region

    def project(self, u: NDArray) -> NDArray:
        """Project onto feasible set."""
        psi = self._evaluate_obstacle(u)
        if self.constraint_type == "lower":
            return np.maximum(u, psi)  # u >= ψ
        else:
            return np.minimum(u, psi)  # u <= ψ

    def get_active_set(self, u: NDArray, tol: float = 1e-10) -> NDArray[np.bool_]:
        """Points where constraint is active."""
        psi = self._evaluate_obstacle(u)
        if self.constraint_type == "lower":
            return u - psi <= tol  # u ≈ ψ
        else:
            return psi - u <= tol

class BilateralConstraint:
    """Bilateral constraint: ψ_lower <= u <= ψ_upper."""

    def __init__(
        self,
        lower_obstacle: NDArray | Callable,
        upper_obstacle: NDArray | Callable,
        region: str | None = None,
    ):
        self.lower = ObstacleConstraint(lower_obstacle, "lower", region)
        self.upper = ObstacleConstraint(upper_obstacle, "upper", region)

    def project(self, u: NDArray) -> NDArray:
        """Project onto box constraint."""
        u = self.lower.project(u)
        u = self.upper.project(u)
        return u
```

**Testing**:
- Projection operator tests (idempotency, contraction)
- Active set detection accuracy

**Effort**: 1 week

---

#### 2.2 VI Solver Integration (2 weeks)

**Objective**: Modify existing solvers to support constraints

**Approach**: Penalty-projection methods (least invasive)

**Solvers to Modify**:
1. **HJB FDM Solver** (`mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`)
2. **HJB Semi-Lagrangian** (`mfg_pde/alg/numerical/hjb_solvers/hjb_sl.py`)
3. **Poisson Solver** (for testing)

**Implementation Pattern** (Penalty Method):
```python
# mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py

def solve_hjb_system(
    self,
    ...,
    constraints: list[ConstraintProtocol] | None = None,
) -> NDArray:
    """Solve HJB with optional constraints."""

    # Standard solver iteration
    for iteration in range(max_iterations):
        # Compute unconstrained update
        U_unconstrained = self._compute_hjb_update(U_prev, ...)

        # Apply constraints if present
        if constraints:
            U_new = U_unconstrained.copy()
            for constraint in constraints:
                U_new = constraint.project(U_new)
        else:
            U_new = U_unconstrained

        # Check convergence
        if self._has_converged(U_new, U_prev):
            break

        U_prev = U_new

    return U_new
```

**Advanced**: Projected Newton Method (higher performance, more complex):
```python
def solve_hjb_with_obstacle_projected_newton(
    problem: MFGProblem,
    obstacle: NDArray,
    max_iterations: int = 100,
) -> NDArray:
    """Solve HJB VI using projected Newton method."""

    U = problem.U_terminal.copy()

    for iteration in range(max_iterations):
        # Compute active and inactive sets
        active = (U - obstacle) <= 1e-10
        inactive = ~active

        # Newton step on inactive set only
        if np.any(inactive):
            residual = compute_hjb_residual(U)
            jacobian = compute_hjb_jacobian(U)

            # Solve only for inactive points
            delta_U = np.zeros_like(U)
            delta_U[inactive] = spsolve(
                jacobian[np.ix_(inactive, inactive)],
                -residual[inactive],
            )

            # Line search with projection
            alpha = 1.0
            U_new = project_onto_obstacle(U + alpha * delta_U, obstacle)

            # Backtracking if needed
            while not is_descent_direction(U_new, U):
                alpha *= 0.5
                U_new = project_onto_obstacle(U + alpha * delta_U, obstacle)
                if alpha < 1e-10:
                    break
        else:
            U_new = U

        if np.linalg.norm(U_new - U) < tolerance:
            break

        U = U_new

    return U
```

**Testing Strategy**:
- **Unit Tests**: Projection correctness, active set detection
- **Integration Tests**: 1D obstacle problem with known solution
- **Validation**: Compare to analytical solutions (where available)

**Test Case 1**: 1D Obstacle Problem (Poisson with obstacle)
```python
# Solve: -u'' = f, u >= ψ in (0,1), u(0)=u(1)=0
# Known solution for f=1, ψ(x) = 0.1*sin(πx)

def test_1d_obstacle_problem():
    # Setup
    x = np.linspace(0, 1, 101)
    dx = x[1] - x[0]
    f = np.ones_like(x)
    psi = 0.1 * np.sin(np.pi * x)

    # Constraint
    constraint = ObstacleConstraint(psi, constraint_type="lower")

    # Solve
    u = solve_poisson_1d(f, dx, constraints=[constraint])

    # Verify
    assert np.all(u >= psi - 1e-10), "Solution violates constraint"

    # Check active set makes sense
    active = constraint.get_active_set(u)
    # Active set should be in middle where ψ is highest
    assert np.any(active[40:60]), "Expected active set in center"
```

**Test Case 2**: HJB with Running Cost Obstacle
```python
# Solve: ∂U/∂t + H(∇U) = 0, U(T,x) = g(x), U >= ψ(x)
# Physical interpretation: Can't have value less than immediate reward

def test_hjb_with_obstacle():
    problem = SimpleMFGProblem(dimension=1, T=1.0, N_t=100, N_x=50)

    # Obstacle: minimum value function (e.g., running cost integral)
    psi = compute_minimum_value_function(problem)

    constraint = ObstacleConstraint(psi, constraint_type="lower")

    # Solve
    solver = HJBFDMSolver(problem)
    U = solver.solve_hjb_system(constraints=[constraint])

    # Verify
    assert np.all(U >= psi - 1e-8), "HJB solution violates obstacle"

    # Check terminal condition
    assert np.allclose(U[-1, :], problem.U_terminal), "Terminal condition violated"
```

**Effort**: 2 weeks (1 week penalty method, 1 week testing + projected Newton)

---

#### 2.3 Documentation and Examples (3-5 days)

**Files to Create**:
- `examples/advanced/obstacle_problem_1d.py` - Simple obstacle problem
- `examples/advanced/mfg_with_capacity_constraints.py` - MFG with crowd capacity limits
- `docs/user/variational_inequalities.md` - User guide for VI problems

**Example 1**: Capacity-Constrained MFG
```python
# examples/advanced/mfg_with_capacity_constraints.py
"""
Capacity-constrained crowd motion (Maury et al. 2010).

Physical setup:
- Agents want to reach exit quickly (running cost = time)
- Density cannot exceed capacity: m(t,x) <= m_max
- Leads to congestion and queue formation

Mathematical formulation:
- HJB: ∂U/∂t + H(∇U) = 0 (standard)
- FP: ∂m/∂t - div(m∇U + σ²/2 ∇m) = 0, m <= m_max (VI)
"""

from mfg_pde import MFGProblem
from mfg_pde.geometry.boundary import ObstacleConstraint
import numpy as np
import matplotlib.pyplot as plt

# Problem setup
problem = MFGProblem(
    dimension=1,
    domain_bounds=np.array([[0.0, 10.0]]),  # Corridor
    T=2.0,
    N_t=200,
    N_x=100,
    sigma=0.1,
)

# Capacity constraint
m_max = 0.5  # Maximum density
capacity_constraint = ObstacleConstraint(
    obstacle=m_max * np.ones(problem.N_x),
    constraint_type="upper",
)

# Solve with constraint
result = problem.solve(
    fp_constraints=[capacity_constraint],
    max_iterations=50,
)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Density evolution
axes[0].imshow(result.M, aspect='auto', origin='lower', vmax=m_max)
axes[0].axhline(m_max, color='red', linestyle='--', label=f'm_max={m_max}')
axes[0].set_title('Density m(t,x) with Capacity Constraint')
axes[0].set_xlabel('Space')
axes[0].set_ylabel('Time')

# Active set (where m = m_max)
active_set = (result.M >= m_max - 1e-6)
axes[1].imshow(active_set, aspect='auto', origin='lower', cmap='RdYlGn_r')
axes[1].set_title('Congestion (Active Set)')
axes[1].set_xlabel('Space')
axes[1].set_ylabel('Time')

# Value function
axes[2].plot(problem.geometry.grid_points[0], result.U[0, :], label='t=0')
axes[2].plot(problem.geometry.grid_points[0], result.U[-1, :], label='t=T')
axes[2].set_title('Value Function U(t,x)')
axes[2].set_xlabel('Space')
axes[2].legend()

plt.tight_layout()
plt.show()
```

**Effort**: 3-5 days

---

### Phase 3: Tier 3 BCs - Dynamic Interfaces (3-4 weeks)

**Objective**: Implement Level Set method for free boundary problems

#### 3.1 Level Set Infrastructure (1.5 weeks)

**Files to Create**:
- `mfg_pde/geometry/level_set/core.py` - Level Set evolution
- `mfg_pde/geometry/level_set/reinitialization.py` - Redistancing
- `mfg_pde/geometry/level_set/curvature.py` - Geometric quantities

**Core Components**:
```python
# mfg_pde/geometry/level_set/core.py

class LevelSetFunction:
    """Container for level set function φ(t,x)."""

    def __init__(
        self,
        phi: NDArray,
        geometry: GeometryProtocol,
        is_signed_distance: bool = False,
    ):
        self.phi = phi
        self.geometry = geometry
        self.is_signed_distance = is_signed_distance

    @property
    def interface_mask(self) -> NDArray[np.bool_]:
        """Points near zero level set (|φ| < band_width)."""
        return np.abs(self.phi) < self.band_width

    def get_normal(self) -> NDArray:
        """Compute interface normal: n = ∇φ / |∇φ|."""
        grad_phi = self.geometry.get_gradient_operator() @ self.phi
        mag = np.linalg.norm(grad_phi, axis=-1, keepdims=True)
        return grad_phi / (mag + 1e-12)

    def get_curvature(self) -> NDArray:
        """Compute mean curvature: κ = div(∇φ / |∇φ|)."""
        normal = self.get_normal()
        div_op = self.geometry.get_divergence_operator()
        return div_op @ normal

class LevelSetEvolver:
    """Evolve level set function according to interface velocity."""

    def __init__(
        self,
        phi0: NDArray,
        geometry: GeometryProtocol,
        scheme: Literal["upwind", "weno"] = "upwind",
        reinit_frequency: int = 10,
    ):
        self.phi = LevelSetFunction(phi0, geometry)
        self.geometry = geometry
        self.scheme = scheme
        self.reinit_frequency = reinit_frequency
        self.step_count = 0

    def evolve_step(
        self,
        velocity: NDArray | Callable,
        dt: float,
    ) -> LevelSetFunction:
        """Advance level set by one time step.

        Solves: ∂φ/∂t + V|∇φ| = 0

        Args:
            velocity: Interface velocity V(x) or V(x, n, κ)
            dt: Time step

        Returns:
            Updated level set function
        """
        # Evaluate velocity (may depend on normal, curvature)
        if callable(velocity):
            V = velocity(
                self.geometry.grid_points,
                self.phi.get_normal(),
                self.phi.get_curvature(),
            )
        else:
            V = velocity

        # Hamilton-Jacobi evolution: ∂φ/∂t + V|∇φ| = 0
        if self.scheme == "upwind":
            phi_new = self._evolve_upwind(V, dt)
        elif self.scheme == "weno":
            phi_new = self._evolve_weno(V, dt)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        # Reinitialize periodically to maintain SDF property
        self.step_count += 1
        if self.step_count % self.reinit_frequency == 0:
            phi_new = self._reinitialize(phi_new)

        self.phi = LevelSetFunction(phi_new, self.geometry, is_signed_distance=True)
        return self.phi

    def _evolve_upwind(self, V: NDArray, dt: float) -> NDArray:
        """First-order upwind scheme."""
        # Compute upwind gradients
        grad_plus, grad_minus = self._compute_upwind_gradients(self.phi.phi)

        # Godunov Hamiltonian
        H = np.where(
            V >= 0,
            V * np.sqrt(np.sum(grad_minus**2, axis=-1)),  # V > 0: use ∇⁻
            V * np.sqrt(np.sum(grad_plus**2, axis=-1)),   # V < 0: use ∇⁺
        )

        # Forward Euler
        return self.phi.phi - dt * H

    def _reinitialize(self, phi: NDArray, max_iterations: int = 20) -> NDArray:
        """Reinitialize to signed distance function.

        Solves: ∂ψ/∂τ + sign(φ)(|∇ψ| - 1) = 0

        Starting from ψ(0) = φ, evolve in pseudo-time τ until
        |∇ψ| ≈ 1 near interface.
        """
        psi = phi.copy()
        dtau = 0.5 * min(self.geometry.get_grid_spacing())

        for _ in range(max_iterations):
            # Sign function (smoothed near interface)
            sign_phi = phi / np.sqrt(phi**2 + (min(self.geometry.get_grid_spacing()))**2)

            # Upwind gradients
            grad_plus, grad_minus = self._compute_upwind_gradients(psi)

            # |∇ψ| using Godunov scheme
            grad_mag = np.sqrt(np.sum(
                np.maximum(grad_minus, 0)**2 + np.minimum(grad_plus, 0)**2,
                axis=-1,
            ))

            # Pseudo-time evolution
            psi = psi - dtau * sign_phi * (grad_mag - 1)

        return psi
```

**Testing**:
- **Reinit Test**: Circle should remain circle after reinitialization
- **Advection Test**: Level set advected by constant velocity
- **Curvature Test**: Analytical curvature for simple shapes

**Effort**: 1.5 weeks

---

#### 3.2 Stefan Problem Implementation (1 week)

**Objective**: Demonstrate free boundary PDE coupling

**Example**: Classical Stefan Problem (ice melting)

**Files to Create**:
- `examples/advanced/stefan_problem_1d.py` - 1D ice melting
- `examples/advanced/stefan_problem_2d.py` - 2D solidification

```python
# examples/advanced/stefan_problem_1d.py
"""
1D Stefan Problem: Ice melting with moving boundary.

Governing equations:
- Heat equation in liquid (x > s(t)): ∂T/∂t = α ∂²T/∂x²
- Heat equation in solid (x < s(t)): ∂T/∂t = α ∂²T/∂x²
- Interface condition: V = -[k ∂T/∂n]  (Stefan condition)
- Interface temperature: T(s(t), t) = T_melt

Level set formulation:
- φ(x, t) > 0: liquid
- φ(x, t) < 0: solid
- φ = 0: interface
- Evolution: ∂φ/∂t + V|∇φ| = 0, V = -[k ∂T/∂n]
"""

from mfg_pde.geometry.level_set import LevelSetEvolver
from mfg_pde.geometry import TensorProductGrid
import numpy as np
import matplotlib.pyplot as plt

# Setup
L = 1.0
N = 200
x = np.linspace(0, L, N)
dx = x[1] - x[0]
dt = 0.0001
T_final = 0.1

geometry = TensorProductGrid(
    dimensions=1,
    bounds=np.array([[0, L]]),
    num_points=N,
)

# Initial interface position
s0 = 0.3
phi0 = x - s0  # Signed distance function

# Initial temperature (piecewise)
T = np.where(x < s0, 0.0, 1.0)  # Liquid at T=1, solid at T=0
T_melt = 0.5

# Level set evolver
ls = LevelSetEvolver(phi0, geometry, scheme="upwind", reinit_frequency=10)

# Time stepping
N_steps = int(T_final / dt)
interface_history = [s0]

for step in range(N_steps):
    # Solve heat equation
    laplacian = geometry.get_laplacian_operator(order=2)
    T_new = T + dt * laplacian @ T

    # Enforce interface temperature
    interface_mask = np.abs(ls.phi.phi) < 2*dx
    T_new[interface_mask] = T_melt

    # Compute interface velocity (Stefan condition)
    grad_T = geometry.get_gradient_operator() @ T_new
    V_interface = -grad_T[interface_mask].mean()  # Simplified

    # Evolve interface
    V_field = V_interface * np.ones(N)
    ls.evolve_step(V_field, dt)

    # Track interface position
    zero_crossing = np.where(np.diff(np.sign(ls.phi.phi)))[0]
    if len(zero_crossing) > 0:
        interface_history.append(x[zero_crossing[0]])

    T = T_new

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(interface_history)
plt.xlabel('Time Step')
plt.ylabel('Interface Position s(t)')
plt.title('Interface Evolution')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, T, label='Temperature T(x)')
plt.axvline(interface_history[-1], color='red', linestyle='--', label='Interface')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.legend()
plt.title(f'Final State (t={T_final})')

plt.subplot(1, 3, 3)
plt.plot(x, ls.phi.phi, label='φ(x)')
plt.axhline(0, color='red', linestyle='--', label='Interface (φ=0)')
plt.xlabel('x')
plt.ylabel('Level Set φ')
plt.legend()
plt.title('Level Set Function')

plt.tight_layout()
plt.show()
```

**Effort**: 1 week (implementation + examples)

---

#### 3.3 MFG with Free Boundaries (1 week)

**Objective**: Demonstrate MFG with moving domains

**Example**: Crowd evacuation with expanding exit

```python
# examples/advanced/mfg_expanding_exit.py
"""
MFG with expanding exit door.

Setup:
- Agents want to reach exit (running cost = 1)
- Exit door expands when crowded (congestion relief)
- Door expansion governed by Level Set evolution

Physics:
- Door velocity V ∝ density at door: V = k * m|_exit
- Level set: ∂φ/∂t + V|∇φ| = 0
"""

# Implementation similar to Stefan problem but coupled to MFG density
```

**Effort**: 1 week

---

### Phase 4: Advanced BC Methods (2-3 weeks)

**Objective**: Nitsche's method, penalty methods, GKS validation

#### 4.1 Nitsche's Method for FEM (1 week)

**Files to Modify**:
- `mfg_pde/geometry/boundary/applicator_fem.py` - Add Nitsche BC application

**Implementation**:
```python
# Nitsche's method for Dirichlet BC (weak imposition)

def apply_nitsche_bc(
    A: sparse.csr_matrix,
    b: NDArray,
    mesh: Mesh,
    bc: BoundaryConditions,
    penalty: float = 10.0,
) -> tuple[sparse.csr_matrix, NDArray]:
    """Apply Dirichlet BC using Nitsche's method.

    Adds terms to weak form:
    - Consistency: -∫_Γ (∂u/∂n) v ds - ∫_Γ u (∂v/∂n) ds
    - Penalty: (penalty/h) ∫_Γ u v ds
    - BC forcing: (penalty/h) ∫_Γ g v ds

    Args:
        A: Stiffness matrix
        b: RHS vector
        mesh: FEM mesh
        bc: Boundary conditions
        penalty: Penalty parameter (typically 10-100)

    Returns:
        Modified (A, b) with Nitsche BC imposed
    """
    # Extract Dirichlet boundaries
    dirichlet_segs = [seg for seg in bc.segments if seg.bc_type == BCType.DIRICHLET]

    for seg in dirichlet_segs:
        # Get boundary elements
        boundary_elements = mesh.get_boundary_elements(seg.boundary)

        for elem in boundary_elements:
            # Compute element matrices
            nodes = elem.nodes
            h = elem.diameter

            # Consistency term: -∫_Γ (∂u/∂n) v ds
            B_consistency = compute_consistency_matrix(elem)

            # Penalty term: (penalty/h) ∫_Γ u v ds
            M_penalty = penalty / h * compute_mass_matrix(elem)

            # Add to global matrix
            A[np.ix_(nodes, nodes)] += M_penalty - B_consistency - B_consistency.T

            # RHS contribution: (penalty/h) ∫_Γ g v ds
            g_values = seg.value * np.ones(len(nodes))
            b[nodes] += penalty / h * compute_mass_matrix(elem) @ g_values

    return A, b
```

**Testing**:
- Convergence test: Poisson with Dirichlet BC (Nitsche vs strong imposition)
- Stability test: Verify penalty parameter independence

**Effort**: 1 week

---

#### 4.2 GKS Stability Validation (1-2 weeks)

**Objective**: Implement GKS condition checker for BC discretizations

**Files to Create**:
- `mfg_pde/geometry/boundary/validation/gks.py` - GKS condition checker
- `tests/validation/test_gks_conditions.py` - GKS tests for all BC types

**Implementation**:
```python
# mfg_pde/geometry/boundary/validation/gks.py

def check_gks_condition(
    operator: LinearOperator,
    bc_operator: LinearOperator,
    pde_type: Literal["parabolic", "hyperbolic", "elliptic"],
    bc_type: BCType,
    order: int = 2,
) -> GKSResult:
    """Check Gustafsson-Kreiss-Sundström stability condition.

    For parabolic problems with BC discretization, the GKS condition
    requires that the combined PDE+BC operator satisfies certain
    eigenvalue bounds.

    Args:
        operator: Interior discretization (e.g., Laplacian matrix)
        bc_operator: BC discretization operator
        pde_type: Type of PDE
        bc_type: Boundary condition type
        order: Discretization order

    Returns:
        GKSResult with stability analysis
    """
    # Construct modified equation with BC
    A_combined = construct_combined_operator(operator, bc_operator)

    # Compute eigenvalues
    eigenvalues = sparse.linalg.eigs(A_combined, k=min(50, A_combined.shape[0]-1))

    # Check stability criterion
    if pde_type == "parabolic":
        # For heat equation: Re(λ) <= 0
        stable = np.all(eigenvalues.real <= 1e-10)
        criterion = "Re(λ) <= 0"
    elif pde_type == "hyperbolic":
        # For wave equation: |Im(λ)| bounded
        stable = np.all(np.abs(eigenvalues.imag) < 10 * np.linalg.norm(operator))
        criterion = "|Im(λ)| bounded"
    else:
        raise ValueError(f"GKS not implemented for {pde_type}")

    return GKSResult(
        stable=stable,
        eigenvalues=eigenvalues,
        criterion=criterion,
        max_real=eigenvalues.real.max(),
        max_imag=eigenvalues.imag.max(),
    )

@dataclass
class GKSResult:
    """Result of GKS stability check."""
    stable: bool
    eigenvalues: NDArray[np.complex128]
    criterion: str
    max_real: float
    max_imag: float

    def __str__(self) -> str:
        status = "✓ STABLE" if self.stable else "✗ UNSTABLE"
        return (
            f"GKS Stability: {status}\n"
            f"Criterion: {self.criterion}\n"
            f"Max Re(λ): {self.max_real:.6e}\n"
            f"Max Im(λ): {self.max_imag:.6e}"
        )
```

**Testing**:
```python
# tests/validation/test_gks_conditions.py

def test_gks_neumann_fdm():
    """Verify Neumann BC with 2nd-order FDM is GKS-stable."""
    # Build 1D Laplacian with Neumann BC
    N = 50
    dx = 1.0 / N
    A = build_laplacian_1d_neumann(N, dx)

    # Check GKS
    result = check_gks_condition(
        operator=A,
        bc_operator=A,  # BC embedded in operator
        pde_type="parabolic",
        bc_type=BCType.NEUMANN,
        order=2,
    )

    assert result.stable, f"Neumann BC should be GKS-stable:\n{result}"

def test_gks_robin_fdm():
    """Verify Robin BC with 2nd-order FDM is GKS-stable."""
    N = 50
    dx = 1.0 / N
    alpha, beta = 1.0, 1.0
    A = build_laplacian_1d_robin(N, dx, alpha, beta)

    result = check_gks_condition(
        operator=A,
        bc_operator=A,
        pde_type="parabolic",
        bc_type=BCType.ROBIN,
        order=2,
    )

    assert result.stable, f"Robin BC should be GKS-stable:\n{result}"
```

**Effort**: 1-2 weeks (theoretical research + implementation)

---

### Phase 5: Documentation and Integration (2 weeks)

#### 5.1 Theory Documentation (1 week)

**Files to Update**:
- `docs/theory/GEOMETRY_BC_ARCHITECTURE_DESIGN.md` - Add implementation status
- `docs/theory/variational_inequalities_theory.md` - VI mathematical background
- `docs/theory/level_set_method.md` - Level Set theory and numerics

**New Files**:
- `docs/theory/gks_lopatinskii_conditions.md` - Stability theory

---

#### 5.2 User Documentation (1 week)

**Files to Create**:
- `docs/user/advanced_boundary_conditions.md` - User guide for Tier 2/3 BCs
- `docs/user/geometry_traits.md` - Guide to geometry trait system
- `docs/user/obstacle_problems.md` - Tutorial on VI problems

**Example Content**:
```markdown
# Advanced Boundary Conditions

## Tier 2: Variational Constraints

### Obstacle Problems

Obstacle problems arise when solutions must satisfy inequality constraints:

u(x) >= ψ(x)  (lower obstacle)
u(x) <= ψ(x)  (upper obstacle)

#### Example: Capacity-Constrained Crowd Motion

from mfg_pde.geometry.boundary import ObstacleConstraint

# Maximum density constraint
m_max = 0.5
constraint = ObstacleConstraint(
    obstacle=m_max * np.ones(N),
    constraint_type="upper",
)

# Solve MFG with constraint
result = problem.solve(fp_constraints=[constraint])


The solver automatically applies projection after each iteration.
```

---

### Phase 6: Testing and Validation (1-2 weeks)

#### 6.1 Unit Tests
- Trait protocol compliance
- Operator composition
- Constraint projection
- Level Set evolution

#### 6.2 Integration Tests
- Full MFG with obstacles
- Stefan problem
- Multi-tier BC combinations

#### 6.3 Performance Benchmarks
- Overhead of constraint projection
- Level Set reinit cost
- GKS validation cost

---

## Plan A Summary

**Total Timeline**: 12-16 weeks

| Phase | Duration | Key Deliverables |
|:------|:---------|:-----------------|
| 1. Geometry Traits | 2-3 weeks | Protocols, retrofits, region registry |
| 2. Tier 2 BCs (VIs) | 3-4 weeks | Constraints, VI solver, examples |
| 3. Tier 3 BCs (Free) | 3-4 weeks | Level Set, Stefan, MFG coupling |
| 4. Advanced Methods | 2-3 weeks | Nitsche, GKS validation |
| 5. Documentation | 2 weeks | Theory + user guides |
| 6. Testing | 1-2 weeks | Unit + integration + benchmarks |

**Risk Mitigation**:
- Incremental delivery (can stop after any phase)
- Backward compatible (no breaking changes)
- Comprehensive testing at each phase

**Recommended For**: Production stability, long-term maintainability

---

## Plan B: Parallel Tracks

**Philosophy**: Run independent work streams in parallel

### Track 1: Geometry Infrastructure (4-5 weeks)
- Trait protocols
- Region registry
- Operator refactoring
- Mesh geometry completion

**Owner**: Developer 1 (geometry expert)

### Track 2: Tier 2 BCs (4-5 weeks)
- Constraint protocols
- VI solver integration
- Examples and tests

**Owner**: Developer 2 (numerical methods expert)

### Track 3: Tier 3 BCs (5-6 weeks)
- Level Set implementation
- Stefan problem
- Free boundary MFG

**Owner**: Developer 3 (PDE expert)

### Integration Phase (1-2 weeks)
- Merge all tracks
- Integration testing
- Documentation consolidation

**Total Timeline**: 8-10 weeks (with 3 developers)

**Risk**: Merge conflicts, integration challenges

---

## Plan C: Minimal Viable Extension

**Philosophy**: Deliver only what's needed for immediate research

### Scope
- **Tier 2 BCs only** (obstacle problems)
- **1D FDM only** (no FEM, no nD)
- **Penalty method only** (no projected Newton)

### Timeline

**Week 1-2**: Constraint protocol + penalty solver
**Week 3**: Validation (1D obstacle problem tests)
**Week 4**: Documentation and examples

**Total**: 4 weeks

**Best For**: Specific research need (e.g., capacity-constrained MFG paper)

---

## Plan D: Aggressive Transformation

**Philosophy**: Complete architecture overhaul, accept breaking changes

### Breaking Changes Allowed
- Deprecate legacy BC APIs
- Require all solvers use operator abstraction
- Unify FDM/FEM/GFDM under single interface
- Remove backward compatibility shims

### Additional Work
- Full nD support for all features
- GPU backend for all operators
- Adaptive mesh refinement
- Parallel time stepping

**Timeline**: 16-20 weeks

**Risk**: High (ecosystem disruption)

**Best For**: v1.0.0 release, major architectural reset

---

## Recommendation

**Use Plan A (Conservative Sequential)** for the following reasons:

1. **Low Risk**: Incremental delivery, can stop at any phase
2. **Production Stability**: No breaking changes
3. **Thorough Validation**: Each phase fully tested before next
4. **Solo Developer Friendly**: Clear linear path
5. **Research-Ready**: Tier 2 delivered by Week 7 (usable for papers)

**Alternatives**:
- **Plan C** if only need obstacles for specific paper
- **Plan B** if have multiple developers available
- **Plan D** only for major version release (v1.0.0)

---

## Resource Requirements

### For Plan A (Recommended)

**Developer Time**: 12-16 weeks × 1 FTE

**Computational Resources**:
- Testing: Standard laptop sufficient
- CI/CD: ~30 min per commit (add ~10 min for new tests)

**Dependencies**:
- No new required dependencies
- Optional: scikit-fem (for mesh geometry, already in optional deps)

---

## Success Metrics

### Phase 1 (Geometry Traits)
- ✅ All existing geometries pass trait compliance tests
- ✅ Operator accuracy tests pass (convergence order verification)
- ✅ Region registry supports all Tier 1 BC applications

### Phase 2 (Tier 2 BCs)
- ✅ 1D obstacle problem matches analytical solution (< 1% error)
- ✅ Capacity-constrained MFG shows queue formation
- ✅ Projection overhead < 5% of total solve time

### Phase 3 (Tier 3 BCs)
- ✅ Stefan problem matches literature results
- ✅ Level Set remains SDF after 100 steps (max|∇φ| - 1| < 0.1)
- ✅ MFG with free boundary converges

### Phase 4 (Advanced Methods)
- ✅ Nitsche BC matches strong imposition (< 1% difference)
- ✅ GKS validation passes for all standard BCs
- ✅ Penalty parameter independence verified

### Phase 5-6 (Docs + Testing)
- ✅ All phases documented
- ✅ >90% test coverage on new code
- ✅ All examples run successfully

---

## Migration Guide

### For Existing Code

**Tier 1 BCs (No Changes Required)**:
```python
# Old code continues to work
bc = neumann_bc(dimension=2)
U = solver.solve(bc=bc)
```

**Tier 2 BCs (New Feature)**:
```python
# Add constraints to existing solver calls
constraint = ObstacleConstraint(psi, "lower")
U = solver.solve(bc=bc, constraints=[constraint])
```

**Geometry Traits (Optional Upgrade)**:
```python
# Old API still works
laplacian_matrix = geometry.get_laplacian_operator()

# New trait-based checks
if isinstance(geometry, SupportsLaplacian):
    laplacian_matrix = geometry.get_laplacian_operator(order=4)
```

---

## Open Questions

1. **Tier 2 Solver Choice**: Penalty vs Projected Newton vs Active Set?
   - **Recommendation**: Start with penalty (simple), add projected Newton in Phase 2.2 if performance critical

2. **Level Set Scheme**: Upwind vs WENO?
   - **Recommendation**: Upwind for Phase 3.1, WENO as optional upgrade

3. **GKS Validation Scope**: All BCs or just new ones?
   - **Recommendation**: All standard BCs (DNR, Robin, Periodic) to catch regressions

4. **UnstructuredMesh Priority**: Include in Phase 1 or defer?
   - **Recommendation**: Defer to separate issue (not blocking for MFG research)

---

## Risk Analysis

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| VI solver doesn't converge | Medium | High | Start with simple penalty method, extensive testing |
| Level Set instability | Medium | Medium | Use conservative CFL, frequent reinitialization |
| Performance overhead | Low | Medium | Profile early, optimize hot paths |
| API design regret | Low | High | Design review before Phase 1 coding |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| Phase underestimated | Medium | Low | Built-in buffer (12-16 weeks), stop early if needed |
| Scope creep | High | Medium | Strict phase boundaries, user approval for additions |
| Integration issues | Low | Medium | Continuous testing, early integration |

---

## Appendix A: Detailed Work Breakdown

### Phase 1.1: Protocol Definition (3-5 days)

**Day 1**:
- Design `SupportsLaplacian`, `SupportsGradient` protocols
- Review with design doc

**Day 2**:
- Implement remaining operator protocols
- Create base trait tests

**Day 3**:
- Design topology protocols (`SupportsManifold`, etc.)
- Implement region marking protocols

**Day 4-5**:
- Write comprehensive protocol tests
- Document all protocols

---

### Phase 1.2: Retrofit Geometries (5-7 days)

**Day 1-2**: TensorProductGrid
- Add missing traits
- Formalize region registry integration
- Tests

**Day 3-4**: ImplicitDomain
- Implement gradient via SDF
- Add boundary query
- Tests

**Day 5-6**: GraphGeometry
- Graph Laplacian
- Topology traits
- Tests

**Day 7**: Integration testing across all geometries

---

### Phase 2.1: Constraint Protocol (1 week)

**Day 1-2**: Design constraint protocol
**Day 3-4**: Implement ObstacleConstraint
**Day 5**: Implement BilateralConstraint
**Day 6-7**: Testing and validation

---

### Phase 2.2: VI Solver Integration (2 weeks)

**Week 1**: Penalty method
- Modify HJB solver
- Modify Poisson solver (for testing)
- Basic tests

**Week 2**: Projected Newton (optional)
- Active set detection
- Projected line search
- Performance comparison

---

## Appendix B: Code Examples

### Example: Checking Geometry Traits
```python
from mfg_pde.geometry.protocols import (
    SupportsLaplacian,
    SupportsAdvection,
    SupportsManifold,
)

# Runtime trait checking
def solve_diffusion(geometry, u0, bc):
    if not isinstance(geometry, SupportsLaplacian):
        raise TypeError(f"{type(geometry)} doesn't support Laplacian")

    laplacian = geometry.get_laplacian_operator(bc=bc)
    # ... solve
```

### Example: Multi-Constraint Problem
```python
# Bilateral constraint: ψ_lower <= u <= ψ_upper
lower = ObstacleConstraint(psi_lower, "lower")
upper = ObstacleConstraint(psi_upper, "upper")

# Regional constraint: u >= 0 in "safe_zone"
geometry.mark_region("safe_zone", lambda x: np.linalg.norm(x) < 0.5)
safety = ObstacleConstraint(
    obstacle=0.0,
    constraint_type="lower",
    region="safe_zone",
)

# Solve with all constraints
U = solver.solve(constraints=[lower, upper, safety])
```

---

**Last Updated**: 2026-01-17
**Author**: Claude (Based on theoretical design doc)
**Status**: Ready for user review and plan selection
