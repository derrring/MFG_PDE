# Technical Protocol: Implicit Matrix Assembly with 2+4 Boundary Architecture

**Protocol for Unified Matrix Assembly with 2+4 Boundary Architecture**

**Version**: 1.0
**Status**: Technical Specification
**Scope**: Finite Difference Methods (FDM) for FP, HJB, Poisson, and general PDEs
**Related**: `bc_architecture_analysis.md`, Issue #486

---

## Executive Summary

This document standardizes the communication protocol between **Implicit Matrix Assembly** and the **2+4 Boundary Condition Architecture**. The core goal is to ensure that implicit solvers (matrix-based) and explicit solvers (ghost cell-based) produce mathematically identical results while maintaining architectural decoupling.

---

## 1. Core Axioms

### Axiom 1: Single Source of Truth

The matrix assembler **MUST NOT** derive boundary logic independently. It MUST query the same `BoundaryConfig` used by explicit solvers.

```python
# WRONG: Assembler hard-codes BC logic
if is_boundary:
    A[i, i] = 1.0  # Dirichlet assumption

# CORRECT: Assembler queries unified BC interface
constraint = bc_adapter.resolve(boundary_config, grid_info)
```

### Axiom 2: Algebraic-Geometric Equivalence

| Solver Type | Mechanism | Domain |
|-------------|-----------|--------|
| **Explicit** | Ghost Cell Padding | Geometric extension |
| **Implicit** | Coefficient Folding | Algebraic modification |

**Axiom**: Both MUST produce identical numerical results:
```
A_implicit @ u = Stencil(u_padded)
```

This equivalence is required for GKS (Gustafsson-Kreiss-Sundström) stability.

### Axiom 3: Semantics First

The matrix assembler should not care about physical names ("Wall", "Inlet"). It only cares about the **4-Tier semantic classification** (State/Gradient/Flux/Artificial).

---

## 2. The Linear Constraint Adapter

### 2.1 Data Structure

All FDM boundary conditions can be expressed as a linear relationship between ghost and interior nodes:

```python
from dataclasses import dataclass

@dataclass
class LinearConstraint:
    """
    Linear constraint: u_ghost = sum(weights[k] * u[inner+k]) + bias

    Attributes:
        weights: {relative_offset: coefficient}
                 offset 0 = boundary-adjacent interior point
                 offset 1 = one cell further inward, etc.
        bias: Constant term (for inhomogeneous BCs)
    """
    weights: dict[int, float]
    bias: float = 0.0
```

### 2.2 Tier-to-Constraint Mapping

| Tier | Semantic | Physical Examples | Constraint Pattern |
|------|----------|-------------------|-------------------|
| **Tier 1: State** | Lock Value | Dirichlet, Exit | `weights={}`, `bias=g` |
| **Tier 2: Gradient** | Lock Shape | Neumann, Symmetry | `weights={0: 1.0}`, `bias=dx*g` |
| **Tier 3: Flux** | Lock Flow | Robin, FP No-Flux | `weights={0: α}`, `bias=0` |
| **Tier 4: Artificial** | Fake Infinity | Linear Extrapolation | `weights={0: 2, 1: -1}`, `bias=0` |

### 2.3 Adapter Implementation

```python
def resolve_constraint(
    bc_config: BoundaryConfig,
    grid_info: GridInfo,
    pde_context: PDEContext,
) -> LinearConstraint:
    """
    Convert BC configuration to LinearConstraint for matrix folding.

    The pde_context allows Tier 3 (Flux) to compute equation-specific
    coefficients (e.g., drift and diffusion for FP).
    """
    tier = bc_config.semantic_tier

    if tier == Tier.STATE:
        # Tier 1: Direct value constraint
        return LinearConstraint(weights={}, bias=bc_config.value)

    elif tier == Tier.GRADIENT:
        # Tier 2: Derivative constraint
        sign = 1.0 if bc_config.side == "max" else -1.0
        return LinearConstraint(
            weights={0: 1.0},
            bias=sign * grid_info.dx * bc_config.gradient
        )

    elif tier == Tier.FLUX:
        # Tier 3: Conservation constraint (equation-dependent)
        alpha = compute_robin_coefficient(
            bc_config, grid_info, pde_context
        )
        return LinearConstraint(weights={0: alpha}, bias=0.0)

    elif tier == Tier.ARTIFICIAL:
        # Tier 4: Extrapolation constraint
        if bc_config.order == 1:  # Linear
            return LinearConstraint(weights={0: 2.0, 1: -1.0}, bias=0.0)
        else:  # Quadratic
            return LinearConstraint(weights={0: 3.0, 1: -3.0, 2: 1.0}, bias=0.0)
```

---

## 3. Standard Matrix Assembly Algorithm

### 3.1 The Two-Phase Assembly Loop

```python
def assemble_matrix(
    N: int,
    stencil: list[tuple[int, float]],  # [(offset, weight), ...]
    bc_configs: dict[str, BoundaryConfig],  # {'left': ..., 'right': ...}
    grid_info: GridInfo,
    pde_context: PDEContext,
) -> tuple[SparseMatrix, NDArray]:
    """
    Standard matrix assembly with 2+4 BC integration.

    Phase 1: Topology (Layer 1) - Handle periodicity
    Phase 2: Physics (Layer 2) - Handle bounded BC via folding
    """
    A = sparse_matrix((N, N))
    b = zeros(N)

    for i in range(N):
        for offset, weight in stencil:
            col = i + offset

            # ============================================
            # PHASE 1: TOPOLOGY CHECK (Layer 1)
            # ============================================
            # Periodic BC is pure geometry - no physics needed
            if is_periodic(col, bc_configs):
                wrapped_col = col % N
                A[i, wrapped_col] += weight
                continue

            # ============================================
            # INTERIOR POINT - Direct assignment
            # ============================================
            if 0 <= col < N:
                A[i, col] += weight
                continue

            # ============================================
            # PHASE 2: PHYSICS FOLDING (Layer 2)
            # ============================================
            # Only for out-of-bounds, non-periodic points

            # Step 1: Determine boundary side
            side = 'left' if col < 0 else 'right'

            # Step 2: Resolve semantic tier to linear constraint
            # KEY DECOUPLING: Assembler doesn't know if it's HJB or FP
            constraint = resolve_constraint(
                bc_configs[side], grid_info, pde_context
            )

            # Step 3: Execute Coefficient Folding

            # A. Fold weights into matrix
            for rel_offset, fold_weight in constraint.weights.items():
                # Map relative offset to global index
                inner_idx = map_to_global_index(side, rel_offset, N)
                # Core formula: A[i, inner] += stencil_w * constraint_w
                A[i, inner_idx] += weight * fold_weight

            # B. Fold bias into RHS (note the sign!)
            # Original: weight * (sum_of_weighted_terms + bias)
            # Moving bias to RHS: b[i] -= weight * bias
            b[i] -= weight * constraint.bias

    return A, b
```

### 3.2 Performance Note: Sparse Matrix Construction

> **Implementation Tip**: Constructing a sparse matrix incrementally using `A[i,j] += val` can be slow in Python/SciPy (LIL format). Collect triplets in lists and create the matrix once:
>
> ```python
> rows, cols, data = [], [], []
> # ... in loop: rows.append(i), cols.append(j), data.append(val)
> A = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
> ```

### 3.3 Index Mapping Utilities

```python
def map_to_global_index(side: str, rel_offset: int, N: int) -> int:
    """
    Map relative offset from boundary to global index.

    For left boundary (ghost at col < 0):
        rel_offset 0 -> index 0 (first interior)
        rel_offset 1 -> index 1 (second interior)

    For right boundary (ghost at col >= N):
        rel_offset 0 -> index N-1 (last interior)
        rel_offset 1 -> index N-2 (second-to-last interior)
    """
    if side == 'left':
        return rel_offset
    else:  # 'right'
        return N - 1 - rel_offset


def is_periodic(col: int, bc_configs: dict) -> bool:
    """Check if this column falls under periodic topology."""
    if col < 0:
        return bc_configs.get('left', {}).get('topology') == 'periodic'
    elif col >= N:
        return bc_configs.get('right', {}).get('topology') == 'periodic'
    return False
```

---

## 4. Equation-Specific Implementation Notes

### 4.1 Fokker-Planck (FP) Equation

**Characteristic**: Mass conservation is critical.

**Tier 3 (Flux) for Walls**:
- Never use Dirichlet (ρ=0) for walls - this destroys mass
- MUST use **Zero-Flux Robin**: `J·n = (v·n)ρ - D∂ρ/∂n = 0`
- The adapter computes local drift `v(x_boundary)` and diffusion `D(x_boundary)`

**Robin Coefficient Computation**:
```python
def compute_fp_robin_alpha(
    drift_velocity: float,  # Normal component, positive = outward
    diffusion_coeff: float,  # D = σ²/2
    dx: float,
    side: str,
) -> float:
    """
    Compute FP no-flux Robin coefficient.

    From J·n = 0: v*ρ - D*∂ρ/∂n = 0
    Discretized: v*(ρ_ghost + ρ_inner)/2 - D*(ρ_ghost - ρ_inner)/dx = 0
    Solving for ρ_ghost: ρ_ghost = α * ρ_inner
    where α = (2D + v*dx) / (2D - v*dx)
    """
    outward_sign = 1.0 if side == 'right' else -1.0
    v_n = drift_velocity * outward_sign
    D = diffusion_coeff

    # Guard against division by zero
    denominator = 2*D - v_n*dx
    if abs(denominator) < 1e-14:
        return 1.0  # Fallback to Neumann

    return (2*D + v_n*dx) / denominator
```

**Assembler Behavior**: Assembler only receives `weights={0: alpha}`, completely unaware of drift/diffusion physics.

### 4.2 Hamilton-Jacobi-Bellman (HJB) Equation

**Characteristic**: Information flows unidirectionally (upwind), strongly nonlinear.

**Tier 2 (Gradient) for Walls**:
- Reflective wall: `∂V/∂n = 0` → `weights={0: 1.0}, bias=0`
- Physical meaning: Hitting wall doesn't change value

**Tier 4 (Artificial) for Far-field**:
- Linear extrapolation for unbounded domains
- `weights={0: 2.0, 1: -1.0}` (assumes linear growth)
- For LQG with quadratic value: use `weights={0: 3, 1: -3, 2: 1}` (quadratic extrapolation)

**Upwind Considerations**:
- HJB often uses one-sided stencils based on gradient sign
- The assembler must still query BC at upwind boundary
- Viscosity solution may require special treatment at boundary

### 4.3 Poisson/Laplace Equation

**Characteristic**: Elliptic, equilibrium problem.

**Tier 1 (State) for Dirichlet**:
- `weights={}`, `bias=g` → Pure RHS modification
- Matrix row at boundary: `[0, ..., 1, ..., 0]` with `b[i] = g`

**Tier 2 (Gradient) for Neumann**:
- `weights={0: 1.0}`, `bias=dx*g`
- Modifies diagonal and RHS

---

## 5. Multi-Dimensional Extensions

### 5.1 Corner Ghost Cells (2D+)

**Problem**: In 2D+, stencils may access corner ghosts that depend on both x and y boundaries.

**Solution**: Recursive folding or dimension splitting

```python
# Option 1: Recursive Folding (complex but general)
# Corner ghost depends on edge ghosts, edge ghosts depend on interior
constraint_x = resolve_constraint(bc_x, ...)
constraint_y = resolve_constraint(bc_y, ...)
# Compose constraints...

# Option 2: Dimension Splitting (simpler, recommended)
# Apply X-direction folding first, then Y-direction
# Use compact stencils that avoid direct corner access
```

**Standard**: Apply boundaries in **X→Y→Z order**. Document this explicitly.

### 5.2 Processing Order in nD

```python
for i_flat in range(N_total):
    multi_idx = unflatten(i_flat, shape)

    for d in range(ndim):
        for offset in stencil_offsets[d]:
            neighbor_idx = multi_idx.copy()
            neighbor_idx[d] += offset

            # Check each dimension independently
            if is_out_of_bounds(neighbor_idx, d, shape):
                # Apply folding for dimension d
                constraint = resolve_constraint_for_dim(d, side, ...)
                apply_folding(A, b, i_flat, constraint, ...)
```

---

## 6. Grid Alignment Considerations

### 6.1 Cell-Centered vs Vertex-Centered

| Grid Type | Boundary Location | Ghost Formula Difference |
|-----------|------------------|-------------------------|
| **Vertex-Centered** | Boundary at grid node | `u[0] = g` for Dirichlet |
| **Cell-Centered** | Boundary at face between ghost and first cell | `u_ghost = 2g - u[0]` for Dirichlet |

**Critical**: The adapter MUST know the grid type. This affects Tier 2 and Tier 3 coefficient computation.

```python
def resolve_constraint(bc_config, grid_info, pde_context):
    grid_type = grid_info.grid_type  # VERTEX or CELL_CENTERED

    if bc_config.tier == Tier.STATE:
        if grid_type == GridType.VERTEX_CENTERED:
            # Boundary value directly assigned
            return LinearConstraint(weights={}, bias=bc_config.value)
        else:  # CELL_CENTERED
            # Ghost cell extrapolation: u_ghost = 2g - u_inner
            return LinearConstraint(weights={0: -1.0}, bias=2*bc_config.value)
```

---

## 7. Developer Checklist

When implementing a new PDE solver or BC type, verify:

- [ ] **Grid Alignment**: Does the adapter know if grid is `VERTEX_CENTERED` or `CELL_CENTERED`?

- [ ] **Topology First**: Does the assembler check periodicity BEFORE physics?

- [ ] **Tier Classification**: Is the BC correctly classified as Tier 1/2/3/4?

- [ ] **RHS Sign**: Is `bias` subtracted (not added) when moving to RHS?

- [ ] **Corner Handling**: For 2D+ with 9-point stencils, is corner folding handled correctly?

- [ ] **Coefficient Consistency**: Does `α` computation for Tier 3 match the explicit ghost formula?

- [ ] **Index Bounds**: Are all `inner_idx` values within `[0, N-1]`?

- [ ] **Equation Context**: For Tier 3, is the correct `pde_context` (drift, diffusion) passed?

---

## 8. Reference Implementation

See `mfg_pde/geometry/boundary/applicator_base.py`:

```python
from mfg_pde.geometry.boundary.applicator_base import (
    LinearConstraint,
    calculator_to_constraint,
    DirichletCalculator,
    NeumannCalculator,
    ZeroFluxCalculator,
    LinearExtrapolationCalculator,
)

# Example: Convert FP no-flux calculator to constraint
calculator = ZeroFluxCalculator(drift_velocity=0.1, diffusion_coeff=0.5)
constraint = calculator_to_constraint(calculator, dx=0.02, side='right')
# constraint.weights = {0: 1.222...}, constraint.bias = 0.0
```

---

## References

1. `bc_architecture_analysis.md` - Section 1 (Four-Tier Constraint Taxonomy)
2. `bc_architecture_analysis.md` - Section 3.4 (Implicit Solver Matrix Assembly)
3. Gustafsson, Kreiss, Oliger (1995). *Time Dependent Problems and Difference Methods*. — GKS stability theory.

---

**Document History**:
- 2025-12: Initial protocol specification
