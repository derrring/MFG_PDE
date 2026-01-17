# Issue #574: Adjoint-Consistent Robin BC for HJB

**Date**: 2026-01-17
**Status**: Design Phase
**Target**: v0.18.0

---

## Problem Statement

Standard Neumann BC (`∂U/∂n = 0`) for HJB at reflecting boundaries is mathematically inconsistent with the equilibrium solution when the stall point is not at the boundary.

### Mathematical Analysis

At Boltzmann-Gibbs equilibrium, the drift satisfies:
```
α* = -∇U* = (σ²/2T_eff) · ∇V
```

For `V(x) = |x - x_stall|` with `x_stall = 0`:
- At `x = 0` (stall point): `∇V = 0` → `∇U* = 0` ✓ (Neumann BC correct)
- At `x = L` (far boundary): `∇V = +1` → `∇U* ≠ 0` ✗ (Neumann BC wrong!)

### Experimental Evidence

From exp14b validation in mfg-research:

| Configuration | x_stall | Domain | Final Error |
|:--------------|:--------|:-------|:------------|
| Centered (symmetric) | 0.0 | [-0.5, 0.5] | **3.70%** |
| Boundary stall | 0.0 | [0, 1] | **9.81%** |

**Error ratio: 2.65x** - Neumann BC at the wrong location forces `∇U = 0` while equilibrium requires `∇U ≈ 0.04`.

---

## Correct Boundary Condition

### Derivation

At reflecting boundaries, the FP equation requires zero total flux:
```
J · n = 0  where  J = -σ²/2 · ∇m + m · α
```

At equilibrium (`∂/∂t = 0`), this holds everywhere, giving:
```
-σ²/2 · ∇m + m · α = 0
```

Solving for drift:
```
α = σ²/2 · ∇m/m = σ²/2 · ∇(ln m)
```

Since `α = -∇U` (for quadratic Hamiltonian):
```
∇U = -σ²/2 · ∇(ln m)
```

### Boundary Condition

At reflecting boundary with outward normal `n`:
```
∂U/∂n = -σ²/2 · ∂(ln m)/∂n
```

This is a **Robin-type BC** that couples HJB to FP through the density gradient.

---

## Implementation Design

### 1. BC Type Options

Introduce a new BC mode for HJB solver:

```python
class HJBBCMode(Enum):
    STANDARD = auto()  # Classical Neumann BC (∂U/∂n = 0)
    ADJOINT_CONSISTENT = auto()  # Coupled BC (∂U/∂n = -σ²/2 · ∂ln(m)/∂n)
```

### 2. Picard Iteration with Coupled BC

**Standard approach (current)**:
```python
for k in range(max_iterations):
    # Solve HJB with fixed Neumann BC: ∂U/∂n = 0
    U_new = solve_hjb(M_current, bc_type="neumann", bc_value=0.0)

    # Solve FP with drift from HJB
    M_new = solve_fp(U_new, bc_type="zero_flux")
```

**Adjoint-consistent approach (proposed)**:
```python
for k in range(max_iterations):
    # Compute density gradient at boundaries
    grad_ln_m_left = compute_boundary_gradient(M_current, side="left")
    grad_ln_m_right = compute_boundary_gradient(M_current, side="right")

    # Set coupled BC values
    bc_value_left = -sigma**2 / 2 * grad_ln_m_left
    bc_value_right = -sigma**2 / 2 * grad_ln_m_right

    # Solve HJB with coupled Neumann BC
    U_new = solve_hjb(
        M_current,
        bc_type="neumann",
        bc_value_left=bc_value_left,
        bc_value_right=bc_value_right,
    )

    # Solve FP with drift from HJB (same as before)
    M_new = solve_fp(U_new, bc_type="zero_flux")
```

### 3. Boundary Gradient Computation

For 1D with FDM grid:

```python
def compute_boundary_log_density_gradient(
    m: NDArray,
    dx: float,
    side: Literal["left", "right"],
    regularization: float = 1e-10,
) -> float:
    """
    Compute ∂(ln m)/∂n at boundary using one-sided finite differences.

    Args:
        m: Density array (interior points only, shape (Nx,))
        dx: Grid spacing
        side: Which boundary ("left" or "right")
        regularization: Added to m to prevent log(0)

    Returns:
        Gradient ∂(ln m)/∂n at boundary (positive outward)
    """
    m_safe = m + regularization
    ln_m = np.log(m_safe)

    if side == "left":
        # Left boundary: outward normal points left (negative x direction)
        # Use forward difference: ∂/∂n ≈ -(ln_m[1] - ln_m[0])/dx
        grad_ln_m = -(ln_m[1] - ln_m[0]) / dx
    else:  # right
        # Right boundary: outward normal points right (positive x direction)
        # Use backward difference: ∂/∂n ≈ (ln_m[-1] - ln_m[-2])/dx
        grad_ln_m = (ln_m[-1] - ln_m[-2]) / dx

    return grad_ln_m
```

### 4. Integration Points

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`

**Option A: Add parameter to HJBFDMSolver**:
```python
def __init__(
    self,
    problem: MFGProblem,
    bc_mode: Literal["standard", "adjoint_consistent"] = "standard",
    ...
):
```

**Option B: Detect automatically**:
- If problem has reflecting BCs (Neumann for both HJB and zero-flux for FP), use adjoint-consistent
- Otherwise, fall back to standard

**Option C: New method**:
```python
def solve_hjb_system_with_coupled_bc(
    self,
    M_density: NDArray,
    U_terminal: NDArray,
    ...
) -> NDArray:
```

**Recommendation**: Option A - explicit parameter for clarity and user control.

---

## Implementation Plan

### Phase 1: Core Functionality (1 day)

1. Add `compute_boundary_log_density_gradient()` utility
2. Modify HJB solver to accept per-boundary BC values
3. Update BC applicator to handle position-dependent Neumann BC

### Phase 2: Integration (1 day)

1. Add `bc_mode` parameter to HJBFDMSolver
2. Implement logic to compute coupled BC values in Picard iteration
3. Add tests for gradient computation

### Phase 3: Validation (0.5 days)

1. Create validation script similar to exp14b
2. Compare errors: standard vs adjoint-consistent
3. Verify 2.65x error reduction

### Phase 4: Documentation (0.5 days)

1. Update TOWEL_ON_BEACH_1D_PROTOCOL.md
2. Add docstring examples
3. Create tutorial notebook

**Total Effort**: ~3 days

---

## API Design

### User-Facing API

```python
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver

# Create problem with reflecting boundaries
problem = TowelOnBeachMFG(x_stall=0.0, domain=[0, 1])

# Standard approach (current)
hjb_solver = HJBFDMSolver(problem, bc_mode="standard")

# Adjoint-consistent approach (new)
hjb_solver = HJBFDMSolver(problem, bc_mode="adjoint_consistent")

# Rest of code unchanged
result = problem.solve()
```

### Internal Changes

**Modify `solve_hjb_system()` signature**:
```python
def solve_hjb_system(
    self,
    M_density: NDArray | None = None,
    U_terminal: NDArray | None = None,
    bc_values: dict[str, float] | None = None,  # NEW: per-boundary BC values
    ...
) -> NDArray:
```

**Compute coupled BC in coupling solver**:
```python
# In fixed_point_iterator.py or Picard iteration logic
if hjb_solver.bc_mode == "adjoint_consistent":
    bc_values = hjb_solver.compute_coupled_bc_values(M_current)
else:
    bc_values = None  # Use default from problem BC

U_new = hjb_solver.solve_hjb_system(
    M_density=M_current,
    U_terminal=U_terminal,
    bc_values=bc_values,
)
```

---

## Validation Strategy

### Test Cases

1. **Unit test**: Verify gradient computation
   - Known density profile (e.g., Gaussian)
   - Check analytical vs numerical gradient

2. **Integration test**: Towel-on-beach convergence
   - Boundary stall configuration
   - Compare standard vs adjoint-consistent
   - Expect ~2.65x error reduction

3. **Regression test**: Centered stall case
   - Should give same results (BC already correct)
   - Ensures backward compatibility

### Validation Metrics

- Convergence rate (Picard iterations)
- Final error vs Boltzmann-Gibbs solution
- BC consistency: verify `∂U/∂n ≈ -σ²/2 · ∂ln(m)/∂n` at boundaries

---

## Limitations and Future Work

### Current Scope (1D only)

This design focuses on 1D FDM. Extensions needed for:
- **2D/nD**: Normal gradients at rectangular boundaries
- **GFDM**: Normal gradients at arbitrary boundary points
- **Non-rectangular domains**: Curved boundaries (#549)

### Known Issues

1. **Regularization dependence**: `ln(m)` singular at `m = 0`
   - Mitigation: Add small constant (1e-10)
   - Better: Ensure m > 0 via solver design

2. **Stiffness**: Coupled BC may slow Picard convergence
   - Mitigation: Damping/relaxation on BC update
   - Monitor: Iteration count before/after

3. **Hamiltonian generality**: Assumes quadratic H
   - Derivation uses `α = -∇U`
   - For non-quadratic, need `α = -∂_p H`

---

## Related Issues

- #573: Drift interface generalization (enables non-quadratic H extension)
- #549: BC framework for non-tensor geometries (enables 2D/curved extension)
- #542: BC fixes (completed, provided foundation)

---

## References

1. **Experimental validation**: mfg-research/experiments/crowd_evacuation_2d/runners/exp14b_interior_stall_test.py
2. **Protocol**: docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md § BC Consistency
3. **Issue discussion**: https://github.com/derrring/MFG_PDE/issues/574

---

**Last Updated**: 2026-01-17
**Author**: Claude (Issue #574 investigation)
**Status**: Design approved, ready for implementation
