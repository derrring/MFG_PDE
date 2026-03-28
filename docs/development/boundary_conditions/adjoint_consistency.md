# Adjoint Consistency in MFG Discretization

**Status**: RESOLVED --- Issue #704, #622, #625
**Date**: 2026-01 (analysis), 2026-03 (consolidated)
**Related**: #574 (Adjoint-Consistent BC), #705 (Full BC support, future)

---

## 1. Problem Statement

The strict adjoint mode (`adjoint_mode="transpose"`, Issue #622) had a fundamental flaw:
**transposing the HJB gradient matrix does NOT yield a valid FP divergence operator**.

| What was attempted | What went wrong |
|--------------------|-----------------|
| `A_gradient.T` used as FP advection | Row sums != 0, violating mass conservation |
| Observed mass drift | ~60% over 100 timesteps (Issue #625 diagnostics) |
| Root cause | Gradient form ($v \cdot \nabla u$) vs divergence form ($\nabla \cdot (vm)$) use different velocity indexing |

### Why simple transpose fails

For the FP equation $\partial_t m + \nabla \cdot (vm) = D \Delta m$, implicit discretization requires:

$$\text{Row sums of } (A^T - D \cdot L) = 0$$

- Laplacian $L$ with Neumann BC: row sums = 0 (correct)
- $A_{\nabla}^T$ from HJB gradient form: row sums != 0 (broken)

Boundary diagnostic output confirmed the issue:

```
A_HJB^T[0, :5]:   [14.5, -13.5, 0, 0, 0]  -> row sum = 1.0
A_HJB^T[-1, -5:]: [0, 0, 0, -33.5, 34.5]  -> row sum = 1.0
```

Non-zero row sums at boundaries AND interior cause mass leakage.

---

## 2. Correct Solution: Scheme-Based Adjoint

The fix is to use matched discretization schemes rather than matrix transpose:

```python
# CORRECT: Use divergence_upwind for FP (this IS the discrete adjoint)
hjb_solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
fp_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")

solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    adjoint_mode="off",  # Standard mode - schemes handle adjoint structure
)
```

### Why this works

The `divergence_upwind` scheme constructs coefficients equivalent to the **Jacobian transpose** (not the gradient matrix transpose):

| Scheme | Operator | Stencil (row $i$) | Mass Conservation |
|--------|----------|---------------------|-------------------|
| `gradient_upwind` | $v \cdot \nabla u$ | $[-v_i/h, +v_i/h, 0]$ | N/A (for HJB) |
| Naive `.T` | $(A_{\nabla})^T$ | $[0, +v_i/h, -v_{i+1}/h]$ | Broken |
| `divergence_upwind` | $\nabla \cdot (vm)$ | $[-v_{i-1}/h, +v_i/h, 0]$ | Conserved |

The critical difference is velocity indexing: `divergence_upwind` uses $v_{i-1}$ and $v_i$, matching the Jacobian transpose structure from Achdou's method. Both schemes use the same velocity field, and the divergence scheme is equivalent to the Jacobian transpose --- not the gradient matrix transpose.

### Deprecated: matrix transpose modes

| Mode | Status | Notes |
|------|--------|-------|
| `adjoint_mode="off"` | **Recommended** | Use scheme pairing instead |
| `adjoint_mode="transpose"` | **DEPRECATED** | Mathematically wrong |
| `adjoint_mode="auto"` | **DEPRECATED** | Based on wrong foundation |

Deprecated code paths:
- `HJBFDMSolver.build_advection_matrix()` --- returns gradient matrix, not suitable for FP
- `FPFDMSolver.solve_fp_step_adjoint_mode()` --- uses wrong matrix
- `FixedPointIterator._solve_fp_strict_adjoint()` --- broken

---

## 3. AdjointConsistentProvider Pattern (Issue #625)

For reflecting boundaries where the HJB boundary condition must couple to FP density, the `AdjointConsistentProvider` pattern resolves BC values at iteration time.

### Architecture

1. `AdjointConsistentProvider` stored as intent in `BCSegment.value`
2. `FixedPointIterator` calls `problem.using_resolved_bc(state)` each iteration
3. Provider computes concrete Robin BC value from current density: $g = -\sigma^2/2 \cdot \partial \ln(m) / \partial n$
4. Solver receives resolved BC with no MFG coupling knowledge needed

### Usage

```python
from mfgarchon.geometry.boundary import (
    AdjointConsistentProvider, BCSegment, BCType, BoundaryConditions, neumann_bc
)

# Standard Neumann BC (default, no coupling)
problem = MFGProblem(..., boundary_conditions=neumann_bc(dimension=1))

# Adjoint-consistent BC via provider pattern (Issue #625)
bc = BoundaryConditions(segments=[
    BCSegment(name="left_ac", bc_type=BCType.ROBIN,
              alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="left", diffusion=sigma),
              boundary="x_min"),
    BCSegment(name="right_ac", bc_type=BCType.ROBIN,
              alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="right", diffusion=sigma),
              boundary="x_max"),
], dimension=1)
problem = MFGProblem(..., boundary_conditions=bc)
```

### When to use

- Reflecting boundaries with stall point at domain boundary
- Near-equilibrium or high-accuracy requirements
- Boundary stall configurations (>1000x improvement observed in some cases)
- NOT needed for interior stall points or periodic BC

### Implementation files

- **Provider**: `mfgarchon/geometry/boundary/providers.py` (`BCValueProvider` protocol, `AdjointConsistentProvider`)
- **BC coupling**: `mfgarchon/geometry/boundary/bc_coupling.py` (`create_adjoint_consistent_bc_1d()`)
- **Iterator integration**: `mfgarchon/alg/numerical/coupling/fixed_point_iterator.py` (resolves providers via `problem.using_resolved_bc(state)`)

---

## 4. Boundary Condition Support for Adjoint Modes

| BC Type | Transpose valid? | Scheme pairing valid? |
|---------|------------------|----------------------|
| Reflecting (zero-flux Neumann) | Yes | Yes |
| Periodic | Yes | Yes |
| Absorbing/Outflow (homogeneous) | Needs correction | Yes |
| Dirichlet ($m = g$) | No | Yes |
| Robin ($\alpha m + \beta \partial m / \partial n = g$) | No | Yes |
| Inflow / non-zero flux | No | Yes |

Key insight: scheme-based adjoint (`gradient_upwind` + `divergence_upwind`) works for all BC types because each solver applies its own boundary treatment. The transpose approach only works for homogeneous BCs.

Full BC support for non-homogeneous cases tracked in Issue #705.

---

## 5. Diagnostics (Still Useful)

The `adjoint/` module diagnostics remain useful for verifying scheme consistency:

```python
from mfgarchon.alg.numerical.adjoint import check_operator_adjoint, diagnose_adjoint_error

# Compare scheme matrices at a given U
A_hjb = hjb_solver._build_gradient_matrix(U)    # gradient_upwind
A_fp = fp_solver._build_divergence_matrix(U)     # divergence_upwind

# Verify consistency (these should satisfy A_fp ~ A_jacobian.T)
report = diagnose_adjoint_error(A_hjb, A_fp, geometry=geometry)
# Identifies boundary vs interior discrepancies
```

Verification can also be enabled at solver level:

```python
solver = FixedPointIterator(
    ...,
    adjoint_verification=True,  # Log warnings for scheme mismatch
    adjoint_rtol=1e-10,
)
```

---

## 6. Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Protocol & validation | COMPLETED | `adjoint/protocols.py`, init-time checks |
| Verification integration | COMPLETED | `adjoint_verification` param, logging |
| Diagnostics integration | COMPLETED | `_generate_adjoint_report()`, metadata |
| User-facing API in `MFGProblem` | Deferred | Users access via solver directly |
| Full BC support (Issue #705) | Future | Dirichlet, Robin, inflow |

---

## 7. Lessons Learned

1. **Simple transpose != discrete adjoint** for gradient/divergence operators
2. **Jacobian transpose** (from linearized HJB) = discrete adjoint, but this is what `divergence_upwind` already implements
3. Matched scheme pairing (`gradient_upwind` + `divergence_upwind`) is the correct and simpler approach
4. Always verify mass conservation when implementing adjoint modes
5. BC coupling for reflecting boundaries requires the provider pattern, not matrix manipulation

---

## 8. References

- **Theory**: `docs/theory/adjoint_discretization_mfg.md` --- complete mathematical foundations
- **Achdou et al. (2010)**: "Mean Field Games: Numerical Methods" --- original structure-preserving method
- **CLAUDE.md**: Boundary Condition Coupling Patterns section
- **Towel-on-beach protocol**: `docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md` (BC consistency)

### Superseded documents

- `adjoint_integration_design.md` --- original design doc (620 lines, contained deprecated transpose approach)
- `issue_625_adjoint_consistency_analysis.md` --- diagnostic analysis (173 lines, recommended scheme-based fix)
