# State-Dependent Boundary Condition Coupling for MFG

## Overview

In Mean Field Game systems with reflecting boundaries, the boundary condition for the HJB equation may depend on the FP density solution. This document explains the mathematical derivation, physical interpretation, and implementation.

**Key formula**:
$$\frac{\partial U}{\partial n} = -\frac{\sigma^2}{2} \cdot \frac{\partial \ln m}{\partial n}$$

## Mathematical Derivation

### Setup

Consider the MFG system on domain $\Omega$ with reflecting (no-flux) boundaries:

**HJB equation** (backward in time):
$$-\partial_t U + H(\nabla U) = f(m)$$

**FP equation** (forward in time):
$$\partial_t m - \frac{\sigma^2}{2} \Delta m - \nabla \cdot (m \cdot \alpha) = 0$$

where $\alpha = -\nabla_p H = -\nabla U$ for quadratic Hamiltonian $H(p) = |p|^2/2$.

### No-Flux Condition for FP

At reflecting boundaries, mass cannot leave the domain. The total flux must vanish:
$$J \cdot n = 0 \quad \text{on } \partial\Omega$$

where the flux is:
$$J = -\frac{\sigma^2}{2} \nabla m + m \cdot \alpha$$

Substituting $\alpha = -\nabla U$:
$$J = -\frac{\sigma^2}{2} \nabla m - m \nabla U$$

### Equilibrium Condition

At the boundary, $J \cdot n = 0$ implies:
$$-\frac{\sigma^2}{2} \frac{\partial m}{\partial n} - m \frac{\partial U}{\partial n} = 0$$

Solving for $\partial U / \partial n$:
$$\frac{\partial U}{\partial n} = -\frac{\sigma^2}{2} \cdot \frac{1}{m} \frac{\partial m}{\partial n} = -\frac{\sigma^2}{2} \cdot \frac{\partial \ln m}{\partial n}$$

This is a **Robin-type BC** for HJB that couples to the FP density gradient.

## Physical Interpretation

| Term | Meaning |
|------|---------|
| $\partial U / \partial n$ | Rate of change of value function normal to boundary |
| $\partial \ln m / \partial n$ | Relative density gradient at boundary |
| $\sigma^2/2$ | Diffusion strength |

**Intuition**: When density piles up at the boundary ($\partial m / \partial n < 0$ pointing inward), the HJB gradient adjusts so that the optimal control reduces drift toward that boundary. This prevents artificial "trapping" of agents at boundaries.

### When This Matters

| Scenario | Standard BC | State-Dependent BC |
|----------|-------------|-------------------|
| Interior stall point | ✓ Sufficient | Not needed |
| Boundary stall point | ✗ Large error | ✓ Required |
| Near equilibrium | ✗ Inconsistent | ✓ Consistent |
| Transient dynamics | ✓ Usually OK | Optional improvement |

**Key insight**: The coupling becomes critical when equilibrium forms at or near domain boundaries.

## Implementation

### Robin BC Framework

The state-dependent coupling is implemented as Robin BC:
$$\alpha \cdot U + \beta \cdot \frac{\partial U}{\partial n} = g$$

with:
- $\alpha = 0$ (no $U$ term)
- $\beta = 1$ (coefficient of $\partial U / \partial n$)
- $g = -\frac{\sigma^2}{2} \cdot \frac{\partial \ln m}{\partial n}$ (computed from density)

### Code Usage (v0.18.0+ Provider Pattern)

```python
from mfg_pde.geometry.boundary import (
    AdjointConsistentProvider, BCSegment, BCType, BoundaryConditions
)

# Store intent in BC object at problem creation time
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

# FixedPointIterator resolves providers each iteration automatically
problem = MFGProblem(geometry=grid, boundary_conditions=bc, ...)
result = problem.solve()
```

### Numerical Computation

The boundary gradient is computed using one-sided finite differences:

**Left boundary** ($x = x_{min}$, outward normal in $-x$ direction):
$$\frac{\partial \ln m}{\partial n}\bigg|_{left} = -\frac{\ln m_1 - \ln m_0}{\Delta x}$$

**Right boundary** ($x = x_{max}$, outward normal in $+x$ direction):
$$\frac{\partial \ln m}{\partial n}\bigg|_{right} = \frac{\ln m_{N-1} - \ln m_{N-2}}{\Delta x}$$

A small regularization $\epsilon \approx 10^{-10}$ is added to prevent $\ln(0)$.

## Architecture Decision

### Why `bc_mode` Was Removed (v0.18.0)

The `bc_mode` parameter on `HJBFDMSolver` was a design mistake — it embedded MFG coupling logic inside a PDE solver. It was deprecated in v0.17.1 and removed in v0.18.0 (Issue #703).

The replacement is the **BCValueProvider pattern** (Issue #625):

```python
# v0.18.0+: Provider pattern — BC intent stored in problem, resolved by iterator
from mfg_pde.geometry.boundary import AdjointConsistentProvider, BCSegment, BCType, BoundaryConditions

bc = BoundaryConditions(segments=[
    BCSegment(name="left_ac", bc_type=BCType.ROBIN,
              alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="left", diffusion=sigma),
              boundary="x_min"),
    # ... right boundary similarly
], dimension=1)
problem = MFGProblem(..., boundary_conditions=bc)
result = problem.solve()  # Iterator resolves providers automatically
```

**Why this is correct**:
1. Solvers solve PDEs — they don't decide physics
2. Coupling decision is explicit in the BC object, not hidden in a solver flag
3. The iterator resolves providers each iteration via `problem.using_resolved_bc(state)`

### Structural vs State-Dependent Adjoint

| Mechanism | How it works | When sufficient |
|-----------|--------------|-----------------|
| **Structural adjoint** | Same BC + same stencil → $A_{FP} = A_{HJB}^T$ | Interior stall, standard problems |
| **State-dependent coupling** | HJB BC = $f(m)$ at boundary | Boundary stall, high accuracy |

Structural adjoint is **free** (by construction). State-dependent coupling is **optional** for enhanced accuracy at boundaries.

## Validation

The coupling can be validated by checking discrete duality:
$$\langle m, L_{HJB} U \rangle = \langle L_{FP}^* m, U \rangle$$

where $L_{FP}^*$ is the adjoint FP operator. With correct BC coupling, this identity holds to machine precision.

See: `mfg-research/experiments/crowd_evacuation_2d/runners/exp14b_fdm_bc_fix_validation.py`

## References

- Issue #574: Adjoint-consistent BC implementation
- Issue #703: Deprecation of `bc_mode` parameter
- `mfg_pde/geometry/boundary/bc_coupling.py`: Implementation
- `docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md`: Validation protocol

## Structural Adjoint: A Closer Look

The claim "same BC + same stencil = adjoint by construction" is a simplification. The reality is more nuanced.

### FDM: Interior vs Boundary

| Region | $A_{FP} = A_{HJB}^T$? | Reason |
|--------|----------------------|--------|
| **Interior** | ✓ Exact | Same upwind stencils |
| **Boundary** | ✗ Approximate | Different BC implementations |

**HJB-FDM** (Neumann $\partial U/\partial n = 0$):
- Uses ghost cell: $U_{ghost} = U_{boundary}$ (symmetric extension)
- Laplacian at boundary: $(U_1 - U_0)/\Delta x^2$

**FP-FDM** (no-flux $J \cdot n = 0$):
- Truncates flux at boundary (no flux term added)
- This is NOT $\partial m/\partial n = 0$

The two approaches give **different matrix entries** at boundary rows/columns.

### Semi-Lagrangian: Interpolation vs Splatting

| Component | HJB-SL | FP-SL | Adjoint? |
|-----------|--------|-------|----------|
| **Interior advection** | Interpolation | Splatting | ✓ Exact |
| **Boundary advection** | Reflect + Interp | Reflect + Splat | ✓ Approximate |
| **Diffusion** | (none in pure SL) | Crank-Nicolson with Neumann | ❌ Breaks adjoint |

**The critical issue for SL**: Operator splitting in FP-SL uses Neumann BC ($\partial m/\partial n = 0$) for the diffusion step, not the physical no-flux condition ($J \cdot n = 0$). This introduces boundary error.

### When Structural Adjoint Suffices

| Scenario | Boundary error impact | Structural adjoint |
|----------|----------------------|-------------------|
| Interior stall point | Small (far from boundary) | ✓ Sufficient |
| Boundary stall point | Large (density piles at boundary) | ✗ Insufficient |
| Large diffusion $\sigma$ | Medium | Often sufficient |
| Small diffusion $\sigma$ | Large (advection-dominated) | ✗ Insufficient |

### When State-Dependent Coupling is Needed

The state-dependent BC coupling ($\partial U/\partial n = -\frac{\sigma^2}{2} \partial \ln m/\partial n$) corrects the boundary adjoint mismatch by:

1. Making HJB "aware" of FP density gradient at boundary
2. Ensuring discrete duality $\langle m, L_{HJB} U \rangle = \langle L_{FP}^* m, U \rangle$ holds including boundary terms
3. Most critical when equilibrium forms at or near domain boundaries

## Extension to nD

The 1D implementation generalizes to nD by:
1. Computing $\nabla \ln m$ at boundary points using geometry's gradient operator
2. Projecting onto outward normal: $\frac{\partial \ln m}{\partial n} = \nabla \ln m \cdot \hat{n}$
3. Creating Robin BC segments for each boundary face

Currently only 1D is implemented. nD extension is tracked in `bc_coupling.py`.
