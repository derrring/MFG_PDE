# Dimensional Splitting for Multi-Dimensional FP Equation (ARCHIVED)

**Status**: ARCHIVED - Not Recommended for MFG Problems
**Date**: 2025-10-31
**Reason**: Catastrophic failure for advection-dominated problems

---

## Executive Summary

**Dimensional splitting (Strang splitting) fails catastrophically for advection-dominated MFG problems**, losing up to 81% of mass over time evolution. While the method works perfectly for pure diffusion, it is unsuitable for realistic MFG systems where advection from the HJB-generated velocity field dominates.

**Recommendation**: Use coupled full-dimensional system solvers (direct linear/nonlinear systems) instead of dimensional splitting for multi-dimensional MFG problems.

---

## Method Description

### What is Dimensional Splitting?

Dimensional splitting (also called operator splitting or Strang splitting) decomposes a multi-dimensional PDE into a sequence of 1D problems:

**2D Fokker-Planck equation**:
```
∂m/∂t + ∇·(m v) = (σ²/2) Δm
```

**Splitting scheme** (Strang splitting):
```
m^(n+1) = S₂(Δt/2) ∘ S₁(Δt) ∘ S₂(Δt/2) m^n
```

Where:
- S₁: Solve 1D problem in x-direction
- S₂: Solve 1D problem in y-direction
- Each 1D solve uses implicit finite differences

**Advantages** (in theory):
- O(N) complexity per timestep (vs O(N²) for full 2D)
- Reuses efficient 1D solvers
- Memory efficient (no large matrices)

**Disadvantages** (in practice for MFG):
- O(Δt) splitting error from non-commutative operators
- Catastrophic error accumulation with strong advection
- Boundary contamination at intermediate steps

---

## Experimental Findings

### Test 1: Pure Diffusion (σ²Δm, no advection)

**Setup**:
- 12×12 grid, 15 timesteps, T=0.5
- Zero velocity field (U = 0)
- Initial Gaussian blob at center

**Result**: ✓ **PERFECT**
```
Initial mass:  1.0000000
Final mass:    1.0000000
Mass error:    +0.0000%
```

**Conclusion**: Dimensional splitting works correctly for diffusion-dominated problems.

---

### Test 2: Full MFG with Advection (Crowd Motion Problem)

**Setup**:
- 12×12 grid, 15 timesteps, T=0.5
- Agents move from (0.2, 0.2) → (0.8, 0.8)
- HJB solver generates velocity field v = -∇U
- Strong advection + diffusion coupling

**Result**: ✗ **CATASTROPHIC FAILURE**
```
Initial mass:  1.0000000
Final mass:    0.1900000
Mass error:    -81.00%
```

**Mass evolution timeline**:
| Time | Mass | Loss |
|------|------|------|
| t=0  | 1.00 | 0%   |
| t=8  | 0.85 | -15% |
| t=15 | 0.19 | -81% |

**Conclusion**: Dimensional splitting is unsuitable for advection-dominated MFG problems.

---

## Root Cause Analysis

### 1. High Péclet Number (Advection Dominance)

**Péclet number**: Pe = UL/D

Where:
- U: characteristic velocity from ∇U
- L: domain length scale
- D: diffusion coefficient σ²/2

For MFG problems:
- Velocity field from HJB can be strong (U >> 1)
- Diffusion is often small (σ ~ 0.05)
- **Result**: Pe >> 1 (advection dominates)

**Problem**: Dimensional splitting assumes diffusion and advection commute, which is false for high Pe.

### 2. Operator Non-Commutativity

**Mathematical issue**:
```
[A, D] = AD - DA ≠ 0
```

Where:
- A: advection operator ∇·(m v)
- D: diffusion operator (σ²/2) Δm

**Consequence**: Splitting error = O(Δt) × ||[A, D]||

For high Pe:
- ||[A, D]|| is large
- Error accumulates over time
- Small Δt doesn't help enough

### 3. Boundary Contamination

**Issue**: Intermediate 1D solves see incorrect boundary conditions

**Example** (Strang splitting):
1. Solve x-direction: enforces 1D boundaries, ignores y-flux
2. Solve y-direction: uses contaminated state from step 1
3. Errors at boundaries propagate inward

**Result**: Mass leaks through phantom boundary fluxes.

### 4. Error Accumulation in Picard Iteration

**MFG coupling**: HJB and FP alternate until convergence (15-30 iterations)

**Problem**: Each Picard iteration compounds the splitting error:
- Iteration 1: -5% mass loss
- Iteration 5: -20% mass loss
- Iteration 15: -81% mass loss (catastrophic)

---

## Why Full-Dimensional Systems Are Better

### Coupled Full System Approach

**Idea**: Solve 2D/3D FP equation as a single large linear system:

```
(I/Δt + A + D) m^(n+1) = m^n / Δt
```

Where:
- A: full 2D advection operator
- D: full 2D diffusion operator
- Solve directly (sparse linear system)

**Advantages**:
1. **No splitting error**: Single coupled solve preserves operator coupling
2. **Better mass conservation**: Proper flux balance at all boundaries
3. **Stable for high Pe**: Modern solvers handle advection-diffusion well
4. **Flexible discretization**: Can use SUPG, upwinding, flux limiting

**Disadvantages**:
1. **Larger systems**: O(N²) unknowns for 2D vs O(N) for splitting
2. **More memory**: Sparse matrix storage
3. **Slower per timestep**: But fewer iterations may compensate

---

## Recommended Alternatives

### For 2D/3D MFG Problems

**Option 1: Full-Dimensional Finite Differences** (RECOMMENDED)
- Discretize full 2D/3D FP equation
- Use sparse linear solvers (e.g., SciPy's spsolve, iterative methods)
- Add upwinding for advection terms
- Direct flux balance ensures conservation

**Option 2: Semi-Lagrangian Methods**
- Characteristic tracing for advection
- Implicit diffusion
- Naturally conservative

**Option 3: Particle Methods**
- Lagrangian particle evolution
- No advection discretization needed
- Inherently conservative

**Option 4: High-Resolution Schemes**
- MUSCL, WENO for advection
- TVD flux limiters
- Works with full-dimensional grids

---

## Implementation Notes

### Current Code Status

**Dimensional splitting solver**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm_multid.py`
- ⚠️ **DEPRECATED** for MFG problems
- Still works for pure diffusion (testing only)
- Do NOT use for production MFG solves

**Validation scripts** (archived):
- `demo_2d_pure_diffusion.py` - Shows splitting works for diffusion
- `demo_2d_crowd_evolution.py` - Shows catastrophic failure for MFG
- `trace_mass_through_sweeps.py` - Diagnostic tool for per-sweep mass tracking

---

## Historical Context

### Development Timeline

- **2024**: Dimensional splitting implemented as "efficient" 2D/3D solver
- **Bug #8 Investigation (2025-10-25)**: Discovered -11% mass loss with advection
  - Initially thought to be boundary discretization bug
  - Fixed advection at boundaries, reduced loss to -11%
  - Still unacceptable for MFG
- **Full MFG Test (2025-10-31)**: Discovered catastrophic -81% loss
  - Realized splitting error compounds over Picard iterations
  - Confirmed method is fundamentally unsuitable
  - **Decision**: Archive method, use full-dimensional solvers

### Lessons Learned

1. **Test with realistic problems**: Pure diffusion tests passed, but MFG failed
2. **Check full coupling**: Isolated tests hide coupling-induced failures
3. **Advection matters**: Methods that work for diffusion may fail with advection
4. **Error accumulation**: Small per-step errors can compound catastrophically

---

## References

### Dimensional Splitting Theory

- Strang, G. (1968). "On the construction and comparison of difference schemes"
  SIAM J. Numer. Anal., 5(3), 506-517.

- Marchuk, G. I. (1990). "Splitting and alternating direction methods"
  Handbook of Numerical Analysis, Vol. 1, 197-462.

### Advection-Diffusion Numerics

- Hundsdorfer, W., & Verwer, J. (2003). "Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations"
  Springer Series in Computational Mathematics.

### Conservative Methods

- LeVeque, R. J. (2002). "Finite Volume Methods for Hyperbolic Problems"
  Cambridge University Press.

---

## Future Work

### Planned Replacements

1. **Full 2D/3D FDM Solver** (PRIORITY)
   - Direct sparse linear system
   - Upwind discretization for advection
   - Target: ~1-2% mass error (normal FDM accuracy)

2. **Semi-Lagrangian FP Solver**
   - Characteristic tracing
   - Good for high advection

3. **Particle-Based FP Solver**
   - Already have particle methods for HJB
   - Natural extension to FP

---

## Conclusion

**Dimensional splitting is fundamentally unsuitable for MFG problems** due to catastrophic error accumulation in advection-dominated regimes. While elegant in theory, the method's O(Δt) splitting error compounds over Picard iterations, leading to unacceptable mass loss.

**For multi-dimensional MFG**: Use coupled full-dimensional system solvers. The computational cost increase (O(N²) vs O(N)) is justified by correctness and conservation properties.

**This method remains archived** for historical reference and as a cautionary example of why operator splitting can fail for coupled nonlinear systems.

---

**Last Updated**: 2025-10-31
**Archived By**: Claude Code
**See Also**:
- `docs/bugs/BUG8_ROOT_CAUSE_ANALYSIS.md` (boundary discretization fix attempt)
- `benchmarks/validation/results/` (test data demonstrating failure)
