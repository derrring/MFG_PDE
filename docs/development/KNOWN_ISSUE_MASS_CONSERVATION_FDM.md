# Known Issue: Mass Conservation in FP FDM Solver

**Status**: 🔴 CRITICAL BUG  
**Created**: 2025-10-05  
**Affects**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`  
**Tests Failing**: 21 mass conservation tests

## Problem Description

The Fokker-Planck Finite Difference Method (FP-FDM) solver exhibits severe mass loss/gain when solving the forward equation with no-flux boundary conditions. Mass can drop from 1.0 to 0.33 in a single time step.

### Reproduction

```python
from mfg_pde import ExampleMFGProblem, create_standard_solver
import numpy as np

problem = ExampleMFGProblem(Nx=20, Nt=8, T=0.5)
solver = create_standard_solver(problem, "fixed_point")  # Uses FP-FDM

result = solver.solve()

# Check mass at each time step
for t in range(problem.Nt + 1):
    mass = np.sum(result.M[t, :]) * problem.Dx
    print(f"t={t}: mass={mass:.6f}")

# Output shows: t=0: 1.000000 → t=1: 0.334419 (66% mass loss!)
```

## Root Cause Analysis

### Suspected Issue: No-Flux Boundary Discretization

The no-flux boundary condition implementation in `fp_fdm.py` (lines 148-210) uses:

**Left Boundary (i=0)**:
- Diagonal coefficient: `1/Dt + σ²/Dx²`
- Off-diagonal: `-σ²/Dx²`

This gives the discrete diffusion operator:
```
σ²(m_0 - m_1)/Dx²
```

**Correct Conservative Discretization**:
For no-flux boundary (∂m/∂x = 0), using the ghost point method with `m_{-1} = m_1`:

```
∂²m/∂x²|_{i=0} ≈ (m_1 - 2m_0 + m_{-1})/Dx² 
                = (m_1 - 2m_0 + m_1)/Dx²  (using m_{-1} = m_1)
                = 2(m_1 - m_0)/Dx²
```

The current implementation has:
1. **Wrong sign** in the flux
2. **Missing factor of 2** in the diffusion coefficient at boundaries
3. **Asymmetric stencil** that doesn't conserve mass

### Theoretical Background

For a conservative finite volume/difference scheme:
1. **Discrete flux conservation**: ∑ᵢ(flux_in - flux_out) = 0
2. **No-flux boundary**: flux at boundary = 0
3. **Interior conservation**: flux(i→i+1) + flux(i-1→i) = diffusion + advection terms

The current discretization violates these principles at boundaries.

## Impact

### Failing Tests (21 total)

**Mathematical Tests** (`tests/mathematical/test_mass_conservation.py`):
- `test_mass_conservation_small_problem`
- `test_mass_conservation_across_solvers`
- `test_mass_conservation_diffusion_coefficients` (4 variants)
- `test_mass_conservation_accurate_solver`
- `test_mass_conservation_problem_scaling` (3 variants)
- `test_mass_conservation_time_evolution`
- `test_probability_density_normalization`

**Integration Tests** (`tests/integration/test_mass_conservation_1d*.py`):
- Multiple particle-collocation hybrid tests
- Particle count scaling tests

### Severity

- 🔴 **Correctness**: Core algorithm produces physically invalid solutions
- 🔴 **Reliability**: Results cannot be trusted for long-time integration
- 🟡 **Performance**: Solver may not converge due to mass violation

## Workarounds

### Immediate: Use Particle Methods

Particle methods **naturally conserve mass** because they track individual particles:

```python
# Instead of FDM (default in create_standard_solver):
solver = create_standard_solver(problem, solver_type="particle_collocation")
```

### Alternative: Explicit Mass Renormalization (Temporary Hack)

```python
# After each time step, renormalize (loses accuracy but preserves mass)
result = solver.solve()
for t in range(result.M.shape[0]):
    result.M[t, :] /= (np.sum(result.M[t, :]) * problem.Dx)
```

## Recommended Fix

### Short-term (1-2 days)

1. **Update no-flux boundary stencil** in `fp_fdm.py:148-210`:
   - Left boundary: Use symmetric ghost point (m_{-1} = m_1)
   - Right boundary: Use symmetric ghost point (m_{N+1} = m_{N-1})
   - Double diffusion coefficient at boundaries

2. **Add mass conservation check** after each time step:
   ```python
   if abs(np.sum(m_new) - np.sum(m_old)) > 1e-10:
       logger.warning(f"Mass violation at step {k}: {mass_error}")
   ```

3. **Update tests** to use particle methods or mark FDM tests as `xfail`

### Long-term (1-2 weeks)

1. **Implement conservative finite volume scheme**:
   - Use flux-based formulation
   - Guarantee discrete conservation by construction
   - Support multiple boundary conditions (Dirichlet, Neumann, Robin)

2. **Add analytical test cases** with known mass-conserving solutions

3. **Benchmark** against particle methods for validation

## References

- LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems
- Morton, K. W., & Mayers, D. F. (2005). Numerical Solution of Partial Differential Equations
- Relevant code: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py` lines 148-250

## Related Issues

- Issue #76: Test Suite Health (21 mass conservation tests failing)
- PRs #74-79: Infrastructure fixes (completed)

---

**Next Action**: Either fix FDM discretization OR mark FDM solver as experimental and recommend particle methods for production use.
