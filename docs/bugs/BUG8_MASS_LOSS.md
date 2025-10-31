# Bug #8: Mass Loss During FP Evolution

**Date Discovered**: 2025-10-31 (during Bug #7 verification)
**Status**: Identified, awaiting investigation
**Severity**: High - Violates fundamental conservation law
**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py` (likely)

---

## Problem Summary

The Fokker-Planck (FP) solver loses approximately 90% of the total mass during time evolution, violating the fundamental conservation property of the continuity equation.

**Observed Behavior**:
- Initial mass: 1.000000
- Final mass: 0.095391
- Mass loss: 90.46%

**Expected Behavior**:
- Mass should be conserved to machine precision (∫ m(t,x) dx = constant)
- Acceptable loss: < 1e-10 (relative)

---

## Mathematical Background

### The Fokker-Planck Equation as a Conservation Law

```
∂m/∂t + ∇·J = 0
```

where the flux is:
```
J = m·v(∇u) - (σ²/2)∇m
```

This is a **conservative form** - the total integral of m should be preserved:
```
d/dt ∫ m(t,x) dx = ∫ ∂m/∂t dx = -∫ ∇·J dx = 0  (by divergence theorem with proper BCs)
```

### Discretization Requirements

For mass conservation in FDM:
1. **Flux-based formulation**: Use conservative differencing
2. **Boundary conditions**: No-flux (Neumann) or periodic
3. **Summation-by-parts**: Discrete divergence should sum to zero

---

## Discovery Context

This bug was discovered while verifying the Bug #7 fix (time-index mismatch). The convergence test showed:

```
✓ SUCCESS: Solver converged in 18 iterations
  Bug #7 fix appears to have resolved the oscillation issue

Mass conservation:
  Initial: 1.000000
  Final: 0.095391
  Loss: 90.46%
```

**Important**: The solver converges despite the mass loss, indicating this is a **separate issue** from the Bug #7 coupling problem.

---

## Potential Root Causes

### 1. Boundary Condition Implementation (Most Likely)

**Hypothesis**: Homogeneous Dirichlet boundary conditions (m=0 at boundaries) cause mass to "leak" out of the domain instead of being reflected or preserved.

**Evidence**:
- FP solver uses sparse matrix A with boundary rows
- Current BC implementation may not preserve flux
- 90% loss suggests systematic boundary leakage, not numerical diffusion

**Check**:
```python
# In fp_fdm.py or fp_fdm_multid.py
# Look for boundary condition setup in matrix construction
```

### 2. Flux Discretization Error

**Hypothesis**: The discrete divergence operator doesn't satisfy summation-by-parts property.

**Evidence**:
- Interior flux terms: `val_A_ii`, `val_A_i_im1`, `val_A_i_ip1`
- If these don't sum correctly, discrete mass may not be conserved

**Mathematical Test**:
```python
# The sum of each row in the FP system matrix should equal zero
# for mass conservation (excluding time derivative term)
sum_j A[i,j] = 1/dt  (only time derivative contributes)
```

### 3. Density Advection Term

**Hypothesis**: The coupling term `m·v(∇u)` is incorrectly discretized.

**Evidence**:
- Uses `npart()` and `ppart()` functions for flux splitting
- If splitting is wrong, mass can accumulate or disappear

**Location**:
```python
# fp_fdm.py:305-327
val_A_ii += float(coefCT * (npart(...) + ppart(...)) / Dx**2)
```

---

## Diagnostic Plan

### Step 1: Verify Boundary Conditions

**Action**: Check what boundary conditions are applied to m

```python
# Examine fp_fdm.py boundary setup
# Expected: No-flux (∇m·n = 0) or periodic
# Problematic: Dirichlet (m = 0)
```

**Test**: Run simple 1D advection with known BC and check mass

### Step 2: Test Mass Conservation Property

**Script**:
```python
# test_mass_conservation.py
import numpy as np
from benchmarks.validation.test_2d_crowd_motion import CrowdMotion2D
from mfg_pde.factory import create_basic_solver

problem = CrowdMotion2D(grid_resolution=16, time_horizon=0.4, num_timesteps=20)
solver = create_basic_solver(problem, max_iterations=50)
result = solver.solve()

dx, dy = problem.geometry.grid.spacing
dV = dx * dy

# Check mass at each timestep
for t_idx in range(len(result.M)):
    mass_t = np.sum(result.M[t_idx]) * dV
    print(f't={t_idx}: mass={mass_t:.8f}')
```

**Expected Output**: Mass constant across all timesteps
**Observed Output**: Mass decreasing monotonically

### Step 3: Examine FP System Matrix

**Action**: Extract and analyze the FP system matrix structure

```python
# In fp_fdm.py, after matrix construction:
row_sums = A.sum(axis=1)  # Should be 1/dt for all rows
print(f'Row sums: min={row_sums.min()}, max={row_sums.max()}')
```

---

## Testing Plan

1. **Minimal test**: 1D problem, 8 grid points, check mass evolution
2. **Boundary test**: Compare Dirichlet vs no-flux boundary conditions
3. **Matrix test**: Verify discrete conservation property of FP operator
4. **Regression test**: Ensure fix doesn't break convergence

---

## Related Issues

- **Bug #7** (time-index mismatch): FIXED - This enabled convergence, which revealed Bug #8
- **Bug #1** (initial density normalization): FIXED - Ensures initial mass = 1.0

---

## Root Cause Analysis - COMPLETED

### Matrix Analysis Results

Analyzed FP discretization matrix for 1D problem with no-flux boundaries:

**Finding**: Boundary rows have incorrect sum!
- **Expected row sum (1/Δt)**: 100.0
- **Boundary rows (i=0, i=N-1)**: 116.67 → **16.67% error**
- **Interior rows**: 100.0 ✓ (correct)

**Test**: Uniform density → 1 timestep
- Mass loss: -4.40% in 1D
- Observed: ~12.6% in 2D (dimensional splitting compounds errors)

### Mathematical Issue

For mass conservation, discrete FP operator must satisfy:
```
Σⱼ Aᵢⱼ = 1/Δt  (for all rows i)
```

**Current boundary discretization** (lines 254-301 in `fp_fdm.py`):
- Adds diffusion term: `val_A_ii += σ²/Δx²`
- Adds advection term using upwind flux splitting
- **But the off-diagonal terms don't cancel the extra diagonal**

Result: Boundary rows sum to `1/Δt + σ²/Δx²` instead of `1/Δt`

### Impact

- **1D problem**: ~5% mass loss per timestep (1 boundary pair)
- **2D problem with splitting**: ~12% per timestep (x-sweep + y-sweep, each with 2 boundaries)
- **Accumulation**: After N timesteps, mass = initial × (1 - loss_rate)^N

### Location of Bug

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`
**Lines**: 254-301 (boundary discretization for no-flux)

**Problem**: The boundary discretization doesn't properly balance diffusion and advection fluxes to achieve conservative form.

---

## Proposed Fix

The no-flux boundary discretization needs to be revised to ensure row sums equal `1/Δt`.

**Option 1**: Use ghost cells with symmetric reflection
```python
# At left boundary (i=0):
# Extend m[-1] = m[1] (reflection for dm/dx = 0)
# Then use standard interior stencil
```

**Option 2**: Reformulate boundary terms
```python
# Ensure diagonal and off-diagonal sum correctly:
# val_A_ii - val_A_i_ip1 = 1/Δt (at left boundary)
# val_A_ii - val_A_i_im1 = 1/Δt (at right boundary)
```

**Option 3**: Directly enforce no-flux as constraint
```python
# Set boundary flux J = 0
# This gives: m·v - (σ²/2)∇m = 0 at boundaries
# Discretize this condition directly
```

---

## Test Data

**Generated Files**:
- `bug8_mass_loss_data.npz` - Mass evolution data (times, masses, densities)
- `bug8_mass_loss_plots.png` - Visualization of mass loss
- `analyze_fp_matrix.log` - Matrix analysis output
- `trace_mass_loss.log` - Timestep-by-timestep mass tracking

**Key Findings**:
- Mass loss rate is **constant** (not accumulating or decreasing)
- Loss occurs at **every timestep** (systematic discretization error)
- Interior discretization is **correct** (verified via row sums)
- Boundary discretization is **incorrect** (16.67% row sum error)

---

## Next Steps

1. ✓ **Identify root cause**: Boundary row sums incorrect
2. **Design fix**: Reformulate boundary discretization for conservative form
3. **Implement fix**: Update `fp_fdm.py` lines 254-301
4. **Test fix**: Verify mass conservation to machine precision
5. **Regression test**: Ensure convergence still works

---

**Status**: Root cause identified. Fix design in progress.
**Priority**: High (violates conservation law, but solver still converges)
