# Mass-Conservative FDM Treatment for Fokker-Planck Equation

**Context**: Analysis of numerical treatments to achieve mass conservation in FP-FDM solver  
**Date**: 2025-10-05  
**For**: Hybrid FP-FDM + HJB-FDM solver (alternative to particle methods)

## Mathematical Foundation

### Continuous Conservation Law

Fokker-Planck equation with no-flux boundaries:
```
∂m/∂t = σ²∂²m/∂x² - ∂(m·v)/∂x,  where v = -α·∂u/∂x
```

No-flux BC: `∂m/∂x|_{x=0,L} = 0`

**Continuous mass conservation:**
```
d/dt ∫₀ᴸ m(t,x) dx = ∫₀ᴸ ∂m/∂t dx 
                    = [σ²∂m/∂x - m·v]₀ᴸ  (integration by parts)
                    = 0  (by no-flux BC)
```

### Discrete Conservation Requirement

For discrete scheme: `m^{k+1} = A⁻¹ b` where `b = m^k/Dt`

**Discrete mass conservation:**
```
∑ᵢ m^{k+1}_i = ∑ᵢ m^k_i  for all k
```

**Necessary condition:**
```
∑ⱼ (A⁻¹)ᵢⱼ = constant for all i  (row sums of A⁻¹)
```

**Equivalent condition:**
```
A·𝟙 = c·𝟙  for some constant c, where 𝟙 = [1,1,...,1]ᵀ
```

This means: **summing each row of A gives the same value**.

## Current Implementation Analysis

### Interior Points (i = 1, ..., N-2)

**Diffusion part:**
```
Diagonal:    1/Dt + σ²/Dx²
Off-diag:    -σ²/(2Dx²)  (both neighbors)
Row sum:     1/Dt + σ²/Dx² - 2·σ²/(2Dx²) = 1/Dt  ✓
```

**Advection part (upwind):**
```
Uses npart(·) and ppart(·) for upwinding
Row sum contribution should cancel if flux-conservative
```

### Boundary Points (Current Implementation)

**Left boundary (i=0):**
```
Diagonal:    1/Dt + σ²/Dx²
Right:       -σ²/Dx²
Row sum:     1/Dt + σ²/Dx² - σ²/Dx² = 1/Dt  ✓
```

**Right boundary (i=N-1):**
```
Diagonal:    1/Dt + σ²/Dx²
Left:        -σ²/Dx²  
Row sum:     1/Dt + σ²/Dx² - σ²/Dx² = 1/Dt  ✓
```

**Problem identified**: The diffusion part has consistent row sums! But advection at boundaries is missing.

## Root Cause: Missing Advection at Boundaries

**Line 157 in current code:**
```python
# For advection terms, assume no velocity at boundary
```

This is **WRONG**! The velocity is not zero at boundaries - it's determined by ∂u/∂x.

**Correct treatment**: Apply upwind advection at boundaries too, but with **one-sided derivative for flux**.

## Recommended Treatments

### Treatment 1: Finite Volume Conservative Form (RECOMMENDED)

**Idea**: Discretize conservation law directly using flux formulation.

**For interior cell i:**
```
(m^{k+1}_i - m^k_i)/Dt = (F_{i-1/2} - F_{i+1/2})/Dx
```

where flux at interface i+1/2:
```
F_{i+1/2} = -σ²(m_{i+1} - m_i)/Dx + v_{i+1/2}·m_{i+1/2}
```

**For boundaries**: Set `F_{-1/2} = 0` and `F_{N+1/2} = 0` (no-flux).

**Mass conservation**: Summing over all cells:
```
∑ᵢ (m^{k+1}_i - m^k_i)/Dt = (F_{-1/2} - F_{N+1/2})/Dx = 0  ✓
```

**Implementation**:
```python
# At left boundary (i=0):
# No flux from left: F_{-1/2} = 0
# Flux to right: F_{1/2} = -σ²(m_1 - m_0)/Dx + ...
# Update: (m^{k+1}_0 - m^k_0)/Dt = -F_{1/2}/Dx

# At right boundary (i=N-1):
# Flux from left: F_{N-3/2}
# No flux to right: F_{N+1/2} = 0
# Update: (m^{k+1}_{N-1} - m^k_{N-1})/Dt = F_{N-3/2}/Dx
```

**Advantages**:
- Guarantees discrete conservation by construction
- Well-established theory (LeVeque 2002)
- Natural flux limiting for positivity

### Treatment 2: Correct Boundary Advection (SIMPLER)

**Fix the current implementation**: Add advection terms at boundaries.

**Left boundary (i=0):**
```python
# Diffusion (current - OK)
val_A_ii = 1.0 / Dt + sigma**2 / Dx**2
val_A_i_ip1 = -sigma**2 / Dx**2

# ADD: Advection using one-sided derivative
# v_0 = -coupling_coefficient * (u_1 - u_0)/Dx
# Upwind flux: if v_0 > 0, flux from left (ghost) = 0
#              if v_0 < 0, flux = v_0 * m_0
val_A_ii += coupling_coefficient * ppart((u_1 - u_0)/Dx) / Dx  # outflow
```

**Check row sum conservation**:
```
For boundary to conserve mass, need:
∑ⱼ A_ij = 1/Dt  (same as interior)
```

### Treatment 3: Projection Method (POST-PROCESSING)

**Idea**: After each time step, project solution to conserve mass.

```python
# After solving A·m^{k+1} = b
total_mass_k = np.sum(m_k) * Dx
total_mass_k1 = np.sum(m_k1) * Dx

# Renormalize
m_k1 *= total_mass_k / total_mass_k1
```

**Advantages**:
- Simple to implement
- Works with any discretization

**Disadvantages**:
- Loses accuracy
- Not conservative in L² sense
- Band-aid solution

## Recommended Implementation for MFG_PDE

### Short-term (1-2 days): Treatment 2

Update `fp_fdm.py` lines 152-167 and 169-183:

```python
if i == 0:
    # Left boundary
    val_A_ii = 1.0 / Dt + sigma**2 / Dx**2
    
    # Advection: one-sided derivative for velocity
    if Nx > 1:
        v_0 = -coupling_coefficient * (u_at_tk[1] - u_at_tk[0]) / Dx
        # Upwind: outflow from cell 0
        val_A_ii += coupling_coefficient * ppart(v_0) / Dx
        # Inflow to cell 0 (from boundary, zero by no-flux)
    
    row_indices.append(i)
    col_indices.append(i)
    data_values.append(val_A_ii)
    
    if Nx > 1:
        val_A_i_ip1 = -sigma**2 / Dx**2
        # Advection: flux to neighbor
        v_0 = -coupling_coefficient * (u_at_tk[1] - u_at_tk[0]) / Dx
        val_A_i_ip1 += -coupling_coefficient * npart(v_0) / Dx
        row_indices.append(i)
        col_indices.append(i + 1)
        data_values.append(val_A_i_ip1)
```

**Validation**: After implementing, check `np.sum(A @ np.ones(Nx))` ≈ `Nx / Dt`.

### Long-term (1-2 weeks): Treatment 1

Rewrite FP-FDM using finite volume formulation:
- Define flux function at interfaces
- Use conservative update
- Add flux limiters for positivity
- Implement TVD schemes for shocks

## Theoretical References

1. **LeVeque, R. J. (2002)**. *Finite Volume Methods for Hyperbolic Problems*  
   Chapter 6: Conservation and Stability
   
2. **Morton & Mayers (2005)**. *Numerical Solution of PDEs*  
   Chapter 4: Finite Difference Methods for Parabolic Equations

3. **Toro, E. F. (2009)**. *Riemann Solvers and Numerical Methods*  
   Chapter 5: Conservative Finite Volume Methods

4. **Cardaliaguet et al. (2019)**. "The Master Equation and the Convergence Problem in Mean Field Games"  
   Appendix: Numerical Schemes for FP equation

## Testing Strategy

### Unit Test: Mass Conservation

```python
def test_fdm_mass_conservation():
    """Test that FDM conserves mass with no-flux BC."""
    problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)
    fp_solver = FPFDMSolver(problem)
    
    # Mock value function (smooth)
    U = np.zeros((problem.Nt + 1, problem.Nx + 1))
    
    result = fp_solver.solve_fp_system(problem.m_init, U)
    
    # Check mass at each time step
    for t in range(problem.Nt + 1):
        mass = np.sum(result[t, :]) * problem.Dx
        assert abs(mass - 1.0) < 1e-10, f"Mass not conserved at t={t}"
```

### Analytical Test: Diffusion Only

For pure diffusion (no advection), analytical solution exists.  
Compare FDM with exact solution to validate discretization.

## Conclusion

**For our MFG case** (FP-FDM + HJB-FDM):

1. **Best short-term**: Add advection terms at boundaries (Treatment 2)
   - Maintains current structure
   - Achieves approximate conservation
   - 1-2 days implementation

2. **Best long-term**: Finite volume formulation (Treatment 1)
   - Guaranteed conservation
   - More robust
   - Industry standard for conservation laws
   - 1-2 weeks implementation

3. **Not recommended**: Projection/renormalization (Treatment 3)
   - Loses accuracy
   - Doesn't address root cause

**Current recommendation**: Use particle methods (as implemented in PR #80) for production. Implement conservative FDM as research/validation tool.
