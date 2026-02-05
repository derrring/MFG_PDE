# Terminal-BC Compatibility for GFDM Solvers

**Status**: [ANALYSIS] Completed 2025-12-20

## Issue Summary

GFDM-based MFG solvers produce oscillatory boundary values and reversed drift when the terminal cost is incompatible with boundary conditions.

## Root Cause

The standard crowd evacuation terminal cost:
```
g(x,y) = 0.5 * ((x - exit_x)^2 + (y - exit_y)^2)
```

Has gradient dg/dx = (x - exit_x). At the left wall (x=0, exit_x=20):
- dg/dx|_{x=0} = -20

But the Neumann BC requires:
- dg/dn|_{boundary} = 0

This creates an algebraically impossible system at t=T:
- PDE residual requires non-zero gradient (from terminal cost)
- BC constraint requires zero normal derivative

### Why FDM Works but GFDM Fails

| Method | BC Enforcement | Conflict Resolution |
|--------|----------------|---------------------|
| FDM    | Implicit in upwind stencil | Numerical viscosity smears discontinuity |
| GFDM   | Explicit row replacement in Jacobian | Newton can't satisfy both constraints |

## Evidence

1. **Newton non-convergence**: |delta_u| = 10.0 (max step limit) every iteration
2. **Gradient oscillation buildup**:
   - Terminal std = 0.07
   - After 5 steps std = 0.17
   - After 60 steps std = 23.7
3. **Velocity reversal**:
   - FDM: v_x = +6.7 (correct, toward exit)
   - GFDM: v_x = -22.3 (incorrect, away from exit)

## Solution: BC-Compatible Terminal Cost

Apply a "flat buffer" near boundaries to ensure dg/dn = 0:

```python
def create_terminal_cost(config, x_grid, y_grid, bc_compatible=False, boundary_buffer=0.5):
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    if bc_compatible:
        # Clamp coordinates to create flat regions near walls
        X_clamped = np.clip(X, boundary_buffer, Lx - boundary_buffer)
        Y_clamped = np.clip(Y, boundary_buffer, Ly - boundary_buffer)
        dist_sq = (X_clamped - exit_x)**2 + (Y_clamped - exit_y)**2
    else:
        dist_sq = (X - exit_x)**2 + (Y - exit_y)**2

    return 0.5 * dist_sq
```

### Physical Interpretation

For agents within `boundary_buffer` of the wall:
- No local gradient pushing them (drift = 0)
- Brownian noise (diffusion) naturally moves them past the buffer
- Once past the buffer, the gradient kicks in and pulls toward exit

### Expected Result

- Newton converges in 3-4 iterations (instead of hitting max iterations)
- No oscillatory boundary values
- Correct drift direction toward exit

### Experimental Result (2025-12-20)

**Hard clamping approach failed in full MFG coupling.**

| Metric | GFDM-LCR Original | GFDM-LCR BC-Compat |
|--------|-------------------|---------------------|
| Boundary U std | 41.9 | 57.5 (worse) |
| U range | [-9.0, 212.5] | [-71.7, 167.4] (worse) |
| Displacement | 48.2% of FDM | 46.3% of FDM (worse) |

**Why it failed**: The hard clamp (`np.clip`) creates a C0 (continuous but non-smooth) terminal cost. The gradient discontinuity at the buffer boundary propagates backward in time, amplifying oscillations rather than suppressing them.

**Next steps**:
1. ~~Use smooth transition (e.g., tanh-based blending) instead of hard clamp~~ (Testing)
2. Consider larger buffer with smooth edges
3. Investigate QP-based boundary handling (experiments running)
4. **Ghost Node Method (recommended by expert)** - see below

## Expert Recommendation: Ghost Nodes (Method of Images)

The expert analysis identifies that GFDM is sensitive to **curvature (∂²g/∂x²)**, not just slope:

> The "Flat Buffer" failed because GFDM is hypersensitive to the C² discontinuity (curvature jump) introduced at the buffer edge.

### Why Ghost Nodes Work

Instead of modifying the terminal cost (which creates kinks), ghost nodes enforce Neumann BC **structurally**:

1. **Mirror nodes**: For boundary node at (ε, y), create ghost node at (-ε, y)
2. **Neumann rule**: Set ghost value = interior value (u_ghost = u_interior)
3. **Symmetric stencil**: Boundary node now has neighbors on both sides
4. **Natural smoothness**: Original parabolic cost preserved, no curvature shocks

### Key Insight

The original terminal cost g(x,y) = 0.5((x-20)² + (y-5)²) has natural smoothness properties. By using ghost nodes, we:
- Keep the terminal cost as-is
- Modify only the STENCIL structure (not the data)
- Allow boundary nodes to behave like interior nodes with symmetric neighbors

### Implementation Approach

1. **Modify neighbor search**: For boundary nodes, mirror interior neighbors across boundary
2. **Modify system matrix**: Ghost node weights combined with corresponding interior node weights
3. **No terminal cost changes needed**

This is the "gold standard" for meshless Neumann problems because it satisfies the BC through geometry, not by forcing the function to be flat

## Related Issues

1. **LCR 180-degree rotation cancellation**: For axis-aligned walls, the LCR rotation matrix is its own inverse, so rotation cancels during rotate-back. This is not a bug but limits LCR effectiveness for rectangular domains.

2. **Geometric degeneracy (secondary)**: For complex geometries with poor angular coverage, sector-based neighbor selection may also be needed.

## Implementation

Added `bc_compatible` flag to `create_terminal_cost()` in:
- `mfg-research/experiments/crowd_evacuation_2d/experiments/exp1_baseline_fdm.py`

New solver variant:
- `gfdm_lcr_compat_fdm`: Uses BC-compatible terminal cost

## References

- Expert analysis on Terminal-BC Incompatibility (parabolic PDE compatibility conditions)
- GFDM literature on row replacement for boundary conditions
- Numerical viscosity in upwind schemes

## Files Modified

- `experiments/crowd_evacuation_2d/experiments/exp1_baseline_fdm.py`: Added `bc_compatible` parameter
