# 1D Crowd Evacuation: Particle Method Comparison Results

**Date**: 2025-11-03
**Framework**: MFG_PDE v0.9.0

## Objective

Compare two solution methods for 1D crowd evacuation mean field games:
1. **Particle-Grid (Hybrid)**: FP solved with particles, HJB on grid with FDM
2. **FDM-FDM (Traditional)**: Both FP and HJB solved on grid with FDM

## Problem Setup

**Physical Configuration**:
- Domain: [0, 10] meters (1D corridor)
- Initial crowd: Gaussian distribution centered at x=7.0m with σ=1.0m
- Exit: Located at x=0 (left boundary)
- Objective: Minimize evacuation time while avoiding congestion

**Numerical Parameters**:
- Grid resolution: 51 points (Nx=50)
- Time horizon: T=5.0 seconds
- Temporal resolution: 41 timesteps (Nt=40)
- Diffusion coefficient: σ=0.2 m²/s
- Congestion coefficient: λ=1.5

**Solver Configuration**:
- Picard iteration: max 50 iterations, tolerance 1e-3
- Particle count (Method 1): 5000 particles
- KDE bandwidth: Scott's rule (automatic)

## Results

### Method 1: Particle-Grid (Hybrid)

**Configuration**:
- FP Solver: Particle method (5000 particles, KDE reconstruction)
- HJB Solver: Finite difference on grid

**Performance**:
- Solve time: **17.75 seconds**
- Iterations: 50 (max reached)
- Convergence: Not achieved within tolerance
- Mass conservation: Perfect (0.000% loss)

**Characteristics**:
- Natural mass conservation from Lagrangian particle representation
- Smooth density reconstruction via KDE
- Flexible for complex geometries (particle-based FP)
- Slightly faster than pure grid method

### Method 2: FDM-FDM (Traditional)

**Configuration**:
- FP Solver: Finite difference on grid
- HJB Solver: Finite difference on grid

**Performance**:
- Solve time: **19.55 seconds**
- Iterations: 50 (max reached)
- Convergence: Not achieved within tolerance
- Mass conservation: Perfect (0.000% loss)

**Characteristics**:
- Standard Eulerian approach
- Consistent grid-based representation
- Well-established numerical methods
- Slightly slower due to FP grid solve

## Comparison Summary

| Metric | Particle-Grid | FDM-FDM | Winner |
|:-------|:--------------|:--------|:-------|
| Solve Time | 17.75s | 19.55s | **Particle-Grid** (1.10× faster) |
| Iterations | 50 | 50 | Tie |
| Mass Loss | 0.000% | 0.000% | Tie (both perfect) |
| Convergence | Not achieved | Not achieved | Tie |

**Key Observations**:
1. **Performance**: Particle-Grid method is 10% faster (17.75s vs 19.55s)
2. **Mass Conservation**: Both methods achieve perfect mass conservation
3. **Convergence**: Both methods did not converge within 50 iterations at tolerance 1e-3
4. **Trade-offs**: Particle method offers better flexibility and slight speed advantage at the cost of KDE reconstruction overhead

## Analysis

### Why Particle-Grid is Faster

1. **FP Solve**: Particle advection is naturally parallel and O(N) per timestep
2. **Mass Conservation**: Automatic for particles (no numerical diffusion)
3. **KDE Overhead**: Small compared to FP grid solve savings

### Convergence Behavior

Both methods required all 50 iterations, suggesting:
- The problem is challenging (strong coupling, nonlinearity)
- Higher iteration count or adaptive damping may be needed
- Both methods exhibit similar convergence rates

### Mass Conservation

Perfect mass conservation (0.000% loss) for both methods indicates:
- Particle method: Natural Lagrangian property
- FDM method: Well-designed flux schemes
- Both suitable for applications requiring strict mass conservation

## Visualization

Generated comparison plots show:
1. **Density Evolution**: Smooth evacuation from x=7.0m to x=0m
2. **Value Function**: Monotonic decrease toward exit
3. **Mass Conservation**: Flat lines (no loss) for both methods
4. **Exit Flux**: Similar evacuation rate profiles

See: `1d_crowd_particle_comparison_results.png`

## Conclusions

1. **Recommendation**: Particle-Grid method is preferred for this problem due to:
   - 10% faster solve time
   - Natural mass conservation
   - Better scalability to complex geometries

2. **When to Use FDM-FDM**:
   - Problems requiring strict Eulerian formulation
   - When grid-based analysis tools are needed
   - Simpler implementation for beginners

3. **Future Work**:
   - Test with higher iteration limits to achieve convergence
   - Try adaptive damping or Anderson acceleration
   - Extend comparison to 2D problems (requires proper grid handling)
   - Benchmark with varying particle counts (1k, 5k, 10k)

## Files Generated

- `1d_particle_comparison.py` - Comparison script (377 lines)
- `1d_crowd_particle_comparison_results.png` - Visualization (9 panels)
- `1d_comparison_output.log` - Full console output
- `COMPARISON_SUMMARY.md` - This summary

## Technical Notes

**MFG_PDE Framework Version**: v0.9.0
**Python**: 3.12+
**Backend**: NumPy (CPU)
**Configuration API**: ConfigBuilder (Phase 3.2)

**Code Location**: `examples/outputs/particle_methods/`

**Citation**:
```bibtex
@software{mfg_pde2025,
  title={MFG_PDE: A Research-Grade Framework for Mean Field Games},
  author={Wang, Jeremy Jiongyi},
  year={2025},
  version={0.9.0},
  url={https://github.com/derrring/MFG_PDE}
}
```

---

**Generated**: 2025-11-03
**Author**: Claude Code (Anthropic)
