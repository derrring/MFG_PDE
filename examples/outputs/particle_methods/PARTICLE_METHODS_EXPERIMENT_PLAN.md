# Particle Methods Experiments for 2D MFG

**Date**: 2025-11-03
**Objective**: Compare different solver combinations for 2D crowd evacuation

## Experiment Design

### Problem Setup

**Physical Configuration**:
- Domain: 10m × 10m square room
- Initial crowd: Gaussian at (5.0, 7.0) with σ=1.5m
- Exit: Bottom boundary at y=0
- Objective: Minimize evacuation time while avoiding congestion

**Dynamics**:
- Hamiltonian: H = (1/2)|p|² + λ·m·|p|² (kinetic + congestion)
- Diffusion: σ = 0.2 m²/s
- Congestion weight: λ = 2.0
- Time horizon: T = 8.0 seconds
- Timesteps: 40

**Numerical Setup**:
- Grid resolution: 15×15 (256 points) for faster testing
- Max Picard iterations: 20
- Damping factor: 0.5
- Tolerance: 1e-4

## Solver Combinations

### 1. Grid-Grid (FDM-FDM)
**HJB**: Finite Difference Method on grid
**FP**: Finite Difference Method on grid

**Characteristics**:
- Standard approach for grid-based MFG
- Both equations solved on same grid
- Known mass conservation issues in 2D
- Fast per-iteration for moderate grids

**Expected Performance**:
- ⚠️ Poor mass conservation (typically 20-35% loss in 2D)
- ✅ Fast solve time
- ✅ Good convergence with proper parameters

### 2. Grid-Particle (FDM-Particle, 3000 particles)
**HJB**: Finite Difference Method on grid
**FP**: Particle method with KDE reconstruction

**Characteristics**:
- Hybrid approach combining strengths
- HJB on grid (fast, accurate gradients)
- FP with particles (perfect mass conservation)
- Particle count affects accuracy vs speed

**Expected Performance**:
- ✅ Perfect mass conservation (particle-based FP)
- ⚠️ Slower than pure grid (KDE reconstruction overhead)
- ✅ More accurate density representation

### 3. Grid-Particle (FDM-Particle, 5000 particles)
**HJB**: Finite Difference Method on grid
**FP**: Particle method with more particles

**Characteristics**:
- Same as (2) but with higher particle count
- Better density resolution
- More computational cost

**Expected Performance**:
- ✅ Perfect mass conservation
- ✅ Better density accuracy than 3k particles
- ⚠️ Slowest method (more particles + KDE)

### 4. Particle-Particle (Future)
**HJB**: Particle method
**FP**: Particle method

**Characteristics**:
- Fully Lagrangian approach
- No grid needed
- Challenging for HJB (requires gradient estimation)

**Status**: Not yet implemented in framework

## Metrics

### Primary Metrics

1. **Convergence**: Did Picard iteration converge within tolerance?
2. **Solve Time**: Total computation time (seconds)
3. **Iterations**: Number of Picard iterations until convergence
4. **Mass Conservation**: |1 - ∫m(T,x)dx| as percentage

### Secondary Metrics

5. **Evacuation Progress**: Movement of center of mass toward exit
6. **Density Quality**: Smoothness and physical realism
7. **Computational Efficiency**: Time per iteration

## Hypotheses

### H1: Mass Conservation
**Hypothesis**: Particle-based FP will achieve significantly better mass conservation than grid-based FP.

**Expected**:
- Grid-Grid: 20-35% mass loss
- Grid-Particle (3k): < 0.1% mass loss
- Grid-Particle (5k): < 0.05% mass loss

**Rationale**: Particle methods naturally conserve mass; grid-based FDM accumulates numerical diffusion.

### H2: Computational Cost
**Hypothesis**: Pure grid methods will be fastest, hybrid methods slower, more particles = slower.

**Expected Ranking** (fastest to slowest):
1. Grid-Grid (FDM-FDM)
2. Grid-Particle (3k particles)
3. Grid-Particle (5k particles)

**Rationale**: KDE reconstruction and particle advection add overhead.

### H3: Accuracy vs Speed Trade-off
**Hypothesis**: Grid-Particle with enough particles offers best accuracy/speed balance.

**Expected**: Grid-Particle (3k) achieves similar accuracy to Grid-Particle (5k) at lower cost.

**Rationale**: Diminishing returns from additional particles for this grid resolution.

## Implementation Details

### Using GridBasedMFGProblem Pattern

All experiments use the proper nD infrastructure:

```python
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem

class CrowdEvacuation2D(GridBasedMFGProblem):
    def __init__(self, grid_resolution=15, ...):
        super().__init__(
            domain_bounds=(0.0, 10.0, 0.0, 10.0),
            grid_resolution=grid_resolution,
            time_domain=(time_horizon, num_timesteps),
            diffusion_coeff=diffusion_coeff,
        )

    def hamiltonian(self, x, m, p, t):
        """Standard signature: H(x, m, p, t)"""
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        return 0.5 * p_squared + self.congestion_weight * m * p_squared
```

### Solver Configuration

**Grid-Grid**:
```python
solver = create_basic_solver(problem, damping=0.5, max_iterations=20)
```

**Grid-Particle**:
```python
config = ConfigBuilder.for_problem(problem) \
    .with_hjb_fdm_solver() \
    .with_fp_particle_solver(num_particles=num_particles) \
    .with_picard_iteration(max_iterations=20, damping_factor=0.5) \
    .build()
solver = PicardIterationSolver(problem, config)
```

## Expected Output

### Visualizations

1. **Solve Time Comparison**: Bar chart with colors indicating convergence
2. **Iteration Count**: Number of Picard iterations per method
3. **Mass Conservation**: Bar chart with 1% threshold line
4. **Summary Table**: Comprehensive results matrix

### Files Generated

- `particle_methods_comparison.png` - 4-panel comparison figure
- `particle_comparison_output.log` - Detailed execution log
- `PARTICLE_METHODS_RESULTS.md` - Analysis and conclusions

## Success Criteria

### Minimum Success
- ✅ At least one method converges successfully
- ✅ Mass conservation metrics captured
- ✅ Comparison visualization generated

### Full Success
- ✅ All methods run to completion
- ✅ Clear performance differences identified
- ✅ Trade-offs quantified
- ✅ Recommendations for future use

### Stretch Goals
- ✅ Validate hypotheses with statistical significance
- ✅ Identify optimal particle count for given grid size
- ✅ Establish performance scaling rules

## Known Issues and Limitations

### Current Limitations

1. **Grid-based FP mass loss**: Known issue in 2D FDM, needs better discretization
2. **Newton convergence warnings**: Normal for this problem, doesn't prevent solution
3. **KDE bandwidth selection**: Using Scott's rule, may not be optimal for all cases

### Future Improvements

1. **Implement particle-particle method**: Requires particle-based HJB solver
2. **Adaptive particle count**: Dynamically adjust based on density variation
3. **GPU acceleration**: Significant speedup possible for particle methods
4. **Higher-order FDM**: Reduce grid-based mass loss

## Comparison with Previous Work

### vs 1D Comparison (Earlier Session)
- 1D: Simple problem, both methods worked well
- 2D: More challenging, clear differences expected

### vs Broken 2D Implementation
- Broken: Used MFGComponents (wrong for nD)
- Current: Uses GridBasedMFGProblem (correct for nD)
- Result: Proper convergence and evacuation

## Timeline

**Experiment execution**: ~15-30 minutes (3 methods × 5-10 min each)
**Analysis**: 10-15 minutes
**Total**: ~40-60 minutes

## References

- Previous 1D comparison: `examples/outputs/particle_methods/1D_PARTICLE_COMPARISON_SUMMARY.md`
- nD solver investigation: `examples/outputs/particle_methods/2D_ND_SOLVER_INVESTIGATION_SUMMARY.md`
- Density evolution results: `examples/outputs/particle_methods/2D_DENSITY_EVOLUTION_SUMMARY.md`

---

**Status**: Experiment running
**Script**: `particle_methods_comparison_2d.py`
**Output**: `particle_comparison_output.log`
