# Density-Adaptive Collocation for MFG: Research Direction

**Date**: 2025-11-03
**Status**: Research idea (not yet implemented)
**Context**: Extension of dual-mode FP solver (feature/dual-mode-fp-solver)

---

## Motivation

Current collocation mode uses **fixed collocation points** throughout the MFG solve. However, the density distribution m(t,x) evolves significantly during:
1. Time evolution (FP forward solve)
2. Picard iteration (convergence to Nash equilibrium)

**Research Question**: Can we improve efficiency and accuracy by adaptively resampling collocation points based on the evolving density distribution?

---

## Core Idea

### Density-Adaptive Resampling

**Principle**: Concentrate collocation points where density is high, sparse where density is low.

**Intuition**:
- High density → many agents → need fine resolution
- Low density → few agents → coarse resolution sufficient
- Similar to adaptive mesh refinement, but for meshfree methods

### Comparison with Existing Adaptive Strategies

| Strategy | Timing | Criterion | MFG_PDE Status |
|:---------|:-------|:----------|:---------------|
| **Equi-spaced grid** | Fixed (initial) | Uniform coverage | ✅ Supported |
| **Geometry-based clustering** | Fixed (initial) | Problem structure (obstacles, boundaries) | ✅ Supported |
| **Density-adaptive** | Dynamic (per iteration) | Solution m(t,x) | ⚠️ Research direction |

---

## Algorithmic Approaches

### Approach 1: Resampling Between Picard Iterations

```python
# Outer loop: Picard iteration
for picard_iter in range(max_picard_iterations):
    # Solve HJB-FP with current collocation points
    U = hjb_solver.solve_hjb_system(M_current, ...)
    M_new = fp_solver.solve_fp_system(M_current[0], U)

    # Adaptive resampling based on density at final time T
    if picard_iter % resample_frequency == 0:
        points_new = resample_by_density(
            density=M_new[-1, :],        # Density at T
            points_current=points,       # Current collocation points
            N_target=N_particles,        # Target number of points
            method='importance_sampling' # Or 'particle_splitting'
        )

        # Rebuild solvers with new collocation points
        hjb_solver = HJBGFDMSolver(problem, collocation_points=points_new)
        fp_solver = FPParticleSolver(problem, mode='collocation', external_particles=points_new)
        points = points_new

    M_current = M_new
```

**Pros**:
- Relatively simple to implement
- Resampling only between Picard iterations (not every timestep)
- Natural convergence criterion: stop resampling when point distribution stabilizes

**Cons**:
- Expensive: Must rebuild GFDM neighbor structures each resampling
- Which density to use? m(t=0), m(t=T), or time-averaged?

---

### Approach 2: Importance Sampling Methods

#### 2a. Weighted Resampling (Particle Filter Style)

```python
def resample_by_density(density, points_current, N_target):
    """
    Resample collocation points using density as importance weights.

    Similar to particle filter resampling in sequential Monte Carlo.
    """
    # Normalize density to probability distribution
    weights = density / np.sum(density)

    # Importance sampling: high-density regions more likely
    indices = np.random.choice(
        len(points_current),
        size=N_target,
        replace=True,  # Allow duplication
        p=weights
    )

    # Add small jitter to avoid exact duplicates
    points_new = points_current[indices] + np.random.normal(0, jitter_scale, (N_target, dim))

    return points_new
```

**Issue**: Can lead to **sample degeneracy** (all points collapse to high-density region)

#### 2b. Stratified Resampling

```python
def stratified_resample(density, points_current, N_target, strata_density_threshold=0.1):
    """
    Partition domain into high/low density strata, resample proportionally.
    """
    high_density_mask = density > strata_density_threshold * np.max(density)

    # Allocate points proportional to mass in each stratum
    mass_high = np.sum(density[high_density_mask])
    mass_total = np.sum(density)
    N_high = int(N_target * mass_high / mass_total)
    N_low = N_target - N_high

    # Resample within each stratum
    points_high = importance_sample(density[high_density_mask], points_current[high_density_mask], N_high)
    points_low = uniform_sample(points_current[~high_density_mask], N_low)

    return np.vstack([points_high, points_low])
```

**Advantage**: Guarantees minimum coverage of low-density regions

---

### Approach 3: Particle Splitting/Merging

```python
def adaptive_refinement_splitting(density, points_current, refinement_threshold=0.8):
    """
    Split particles in high-density regions, merge in low-density regions.

    Maintains approximate particle count while adapting distribution.
    """
    points_new = []

    for i, (x, m) in enumerate(zip(points_current, density)):
        if m > refinement_threshold * np.max(density):
            # High density: split into 2-4 particles
            n_children = 2
            children = x + np.random.normal(0, split_radius, (n_children, dim))
            points_new.extend(children)
        elif m < (1 - refinement_threshold) * np.max(density):
            # Low density: merge with probability (skip some particles)
            if np.random.rand() > 0.5:
                points_new.append(x)
        else:
            # Medium density: keep as is
            points_new.append(x)

    return np.array(points_new)
```

**Advantage**: Local refinement/coarsening without global resampling
**Challenge**: Variable particle count (need normalization)

---

## Mathematical Considerations

### 1. Consistency and Convergence

**Question**: Does density-adaptive collocation preserve convergence of the MFG Picard iteration?

**Challenges**:
- Changing discretization between iterations
- Need to prove: $\|U^{(k+1)} - U^{(k)}\| \to 0$ and $\|M^{(k+1)} - M^{(k)}\| \to 0$ despite point changes
- Interpolation errors when transferring solutions between different point sets

**Potential approach**:
- Require minimum point density everywhere (avoid empty regions)
- Limit rate of point redistribution (smoothness constraint)
- Interpolate solutions conservatively (maintain mass, boundedness)

---

### 2. Mass Conservation

When resampling collocation points, we must preserve total mass:

```python
# Before resampling
mass_before = np.sum(density)  # Should be 1.0

# After resampling
density_new = interpolate_to_new_points(density, points_old, points_new)
mass_after = np.sum(density_new)

# Renormalize if necessary
density_new = density_new * (mass_before / mass_after)
```

**Methods**:
- Conservative interpolation (e.g., Voronoi-based)
- Projection onto mass-preserving subspace
- Lagrange multiplier for exact conservation

---

### 3. Interpolation Between Point Sets

When collocation points change, we need to transfer:
1. Density: $m_{\text{old}}(x_{\text{old}}) \to m_{\text{new}}(x_{\text{new}})$
2. Value function: $U_{\text{old}}(x_{\text{old}}) \to U_{\text{new}}(x_{\text{new}})$

**Options**:
- **Nearest neighbor**: Fast but non-smooth
- **RBF interpolation**: Smooth but expensive
- **KDE**: Natural for density (already used in hybrid mode)
- **Shepard's method**: Local polynomial approximation

---

## Implementation Plan (Future Work)

### Phase 1: Prototype (Research Repo)
1. Implement basic importance sampling resampling
2. Test on simple 2D LQ-MFG problem
3. Compare accuracy vs fixed collocation (equi-spaced, random uniform)
4. Measure computational overhead

### Phase 2: Algorithm Refinement
1. Test stratified resampling (avoid sample degeneracy)
2. Implement particle splitting/merging
3. Study convergence properties empirically
4. Determine optimal resampling frequency

### Phase 3: Theoretical Analysis
1. Prove convergence under assumptions (minimum density, bounded redistribution)
2. Derive error estimates (bias-variance tradeoff)
3. Compare with adaptive mesh refinement theory

### Phase 4: Production Integration (MFG_PDE)
1. Add optional `adaptive_resampling` parameter to collocation mode
2. Implement robust interpolation methods
3. Add safeguards (mass conservation, minimum density)
4. Comprehensive testing and documentation

---

## Research Questions

### Theoretical
1. **Convergence**: Does Picard iteration converge with changing collocation points?
2. **Error bounds**: How does resampling error propagate through iterations?
3. **Optimal strategy**: Importance sampling vs stratified vs splitting/merging?

### Practical
1. **Resampling frequency**: Every iteration? Every K iterations? Adaptive criterion?
2. **Density metric**: Use $m(t=0)$, $m(t=T)$, $\int_0^T m(t) dt$, or $\max_t m(t)$?
3. **Computational cost**: When does resampling overhead exceed accuracy benefit?
4. **High dimensions**: Does density-adaptive help more in d>3?

### Implementation
1. **GFDM rebuild cost**: Can we update neighbor structures incrementally instead of full rebuild?
2. **Interpolation quality**: Which method balances accuracy and speed?
3. **Stability**: How to prevent sample degeneracy or point clustering?

---

## Related Literature

### Adaptive Methods in MFG
- **Adaptive finite elements for MFG**: A posteriori error estimators, mesh refinement
- **Moving mesh methods**: Grid points follow solution features
- **Particle methods with resampling**: Sequential Monte Carlo for MFG

### Meshfree Adaptivity
- **h-adaptivity for GFDM**: Local refinement by adding/removing collocation points
- **Moving least squares with adaptive support**: Variable kernel sizes
- **Particle splitting/merging**: Smoothed particle hydrodynamics (SPH)

### Resampling Theory
- **Particle filters**: Systematic, stratified, residual resampling
- **Sequential Monte Carlo**: Effective sample size, sample degeneracy
- **Importance sampling**: Variance reduction, optimal proposal distributions

---

## Connection to Current Implementation

### Dual-Mode FP Solver (Implemented)
```python
# Fixed collocation points (current)
points = domain.sample_uniform(N_particles, seed=42)
fp_solver = FPParticleSolver(problem, mode='collocation', external_particles=points)

# Solve MFG (points don't change)
for iteration in range(max_iterations):
    U = hjb_solver.solve_hjb_system(M_current, ...)
    M_new = fp_solver.solve_fp_system(M_current[0], U)
    M_current = M_new
```

### Density-Adaptive Extension (Proposed)
```python
# Initial collocation points
points = domain.sample_uniform(N_particles, seed=42)

# Solve MFG with adaptive resampling
for iteration in range(max_iterations):
    # Solve with current points
    hjb_solver = HJBGFDMSolver(problem, collocation_points=points)
    fp_solver = FPParticleSolver(problem, mode='collocation', external_particles=points)

    U = hjb_solver.solve_hjb_system(M_current, ...)
    M_new = fp_solver.solve_fp_system(M_current[0], U)

    # Adaptive resampling (new)
    if should_resample(iteration, M_new, points):
        points = adaptive_resample(M_new[-1, :], points, N_particles)
        M_current = interpolate_density(M_new, points_old, points)
    else:
        M_current = M_new
```

---

## Experimental Validation Plan

### Test Problem 1: 2D Crowd Motion with Narrow Exit
**Why**: Density concentrates near exit - perfect for adaptive methods

**Comparison**:
- Fixed equi-spaced grid (32×32 = 1024 points)
- Fixed random uniform (1024 points)
- Density-adaptive importance sampling (1024 points)
- Density-adaptive stratified (1024 points)

**Metrics**:
- Solution accuracy (compare to fine grid reference)
- Computational time (including resampling overhead)
- Point distribution evolution (visualize)

### Test Problem 2: High-Dimensional LQ-MFG (d=5)
**Why**: Curse of dimensionality - grid infeasible, adaptive may help

**Comparison**:
- Fixed random uniform (10,000 points)
- Density-adaptive (10,000 points)

**Metrics**:
- Convergence rate of Picard iteration
- Final solution quality

### Test Problem 3: Multi-Population MFG
**Why**: Multiple density peaks - challenging for single-stratum methods

**Comparison**:
- Fixed collocation
- Density-adaptive with multi-modal awareness

---

## Next Steps (Research Repo)

1. **Implement basic prototype** in `mfg-research/adaptive_collocation/`
   - `importance_sampling_resampler.py`
   - `stratified_resampler.py`
   - Test on 2D LQ-MFG

2. **Benchmark experiments**
   - Compare with fixed collocation
   - Visualize point distribution evolution
   - Measure overhead vs accuracy gain

3. **Write research note** documenting findings
   - Does it work? When is it beneficial?
   - Optimal hyperparameters (resampling frequency, stratification threshold)

4. **If successful**: Write paper, then migrate stable implementation to MFG_PDE

---

## Open Questions for Discussion

1. **Resampling metric**: Should we use density $m(t,x)$ or some other criterion (gradient of value function, local error estimate)?

2. **Temporal adaptivity**: Should points also change with time $t$ during FP forward solve? (Not just between Picard iterations)

3. **Hybrid approach**: Combine density-adaptive collocation for FP with fixed-grid HJB? (Asymmetric discretization)

4. **Multi-fidelity**: Start with coarse points, progressively refine as Picard converges?

---

## References to Implement

- Particle filter resampling: Douc & Cappé (2005) "Comparison of resampling schemes"
- Adaptive GFDM: Benito et al. (2007) "Adaptive h-refinement for generalized finite difference method"
- Moving mesh for MFG: Similar to moving mesh for conservation laws (Huang & Russell, 2010)

---

**Document Status**: Research proposal - awaiting experimental validation
**Next Milestone**: Prototype implementation in mfg-research repository
**Timeline**: 1-2 months for initial experiments

---

**Author**: User idea, documented by Claude Code
**Date**: 2025-11-03
**Related**: Dual-mode FP solver (feature/dual-mode-fp-solver branch)
