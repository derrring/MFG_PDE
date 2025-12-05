# Adaptive Collocation Analysis for MFG Solvers

**Date**: 2025-12-03
**Context**: Analysis of GFDM+Particle solver performance in 2D crowd evacuation experiment
**Related Code**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`, `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Observation](#observation) — Experimental evidence
- [The Fundamental Conflict](#the-fundamental-conflict-information-direction) — HJB vs FP information flow
- [Why Adaptive Resampling Hurts HJB](#why-adaptive-resampling-hurts-hjb) — Four failure modes
- [Why HJB Needs Uniform Coverage](#why-hjb-needs-uniform-coverage) — Mathematical analysis
- [Error Propagation Chain](#error-propagation-in-mfg-solving-chain) — Complete error flow
- [Why Moving Collocation Fails for GFDM](#why-moving-collocation-is-bad-even-for-gfdm) — GFDM-specific issues
- [The Solution: Hybrid Eulerian-Lagrangian](#the-solution-hybrid-eulerian-lagrangian-architecture) — Recommended architecture
- [Conclusion](#conclusion)
- [References](#references)

---

## Executive Summary

This document analyzes why **density-aware adaptive collocation fails for HJB solvers** while the **hybrid Eulerian-Lagrangian approach** (fixed grid for HJB + particles for FP) emerges as the preferred solution. The analysis connects numerical stability theory, error propagation mechanisms, and the fundamental information-flow asymmetry between HJB and Fokker-Planck equations.

**Key Insight**: GFDM's scattered-point capability is designed for *static irregular geometries*, not *dynamic point clouds*. Moving collocation points during iteration destroys Picard convergence.

---

## Observation

In the 2D crowd evacuation experiment, **density-aware adaptive collocation performed worse than fixed uniform collocation** for the HJB solver:

| Algorithm | err_U (iter 1$\to$6) | err_M (iter 1$\to$6) | Convergence |
|-----------|------------------|------------------|-------------|
| GFDM+Particle (fixed) | 2.14 $\to$ 0.03 | 0.72 $\to$ 0.15 | Monotonic $\downarrow$ |
| GFDM+Particle (adaptive) | 2.64 $\to$ 0.21 | 0.98 $\to$ 0.33 | Oscillates |

The adaptive version resampled collocation points every 2 iterations based on the current density distribution, expecting to improve accuracy where mass is concentrated. Instead, it caused oscillations and slower convergence.

---

## The Fundamental Conflict: Information Direction

HJB and Fokker-Planck have **opposing information flows**, which dictates their optimal discretization:

| Feature | Hamilton-Jacobi-Bellman (HJB) | Fokker-Planck (FP) |
|:--------|:------------------------------|:-------------------|
| **Time Direction** | Backward $(T \to 0)$ | Forward $(0 \to T)$ |
| **Physics** | Information/value propagation | Mass transport |
| **Spatial Relevance** | Global (values exist everywhere) | Local (mass is localized) |
| **Ideal Mesh** | Fixed, uniform (Eulerian) | Adaptive/moving (Lagrangian) |

**Why "Density-Aware" fails for HJB:**
The value function $u(x,t)$ represents a *potential* outcome. Even if no agent is currently at location $x$, the solver must know exactly what *would* happen if an agent arrived there, because agents in neighboring regions need that information to decide their velocity. If you under-sample empty regions, you destroy the "navigation map" for agents who might enter those regions later.

---

## Why Adaptive Resampling Hurts HJB

### 1. The "Moving Goalposts" Problem (Picard Stability)

When collocation points are resampled, GFDM stencils change completely. The solution from the previous iteration was computed on different points:

```
Iter 2: err_U=0.84  (computed on point set A)
[Resample → new point set B]
Iter 3: err_U=0.73  ← should continue decreasing, but jumps due to new stencils
```

Mathematically, Picard iteration seeks a fixed point $u^*$ such that $u^* = \Phi(u^*)$:

- **Fixed Grid:** The operator $\Phi$ is constant. The iteration contracts purely based on the physics of the MFG.
- **Moving Grid:** You are solving $u^{(n+1)} = \Phi_n(u^{(n)})$. The operator itself changes every iteration because the mesh changes.

The error introduced by projecting the solution from Mesh $n$ to Mesh $n+1$ acts as **artificial noise**. If this projection error is larger than the contraction rate of the fixed point map, the solver will oscillate endlessly (limit cycle) rather than converge.

### 2. HJB Requires Global Coverage, Not Density-Weighted

The value function $u(x,t)$ represents the **optimal cost-to-go from any point $x$**, regardless of whether agents are currently there.

The optimal control is:
$$\alpha^*(x,t) = -\frac{1}{\lambda} \nabla u(x,t)$$

If collocation points are concentrated in high-density regions:
- **High-$m$ regions**: Many points $\to$ good $u$ accuracy
- **Low-$m$ regions**: Few points $\to$ poor $u$ accuracy $\to$ wrong gradients $\to$ wrong control

Agents may need to pass through or navigate around low-density regions. Inaccurate $u$ there leads to suboptimal trajectories.

### 3. Timing Mismatch

The resampling workflow has inherent lag:
1. Resample based on $m^n$
2. Solve HJB on new points → get $u^{n+1}$
3. Solve FP → get $m^{n+1}$ which differs from $m^n$

The density used for sampling is already stale by the time FP completes.

### 4. Resampling Frequency

Resampling every 2 iterations is too aggressive. The solution hasn't stabilized enough for the density to reliably guide point placement.

---

## Why HJB Needs Uniform Coverage

### Mathematical Nature

The HJB equation:
$$-\frac{\partial u}{\partial t} - \frac{\sigma^2}{2} \Delta u + H(x, \nabla u, m) = 0$$

Key properties:

1. **Backward Propagation**: HJB solves backward from terminal condition. Information flows from exit $\to$ everywhere. Low-density regions at time $t$ may be crucial pathways at earlier times.

2. **Elliptic Error Pollution**: The diffusion term $\frac{\sigma^2}{2}\Delta u$ couples the solution globally. Because the HJB equation is **elliptic** (or parabolic in time, which behaves elliptically in space per timestep), the inverse operator is dense.

   **The Consequence:** You cannot "save computational cost" by ignoring low-density regions in the HJB step. If the grid is coarse in empty regions, the truncation error $\tau$ there becomes a "source term" for error that **instantly spreads** to the high-density regions where you care about accuracy.

3. **Control Extraction**: The gradient $\nabla u$ must be accurate everywhere to compute correct optimal controls.

---

## Error Propagation in MFG Solving Chain

### Complete Iteration Flow

```
m⁰ (initial guess)
    ↓
┌─────────────────────────────────────────────────────────┐
│  ITERATION n                                            │
│                                                         │
│  1. HJB (backward): Given mⁿ → solve for uⁿ⁺¹          │
│     -∂u/∂t - (σ²/2)Δu + H(x,∇u,mⁿ) = 0                 │
│                                                         │
│  2. FP (forward): Given uⁿ⁺¹ → solve for mⁿ⁺¹          │
│     ∂m/∂t - (σ²/2)Δm - ∇·(m·∇ₚH) = 0                   │
│     where α* = -∇u/λ (optimal control from uⁿ⁺¹)       │
│                                                         │
│  3. Check: ‖uⁿ⁺¹-uⁿ‖ < ε and ‖mⁿ⁺¹-mⁿ‖ < ε ?         │
└─────────────────────────────────────────────────────────┘
    ↓ (repeat until convergence)
(u*, m*) fixed point
```

### Error Sources by Stage

#### Stage 1: HJB Solver (backward $t: T \to 0$)

| Error | Source | Impact |
|-------|--------|--------|
| E1 | Interpolation: $m^n_{\text{grid}} \to m^n_{\text{colloc}}$ | Affects Hamiltonian |
| E2 | Stencil accuracy: GFDM weights for $\Delta u$, $\nabla u$ | Sparse regions degrade to $O(h)$ |
| E3 | Hamiltonian evaluation: $H(x, \nabla u, m)$ | Compounds E1, E2 |
| E4 | Time stepping: backward Euler/Crank-Nicolson | $O(\Delta t)$ or $O(\Delta t^2)$ |

#### Stage 2: Optimal Control Extraction

$$\alpha^* = -\frac{1}{\lambda}\nabla u^{n+1}$$

| Error | Source | Impact |
|-------|--------|--------|
| E5 | Gradient computation | **Amplified**: $\|\nabla u_{\text{num}} - \nabla u_{\text{exact}}\| \sim O(h)$ not $O(h^2)$ |
| E6 | Control magnitude | Wrong velocity direction and magnitude |

**Critical**: $\epsilon_{\text{ctrl}} \sim \epsilon_{\text{HJB}} / h$ — differentiation amplifies errors.

#### Stage 3: FP Solver (forward $t: 0 \to T$)

**Option A: FDM**

| Error | Source | Impact |
|-------|--------|--------|
| E7a | Advection discretization | Upwind diffusive, central oscillatory |
| E8a | Mass conservation | Boundary stencils leak, clipping destroys mass |

**Option B: Particle (SDE)**

| Error | Source | Impact |
|-------|--------|--------|
| E7b | Velocity interpolation at particles | Compounds E5 |
| E8b | SDE integration (Euler-Maruyama) | $O(\sqrt{\Delta t})$ |
| E9b | Density estimation (KDE) | Smoothing bias |

Particle method: **Mass exactly conserved** (just count particles).

#### Stage 4: Picard Update

| Error | Source | Impact |
|-------|--------|--------|
| E10 | Damping | Helps stability, slows convergence |
| E11 | Resampling projection | **Destroys iteration consistency** |

### Critical Amplification Points

| Stage | Error Type | Amplification | Why Critical |
|-------|-----------|---------------|--------------|
| E2$\to$E5 | $u \to \nabla u$ | $O(1/h)$ | Differentiation amplifies |
| E5$\to$E7b | $\nabla u \to$ particle velocity | $O(1)$ | Wrong trajectories |
| E8a | Advection | Cumulative over $T$ | Mass leaks every timestep |
| E11 | Resampling | Resets accuracy | Breaks iteration consistency |

### The "Differentiation Trap"

The error chain reveals a devastating amplification loop:

$$\epsilon_{control} \approx \frac{\epsilon_{HJB}}{h}$$

Since the control $\alpha^* = -\nabla u / \lambda$ depends on the *gradient* $\nabla u$, any error in the value function is amplified by $1/h$ (numerical differentiation).

With moving mesh (GFDM adaptive):
1. Local stencil quality fluctuates as points move
2. This causes $\epsilon_{\text{HJB}}$ to fluctuate
3. The gradient $\nabla u$ becomes noisy
4. Particles in FP receive "jittery" velocity commands
5. Causes artificial diffusion or instability in density $m$

This is why fixed grids with stable, precomputed stencils are essential for HJB.

---

## Why Moving Collocation is Bad Even for GFDM

GFDM is designed for scattered points, but **moving collocation still hurts HJB**:

### 1. Temporal Consistency

HJB solves backward in time. At each timestep:
$$u(t, x) = u(t+dt, x) + dt \cdot [\text{RHS}]$$

If points move between timesteps:
```
t=T:    u computed at points {x₁, x₂, ..., xₙ}
t=T-dt: points moved to {x'₁, x'₂, ..., x'ₙ}
        → must interpolate u(T) to new locations
        → interpolation error EVERY timestep
        → O(Nt) error accumulation over backward sweep
```

### 2. Stencil Recomputation Cost

GFDM precomputes stencil weights by solving local least-squares problems:
- **Fixed points**: Compute stencils once $\to$ reuse for all $N_t$ timesteps
- **Moving points**: Recompute stencils every timestep $\to$ $N_t \times$ more work

For the experiment: 15 timesteps $\times$ ~2 min stencil computation = 30 min vs 2 min.

### 3. Solution Smoothness Assumption

GFDM approximates derivatives assuming the solution is smooth over the stencil. When points move:
- Stencil neighbors change discontinuously
- Local polynomial fit changes discontinuously
- Derivative estimates jump even if true solution is smooth

### 4. Condition Number Instability

GFDM stencil quality depends on point configuration. Random resampling can create:
- Nearly collinear points $\to$ ill-conditioned local system
- Gaps in coverage $\to$ large stencils $\to$ accuracy loss
- Clusters $\to$ redundant information, wasted points

### What GFDM Flexibility Is Actually For

| Use Case | GFDM Advantage | Moving Points? |
|----------|----------------|----------------|
| Complex geometry | Fits irregular boundaries | No - fixed boundary-conforming |
| Local refinement | Dense near features | No - fixed refined regions |
| Unstructured mesh | Works on any point cloud | No - fixed mesh |
| Adaptive collocation | ❌ Not this | Causes more problems than it solves |

**GFDM's scattered-point capability is for static irregular geometries, not dynamic point clouds.**

---

## The Solution: Hybrid Eulerian-Lagrangian Architecture

The conclusion of this analysis—using a hybrid approach—aligns with the standard recommendation for high-performance MFG solvers today.

### The "Best of Both Worlds" Architecture

1. **HJB (The "Map"):** Use a **Fixed, Uniform Grid** (Eulerian)
   - *Why:* Ensures the "map" is stable, smooth, and accurate everywhere
   - The Laplacian pollution is minimized
   - Stencils precomputed once, reused for all timesteps

2. **FP (The "Crowd"):** Use **Particles / SDEs** (Lagrangian)
   - *Why:* Particles naturally concentrate where probability mass is high
   - No need to "mesh" the empty space
   - Effectively provides infinite resolution in high-density areas and zero resolution in empty areas—exactly what is needed for mass transport
   - Mass conservation is exact (just count particles)

### Why This Works

```
Grid HJB:   Uniform coverage → accurate u everywhere
                ↓
            Accurate ∇u → accurate α*
                ↓
Particle FP: Particles follow α* → exact mass conservation
                ↓
            KDE/project to grid → mⁿ⁺¹ for next HJB
                ↓
            Errors don't amplify through mass loss
```

The particle method breaks the mass loss chain (E8a), which is the dominant error in FDM-based FP solvers.

### Practical Recommendations

**For HJB Solver:**
1. Use **fixed uniform collocation** (grid or quasi-uniform scattered)
2. Refine near boundaries/features but keep points static
3. Precompute GFDM stencils once and reuse

**For FP Solver:**
1. **Particle methods** naturally adapt to density
2. Mass conservation is exact (count particles)
3. Use KDE or histogram for density projection to grid

**If Adaptive Collocation Is Required:**
1. Resample very infrequently (every 10+ iterations, or only on stall)
2. Use mixed strategy: fixed base grid + density-weighted refinement overlay
3. Smooth transition: blend old/new points instead of full replacement
4. Consider p-refinement (increase polynomial order) over h-refinement (move points)

---

## Conclusion

The experiment confirms that **uniform coverage beats density-aware collocation for HJB** because:

1. Value function accuracy is needed globally, not just where density is high
2. Moving points breaks temporal consistency in backward solve
3. Resampling destroys Picard iteration momentum (the "moving goalposts" problem)
4. The differentiation trap amplifies errors by $O(1/h)$
5. Elliptic error pollution spreads truncation error globally
6. GFDM stencil recomputation is expensive and can degrade quality

The hybrid Eulerian-Lagrangian approach (fixed grid for HJB + particles for FP) leverages the complementary strengths of each representation:
- **Global accuracy** where needed (HJB navigation map)
- **Natural density adaptation** where beneficial (FP mass transport)
- **Exact mass conservation** through particle counting

---

## References

1. Achdou, Y., & Laurière, M. (2020). *Mean Field Games and Applications: Numerical Aspects*. EMS Surveys in Mathematical Sciences. (Discusses the necessity of stable grids for the HJB backward sweep).

2. Guéant, O. (2016). *Mean Field Games: A Partial Differential Equation Approach*. Paris-Dauphine University. (See chapters on numerical schemes and the coupling of HJB-KFP systems).

3. Achdou, Y., & Capuzzo-Dolcetta, I. (2010). Mean field games: numerical methods. *SIAM Journal on Numerical Analysis*, 48(3), 1136-1162.

4. Carlini, E., & Silva, F. J. (2014). A fully discrete semi-Lagrangian scheme for a first order mean field game problem. *SIAM Journal on Numerical Analysis*, 52(1), 45-67.
