# Advection Discretization Methods for Fokker-Planck Equations

A comprehensive comparison of discretization approaches for advection-diffusion PDEs, focusing on mass conservation properties critical for Mean Field Games.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Setting](#problem-setting)
3. [Part I: Discretization Methods](#part-i-discretization-methods)
   - [Gradient FDM (Non-Conservative)](#1-gradient-fdm-non-conservative)
   - [Flux FDM (Conservative)](#2-flux-fdm-conservative)
   - [Finite Volume Method (FVM)](#3-finite-volume-method-fvm)
4. [Part II: Method Comparison](#part-ii-method-comparison)
   - [Summary Table](#summary-table)
   - [Higher-Order Accuracy Analysis](#higher-order-accuracy-analysis)
   - [Method Selection Guide](#method-selection-guide)
5. [Part III: MFG Context](#part-iii-mfg-context)
   - [Why Method Diversity Matters](#why-method-diversity-matters)
   - [MFG_PDE Solver Portfolio](#mfg_pde-solver-portfolio)
6. [Implementation Notes](#implementation-notes)
7. [References](#references)

---

## Executive Summary

| Method | Conservative? | Accuracy | Complexity | Best For |
|--------|--------------|----------|------------|----------|
| **Gradient FDM** | No | $O(\Delta x)$ | Low | Prototyping, HJB equations |
| **Flux FDM** | Yes | $O(\Delta x)$ | Low | Quick mass-preserving FP |
| **FVM** | Yes | $O(\Delta x^2)$ possible | Medium | Production FP solvers |
| **Particle** | Yes (inherent) | Depends | Medium | High-D, moving domains |

**Key insight**: For Fokker-Planck, mass conservation requires **column sums = 0** in the discretization matrix. Gradient FDM fails this; Flux FDM and FVM satisfy it by construction.

---

## Problem Setting

The Fokker-Planck equation describes probability density evolution:

$$
\frac{\partial m}{\partial t} + \nabla \cdot (\alpha m) = D \Delta m
$$

**Notation**:
- $m(t,x)$: probability density, must satisfy $\int m \, dx = 1$
- $\alpha(x) = -\nabla U(x)$: drift velocity from potential $U$
- $D = \sigma^2/2$: diffusion coefficient

**Conservation requirement**: With no-flux boundary conditions:
$$
\frac{d}{dt} \int_\Omega m \, dx = 0
$$

This is the fundamental constraint that distinguishes FP solvers from general advection-diffusion solvers.

---

# Part I: Discretization Methods

## 1. Gradient FDM (Non-Conservative)

### Mathematical Formulation

Using the product rule $\nabla \cdot (\alpha m) = \alpha \cdot \nabla m + m \nabla \cdot \alpha$, if we assume $\nabla \cdot \alpha = 0$:

$$
\frac{\partial m}{\partial t} + \alpha \cdot \nabla m = D \Delta m
$$

This is the **gradient form** — we discretize $\alpha \cdot \nabla m$ directly.

### Upwind Discretization (1D)

At grid point $i$ with velocity $\alpha_i$:

$$
(\alpha \cdot \nabla m)_i \approx
\begin{cases}
\alpha_i \dfrac{m_i - m_{i-1}}{\Delta x} & \text{if } \alpha_i > 0 \\[8pt]
\alpha_i \dfrac{m_{i+1} - m_i}{\Delta x} & \text{if } \alpha_i < 0
\end{cases}
$$

**Matrix form** (implicit time stepping):
$$
\left(\frac{I}{\Delta t} + L_{\text{grad}}\right) m^{n+1} = \frac{m^n}{\Delta t}
$$

### Conservation Analysis

For an implicit scheme $A m^{n+1} = b$, mass conservation requires:
$$
\sum_j m_j^{n+1} = \sum_j m_j^n \quad \Longleftrightarrow \quad \text{column sums of } A = \frac{1}{\Delta t}
$$

**Gradient FDM fails this test** because the upwind direction at point $i$ depends on $\alpha_i$, not on a shared interface velocity. Column sums vary with the velocity field structure.

**Result**: Mass NOT conserved unless $\nabla \cdot \alpha = 0$ everywhere.

### Code Reference

Current implementation in `fp_fdm.py:_add_interior_entries`:
```python
coeff_plus += -coupling * ppart(u_plus - u_center) / dx_sq
diagonal   += +coupling * ppart(u_plus - u_center) / dx_sq
```

---

## 2. Flux FDM (Conservative)

### Mathematical Formulation

Keep the divergence form without approximation:
$$
\frac{\partial m}{\partial t} + \nabla \cdot (\alpha m) = D \Delta m
$$

### Flux-Difference Discretization (1D)

Discretize divergence as difference of fluxes at cell interfaces:
$$
(\nabla \cdot (\alpha m))_i \approx \frac{F_{i+1/2} - F_{i-1/2}}{\Delta x}
$$

**Interface velocity** (computed at cell face, not cell center):
$$
\alpha_{i+1/2} = -\frac{U_{i+1} - U_i}{\Delta x}
$$

**Upwind flux**:
$$
F_{i+1/2} = \alpha_{i+1/2} \cdot m_{\text{upwind}} =
\begin{cases}
\alpha_{i+1/2} \cdot m_i & \text{if } \alpha_{i+1/2} \geq 0 \\
\alpha_{i+1/2} \cdot m_{i+1} & \text{if } \alpha_{i+1/2} < 0
\end{cases}
$$

### Conservation Analysis

**Telescoping property**: Summing over all cells:
$$
\sum_{i=1}^{N} (F_{i+1/2} - F_{i-1/2}) = F_{N+1/2} - F_{1/2} = 0 \quad \text{(with no-flux BC)}
$$

**Column sum proof**: Consider how $m_j$ affects all equations:
- The flux $F_{j+1/2}$ appears with $+1/\Delta x$ in row $j$ (outgoing)
- The same flux appears with $-1/\Delta x$ in row $j+1$ (incoming)
- These cancel exactly → column sum = 0

**Result**: Mass CONSERVED by construction.

### Implementation

```python
def flux_fdm_advection(i, u, dx, coupling):
    # Interface velocities (defined at cell faces)
    alpha_right = -coupling * (u[i+1] - u[i]) / dx   # at x_{i+1/2}
    alpha_left  = -coupling * (u[i] - u[i-1]) / dx   # at x_{i-1/2}

    # Flux F_{i+1/2} contribution
    if alpha_right >= 0:
        diagonal += alpha_right / dx      # upwind from m_i
    else:
        coeff_right += alpha_right / dx   # upwind from m_{i+1}

    # Flux -F_{i-1/2} contribution
    if alpha_left >= 0:
        coeff_left -= alpha_left / dx     # upwind from m_{i-1}
    else:
        diagonal -= alpha_left / dx       # upwind from m_i
```

---

## 3. Finite Volume Method (FVM)

### Mathematical Formulation

Integrate the PDE over control volume $\Omega_i = [x_{i-1/2}, x_{i+1/2}]$:
$$
\frac{d}{dt} \int_{\Omega_i} m \, dx + \oint_{\partial\Omega_i} (\alpha m - D\nabla m) \cdot n \, dS = 0
$$

**Primary variable**: Cell average
$$
\bar{m}_i = \frac{1}{|\Omega_i|} \int_{\Omega_i} m \, dx
$$

### Semi-Discrete Form

$$
\frac{d\bar{m}_i}{dt} = -\frac{F_{i+1/2} - F_{i-1/2}}{\Delta x}
$$

where the numerical flux combines advection and diffusion:
$$
F_{i+1/2} = \alpha_{i+1/2} m_{i+1/2} - D \frac{m_{i+1} - m_i}{\Delta x}
$$

### Reconstruction for Higher Order

FVM achieves higher accuracy through **reconstruction** from cell averages:

| Order | Method | Description |
|-------|--------|-------------|
| 1st | Piecewise constant | $m_{i+1/2} = \bar{m}_i$ (upwind) |
| 2nd | Piecewise linear + limiter | $m_{i+1/2} = \bar{m}_i + \sigma_i \Delta x/2$ |
| 3rd+ | WENO, PPM | Weighted nonlinear combinations |

**Piecewise linear with MinMod limiter**:
$$
\sigma_i = \text{minmod}\left(\frac{\bar{m}_i - \bar{m}_{i-1}}{\Delta x}, \frac{\bar{m}_{i+1} - \bar{m}_i}{\Delta x}\right)
$$

The minmod function:
$$
\text{minmod}(a, b) =
\begin{cases}
\text{sign}(a) \min(|a|, |b|) & \text{if } ab > 0 \\
0 & \text{otherwise}
\end{cases}
$$

### Conservation Property

FVM is **inherently conservative** because:
1. Fluxes are defined at shared interfaces
2. Outflow from cell $i$ = inflow to cell $i+1$ (exactly)
3. Global sum: $\sum_i \frac{d\bar{m}_i}{dt} |\Omega_i| = -(F_{N+1/2} - F_{1/2}) = 0$

---

# Part II: Method Comparison

## Summary Table

| Property | Gradient FDM | Flux FDM | FVM |
|----------|-------------|----------|-----|
| **PDE form** | $\alpha \cdot \nabla m$ | $\nabla \cdot (\alpha m)$ | $\nabla \cdot (\alpha m)$ |
| **Primary variable** | Point values $m_i$ | Point values $m_i$ | Cell averages $\bar{m}_i$ |
| **Velocity location** | Cell centers | Cell interfaces | Cell interfaces |
| **Mass conservation** | No | Yes | Yes |
| **Matrix column sums** | $\neq 0$ | $= 0$ | $= 0$ |
| **Base accuracy** | $O(\Delta x)$ | $O(\Delta x)$ | $O(\Delta x)$ |
| **Higher-order path** | Wider stencils | Limited | Reconstruction + limiters |
| **Achievable accuracy** | $O(\Delta x)$ practical | $O(\Delta x)$ | $O(\Delta x^2)$ or higher |
| **Implementation** | Simple | Moderate | More complex |

---

## Higher-Order Accuracy Analysis

### Why FDM Struggles with Higher Order

FDM works with **point values** $m_i = m(x_i)$. Higher-order derivatives require wider stencils:

| Order | Scheme | Stencil | Problem |
|-------|--------|---------|---------|
| 1st | Upwind | 2 points | Stable, diffusive |
| 2nd | Central | 3 points | **Unstable for advection** |
| 3rd | QUICK | 4 points | Boundary complications |

**Central differences** ($O(\Delta x^2)$) lack numerical diffusion and produce oscillations in advection-dominated flows. Adding artificial diffusion restores stability but degrades accuracy.

### Why FVM Naturally Achieves Higher Order

FVM works with **cell averages** $\bar{m}_i$. To compute interface flux, it must **reconstruct** point values:

```
Given:     |  m̄_{i-1}  |   m̄_i   |  m̄_{i+1}  |
                      ↓
Reconstruct polynomial within each cell
                      ↓
Evaluate:  m_{i+1/2}^L  and  m_{i+1/2}^R
```

**Key advantages**:
1. **Reconstruction is well-posed**: Polynomial interpolation from averages is standard
2. **Limiters integrate naturally**: Bound the reconstruction slope to prevent oscillations
3. **Modular design**: Change limiter without changing flux calculation
4. **Mature frameworks**: WENO, PPM, etc. are well-understood

### Comparison: Achieving 2nd-Order

**FDM approach** (problematic):
- Use central difference: $\alpha_i (m_{i+1} - m_{i-1})/(2\Delta x)$
- $O(\Delta x^2)$ truncation error
- **Unstable** — oscillations grow without bound

**FVM approach** (stable):
1. Compute limited slope: $\sigma_i = \text{minmod}(\text{backward}, \text{forward})$
2. Reconstruct: $m_{i+1/2}^L = \bar{m}_i + \sigma_i \Delta x/2$
3. Use upwind based on velocity sign
4. **TVD** — Total Variation Diminishing, no new oscillations

---

## Method Selection Guide

### When to Use Gradient FDM
- Rapid prototyping and algorithm development
- Problems where $\nabla \cdot \alpha \approx 0$ (incompressible flow)
- HJB equations (conservation not required)
- When mass loss is acceptable (~5-20% typical)

### When to Use Flux FDM
- Need mass conservation on structured grids
- Minimal code changes from gradient FDM
- First-order accuracy sufficient
- Quick upgrade path from non-conservative solver

### When to Use FVM
- Production-quality Fokker-Planck simulations
- Higher-order accuracy required
- Complex geometries (with unstructured mesh capability)
- Standard choice in CFD and transport communities

---

# Part III: MFG Context

## Why Method Diversity Matters

FVM is excellent for Fokker-Planck, but MFG problems require a **diverse toolkit**.

### 1. Two Coupled PDEs with Different Requirements

| Equation | Type | Conservation? | Recommended Methods |
|----------|------|---------------|---------------------|
| **Fokker-Planck** | Forward parabolic | Critical | FVM, Flux FDM, Particle |
| **Hamilton-Jacobi-Bellman** | Backward parabolic | Not needed | FDM (monotone), FEM |

HJB requires **viscosity solution** properties (monotonicity, stability at kinks), not conservation. Simple upwind FDM is natural and effective.

### 2. Geometry Constraints

| Domain Type | Best Choice | Rationale |
|-------------|-------------|-----------|
| Rectangle | FDM/FVM | Simplest, optimal stencils |
| Polygonal (L-shape) | FEM | Mesh conforms to boundary |
| Curved boundaries | FEM or GFDM | Avoid staircase artifacts |
| Moving domains | Particle, GFDM | No remeshing required |
| SDF-defined | GFDM | Natural SDF integration |

### 3. Curse of Dimensionality

| Dimension | Grid Methods | Particle Methods |
|-----------|--------------|------------------|
| 1D–3D | $O(N^d)$ DOFs — feasible | $O(N_p)$ particles |
| 5D–10D | Impractical | Still manageable |
| 100D+ | Impossible | Monte Carlo required |

For high-dimensional MFG (multi-agent systems), **only particle/Monte Carlo methods scale**.

### 4. Development Stage Trade-offs

| Stage | Priority | Recommended |
|-------|----------|-------------|
| Research/prototyping | Speed, simplicity | Gradient FDM |
| Algorithm validation | Correctness | Flux FDM |
| Production | Accuracy, robustness | FVM or Particle |
| High-D applications | Scalability | Particle + DGM/PINN |

---

## MFG_PDE Solver Portfolio

Our solver ecosystem reflects these trade-offs:

```
FP Solvers (density evolution):
├── FDM (Gradient)   → Fast prototyping, acceptable mass loss
├── GFDM + Particle  → Production-quality, mass-preserving, meshfree
└── [Future: FVM]    → High-order conservative, structured grids

HJB Solvers (value function):
├── FDM (upwind)     → Monotone, viscosity-consistent
├── Semi-Lagrangian  → Large timesteps, unconditional stability
└── DGM/PINN         → High dimensions, neural network based

Coupled MFG:
├── Fixed-point iteration with solver mixing
└── Automatic solver selection based on problem characteristics
```

---

# Implementation Notes

## Current State in MFG_PDE

- `FPFDMSolver` uses **Gradient FDM** (non-conservative)
- `GFDM+Particle` solver is **inherently conservative** (particles carry fixed mass)

## Potential Enhancement

Add `conservative=True` option to `FPFDMSolver`:
```python
solver = FPFDMSolver(problem, conservative=True)  # Uses Flux FDM
```

## Boundary Conditions for Conservative Methods

For no-flux BC at left boundary with Flux FDM:

**Requirement**: $F_{1/2} = 0$

**Implementation**:
- Set interface velocity $\alpha_{1/2} = 0$ (or use one-sided difference)
- Ghost point: $m_0 = m_1$ (reflection)
- Result: $F_{1/2} = \alpha_{1/2} m_{\text{upwind}} - D(m_1 - m_0)/\Delta x = 0$ ✓

---

# References

1. LeVeque, R.J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.

2. Toro, E.F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. 3rd ed. Springer.

3. Hundsdorfer, W. & Verwer, J.G. (2003). *Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations*. Springer.

4. Achdou, Y. & Laurière, M. (2020). *Mean Field Games and Applications: Numerical Aspects*. In Mean Field Games, Springer.

---

**Document Info**
- **Last Updated**: 2025-12
- **Related Files**: `fp_fdm.py`, `boundary_conditions_and_geometry.md`
- **Related Issues**: #382 (FDM mass conservation investigation)
- **Author**: MFG_PDE Development Team
