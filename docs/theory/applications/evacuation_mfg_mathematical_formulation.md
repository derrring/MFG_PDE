# Evacuation Mean Field Games: Mathematical Formulation

**Document Type**: Application-Specific MFG Formulation
**Created**: October 2025
**Status**: Enhanced with Mathematical Rigor
**Related**: `anisotropic_mfg_mathematical_formulation.md`, `mathematical_background.md`, `convergence_criteria.md`

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [The Mass Conservation Paradox](#2-the-mass-conservation-paradox)
3. [Three Mathematical Approaches](#3-three-mathematical-approaches)
4. [Application: Two-Door Room Evacuation](#4-application-two-door-room-evacuation)
5. [Anisotropic Extensions](#5-anisotropic-extensions)
6. [Numerical Methods and Stability](#6-numerical-methods-and-stability)
7. [References](#references)

---

## 1. Overview and Motivation

This document provides rigorous mathematical formulations for **evacuation games** using Mean Field Game (MFG) theory, addressing the fundamental challenge: in real evacuations, **total mass decreases** as agents exit, violating the standard MFG assumption of mass conservation.

**Practical Applications**:
- Emergency evacuation planning (buildings, stadiums, transportation hubs)
- Crowd management in large events
- Urban planning for pedestrian infrastructure
- Safety regulation compliance (exit capacity analysis)

**Mathematical Challenge**: Standard MFG theory assumes $\int_\Omega m(t,x) dx = \text{const}$, but evacuation requires $\int_\Omega m(t,x) dx \to 0$ as $t \to T$.

---

## 2. The Mass Conservation Paradox

### 2.1 Standard MFG Framework

**Definition 2.1 (Standard MFG System)**[^1]:
Let $\Omega \subset \mathbb{R}^d$ be a bounded domain, $T > 0$ a time horizon. The classical MFG system consists of:

**Hamilton-Jacobi-Bellman (HJB) Equation** (backward):
$$-\frac{\partial u}{\partial t} + H(x, \nabla u, m) = f(x, m), \quad (t,x) \in [0,T] \times \Omega$$

**Fokker-Planck-Kolmogorov (FPK) Equation** (forward):
$$\frac{\partial m}{\partial t} - \nabla \cdot (m \nabla_p H(x, \nabla u, m)) - \sigma \Delta m = 0, \quad (t,x) \in [0,T] \times \Omega$$

**Boundary Conditions**:
- **Terminal condition**: $u(T,x) = g(x)$ for all $x \in \Omega$
- **Initial condition**: $m(0,x) = m_0(x)$ with $\int_\Omega m_0 dx = 1$
- **Spatial boundary**: $\frac{\partial u}{\partial n} = 0$, $\frac{\partial m}{\partial n} = 0$ on $\partial \Omega$ (Neumann)

**Mass Conservation Property**[^2]:
For classical MFG with Neumann boundaries:
$$\frac{d}{dt} \int_\Omega m(t,x) dx = 0 \quad \Rightarrow \quad \int_\Omega m(t,x) dx = \int_\Omega m_0(x) dx = 1, \quad \forall t \in [0,T]$$

**Proof Sketch**: Integrate FPK equation over $\Omega$ and use divergence theorem:
$$\frac{d}{dt} \int_\Omega m dx = \int_\Omega \nabla \cdot (\cdots) dx + \sigma \int_\Omega \Delta m dx = \int_{\partial\Omega} (\cdots) \cdot n \, dS = 0$$

### 2.2 Evacuation Reality: Non-Conservative Systems

**Problem Statement**: In evacuation scenarios, agents **permanently leave** the system through exits $\Gamma_{\text{exit}} \subset \partial \Omega$, leading to:

$$\frac{d}{dt} \int_\Omega m(t,x) dx < 0, \quad \int_\Omega m(T,x) dx \ll 1$$

**Physical Interpretation**:
- Mass $m(t,x)$ represents density of people still in $\Omega$ at time $t$
- Exits act as **absorbing boundaries** or **sinks**
- Total evacuation: $\lim_{t \to T} \int_\Omega m(t,x) dx = 0$

**Mathematical Challenge**: Standard MFG solvers and convergence theory[^3] assume mass conservation. Evacuation requires modified formulations.

---

## 3. Three Mathematical Approaches

### 3.1 Approach 1: Absorbing Boundary Conditions

**Idea**: Add **killing term** to FPK equation to remove mass near exits.

#### 3.1.1 Mathematical Formulation

**Definition 3.1 (Evacuation MFG with Absorption)**:
The MFG system with absorption consists of:

**HJB Equation** (unchanged):
$$-\frac{\partial u}{\partial t} + H(x, \nabla u, m) = f(x, m), \quad (t,x) \in [0,T] \times \Omega$$

**Modified FPK Equation**:
$$\frac{\partial m}{\partial t} - \nabla \cdot (m \nabla_p H) - \sigma \Delta m = -k(x) m, \quad (t,x) \in [0,T] \times \Omega$$

where $k : \Omega \to \mathbb{R}_{\geq 0}$ is the **exit rate function** (absorption coefficient).

**Definition 3.2 (Exit Rate Function)**:
Given exit set $\Gamma_{\text{exit}} \subset \partial \Omega$, define:
$$k(x) = \kappa \cdot \exp\left(-\alpha \cdot d(x, \Gamma_{\text{exit}})^2\right)$$

where:
- $d(x, \Gamma_{\text{exit}}) = \inf_{y \in \Gamma_{\text{exit}}} \|x - y\|$ (distance to nearest exit)
- $\kappa > 0$ is **absorption strength** (exit capacity)
- $\alpha > 0$ is **spatial sharpness** parameter

**Theorem 3.3 (Mass Decay Rate)**[^4]:
For the absorption model, the total mass satisfies:
$$M(t) := \int_\Omega m(t,x) dx$$
$$\frac{dM}{dt} = -\int_\Omega k(x) m(t,x) dx \leq 0$$

with exponential decay when $k \geq k_{\min} > 0$:
$$M(t) \leq M(0) e^{-k_{\min} t}$$

*Proof*: Integrate modified FPK over $\Omega$:
$$\frac{dM}{dt} = \int_\Omega \frac{\partial m}{\partial t} dx = -\int_\Omega k(x) m(t,x) dx$$
using divergence theorem on flux and diffusion terms (boundary contributions vanish).

**Definition 3.4 (Evacuated Mass)**:
The cumulative evacuated mass is:
$$E(t) = \int_0^t \int_\Omega k(x) m(s,x) dx \, ds$$

**Conservation Law**:
$$M(t) + E(t) = M(0) \quad \forall t \in [0,T]$$

#### 3.1.2 Numerical Stability Analysis

**Proposition 3.5 (Stiffness of Absorption Term)**[^5]:
The absorption term $-k(x)m$ introduces **stiffness** when $\max_x k(x) \gg 1$.

For explicit Euler discretization, stability requires:
$$\Delta t \cdot \max_{x \in \Omega} k(x) < 1$$

This can be severely restrictive when exits have high capacity ($\kappa \gg 1$).

**Solution 3.6 (Semi-Implicit Treatment)**:
Treat absorption implicitly:
$$(1 + \Delta t \cdot k(x)) m^{n+1}(x) = m^n(x) + \Delta t \cdot [\text{transport} + \text{diffusion}]^n$$

This gives **unconditional stability** for the absorption term.

**Algorithm 3.7 (IMEX Scheme)**:
1. Compute transport-diffusion explicitly:
   $$\tilde{m} = m^n + \Delta t \left[-\nabla \cdot (m^n \nabla_p H^n) + \sigma \Delta m^n\right]$$
2. Apply implicit absorption:
   $$m^{n+1} = \frac{\tilde{m}}{1 + \Delta t \cdot k}$$

### 3.2 Approach 2: Target Domain with Terminal Payoff

**Idea**: Keep mass-conserving FPK, but give **large terminal reward** for reaching exit regions. Mass remains conserved, but concentrates at exits.

#### 3.2.1 Mathematical Formulation

**Definition 3.8 (Target Payoff Formulation)**:
Standard MFG system with modified terminal condition:

**MFG System**:
$$\begin{cases}
-\frac{\partial u}{\partial t} + H(x, \nabla u, m) = f(x, m), & (t,x) \in [0,T] \times \Omega \\
\frac{\partial m}{\partial t} - \nabla \cdot (m \nabla_p H) - \sigma \Delta m = 0, & (t,x) \in [0,T] \times \Omega \\
u(T,x) = g_{\text{exit}}(x), & x \in \Omega \\
m(0,x) = m_0(x), & x \in \Omega
\end{cases}$$

**Exit Target Function**:
$$g_{\text{exit}}(x) = \begin{cases}
-C_{\text{exit}} & \text{if } d(x, \Gamma_{\text{exit}}) < r_{\text{exit}} \\
0 & \text{otherwise}
\end{cases}$$

where:
- $C_{\text{exit}} \gg 1$ is large reward magnitude (e.g., 100-1000)
- $r_{\text{exit}} > 0$ is exit capture radius
- Negative cost = reward (agents minimize cost)

**Definition 3.9 (Exit Region)**:
$$\Omega_{\text{exit}} = \{x \in \Omega : d(x, \Gamma_{\text{exit}}) < r_{\text{exit}}\}$$

**Remark 3.10**: Mass remains conserved:
$$\int_\Omega m(t,x) dx = \int_\Omega m_0(x) dx = 1, \quad \forall t \in [0,T]$$

but concentration at exits increases:
$$\int_{\Omega_{\text{exit}}} m(T,x) dx \to 1 \text{ as } C_{\text{exit}} \to \infty$$

#### 3.2.2 Evacuation Metrics

**Definition 3.11 (Success Rate)**:
The **evacuation success rate** at time $T$ is:
$$S(T) = \frac{\int_{\Omega_{\text{exit}}} m(T,x) dx}{\int_\Omega m_0(x) dx}$$

Ideally, $S(T) \approx 1$ (all mass concentrated at exits).

**Definition 3.12 (Expected Evacuation Time)**[^6]:
For initial position $x_0$, the expected evacuation time is approximated by:
$$\tau(x_0) \approx u(0, x_0)$$

where $u$ is the value function. The **average evacuation time** is:
$$\bar{\tau} = \int_\Omega u(0,x) m_0(x) dx$$

**Theorem 3.13 (Concentration at Exits)**:
As $C_{\text{exit}} \to \infty$ with fixed $f(x,m) = c > 0$, the equilibrium density $m(T,x)$ concentrates on $\Omega_{\text{exit}}$:
$$\lim_{C_{\text{exit}} \to \infty} S(T) = 1$$

*Proof Sketch*: Large terminal reward creates steep gradient $|\nabla u(T,x)| \approx C_{\text{exit}}/r_{\text{exit}}$ near $\partial \Omega_{\text{exit}}$, driving flux toward exits throughout $[0,T]$. See [^7] for rigorous analysis in optimal transport framework.

#### 3.2.3 Advantages and Limitations

**Advantages**:
- ✅ **Numerically stable**: Standard mass-conserving solvers apply
- ✅ **No stiffness**: No absorption term
- ✅ **Well-tested**: Leverages existing MFG convergence theory[^3]
- ✅ **Clear game interpretation**: Agents minimize expected cost to reach exit

**Limitations**:
- ❌ **Mass artificially conserved**: People don't physically leave (mathematical artifact)
- ❌ **Indirect metric**: Must measure concentration $S(T)$, not actual evacuation count
- ❌ **Parameter tuning**: Requires careful choice of $C_{\text{exit}}$ and $r_{\text{exit}}$

**Recommended for Production**: This approach is most robust for practical applications[^8].

### 3.3 Approach 3: Free Boundary Problem

**Idea**: Model evacuation front as **moving boundary** where occupied region $\Omega(t)$ shrinks over time.

#### 3.3.1 Mathematical Formulation

**Definition 3.14 (Free Boundary MFG)**:
The occupied region evolves:
$$\Omega(t) = \{x \in \Omega : m(t,x) > 0\}, \quad \Gamma(t) = \partial \Omega(t) \cap \Omega$$

**MFG System with Free Boundary**:
$$\begin{cases}
-\frac{\partial u}{\partial t} + H(x, \nabla u, m) = f(x,m), & x \in \Omega(t), \, t \in [0,T] \\
\frac{\partial m}{\partial t} - \nabla \cdot (m \nabla_p H) - \sigma \Delta m = 0, & x \in \Omega(t), \, t \in [0,T] \\
m(t,x) = 0, \quad u(t,x) = g(x), & x \in \Gamma(t) \cap \Gamma_{\text{exit}}
\end{cases}$$

**Stefan-Like Condition**[^9]:
The front $\Gamma(t)$ evolves with normal velocity:
$$v_n = -\sigma \frac{\nabla m \cdot n}{m} \quad \text{on } \Gamma(t)$$

where $n$ is outward normal to $\Omega(t)$.

#### 3.3.2 Level Set Formulation

**Definition 3.15 (Level Set Representation)**[^10]:
Represent $\Gamma(t)$ implicitly via level set function $\phi(t,x)$:
$$\Gamma(t) = \{x : \phi(t,x) = 0\}$$
$$\Omega(t) = \{x : \phi(t,x) > 0\}$$

**Evolution Equation**:
$$\frac{\partial \phi}{\partial t} + V(t,x) |\nabla \phi| = 0$$

where $V(t,x)$ is front speed derived from FPK solution.

#### 3.3.3 Numerical Methods

**Remark 3.16**: Free boundary MFG is **highly challenging** numerically:
- Requires coupled solution of PDEs + front tracking
- Level set methods (Osher-Sethian[^10]) or phase field approaches (Allen-Cahn)
- **Not production-ready** for evacuation applications

---

## 4. Application: Two-Door Room Evacuation

### 4.1 Problem Setup

**Geometry**:
- Room: $\Omega = [0,1] \times [0,1]$
- Doors: $\mathbf{d}_1 = (0.3, 1.0)$, $\mathbf{d}_2 = (0.7, 1.0)$ on top wall
- Initial crowd: $m_0(x,y) = \frac{1}{Z} \exp\left(-\frac{(x-0.5)^2 + (y-0.4)^2}{2\sigma_0^2}\right)$ with $\sigma_0 = 0.1$

### 4.2 Target Payoff Implementation (Approach 2)

**Terminal Cost** (dual-well structure):
$$g_{\text{exit}}(x,y) = -C_{\text{exit}} \cdot \mathbb{1}_{\Omega_{\text{exit}}}(x,y)$$

where exit regions are:
$$\Omega_{\text{exit}} = \bigcup_{j=1}^2 B_{r_{\text{exit}}}(\mathbf{d}_j) = \{(x,y) : \min_j \|(x,y) - \mathbf{d}_j\| < r_{\text{exit}}\}$$

Typical values: $C_{\text{exit}} = 500$, $r_{\text{exit}} = 0.15$.

**Hamiltonian with Congestion**[^11]:
$$H(x, p, m) = \frac{1}{2} |p|^2 + \gamma m |p|^2$$

where $\gamma > 0$ is **congestion coefficient** (slows movement in dense crowds).

**Velocity Reconstruction**:
$$v(t,x) = -\nabla_p H(x, \nabla u, m) = -(1 + 2\gamma m) \nabla u$$

Congestion term $(1 + 2\gamma m)$ increases effective "mass" of moving agents.

**Running Cost**:
$$f(x,m) = 1 \quad \text{(minimize time)}$$

### 4.3 Evacuation Metrics

**Success Rate** (Definition 3.11):
$$S(T) = \frac{\int_{\Omega_{\text{exit}}} m(T,x,y) dx dy}{\int_\Omega m_0(x,y) dx dy}$$

Numerical integration over grid cells in exit regions.

**Average Evacuation Time** (Definition 3.12):
$$\bar{\tau} = \int_\Omega u(0,x,y) m_0(x,y) dx dy$$

Weighted average of initial value function.

### 4.4 Expected Equilibrium Behavior

**Theorem 4.1 (Symmetry Breaking)**[^12]:
For symmetric geometry and initial condition, the MFG equilibrium may exhibit **spontaneous symmetry breaking**, with unequal door usage:
$$\int_{B_{r}(\mathbf{d}_1)} m(T,x) dx \neq \int_{B_{r}(\mathbf{d}_2)} m(T,x) dx$$

This depends on:
- Congestion strength $\gamma$
- Initial distribution concentration
- Noise level $\sigma$

**Physical Interpretation**: Herding behavior and congestion avoidance lead to **load balancing** at exits.

---

## 5. Anisotropic Extensions

Combine evacuation formulation with **anisotropic diffusion** for realistic directional crowd behavior[^13].

### 5.1 Anisotropic Hamiltonian

**Definition 5.1 (Anisotropic Evacuation Hamiltonian)**:
$$H(x, p, m) = \frac{1}{2} p^T A(x) p + \gamma m \cdot p^T A(x) p$$

where $A(x) \in \mathbb{R}^{d \times d}$ is **anisotropy matrix** (symmetric positive definite):
$$A(x) = \begin{bmatrix} 1 & \rho(x) \\ \rho(x) & 1 \end{bmatrix}$$

with $\rho(x) \in (-1, 1)$ encoding directional preference.

**Physical Interpretation**:
- $\rho > 0$: Diagonal movement preferred (corridors, channeling)
- $\rho < 0$: Axis-aligned movement preferred (open rooms)
- $\rho = 0$: Isotropic (standard)

### 5.2 Exit Channeling Model

**Definition 5.2 (Exit-Enhanced Anisotropy)**:
Near exits, enhance channeling:
$$\rho(x) = \rho_{\text{base}}(x) + \sum_{j=1}^{N_{\text{exit}}} c_j \exp\left(-\beta \|x - \mathbf{d}_j\|^2\right)$$

where:
- $\rho_{\text{base}}(x)$ is background anisotropy (e.g., room geometry)
- $c_j > 0$ is channeling strength at door $j$
- $\beta > 0$ controls spatial extent

**Remark 5.3**: Full mathematical treatment in `docs/theory/applications/anisotropic_mfg_mathematical_formulation.md`.

---

## 6. Numerical Methods and Stability

### 6.1 FDM Scheme Pairing for MFG

When using FDM solvers for both HJB and FP, the advection scheme selection is **mathematically determined** by the equation structure.

**HJB (non-divergence form):** $-\partial_t u + \mathbf{b} \cdot \nabla u = 0$
- Must use **Gradient schemes**: `gradient_upwind` or `gradient_centered`
- Divergence schemes are structurally impossible (no flux to diverge)

**FP (divergence form, adjoint of HJB):** $\partial_t m + \nabla \cdot (m \mathbf{b}) = 0$
- Must use **Divergence schemes**: `divergence_upwind` or `divergence_centered`
- Using gradient schemes is a type error that violates mass conservation

**Required pairings:**
| HJB Scheme | FP Scheme | Properties |
|------------|-----------|------------|
| `gradient_upwind` | `divergence_upwind` | Monotone, positivity-preserving |
| `gradient_centered` | `divergence_centered` | Second-order, may oscillate |

```python
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver

hjb_solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
fp_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")
```

**Reference:** See `docs/theory/adjoint_discretization_mfg.md` §8 for detailed analysis.

### 6.2 Semi-Lagrangian Scheme for Evacuation MFG

**Algorithm 6.1 (Semi-Lagrangian + IMEX for Absorption)**[^14]:

**Initialization**:
- Grid: $\{x_i\}_{i=1}^{N_x}$, time steps $\{t^n\}_{n=0}^{N_t}$ with $\Delta t = T/N_t$
- $u^{N_t}(x) = g_{\text{exit}}(x)$, $m^0(x) = m_0(x)$

**Time Loop** ($n = N_t - 1, \ldots, 0$):

1. **HJB Step** (backward, semi-Lagrangian):
   - Compute characteristics: $x^*(x_i) = x_i - \Delta t \nabla_p H(x_i, \nabla u^{n+1}, m^n)$
   - Update: $u^n(x_i) = \text{Interpolate}(u^{n+1}, x^*) + \Delta t \cdot f(x_i, m^n)$

2. **FPK Step** (forward):
   - **Explicit transport-diffusion**:
     $$\tilde{m}^{n+1} = m^n + \Delta t \left[-\nabla \cdot (m^n \nabla_p H) + \sigma \Delta m^n\right]$$
   - **Implicit absorption** (if using Approach 1):
     $$m^{n+1} = \frac{\tilde{m}^{n+1}}{1 + \Delta t \cdot k}$$
   - **No absorption** (if using Approach 2): $m^{n+1} = \tilde{m}^{n+1}$

**Proposition 6.2 (Stability)**:
The IMEX scheme is **unconditionally stable** for absorption term, with CFL condition only on transport:
$$\Delta t \cdot \frac{\max |\nabla_p H|}{\Delta x} < 1$$

### 6.3 Comparison of Approaches

| Approach | Mass Behavior | CFL Restriction | Stiffness | Production-Ready | Realism |
|:---------|:--------------|:----------------|:----------|:-----------------|:--------|
| **Absorbing Boundaries** | Decreases | Transport only (IMEX) | High (needs IMEX) | ⚠️ Moderate | High |
| **Target Payoff** | Conserved | Standard | None | ✅ Yes | Medium |
| **Free Boundary** | Decreases | Very restrictive | Very High | ❌ No | High |

**Production Recommendation**[^8]: **Target Payoff (Approach 2)** for practical applications.

---

## References

[^1]: Achdou, Y., & Capuzzo-Dolcetta, I. (2010). "Mean field games: numerical methods." *SIAM Journal on Numerical Analysis*, 48(3), 1136-1162.

[^2]: Lions, P.-L. (2007-2011). *Cours au Collège de France: Théorie des jeux à champs moyen*. Available: https://www.college-de-france.fr

[^3]: Cardaliaguet, P. (2013). "Notes on Mean Field Games." Available: https://www.ceremade.dauphine.fr/~cardalia/MFG20130420.pdf

[^4]: Pazy, A. (1983). *Semigroups of Linear Operators and Applications to Partial Differential Equations*. Springer-Verlag.

[^5]: Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer-Verlag.

[^6]: Fleming, W. H., & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions* (2nd ed.). Springer.

[^7]: Benamou, J.-D., & Carlier, G. (2015). "Augmented Lagrangian methods for transport optimization, mean field games and degenerate elliptic equations." *Journal of Optimization Theory and Applications*, 167(1), 1-26.

[^8]: Lachapelle, A., & Wolfram, M.-T. (2011). "On a mean field game approach modeling congestion and aversion in pedestrian crowds." *Transportation Research Part B: Methodological*, 45(10), 1572-1589.

[^9]: Friedman, A. (1982). *Variational Principles and Free-Boundary Problems*. Wiley-Interscience.

[^10]: Osher, S., & Sethian, J. A. (1988). "Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations." *Journal of Computational Physics*, 79(1), 12-49.

[^11]: Maury, B., Roudneff-Chupin, A., & Santambrogio, F. (2010). "A macroscopic crowd motion model of gradient flow type." *Mathematical Models and Methods in Applied Sciences*, 20(10), 1787-1821.

[^12]: Dogbé, C. (2010). "Modeling crowd dynamics by the mean-field limit approach." *Mathematical and Computer Modelling*, 52(9-10), 1506-1520.

[^13]: Appert-Rolland, C., Cividini, J., Hilhorst, H. J., & Degond, P. (2020). "Microscopic and macroscopic dynamics of a pedestrian cross-flow: Part I, experimental analysis." *Physica A: Statistical Mechanics and its Applications*, 549, 124295.

[^14]: Carlini, E., & Silva, F. J. (2015). "A semi-Lagrangian scheme for a degenerate second order mean field game system." *Discrete and Continuous Dynamical Systems*, 35(9), 4269-4292.

[^15]: Hoogendoorn, S. P., & Bovy, P. H. L. (2004). "Pedestrian route-choice and activity scheduling theory and models." *Transportation Research Part B: Methodological*, 38(2), 169-190.

### Additional References

**Numerical Methods for MFG with Boundaries**:
- Achdou, Y., Camilli, F., & Capuzzo-Dolcetta, I. (2012). "Mean field games: numerical methods for the planning problem." *SIAM Journal on Control and Optimization*, 50(1), 77-109.
- Camilli, F., & Silva, F. J. (2018). "A semi-discrete approximation for a first order mean field game problem." *Networks and Heterogeneous Media*, 13(1), 149-174.

**Crowd Dynamics and Pedestrian Flow**:
- Helbing, D., Farkas, I., & Vicsek, T. (2000). "Simulating dynamical features of escape panic." *Nature*, 407(6803), 487-490.
- Hughes, R. L. (2002). "A continuum theory for the flow of pedestrians." *Transportation Research Part B*, 36(6), 507-535.

**Optimal Control and Exit Strategies**:
- Bardi, M., & Capuzzo-Dolcetta, I. (1997). *Optimal Control and Viscosity Solutions of Hamilton-Jacobi-Bellman Equations*. Birkhäuser.
- Bressan, A., & Piccoli, B. (2007). *Introduction to the Mathematical Theory of Control*. AIMS Series on Applied Mathematics, Vol. 2.

---

## Implementation Notes

### Code References

**Examples**:
- `examples/basic/geometry/2d_crowd_motion_fdm.py` - 2D crowd motion with FDM
- `mfg-research/experiments/crowd_evacuation_2d/` - Full evacuation experiment (separate repo)

**Theory Documents**:
- `docs/theory/applications/anisotropic_mfg_mathematical_formulation.md` - Anisotropic framework
- `docs/theory/adjoint_discretization_mfg.md` - FDM scheme pairing, adjoint structure (§8)
- `docs/theory/semi_lagrangian_methods_for_hjb.md` - Semi-Lagrangian methods

**Core Solvers**:
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py` - HJB FDM solver with `advection_scheme` parameter
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py` - FP FDM solver with `advection_scheme` parameter
- `mfg_pde/alg/numerical/coupling/` - Coupled MFG iteration (FixedPointIterator)

### Production Recommendations

For **practical evacuation simulations**, use:

1. **Target Payoff Approach** (§3.2) with large terminal reward at exits
2. **Congestion Hamiltonian** $H = \frac{1}{2}|p|^2 + \gamma m |p|^2$ with $\gamma \in [0.1, 0.5]$
3. **Correct FDM Scheme Pairing** (§6.1): `gradient_upwind` (HJB) + `divergence_upwind` (FP)
4. **Success Metric** $S(T)$ (Definition 3.11) for evacuation effectiveness
5. **Anisotropic Extensions** (§5) for realistic crowd channeling (optional)

**Avoid**:
- Explicit absorption (Approach 1) unless implementing IMEX schemes
- Using `gradient_*` schemes for FP (violates mass conservation)

---

**Document Status**: Enhanced with FDM scheme pairing requirements
**Last Updated**: December 2025
**Version**: 2.1 (Added §6.1 FDM Scheme Pairing, fixed dead references)
**Notation**: Follows `NOTATION_STANDARDS.md`
