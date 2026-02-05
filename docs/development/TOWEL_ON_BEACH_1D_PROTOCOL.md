# 1D Towel-on-the-Beach Problem Protocol

## Problem Overview

The **Towel-on-the-Beach** problem (French: "Serviette sur la plage"), also known as the **Beach Bar Process**, is a classical MFG benchmark introduced by Guéant (2009). It models beachgoers choosing where to place their towels on a beach strip.

**Physical Interpretation**:
- Agents choose positions on a 1D beach $x \in [0, L]$
- An attractive point (ice cream stall) at position $x_{stall}$ draws agents
- Agents dislike crowded areas (congestion aversion)
- Equilibrium emerges from the tension between attraction and crowd aversion

**This problem is mathematically distinct from the El Farol Bar problem** (attendance coordination). Here, agents choose **continuous spatial positions**, not binary attendance decisions.

---

## Mathematical Formulation

### Domain and State Space

$$\Omega = [0, L], \quad L = 1.0$$

- $x_{stall} \in [0, L]$: Location of attractive point (ice cream stall)
- Common choice: $x_{stall} = 0$ (water's edge) or $x_{stall} = 0.5$ (center)

### Agent Dynamics (Microscopic Model)

Each agent controls their velocity $u_t$:

$$dX_t = u_t \, dt + \sigma \, dW_t$$

where $u_t$ is the control input (velocity), $\sigma$ is constant volatility, and $W_t$ is standard Brownian motion.

### Agent Objective Functional

Each agent minimizes expected cost over horizon $[0, T]$:

$$J(u) = \mathbb{E} \left[ \int_0^T \left( \underbrace{\frac{1}{2}u_t^2}_{\text{Kinetic cost}} + \underbrace{V(X_t)}_{\text{Position preference}} + \underbrace{g(m(t, X_t))}_{\text{Crowd aversion}} \right) dt + \underbrace{\Phi(X_T, m(T, X_T))}_{\text{Terminal cost}} \right]$$

### Running Cost (Lagrangian)

The instantaneous cost for an agent at position $x$ with velocity $u$ in density $m$:

$$L(x, u, m) = \underbrace{V(x)}_{\text{Position cost}} + \underbrace{\lambda \ln(m(x))}_{\text{Congestion cost}} + \underbrace{\frac{1}{2}u^2}_{\text{Movement cost}}$$

**Potential function** (attraction to stall):
$$V(x) = |x - x_{stall}|$$

Or quadratic variant:
$$V(x) = \frac{1}{2}(x - x_{stall})^2$$

**Key parameter**: $\lambda > 0$ is the **crowd aversion parameter**. It governs the balance between proximity preference and congestion avoidance.

### Hamiltonian Derivation

From the Lagrangian, the Hamiltonian is obtained by optimizing over control $u$:

$$H(x, p, m) = \sup_{u} \left\{ -p \cdot u - L(x, u, m) \right\}$$

With optimal control $u^* = -p$:

$$H(x, p, m) = \frac{1}{2}p^2 - V(x) - \lambda \ln(m)$$

### Macroscopic Model: HJB-FP System

As $N \to \infty$, the Nash equilibrium is characterized by the coupled HJB-Fokker-Planck system.

#### HJB Equation (Backward in time)

$$-\frac{\partial U}{\partial t} - \frac{\sigma^2}{2}\frac{\partial^2 U}{\partial x^2} + \frac{1}{2}\left(\frac{\partial U}{\partial x}\right)^2 = V(x) + \lambda \ln(m)$$

with terminal condition $U(T, x) = \Phi(x, m(T))$.

#### Fokker-Planck Equation (Forward in time)

$$\frac{\partial m}{\partial t} - \frac{\sigma^2}{2} \frac{\partial^2 m}{\partial x^2} - \frac{\partial}{\partial x}\left(m \frac{\partial U}{\partial x}\right) = 0$$

with initial condition $m(0, x) = m_0(x)$.

The optimal control is $u^*(t, x) = -\nabla U(t, x)$.

### Boundary Conditions

| Location | HJB ($U$) | FP ($m$) | Physical Meaning |
|:---------|:----------|:---------|:-----------------|
| $x = 0$ | Neumann: $\frac{\partial U}{\partial x} = 0$ | No-flux | Reflecting boundary |
| $x = L$ | Neumann: $\frac{\partial U}{\partial x} = 0$ | No-flux | Reflecting boundary |

**Both boundaries are reflecting** — agents cannot leave the beach domain.

**Explicit No-Flux Condition for FP**:
$$\frac{\sigma^2}{2} \frac{\partial m}{\partial n} + m \frac{\partial U}{\partial n} = 0 \quad \text{on } \partial \Omega$$

This ensures the probability flux $J = -\frac{\sigma^2}{2}\nabla m + m \nabla U$ vanishes at boundaries.

### ⚠️ Boundary Condition Consistency Issue

**WARNING**: The standard Neumann BC ($\partial U/\partial n = 0$) for HJB may NOT be consistent with the equilibrium solution!

At equilibrium (Boltzmann-Gibbs), the drift is:
$$\alpha^* = -\nabla U^* = \frac{\sigma^2}{2 T_{eff}} \nabla V$$

For $V(x) = |x - x_{stall}|$ with $x_{stall} = 0$:
- At $x = 0$: $\nabla V = 0$ (kink point) → $\nabla U^* = 0$ ✓ (Neumann OK)
- At $x = L$: $\nabla V = +1$ → $\nabla U^* = \frac{\sigma^2}{2 T_{eff}} \neq 0$ ✗ (Neumann WRONG)

**Consequence**: Imposing $\partial U/\partial n = 0$ at $x = L$ prevents convergence to the true equilibrium. The numerical solution will be **flatter** than the analytic Boltzmann-Gibbs.

**Experimental Validation (exp14b)**: Tested BC consistency hypothesis with two configurations:

| Configuration | Domain | x_stall | Final Error | Notes |
|:--------------|:-------|:--------|:------------|:------|
| Centered | [-0.5, 0.5] | 0.0 | **3.70%** | Neumann consistent at both boundaries |
| Boundary stall | [0, 1] | 0.0 | **9.81%** | Neumann inconsistent at x=1 |

**Result**: 2.65x error improvement with centered stall. Error in boundary case is concentrated at x=0 where Neumann forces ∇U=0 while equilibrium requires ∇U≠0.

#### ✅ Solution: Adjoint-Consistent Boundary Conditions (Issue #574)

**Implementation Status**: Available in v0.17.1+ (proper Robin BC framework architecture)

The HJB solver now supports **adjoint-consistent Robin BC** that couples to the FP density gradient at reflecting boundaries. This fixes the equilibrium inconsistency when stall points occur at domain boundaries.

**Mathematical Formula**: At reflecting boundaries with zero total flux $J \cdot n = 0$ where $J = -\frac{\sigma^2}{2}\nabla m + m \alpha$, the adjoint-consistent BC for quadratic Hamiltonians is:

$$\frac{\partial U}{\partial n} = -\frac{\sigma^2}{2} \frac{\partial \ln(m)}{\partial n}$$

**Architecture**: Uses the existing Robin BC framework (`BCType.ROBIN` with $\alpha=0$, $\beta=1$) for dimension-agnostic support. The solver automatically creates proper `BoundaryConditions` objects with Robin BC segments when `bc_mode="adjoint_consistent"`.

**Usage**:
```python
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.alg.numerical.coupling import FixedPointIterator
from mfg_pde.geometry.boundary import neumann_bc

# Create problem with reflecting boundaries
problem = MFGProblem(
    domain=[0, 1],
    Nx=50,
    Nt=50,
    T=1.0,
    sigma=0.2,
    boundary_conditions=neumann_bc(dimension=1),
)

# Standard BC mode (default, backward compatible)
hjb_solver_std = HJBFDMSolver(problem, bc_mode="standard")

# Adjoint-consistent BC mode (recommended for boundary stall)
hjb_solver_ac = HJBFDMSolver(problem, bc_mode="adjoint_consistent")

# Use in Picard iteration (no changes needed)
iterator = FixedPointIterator(problem, hjb_solver_ac, FPFDMSolver(problem))
result = iterator.solve(max_iterations=30, tolerance=1e-6)
```

**When to Use Adjoint-Consistent BC**:
- ✅ Stall point at domain boundary ($x_{stall} = 0$ or $x_{stall} = L$)
- ✅ Reflecting boundaries (Neumann/no-flux BC)
- ✅ Near equilibrium or high-accuracy simulations
- ❌ Not needed when stall is interior or with periodic/Dirichlet BC

**Performance Impact**: Negligible (<0.1% overhead). Often reduces total iterations due to better consistency.

**Validation Results**:
- Original validation (mfg-research exp14b): 2.13x convergence improvement
- Framework architecture validation: **1932x improvement** (standard BC diverges, adjoint-consistent converges)
- The dramatic improvement demonstrates the importance of BC consistency for equilibrium problems

**Implementation Details**:
- Core utilities: `mfg_pde/geometry/boundary/bc_coupling.py`
  - `create_adjoint_consistent_bc_1d()`: Creates Robin BC from density
  - `compute_boundary_log_density_gradient_1d()`: Computes ∂ln(m)/∂n
- Solver integration: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`
  - Automatically creates BC when `bc_mode="adjoint_consistent"`
- Design document: `docs/archive/bc_completed_2026-02/issue_574_robin_bc_design.md` (archived, implemented v0.17.1)
- GitHub Issue: #574

**Alternative Approaches** (for reference):
1. **Natural BC**: Don't impose anything on $U$ at boundary (not currently supported)
2. **State-constrained**: Use viscosity solution techniques (requires solver modifications)
3. **Centered stall**: Place $x_{stall}$ at domain center to make Neumann consistent at both boundaries (simple workaround)

### Initial and Terminal Conditions

**FP Initial Condition**: $m(0, x) = m_0(x)$
- Uniform: $m_0(x) = 1/L$
- Gaussian: $m_0(x) \propto \exp(-\alpha(x - x_0)^2)$

**HJB Terminal Condition**: $U(T, x) = g(x)$
- Zero: $g(x) = 0$
- Position-dependent: $g(x) = V(x)$

---

## Multi-Peak Analysis (Critical for Debugging!)

### When Multi-Peaks Do NOT Occur

**In the standard Towel-on-Beach game with repulsive congestion ($g'(m) > 0$) and a unimodal potential $V(x)$, multiple peaks NEVER arise spontaneously.**

**Mathematical Reason**: The energy functional is strictly convex:
$$\mathcal{E}[m] = \int \left( m V + G(m) + \frac{\sigma^2}{2} m \ln m \right) dx$$

If $V$ is convex (bowl-shaped) and $G$ is convex (standard congestion), the sum is convex. There is a **unique minimizer** $m^*$.

**Physical Behavior**: As you increase repulsion $\lambda$, the single peak simply gets **wider and flatter** (the "table-top" or "bathtub" effect), but it will **NOT split into two peaks**.

> **Debugging Implication**: If your numerical solver produces multiple peaks with a single-well potential and repulsive congestion, this is a **NUMERICAL BUG**, not a physical phenomenon!

### When Multi-Peaks DO Occur

#### 1. Multi-Well Potential (Environment-Driven)

If the beach has multiple attractive spots (e.g., ice cream at $x=-1$ and bar at $x=1$):

$$V(x) = h(x^2 - 1)^2$$

This double-well potential naturally produces two density peaks.

**Interaction with congestion**:
- Low $\lambda$: Sharp peaks at both minima
- High $\lambda$: Peaks widen; valley fills as agents are pushed out of prime spots

#### 2. Attractive Coupling (Clustering)

If agents *want* to be near others ($g'(m) < 0$), spontaneous clustering occurs even with flat potential $V(x) = 0$.

**Warning**: This makes the HJB-FP system mathematically unstable and requires specialized monotonizing schemes.

---

## Equilibrium Structure

### Equilibrium Types (for Single-Well $V(x)$)

The equilibrium density depends qualitatively on the crowd aversion parameter $\lambda$:

#### 1. Single Peak Equilibrium ($\lambda$ small)
- **Pattern**: Density maximum at stall location $x_{stall}$
- **Interpretation**: Weak congestion penalty allows concentration
- **Shape**: Sharp peak centered at $x_{stall}$

#### 2. Flattened Peak / "Table-Top" ($\lambda$ large)
- **Pattern**: Density still maximum at stall, but much flatter profile
- **Interpretation**: Strong congestion spreads agents toward boundaries
- **Shape**: Broad, flat-topped distribution

#### 3. Crater Equilibrium (Requires $x_{stall}$ at center)
- **Pattern**: Density **minimum** at stall, peaks on both sides
- **Interpretation**: With stall at center ($x_{stall} = L/2$) and very high $\lambda$
- **Shape**: Two peaks flanking $x_{stall}$
- **Note**: This is still a single equilibrium, not multi-peak in the pathological sense

### Analytic Stationary Solution (Validation!)

As $t \to \infty$, time derivatives vanish. The stationary density $m^*(x)$ can be derived analytically for the logarithmic congestion case.

#### Complete Derivation of Boltzmann-Gibbs Equilibrium

**Step 1: Stationary HJB-FP System**

At equilibrium, time derivatives vanish. The coupled system becomes:

$$\text{HJB:} \quad -\frac{\sigma^2}{2} \partial_{xx} U^* + \frac{1}{2}|\partial_x U^*|^2 = V(x) + \lambda \ln(m^*)$$

$$\text{FP:} \quad -\frac{\sigma^2}{2} \partial_{xx} m^* - \partial_x(m^* \cdot \partial_x U^*) = 0$$

**Step 2: Zero-Flux Condition**

The FP equation can be written in conservation form:
$$\partial_x J = 0, \quad \text{where } J = -\frac{\sigma^2}{2} \partial_x m^* + m^* \cdot \alpha^*$$

and $\alpha^* = -\partial_x U^*$ is the optimal drift (velocity).

With reflecting boundaries (no-flux: $J = 0$ at $x = 0$ and $x = L$), we have $J = 0$ **everywhere** in the domain:
$$\boxed{-\frac{\sigma^2}{2} \partial_x m^* + m^* \cdot \alpha^* = 0}$$

**Step 3: Equilibrium Drift-Density Relation**

Solving for the drift:
$$\alpha^* = \frac{\sigma^2}{2} \frac{\partial_x m^*}{m^*} = \frac{\sigma^2}{2} \partial_x (\ln m^*)$$

This is the **fluctuation-dissipation relation**: at equilibrium, drift exactly balances diffusion.

**Step 4: Substitution into Stationary HJB**

Since $\alpha^* = -\partial_x U^*$, we have:
$$\partial_x U^* = -\frac{\sigma^2}{2} \partial_x (\ln m^*)$$

Integrating:
$$U^*(x) = -\frac{\sigma^2}{2} \ln m^*(x) + C_1$$

**Step 5: Verify Consistency with HJB**

Substituting into the stationary HJB:
$$-\frac{\sigma^2}{2} \partial_{xx} U^* + \frac{1}{2}|\partial_x U^*|^2 = V(x) + \lambda \ln(m^*)$$

Computing derivatives:
- $\partial_x U^* = -\frac{\sigma^2}{2} \frac{\partial_x m^*}{m^*}$
- $\partial_{xx} U^* = -\frac{\sigma^2}{2} \left( \frac{\partial_{xx} m^*}{m^*} - \frac{(\partial_x m^*)^2}{(m^*)^2} \right)$

After substitution and simplification (using the chain rule and cancellations), the HJB reduces to:
$$\frac{\sigma^2}{2} \ln m^* + \lambda \ln m^* = -V(x) + C_2$$

**Step 6: Solve for $m^*(x)$**

Combining logarithmic terms:
$$\left(\lambda + \frac{\sigma^2}{2}\right) \ln m^*(x) = -V(x) + C_2$$

Define the **effective temperature**:
$$\boxed{T_{eff} = \lambda + \frac{\sigma^2}{2}}$$

Exponentiating:
$$m^*(x) = \exp\left(\frac{-V(x) + C_2}{T_{eff}}\right) = e^{C_2/T_{eff}} \cdot \exp\left(-\frac{V(x)}{T_{eff}}\right)$$

**Step 7: Normalization**

The normalization constant $Z$ (partition function) is determined by $\int_0^L m^*(x) dx = 1$:
$$Z = \int_0^L \exp\left(-\frac{V(x)}{T_{eff}}\right) dx$$

#### Final Result: Boltzmann-Gibbs Distribution

$$\boxed{m^*(x) = \frac{1}{Z} \exp\left(-\frac{V(x)}{T_{eff}}\right), \quad T_{eff} = \lambda + \frac{\sigma^2}{2}}$$

where:
- $V(x) = c_1 |x - x_{stall}|$ is the position cost (potential)
- $\lambda = c_2$ is the crowd aversion parameter
- $\sigma$ is the diffusion coefficient
- $Z = \int \exp(-V/T_{eff}) dx$ is the partition function

#### Physical Interpretation

| Component | Physical Meaning |
|:----------|:-----------------|
| $V(x)/T_{eff}$ | Dimensionless "energy" at position $x$ |
| $T_{eff} = \lambda + \sigma^2/2$ | **Effective temperature** combining diffusion and crowd aversion |
| $\lambda$ contribution | Crowd aversion acts like **additional thermal noise** |
| $\sigma^2/2$ contribution | Brownian diffusion contribution |
| $Z$ | Partition function (ensures $\int m^* = 1$) |

**Key insight**: The crowd aversion term $\lambda \ln(m)$ in the running cost produces the **same effect as additional diffusion**. Both mechanisms spread agents away from the peak. The equilibrium doesn't distinguish between "agents avoiding crowds" and "agents randomly diffusing" — they are mathematically equivalent at stationarity.

#### Explicit Formula for Linear Potential

For $V(x) = c_1 |x - x_{stall}|$ with $x_{stall} = 0$ on domain $[0, L]$:

$$m^*(x) = \frac{c_1/T_{eff}}{\exp(c_1 L/T_{eff}) - 1} \exp\left(-\frac{c_1 x}{T_{eff}}\right)$$

This is an **exponential decay** from $x = 0$ (stall location) toward $x = L$.

#### Why This is a TRUE Analytic Solution

The Boltzmann-Gibbs formula is **exact** (not an approximation) because:
1. It satisfies the stationary FP equation exactly (zero flux)
2. It satisfies the stationary HJB equation exactly (verified by substitution)
3. The only numerical step is computing the normalization integral $Z$, which for smooth exponentials has negligible quadrature error (~$10^{-10}$)

**Validation error = Numerical solver error**, not comparison against another approximation.

#### Special Case: Linear Congestion

For $g(m) = \lambda m$, the implicit relation is:
$$\lambda m^* + \frac{\sigma^2}{2} \ln m^* = C - V(x)$$

This can be solved using the **Lambert W function**:
$$m^*(x) = \frac{\sigma^2}{2\lambda} W\left( \frac{2\lambda}{\sigma^2} \exp\left( \frac{2(C - V(x))}{\sigma^2} \right) \right)$$

**Practical Algorithm**: Use a numerical root finder at each grid point, then adjust $C$ to satisfy normalization.

### Thomas-Fermi Limit ($\sigma \to 0$)

In the zero-noise limit with linear congestion $\kappa m$:
$$m^*(x) = \max\left(0, \frac{C - V(x)}{\kappa}\right)$$

This gives an **inverted parabola** with compact support (drops to zero at boundaries).

---

## Numerical Algorithm

### Discretization Setup (Finite Difference)

**Space**: Grid $x_i = i \Delta x$ for $i = 0, \ldots, N_x$, where $\Delta x = L / N_x$.

**Time**: Grid $t_n = n \Delta t$ for $n = 0, \ldots, N_t$, where $\Delta t = T / N_t$.

**Numerical Scheme**:
- **Implicit Euler** for time stepping (stability on diffusion term)
- **Upwind scheme** for advection term ($\nabla \phi \cdot \nabla m$) in FP equation (ensures positivity of mass)
- **Central differences** for diffusion term

### Fixed-Point Iteration (Picard Method)

Since HJB depends on $m$ and FP depends on $U$, we use iterative coupling.

#### Algorithm

**Step 0: Initialization**
- Guess initial flow of distributions $m^{(0)}(t, x)$ (e.g., $m^{(0)}(t, x) = m_0(x)$ for all $t$)

**Step k: Iteration**

1. **Solve HJB (Backward)**: Using $m^{(k-1)}$, solve for $U^{(k)}$ from $t = T$ to $t = 0$:
   $$-\frac{U_i^{n} - U_i^{n-1}}{\Delta t} - \frac{\sigma^2}{2} D_{xx} U_i^{n-1} + \frac{1}{2}|D_x U_i^{n-1}|^2 = V(x_i) + \lambda \ln(m^{(k-1)}(t_n, x_i))$$

2. **Solve FP (Forward)**: Using $U^{(k)}$, solve for $m^{(k)}$ from $t = 0$ to $t = T$:
   $$\frac{m_i^{n+1} - m_i^{n}}{\Delta t} - \frac{\sigma^2}{2} D_{xx} m_i^{n+1} - D_x(m_i^{n+1} D_x U_i^{(k)}) = 0$$

3. **Check Convergence**: Compute error $E = \max_{t,x} |m^{(k)} - m^{(k-1)}|$.
   If $E < \epsilon$ (tolerance), stop. Otherwise, set $k \leftarrow k + 1$ and repeat.

**Damping (Relaxation)**: For stability, use damped updates:
$$m^{(k)} \leftarrow \alpha \cdot m^{(k)}_{new} + (1 - \alpha) \cdot m^{(k-1)}$$

where $\alpha \in [0.3, 0.7]$ is the relaxation factor.

---

## HJB-FP Coupling: General Design Principle

### The Optimal Control as the Bridge

The HJB and FP equations are coupled through the **optimal control** (drift/advection velocity). This relationship is fundamental to MFG numerics.

#### General Case

For a general Hamiltonian $H(x, p, m)$, the optimal control is derived from:

$$\alpha^*(x, t) = -\partial_p H(x, \nabla U, m)$$

This is the **advection velocity** that appears in the Fokker-Planck equation:

$$\frac{\partial m}{\partial t} + \nabla \cdot (m \, \alpha^*) = \frac{\sigma^2}{2} \Delta m$$

#### Coupling Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  HJB Solver                                                      │
│  Input: m(t,x)                                                   │
│  Output: U(t,x) (value function)                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Optimal Control Computation                                     │
│  α*(t,x) = -∂_p H(x, ∇U, m)                                     │
│  This is the DRIFT / ADVECTION VELOCITY                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  FP Solver                                                       │
│  Input: α*(t,x) (drift velocity)                                │
│  Output: m(t,x) (density)                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Quadratic Hamiltonian (Special Case)

For the standard quadratic Hamiltonian:

$$H(x, p, m) = \frac{1}{2}|p|^2 - f(x, m)$$

We have $\partial_p H = p$, so:

$$\alpha^* = -p = -\nabla U$$

**This simplification only holds for quadratic control costs.** In this case, passing the value function $U$ to the FP solver and computing $-\nabla U$ internally is equivalent to passing the optimal control directly.

### Non-Quadratic Hamiltonians

For general Hamiltonians, $\alpha^* \neq -\nabla U$. Examples:

| Hamiltonian | Optimal Control |
|:------------|:----------------|
| $H = \frac{1}{2}p^2$ | $\alpha^* = -\nabla U$ |
| $H = \|p\|$ (L1 control) | $\alpha^* = -\text{sign}(\nabla U)$ |
| $H = \frac{1}{4}p^4$ | $\alpha^* = -(\nabla U)^{1/3}$ |
| State constraints | Projected gradient |

### Sign Convention Warning

The sign of optimal control depends on the **optimization direction** (sup vs inf):

| Formulation | Hamiltonian Definition | Optimal Control |
|:------------|:-----------------------|:----------------|
| Cost minimization | $H = \sup_\alpha \{-p \cdot \alpha - L\}$ | $\alpha^* = -\partial_p H$ |
| Profit maximization | $H = \inf_\alpha \{p \cdot \alpha + L\}$ | $\alpha^* = +\partial_p H$ |

**This protocol uses cost minimization**, hence $\alpha^* = -\partial_p H$.

### Implementation Implications

**For numerical solvers:**

1. **FP solver input should be the drift velocity $\alpha^*$**, not the value function $U$
2. The HJB solver computes $U$; a separate step computes $\alpha^* = -\partial_p H$
3. Only for quadratic $H$ can we pass $U$ and let the FP solver compute $-\nabla U$ internally
4. **The FP solver should be sign-agnostic** — caller computes correct $\alpha^*$ based on their convention

**Current FP FDM Solver Limitation:**
The production FP FDM solver assumes quadratic Hamiltonian. It accepts what it calls `drift_field` but internally treats it as $U$ and computes velocity as $-c \nabla U$. This works for Towel-on-Beach but **will not generalize** to non-quadratic problems.

**GFDM Solver Design:**
The GFDM solver uses a `drift_query` interface that explicitly queries for $\alpha^*(x, t)$ at each point, avoiding this limitation.

---

## Numerical Implementation

### Running Cost Function

```python
def towel_running_cost(
    x: np.ndarray,
    m: np.ndarray,
    x_stall: float = 0.0,
    lambda_crowd: float = 1.0
) -> np.ndarray:
    """
    Towel-on-beach running cost: position preference + logarithmic congestion.

    f(x, m) = |x - x_stall| + lambda * ln(m)

    Args:
        x: Spatial positions
        m: Agent density at each position
        x_stall: Location of attractive point (ice cream stall)
        lambda_crowd: Crowd aversion parameter

    Returns:
        Running cost at each point
    """
    m_safe = np.maximum(m, 1e-10)  # Regularize log singularity
    position_cost = np.abs(x - x_stall)
    congestion_cost = lambda_crowd * np.log(m_safe)
    return position_cost + congestion_cost
```

### Boundary Condition Setup

```python
from mfg_pde.geometry.boundary import BCSegment, BCType, mixed_bc
import numpy as np

L = 1.0

# Neumann (reflecting) on both ends
left_bc = BCSegment(
    name="left", bc_type=BCType.NEUMANN, value=0.0, boundary="x_min"
)
right_bc = BCSegment(
    name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max"
)

beach_bc = mixed_bc(
    segments=[left_bc, right_bc],
    dimension=1,
    domain_bounds=np.array([[0.0, L]]),
)
```

### Analytic Solution for Validation

```python
def analytic_stationary_density(
    x: np.ndarray,
    x_stall: float = 0.0,
    sigma: float = 0.2,
    lambda_crowd: float = 1.0
) -> np.ndarray:
    """
    Boltzmann-Gibbs stationary solution for logarithmic congestion.

    m*(x) = Z^{-1} exp(-V(x) / (sigma^2/2 + lambda))
    """
    effective_temp = sigma**2 / 2 + lambda_crowd
    V = np.abs(x - x_stall)  # Linear potential
    unnormalized = np.exp(-V / effective_temp)
    Z = np.trapezoid(unnormalized, x)
    return unnormalized / Z
```

---

## Validation Protocol

### Experimental Cases (Scenario Design)

Run the following cases to validate the physics of the model:

#### Case A: "Free Beach" (Baseline)
- **Parameters**: $\lambda = 0$ (no crowd aversion), $\sigma = 0.1$, $V(x) = \frac{1}{2}x^2$
- **Expectation**: Agents ignore each other and congregate at the minimum of $V(x)$. Final density should be a sharp Gaussian centered at $x = 0$.
- **Purpose**: Verify FP solver (pure diffusion with potential)

#### Case B: "Crowded Holiday"
- **Parameters**: $\lambda = 5.0$ (high crowd aversion), $\sigma = 0.1$, $V(x) = \frac{1}{2}x^2$
- **Expectation**: Density $m(T, x)$ should be much flatter than Case A. Agents spread toward boundaries to avoid high cost at center. This is the "flattening effect."
- **Purpose**: Verify congestion effect; compare with Boltzmann-Gibbs solution

#### Case C: "High Noise" (Windy Day)
- **Parameters**: $\lambda = 2.0$, $\sigma = 0.5$ (high diffusion)
- **Expectation**: Distribution will be flatter due to diffusion, even with moderate crowd aversion.
- **Purpose**: Validate stability of diffusion term in solver

#### Case D: "Double-Well Beach" (Multi-Peak Validation)
- **Parameters**: $V(x) = 2(x^2 - 1)^2$ on $[-2, 2]$, $\lambda = 0.5$, $\sigma = 0.3$
- **Expectation**: Two distinct peaks at $x = \pm 1$ (the potential minima). Density at $x = 0$ should be near zero.
- **Purpose**: Verify multi-peak solutions arise from environment (double-well), not numerical artifacts

### Test Cases Summary

| Case | Parameters | Expected Result | Purpose |
|:-----|:-----------|:----------------|:--------|
| **A. Free Beach** | $\lambda=0$, $V(x)=\frac{1}{2}x^2$ | Sharp Gaussian at $x=0$ | Verify FP + potential |
| **B. Crowded Holiday** | $\lambda=5$, $V(x)=\frac{1}{2}x^2$ | Flattened distribution | Verify congestion |
| **C. High Noise** | $\sigma=0.5$, $\lambda=2$ | Flat from diffusion | Verify diffusion stability |
| **D. Double-Well** | $V(x)=2(x^2-1)^2$ | Two peaks at $\pm 1$ | Verify environment-driven peaks |

### Validation Metrics

1. **Mass conservation**: $\int_0^L m(t,x) dx = 1$ (within tolerance ~1-2%)
2. **Stationary convergence**: $\|m^{num} - m^{analytic}\|_\infty < \epsilon$
3. **No-flux verification**: $J(t, 0) = J(t, L) = 0$
4. **Monotonicity** (single-peak case): $m(x_1) > m(x_2)$ when $|x_1 - x_{stall}| < |x_2 - x_{stall}|$
5. **Grid convergence**: Error should decrease with order $O(\Delta x)$ or $O(\Delta x^2)$

### Data Analysis & Visualization

For each case, produce:

1. **Density Evolution**: 3D surface plot or heatmap of $m(t, x)$ showing spreading/concentration over time
2. **Terminal Slice**: 2D plot comparing $m(T, x)$ across cases (A vs B proves the "Towel" game theory)
3. **Control Policy**: Plot $u^*(0, x) = -\nabla U(0, x)$; in Case B should show velocities pushing away from center
4. **Convergence Plot**: Picard iteration error vs iteration number

### Numerical Considerations

1. **Regularization**: Use $\ln(m + \epsilon)$ with $\epsilon \sim 10^{-10}$ to prevent singularities
2. **Grid resolution**: Fine discretization needed for sharp peaks and crater patterns
3. **Convergence**: Higher $\lambda$ may require more Picard iterations
4. **Damping**: Use relaxation factor 0.3-0.7 for stability
5. **CFL condition**: For advection stability, ensure $|u| \Delta t / \Delta x < 1$

---

## Comparison with Related Problems

| Aspect | Towel-on-Beach | Corridor Evacuation | El Farol Bar |
|:-------|:---------------|:--------------------|:-------------|
| **Decision** | Spatial position | Spatial position | Binary attendance |
| **State space** | Continuous $[0,L]$ | Continuous $[0,L]$ | Discrete {0,1} |
| **Attraction** | Ice cream stall | Exit | Social benefit |
| **Boundaries** | Neumann + Neumann | Neumann + Dirichlet | N/A |
| **Equilibrium** | Spatial distribution | Depletion | Attendance rate |

---

## Applications

1. **Urban Planning**: Retail location choice under congestion
2. **Traffic Flow**: Route selection with crowding costs
3. **Market Competition**: Firm location decisions (Hotelling model)
4. **Ecology**: Animal territory selection

---

## References

1. Guéant, O. (2009). *A reference case for mean field games models*. Journal de mathématiques pures et appliquées, 92(3), 276-294.
2. Lasry, J.-M. & Lions, P.-L. (2007). *Mean Field Games*. Japanese Journal of Mathematics, 2(1), 229-260.
3. Achdou, Y. & Capuzzo-Dolcetta, I. (2010). *Mean Field Games: Numerical Methods*. SIAM Journal on Numerical Analysis, 48(3), 1136-1162.
4. arXiv:2007.03458 - "Beach Bar Process" (RL formulation)
5. Ullmo, P., Swiecicki, I., & Gobron, T. (2019). *On the quadratic mean field games with double-well potential*. Journal of Statistical Mechanics: Theory and Experiment, 2019(2), 023207.

---

**Last Updated**: 2026-01-14

**Changelog**:
- 2026-01-14: Added complete step-by-step Boltzmann-Gibbs equilibrium derivation (Steps 1-7), including zero-flux condition, fluctuation-dissipation relation, effective temperature derivation, and physical interpretation. Cleaned up redundant content.
- 2026-01-13: Added multi-peak analysis (critical for debugging), agent objective functional, explicit FP no-flux BC formula, Picard iteration algorithm, stationary solution derivation, Lambert W function solution for linear congestion, experimental cases (A-D), data analysis guidelines.
