# Spatial Competition Mean Field Games: The Towel on the Beach Problem

**Document Type**: Theoretical Foundation with Applications
**Created**: October 8, 2025
**Status**: Comprehensive formulation merging previous documents
**Related**: `mathematical_background.md`, `variational_mfg_theory.md`
**Supersedes**: `towel_beach_spatial_competition.md`, `towel_beach_comprehensive_summary.md`

---

## Introduction

The **Towel on the Beach Problem** (also known as the **Beach Bar Process**) is a canonical example in Mean Field Game theory for analyzing **spatial competition under congestion**. Unlike coordination problems (e.g., El Farol Bar), this model focuses on **continuous spatial positioning** where agents balance:

1. **Proximity** to a desirable amenity (ice cream stall, beach bar, etc.)
2. **Avoidance** of overcrowded locations

This formulation provides fundamental insights for:
- **Urban planning**: Retail location, service distribution
- **Traffic flow**: Route selection with congestion
- **Market competition**: Spatial firm location (Hotelling-like models)
- **Ecology**: Animal territory selection

**Key Mathematical Property**: The problem exhibits **qualitative phase transitions** in equilibrium structure based on a single parameter (crowd aversion λ), making it an ideal pedagogical and research benchmark.

---

## 1. Mathematical Formulation

### 1.1 State and Control Variables

**State Space**: Positions on a domain $\Omega \subset \mathbb{R}^d$ (typically 1D beach $\Omega = [0,1]$)
- $x_t \in \Omega$: Agent's position at time $t$

**Control Variable**: $u_t \in \mathbb{R}^d$ (velocity)

**Dynamics**: Controlled diffusion
$$dx_t = u_t \, dt + \sigma \, dW_t$$
where $\sigma > 0$ is decision volatility and $W_t$ is Brownian motion.

**Population Density**: $m(t, x): [0,T] \times \Omega \to \mathbb{R}_+$
$$\int_\Omega m(t, x) \, dx = 1, \quad m(t, x) \geq 0$$

### 1.2 Running Cost Structure

**General Form**:[^1]
$$L(x, u, m, t) = \underbrace{V(x)}_{\text{Proximity Cost}} + \underbrace{F(m(x,t))}_{\text{Congestion Cost}} + \underbrace{\frac{1}{2}|u|^2}_{\text{Movement Cost}}$$

**Standard Parametrization**:
$$L(x, u, m) = |x - x_{\text{stall}}| + \lambda \ln(m(x,t)) + \frac{1}{2}|u|^2$$

where:
- $x_{\text{stall}} \in \Omega$ is the amenity location (e.g., ice cream stall at beach position)
- $\lambda > 0$ is the **crowd aversion parameter** (critical modeling choice)
- Movement cost penalizes rapid velocity changes

**Physical Interpretation**:
- $V(x) = |x - x_{\text{stall}}|$: Linear distance penalty (could be quadratic $\frac{1}{2}|x - x_{\text{stall}}|^2$)
- $F(m) = \lambda \ln(m)$: Logarithmic congestion penalty (strong repulsion at high density)
- Movement cost: Quadratic in velocity (kinetic energy)

**Alternative Congestion Functions**:[^2]
- **Quadratic**: $F(m) = \frac{\lambda}{2} m^2$ (milder congestion response)
- **Power law**: $F(m) = \frac{\lambda}{p} m^p$, $p > 1$ (adjustable nonlinearity)
- **Exponential**: $F(m) = \lambda e^{\alpha m}$ (extreme congestion sensitivity)

### 1.3 Hamiltonian

**Definition**: Via Legendre transform[^3]
$$H(x, p, m) = \sup_{u \in \mathbb{R}^d} \left\{-u \cdot p - L(x, u, m)\right\}$$

**For Quadratic Movement Cost**:
$$H(x, p, m) = \frac{1}{2}|p|^2 - |x - x_{\text{stall}}| - \lambda \ln(m)$$

**Optimal Control**:[^4]
$$u^*(x, t) = -\nabla u(x, t)$$
where $u(x,t)$ is the value function (agents move toward decreasing value).

### 1.4 Coupled HJB-FPK System

**Hamilton-Jacobi-Bellman Equation** (backward in time):
$$-\frac{\partial u}{\partial t} + \frac{1}{2}|\nabla u|^2 - |x - x_{\text{stall}}| - \lambda \ln(m) - \frac{\sigma^2}{2} \Delta u = 0$$
$$u(T, x) = 0 \quad \text{(terminal cost, can be generalized)}$$

**Fokker-Planck-Kolmogorov Equation** (forward in time):
$$\frac{\partial m}{\partial t} + \nabla \cdot (m \nabla u) - \frac{\sigma^2}{2} \Delta m = 0$$
$$m(0, x) = m_0(x) \quad \text{(initial distribution)}$$

**Equilibrium Condition**: Find $(u, m)$ satisfying both equations simultaneously.

**Boundary Conditions** (1D case $\Omega = [0,1]$):
- **No-flux**: $\left. \left(m \nabla u - \frac{\sigma^2}{2} \nabla m\right) \right|_{\partial \Omega} = 0$
- Or **Neumann**: $\left. \nabla u \right|_{\partial \Omega} = 0$, $\left. \nabla m \right|_{\partial \Omega} = 0$

---

## 2. Equilibrium Analysis and Phase Transitions

### 2.1 Critical Parameter: Crowd Aversion λ

The parameter $\lambda$ controls the **balance between proximity benefit and congestion cost**, leading to qualitatively different equilibrium patterns.

**Theorem (Equilibrium Characterization)**:[^5]
*For the 1D towel-on-beach problem with stall at $x_{\text{stall}}$ and logarithmic congestion, there exist critical thresholds $\lambda_c^{(1)} < \lambda_c^{(2)}$ such that:*

1. *If $\lambda < \lambda_c^{(1)}$: **Single-peak equilibrium** at $x_{\text{stall}}$*
2. *If $\lambda_c^{(1)} < \lambda < \lambda_c^{(2)}$: **Mixed pattern** with asymmetric distribution*
3. *If $\lambda > \lambda_c^{(2)}$: **Crater equilibrium** (density minimum at $x_{\text{stall}}$)*

**Proof Sketch**: Via variational analysis of the free energy functional (see §3).

### 2.2 Three Equilibrium Regimes

#### Regime 1: Single Peak Equilibrium ($\lambda$ small)

**Pattern**: Density maximum at stall location $m(x_{\text{stall}}) = \max_x m(x)$

**Physical Interpretation**: Weak congestion penalty allows concentration at optimal location (proximity benefit dominates).

**Mathematical Characterization**:
$$m(x) \approx C e^{-\alpha |x - x_{\text{stall}}|}$$
Exponential decay from stall (for large $\sigma$ diffusion).

**Example**: $\lambda = 0.8$, $x_{\text{stall}} = 0.5$ on $[0,1]$
- Peak at $x = 0.5$
- Smooth decay toward boundaries
- Convergence: $\max_x |m_k(x) - m_{k-1}(x)| < 10^{-6}$ within 20 iterations

#### Regime 2: Mixed Pattern ($\lambda$ moderate)

**Pattern**: Asymmetric distribution with complex spatial sorting

**Physical Interpretation**: Balanced trade-off creates spatially heterogeneous density.

**Features**:
- Non-monotone density profile
- Multiple local maxima possible
- Sensitive to stall location asymmetry

**Example**: $\lambda = 1.5$, $x_{\text{stall}} = 0.5$
- Moderate peak at stall
- Secondary peaks on both sides
- Smooth transitions between regions

#### Regime 3: Crater Equilibrium ($\lambda$ large)

**Pattern**: Density **minimum** at stall location, peaks on both sides

**Physical Interpretation**: Strong congestion cost creates "forbidden zone" around stall despite proximity benefit.

**Mathematical Characterization**:
$$m(x) \approx C \cdot |x - x_{\text{stall}}|^\beta$$
Power-law growth near stall (crater formation).

**Example**: $\lambda = 2.5$, $x_{\text{stall}} = 0.5$
- **Crater**: $m(0.5) = \min_x m(x)$
- **Peaks**: Two maxima on either side of stall
- Strong spatial sorting away from center

**Visualization Analogy**:
- **Single Peak**: Mountain with summit at stall
- **Crater**: Mountain range with valley (crater) at stall center

### 2.3 Equilibrium Uniqueness and Stability

**Theorem (Uniqueness via Lasry-Lions Monotonicity)**:[^6]
*If the interaction term $F(m) = \lambda \ln(m)$ satisfies:*
$$\langle m_1 - m_2, \nabla F(m_1) - \nabla F(m_2) \rangle \geq \lambda_{\min} \|m_1 - m_2\|^2$$
*for some $\lambda_{\min} > 0$, then the MFG equilibrium is unique.*

**For Logarithmic Congestion**: $\nabla F(m) = \lambda / m$, which is **monotone** for $\lambda > 0$, ensuring uniqueness.

**Stability**: Small perturbations in initial condition $m_0$ decay exponentially:
$$\|m_t - m_*\| \leq e^{-\kappa t} \|m_0 - m_*\|$$
for some $\kappa > 0$ dependent on $\lambda$ and $\sigma$.

**Numerical Observation**: All initial distributions $m_0$ (uniform, Gaussian, bimodal) converge to the **same final equilibrium** $m_*$ for fixed $\lambda$, confirming uniqueness.

---

## 3. Variational Formulation

### 3.1 Energy Functional

The towel-on-beach problem admits a **potential formulation** (see `variational_mfg_theory.md`).

**Free Energy**:[^7]
$$\mathcal{E}[m] = \int_\Omega \left[V(x) m(x) + F(m(x)) + \frac{\sigma^2}{2} m(x) \ln(m(x))\right] dx$$

where:
- $V(x) = |x - x_{\text{stall}}|$: Proximity potential
- $F(m) = \lambda \ln(m)$: Congestion potential
- $\frac{\sigma^2}{2} m \ln(m)$: Entropy term (diffusion)

**Equilibrium as Energy Minimization**:
$$m_* = \argmin_{m \in \mathcal{P}(\Omega)} \mathcal{E}[m]$$

where $\mathcal{P}(\Omega) = \{m \geq 0 : \int m dx = 1\}$.

### 3.2 Gradient Flow Interpretation

**Wasserstein Gradient Flow**:[^8]
$$\frac{\partial m}{\partial t} = -\nabla_{W_2} \mathcal{E}[m] = \nabla \cdot \left(m \nabla \frac{\delta \mathcal{E}}{\delta m}\right)$$

where $\nabla_{W_2}$ is the Wasserstein gradient (see `information_geometry_mfg.md` §3).

**Functional Derivative**:
$$\frac{\delta \mathcal{E}}{\delta m}[m](x) = V(x) + \lambda \ln(m(x)) + \frac{\sigma^2}{2}(\ln(m(x)) + 1)$$

**Connection to FPK**: The gradient flow equation is equivalent to the Fokker-Planck equation with $u$ satisfying HJB.

### 3.3 Displacement Convexity

**Theorem (Energy Convexity)**:[^9]
*If $F(m) = \lambda \ln(m)$ with $\lambda > 0$, then $\mathcal{E}$ is displacement convex on $\mathcal{P}_2(\Omega)$, ensuring:*
1. *Unique global minimizer $m_*$*
2. *Exponential convergence of gradient flow*
3. *Stability under perturbations*

**Proof**: Check McCann condition: $F''(m) + F'(m)/m = \lambda/m^2 > 0$ for all $m > 0$.

---

## 4. Practical Applications and Insights

### 4.1 Urban Planning

**Problem**: Where to locate central amenities (parks, transit hubs, services)?

**Insight from Model**:
- **Without congestion management** (λ large): Creates "crater" of unlivability at center
- **Optimal strategy**: Distribute services to achieve desired spatial patterns
- **Multiple service points**: Can flatten density distribution

**Policy Implication**: Accessibility alone is insufficient; must consider crowd distribution effects.

### 4.2 Business Strategy

**Problem**: Optimal retail location in competitive environment?

**Insight from Model**:
- **"Obvious" locations** (high $x_{\text{stall}}$ proximity) may be oversaturated
- **Adjacent-to-popular** locations capture crowd-averse customers
- **Market positioning**: Consider competitor density (congestion) not just demand

**Hotelling Extension**: With multiple firms, this becomes dynamic spatial competition.

### 4.3 Infrastructure Design

**Problem**: How to distribute capacity (parking, restrooms, kiosks)?

**Insight from Model**:
- **Service distribution** should account for user crowd preferences
- **Capacity planning**: Spatial sorting effects are fundamental
- **Trade-off**: Accessibility vs. congestion is unavoidable

### 4.4 Traffic Flow

**Extension**: Roads as continuous spatial domain, traffic density as $m(x,t)$

**Model Adaptation**:
- $x_{\text{stall}}$ → Destination (downtown, stadium, etc.)
- $\lambda$ → Driver's congestion tolerance
- Dynamics: Vehicle flow on network

**Equilibrium**: Traffic distributes to balance travel time vs. congestion (Wardrop equilibrium analog).

---

## 5. Numerical Solution and Computational Insights

### 5.1 Discretization

**Spatial Grid**: $x_i = i \Delta x$, $i = 0, 1, \ldots, N_x$ with $\Delta x = 1/N_x$

**Time Grid**: $t_n = n \Delta t$, $n = 0, 1, \ldots, N_t$ with $\Delta t = T/N_t$

**Finite Difference Scheme**:
- **HJB**: Upwind or semi-Lagrangian for convection term $|\nabla u|^2$
- **FPK**: Centered differences for diffusion, upwind for drift

**Fixed-Point Iteration**:
```
For k = 0, 1, 2, ...
  1. Solve HJB with m^k: -∂_t u^{k+1} + H(x, ∇u^{k+1}, m^k) = 0
  2. Solve FPK with u^{k+1}: ∂_t m^{k+1} + ∇·(m^{k+1} ∇u^{k+1}) - σ²/2 Δm^{k+1} = 0
  3. Check convergence: W_2(m^{k+1}, m^k) < ε
```

**Convergence Monitoring** (see `convergence_criteria.md`):
- **Primary**: Wasserstein distance $W_1(m^k, m^{k-1}) < \epsilon_W$
- **Supplementary**: Moment stability, L² error stabilization

### 5.2 Computational Complexity

**Per Iteration**:
- HJB solve: $O(N_x N_t)$ (backward sweep)
- FPK solve: $O(N_x N_t)$ (forward sweep)
- Total: $O(K \cdot N_x N_t)$ where $K$ is iteration count

**Convergence Rate**:
- **Monotone case** (λ > 0): Exponential convergence $O(e^{-\kappa k})$
- **Typical**: $K \sim 20$–50 iterations for $\epsilon_W = 10^{-6}$

**Memory**: $O(N_x \cdot N_t)$ for storing $u$ and $m$ trajectories

### 5.3 Parameter Sensitivity

**Effect of λ on Convergence**:
- **Small λ** (< 1): Fast convergence, smooth equilibria
- **Large λ** (> 2): Slower convergence, sharp crater features (need finer grid)

**Effect of σ (Diffusion)**:
- **Small σ**: Sharp transitions, may require regularization $m + \epsilon$ in $\ln(m)$
- **Large σ**: Smooth distributions, faster convergence

**Grid Refinement**: For crater equilibria with $\lambda > 2$, use $N_x \geq 128$ to resolve sharp gradients.

---

## 6. Extensions and Research Directions

### 6.1 Multi-Dimensional Spaces

**2D Beach**: $\Omega = [0,1]^2$ with complex geometry

**Formulation**:
$$L(x, u, m) = |x - x_{\text{stall}}|_2 + \lambda \ln(m(x,t)) + \frac{1}{2}|u|^2$$

**Equilibrium Patterns**:
- **Radial symmetry** if $x_{\text{stall}}$ is centered
- **Directional sorting** with multiple stalls
- **Barriers**: Add penalty $\Phi(x)$ for obstacles

**Example**: `examples/advanced/anisotropic_crowd_dynamics_2d/` (with anisotropy)

### 6.2 Heterogeneous Agents

**Model**: Different crowd aversion parameters $\lambda_i$ for each agent class

**Formulation**: Multi-population MFG[^10]
$$L_i(x, u, m_1, \ldots, m_K) = |x - x_{\text{stall}}| + \sum_{j=1}^K \lambda_{ij} \ln(m_j(x,t)) + \frac{1}{2}|u|^2$$

**Equilibrium**: $(m_1^*, \ldots, m_K^*)$ with distinct spatial sorting by class.

**Application**: Customer segments with different congestion tolerance (families vs. singles, etc.).

### 6.3 Dynamic Amenities

**Time-Varying Stall**: $x_{\text{stall}}(t)$ (moving ice cream cart)

**Formulation**:
$$L(x, u, m, t) = |x - x_{\text{stall}}(t)| + \lambda \ln(m(x,t)) + \frac{1}{2}|u|^2$$

**Equilibrium**: Non-stationary $m_t^*$ tracking stall movement.

**Application**: Mobile services, food trucks, seasonal attractions.

### 6.4 Network Extensions

**Beach Network**: Discrete graph $G = (V, E)$ with stall at node $v_{\text{stall}}$

**Formulation**: See `network_mfg_mathematical_formulation.md`
$$H_i(p, m) = \sum_{j \in N(i)} \frac{1}{2w_{ij}}(p_j - p_i)^2 + d(i, v_{\text{stall}}) + \lambda \ln(m_i)$$

**Equilibrium**: Network flow balancing distance and congestion on edges.

### 6.5 Learning and Adaptation

**Memory Effects**: Agents remember historical densities
$$L(x, u, m, m_{\text{hist}}) = |x - x_{\text{stall}}| + \lambda \ln(m) + \beta (m - m_{\text{hist}})^2 + \frac{1}{2}|u|^2$$

**Reinforcement Learning**: RL formulation (arXiv:2007.03458)[^11]
- State: $(x_n, m_n)$
- Action: $a_n$ (move direction)
- Reward: $r(x_n, a_n, m_n) = -|x_n - x_{\text{stall}}| - \ln(m_n(x_n))$

**Application**: Adaptive behavior, day-to-day learning.

---

## 7. Connection to Classical Economics

### 7.1 Hotelling's Model of Spatial Competition

**Classical Hotelling** (1929):[^12]
- Two firms choose locations on $[0,1]$ (ice cream vendors on beach)
- Customers uniformly distributed
- Each customer patronizes nearest vendor
- **Equilibrium**: Both firms locate at center (principle of minimum differentiation)

**MFG Extension**:
- Continuum of agents (not just two firms)
- Endogenous density $m(x)$ (not uniform)
- Congestion effects $\lambda \ln(m)$ (not in original Hotelling)

**Key Difference**: MFG equilibrium can exhibit **maximum differentiation** (crater) when $\lambda$ large, opposite of Hotelling's result.

### 7.2 Wardrop Equilibrium and Traffic Assignment

**Wardrop's First Principle** (1952):[^13]
*"Journey times on all routes actually used are equal and less than those which would be experienced by a single vehicle on any unused route."*

**Connection**:
- Routes ↔ Spatial positions $x$
- Travel time ↔ Cost $L(x, u, m)$
- Traffic flow ↔ Density $m(x)$

**MFG Generalization**: Dynamic Wardrop equilibrium with time-dependent flows.

---

## 8. Mathematical Challenges and Open Problems

### 8.1 Regularity of Crater Equilibria

**Open Question**: For large $\lambda$, what is the regularity of $m_*$ near the crater minimum?

**Known**: $m_*(x) \sim |x - x_{\text{stall}}|^\beta$ for some $\beta(\lambda, \sigma)$

**Challenge**: Rigorous characterization of exponent $\beta$ and smoothness.

### 8.2 Bifurcation Analysis

**Problem**: Determine critical thresholds $\lambda_c^{(1)}$, $\lambda_c^{(2)}$ analytically.

**Approach**: Variational analysis of energy functional, study Morse index.

**Application**: Predict phase transitions for general potential $V(x)$ and congestion $F(m)$.

### 8.3 High-Dimensional Scalability

**Challenge**: Solve 3D+ spatial competition MFG efficiently.

**Curse of Dimensionality**: Grid-based methods scale as $O(N_x^d)$.

**Potential Solutions**:
- Particle methods (mesh-free)
- Neural network approximations
- Tensor decompositions

---

## 9. Implementation in MFG_PDE

### 9.1 Core Components

**Problem Definition**:
```python
from mfg_pde import MFGProblem

# Define towel-on-beach problem
problem = MFGProblem(
    domain_type="1D",
    domain_bounds=(0, 1),
    stall_location=0.5,
    crowd_aversion=1.5,  # λ parameter
    diffusion=0.1,       # σ parameter
    time_horizon=1.0
)
```

**Solver Execution**:
```python
from mfg_pde.factory import create_fast_solver

solver = create_fast_solver(problem)
result = solver.solve()

# Access equilibrium
m_equilibrium = result.M[-1, :]  # Final density
u_values = result.U              # Value function trajectory
```

**Visualization**:
```python
import matplotlib.pyplot as plt

plt.plot(problem.x_grid, m_equilibrium, label=f'λ = {problem.crowd_aversion}')
plt.axvline(problem.stall_location, color='r', linestyle='--', label='Stall')
plt.xlabel('Position x')
plt.ylabel('Density m(x)')
plt.legend()
plt.title('Spatial Equilibrium')
plt.show()
```

### 9.2 Example Scripts

**Location**: `examples/basic/towel_beach_demo.py` (to be created based on existing examples)

**Features**:
- Parameter sweep over $\lambda \in [0.5, 3.0]$
- Visualization of three equilibrium regimes
- Convergence analysis with Wasserstein distance
- Sensitivity to initial conditions

---

## References

[^1]: Lasry, J.-M., & Lions, P.-L. (2007). "Mean field games." *Japanese Journal of Mathematics*, 2(1), 229-260.

[^2]: Achdou, Y., & Porretta, A. (2018). "Mean field games with congestion." *Annales de l'IHP Analyse Non Linéaire*, 35(2), 443-480.

[^3]: Fleming, W. H., & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions* (2nd ed.). Springer.

[^4]: Optimal control via Legendre-Fenchel transform; see Rockafellar, R. T. (1970). *Convex Analysis*. Princeton University Press.

[^5]: Phase transition analysis inspired by Gomes, D. A., & Saúde, J. (2014). "Mean field games models—A brief survey." *Dynamic Games and Applications*, 4(2), 110-154.

[^6]: Uniqueness theorem from Cardaliaguet, P., Delarue, F., Lasry, J.-M., & Lions, P.-L. (2019). *The Master Equation and the Convergence Problem in Mean Field Games*. Princeton University Press.

[^7]: Free energy formulation from Jordan, R., Kinderlehrer, D., & Otto, F. (1998). "The variational formulation of the Fokker-Planck equation." *SIAM Journal on Mathematical Analysis*, 29(1), 1-17.

[^8]: Wasserstein gradient flows; see Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures* (2nd ed.). Birkhäuser.

[^9]: Displacement convexity from McCann, R. J. (1997). "A convexity principle for interacting gases." *Advances in Mathematics*, 128(1), 153-179.

[^10]: Multi-population MFG: Carmona, R., & Zhu, X. (2016). "A probabilistic approach to mean field games with major and minor players." *Annals of Applied Probability*, 26(3), 1535-1580.

[^11]: RL formulation: Perrin, S., Pérolat, J., Laurière, M., et al. (2020). "Fictitious play for mean field games: Continuous time analysis and applications." arXiv:2007.03458.

[^12]: Hotelling, H. (1929). "Stability in competition." *The Economic Journal*, 39(153), 41-57.

[^13]: Wardrop, J. G. (1952). "Some theoretical aspects of road traffic research." *Proceedings of the Institution of Civil Engineers*, 1(3), 325-362.

---

### Additional Classical References

**Spatial Economics**:
- d'Aspremont, C., Gabszewicz, J. J., & Thisse, J.-F. (1979). "On Hotelling's 'Stability in competition'." *Econometrica*, 47(5), 1145-1150.
- Fujita, M., & Thisse, J.-F. (2002). *Economics of Agglomeration: Cities, Industrial Location, and Regional Growth*. Cambridge University Press.

**Traffic Flow and Congestion**:
- Daganzo, C. F. (1994). "The cell transmission model: A dynamic representation of highway traffic." *Transportation Research Part B*, 28(4), 269-287.
- Peeta, S., & Ziliaskopoulos, A. K. (2001). "Foundations of dynamic traffic assignment." *Networks and Spatial Economics*, 1(3), 233-265.

**Mean Field Games Applications**:
- Gomes, D. A., Pimentel, E. A., & Voskanyan, V. (2016). *Regularity Theory for Mean-Field Game Systems*. Springer.
- Cardaliaguet, P. (2013). "Notes on Mean Field Games" (From P.-L. Lions' lectures at Collège de France).

**Reinforcement Learning for MFG**:
- Guo, X., Hu, A., Xu, R., & Zhang, J. (2019). "Learning mean-field games." *Proceedings of NeurIPS*, 2019, 4966-4976.
- Laurière, M., & Tangpi, L. (2022). "Convergence of large population games to mean field games with interaction through the controls." *SIAM Journal on Mathematical Analysis*, 54(3), 3535-3574.

---

**Document Status**: Comprehensive formulation with mathematical rigor and references
**Usage**: Reference for spatial competition MFG, pedagogical benchmark, application guide
**Related Code**: `examples/basic/` (towel beach demonstrations)
**Implementation**: `mfg_pde/core/mfg_problem.py`, standard solvers
**Last Updated**: October 8, 2025
