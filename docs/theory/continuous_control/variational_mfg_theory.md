# Variational Mean Field Games: Theory and Computational Methods

**Document Type**: Theoretical Foundation
**Created**: October 8, 2025
**Status**: Comprehensive mathematical framework
**Related**: `mathematical_background.md`, `information_geometry_mfg.md`, `convergence_criteria.md`

---

## Introduction

**Variational Mean Field Games** (Variational MFG) reformulate the classical HJB-FPK system as an **optimization problem** on the space of probability measures. This perspective:

1. **Unifies** MFG with optimal transport and calculus of variations
2. **Enables** gradient-based numerical methods (JKO schemes, Wasserstein gradient flows)
3. **Reveals** convexity properties ensuring uniqueness of equilibria
4. **Connects** to statistical physics (free energy minimization)
5. **Extends** to non-potential games via monotonicity conditions

This document provides rigorous mathematical foundations for variational formulations and their computational implementation in MFG_PDE.

---

## 1. Classical MFG vs. Variational Formulation

### 1.1 Classical HJB-FPK System

**Standard MFG**:[^1]
$$\begin{cases}
-\partial_t u + H(x, \nabla u, m) = 0, & u(T, x) = g(x, m(T)) \\
\partial_t m - \text{div}(m \nabla_p H(x, \nabla u, m)) - \sigma^2 \Delta m = 0, & m(0, x) = m_0(x)
\end{cases}$$

**Challenges**:
- Coupled nonlinear PDEs (backward-forward system)
- Non-convex in general (multiple equilibria possible)
- Convergence of fixed-point iteration not guaranteed

### 1.2 Variational Formulation (Potential MFG)

**Definition (Potential MFG)**:[^2]
A MFG is **potential** if there exists a functional $\mathcal{E}: \mathcal{C}([0,T]; \mathcal{P}(\Omega)) \to \mathbb{R}$ such that:

**Equilibrium condition**: $m^* \in \argmin_{m} \mathcal{E}[m]$ subject to:
$$\partial_t m = \text{div}(m \nabla_p H(x, \nabla u^m)) + \sigma^2 \Delta m, \quad m(0) = m_0$$

where $u^m$ solves the HJB equation for fixed $m$.

**Energy Functional** (Benamou-Brenier-Achdou form):[^3]
$$\mathcal{E}[m] = \int_0^T \int_\Omega \left[\frac{1}{2}|v|^2 m + F(x, m, t)\right] dx \, dt + \int_\Omega g(x, m(T)) dx$$

subject to continuity equation:
$$\partial_t m + \text{div}(m v) = \sigma^2 \Delta m, \quad m(0) = m_0$$

**Key Insight**: Instead of solving coupled PDEs, we minimize a functional over trajectory space.

---

## 2. Theoretical Foundations

### 2.1 Existence of Potential Functions

**Theorem (Characterization of Potential MFG)**:[^4]
*A MFG is potential if and only if the Hamiltonian $H$ satisfies:*
$$\frac{\partial H}{\partial m}(x, p, m) = \nabla_m \mathcal{F}[m](x)$$
*for some functional $\mathcal{F}: \mathcal{P}(\Omega) \to \mathbb{R}$.*

**Equivalently**: The system admits a variational structure if:
$$\frac{\partial^2 H}{\partial m \partial p} = \left(\frac{\partial^2 H}{\partial p \partial m}\right)^T$$
(symmetry of mixed derivatives in distributional sense).

**Example (Potential Hamiltonian)**:
$$H(x, p, m) = \frac{1}{2}|p|^2 + V(x) + F(m(x))$$

where $F: \mathbb{R}_+ \to \mathbb{R}$ is the **local interaction function**.

**Common Choices**:
1. **Quadratic congestion**: $F(m) = \frac{\lambda}{2} m^2$ (potential)
2. **Logarithmic**: $F(m) = -\lambda m \log m$ (entropy, potential)
3. **Power law**: $F(m) = \frac{\lambda}{p} m^p$ (potential if $p > 1$)

**Non-Example (Non-Potential)**:
$$H(x, p, m) = \frac{1}{2}|p|^2 + V(x) + \lambda m(x+1)$$
where $m(x+1)$ is non-local interaction (not potential in general).

### 2.2 Convexity and Displacement Convexity

**Definition (Displacement Convexity)**:[^5]
A functional $\mathcal{E}: \mathcal{P}_2(\Omega) \to \mathbb{R}$ is **$\lambda$-displacement convex** if for any geodesic $(m_t)_{t \in [0,1]}$ in Wasserstein space $W_2$:
$$\mathcal{E}[m_t] \leq (1-t) \mathcal{E}[m_0] + t \mathcal{E}[m_1] - \frac{\lambda}{2} t(1-t) W_2^2(m_0, m_1)$$

**Physical Interpretation**: Energy decreases along optimal transport paths faster than linear interpolation.

**Theorem (Displacement Convexity Implies Uniqueness)**:[^6]
*If $\mathcal{E}$ is $\lambda$-displacement convex with $\lambda > 0$, then:*
1. *$\mathcal{E}$ has a unique minimizer $m^* \in \mathcal{P}_2(\Omega)$*
2. *Gradient flow $\partial_t m = -\nabla_{W_2} \mathcal{E}[m]$ converges exponentially:*
   $$\mathcal{E}[m_t] - \mathcal{E}[m_*] \leq e^{-2\lambda t} (\mathcal{E}[m_0] - \mathcal{E}[m_*])$$

**Condition for Displacement Convexity**:[^7]
For potential MFG with Hamiltonian $H(x, p, m) = \frac{1}{2}|p|^2 + V(x) + F(m(x))$:

$$F''(m) + F'(m)/m \geq \lambda > 0 \quad \text{(McCann condition)}$$

**Examples**:
- $F(m) = \frac{c}{2} m^2$: Displacement convex with $\lambda = c$
- $F(m) = m \log m$: Displacement convex with $\lambda = 1$ (Talagrand inequality)
- $F(m) = m^p$, $p \in (1, 2)$: Displacement convex on compact domains

### 2.3 Lagrangian Formulation and Benamou-Brenier

**Benamou-Brenier Dynamic Formulation**:[^8]
The Wasserstein geodesic problem can be written as:
$$W_2^2(m_0, m_1) = \inf \int_0^1 \int_\Omega |v_t(x)|^2 m_t(x) \, dx \, dt$$
subject to:
$$\partial_t m_t + \text{div}(m_t v_t) = 0, \quad m(0) = m_0, \, m(1) = m_1$$

**Lagrangian Reformulation**:
Define action functional:
$$\mathcal{A}[m, v] = \int_0^T \int_\Omega \left[\frac{1}{2}|v|^2 m + L(x, m)\right] dx \, dt + \int_\Omega g(x, m(T)) dx$$

**Theorem (Equivalence to MFG)**:[^9]
*The minimizer $(m^*, v^*)$ of $\mathcal{A}$ subject to the continuity equation satisfies:*
$$v^* = -\nabla u^*, \quad -\partial_t u^* + \frac{1}{2}|\nabla u^*|^2 + \frac{\delta \mathcal{A}}{\delta m}[m^*] = 0$$
*which is the HJB-FPK system for potential MFG.*

---

## 3. Jordan-Kinderlehrer-Otto (JKO) Scheme

### 3.1 Implicit Euler in Wasserstein Space

**Definition (JKO Scheme)**:[^10]
The **JKO scheme** is implicit Euler discretization in Wasserstein metric:

**Time discretization**: $t_n = n \tau$, $n = 0, 1, \ldots, N$ with $\tau = T/N$

**Iterative Minimization**:
$$m_{n+1} = \argmin_{m \in \mathcal{P}_2(\Omega)} \left\{\frac{1}{2\tau} W_2^2(m, m_n) + \mathcal{E}[m]\right\}$$

**Physical Interpretation**: Each step minimizes **kinetic energy** (Wasserstein distance) plus **potential energy** (functional $\mathcal{E}$).

**Theorem (JKO Convergence)**:[^11]
*If $\mathcal{E}$ is $\lambda$-displacement convex, then the JKO scheme converges as $\tau \to 0$:*
$$\sup_{t \in [0,T]} W_2(m_\tau(t), m_*(t)) \leq C \tau^{1/2}$$
*where $m_*$ is the gradient flow solution.*

### 3.2 Computational Implementation

**Algorithm (JKO for MFG)**:

**Input**: Initial density $m_0$, time step $\tau$, energy functional $\mathcal{E}$

**For** $n = 0, 1, \ldots, N-1$:
1. **Solve optimization problem**:
   $$m_{n+1} = \argmin_{m} \left\{\frac{1}{2\tau} W_2^2(m, m_n) + \mathcal{E}[m]\right\}$$

2. **Wasserstein distance computation**:
   - **1D**: Exact via quantile functions, $O(N_x \log N_x)$
   - **High-D**: Sinkhorn algorithm (entropic regularization), $O(N_x^2 K)$ where $K$ is iteration count

3. **Energy minimization**: Use gradient descent or Newton's method in discretized space

**Practical Regularization**:[^12]
Replace $W_2$ with **Sinkhorn divergence**:
$$W_{2,\epsilon}^2(m, n) = \langle m - n, K_\epsilon^{-1}(m - n) \rangle$$
where $K_\epsilon = e^{-c(x,y)/\epsilon}$ is the Gibbs kernel.

**Advantage**: Differentiable with respect to $(m, n)$, enabling gradient-based optimization.

### 3.3 Energy Dissipation and Lyapunov Functions

**Theorem (Energy Dissipation)**:[^13]
*The JKO scheme satisfies the discrete energy dissipation inequality:*
$$\mathcal{E}[m_{n+1}] + \frac{1}{2\tau} W_2^2(m_{n+1}, m_n) \leq \mathcal{E}[m_n]$$

**Corollary**: $\mathcal{E}[m_n]$ is monotone decreasing, providing a **Lyapunov function** for convergence.

**Gradient Flow Characterization**:
$$\frac{m_{n+1} - m_n}{\tau} \approx -\nabla_{W_2} \mathcal{E}[m_{n+1}]$$

where $\nabla_{W_2}$ is the Wasserstein gradient (see `information_geometry_mfg.md` §3).

---

## 4. Primal-Dual Formulations

### 4.1 Kantorovich Duality

**Primal Problem** (Wasserstein distance):
$$W_2^2(m_0, m_1) = \inf_{\pi \in \Pi(m_0, m_1)} \int_{\Omega \times \Omega} |x - y|^2 \, d\pi(x, y)$$

**Dual Problem** (Kantorovich):[^14]
$$W_2^2(m_0, m_1) = \sup_{\phi, \psi} \left\{\int \phi \, dm_0 + \int \psi \, dm_1 : \phi(x) + \psi(y) \leq |x - y|^2\right\}$$

**Optimal Transport Map**: If $m_0$ is absolutely continuous, there exists a unique map $T$ such that:
$$T_\# m_0 = m_1, \quad T = \nabla \phi \quad \text{(Brenier's theorem)}$$

where $\phi$ solves the Monge-Ampère equation:
$$\det(\nabla^2 \phi) = \frac{m_0}{\m_1 \circ \nabla \phi}$$

### 4.2 Augmented Lagrangian Method

**Constrained Optimization**:
$$\min_{m, v} \mathcal{A}[m, v] \quad \text{s.t.} \quad \partial_t m + \text{div}(m v) = 0$$

**Augmented Lagrangian**:[^15]
$$\mathcal{L}_r[m, v, \lambda] = \mathcal{A}[m, v] + \langle \lambda, \partial_t m + \text{div}(m v) \rangle + \frac{r}{2} \|\partial_t m + \text{div}(m v)\|^2$$

**ADMM Algorithm**:
1. **Update $v$**: $v^{k+1} = \argmin_v \mathcal{L}_r[m^k, v, \lambda^k]$
2. **Update $m$**: $m^{k+1} = \argmin_m \mathcal{L}_r[m, v^{k+1}, \lambda^k]$
3. **Update $\lambda$**: $\lambda^{k+1} = \lambda^k + r(\partial_t m^{k+1} + \text{div}(m^{k+1} v^{k+1}))$

**Convergence**:[^16] Under convexity of $\mathcal{A}$, ADMM converges linearly to $(m^*, v^*, \lambda^*)$.

---

## 5. Monotone MFG and Variational Inequalities

### 5.1 Non-Potential MFG with Monotonicity

**Definition (Monotone MFG)**:[^17]
A MFG is **monotone** if:
$$\langle m_1 - m_2, \nabla_m H(\cdot, \nabla u_1, m_1) - \nabla_m H(\cdot, \nabla u_2, m_2) \rangle \geq \lambda \|m_1 - m_2\|^2$$

**Remark**: Potential MFG are automatically monotone with $\lambda > 0$ if $\mathcal{E}$ is displacement convex.

**Theorem (Uniqueness for Monotone MFG)**:[^18]
*If a MFG is $\lambda$-monotone with $\lambda > 0$, then it has a unique equilibrium $(u^*, m^*)$.*

**Proof Sketch**: Use contraction mapping theorem in product space with metric:
$$d((u_1, m_1), (u_2, m_2)) = \|u_1 - u_2\|_{L^2} + W_2(m_1, m_2)$$

### 5.2 Variational Inequality Formulation

**Equivalent Formulation**:[^19]
Find $(u^*, m^*) \in \mathcal{H} \times \mathcal{P}_2(\Omega)$ such that:
$$\langle F(u^*, m^*), (u - u^*, m - m^*) \rangle \geq 0, \quad \forall (u, m) \in \mathcal{H} \times \mathcal{P}_2(\Omega)$$

where $F: (u, m) \mapsto (-\partial_t u + H(x, \nabla u, m), \partial_t m - \text{div}(m \nabla_p H))$.

**Computational Method**: Extragradient algorithm[^20]
1. **Prediction**: $(u^{k+1/2}, m^{k+1/2}) = (u^k, m^k) - \tau F(u^k, m^k)$
2. **Correction**: $(u^{k+1}, m^{k+1}) = (u^k, m^k) - \tau F(u^{k+1/2}, m^{k+1/2})$

**Convergence**: For $\lambda$-monotone $F$ and $\tau < 1/L$ (Lipschitz constant), converges linearly.

---

## 6. Entropy Regularization and Schrödinger Bridge

### 6.1 Entropic Regularization of Wasserstein Distance

**Regularized Wasserstein**:[^21]
$$W_{2,\epsilon}^2(m_0, m_1) = \inf_{\pi \in \Pi(m_0, m_1)} \left\{\int |x - y|^2 d\pi(x, y) + \epsilon D_{KL}(\pi \| m_0 \otimes m_1)\right\}$$

where $D_{KL}$ is Kullback-Leibler divergence.

**Sinkhorn Algorithm**:[^22]
The optimizer has the form $\pi^* = \text{diag}(u) K \text{diag}(v)$ where $K_{ij} = e^{-|x_i - y_j|^2/\epsilon}$.

**Iteration**:
$$u^{n+1} = m_0 \oslash (K v^n), \quad v^{n+1} = m_1 \oslash (K^T u^{n+1})$$
where $\oslash$ is element-wise division.

**Complexity**: $O(N^2 K)$ where $K \sim \log(1/\delta)$ for accuracy $\delta$.

### 6.2 Schrödinger Bridge Problem

**Stochastic Control Formulation**:[^23]
$$\inf_{u} \mathbb{E}\left[\int_0^T \frac{1}{2}|u_t|^2 dt\right]$$
subject to:
$$dX_t = u_t dt + \sigma dW_t, \quad \mathcal{L}(X_0) = m_0, \, \mathcal{L}(X_T) = m_1$$

**Equivalence to Entropic OT**:[^24]
The Schrödinger bridge is the continuous-time limit of entropic optimal transport:
$$\lim_{\epsilon \to 0} \epsilon W_{2,\epsilon}^2(m_0, m_1) = \text{Schrödinger cost}$$

**Application to MFG**: Regularized MFG equilibria can be computed via Schrödinger bridge solvers.

---

## 7. Numerical Methods Summary

### 7.1 Method Comparison

| Method | Pros | Cons | Best For |
|:-------|:-----|:-----|:---------|
| **JKO Scheme** | Structure-preserving, energy dissipation | $O(N^3)$ per step (high-D) | Small-scale, displacement convex |
| **Sinkhorn JKO** | Fast ($O(N^2 K)$), GPU-friendly | Entropic bias, parameter $\epsilon$ | Large-scale, smooth densities |
| **ADMM** | Handles constraints naturally | Requires tuning $r$ | Constrained MFG |
| **Extragradient** | Works for non-potential | No energy guarantee | Monotone non-potential MFG |
| **Primal-Dual** | Stable, well-conditioned | Slower convergence | General MFG |

### 7.2 Discretization Hierarchy

**Spatial Discretization**:
1. **Finite Differences**: Simple, explicit computation
2. **Finite Elements**: Better for irregular domains
3. **Particle Methods**: Mesh-free, high-dimensional

**Temporal Discretization**:
1. **Explicit Euler**: $m^{n+1} = m^n - \tau \nabla_{W_2} \mathcal{E}[m^n]$ (CFL condition required)
2. **Implicit Euler (JKO)**: $m^{n+1} = \argmin \{\frac{1}{2\tau} W_2^2(m, m^n) + \mathcal{E}[m]\}$ (unconditionally stable)
3. **Semi-implicit**: Mixed explicit-implicit for efficiency

---

## 8. Implementation in MFG_PDE

### 8.1 Core Components

**Location**: `mfg_pde/alg/optimization/`

**Key Classes**:
1. **`WassersteinGradientFlow`**: JKO scheme implementation
   - Wasserstein distance via `scipy.stats` (1D) or `POT` library (multi-D)
   - Energy functional interface
   - Adaptive time stepping

2. **`SinkhornSolver`**: Entropic optimal transport
   - GPU-accelerated via `cupy` (optional)
   - Log-domain stabilization
   - Dual potentials extraction

3. **`VariationalMFGSolver`**: General variational MFG solver
   - Benamou-Brenier formulation
   - ADMM and primal-dual methods
   - Convergence monitoring

### 8.2 Example Usage

**Simple Quadratic Congestion MFG**:

```python
from mfg_pde.alg.optimization import WassersteinGradientFlow
from mfg_pde.core import VariationalMFGProblem
import numpy as np

# Define energy functional
def energy(m):
    # E[m] = ∫ (V(x) m(x) + (λ/2) m(x)^2) dx
    V = potential(x_grid)
    return np.sum((V * m + 0.5 * lambda_cong * m**2) * dx)

# Initial density
m0 = np.exp(-x_grid**2)
m0 /= np.sum(m0) * dx

# JKO scheme
jko = WassersteinGradientFlow(energy=energy, tau=0.01)
m_trajectory = jko.solve(m0, T=1.0)
```

**Sinkhorn-Based Solver**:

```python
from mfg_pde.alg.optimization import SinkhornJKO

# Entropic JKO with GPU acceleration
solver = SinkhornJKO(
    energy=energy,
    epsilon=0.01,  # Entropic regularization
    tau=0.01,      # Time step
    use_gpu=True
)

m_trajectory = solver.solve(m0, T=1.0, tol=1e-6)
```

### 8.3 Validation Examples

**Location**: `examples/advanced/variational_mfg_demo.py`

**Test Cases**:
1. **Quadratic congestion**: Verify displacement convexity
2. **Logarithmic entropy**: Check exponential convergence
3. **Non-potential with monotonicity**: Extragradient method
4. **High-dimensional ($d=10$)**: Sinkhorn scaling

---

## 9. Connections to Other Formulations

### 9.1 Relationship to Classical MFG

**Theorem (Equivalence)**:[^25]
*For potential MFG, the following are equivalent:*
1. *(u, m)* *solves HJB-FPK system*
2. *m minimizes variational energy* $\mathcal{E}[m]$
3. *m is the gradient flow of* $\mathcal{E}$ *starting from* $m_0$

**Proof**: Via optimality conditions and Lions derivative calculus (see `information_geometry_mfg.md`).

### 9.2 Link to Optimal Control

**Pontryagin Maximum Principle**: For individual agent with control $u$:
$$\max_u \left\{-L(x, u, m) - p \cdot f(x, u)\right\} = H(x, p, m)$$

**Variational Reformulation**: Same problem as:
$$\min_{(x, u)} \int_0^T L(x, u, m) dt \quad \text{s.t.} \quad \dot{x} = f(x, u)$$

**Connection**: Hamiltonian $H$ is Legendre transform of Lagrangian $L$.

### 9.3 Statistical Physics Analogy

**Free Energy**: In statistical mechanics:
$$F[\rho] = \int \rho V dx + kT \int \rho \log \rho dx$$

**Boltzmann Equilibrium**: Minimizer is $\rho^* = e^{-V/kT} / Z$

**MFG Analogy**: Logarithmic MFG with $F(m) = m \log m$ corresponds to thermal equilibrium at temperature $1/\lambda$.

---

## References

[^1]: Lasry, J.-M., & Lions, P.-L. (2007). "Mean field games." *Japanese Journal of Mathematics*, 2(1), 229-260.

[^2]: Achdou, Y., & Porretta, A. (2018). "Mean field games with congestion." *Annales de l'IHP Analyse Non Linéaire*, 35(2), 443-480.

[^3]: Benamou, J.-D., & Brenier, Y. (2000). "A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem." *Numerische Mathematik*, 84(3), 375-393.

[^4]: Graber, P. J., & Mészáros, A. R. (2019). "Sobolev regularity for first order mean field games." *Annales de l'IHP Analyse Non Linéaire*, 36(6), 1557-1576.

[^5]: McCann, R. J. (1997). "A convexity principle for interacting gases." *Advances in Mathematics*, 128(1), 153-179.

[^6]: Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures* (2nd ed.). Birkhäuser.

[^7]: Displacement convexity conditions from McCann (1997) and Villani, C. (2009). *Optimal Transport: Old and New*. Springer.

[^8]: Benamou & Brenier (2000).

[^9]: Cardaliaguet, P., & Porretta, A. (2019). "Long time average of mean field games with a nonlocal coupling." *SIAM Journal on Control and Optimization*, 51(5), 3558-3591.

[^10]: Jordan, R., Kinderlehrer, D., & Otto, F. (1998). "The variational formulation of the Fokker-Planck equation." *SIAM Journal on Mathematical Analysis*, 29(1), 1-17.

[^11]: JKO convergence: Ambrosio et al. (2008), Chapter 11.

[^12]: Cuturi, M. (2013). "Sinkhorn distances: Lightspeed computation of optimal transport." *Proceedings of NIPS*, 2013, 2292-2300.

[^13]: Energy dissipation for JKO: Jordan et al. (1998).

[^14]: Villani, C. (2009). *Optimal Transport: Old and New*. Springer.

[^15]: Benamou, J.-D., & Carlier, G. (2015). "Augmented Lagrangian methods for transport optimization, mean field games and degenerate elliptic equations." *Journal of Optimization Theory and Applications*, 167(1), 1-26.

[^16]: ADMM convergence: Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). "Distributed optimization and statistical learning via ADMM." *Foundations and Trends in Machine Learning*, 3(1), 1-122.

[^17]: Monotone MFG definition from Lasry & Lions (2007).

[^18]: Cardaliaguet, P., Delarue, F., Lasry, J.-M., & Lions, P.-L. (2019). *The Master Equation and the Convergence Problem in Mean Field Games*. Princeton University Press.

[^19]: Kinderlehrer, D., & Stampacchia, G. (2000). *An Introduction to Variational Inequalities and Their Applications*. SIAM.

[^20]: Korpelevich, G. M. (1976). "The extragradient method for finding saddle points and other problems." *Matecon*, 12, 747-756.

[^21]: Cuturi (2013).

[^22]: Sinkhorn, R., & Knopp, P. (1967). "Concerning nonnegative matrices and doubly stochastic matrices." *Pacific Journal of Mathematics*, 21(2), 343-348.

[^23]: Léonard, C. (2014). "A survey of the Schrödinger problem and some of its connections with optimal transport." *Discrete and Continuous Dynamical Systems A*, 34(4), 1533-1574.

[^24]: Chen, Y., Georgiou, T. T., & Pavon, M. (2016). "Optimal transport over a linear dynamical system." *IEEE Transactions on Automatic Control*, 62(5), 2137-2152.

[^25]: Equivalence of formulations: See Porretta, A. (2015). "Weak solutions to Fokker-Planck equations and mean field games." *Archive for Rational Mechanics and Analysis*, 216(1), 1-62.

---

### Additional Classical References

**Optimal Transport Theory**:
- Santambrogio, F. (2015). *Optimal Transport for Applied Mathematicians*. Birkhäuser.
- Peyré, G., & Cuturi, M. (2019). "Computational optimal transport." *Foundations and Trends in Machine Learning*, 11(5-6), 355-607.

**Calculus of Variations**:
- Dacorogna, B. (2008). *Direct Methods in the Calculus of Variations* (2nd ed.). Springer.
- Evans, L. C. (2010). *Partial Differential Equations* (2nd ed.). AMS.

**Gradient Flows and Evolution Equations**:
- Otto, F. (2001). "The geometry of dissipative evolution equations: The porous medium equation." *Communications in Partial Differential Equations*, 26(1-2), 101-174.
- Mielke, A. (2011). "A gradient structure for reaction-diffusion systems and for energy-drift-diffusion systems." *Nonlinearity*, 24(4), 1329-1346.

**Computational Optimal Transport**:
- Flamary, R., & Courty, N. (2017). "POT Python optimal transport library." https://pythonot.github.io/
- Schmitzer, B. (2019). "Stabilized sparse scaling algorithms for entropy regularized transport problems." *SIAM Journal on Scientific Computing*, 41(3), A1443-A1481.

**Mean Field Games Variational Methods**:
- Gomes, D. A., Pimentel, E. A., & Voskanyan, V. (2016). *Regularity Theory for Mean-Field Game Systems*. Springer.
- Laurière, M., & Pironneau, O. (2016). "Dynamic programming for mean-field type control." *Comptes Rendus Mathematique*, 352(9), 707-713.

---

**Document Status**: Comprehensive mathematical foundation for variational MFG
**Usage**: Reference for optimization-based MFG solvers and theoretical analysis
**Related Code**: `mfg_pde/alg/optimization/wasserstein_gradient_flow.py`, `mfg_pde/alg/optimization/sinkhorn_solver.py`
**Related Theory**: `information_geometry_mfg.md` (geometric perspective), `mathematical_background.md` (foundations)
**Last Updated**: October 8, 2025
