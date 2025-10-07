# Stochastic Differential Games: Mathematical Foundations

**Document Type**: Theoretical Foundation
**Created**: October 8, 2025
**Status**: Planning Document for Phase 4 Implementation
**Related**: `STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` Phase 4

---

## Table of Contents

1. [Mathematical Preliminaries](#1-mathematical-preliminaries)
2. [N-Player Stochastic Differential Games](#2-n-player-stochastic-differential-games)
3. [Mean Field Game Limit](#3-mean-field-game-limit)
4. [Convergence Theory](#4-convergence-theory)
5. [Advanced Stochastic Calculus](#5-advanced-stochastic-calculus)
6. [Lévy Processes and Jump Diffusions](#6-lévy-processes-and-jump-diffusions)
7. [Stochastic PDEs](#7-stochastic-pdes)
8. [Rough Path Theory](#8-rough-path-theory)
9. [Regime-Switching Dynamics](#9-regime-switching-dynamics)
10. [Mean Field Games of Controls](#10-mean-field-games-of-controls)
11. [Computational Considerations](#11-computational-considerations)

---

## 1. Mathematical Preliminaries

### 1.1 Probability Spaces and Filtrations

**Probability Space**: $(\Omega, \mathcal{F}, \mathbb{P})$
- $\Omega$: Sample space
- $\mathcal{F}$: $\sigma$-algebra of events
- $\mathbb{P}$: Probability measure

**Filtration**: $\{\mathcal{F}_t\}_{t \geq 0}$ satisfying:
- **Right-continuous**: $\mathcal{F}_t = \bigcap_{s > t} \mathcal{F}_s$
- **$\mathbb{P}$-complete**: Contains all $\mathbb{P}$-null sets
- **Increasing**: $\mathcal{F}_s \subseteq \mathcal{F}_t$ for $s \leq t$

**Adapted Process**: $X_t$ is $\mathcal{F}_t$-measurable for all $t \geq 0$

### 1.2 Wiener Processes

**Definition**: $W_t$ is a standard Brownian motion if:
1. $W_0 = 0$ a.s.
2. Independent increments: $W_t - W_s \perp \mathcal{F}_s$ for $t > s$
3. Gaussian increments: $W_t - W_s \sim \mathcal{N}(0, t-s)$
4. Continuous paths: $t \mapsto W_t(\omega)$ is continuous a.s.

**Quadratic Variation**: $[W, W]_t = t$ (deterministic)

**Covariation**: For independent Brownian motions $W^1, W^2$:
$$[W^1, W^2]_t = 0$$

**Multidimensional Brownian Motion**: $W_t = (W_t^1, \ldots, W_t^d)$
- Each component is independent standard Brownian motion
- Covariance matrix: $\mathbb{E}[W_t W_t^T] = t I_d$

### 1.3 Stochastic Integrals

**Itô Integral** (for adapted, square-integrable processes):
$$I_t = \int_0^t f_s \, dW_s$$

**Properties**:
- **Martingale**: $\mathbb{E}[I_t | \mathcal{F}_s] = I_s$ for $s \leq t$
- **Isometry**: $\mathbb{E}[I_t^2] = \mathbb{E}\left[\int_0^t f_s^2 \, ds\right]$
- **Zero expectation**: $\mathbb{E}[I_t] = 0$

**Stratonovich Integral**:
$$\int_0^t f_s \circ dW_s = \int_0^t f_s \, dW_s + \frac{1}{2} \int_0^t \frac{\partial f}{\partial x}(W_s) \, ds$$

### 1.4 Itô's Formula

**Scalar Case**: For $X_t$ satisfying $dX_t = b_t \, dt + \sigma_t \, dW_t$:
$$df(t, X_t) = \left(\frac{\partial f}{\partial t} + b_t \frac{\partial f}{\partial x} + \frac{1}{2} \sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma_t \frac{\partial f}{\partial x} \, dW_t$$

**Vector Case**: For $X_t \in \mathbb{R}^d$ with $dX_t^i = b_t^i \, dt + \sum_{j=1}^m \sigma_t^{ij} \, dW_t^j$:
$$df(t, X_t) = \frac{\partial f}{\partial t} dt + \sum_{i=1}^d \frac{\partial f}{\partial x^i} dX_t^i + \frac{1}{2} \sum_{i,j=1}^d \frac{\partial^2 f}{\partial x^i \partial x^j} d[X^i, X^j]_t$$

where the quadratic covariation is:
$$d[X^i, X^j]_t = \sum_{k=1}^m \sigma_t^{ik} \sigma_t^{jk} \, dt$$

---

## 2. N-Player Stochastic Differential Games

### 2.1 Game Formulation

**State Dynamics** for player $i \in \{1, \ldots, N\}$:
$$dX_t^i = \alpha_t^i(X_t^i, \mu_t^N) \, dt + \sigma^i dW_t^i + \gamma^i dW_t^0$$

where:
- $X_t^i \in \mathbb{R}^d$: State of player $i$
- $\alpha_t^i$: Control of player $i$ (adapted process)
- $W_t^i$: Idiosyncratic Brownian motion (independent across $i$)
- $W_t^0$: Common noise (shared by all players)
- $\mu_t^N = \frac{1}{N} \sum_{j=1}^N \delta_{X_t^j}$: Empirical measure

**Cost Functional** for player $i$:
$$J^i(\alpha^1, \ldots, \alpha^N) = \mathbb{E}\left[\int_0^T L^i(X_t^i, \alpha_t^i, \mu_t^N) \, dt + g^i(X_T^i, \mu_T^N)\right]$$

### 2.2 Nash Equilibrium

**Definition**: $(\alpha^{1*}, \ldots, \alpha^{N*})$ is a Nash equilibrium if for all $i$:
$$J^i(\alpha^{i*}, \alpha^{-i*}) \leq J^i(\alpha^i, \alpha^{-i*})$$
for all admissible $\alpha^i$, where $\alpha^{-i*} = (\alpha^{1*}, \ldots, \alpha^{i-1*}, \alpha^{i+1*}, \ldots, \alpha^{N*})$.

### 2.3 Hamilton-Jacobi-Bellman Equations

**Value Function** for player $i$:
$$V^{i,N}(t, x, \mu^N) = \inf_{\alpha^i} \mathbb{E}\left[\int_t^T L^i(X_s^i, \alpha_s^i, \mu_s^N) \, ds + g^i(X_T^i, \mu_T^N) \mid X_t^i = x\right]$$

**Coupled HJB System** (at Nash equilibrium):
$$\begin{cases}
\frac{\partial V^{i,N}}{\partial t} + H^i(x, \nabla_x V^{i,N}, \mu^N) + \frac{(\sigma^i)^2}{2} \Delta_x V^{i,N} + \frac{(\gamma^i)^2}{2} \mathbb{E}[\Delta_x V^{i,N} \mid W^0] = 0 \\
V^{i,N}(T, x, \mu^N) = g^i(x, \mu^N)
\end{cases}$$

where the Hamiltonian is:
$$H^i(x, p, \mu^N) = \inf_{\alpha} \left\{\alpha \cdot p + L^i(x, \alpha, \mu^N)\right\}$$

**Optimal Control**:
$$\alpha^{i*}(x, \mu^N) = \arg\min_{\alpha} \left\{\alpha \cdot \nabla_x V^{i,N}(t, x, \mu^N) + L^i(x, \alpha, \mu^N)\right\}$$

### 2.4 Empirical Measure Evolution

**Fokker-Planck Equation** for $\mu_t^N = \frac{1}{N}\sum_{j=1}^N \delta_{X_t^j}$:

In the weak formulation, for any test function $\varphi$:
$$\frac{d}{dt} \int \varphi(x) \, \mu_t^N(dx) = \frac{1}{N} \sum_{j=1}^N \left[\alpha^{j*}(X_t^j, \mu_t^N) \cdot \nabla \varphi(X_t^j) + \frac{(\sigma^j)^2}{2} \Delta \varphi(X_t^j)\right]$$

---

## 3. Mean Field Game Limit

### 3.1 MFG System (N → ∞ Limit)

**Hamilton-Jacobi-Bellman Equation**:
$$\begin{cases}
\frac{\partial u}{\partial t} + H(x, \nabla u, m) + \frac{\sigma^2}{2} \Delta u = 0, & (t,x) \in [0,T] \times \mathbb{R}^d \\
u(T, x) = g(x, m_T)
\end{cases}$$

**Fokker-Planck Equation**:
$$\begin{cases}
\frac{\partial m}{\partial t} - \text{div}(m \nabla_p H(x, \nabla u, m)) - \frac{\sigma^2}{2} \Delta m = 0, & (t,x) \in [0,T] \times \mathbb{R}^d \\
m(0, x) = m_0(x)
\end{cases}$$

**Mean Field Hamiltonian**:
$$H(x, p, m) = \inf_{\alpha} \left\{\alpha \cdot p + L(x, \alpha, m)\right\}$$

### 3.2 Fixed Point Formulation

The MFG system seeks $(u, m)$ such that:
1. $u$ solves HJB with measure $m$
2. $m$ is the distribution of optimal trajectories under control $\alpha^* = \arg\min_\alpha [\alpha \cdot \nabla u + L]$

**Fixed Point Operator**: $\Phi: m \mapsto m'$ where:
- Solve HJB for $u$ given $m$
- Compute optimal control $\alpha^*(x) = -\nabla_p H(x, \nabla u(x), m)$
- Solve FP with drift $\alpha^*$ to get $m'$
- Fixed point: $\Phi(m) = m$

### 3.3 Common Noise Extension

**Conditional MFG System** (conditioned on common noise path $\theta_t = W_t^0$):

**Conditional HJB**:
$$\frac{\partial u^\theta}{\partial t} + H(x, \nabla u^\theta, m^\theta, \theta_t) + \frac{\sigma^2}{2} \Delta u^\theta = 0$$

**Conditional Fokker-Planck**:
$$\frac{\partial m^\theta}{\partial t} - \text{div}(m^\theta \nabla_p H(x, \nabla u^\theta, m^\theta, \theta_t)) - \frac{\sigma^2}{2} \Delta m^\theta = 0$$

**Solution via Monte Carlo**: Average over many common noise realizations:
$$\bar{u}(t,x) = \mathbb{E}[u^{\theta}(t,x)], \quad \bar{m}(t,x) = \mathbb{E}[m^{\theta}(t,x)]$$

---

## 4. Convergence Theory

### 4.1 Convergence of Value Functions

**Theorem (Non-Asymptotic Convergence)**:
Under regularity assumptions (Lipschitz Hamiltonian, bounded derivatives), there exists $C > 0$ such that:
$$\sup_{t \in [0,T], x \in \mathbb{R}^d} \left|V^{i,N}(t, x, \mu^N) - u(t, x, m)\right| \leq \frac{C}{\sqrt{N}}$$

where:
- $V^{i,N}$: Value function for player $i$ in N-player game
- $u$: MFG value function
- $m$: MFG measure (limit of $\mu^N$)

**Proof Sketch**:
1. **Master Equation**: Lift to Wasserstein space $\mathcal{P}_2(\mathbb{R}^d)$
2. **Propagation of Chaos**: $\mu^N \to m$ in $W_2$ distance with rate $1/\sqrt{N}$
3. **Stability Estimate**: Lipschitz dependence of HJB solution on measure

### 4.2 Convergence Rates

**Empirical Measure Convergence**:
$$\mathbb{E}\left[W_2^2(\mu_t^N, m_t)\right] \leq \frac{C}{N}$$

where $W_2$ is the 2-Wasserstein distance.

**Trajectory Convergence**:
For the law $\mathcal{L}(X_t^i \mid \mu_0^N)$ of a tagged particle:
$$W_2\left(\mathcal{L}(X_t^i \mid \mu_0^N), m_t\right) \leq \frac{C}{\sqrt{N}}$$

### 4.3 Regularity Assumptions

**Conditions for $O(1/\sqrt{N})$ Convergence**:

1. **Lipschitz Hamiltonian**:
   $$|H(x, p, \mu) - H(x, p, \nu)| \leq L_H W_1(\mu, \nu)$$

2. **Regularity of MFG Solution**:
   $$\|u\|_{W^{2,\infty}} + \|\nabla_x u\|_{\infty} + \|\nabla_m u\|_{\infty} \leq C$$

3. **Interaction Kernel Smoothness**:
   $$L(x, \alpha, m) = \int K(x, y) \, m(dy) + \ell(x, \alpha)$$
   with $K \in C^{2,1}$

4. **Finite Moments**:
   $$\sup_{t \in [0,T]} \mathbb{E}\left[\int |x|^p \, m_t(dx)\right] < \infty, \quad p \geq 2$$

---

## 5. Advanced Stochastic Calculus

### 5.1 Malliavin Calculus

**Malliavin Derivative** $D_t$: Operator on Wiener functionals
$$D_t F = \lim_{h \to 0} \frac{F(W + h \mathbb{1}_{[t,T]}) - F(W)}{h}$$

**Properties**:
- **Chain Rule**: $D_t[f(F)] = f'(F) D_t F$
- **Product Rule**: $D_t[FG] = F D_t G + G D_t F$
- **Integration by Parts**:
  $$\mathbb{E}\left[F \int_0^T u_s \, dW_s\right] = \mathbb{E}\left[\int_0^T u_s D_s F \, ds\right]$$

**Skorohod Integral** (adjoint of $D$):
$$\delta(u) = \int_0^T u_s \, dW_s - \int_0^T D_s u_s \, ds$$

**Applications**:
- **Sensitivity Analysis**: Compute $\frac{\partial}{\partial \theta} \mathbb{E}[F(\theta)]$ via Malliavin derivatives
- **Variance Reduction**: Control variates using $D_t F$
- **Greeks in Finance**: $\Delta = \frac{\partial V}{\partial S_0}$ computed via Malliavin calculus

### 5.2 Girsanov Theorem

**Measure Change**: Define new measure $\mathbb{Q}$ via Radon-Nikodym derivative:
$$\frac{d\mathbb{Q}}{d\mathbb{P}} = \exp\left(\int_0^T \theta_s \, dW_s - \frac{1}{2} \int_0^T \theta_s^2 \, ds\right)$$

**Transformed Brownian Motion**: Under $\mathbb{Q}$:
$$\tilde{W}_t = W_t - \int_0^T \theta_s \, ds$$
is a Brownian motion.

**Application to Control**: Transform optimal control problem via measure change
$$\mathbb{E}^{\mathbb{P}}\left[e^{-\int_0^T r_s ds} g(X_T)\right] = \mathbb{E}^{\mathbb{Q}}[g(X_T)]$$

### 5.3 Doob-Meyer Decomposition

**Theorem**: Every submartingale $Y_t$ admits unique decomposition:
$$Y_t = M_t + A_t$$
where $M_t$ is a martingale and $A_t$ is predictable increasing process.

**Application**: Decompose value function evolution into martingale + drift components.

---

## 6. Lévy Processes and Jump Diffusions

### 6.1 Lévy Processes

**Definition**: $L_t$ is a Lévy process if:
1. $L_0 = 0$ a.s.
2. Independent increments: $L_t - L_s \perp \mathcal{F}_s$
3. Stationary increments: $L_t - L_s \sim L_{t-s}$
4. Stochastic continuity: $L_t \to L_s$ in probability as $t \to s$

**Lévy-Khintchine Representation**:
$$\mathbb{E}[e^{i\theta L_t}] = \exp\left(t \psi(\theta)\right)$$
where the characteristic exponent is:
$$\psi(\theta) = i b \theta - \frac{\sigma^2}{2} \theta^2 + \int_{\mathbb{R}} (e^{i\theta x} - 1 - i\theta x \mathbb{1}_{|x| < 1}) \, \nu(dx)$$

Components:
- $b \in \mathbb{R}$: Drift
- $\sigma \geq 0$: Diffusion coefficient
- $\nu$: Lévy measure satisfying $\int_{\mathbb{R}} (1 \wedge x^2) \, \nu(dx) < \infty$

### 6.2 Compound Poisson Process

**Definition**:
$$J_t = \sum_{i=1}^{N_t} Y_i$$
where $N_t$ is Poisson($\lambda t$) and $Y_i \sim F$ are i.i.d. jump sizes.

**Lévy Measure**:
$$\nu(dx) = \lambda F(dx)$$

**Moments**:
$$\mathbb{E}[J_t] = \lambda t \mathbb{E}[Y], \quad \text{Var}(J_t) = \lambda t \mathbb{E}[Y^2]$$

### 6.3 Jump-Diffusion MFG

**State Dynamics with Jumps**:
$$dX_t = \alpha_t \, dt + \sigma dW_t + dJ_t$$

where $J_t$ is a compound Poisson process with intensity $\lambda$ and jump distribution $F$.

**PIDE (Partial Integro-Differential Equation)**:
$$\begin{multline}
\frac{\partial u}{\partial t} + H(x, \nabla u, m) + \frac{\sigma^2}{2} \Delta u \\
+ \lambda \int_{\mathbb{R}^d} [u(t, x+y) - u(t,x) - \nabla u(t,x) \cdot y \mathbb{1}_{|y| < 1}] \, F(dy) = 0
\end{multline}$$

**Jump Fokker-Planck**:
$$\begin{multline}
\frac{\partial m}{\partial t} - \text{div}(m \nabla_p H) - \frac{\sigma^2}{2} \Delta m \\
+ \lambda \int_{\mathbb{R}^d} [m(t, x-y) F(dy) - m(t,x)] = 0
\end{multline}$$

### 6.4 Alpha-Stable Processes

**Definition**: $L_t$ is $\alpha$-stable if:
$$\mathbb{E}[e^{i\theta L_t}] = \exp\left(-t |\theta|^\alpha e^{-i \frac{\pi \alpha}{2} \beta \text{sign}(\theta)}\right)$$

Parameters:
- $\alpha \in (0, 2]$: Stability index (tail behavior)
- $\beta \in [-1, 1]$: Skewness parameter

**Heavy Tails**: For $\alpha < 2$, infinite variance:
$$\mathbb{P}(|L_t| > x) \sim C x^{-\alpha} \text{ as } x \to \infty$$

**Applications**:
- Financial models with extreme events (crashes)
- Anomalous diffusion in physics
- Heavy-tailed interaction kernels in MFG

---

## 7. Stochastic PDEs

### 7.1 General SPDE Framework

**Abstract SPDE**:
$$dU_t = (AU_t + F(U_t)) \, dt + G(U_t) \, dW_t$$

where:
- $A$: Linear differential operator (e.g., Laplacian)
- $F$: Nonlinear drift
- $G$: Diffusion operator
- $W_t$: Cylindrical Wiener process on Hilbert space

### 7.2 Stochastic Heat Equation

**Additive Noise**:
$$\frac{\partial u}{\partial t} = \kappa \Delta u + \sigma \dot{W}(t, x)$$

where $\dot{W}(t, x)$ is space-time white noise.

**Mild Solution**: Using semigroup $S(t) = e^{t\kappa\Delta}$:
$$u(t, x) = \int_{\mathbb{R}^d} S(t)(x - y) u_0(y) \, dy + \sigma \int_0^t \int_{\mathbb{R}^d} S(t-s)(x-y) \, W(ds, dy)$$

**Multiplicative Noise**:
$$\frac{\partial u}{\partial t} = \kappa \Delta u + \sigma u \dot{W}(t,x)$$

**Stratonovich vs Itô**: Multiplicative noise interpretation affects drift term:
$$\text{Stratonovich: } \frac{\partial u}{\partial t} = \kappa \Delta u + \frac{\sigma^2}{2} u + \sigma u \circ \dot{W}$$

### 7.3 Numerical Methods for SPDE

**Spatial Discretization**:

1. **Finite Differences**: $u(t,x) \approx u_i^n$ on grid
   $$\frac{u_i^{n+1} - u_i^n}{\Delta t} = \kappa \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{\Delta x^2} + \sigma \Delta W_i^n$$

2. **Finite Elements**: $u_h(t,x) = \sum_{j=1}^N u_j(t) \phi_j(x)$
   - Weak formulation with test functions
   - Mass matrix $M$ and stiffness matrix $K$
   - $M \frac{du}{dt} = -K u + \text{noise}$

3. **Spectral Methods**: $u(t,x) = \sum_{k} u_k(t) e^{ikx}$
   - Fourier transform diagonalizes Laplacian
   - Mode-by-mode stochastic ODE

**Time Integration**:

1. **Euler-Maruyama**: $u^{n+1} = u^n + \Delta t (Au^n + F(u^n)) + \sqrt{\Delta t} G(u^n) \xi^n$
2. **Exponential Euler**: $u^{n+1} = e^{\Delta t A} u^n + \int_0^{\Delta t} e^{({\Delta t - s})A} G(u^n) \, dW_s$
3. **Crank-Nicolson**: Implicit-explicit scheme for stability

---

## 8. Rough Path Theory

### 8.1 Rough Paths and Signatures

**Rough Path**: Pair $(X, \mathbb{X})$ where:
- $X : [0,T] \to \mathbb{R}^d$ is $\alpha$-Hölder continuous, $\alpha > 1/3$
- $\mathbb{X}_{s,t} = \int_s^t (X_r - X_s) \otimes dX_r$ (Lévy area, iterated integral)

**Signature**: Infinite sequence of iterated integrals
$$\text{Sig}(X)_{s,t} = \left(1, X_{s,t}, \mathbb{X}_{s,t}, \int_s^t (X_r - X_s) \otimes (X_r - X_s) \otimes dX_r, \ldots\right)$$

**Truncated Signature**: Keep terms up to level $N$
$$\text{Sig}^{(N)}(X)_{s,t} = (1, X_{s,t}, \mathbb{X}_{s,t}, \ldots, \int_{s < r_1 < \cdots < r_N < t} dX_{r_1} \otimes \cdots \otimes dX_{r_N})$$

### 8.2 Rough Differential Equations

**RDE Driven by Rough Path**:
$$dY_t = f(Y_t) \, dX_t$$

where $X$ is $\alpha$-Hölder with $\alpha \in (1/3, 1/2]$ (too rough for classical Itô/Stratonovich).

**Universal Limit Theorem**: Solution $Y$ depends continuously on rough path $(X, \mathbb{X})$:
$$\|Y - Y'\|_{\alpha} \leq C \|(X, \mathbb{X}) - (X', \mathbb{X}')\|_{\alpha\text{-Hölder}}$$

**Application to SDEs**: Enhanced Brownian motion $(W, \mathbb{W})$ with Lévy area
$$\mathbb{W}_{s,t}^{ij} = \int_s^t (W_r^i - W_s^i) \, dW_r^j$$

allows pathwise SDE theory for irregular coefficients.

### 8.3 Signatures for Machine Learning

**Feature Extraction**: Map path $X_{[0,T]}$ to signature $\text{Sig}^{(N)}(X)_{0,T}$
- **Universality**: Signature uniquely characterizes path (up to reparametrization)
- **Invariance**: Signature is invariant under time reparametrization
- **Linearity**: $\text{Sig}(X \oplus Y) = \text{Sig}(X) \otimes \text{Sig}(Y)$ (concatenation)

**Neural Signature Layers**: Learn transformations of signatures
$$h = \text{ReLU}(W \cdot \text{Sig}^{(N)}(X) + b)$$

---

## 9. Regime-Switching Dynamics

### 9.1 Piecewise Deterministic Markov Processes

**Hybrid State**: $(X_t, \theta_t)$ where:
- $X_t \in \mathbb{R}^d$: Continuous state
- $\theta_t \in \{1, \ldots, K\}$: Discrete regime

**Dynamics**:
$$dX_t = \alpha_{\theta_t}(X_t, m_t) \, dt + \sigma_{\theta_t} \, dW_t$$

**Regime Transitions**: Markov chain with rate matrix $Q = (Q_{ij})$
- $\theta_t$ jumps from $i$ to $j$ with rate $Q_{ij}$
- Generator: $\mathcal{Q} f(\theta) = \sum_{j} Q_{\theta j} [f(j) - f(\theta)]$

**Infinitesimal Generator**: For $F(x, \theta)$:
$$\mathcal{L} F = \alpha_\theta \cdot \nabla_x F + \frac{\sigma_\theta^2}{2} \Delta_x F + \sum_{j} Q_{\theta j} [F(x, j) - F(x, \theta)]$$

### 9.2 Regime-Switching MFG

**Coupled HJB System**:
$$\frac{\partial u_\theta}{\partial t} + H_\theta(x, \nabla u_\theta, m_\theta) + \frac{\sigma_\theta^2}{2} \Delta u_\theta + \sum_{k=1}^K Q_{\theta k} (u_k - u_\theta) = 0$$

for each regime $\theta \in \{1, \ldots, K\}$.

**Coupled Fokker-Planck**:
$$\frac{\partial m_\theta}{\partial t} - \text{div}(m_\theta \nabla_p H_\theta) - \frac{\sigma_\theta^2}{2} \Delta m_\theta + \sum_{k=1}^K (Q_{k\theta} m_k - Q_{\theta k} m_\theta) = 0$$

**Equilibrium Measure**: Aggregate distribution
$$m_{\text{total}}(t, x) = \sum_{\theta=1}^K \pi_\theta(t) m_\theta(t, x)$$

where $\pi_\theta(t)$ is probability of being in regime $\theta$ at time $t$.

### 9.3 Applications

**Financial Markets**:
- **Bull/Bear Regimes**: $\theta \in \{\text{bull}, \text{bear}\}$
  - Bull: High expected return, low volatility
  - Bear: Low/negative return, high volatility
- **Transition rates**: Asymmetric (easier to enter bear than exit)

**Epidemic Models**:
- **Seasonal Transmission**: $\theta \in \{\text{winter}, \text{summer}\}$
  - Winter: High transmission rate $\beta_{\text{winter}}$
  - Summer: Low transmission rate $\beta_{\text{summer}}$

**Energy Markets**:
- **Peak/Off-Peak Pricing**: $\theta \in \{\text{peak}, \text{off-peak}\}$
  - Peak hours: High demand, high price
  - Off-peak: Low demand, low price

---

## 10. Mean Field Games of Controls (MFGC)

### 10.1 Formulation

**Key Difference from Classical MFG**: Interaction through distribution of **controls** $\nu_t = \text{Law}(\alpha_t)$, not states.

**State Dynamics**:
$$dX_t = \alpha_t \, dt + \sigma \, dW_t$$

**Cost Functional**:
$$J(\alpha) = \mathbb{E}\left[\int_0^T L(X_t, \alpha_t, \nu_t) \, dt + g(X_T)\right]$$

where $L$ depends on control distribution $\nu_t \in \mathcal{P}(\mathbb{R}^d)$.

**Fixed Point Condition**:
$$\nu_t = \text{Law}(\alpha_t^*), \quad \alpha_t^* = \arg\min_\alpha \left\{\alpha \cdot \nabla u(t, X_t) + L(X_t, \alpha, \nu_t)\right\}$$

### 10.2 MFGC System

**Hamilton-Jacobi-Bellman**:
$$\frac{\partial u}{\partial t} + \inf_{\alpha} \left\{\alpha \cdot \nabla u + L(x, \alpha, \nu_t)\right\} + \frac{\sigma^2}{2} \Delta u = 0$$

**Fokker-Planck for State Distribution** $m_t = \text{Law}(X_t)$:
$$\frac{\partial m}{\partial t} - \text{div}(m \alpha^*(\cdot, \nu_t)) - \frac{\sigma^2}{2} \Delta m = 0$$

**Control Distribution Evolution**: For any test function $\psi$:
$$\int \psi(\alpha) \, \nu_t(d\alpha) = \int \psi(\alpha^*(x, \nu_t)) \, m_t(dx)$$

**Self-Consistency**:
$$\nu_t(A) = m_t\left(\{x : \alpha^*(x, \nu_t) \in A\}\right)$$

### 10.3 Examples

**Energy Market (Generator Dispatch)**:
- **State**: Generator capacity $X_t \in [0, C]$
- **Control**: Production level $\alpha_t \in [0, X_t]$
- **Interaction**: Price depends on total production $\int \alpha \, \nu_t(d\alpha)$
- **Cost**: $L(x, \alpha, \nu_t) = -\alpha \cdot p(\bar{\alpha}_t) + c(\alpha)$ where $\bar{\alpha}_t = \int \alpha \, \nu_t(d\alpha)$

**Autonomous Vehicle Speed Choice**:
- **State**: Position $X_t$ on road
- **Control**: Speed $\alpha_t$
- **Interaction**: Congestion depends on speed distribution $\nu_t$
- **Cost**: Travel time + congestion penalty depending on $\nu_t$

---

## 11. Computational Considerations

### 11.1 N-Player Game Solvers

**Coupled System Size**: $N$ coupled HJB equations
- **Curse of Dimensionality**: Exponential in $N$ if solved directly
- **Mean Field Approximation**: Reduce to single HJB + FP (linear in $N$)

**Numerical Methods**:
1. **Policy Iteration**:
   - Initialize $\alpha^{i,0}$ for all $i$
   - Iterate: Solve HJB given $\alpha^{-i,k}$, update $\alpha^{i,k+1}$
   - Converges to Nash equilibrium

2. **Fictitious Play**:
   - Players best-respond sequentially to current strategy profile
   - Average empirical play converges to equilibrium

3. **Monte Carlo Sampling**: Approximate $\mu^N$ via particles
   - Avoid full discretization of measure space
   - Update particles according to optimal control

### 11.2 Convergence Rate Estimation

**Empirical Rate Fitting**: For sequence $N = 10, 20, 50, 100, \ldots$:
1. Compute $E_N = |V^{i,N} - V^{\text{MFG}}|$
2. Fit power law: $E_N \approx C N^{-\alpha}$
3. Estimate $\alpha$ via log-log regression: $\log E_N \approx \log C - \alpha \log N$
4. Compare with theoretical $\alpha = 0.5$

**Bootstrap Confidence Intervals**:
1. Resample data $(N_i, E_{N_i})$ with replacement
2. Refit power law on bootstrap sample
3. Repeat to get distribution of $\hat{\alpha}$
4. Compute 95% confidence interval for $\alpha$

### 11.3 PIDE Solvers for Jump Diffusions

**Finite Difference + Quadrature**:
$$\frac{u_i^{n+1} - u_i^n}{\Delta t} = \text{(diffusion terms)} + \lambda \sum_{j} w_j [u(x_i + y_j) - u(x_i)]$$

where $(y_j, w_j)$ are quadrature points/weights for Lévy measure $\nu$.

**Lévy Measure Truncation**:
- Small jumps: $|y| < \epsilon$ treated as diffusion correction
- Large jumps: $|y| \geq \epsilon$ handled explicitly via quadrature

**Fast Fourier Transform (FFT)**:
- Convolution structure: $\int u(x+y) \nu(dy)$ via FFT
- $O(N \log N)$ complexity vs $O(N^2)$ direct summation

### 11.4 SPDE Spatial Discretization

**Spectral Galerkin**:
- Expand: $u(t,x) = \sum_{k} u_k(t) \phi_k(x)$
- Project SPDE onto each mode
- Diagonal Laplacian in Fourier basis

**Finite Element Method**:
- Triangulation $\mathcal{T}_h$ with element size $h$
- Basis functions $\{\phi_j\}_{j=1}^{N_h}$
- Weak formulation: $\int \phi_i (\partial_t u - \Delta u) dx = \int \phi_i \sigma \dot{W} dx$
- Mass matrix $M_{ij} = \int \phi_i \phi_j$ and stiffness $K_{ij} = \int \nabla \phi_i \cdot \nabla \phi_j$

**Convergence**: For P1 elements (piecewise linear):
$$\mathbb{E}\left[\|u - u_h\|_{L^2}^2\right] \leq C h^2 \|u\|_{H^2}^2$$

---

## References

### Classical MFG Theory
- Lasry, J. M., & Lions, P. L. (2007). "Mean field games." *Japanese Journal of Mathematics*.
- Cardaliaguet, P. (2013). "Notes on Mean Field Games." [Lecture notes]

### Convergence Theory
- Cardaliaguet, P., et al. (2019). "Master equation for the finite state space mean field game problem."
- Delarue, F., & Lacker, D. (2022). "From the master equation to mean field games." *Probability Theory and Related Fields*.

### Stochastic Calculus
- Øksendal, B. (2003). *Stochastic Differential Equations*. Springer.
- Nualart, D. (2006). *The Malliavin Calculus and Related Topics*. Springer.

### Lévy Processes
- Applebaum, D. (2009). *Lévy Processes and Stochastic Calculus*. Cambridge University Press.
- Cont, R., & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall.

### Rough Paths
- Friz, P., & Hairer, M. (2014). *A Course on Rough Paths*. Springer.
- Lyons, T., Caruana, M., & Lévy, T. (2007). *Differential Equations Driven by Rough Paths*. Springer.

### Stochastic PDEs
- Da Prato, G., & Zabczyk, J. (2014). *Stochastic Equations in Infinite Dimensions*. Cambridge University Press.
- Lord, G. J., Powell, C. E., & Shardlow, T. (2014). *An Introduction to Computational Stochastic PDEs*. Cambridge University Press.

### Regime-Switching Models
- Davis, M. H. A. (1984). "Piecewise-deterministic Markov processes." *Journal of the Royal Statistical Society*.
- Guo, X., & Zhang, Q. (2004). *Closed-form Solutions for Perpetual American Options in Regime-Switching Models*. Springer.

### Mean Field Games of Controls
- Carmona, R., & Delarue, F. (2018). *Probabilistic Theory of Mean Field Games with Applications*. Springer.
- Cardaliaguet, P., & Porretta, A. (2019). "An introduction to mean field game theory." *Paris-Princeton Lectures on Mathematical Finance*.

---

**Document Status**: Planning and theoretical foundation for Phase 4 implementation
**Next Steps**: Implement computational modules following this mathematical framework
**Related Implementation**: `STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` Phase 4.0-4.5
