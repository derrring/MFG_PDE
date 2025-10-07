# Mathematical Background for Mean Field Games

**Document Type**: Foundational Reference
**Created**: October 2025
**Status**: Complete mathematical foundations
**Related**: All theory documents, `NOTATION_STANDARDS.md`

---

## Table of Contents

1. [Function Spaces and Topologies](#1-function-spaces-and-topologies)
2. [Stochastic Processes](#2-stochastic-processes)
3. [Partial Differential Equations](#3-partial-differential-equations)
4. [Optimal Control Theory](#4-optimal-control-theory)
5. [Measure Theory and Weak Convergence](#5-measure-theory-and-weak-convergence)
6. [Viscosity Solutions](#6-viscosity-solutions)
7. [Numerical Analysis Foundations](#7-numerical-analysis-foundations)

---

## 1. Function Spaces and Topologies

### 1.1 Continuous Function Spaces

**Definition**: For domain $\Omega \subseteq \mathbb{R}^d$:[^1]

- **$C(\Omega)$**: Continuous functions on $\Omega$
- **$C^k(\Omega)$**: $k$-times continuously differentiable functions
- **$C_b(\Omega)$**: Bounded continuous functions
  $$\|f\|_{C_b} = \sup_{x \in \Omega} |f(x)| < \infty$$
- **$C_c(\Omega)$**: Continuous functions with compact support
- **$C^\infty_c(\Omega)$**: Smooth functions with compact support (test functions)

### 1.2 Lebesgue Spaces

**Definition**: For $p \in [1, \infty]$:[^2]

**$L^p(\Omega)$**: Lebesgue $p$-integrable functions
$$\|f\|_{L^p} = \left(\int_\Omega |f(x)|^p dx\right)^{1/p} < \infty$$

**$L^\infty(\Omega)$**: Essentially bounded functions
$$\|f\|_{L^\infty} = \text{ess sup}_{x \in \Omega} |f(x)| < \infty$$

**Key Properties**:
- **Completeness**: $(L^p, \|\cdot\|_{L^p})$ is a Banach space
- **Duality**: $(L^p)^* = L^q$ where $1/p + 1/q = 1$
- **Hölder Inequality**: $\|fg\|_{L^1} \leq \|f\|_{L^p} \|g\|_{L^q}$

### 1.3 Sobolev Spaces

**Definition**: [^3]

**$W^{k,p}(\Omega)$**: Functions with weak derivatives up to order $k$ in $L^p$
$$W^{k,p}(\Omega) = \{f \in L^p(\Omega) : D^\alpha f \in L^p(\Omega), \, |\alpha| \leq k\}$$

**Norm**:
$$\|f\|_{W^{k,p}} = \left(\sum_{|\alpha| \leq k} \|D^\alpha f\|_{L^p}^p\right)^{1/p}$$

**Special Case**: $H^k(\Omega) = W^{k,2}(\Omega)$ (Hilbert space)

**Sobolev Embedding** (Continuous injection): For $d < p$,
$$W^{1,p}(\mathbb{R}^d) \hookrightarrow C_b(\mathbb{R}^d)$$

### 1.4 Spaces of Probability Measures

**Definition**: [^4]

**$\mathcal{P}(\mathbb{R}^d)$**: All probability measures on $\mathbb{R}^d$

**$\mathcal{P}_{ac}(\mathbb{R}^d)$**: Absolutely continuous measures w.r.t. Lebesgue
$$\mu(dx) = m(x) dx, \quad m \in L^1(\mathbb{R}^d), \, \int m = 1$$

**$\mathcal{P}_p(\mathbb{R}^d)$**: Measures with finite $p$-th moment
$$\mathcal{P}_p = \left\{\mu : \int |x|^p \mu(dx) < \infty\right\}$$

**Weak Convergence**: $\mu_n \rightharpoonup \mu$ if
$$\int f \, d\mu_n \to \int f \, d\mu \quad \forall f \in C_b(\mathbb{R}^d)$$

---

## 2. Stochastic Processes

### 2.1 Brownian Motion

**Definition**: [^5] $W_t$ is a standard $d$-dimensional Brownian motion if:
1. $W_0 = 0$ almost surely
2. Independent increments: $W_t - W_s \perp \mathcal{F}_s$ for $t > s$
3. Gaussian increments: $W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$
4. Continuous paths: $t \mapsto W_t(\omega)$ is continuous a.s.

**Properties**:
- **Martingale**: $\mathbb{E}[W_t | \mathcal{F}_s] = W_s$
- **Quadratic Variation**: $[W, W]_t = t$ (deterministic)
- **Markov Property**: $\mathbb{P}(W_t \in A | \mathcal{F}_s) = \mathbb{P}(W_t \in A | W_s)$

### 2.2 Itô Calculus

**Itô Integral**: For adapted $f \in L^2([0,T] \times \Omega)$:
$$I_t = \int_0^t f_s \, dW_s$$

**Properties**:
- **Martingale**: $\mathbb{E}[I_t | \mathcal{F}_s] = I_s$
- **Isometry**: $\mathbb{E}[I_t^2] = \mathbb{E}\left[\int_0^t f_s^2 ds\right]$

**Itô's Formula**: [^6] For $X_t$ with $dX_t = b_t dt + \sigma_t dW_t$ and $f \in C^{1,2}$:
$$df(t, X_t) = \left(\frac{\partial f}{\partial t} + b_t \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma_t \frac{\partial f}{\partial x} dW_t$$

**Key Point**: The second-order term $\frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}$ is the **Itô correction**.

### 2.3 Stochastic Differential Equations

**SDE**: [^7]
$$dX_t = b(X_t, t) dt + \sigma(X_t, t) dW_t, \quad X_0 = x_0$$

**Theorem (Existence and Uniqueness)**: If $b, \sigma$ are Lipschitz and have linear growth, there exists a unique strong solution.

**Fokker-Planck-Kolmogorov Equation**: The density $m(t,x)$ of $X_t$ satisfies:
$$\frac{\partial m}{\partial t} + \text{div}(b m) = \frac{1}{2}\text{div}(\sigma \sigma^T \nabla m)$$

---

## 3. Partial Differential Equations

### 3.1 Classification of PDEs

**Parabolic PDE** (e.g., heat equation, Fokker-Planck):
$$\frac{\partial u}{\partial t} = \Delta u + \text{lower order terms}$$

**Elliptic PDE** (e.g., Poisson equation):
$$-\Delta u = f$$

**Hyperbolic PDE** (e.g., wave equation):
$$\frac{\partial^2 u}{\partial t^2} = c^2 \Delta u$$

**Fully Nonlinear** (e.g., Hamilton-Jacobi-Bellman):
$$F(x, u, Du, D^2u) = 0$$

### 3.2 Hamilton-Jacobi-Bellman Equation

**General Form**: [^8]
$$\begin{cases}
-\frac{\partial u}{\partial t} + H(x, \nabla u, D^2u) = 0, & (t,x) \in [0,T] \times \Omega \\
u(T, x) = g(x), & x \in \Omega
\end{cases}$$

**For MFG**: Hamiltonian depends on measure $m$:
$$-\frac{\partial u}{\partial t} + H(x, \nabla u, m) + \frac{\sigma^2}{2}\Delta u = 0$$

**Optimal Control Representation**:
$$H(x, p, m) = \inf_{\alpha \in \mathcal{A}} \{\alpha \cdot p + L(x, \alpha, m)\}$$

**Convexity**: $H$ is convex in $p$ (fundamental for MFG theory)

### 3.3 Fokker-Planck Equation

**Forward Equation** for density evolution:[^9]
$$\frac{\partial m}{\partial t} - \text{div}(m \nabla_p H(x, \nabla u, m)) - \frac{\sigma^2}{2}\Delta m = 0$$

**Conservation of Mass**:
$$\frac{d}{dt}\int_{\mathbb{R}^d} m(t,x) dx = 0$$

**Connection to SDE**: If agents follow SDE with drift $\alpha^*(x) = -\nabla_p H$, then $m$ is the resulting density.

### 3.4 Boundary Conditions

**Dirichlet**: $u|_{\partial \Omega} = g$ (prescribed value)

**Neumann**: $\frac{\partial u}{\partial n}\bigg|_{\partial \Omega} = 0$ (no-flux)

**Periodic**: $u(0,x) = u(L,x)$ (periodic domain)

**Natural Boundary**: For $\Omega = \mathbb{R}^d$, impose decay at infinity:
$$\lim_{|x| \to \infty} u(t,x) = 0$$

---

## 4. Optimal Control Theory

### 4.1 Dynamic Programming Principle

**Value Function**: [^10]
$$V(t,x) = \inf_{\alpha \in \mathcal{A}} \mathbb{E}\left[\int_t^T L(X_s, \alpha_s) ds + g(X_T) \mid X_t = x\right]$$

**Bellman Optimality**:
$$V(t,x) = \inf_{\alpha} \mathbb{E}\left[L(x, \alpha) dt + V(t+dt, x + \alpha dt + \sigma dW_t)\right]$$

**HJB Derivation**: Apply Itô's formula to $V(t+dt, X_{t+dt})$ and take $dt \to 0$.

### 4.2 Verification Theorem

**Theorem**: [^11] If $u$ solves the HJB equation and $u$ has sufficient regularity, then:
1. $u(t,x) = V(t,x)$ (value function)
2. $\alpha^*(x) = \arg\min_\alpha \{\alpha \cdot \nabla u + L(x,\alpha)\}$ is optimal

**Sufficient Conditions**:
- $u \in C^{1,2}$
- Polynomial growth: $|u(t,x)| \leq C(1 + |x|^p)$

### 4.3 Pontryagin Maximum Principle

**Adjoint Equation** (co-state $p_t$):
$$\frac{dp_t}{dt} = -\frac{\partial H}{\partial x}(X_t, p_t, \alpha_t)$$

**Optimality Condition**:
$$\alpha_t^* = \arg\max_\alpha H(X_t, p_t, \alpha)$$

**Connection to HJB**: $p_t = \nabla u(t, X_t)$ (adjoint = gradient of value function)

---

## 5. Measure Theory and Weak Convergence

### 5.1 Wasserstein Distance

**Definition**: [^12] For $\mu, \nu \in \mathcal{P}_2(\mathbb{R}^d)$:
$$W_2(\mu, \nu) = \inf_{\pi \in \Pi(\mu,\nu)} \left(\int |x-y|^2 \pi(dx,dy)\right)^{1/2}$$

where $\Pi(\mu,\nu)$ are couplings (joint measures with marginals $\mu, \nu$).

**Metric Properties**:
- Triangle inequality: $W_2(\mu, \nu) \leq W_2(\mu, \rho) + W_2(\rho, \nu)$
- Metrizes weak convergence on $\mathcal{P}_2$

### 5.2 Empirical Measures

**Definition**: For i.i.d. samples $X_1, \ldots, X_N \sim \mu$:
$$\mu^N = \frac{1}{N}\sum_{i=1}^N \delta_{X_i}$$

**Law of Large Numbers**:
$$W_2(\mu^N, \mu) \xrightarrow{a.s.} 0 \quad \text{as } N \to \infty$$

**Rate of Convergence**:
$$\mathbb{E}[W_2^2(\mu^N, \mu)] = \mathcal{O}(1/N)$$

### 5.3 Compactness in Measure Spaces

**Prokhorov's Theorem**: [^13] A family $\{\mu_\alpha\} \subset \mathcal{P}(\mathbb{R}^d)$ is relatively compact in weak topology iff:
1. **Tightness**: $\forall \epsilon > 0, \exists K$ compact: $\sup_\alpha \mu_\alpha(K^c) < \epsilon$

**Application to MFG**: Tightness ensures existence of limit measures as $N \to \infty$.

---

## 6. Viscosity Solutions

### 6.1 Motivation

**Problem**: Classical solutions to HJB may not exist (discontinuities, lack of regularity).

**Solution**: Weaken the notion of solution to allow for non-smooth functions.[^14]

### 6.2 Definition

**Subsolution**: $u$ is a viscosity subsolution if for any $\phi \in C^2$ with $u - \phi$ having a local maximum at $x_0$:
$$-\frac{\partial \phi}{\partial t}(x_0) + H(x_0, \nabla \phi(x_0), D^2\phi(x_0)) \leq 0$$

**Supersolution**: $u$ is a viscosity supersolution if for any $\phi \in C^2$ with $u - \phi$ having a local minimum at $x_0$:
$$-\frac{\partial \phi}{\partial t}(x_0) + H(x_0, \nabla \phi(x_0), D^2\phi(x_0)) \geq 0$$

**Viscosity Solution**: Both subsolution and supersolution.

### 6.3 Comparison Principle

**Theorem**: [^15] If $H$ satisfies proper monotonicity conditions, then:
- **Uniqueness**: Viscosity solution is unique
- **Comparison**: If $u$ is subsolution, $v$ is supersolution, and $u(T,\cdot) \leq v(T,\cdot)$, then $u \leq v$

**Application**: Guarantees well-posedness of HJB equation even without classical regularity.

---

## 7. Numerical Analysis Foundations

### 7.1 Finite Difference Methods

**Central Difference** (2nd order):
$$\frac{\partial u}{\partial x}(x_i) \approx \frac{u_{i+1} - u_{i-1}}{2h} + \mathcal{O}(h^2)$$

**Upwind Scheme** (1st order, stable for transport):
$$\frac{\partial u}{\partial x}(x_i) \approx \begin{cases}
\frac{u_i - u_{i-1}}{h} & \text{if } a > 0 \\
\frac{u_{i+1} - u_i}{h} & \text{if } a < 0
\end{cases}$$

**Laplacian** (2nd order):
$$\Delta u(x_i) \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} + \mathcal{O}(h^2)$$

### 7.2 Stability Analysis

**Von Neumann Stability**: For scheme $u^{n+1}_j = G u^n_j$, stable if:
$$|\lambda(G)| \leq 1 \quad \forall \text{ eigenvalues } \lambda$$

**CFL Condition**: For explicit time-stepping with advection:
$$\frac{a \Delta t}{\Delta x} \leq 1$$

### 7.3 Convergence Theorems

**Lax Equivalence Theorem**: [^16] For consistent, linear scheme:
$$\text{Convergence} \Longleftrightarrow \text{Stability}$$

**Error Estimates**: For $p$-th order method with step size $h$:
$$\|u^h - u\|_{L^\infty} \leq C h^p$$

where $C$ depends on derivatives of $u$.

### 7.4 Monotone Schemes

**Definition**: Scheme is monotone if:
$$F(u_1, \ldots, u_k) \text{ is non-decreasing in each argument}$$

**Theorem (Crandall-Lions)**: [^17] Monotone, consistent scheme converges to viscosity solution.

**Application**: Guarantees convergence for HJB even with non-smooth solutions.

---

## References

[^1]: Rudin, W. (1991). *Functional Analysis* (2nd ed.). McGraw-Hill. Chapters 1-2 on normed spaces and topologies.

[^2]: Brezis, H. (2011). *Functional Analysis, Sobolev Spaces and Partial Differential Equations*. Springer. Chapter 4 on $L^p$ spaces.

[^3]: Adams, R. A., & Fournier, J. J. F. (2003). *Sobolev Spaces* (2nd ed.). Academic Press. Chapters 2-3 on Sobolev space theory.

[^4]: Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures* (2nd ed.). Birkhäuser. Chapter 5 on spaces of measures.

[^5]: Øksendal, B. (2003). *Stochastic Differential Equations* (6th ed.). Springer. Chapter 2 on Brownian motion.

[^6]: Itô's formula: Øksendal (2003), Chapter 4, Theorem 4.1.2 (Itô's formula for scalar case) and Theorem 4.2.4 (vector case).

[^7]: SDE existence and uniqueness: Øksendal (2003), Chapter 5, Theorem 5.2.1.

[^8]: Fleming, W. H., & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions* (2nd ed.). Springer. Chapter III on Hamilton-Jacobi-Bellman equations.

[^9]: Fokker-Planck equation derivation: Risken, H. (1996). *The Fokker-Planck Equation* (2nd ed.). Springer. Chapter 4.

[^10]: Dynamic programming principle: Bertsekas, D. P. (2000). *Dynamic Programming and Optimal Control* (Vol. I, 2nd ed.). Athena Scientific. Chapter 1.

[^11]: Verification theorem: Fleming & Soner (2006), Chapter III, Theorem 4.1.

[^12]: Wasserstein distance: Villani, C. (2003). *Topics in Optimal Transportation*. American Mathematical Society. Chapter 6.

[^13]: Prokhorov's theorem: Billingsley, P. (1999). *Convergence of Probability Measures* (2nd ed.). Wiley. Theorem 5.1.

[^14]: Viscosity solutions: Crandall, M. G., Ishii, H., & Lions, P.-L. (1992). "User's guide to viscosity solutions of second order partial differential equations." *Bulletin of the American Mathematical Society*, 27(1), 1-67.

[^15]: Comparison principle: Crandall, Ishii, & Lions (1992), Theorem 3.3 (comparison for HJB equations).

[^16]: Lax equivalence theorem: Strikwerda, J. C. (2004). *Finite Difference Schemes and Partial Differential Equations* (2nd ed.). SIAM. Theorem 1.4.1.

[^17]: Crandall-Lions theorem: Barles, G., & Souganidis, P. E. (1991). "Convergence of approximation schemes for fully nonlinear second order equations." *Asymptotic Analysis*, 4(3), 271-283.

### Additional Classical References

**Functional Analysis**:
- Rudin, W. (1991). *Functional Analysis* (2nd ed.). McGraw-Hill.
- Brezis, H. (2011). *Functional Analysis, Sobolev Spaces and Partial Differential Equations*. Springer.

**Stochastic Analysis**:
- Øksendal, B. (2003). *Stochastic Differential Equations* (6th ed.). Springer.
- Karatzas, I., & Shreve, S. E. (1991). *Brownian Motion and Stochastic Calculus* (2nd ed.). Springer.

**PDE Theory**:
- Evans, L. C. (2010). *Partial Differential Equations* (2nd ed.). American Mathematical Society.
- Gilbarg, D., & Trudinger, N. S. (2001). *Elliptic Partial Differential Equations of Second Order*. Springer.

**Optimal Control**:
- Fleming, W. H., & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions* (2nd ed.). Springer.
- Bertsekas, D. P. (2000). *Dynamic Programming and Optimal Control* (Vols. I-II, 2nd ed.). Athena Scientific.

**Viscosity Solutions**:
- Crandall, M. G., Ishii, H., & Lions, P.-L. (1992). "User's guide to viscosity solutions of second order partial differential equations." *Bulletin of the American Mathematical Society*, 27(1), 1-67.
- Barles, G. (2013). *An Introduction to the Theory of Viscosity Solutions for First-order Hamilton-Jacobi Equations and Applications*. Springer.

**Measure Theory and Optimal Transport**:
- Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures* (2nd ed.). Birkhäuser.
- Villani, C. (2003). *Topics in Optimal Transportation*. American Mathematical Society.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.

**Numerical Analysis**:
- Strikwerda, J. C. (2004). *Finite Difference Schemes and Partial Differential Equations* (2nd ed.). SIAM.
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.

---

**Document Status**: Complete mathematical foundations
**Usage**: Reference for all MFG_PDE theory and implementation
**Notation Standards**: See `NOTATION_STANDARDS.md`
**Implementation**: Mathematical foundations for all numerical solvers in `mfg_pde.alg.numerical`
