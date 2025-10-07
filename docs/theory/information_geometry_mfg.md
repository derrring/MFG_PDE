# Information Geometry Meets Mean Field Games: Mathematical Foundations

**Document Type**: Theoretical Foundation
**Created**: October 8, 2025
**Status**: Planning Document for Phase 4.6 Implementation
**Related**: `STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` Phase 4.6

---

## Executive Summary

This document establishes the mathematical foundations for integrating **information geometry** with **mean field games (MFGs)**. The intersection of these fields provides a powerful geometric framework for understanding and solving large-scale strategic interaction problems. By treating the space of probability distributions as a statistical manifold equipped with Riemannian metrics and divergence functions, we gain access to:

- **Structure-preserving numerical methods** that respect the geometry of the measure space
- **Natural gradient flows** with provably better convergence properties
- **Regularization techniques** via entropic penalties and KL divergence
- **Geometric optimization** on probability manifolds

The framework bridges optimal transport theory, differential geometry, and game theory, offering both theoretical insights and computational advantages.

---

## Table of Contents

1. [The Space of Probability Measures](#1-the-space-of-probability-measures)
2. [Fisher-Rao Metric and Information Geometry](#2-fisher-rao-metric-and-information-geometry)
3. [Wasserstein Geometry and Optimal Transport](#3-wasserstein-geometry-and-optimal-transport)
4. [Connections Between Fisher-Rao and Wasserstein](#4-connections-between-fisher-rao-and-wasserstein)
5. [Divergence Functions and Bregman Geometry](#5-divergence-functions-and-bregman-geometry)
6. [Gradient Flows on Statistical Manifolds](#6-gradient-flows-on-statistical-manifolds)
7. [Natural Gradient Descent](#7-natural-gradient-descent)
8. [KL-Regularized Mean Field Games](#8-kl-regularized-mean-field-games)
9. [Schrödinger Bridges and Entropic Regularization](#9-schrödinger-bridges-and-entropic-regularization)
10. [Mirror Descent on Measure Spaces](#10-mirror-descent-on-measure-spaces)
11. [Information-Geometric Learning in MFG](#11-information-geometric-learning-in-mfg)
12. [Computational Methods](#12-computational-methods)

---

## 1. The Space of Probability Measures

### 1.1 Basic Definitions

**Space of Probability Measures**: $\mathcal{P}(\mathbb{R}^d)$
- Set of all Borel probability measures on $\mathbb{R}^d$
- Elements: $\mu, \nu \in \mathcal{P}(\mathbb{R}^d)$
- Properties: $\mu(\mathbb{R}^d) = 1$, $\mu(A) \geq 0$ for all Borel sets $A$

**Absolutely Continuous Measures**: $\mathcal{P}_{ac}(\mathbb{R}^d)$
- Measures with density: $\mu(dx) = m(x) dx$ where $m \in L^1(\mathbb{R}^d)$
- Positivity: $m(x) > 0$ a.e.
- Normalization: $\int_{\mathbb{R}^d} m(x) dx = 1$

**Finite Second Moment**: $\mathcal{P}_2(\mathbb{R}^d)$
$$\mathcal{P}_2(\mathbb{R}^d) = \left\{\mu \in \mathcal{P}(\mathbb{R}^d) : \int_{\mathbb{R}^d} |x|^2 \, \mu(dx) < \infty\right\}$$

This is the natural space for Wasserstein geometry.

### 1.2 Tangent Space to $\mathcal{P}(\mathbb{R}^d)$

**Formal Tangent Space**: For $\mu \in \mathcal{P}_{ac}(\mathbb{R}^d)$ with density $m$:

**Otto Calculus Perspective** (Wasserstein geometry):
$$T_\mu \mathcal{P}_2(\mathbb{R}^d) = \{\xi = -\text{div}(m \nabla \phi) : \phi \in C^\infty_c(\mathbb{R}^d)\}$$

Tangent vectors are continuity equations: $\partial_t m = \xi$.

**Fisher-Rao Perspective**:
$$T_\mu \mathcal{P}_{ac}(\mathbb{R}^d) = \left\{\xi : \int_{\mathbb{R}^d} \xi(x) \, dx = 0, \, \int_{\mathbb{R}^d} \frac{\xi^2(x)}{m(x)} \, dx < \infty\right\}$$

Tangent vectors are functions with zero mean and finite Fisher norm.

### 1.3 Measure-Valued Curves

**Curve in Measure Space**: $\mu : [0,T] \to \mathcal{P}(\mathbb{R}^d)$
- Continuous in weak topology: $\int \varphi \, d\mu_t$ is continuous for all $\varphi \in C_b(\mathbb{R}^d)$

**Continuity Equation**: If $\mu_t$ has density $m_t$ and velocity field $v_t$:
$$\frac{\partial m_t}{\partial t} + \text{div}(m_t v_t) = 0$$

**Action Functional** (Benamou-Brenier):
$$\mathcal{A}[\mu_{0:T}] = \int_0^T \int_{\mathbb{R}^d} \frac{|v_t(x)|^2}{2} m_t(x) \, dx \, dt$$

---

## 2. Fisher-Rao Metric and Information Geometry

### 2.1 Fisher Information Metric

**Definition**: For a parametric family $\{p_\theta : \theta \in \Theta \subseteq \mathbb{R}^k\}$:
$$g_{ij}(\theta) = \mathbb{E}_{p_\theta}\left[\frac{\partial \log p_\theta}{\partial \theta^i} \frac{\partial \log p_\theta}{\partial \theta^j}\right] = \int_{\mathbb{R}^d} \frac{\partial_i p_\theta \cdot \partial_j p_\theta}{p_\theta} \, dx$$

This defines a Riemannian metric on the parameter space $\Theta$.

**Fisher-Rao Metric on $\mathcal{P}_{ac}(\mathbb{R}^d)$**: For tangent vectors $\xi, \eta \in T_\mu \mathcal{P}_{ac}$:
$$g_{FR}(\mu)[\xi, \eta] = \int_{\mathbb{R}^d} \frac{\xi(x) \eta(x)}{m(x)} \, dx$$

where $\mu = m \, dx$.

**Score Function Representation**: If $\xi = a \cdot m$ and $\eta = b \cdot m$ for functions $a, b$:
$$g_{FR}(\mu)[\xi, \eta] = \int_{\mathbb{R}^d} a(x) b(x) m(x) \, dx$$

This is the $L^2(m)$ inner product of score-like functions.

### 2.2 Properties of Fisher-Rao Metric

**Invariance**: The Fisher-Rao metric is invariant under sufficient statistics transformations.

**Geodesics**: Geodesics in Fisher-Rao metric follow:
$$\frac{d^2 m}{dt^2} = \left(\frac{dm/dt}{m}\right)^2 m - \frac{(dm/dt)^2}{m}$$

**Distance Formula**: Fisher-Rao distance between $\mu_0 = m_0 dx$ and $\mu_1 = m_1 dx$:
$$d_{FR}(\mu_0, \mu_1) = 2 \arccos\left(\int_{\mathbb{R}^d} \sqrt{m_0(x) m_1(x)} \, dx\right)$$

This is also called the **Hellinger distance** (up to a constant).

### 2.3 Exponential and Mixture Families

**Exponential Family**:
$$p_\theta(x) = \exp\left(\sum_{i=1}^k \theta^i T_i(x) - \psi(\theta)\right) h(x)$$

where $T_i$ are sufficient statistics and $\psi(\theta) = \log \int \exp(\theta \cdot T(x)) h(x) dx$ is the log-partition function.

**Fisher Information for Exponential Families**:
$$g_{ij}(\theta) = \frac{\partial^2 \psi}{\partial \theta^i \partial \theta^j}(\theta) = \text{Cov}[T_i, T_j]$$

**Mixture Families**: Convex combinations
$$p_\eta = \sum_{i=1}^k \eta^i q_i, \quad \eta \in \Delta_{k-1}$$

where $\Delta_{k-1}$ is the probability simplex.

### 2.4 Affine Connections

**$e$-Connection** (Exponential): $\nabla^{(e)}$
- Flat for exponential families
- Dual to $m$-connection

**$m$-Connection** (Mixture): $\nabla^{(m)}$
- Flat for mixture families
- Dual to $e$-connection

**Duality**:
$$g(\nabla^{(e)}_X Y, Z) + g(Y, \nabla^{(m)}_X Z) = X g(Y, Z)$$

This dualistic structure is fundamental to information geometry.

---

## 3. Wasserstein Geometry and Optimal Transport

### 3.1 Optimal Transport Problem

**Monge Problem**: Find $T : \mathbb{R}^d \to \mathbb{R}^d$ such that:
$$\min_{T} \int_{\mathbb{R}^d} c(x, T(x)) \, \mu_0(dx)$$
subject to $T_\# \mu_0 = \mu_1$ (pushforward constraint).

**Kantorovich Relaxation**: Find coupling $\pi \in \Pi(\mu_0, \mu_1)$:
$$W_c(\mu_0, \mu_1) = \min_{\pi \in \Pi(\mu_0, \mu_1)} \int_{\mathbb{R}^d \times \mathbb{R}^d} c(x, y) \, \pi(dx, dy)$$

where $\Pi(\mu_0, \mu_1) = \{\pi : \pi(\cdot, \mathbb{R}^d) = \mu_0, \pi(\mathbb{R}^d, \cdot) = \mu_1\}$.

**Wasserstein-2 Distance**: For $c(x,y) = |x-y|^2/2$:
$$W_2^2(\mu_0, \mu_1) = \min_{\pi \in \Pi(\mu_0, \mu_1)} \int |x - y|^2 \, \pi(dx, dy)$$

### 3.2 Benamou-Brenier Formula

**Dynamic Formulation**: Wasserstein distance via curves:
$$W_2^2(\mu_0, \mu_1) = \inf_{\substack{(\mu_t, v_t) \\ \mu_0 = \text{given}, \mu_1 = \text{given}}} \int_0^1 \int_{\mathbb{R}^d} |v_t(x)|^2 m_t(x) \, dx \, dt$$

subject to continuity equation:
$$\frac{\partial m_t}{\partial t} + \text{div}(m_t v_t) = 0$$

**Interpretation**: Wasserstein distance is the minimal "kinetic energy" to transport $\mu_0$ to $\mu_1$.

### 3.3 Wasserstein Metric as Riemannian Geometry

**Otto Metric** on $\mathcal{P}_2(\mathbb{R}^d)$: For tangent vectors $\xi = -\text{div}(m \nabla \phi_\xi)$, $\eta = -\text{div}(m \nabla \phi_\eta)$:
$$g_W(\mu)[\xi, \eta] = \int_{\mathbb{R}^d} \nabla \phi_\xi(x) \cdot \nabla \phi_\eta(x) \, m(x) \, dx$$

**Formal Riemannian Structure**: $(\mathcal{P}_2(\mathbb{R}^d), g_W)$ is a formal Riemannian manifold.

**Geodesic Equation**: Geodesics satisfy:
$$\frac{\partial v_t}{\partial t} + (v_t \cdot \nabla) v_t + \nabla p_t = 0$$
$$\frac{\partial m_t}{\partial t} + \text{div}(m_t v_t) = 0$$

This is the **inviscid Burgers equation** on measure space.

### 3.4 Displacement Convexity

**$\lambda$-Convexity**: Functional $\mathcal{F} : \mathcal{P}_2 \to \mathbb{R}$ is $\lambda$-convex if along constant-speed geodesics $\mu_t$:
$$\mathcal{F}(\mu_t) \leq (1-t) \mathcal{F}(\mu_0) + t \mathcal{F}(\mu_1) - \frac{\lambda}{2} t(1-t) W_2^2(\mu_0, \mu_1)$$

**Examples**:
- **Entropy**: $S[m] = \int m \log m \, dx$ is displacement convex on $\mathbb{R}^d$ ($\lambda = 0$)
- **Internal Energy**: $\mathcal{E}[m] = \int V(x) m(x) \, dx$ is $\lambda$-convex if $\nabla^2 V \geq \lambda I$

---

## 4. Connections Between Fisher-Rao and Wasserstein

### 4.1 Otto Calculus vs Fisher-Rao

**Wasserstein Tangent Vectors**: Velocity fields $v$ such that $\xi = -\text{div}(m v)$
- Represent mass transport
- Inner product: $\langle v, w \rangle_W = \int v \cdot w \, m \, dx$

**Fisher-Rao Tangent Vectors**: Functions $\xi$ with $\int \xi = 0$
- Represent infinitesimal changes in density
- Inner product: $\langle \xi, \eta \rangle_{FR} = \int \xi \eta / m \, dx$

**Relationship via Score Function**: If $\xi = -\text{div}(m v)$:
$$\text{Score}(\xi) = -\text{div}(v) - v \cdot \nabla \log m$$

### 4.2 Hessian Metrics

**Wasserstein as Hessian Metric**: The Wasserstein metric is the Hessian of the entropy functional:
$$g_W(\mu)[\xi, \eta] = \text{Hess}_S(\mu)[\xi, \eta]$$

where $S[m] = \int m \log m \, dx$ is the Boltzmann entropy.

**Fisher-Rao as Induced Metric**: Fisher-Rao can be obtained from Wasserstein via:
$$g_{FR} = g_W + \text{(additional curvature terms)}$$

### 4.3 Lott's Relationship

**Lott (2008) Connection**: For probability measures on a Riemannian manifold $(M, g)$:
$$g_{W_2}^{(M)}(\mu) = g_{FR}^{(M \times [0,\infty))}(\tilde{\mu})$$

where $\tilde{\mu}$ is a lifted measure on $M \times [0,\infty)$.

**Intuition**: Wasserstein metric on base manifold equals Fisher-Rao metric on extended space.

### 4.4 Interpolation Between Metrics

**$\alpha$-Geometry**: One-parameter family interpolating Fisher-Rao and Wasserstein:
$$g_\alpha(\mu)[\xi, \eta] = (1 - \alpha) g_{FR}(\mu)[\xi, \eta] + \alpha g_W(\mu)[\xi, \eta]$$

for $\alpha \in [0, 1]$.

---

## 5. Divergence Functions and Bregman Geometry

### 5.1 Kullback-Leibler Divergence

**Definition**: For $\mu, \nu \in \mathcal{P}_{ac}(\mathbb{R}^d)$ with densities $m, n$:
$$D_{KL}(\mu \| \nu) = \int_{\mathbb{R}^d} m(x) \log \frac{m(x)}{n(x)} \, dx$$

if $\mu \ll \nu$, otherwise $D_{KL}(\mu \| \nu) = +\infty$.

**Properties**:
- **Non-negativity**: $D_{KL}(\mu \| \nu) \geq 0$ with equality iff $\mu = \nu$ (Gibbs' inequality)
- **Asymmetry**: $D_{KL}(\mu \| \nu) \neq D_{KL}(\nu \| \mu)$ in general
- **Convexity**: $D_{KL}(\cdot \| \nu)$ is convex

**Pythagorean Theorem**: For $\mu, \nu, \sigma$ with $\nu$ in the exponential family closure of $\sigma$:
$$D_{KL}(\mu \| \sigma) = D_{KL}(\mu \| \nu) + D_{KL}(\nu \| \sigma)$$

### 5.2 Bregman Divergence

**Convex Function**: Let $\phi : \mathcal{X} \to \mathbb{R}$ be strictly convex and differentiable.

**Bregman Divergence**:
$$D_\phi(x, y) = \phi(x) - \phi(y) - \langle \nabla \phi(y), x - y \rangle$$

**KL as Bregman**: With $\phi[m] = \int m \log m \, dx$ (negative entropy):
$$D_\phi(m, n) = \int m \log \frac{m}{n} - (m - n) \, dx = D_{KL}(m \| n)$$

**Induced Geometry**: Bregman divergence induces:
- Riemannian metric: $g_{ij} = \partial_i \partial_j \phi$
- Dual coordinates: $\theta^i$ (expectation parameters) and $\eta_i$ (natural parameters)
- Geodesics: Bregman projections

### 5.3 $\alpha$-Divergences

**Amari $\alpha$-Divergence**: Generalization of KL divergence:
$$D_\alpha(\mu \| \nu) = \begin{cases}
\frac{4}{1 - \alpha^2} \left(1 - \int \left(\frac{m}{n}\right)^{(1+\alpha)/2} n \, dx\right) & \alpha \neq \pm 1 \\
D_{KL}(\mu \| \nu) & \alpha = 1 \\
D_{KL}(\nu \| \mu) & \alpha = -1
\end{cases}$$

**Symmetry**: $D_\alpha(\mu \| \nu) = D_{-\alpha}(\nu \| \mu)$

**Special Cases**:
- $\alpha = 0$: Squared Hellinger distance
- $\alpha = 1$: Forward KL divergence
- $\alpha = -1$: Reverse KL divergence

### 5.4 Csiszár $f$-Divergences

**General Form**: For convex $f : (0, \infty) \to \mathbb{R}$ with $f(1) = 0$:
$$D_f(\mu \| \nu) = \int_{\mathbb{R}^d} n(x) f\left(\frac{m(x)}{n(x)}\right) dx$$

**Examples**:
- $f(t) = t \log t$: KL divergence
- $f(t) = (t - 1)^2$: Pearson $\chi^2$ divergence
- $f(t) = |t - 1|$: Total variation distance
- $f(t) = (\sqrt{t} - 1)^2$: Hellinger distance

---

## 6. Gradient Flows on Statistical Manifolds

### 6.1 Wasserstein Gradient Flow

**Energy Functional**: $\mathcal{E} : \mathcal{P}_2(\mathbb{R}^d) \to \mathbb{R}$

**Gradient Flow**: Curve $\mu_t$ satisfying:
$$\frac{\partial m_t}{\partial t} = -\nabla_{W_2} \mathcal{E}[m_t]$$

**Characterization**: $v_t = -\nabla \frac{\delta \mathcal{E}}{\delta m}[m_t]$ where:
$$\frac{\delta \mathcal{E}}{\delta m}[m] = \lim_{\epsilon \to 0} \frac{\mathcal{E}[m + \epsilon \delta_x] - \mathcal{E}[m]}{\epsilon}$$

**Example - Fokker-Planck**: For $\mathcal{E}[m] = \int V(x) m(x) dx + \epsilon \int m \log m \, dx$:
$$\frac{\partial m}{\partial t} = \text{div}(m \nabla V) + \epsilon \Delta m$$

This is the **Wasserstein gradient flow** of the free energy.

### 6.2 JKO Scheme

**Implicit Euler** on Wasserstein space (Jordan-Kinderlehrer-Otto, 1998):
$$m^{k+1} = \arg\min_{m} \left\{\mathcal{E}[m] + \frac{1}{2\tau} W_2^2(m, m^k)\right\}$$

**Properties**:
- **Energy dissipation**: $\mathcal{E}[m^{k+1}] \leq \mathcal{E}[m^k]$
- **Unconditional stability**: Works for any $\tau > 0$
- **Variational interpretation**: Proximal point method on $(\mathcal{P}_2, W_2)$

**Convergence**: As $\tau \to 0$, $m^k_\tau(t)$ converges to gradient flow $m_t$.

### 6.3 Fisher-Rao Gradient Flow

**Gradient Flow in Fisher-Rao Metric**:
$$\frac{\partial m}{\partial t} = -m \nabla_{FR} \mathcal{E}[m]$$

where $\nabla_{FR} \mathcal{E} = m \frac{\delta \mathcal{E}}{\delta m}$ (score-based gradient).

**Replicator Equation**: For $\mathcal{E}[m] = \int V(x) m(x) dx$:
$$\frac{\partial m}{\partial t} = m \left(\bar{V} - V(x)\right), \quad \bar{V} = \int V(y) m(y) dy$$

This is the **replicator dynamics** in evolutionary game theory.

### 6.4 Dual Gradient Flows

**Primal Flow** (Wasserstein): $\frac{\partial m}{\partial t} = -\nabla_{W_2} \mathcal{E}[m]$

**Dual Flow** (Fisher-Rao): $\frac{\partial u}{\partial t} = -\nabla_{FR}^* \mathcal{E}^*[u]$

where $\mathcal{E}^*$ is the Legendre-Fenchel conjugate.

**Coupling via Optimality**: $(m, u)$ satisfies:
$$\nabla u = -\frac{\delta \mathcal{E}}{\delta m}[m], \quad m = \nabla \mathcal{E}^*[u]$$

---

## 7. Natural Gradient Descent

### 7.1 Steepest Descent in Riemannian Manifolds

**Standard Gradient**: In Euclidean space $\mathbb{R}^n$:
$$\theta^{k+1} = \theta^k - \alpha \nabla_\theta J(\theta^k)$$

**Riemannian Gradient**: On manifold $(M, g)$:
$$\theta^{k+1} = \exp_{\theta^k}(-\alpha \nabla^g_\theta J(\theta^k))$$

where $\nabla^g_\theta J = g^{-1}(\theta) \nabla_\theta J$ and $\exp$ is the exponential map.

**Steepest Descent Direction**: Solve
$$\min_{\delta \theta} J(\theta + \delta \theta) \quad \text{subject to} \quad g(\theta)[\delta \theta, \delta \theta] = \epsilon^2$$

Solution: $\delta \theta = -\epsilon g^{-1}(\theta) \nabla J(\theta)$.

### 7.2 Natural Gradient for Parametric Families

**Fisher Information**: $F(\theta) = \mathbb{E}_{p_\theta}[\nabla \log p_\theta \nabla \log p_\theta^T]$

**Natural Gradient**:
$$\tilde{\nabla}_\theta J(\theta) = F^{-1}(\theta) \nabla_\theta J(\theta)$$

**Update Rule**:
$$\theta^{k+1} = \theta^k - \alpha F^{-1}(\theta^k) \nabla_\theta J(\theta^k)$$

**Geometric Interpretation**: Steepest descent in Fisher-Rao metric, not Euclidean metric.

### 7.3 Natural Gradient on Measure Space

**Measure-Valued Natural Gradient**: For functional $\mathcal{J} : \mathcal{P}_{ac} \to \mathbb{R}$:
$$\tilde{\nabla}_{FR} \mathcal{J}[m] = m \frac{\delta \mathcal{J}}{\delta m}[m]$$

**Update**:
$$m^{k+1}(x) = m^k(x) \exp\left(-\alpha \frac{\delta \mathcal{J}}{\delta m}[m^k](x)\right) / Z^{k+1}$$

where $Z^{k+1} = \int m^k(x) \exp(-\alpha \frac{\delta \mathcal{J}}{\delta m}[m^k](x)) dx$ is the normalization.

**Automatic Properties**:
- Preserves positivity: $m^{k+1} > 0$ if $m^k > 0$
- Preserves mass: $\int m^{k+1} = 1$
- No projection needed

### 7.4 Convergence Advantages

**Condition Number**: Fisher information preconditioning reduces condition number:
$$\kappa(F^{-1} H) \ll \kappa(H)$$

where $H = \nabla^2 J$ is the Hessian.

**Covariant Convergence**: Natural gradient is invariant under reparametrization:
$$\tilde{\nabla}_\theta J = \tilde{\nabla}_{\phi} J \cdot \frac{\partial \phi}{\partial \theta}$$

This ensures consistent convergence regardless of parametrization choice.

---

## 8. KL-Regularized Mean Field Games

### 8.1 Entropic Regularization

**Regularized Cost Functional**:
$$J_\lambda(\alpha) = \mathbb{E}\left[\int_0^T L(X_t, \alpha_t, m_t) dt + g(X_T, m_T)\right] + \lambda D_{KL}(m \| m_{\text{ref}})$$

where:
- $m$ is the distribution induced by control $\alpha$
- $m_{\text{ref}}$ is a reference/prior distribution
- $\lambda > 0$ is regularization parameter

**Interpretation**:
- **Robust Control**: Stay close to nominal distribution
- **Risk-Sensitive**: Penalize deviations from safe distribution
- **Smooth Equilibria**: Regularization smooths Nash equilibria

### 8.2 Modified HJB Equation

**Standard HJB**: $\frac{\partial u}{\partial t} + H(x, \nabla u, m) + \frac{\sigma^2}{2} \Delta u = 0$

**KL-Regularized HJB**:
$$\frac{\partial u}{\partial t} + H(x, \nabla u, m) + \frac{\sigma^2}{2} \Delta u + \lambda \log \frac{m(x)}{m_{\text{ref}}(x)} = 0$$

**Entropy Penalty Term**: $\lambda \log(m/m_{\text{ref}})$ acts as additional running cost.

### 8.3 Modified Fokker-Planck

**Regularized FP**: The dual equation becomes:
$$\frac{\partial m}{\partial t} - \text{div}(m \nabla_p H(x, \nabla u, m)) - \frac{\sigma^2}{2} \Delta m - \lambda \Delta \left(\log \frac{m}{m_{\text{ref}}}\right) = 0$$

**Gradient Flow Interpretation**: This is the Wasserstein gradient flow of:
$$\mathcal{E}[m, u] = \int H(x, \nabla u, m) m \, dx + \lambda D_{KL}(m \| m_{\text{ref}})$$

### 8.4 Fixed Point with Regularization

**Regularized Fixed Point**: Seek $(u, m)$ such that:
$$\Phi_\lambda(m) = m$$

where $\Phi_\lambda$ includes KL penalty in the forward map.

**Contraction Property**: For sufficiently large $\lambda$:
$$\|\Phi_\lambda(m_1) - \Phi_\lambda(m_2)\|_{TV} \leq \gamma \|m_1 - m_2\|_{TV}, \quad \gamma < 1$$

Regularization induces contraction, guaranteeing unique equilibrium.

---

## 9. Schrödinger Bridges and Entropic Regularization

### 9.1 Schrödinger Bridge Problem

**Classical Optimal Transport**: $\min_{\pi \in \Pi(\mu_0, \mu_1)} \int c(x,y) \pi(dx, dy)$

**Entropic Regularization**:
$$\min_{\pi \in \Pi(\mu_0, \mu_1)} \left\{\int c(x,y) \pi(dx, dy) + \epsilon D_{KL}(\pi \| \mu_0 \otimes \mu_1)\right\}$$

**Schrödinger Bridge** (Dynamic Version):
$$\min_{\substack{m_{0:T} \\ m_0 = \mu_0, m_T = \mu_1}} \left\{\frac{1}{2} \int_0^T \int |v_t|^2 m_t \, dx \, dt + \epsilon D_{KL}(m_{0:T} \| m_{\text{ref}, 0:T})\right\}$$

subject to $\frac{\partial m_t}{\partial t} + \text{div}(m_t v_t) = 0$.

### 9.2 Forward-Backward System

**Schrödinger Bridge Solution**: $(m_t, \psi_t, \phi_t)$ satisfying:

**Forward SDE**:
$$\frac{\partial m}{\partial t} = -\text{div}(m \nabla \psi) + \epsilon \Delta m$$

**Backward HJB**:
$$-\frac{\partial \psi}{\partial t} + \frac{|\nabla \psi|^2}{2} - \epsilon \Delta \psi = 0$$

**Dual Backward**:
$$\frac{\partial \phi}{\partial t} + \frac{|\nabla \phi|^2}{2} - \epsilon \Delta \phi = 0$$

**Boundary Conditions**: $m_0 = \mu_0$, $m_T = \mu_1$.

### 9.3 Connection to MFG

**MFG as Schrödinger Bridge**: Consider MFG with entropy regularization:
$$\min \int_0^T \int L(x, \alpha, m) m \, dx \, dt + \epsilon \int m_T \log m_T \, dx$$

This is equivalent to a Schrödinger bridge with terminal entropy penalty.

**Sinkhorn Algorithm**: Iterative proportional fitting converges to Schrödinger bridge:
$$m^{k+1}_t(x) \propto m^k_t(x) \exp\left(-\alpha \frac{\delta \mathcal{E}}{\delta m}[m^k_t](x)\right)$$

### 9.4 Applications

**Path Planning**: Find optimal trajectory distribution between two endpoint distributions.

**Generative Modeling**: Schrödinger bridges for flow-based generative models.

**Control**: Distributional control with soft constraints on initial/final states.

---

## 10. Mirror Descent on Measure Spaces

### 10.1 Bregman Divergence Review

**Convex Potential**: $\phi : \mathcal{P}_{ac} \to \mathbb{R}$

**Bregman Divergence**:
$$D_\phi(m, n) = \phi[m] - \phi[n] - \left\langle \frac{\delta \phi}{\delta n}, m - n \right\rangle$$

**Negative Entropy**: $\phi[m] = \int m \log m \, dx$ gives:
$$D_\phi(m, n) = D_{KL}(m \| n)$$

### 10.2 Mirror Descent Algorithm

**Primal Update** (Gradient Descent):
$$m^{k+1} = \arg\min_m \left\{\langle \nabla \mathcal{E}[m^k], m \rangle + \frac{1}{2\alpha} \|m - m^k\|^2\right\}$$

**Mirror Descent** (with Bregman divergence):
$$m^{k+1} = \arg\min_m \left\{\langle \nabla \mathcal{E}[m^k], m \rangle + \frac{1}{\alpha} D_\phi(m, m^k)\right\}$$

**Closed Form** (for $\phi = $ negative entropy):
$$m^{k+1}(x) = m^k(x) \exp\left(-\alpha \frac{\delta \mathcal{E}}{\delta m}[m^k](x)\right) / Z$$

### 10.3 Convergence Analysis

**Strong Convexity**: If $\mathcal{E}$ is $\lambda$-convex in $D_\phi$:
$$\mathcal{E}[m] \geq \mathcal{E}[n] + \left\langle \frac{\delta \mathcal{E}}{\delta n}, m - n \right\rangle + \frac{\lambda}{2} D_\phi(m, n)$$

**Convergence Rate**: For strongly convex $\mathcal{E}$:
$$\mathcal{E}[m^k] - \mathcal{E}[m^*] \leq \frac{D_\phi(m^*, m^0)}{\lambda k}$$

Linear convergence in function value.

### 10.4 Composite Objectives

**Composite Problem**:
$$\min_m \left\{\mathcal{E}[m] + \mathcal{R}[m]\right\}$$

where $\mathcal{E}$ is smooth and $\mathcal{R}$ is potentially non-smooth (e.g., constraints).

**Proximal Mirror Descent**:
$$m^{k+1} = \arg\min_m \left\{\langle \nabla \mathcal{E}[m^k], m \rangle + \mathcal{R}[m] + \frac{1}{\alpha} D_\phi(m, m^k)\right\}$$

---

## 11. Information-Geometric Learning in MFG

### 11.1 Natural Policy Gradient for MFRL

**Policy**: $\pi_\theta(a | s)$ parameterized by $\theta \in \mathbb{R}^p$

**Value Function**: $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t r(s_t, a_t)]$

**Standard Policy Gradient**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)\right]$$

**Fisher Information Matrix**:
$$F(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}} \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T\right]$$

**Natural Policy Gradient**:
$$\tilde{\nabla}_\theta J(\theta) = F^{-1}(\theta) \nabla_\theta J(\theta)$$

### 11.2 Mean Field Natural Gradient

**Mean Field Policy**: $\pi_\theta$ induces distribution $m_\theta$ of agent states.

**Mean Field Fisher Information**:
$$F_{MF}(\theta) = \int \nabla_\theta \log m_\theta(x) \nabla_\theta \log m_\theta(x)^T \, m_\theta(x) \, dx$$

**Update**:
$$\theta^{k+1} = \theta^k - \alpha F_{MF}^{-1}(\theta^k) \nabla_\theta J(\theta^k)$$

**Geometric Interpretation**: Steepest descent in Fisher-Rao metric on space of population distributions.

### 11.3 Trust Region Methods

**Trust Region Policy Optimization (TRPO)**: Constrain KL divergence between policies:
$$\max_\theta J(\theta) \quad \text{subject to} \quad D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta$$

**Linearization**:
$$\max_\theta \nabla_\theta J(\theta_{\text{old}})^T (\theta - \theta_{\text{old}}) \quad \text{s.t.} \quad \frac{1}{2}(\theta - \theta_{\text{old}})^T F(\theta_{\text{old}}) (\theta - \theta_{\text{old}}) \leq \delta$$

**Solution** (conjugate gradient):
$$\theta = \theta_{\text{old}} + \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g, \quad g = \nabla_\theta J(\theta_{\text{old}})$$

### 11.4 Distributional Reinforcement Learning

**Value Distribution**: Learn distribution $Z(s,a)$ of returns, not just expectation.

**Categorical Representation**: $Z(s,a) = \sum_{i=1}^N p_i(s,a) \delta_{z_i}$

**Wasserstein Loss**: Minimize Wasserstein distance between predicted and target distributions:
$$\mathcal{L}_W(\theta) = \mathbb{E}_{s,a,r,s'}\left[W_1(Z_\theta(s,a), \mathcal{T} Z_{\theta^-}(s,a))\right]$$

where $\mathcal{T}$ is the distributional Bellman operator.

---

## 12. Computational Methods

### 12.1 Particle-Based Wasserstein Gradient Flow

**Particle Representation**: $m^k \approx \frac{1}{N} \sum_{i=1}^N \delta_{x_i^k}$

**Wasserstein Gradient Flow**:
$$\frac{dx_i}{dt} = -\nabla_{x_i} \mathcal{E}\left[\frac{1}{N} \sum_j \delta_{x_j}\right]$$

**Discretization** (Forward Euler):
$$x_i^{k+1} = x_i^k - \tau \nabla_{x_i} \mathcal{E}[m^k]$$

**Interaction Energy**: For $\mathcal{E}[m] = \iint K(x,y) m(dx) m(dy)$:
$$\nabla_{x_i} \mathcal{E}[m^k] = \frac{2}{N} \sum_{j=1}^N \nabla_x K(x_i, x_j)$$

### 12.2 JKO Scheme Implementation

**Discrete JKO**:
$$x^{k+1} = \arg\min_{x} \left\{\mathcal{E}\left[\frac{1}{N} \sum_i \delta_{x_i}\right] + \frac{1}{2\tau} \sum_i |x_i - x_i^k|^2\right\}$$

**Gradient**: Using calculus of variations:
$$\frac{\partial}{\partial x_i}\left[\mathcal{E}[m] + \frac{1}{2\tau} \sum_j |x_j - x_j^k|^2\right] = \nabla_{x_i} \mathcal{E}[m] + \frac{1}{\tau}(x_i - x_i^k)$$

**Implicit Update**: Solve for $x^{k+1}$:
$$x_i^{k+1} = x_i^k - \tau \nabla_{x_i} \mathcal{E}[m^{k+1}]$$

Requires iterative solver (Newton or gradient descent).

### 12.3 Sinkhorn Algorithm for Entropic OT

**Entropic OT**: $\min_{\pi \in \Pi(\mu, \nu)} \langle C, \pi \rangle + \epsilon D_{KL}(\pi \| \mu \otimes \nu)$

**Sinkhorn Iterations**: For discrete measures $\mu = \sum_i a_i \delta_{x_i}$, $\nu = \sum_j b_j \delta_{y_j}$:
$$u_i^{k+1} = \frac{a_i}{\sum_j v_j^k K_{ij}}, \quad v_j^{k+1} = \frac{b_j}{\sum_i u_i^{k+1} K_{ij}}$$

where $K_{ij} = \exp(-C_{ij}/\epsilon)$.

**Coupling**: $\pi_{ij} = u_i v_j K_{ij}$

**Convergence**: Exponentially fast for $\epsilon > 0$.

### 12.4 Natural Gradient via Fisher Information

**Empirical Fisher**: For parametric $p_\theta(x) = \frac{1}{N} \sum_i \delta_{X_i^\theta}$:
$$\hat{F}(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(X_i) \nabla_\theta \log p_\theta(X_i)^T$$

**Update** (with damping):
$$\theta^{k+1} = \theta^k - \alpha (\hat{F}(\theta^k) + \lambda I)^{-1} \nabla_\theta J(\theta^k)$$

**Conjugate Gradient**: Avoid matrix inversion via CG for $\hat{F} d = g$:
1. Initialize $d_0 = g$, $r_0 = g$
2. Iterate: $d_{i+1} = r_i + \beta_i d_i$ where $\beta_i = \|r_i\|^2 / \|r_{i-1}\|^2$
3. Update $r_{i+1}$ via Fisher-vector product

### 12.5 Hybrid Discrete-Continuous Methods

**Measure Splitting**: $m = \sum_{i=1}^K w_i m_i$ where $\sum w_i = 1$

**Component Optimization**: Alternate between:
1. **Weight Update**: $w^{k+1} = \arg\min_w \mathcal{E}[\sum w_i m_i^k]$
2. **Shape Update**: $m_i^{k+1} = \arg\min_{m_i} \mathcal{E}[\sum w_i^{k+1} m_i]$

**Wasserstein Mixture Flow**: Each $m_i$ evolves via Wasserstein gradient, weights via Fisher-Rao flow.

---

## References

### Information Geometry Foundations
- Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
- Ay, N., et al. (2017). *Information Geometry*. Springer.

### Optimal Transport & Wasserstein Geometry
- Villani, C. (2003). *Topics in Optimal Transportation*. AMS.
- Santambrogio, F. (2015). *Optimal Transport for Applied Mathematicians*. Birkhäuser.
- Ambrosio, L., et al. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkhäuser.

### Fisher-Rao and Wasserstein Connections
- Otto, F. (2001). "The geometry of dissipative evolution equations: the porous medium equation." *Comm. PDE*.
- Lott, J. (2008). "Some geometric calculations on Wasserstein space." *Comm. Math. Phys*.

### KL Regularization and Schrödinger Bridges
- Léonard, C. (2014). "A survey of the Schrödinger problem and some of its connections with optimal transport." *Discrete Contin. Dyn. Syst. A*.
- Chen, Y., et al. (2021). "Likelihood training of Schrödinger bridge using forward-backward SDEs theory." *ICLR*.

### Natural Gradient Methods
- Amari, S. (1998). "Natural gradient works efficiently in learning." *Neural Computation*.
- Martens, J., & Grosse, R. (2015). "Optimizing neural networks with Kronecker-factored approximate curvature." *ICML*.

### Information Geometry + Mean Field Games
- Chizat, L., & Bach, F. (2018). "On the global convergence of gradient descent for over-parameterized models using optimal transport." *NeurIPS*.
- Rotskoff, G., & Vanden-Eijnden, E. (2022). "Trainability and accuracy of neural networks: An interacting particle system approach." *CPAM*.
- Maoutsa, D., et al. (2020). "Interacting particle solutions of Fokker–Planck equations through gradient–log–density estimation." *Entropy*.

### Mirror Descent and Bregman Geometry
- Beck, A., & Teboulle, M. (2003). "Mirror descent and nonlinear projected subgradient methods for convex optimization." *Operations Research Letters*.
- Bauschke, H., & Borwein, J. (1997). "Legendre functions and the method of random Bregman projections." *J. Convex Analysis*.

### Computational Methods
- Cuturi, M. (2013). "Sinkhorn distances: Lightspeed computation of optimal transport." *NeurIPS*.
- Peyré, G., & Cuturi, M. (2019). "Computational optimal transport." *Foundations and Trends in Machine Learning*.
- Jordan, R., et al. (1998). "The variational formulation of the Fokker–Planck equation." *SIAM J. Math. Analysis*.

---

## Implementation Roadmap

### Phase 4.6.1: Core Information Geometry Tools (3-4 weeks)
**Deliverables**:
- `mfg_pde/information_geometry/metrics.py`: Fisher-Rao and Wasserstein metrics
- `mfg_pde/information_geometry/divergences.py`: KL, Wasserstein, $\alpha$-divergences
- Unit tests with analytical benchmarks

### Phase 4.6.2: Optimization Methods (3-4 weeks)
**Deliverables**:
- `mfg_pde/information_geometry/optimization.py`: Natural gradient descent, mirror descent
- `mfg_pde/information_geometry/flows.py`: JKO scheme, Wasserstein gradient flows
- Particle-based implementations

### Phase 4.6.3: MFG Integration (4-5 weeks)
**Deliverables**:
- `mfg_pde/information_geometry/solvers/kl_regularized_mfg.py`: KL-regularized MFG solver
- `mfg_pde/information_geometry/solvers/natural_gradient_mfg.py`: Natural gradient MFG
- Examples: Robust control, safe learning

### Phase 4.6.4: Advanced Topics (Optional, 3-4 weeks)
**Deliverables**:
- Schrödinger bridge solver
- Information-geometric MFRL
- Geometric integrators

**Total Timeline**: 13-17 weeks (3-4 months)

---

**Document Status**: Mathematical foundation for Phase 4.6 implementation
**Next Steps**: Create module structure `mfg_pde/information_geometry/` and implement core components
**Related**: `STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` Phase 4.6 (to be added)
