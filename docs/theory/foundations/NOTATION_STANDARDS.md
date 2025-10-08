# MFG_PDE Theory Documentation: Notation Standards and Consistency Guide

**Document Type**: Cross-Document Standards
**Created**: October 8, 2025
**Status**: Active Reference Guide
**Purpose**: Ensure mathematical consistency across all theory documentation

---

## 1. Core Notation Standards

### 1.1 Probability and Measure Theory

**Consistent Usage**:
- $\mu, \nu, \pi$: **Probability measures** (general)
- $m, n, \rho$: **Probability densities** (when absolutely continuous)
  - **Primary**: Use $m$ for MFG equilibrium density
  - **Alternative**: Use $\rho$ only in fluid dynamics context
- $\mathbb{P}, \mathbb{Q}$: **Probability measures** on path space $(\Omega, \mathcal{F})$
- $\mathbb{E}, \mathbb{E}^\mathbb{P}$: **Expectation** operator
- $\mathcal{L}(X)$: **Law/distribution** of random variable $X$

**Convention**: When both measure and density appear:
$$\mu(dx) = m(x) \, dx \quad \text{(Radon-Nikodym derivative)}$$

### 1.2 Mean Field Game Components

**Value Function**:
- $u(t, x)$ or $u(t, x, m)$: **MFG value function** (limit)
- $V^{i,N}(t, x, \mu^N)$: **N-player value function** for player $i$
- $U(t, x, \mu)$: **Master equation** solution on $\mathcal{P}_2$

**Empirical Measure**:
- $\mu^N_t = \frac{1}{N} \sum_{i=1}^N \delta_{X_t^i}$: **Empirical measure** (N-player)
- $m_t$: **MFG equilibrium measure/density** (limit)

**Controls**:
- $\alpha, \alpha_t, \alpha(t,x)$: **Control/strategy**
- $\mathcal{A}$: **Admissible control set**

### 1.3 Hamiltonian and Running Cost

**Standard Form**:
$$H(x, p, m) = \sup_{\alpha \in \mathcal{A}} \{-\alpha \cdot p - L(x, \alpha, m)\}$$

where:
- $L(x, \alpha, m)$: **Running cost** (instantaneous)
- $g(x, m)$: **Terminal cost**
- $p = \nabla u$: **Co-state variable**

**Alternative** (minimization):
$$H(x, p, m) = \inf_{\alpha} \{\alpha \cdot p + L(x, \alpha, m)\}$$

**Convention**: Use infimum form for consistency with control theory literature.

### 1.4 Stochastic Processes

**Brownian Motion**:
- $W_t, B_t$: **Standard Brownian motion**
  - **Primary**: Use $W_t$ for general diffusions
  - **Alternative**: Use $B_t$ only when distinguishing multiple BMs
- $W_t^i$: **Idiosyncratic noise** for player $i$
- $W_t^0$ or $\theta_t$: **Common noise** (shared by all players)

**State Process**:
- $X_t, X_t^i$: **State variable** (general)
- $x$: **State variable** (deterministic/spatial)

### 1.5 Function Spaces

**Standard Notation**:
- $C^k(\mathbb{R}^d)$: $k$-times continuously differentiable
- $C_b(\mathbb{R}^d)$: Bounded continuous functions
- $C_c(\mathbb{R}^d)$: Compactly supported continuous
- $C^\infty_c(\mathbb{R}^d)$: Smooth compactly supported
- $L^p(\mathbb{R}^d)$: Lebesgue $p$-integrable, $p \in [1, \infty]$
- $W^{k,p}(\mathbb{R}^d)$: Sobolev space with $k$ weak derivatives in $L^p$
- $H^k(\mathbb{R}^d) = W^{k,2}(\mathbb{R}^d)$: Sobolev-Hilbert space

**Measure Spaces**:
- $\mathcal{P}(\mathbb{R}^d)$: All probability measures
- $\mathcal{P}_{ac}(\mathbb{R}^d)$: Absolutely continuous w.r.t. Lebesgue
- $\mathcal{P}_p(\mathbb{R}^d)$: $p$-th moment finite, $\int |x|^p \mu(dx) < \infty$

### 1.6 Operators and Derivatives

**Gradient and Divergence**:
- $\nabla f$, $\nabla_x f$: **Spatial gradient**
- $\text{div}(v)$, $\nabla \cdot v$: **Divergence** (prefer $\text{div}$ for clarity)
- $\Delta f = \nabla \cdot \nabla f$: **Laplacian**

**Functional Derivatives**:
- $\frac{\delta F}{\delta m}[m](x)$: **Lions derivative** (first variation)
- $\nabla_m F$: **Wasserstein gradient** w.r.t. measure

**Time Derivatives**:
- $\frac{\partial u}{\partial t}$, $\partial_t u$, $u_t$: **Partial time derivative**
- $\frac{du}{dt}$: **Total derivative** along trajectory

---

## 2. Cross-Document Mapping

### 2.1 Stochastic Calculus → Stochastic Games

**Connection Points**:
1. **Itô's formula** (stochastic_calculus) → **HJB equation derivation** (stochastic_games)
2. **Martingale representation** → **Optimal control characterization**
3. **Quadratic variation** → **Diffusion term in HJB**

**Notation Bridge**:
```
Stochastic Calculus:          Stochastic Games:
- dX_t = b dt + σ dW_t       →  dX_t^i = α_t^i dt + σ^i dW_t^i
- Itô integral ∫ f dW        →  Noise term in dynamics
- E[f(X_T) | F_t]            →  Value function u(t,X_t)
```

### 2.2 Information Geometry → Optimization

**Connection Points**:
1. **Fisher-Rao metric** (info_geom) → **Natural gradient** (optimization)
2. **Wasserstein distance** (info_geom) → **JKO scheme** (optimization)
3. **KL divergence** (info_geom) → **Entropic regularization** (optimization)

**Notation Bridge**:
```
Information Geometry:         Optimization:
- g_FR(μ)[ξ,η]               →  Fisher information matrix F(θ)
- W_2(μ,ν)                   →  Wasserstein metric in JKO
- D_KL(μ||ν)                 →  Regularization term in MFG
```

### 2.3 Convergence Theory → Implementation

**Connection Points**:
1. **Theoretical $O(1/\sqrt{N})$ rate** → **Empirical validation** (code)
2. **Regularity assumptions** → **Discretization requirements**
3. **Master equation** → **Finite-dimensional approximation**

**Gap to Address**:
- Theory: Continuous state/time on $\mathbb{R}^d \times [0,T]$
- Code: Discrete grid $(x_i, t_n) \in \mathbb{R}^{d \times N_x} \times \mathbb{R}^{N_t}$
- **Needed**: Discretization error analysis linking theory to practice

---

## 3. Theorem Numbering and Referencing

### 3.1 Numbering Scheme

**Format**: `Document.Section.Number`

Examples:
- **SDG.4.1**: Stochastic Differential Games, Section 4, Theorem 1
- **IG.7.2**: Information Geometry, Section 7, Theorem 2

### 3.2 Cross-Document References

**Standard Format**:
> See [SDG.4.1] for propagation of chaos theorem.
> This connects to the JKO scheme [IG.6.1] via Wasserstein gradient flow structure.

### 3.3 External References and Citation Standards

**Citation Format**: Use footnotes for all citations with full bibliographic details

**Inline Citation Style**:
```markdown
The convergence rate is optimal[^1].
```

**Footnote Citation Formats**:

1. **Journal Articles**:
```markdown
[^1]: Author1, A., & Author2, B. (Year). "Article title." *Journal Name*, Volume(Issue), Pages.

Example:
[^1]: Delarue, F., & Lacker, D. (2018). "From the master equation to mean field games." *Probability Theory and Related Fields*, 193(3-4), 573-649.
```

2. **Books**:
```markdown
[^2]: Author, A. (Year). *Book Title* (Edition if not first). Publisher.

Example:
[^2]: Carmona, R., & Delarue, F. (2018). *Probabilistic Theory of Mean Field Games with Applications* (Vol. II). Springer.
```

3. **Book Chapters**:
```markdown
[^3]: Author, A. (Year). "Chapter title." In Editor, E. (Ed.), *Book Title* (pp. pages). Publisher.

Example:
[^3]: Achdou, Y., & Laurière, M. (2020). "Mean field games and applications: Numerical aspects." In P.-L. Lions & B. Perthame (Eds.), *Partial Differential Equations and Applications* (pp. 249-307). Springer.
```

4. **Conference Papers**:
```markdown
[^4]: Author, A., & Author, B. (Year). "Paper title." *Proceedings of Conference*, Pages.

Example:
[^4]: Chizat, L., & Bach, F. (2018). "On the global convergence of gradient descent for over-parameterized models using optimal transport." *Proceedings of NeurIPS*, 2018, 3036-3046.
```

5. **Derived/Combined Results**:
```markdown
[^5]: Description of derivation from: Primary Source. Additional context from: Secondary Source.

Example:
[^5]: Convergence rate theorem derived from combining Sznitman (1991) propagation of chaos with master equation stability estimates from Cardaliaguet et al. (2019).
```

**Additional References Section**: After footnotes, include a categorized bibliography:

```markdown
### Additional Classical References

**Category Name**:
- Full citation in same format as footnotes
- Organized by topic for easy reference
```

**Cross-Document Citation Style**:
```markdown
See Section 4.1 of `stochastic_differential_games_theory.md` for propagation of chaos.
This connects to the JKO scheme discussed in `information_geometry_mfg.md` §6.2.
```

---

## 4. Mathematical Rigor Checklist

### 4.1 For Every Theorem

- [ ] **Hypotheses**: Explicitly listed (H1), (H2), ...
- [ ] **Statement**: Precise mathematical conclusion
- [ ] **Proof**: Sketch provided or reference given
- [ ] **Sharpness**: Discuss necessity of assumptions (if known)

### 4.2 For Every Definition

- [ ] **Domain**: Specify where object lives
- [ ] **Properties**: List key characteristics
- [ ] **Examples**: Provide concrete instances
- [ ] **Non-examples**: Show what doesn't satisfy definition

### 4.3 For Every Algorithm

- [ ] **Input**: Specify data and assumptions
- [ ] **Output**: Describe result
- [ ] **Complexity**: Provide computational cost
- [ ] **Convergence**: State guarantees (if any)

---

## 5. Common Pitfalls and Corrections

### 5.1 Measure vs Density Confusion

❌ **Incorrect**: "The measure $m(x)$ evolves..."
✅ **Correct**: "The density $m(x)$ of measure $\mu$ evolves..."

❌ **Incorrect**: "$\mu_t$ satisfies Fokker-Planck: $\partial_t \mu = ...$"
✅ **Correct**: "The density $m_t$ of $\mu_t$ satisfies: $\partial_t m_t = ...$"

### 5.2 Index Notation

❌ **Incorrect**: "$\sum_i x_i$" when summation range unclear
✅ **Correct**: "$\sum_{i=1}^N x_i$" or "$\sum_{i \in I} x_i$"

❌ **Incorrect**: Mixing Einstein summation without declaration
✅ **Correct**: State "Einstein summation convention: repeated indices summed"

### 5.3 Functional Derivative

❌ **Incorrect**: "$\nabla_m u$" without specifying type of derivative
✅ **Correct**: "$\frac{\delta u}{\delta m}[m](y)$" (Lions derivative) or "$\nabla_{W_2} u$" (Wasserstein gradient)

### 5.4 Stochastic Calculus

❌ **Incorrect**: "$du = \frac{\partial u}{\partial t} dt + \nabla u \cdot dX$"
✅ **Correct**: "$du = \left(\frac{\partial u}{\partial t} + \nabla u \cdot b + \frac{1}{2} \text{tr}(\sigma \sigma^T \nabla^2 u)\right) dt + \nabla u \cdot \sigma dW$"
(Include Itô correction term!)

---

## 6. Document-Specific Conventions

### 6.1 Stochastic Differential Games

**Key Notation**:
- $V^{i,N}$: N-player value
- $\mu^N$: Empirical measure
- $\alpha^{i,N}$: Control of player $i$ in N-player game

**Asymptotic Notation**:
- Use $O(1/N)$, $o(1)$, $\Theta(1/\sqrt{N})$ rigorously
- State constants explicitly when possible

### 6.2 Information Geometry

**Key Notation**:
- $g_{FR}$: Fisher-Rao metric
- $g_W$: Wasserstein (Otto) metric
- $D_{KL}$: Kullback-Leibler divergence
- $W_p$: $p$-Wasserstein distance

**Manifold Notation**:
- $\mathcal{M}$: Statistical manifold
- $T_\mu \mathcal{M}$: Tangent space at $\mu$
- $\nabla^{(e)}$, $\nabla^{(m)}$: Exponential and mixture connections

### 6.3 Stochastic MFG

**Key Notation**:
- $\theta_t$ or $W_t^0$: Common noise
- $u^\theta(t,x)$: Conditional value function
- $m^\theta(t,x)$: Conditional density

**Ensemble Averaging**:
- $\bar{u} = \mathbb{E}[u^\theta]$: Averaged value
- $\bar{m} = \mathbb{E}[m^\theta]$: Averaged density

---

## 7. Implementation Connections

### 7.1 Theory → Code Mapping

| Theory Object | Code Location | Notes |
|---------------|---------------|-------|
| $u(t,x)$ | `result.U` | Discretized on `(t_grid, x_grid)` |
| $m(t,x)$ | `result.M` | Empirical or discretized density |
| $H(x,p,m)$ | `problem.hamiltonian()` | Implemented as method |
| $\nabla_{W_2} \mathcal{E}$ | `WassersteinGradientFlow` | Class in `alg/optimization/` |
| $F^{-1}(\theta) \nabla J$ | `natural_gradient()` | Function in `utils/optimization/` |

### 7.2 Discretization Standards

**Spatial**: Use notation $(x_i)_{i=1}^{N_x}$ for grid points
**Temporal**: Use $(t_n)_{n=0}^{N_t}$ for time grid
**Discrete Solution**: $u_i^n \approx u(t_n, x_i)$

**Error Analysis**: Connect to theory via
$$\|u^h - u\|_{L^\infty} \leq C h^p$$
where $h = \Delta x$, $p$ is order of convergence.

---

## 8. Review Checklist

Before finalizing any theory document:

- [ ] All symbols defined before first use
- [ ] Notation consistent with this guide
- [ ] Cross-references use standard format
- [ ] Theorems numbered and complete
- [ ] Proofs sketched or referenced
- [ ] Examples provided for key concepts
- [ ] Connection to code noted (if applicable)
- [ ] References complete and formatted as footnotes

---

**Document Status**: Active reference guide for all theory documentation
**Usage**: Consult when writing or reviewing any theory document in `docs/theory/`
**Related**: All theory documents in `docs/theory/`
