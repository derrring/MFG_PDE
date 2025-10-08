# Establishing Robust Convergence Criteria for Coupled Numerical Systems

**Document Type**: Numerical Analysis
**Created**: Original date unknown
**Enhanced**: October 8, 2025
**Status**: Mathematical foundation for MFG solver convergence
**Related**: `mathematical_background.md`, `stochastic_differential_games_theory.md`

---

## Introduction

This document provides rigorous mathematical foundations for defining convergence criteria in **coupled HJB-FPK systems** arising from Mean Field Games. Standard single-equation convergence tests fail for coupled systems due to:

1. **Stochastic sampling noise** in particle-based FPK approximations
2. **Feedback oscillations** between the forward (FPK) and backward (HJB) equations
3. **Lack of monotone convergence** in non-contractive iterations

We present mathematically sound criteria based on probability metrics, statistical stability, and oscillation analysis, with theoretical justification and implementation guidance.

---

## The Challenge with Standard $L^2$ Error

### 1.1 Why Simple Error Checks Fail

**Naive Convergence Test**: Declare convergence if $\|x_k - x_{k-1}\|_{L^2} < \epsilon$

**Failure Modes**:

1. **Statistical Noise in Particle Methods**:[^1]
   For particle approximation $m_k^N = \frac{1}{N}\sum_{i=1}^N \delta_{X_i^k}$ of density $m_k$:
   $$\mathbb{E}[W_2^2(m_k^N, m_k)] = O(N^{-1/d})$$
   This variance prevents deterministic convergence below $O(N^{-1/(2d)})$ tolerance.

2. **Coupled System Oscillation**:[^2]
   In the MFG fixed-point iteration:
   $$u_{k+1} = \mathcal{T}_{\text{HJB}}[m_k], \quad m_{k+1} = \mathcal{T}_{\text{FPK}}[u_{k+1}]$$
   Noise in $m_k$ induces noise in $u_{k+1}$, creating a **non-contractive feedback loop**.

3. **Non-Uniqueness and Oscillation Between Equilibria**:[^3]
   When the Lasry-Lions monotonicity condition fails, multiple equilibria may exist:
   $$\langle m - n, \delta_m H - \delta_n H \rangle \geq \lambda \|m - n\|^2$$
   Iterations may oscillate between basins of attraction.

### 1.2 Theoretical Foundation: Fixed-Point Iterations

**Definition (MFG Fixed-Point Map)**: Define $\Phi: \mathcal{P}_2(\mathbb{R}^d) \to \mathcal{P}_2(\mathbb{R}^d)$ by:
$$\Phi(m) = \mathcal{T}_{\text{FPK}}[\mathcal{T}_{\text{HJB}}[m]]$$
where $\mathcal{P}_2$ is the space of probability measures with finite second moment.

**Theorem (Contraction in Monotone Case)**:[^4]
*If the MFG system satisfies the Lasry-Lions monotonicity condition with constant $\lambda > 0$, then $\Phi$ is a contraction in Wasserstein distance $W_2$:*
$$W_2(\Phi(m), \Phi(n)) \leq \kappa W_2(m, n), \quad \kappa < 1$$
*and the Picard iteration $m_{k+1} = \Phi(m_k)$ converges exponentially:*
$$W_2(m_k, m_*) \leq \kappa^k W_2(m_0, m_*)$$

**Corollary**: In the monotone case, $W_2(m_k, m_{k-1}) \to 0$ exponentially.

**Remark**: When monotonicity fails, $\Phi$ may not be contractive, and $W_2(m_k, m_{k-1})$ may oscillate around a small positive value. This motivates **stabilization criteria** rather than vanishing residual criteria.

---

## 1. Convergence Criteria for the Distribution $m$

To handle noisy, particle-based distributions, we use **probability metrics** robust to sampling variance.

### 1.1 Statistical Moment Stability

**Rationale**: Moments are continuous linear functionals on $\mathcal{P}_2$ and have lower variance than pointwise densities.

**Definition (First and Second Moments)**:
$$\mu[m] = \int x \, m(dx), \quad \Sigma[m] = \int (x - \mu[m])(x - \mu[m])^T m(dx)$$

**Convergence Criterion (Moment Stability)**:
$$|\mu[m_k] - \mu[m_{k-1}]| < \epsilon_{\mu}, \quad \|\Sigma[m_k] - \Sigma[m_{k-1}]\|_F < \epsilon_{\Sigma}$$
where $\|\cdot\|_F$ is the Frobenius norm.

**Theorem (Moment Convergence Implies Weak Convergence)**:[^5]
*If $\{m_k\}$ is tight and all moments converge, then $m_k$ converges weakly to a limit $m_*$.*

**Limitation**: Moment convergence does not imply convergence in stronger metrics like Wasserstein distance.

### 1.2 Kullback-Leibler (KL) Divergence

**Definition (KL Divergence)**:[^6]
For probability measures $\mu, \nu$ with $\mu \ll \nu$ (absolutely continuous):
$$D_{KL}(\mu \| \nu) = \int \log\left(\frac{d\mu}{d\nu}\right) d\mu = \int m(x) \log\left(\frac{m(x)}{n(x)}\right) dx$$

**Properties**:
- $D_{KL}(\mu \| \nu) \geq 0$ with equality iff $\mu = \nu$ (Gibbs inequality)
- **Not a metric**: Asymmetric and violates triangle inequality
- Sensitive to tail behavior and zero-density regions

**Convergence Criterion**:
$$D_{KL}(m_k \| m_{k-1}) < \epsilon_{KL}$$

**Numerical Stability**: Add regularization $\delta \sim 10^{-10}$ to avoid $\log(0)$:
$$D_{KL}^{\delta}(m_k \| m_{k-1}) = \sum_i (m_k(x_i) + \delta) \log\left(\frac{m_k(x_i) + \delta}{m_{k-1}(x_i) + \delta}\right)$$

**Caveat**: KL divergence is **not robust to non-overlapping supports**, making it unreliable for early iterations.

### 1.3 Wasserstein Distance (Recommended)

**Definition (Wasserstein-$p$ Distance)**:[^7]
$$W_p(\mu, \nu) = \left(\inf_{\pi \in \Pi(\mu, \nu)} \int |x - y|^p \, \pi(dx, dy)\right)^{1/p}$$
where $\Pi(\mu, \nu)$ is the set of couplings (joint measures with marginals $\mu, \nu$).

**Wasserstein-1 (Earth Mover's Distance)**:
$$W_1(\mu, \nu) = \sup_{f \in \text{Lip}_1} \left|\int f \, d\mu - \int f \, d\nu\right|$$
where $\text{Lip}_1 = \{f : |f(x) - f(y)| \leq |x - y|\}$ (Kantorovich-Rubinstein duality).

**Key Properties**:[^8]
1. **Metrizes weak convergence**: $W_p(m_k, m_*) \to 0 \iff m_k \xrightarrow{w} m_*$ (on $\mathcal{P}_p$)
2. **Robust to support differences**: Well-defined even for disjoint supports
3. **Continuous w.r.t. moments**: $|W_2^2(\mu, \nu) - W_2^2(\mu', \nu')| \leq C(|\mu[\cdot] - \mu'[\cdot]| + \ldots)$
4. **Compatible with MFG theory**: Natural metric for gradient flows (JKO scheme)

**Convergence Criterion (RECOMMENDED)**:
$$W_1(m_k, m_{k-1}) < \epsilon_W \quad \text{or} \quad W_2(m_k, m_{k-1}) < \epsilon_W$$

**Theorem (Stability Under Sampling Noise)**:[^9]
*For particle approximations $m_k^N$ with $N$ particles:*
$$\mathbb{E}[W_2(m_k^N, m_k)] \leq C N^{-1/(2d)}$$
*Therefore, choosing $\epsilon_W \gg N^{-1/(2d)}$ ensures convergence is not masked by sampling noise.*

**Computational Cost**:
- **1D**: $O(N \log N)$ via sorting
- **High-D**: $O(N^3)$ via linear programming or $O(N^2 \log N)$ via Sinkhorn algorithm[^10]

---

## 2. Convergence Criteria for the Value Function $u$

The value function inherits noise from $m$ through the Hamiltonian coupling. Standard residual tests fail because:
$$\|u_k - u_{k-1}\|_{L^2} \not\to 0 \quad \text{(oscillates around small positive value)}$$

### 2.1 Oscillation Analysis: Stabilization vs. Convergence

**Definition (Stabilization)**: A sequence $\{e_k\}$ is **stabilized** if:
1. **Bounded oscillation**: $\limsup_{k \to \infty} e_k < \epsilon$
2. **Stationary statistics**: $\text{Var}(e_{k:k+N}) \to 0$ as $k \to \infty$

**Lemma (Oscillation Bound in Non-Monotone Case)**:[^11]
*If $W_2(m_k, m_{k-1}) \leq \delta_m$ (small but non-zero due to sampling noise), and the HJB operator $\mathcal{T}_{\text{HJB}}$ is Lipschitz with constant $L_H$, then:*
$$\|u_k - u_{k-1}\|_{L^2} \leq L_H \delta_m$$
*Thus, the value function error stabilizes at amplitude proportional to the noise in $m$.*

### 2.2 Statistical Stabilization Criteria

**Criterion (Two-Part Stabilization Test)**:
Monitor the recent history of $L^2$ errors: $\mathbf{e}_u = (e_k, e_{k-1}, \ldots, e_{k-N+1})$ where $e_k = \|u_k - u_{k-1}\|_{L^2}$.

1. **Small Magnitude** (oscillation amplitude is acceptably small):
   $$\overline{e}_u := \frac{1}{N}\sum_{j=0}^{N-1} e_{k-j} < \epsilon_{u,\text{mag}}$$

2. **Stability** (oscillation has reached steady state):
   $$s_u := \sqrt{\frac{1}{N}\sum_{j=0}^{N-1} (e_{k-j} - \overline{e}_u)^2} < \epsilon_{u,\text{stab}}$$

**Interpretation**:
- $\overline{e}_u < \epsilon_{u,\text{mag}}$ ensures we are near equilibrium
- $s_u < \epsilon_{u,\text{stab}}$ (very small) ensures the oscillation has stabilized, not still decaying

**Choice of Window Size**: Typically $N = 10$ iterations. Too small misses oscillation pattern; too large delays convergence detection.

### 2.3 Alternative: Gradient Norm Criterion

**Motivation**: For HJB equations, the gradient $\nabla u$ determines optimal controls. Stability in $\nabla u$ may be more meaningful than pointwise $u$.

**Criterion**:
$$\|\nabla u_k - \nabla u_{k-1}\|_{L^2(\Omega)} < \epsilon_{\nabla u}$$

**Advantage**: Less sensitive to constant shifts in $u$ (which do not affect controls).

**Caveat**: Requires additional computational cost for gradient computation.

---

## 3. Recommended Multi-Criteria Convergence Checklist

The most robust approach combines multiple perspectives. Declare convergence when **all** conditions hold:

### 3.1 Combined Convergence Conditions

**For the distribution $m$** (choose one or more):
- **(Primary)** Wasserstein distance criterion:
  $$W_1(m_k, m_{k-1}) < \epsilon_W$$
- **(Supplementary)** Moment stability:
  $$|\mu[m_k] - \mu[m_{k-1}]| < \epsilon_{\mu} \quad \text{AND} \quad \|\Sigma[m_k] - \Sigma[m_{k-1}]\|_F < \epsilon_{\Sigma}$$

**For the value function $u$**:
- **(Required)** Statistical stabilization:
  $$\overline{e}_u < \epsilon_{u,\text{mag}} \quad \text{AND} \quad s_u < \epsilon_{u,\text{stab}}$$
- **(Optional)** Gradient stability (if applicable):
  $$\|\nabla u_k - \nabla u_{k-1}\|_{L^2} < \epsilon_{\nabla u}$$

### 3.2 Tolerance Calibration

**Guideline**: Choose tolerances based on:
1. **Sampling variance**: For $N$ particles, set $\epsilon_W \sim \max(10^{-4}, 5 N^{-1/(2d)})$
2. **HJB Lipschitz constant**: Set $\epsilon_{u,\text{mag}} \sim 10 L_H \epsilon_W$
3. **Stabilization threshold**: Set $\epsilon_{u,\text{stab}} \sim 0.1 \epsilon_{u,\text{mag}}$

**Theorem (Convergence with Calibrated Tolerances)**:[^12]
*Under the above calibration, if all criteria are satisfied, then the numerical solution $(u_k, m_k)$ approximates an MFG equilibrium $(u_*, m_*)$ with error:*
$$\|u_k - u_*\|_{L^2} + W_2(m_k, m_*) \lesssim \epsilon_W + h^p + \text{iter\_tol}$$
*where $h$ is spatial discretization, $p$ is scheme order, and iter_tol is iteration tolerance.*

---

## 4. Theoretical Foundations

### 4.1 Fixed-Point Theorems

**Banach Fixed-Point Theorem**: If $\Phi: X \to X$ is a contraction on complete metric space $(X, d)$:
$$d(\Phi(x), \Phi(y)) \leq \kappa d(x, y), \quad \kappa < 1$$
then $\Phi$ has unique fixed point $x_* \in X$, and Picard iteration converges:
$$d(x_k, x_*) \leq \frac{\kappa^k}{1 - \kappa} d(x_1, x_0)$$

**Application to MFG**: In the monotone case, $(\mathcal{P}_2, W_2)$ is complete, and $\Phi$ is a contraction.[^13]

**Non-Contractive Case**: Use Brouwer or Schauder fixed-point theorems, which guarantee existence but not uniqueness or convergence rates.[^14]

### 4.2 Propagation of Noise Through Coupled Systems

**Lemma (Noise Propagation)**:[^15]
*Let $u_k = \mathcal{T}_{\text{HJB}}[m_k]$ where $\mathcal{T}_{\text{HJB}}$ is Lipschitz with constant $L_H$. If $W_2(m_k, m_{k-1}) \leq \delta$, then:*
$$\|u_k - u_{k-1}\|_{L^{\infty}} \leq L_H \delta$$

**Proof Sketch**: By definition of $\mathcal{T}_{\text{HJB}}$, the Hamiltonian depends continuously on $m$. Lipschitz continuity in the $W_2$ metric follows from stability estimates for viscosity solutions.[^16]

### 4.3 Statistical Consistency of Particle Approximations

**Theorem (Monte Carlo Convergence Rate)**:[^17]
*For $m_k^N = \frac{1}{N}\sum_{i=1}^N \delta_{X_i^k}$ with i.i.d. samples $X_i \sim m_k$:*
$$\mathbb{E}[W_2^2(m_k^N, m_k)] = O(N^{-1/d}) \quad \text{(curse of dimensionality)}$$

**Implication**: In high dimensions, very large $N$ is required for $\epsilon_W < 10^{-4}$.

**Advanced Methods**: Quasi-Monte Carlo and variance reduction can improve to $O(N^{-1} (\log N)^d)$.[^18]

---

## 5. Implementation in MFG_PDE

The MFG_PDE package implements these convergence criteria through:

### 5.1 Core Components

**Location**: `mfg_pde/utils/convergence.py`

**Classes**:
1. **`AdvancedConvergenceMonitor`** - Tracks multiple convergence metrics simultaneously
   - Wasserstein distance via `scipy.stats.wasserstein_distance` (1D) or `POT` library (multi-D)
   - KL divergence with regularization
   - Moment tracking (mean, variance, higher moments)

2. **`StabilizationDetector`** - Historical error analysis for oscillating systems
   - Sliding window statistics
   - Trend analysis (increasing, decreasing, stabilized)
   - Autocorrelation detection

3. **`MultiCriteriaValidator`** - Combined convergence assessment
   - Logical combination of criteria (AND/OR logic)
   - Weighted scoring for multi-objective convergence
   - Early stopping with partial convergence

### 5.2 Integration with Solvers

**Example Usage** (see `mfg_pde/alg/numerical/hjb_solvers/`):
```python
from mfg_pde.utils.convergence import AdvancedConvergenceMonitor

monitor = AdvancedConvergenceMonitor(
    metrics=['wasserstein', 'moment_stability'],
    tolerances={'wasserstein': 1e-4, 'moment_mean': 1e-5},
    window_size=10
)

for k in range(max_iter):
    u_k = solve_hjb(m_k)
    m_k = solve_fpk(u_k)

    converged = monitor.check_convergence(m_k, m_prev, u_k, u_prev)
    if converged:
        break
```

### 5.3 Visualization and Diagnostics

**Tools**:
- Convergence history plots (error vs. iteration)
- Phase portraits (u-error vs. m-error)
- Spectral analysis of oscillations
- Diagnostic reports with convergence assessment

**Location**: `mfg_pde/utils/visualization/convergence_diagnostics.py`

---

## 6. Practical Recommendations

### 6.1 Choosing the Right Criteria

| Scenario | Recommended Primary Criterion | Supplementary |
|:---------|:------------------------------|:--------------|
| Smooth, monotone MFG | Wasserstein $W_1$ or $W_2$ | Moment stability |
| Non-monotone, oscillatory | Statistical stabilization | Wasserstein |
| High-dimensional ($d \geq 4$) | Moment stability | Sinkhorn-Wasserstein |
| Particle methods | Wasserstein (robust to noise) | KL (if smooth) |

### 6.2 Debugging Non-Convergence

**If convergence stalls**:
1. **Check monotonicity**: Compute $\langle m_k - m_{k-1}, \delta_m H \rangle$ to verify Lasry-Lions condition
2. **Inspect oscillation frequency**: FFT of error sequence to detect periodic oscillations
3. **Increase particle count**: If $\epsilon_W \sim N^{-1/(2d)}$, noise floor is hit
4. **Relax tolerances**: Ensure $\epsilon_W \gg$ discretization error $h^p$

**If convergence is too slow**:
1. **Use Anderson acceleration**: Extrapolate from history of iterates[^19]
2. **Adaptive relaxation**: $m_{k+1} = \omega \Phi(m_k) + (1 - \omega) m_k$ with optimal $\omega$
3. **Multigrid methods**: Coarse-grid correction for faster convergence

---

## References

[^1]: Fournier, N., & Guillin, A. (2015). "On the rate of convergence in Wasserstein distance of the empirical measure." *Probability Theory and Related Fields*, 162(3-4), 707-738.

[^2]: Achdou, Y., & Capuzzo-Dolcetta, I. (2010). "Mean field games: Numerical methods." *SIAM Journal on Numerical Analysis*, 48(3), 1136-1162.

[^3]: Lasry, J.-M., & Lions, P.-L. (2007). "Mean field games." *Japanese Journal of Mathematics*, 2(1), 229-260.

[^4]: Cardaliaguet, P., Delarue, F., Lasry, J.-M., & Lions, P.-L. (2019). *The Master Equation and the Convergence Problem in Mean Field Games*. Princeton University Press.

[^5]: Billingsley, P. (1995). *Probability and Measure* (3rd ed.). Wiley.

[^6]: Kullback, S., & Leibler, R. A. (1951). "On information and sufficiency." *Annals of Mathematical Statistics*, 22(1), 79-86.

[^7]: Villani, C. (2009). *Optimal Transport: Old and New*. Springer.

[^8]: Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures* (2nd ed.). Birkhäuser.

[^9]: Derived from Fournier & Guillin (2015) with constants depending on dimension $d$ and support diameter.

[^10]: Cuturi, M. (2013). "Sinkhorn distances: Lightspeed computation of optimal transport." *Proceedings of NIPS*, 2013, 2292-2300.

[^11]: Derived from Lipschitz stability of viscosity solutions (Barles & Souganidis 1991) combined with Wasserstein stability.

[^12]: Combined result from discretization error analysis (Achdou et al. 2010) and iterative solver tolerance (Saude 2015).

[^13]: See Cardaliaguet et al. (2019), Chapter 4, Theorem 4.2 for proof in MFG context.

[^14]: Brouwer, L. E. J. (1911). "Über Abbildung von Mannigfaltigkeiten." *Mathematische Annalen*, 71(1), 97-115.

[^15]: Stability estimate for HJB equations from Evans, L. C. (2010). *Partial Differential Equations* (2nd ed.). AMS.

[^16]: Barles, G., & Souganidis, P. E. (1991). "Convergence of approximation schemes for fully nonlinear second order equations." *Asymptotic Analysis*, 4(3), 271-283.

[^17]: Standard Monte Carlo theory; see Fournier & Guillin (2015) for sharp constants in Wasserstein distance.

[^18]: Caflisch, R. E. (1998). "Monte Carlo and quasi-Monte Carlo methods." *Acta Numerica*, 7, 1-49.

[^19]: Anderson, D. G. (1965). "Iterative procedures for nonlinear integral equations." *Journal of the ACM*, 12(4), 547-560.

---

### Additional Classical References

**Convergence Theory for Iterative Methods**:
- Saad, Y., & Schultz, M. H. (1986). "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems." *SIAM Journal on Scientific and Statistical Computing*, 7(3), 856-869.
- Walker, H. F., & Ni, P. (2011). "Anderson acceleration for fixed-point iterations." *SIAM Journal on Numerical Analysis*, 49(4), 1715-1735.

**Probability Metrics and Optimal Transport**:
- Rachev, S. T., & Rüschendorf, L. (1998). *Mass Transportation Problems* (Vols. I-II). Springer.
- Peyré, G., & Cuturi, M. (2019). "Computational optimal transport." *Foundations and Trends in Machine Learning*, 11(5-6), 355-607.

**Convergence of MFG Numerical Schemes**:
- Carlini, E., & Silva, F. J. (2014). "A fully discrete semi-Lagrangian scheme for a first order mean field game problem." *SIAM Journal on Numerical Analysis*, 52(1), 45-67.
- Benamou, J.-D., & Carlier, G. (2015). "Augmented Lagrangian methods for transport optimization, mean field games and degenerate elliptic equations." *Journal of Optimization Theory and Applications*, 167(1), 1-26.

**Variance Reduction and Quasi-Monte Carlo**:
- Owen, A. B. (2013). *Monte Carlo Theory, Methods and Examples*. Online textbook.
- Dick, J., & Pillichshammer, F. (2010). *Digital Nets and Sequences: Discrepancy Theory and Quasi-Monte Carlo Integration*. Cambridge University Press.

---

**Document Status**: Enhanced with mathematical rigor and comprehensive references
**Usage**: Reference for MFG solver convergence criteria and implementation
**Related Code**: `mfg_pde/utils/convergence.py`, solver implementations in `mfg_pde/alg/numerical/`
