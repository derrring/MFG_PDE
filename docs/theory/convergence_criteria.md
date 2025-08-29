# Establishing Robust Convergence Criteria for Coupled Numerical Systems

This document summarizes strategies for defining reliable convergence criteria for coupled systems, such as the Mean Field Game (MFG) system involving the Hamilton-Jacobi-Bellman (HJB) and Fokker-Planck equations. Standard error checks can often fail in these scenarios, requiring more sophisticated methods.

## The Challenge with Standard L2 Error

A simple L2 error check, where convergence is declared if $||x_k - x_{k-1}||_2 < \epsilon$, can be unreliable for two primary reasons:

1.  **Statistical Noise:** Methods that use particles to approximate a probability distribution (`m`), like the Fokker-Planck equation solver, have inherent statistical noise. This noise can prevent the L2 error from consistently staying below a strict tolerance.
2.  **System Oscillation:** In a coupled system, the output of one equation is the input for the other. Noise from the distribution `m` is fed into the HJB solver, causing its solution, the value function `u`, to oscillate. This feedback loop can prevent the numerical solutions from ever settling down perfectly, even when they have reached a stable equilibrium.

---

## 1. Criteria for the Distribution (`m`)

To handle the noisy, particle-based distribution `m`, we must use criteria that are robust to small statistical fluctuations.

### Monitoring Statistical Moments

This approach focuses on stable, global properties of the distribution rather than noisy, point-wise values. Convergence is declared when the mean ($\mu$) and variance ($\sigma^2$) of the distribution stop changing significantly.

* **Mean Stability:**
    $|\text{mean}(m_k) - \text{mean}(m_{k-1})| < \epsilon_{\mu}$
* **Variance Stability:**
    $|\text{var}(m_k) - \text{var}(m_{k-1})| < \epsilon_{\sigma^2}$

### KL Divergence

Kullback-Leibler (KL) divergence is an information-theoretic measure of how one probability distribution differs from another. A value close to zero indicates similarity.

* **Check:**
    $D_{KL}(m_k || m_{k-1}) < \epsilon_{KL}$
* **Formula:**
    $D_{KL}(m_k || m_{k-1}) = \sum_{i} m_k(x_i) \log\left(\frac{m_k(x_i)}{m_{k-1}(x_i)}\right)$
    *Note: A small value, $\delta$, must be added to the distributions to avoid numerical errors from `log(0)`.*

### Wasserstein Distance

The Wasserstein-1 distance, or "Earth Mover's Distance," is often considered the **gold standard** for comparing probability distributions. It measures the minimum "work" required to transform one distribution into the other and is robust even for non-overlapping distributions.

* **Check:**
    $W_1(m_k, m_{k-1}) < \epsilon_{W}$

---

## 2. Criteria for the Value Function (`u`)

The value function `u` oscillates due to the noisy `m` input. Therefore, its L2 error, $e_k = ||u_k - u_{k-1}||_2$, will not converge to zero but will **stabilize at a small, non-zero value**. The goal is to detect this stabilization.

This requires a two-part check on the recent history of the L2 error (e.g., the last $N=10$ values):

1.  **Check for Small Magnitude:** The amplitude of the oscillation is acceptably small.
    $\text{mean}(e_k, e_{k-1}, \dots, e_{k-N+1}) < \epsilon_{\text{mag}}$

2.  **Check for Stability:** The error has stopped decreasing and the oscillation has reached a steady state.
    $\text{std}(e_k, e_{k-1}, \dots, e_{k-N+1}) < \epsilon_{\text{stab}}$
    *Here, $\epsilon_{\text{stab}}$ would be a very small number, confirming the error values themselves are no longer changing.*

---

## 3. Recommended Multi-Criteria Checklist

The most reliable approach is to combine these methods into a final checklist. Convergence is declared only when the system is stable from multiple perspectives simultaneously.

A robust checklist would be:

* **For the distribution `m`:** The Wasserstein distance has fallen below its tolerance.
    $W_1(m_k, m_{k-1}) < \epsilon_{W}$
* **For the value function `u`:** The L2 error has stabilized at a small magnitude.
    $\text{mean}(\mathbf{e}_u) < \epsilon_{u, \text{mag}} \quad \text{AND} \quad \text{std}(\mathbf{e}_u) < \epsilon_{u, \text{stab}}$

## Implementation in MFG_PDE

The MFG_PDE package implements these convergence criteria through:

1. **`AdvancedConvergenceMonitor`** class - Tracks multiple convergence metrics
2. **Distribution comparison utilities** - Wasserstein distance, KL divergence, moment tracking
3. **Stabilization detection** - Historical error analysis for oscillating systems
4. **Multi-criteria validation** - Combined convergence assessment

See the implementation in `mfg_pde/utils/convergence.py` and integration examples in the solver documentation.
