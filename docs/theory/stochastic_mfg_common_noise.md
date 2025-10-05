# Common Noise Mean Field Games: Mathematical Formulation

**Author**: MFG_PDE Development Team
**Date**: October 2025
**Status**: Production Implementation
**Related**: Phase 2.2 Stochastic MFG Extensions

---

## Overview

Common Noise Mean Field Games extend the classical MFG framework to scenarios where all agents observe a shared stochastic process $\theta_t$ that affects their decision-making. This framework is essential for modeling:

- **Financial Markets**: Shared market indices (VIX, interest rates) affecting all traders
- **Epidemiology**: Random policy changes or external events affecting population dynamics
- **Robotics**: Shared sensor measurements with noise affecting multi-agent coordination

**Key References**:
- Carmona & Delarue (2018): *Probabilistic Theory of Mean Field Games*
- Carmona, Fouque, & Sun (2015): *Mean Field Games and Systemic Risk*

---

## Mathematical Framework

### Problem Statement

Consider $N$ agents with individual states $X_i^{(N)} \in \mathbb{R}^d$ and a **common noise** process $\theta_t \in \mathbb{R}^m$ observable by all agents.

**Individual Agent Dynamics**:
```math
dX_i^{(N)} = b(X_i^{(N)}, \alpha_i, \mu_N, \theta_t) dt + \sigma dW_i^t
```

where:
- $\alpha_i$ is the individual control
- $\mu_N = \frac{1}{N}\sum_{j=1}^N \delta_{X_j^{(N)}}$ is the empirical measure
- $W_i^t$ are independent Brownian motions (idiosyncratic noise)
- $\theta_t$ is the common noise process

**Common Noise Process**:
```math
d\theta_t = \mu(\theta_t, t) dt + \sigma_\theta(\theta_t, t) dB_t
```

where $B_t$ is a Brownian motion independent of all $W_i^t$.

**Individual Cost**:
```math
J_i(\alpha_i, \mu_N, \theta) = \mathbb{E}\left[\int_0^T L(X_i^{(N)}, \alpha_i, \mu_N, \theta_t) dt + g(X_i^{(N)}_T, \theta_T)\right]
```

### Mean Field Limit

As $N \to \infty$, the system converges to a **Common Noise MFG** characterized by:

**Conditional HJB Equation** (given noise path $\theta$):
```math
\frac{\partial u^\theta}{\partial t} + H(x, \nabla u^\theta, m^\theta, \theta_t) + \frac{\sigma^2}{2} \Delta u^\theta = 0
```

**Terminal Condition**:
```math
u^\theta(T, x) = g(x, \theta_T)
```

**Conditional Fokker-Planck Equation**:
```math
\frac{\partial m^\theta}{\partial t} - \text{div}(m^\theta \nabla_p H(x, \nabla u^\theta, m^\theta, \theta_t)) - \frac{\sigma^2}{2} \Delta m^\theta = 0
```

**Initial Condition**:
```math
m^\theta(0, x) = m_0(x)
```

**Key Property**: For each realization $\theta$ of the common noise, we have a **conditional MFG** problem.

---

## Conditional vs Unconditional Solutions

### Conditional Solution

Given a specific noise path $\theta_t$, the conditional solution $(u^\theta, m^\theta)$ represents:
- $u^\theta(t, x)$: Value function for an agent at state $x$ given noise history
- $m^\theta(t, x)$: Population density conditioned on noise realization

### Unconditional Solution (Expectation)

The **unconditional** quantities are obtained by averaging over noise realizations:

```math
\bar{u}(t, x) = \mathbb{E}_\theta[u^\theta(t, x)]
```
```math
\bar{m}(t, x) = \mathbb{E}_\theta[m^\theta(t, x)]
```

These represent the expected value function and density over all possible noise paths.

---

## Numerical Solution: Monte Carlo Method

### Algorithm Overview

The Common Noise MFG is solved using **Monte Carlo sampling** over noise realizations:

**Step 1: Sample Noise Paths**
Generate $K$ independent paths of the common noise process:
```math
\theta^1, \theta^2, \ldots, \theta^K
```

**Step 2: Solve Conditional MFG**
For each noise path $k = 1, \ldots, K$, solve the conditional MFG:
```math
(u^k, m^k) = \text{solve\_conditional\_MFG}(\theta^k)
```

**Step 3: Monte Carlo Aggregation**
Compute expectations via sample average:
```math
\bar{u}(t, x) \approx \frac{1}{K} \sum_{k=1}^K u^k(t, x)
```
```math
\bar{m}(t, x) \approx \frac{1}{K} \sum_{k=1}^K m^k(t, x)
```

### Variance Reduction

Standard Monte Carlo has error $\mathcal{O}(K^{-1/2})$. We improve convergence using:

#### 1. Quasi-Monte Carlo (QMC)

Use **Sobol sequences** instead of random sampling for better coverage:
- Sobol points have low discrepancy in $[0,1]^d$
- Typical error reduction: $\mathcal{O}((\log K)^d / K)$ vs $\mathcal{O}(K^{-1/2})$
- Implementation: `QuasiMCSampler` with `sequence_type="sobol"`

#### 2. Control Variates

If a related problem has known solution $v(t,x)$:
```math
\bar{u}_{\text{CV}} = \bar{u}_{\text{MC}} + \beta(\mathbb{E}[v] - \bar{v}_{\text{MC}})
```

Optimal $\beta$ minimizes variance: $\beta^* = \frac{\text{Cov}(u,v)}{\text{Var}(v)}$

#### 3. Antithetic Variables

For symmetric noise processes, sample in pairs:
```math
(\theta^k, -\theta^k) \quad \text{for } k = 1, \ldots, K/2
```

Reduces variance when function has symmetric structure.

---

## Computational Complexity

### Sequential Implementation

For $K$ noise samples with grid size $N_x \times N_t$:

**Time Complexity**: $\mathcal{O}(K \cdot T_{\text{MFG}})$

where $T_{\text{MFG}}$ is the cost of solving one conditional MFG:
- Finite Difference: $\mathcal{O}(N_x N_t \log N_x)$ per iteration
- Neural: $\mathcal{O}(N_{\text{epochs}} \cdot N_{\text{samples}})$

**Space Complexity**: $\mathcal{O}(K \cdot N_x \cdot N_t)$ to store all samples

### Parallel Implementation

The $K$ conditional MFG problems are **embarrassingly parallel**:

**Ideal Speedup**: $K$ times faster with $K$ processors

**Practical Speedup**: $\eta \cdot K$ where $\eta \approx 0.8-0.95$ is parallel efficiency

**Implementation**: `ProcessPoolExecutor` with $P$ workers solves $K$ problems in time:
```math
T_{\text{parallel}} \approx \frac{K}{P} \cdot T_{\text{MFG}}
```

---

## Convergence Analysis

### Monte Carlo Error

The Monte Carlo estimator $\bar{u}_K$ has error:
```math
\mathbb{E}[|\bar{u}_K - \mathbb{E}[u^\theta]|^2] = \frac{\text{Var}(u^\theta)}{K}
```

**Standard Error**: $\text{SE} = \frac{\sigma_u}{\sqrt{K}}$ where $\sigma_u^2 = \text{Var}(u^\theta)$

**Confidence Interval** (95%):
```math
\bar{u}_K \pm 1.96 \cdot \frac{\sigma_u}{\sqrt{K}}
```

### Convergence in Number of Samples

**Weak Convergence**: As $K \to \infty$,
```math
\bar{u}_K \xrightarrow{L^2} \mathbb{E}[u^\theta]
```

with rate $\mathcal{O}(K^{-1/2})$ for standard MC, improved to $\mathcal{O}((\log K)^d/K)$ for QMC.

### Spatial Discretization Error

Each conditional MFG has discretization error $\mathcal{O}(h^p)$ where:
- $h = \Delta x$ is grid spacing
- $p$ is the order of the numerical method (e.g., $p=2$ for central differences)

**Total Error**:
```math
\text{Total Error} \leq \text{MC Error} + \text{Discretization Error} = \mathcal{O}(K^{-1/2}) + \mathcal{O}(h^p)
```

**Optimal Balance**: Choose $K$ and $h$ such that both errors are comparable:
```math
K \sim h^{-2p}
```

---

## Special Case: Linear-Quadratic Common Noise MFG

### Problem Setup

**Hamiltonian**: $H(x, p, m, \theta) = \frac{p^2}{2} + \frac{x^2}{2} + \alpha \theta x$

**Common Noise**: $d\theta_t = -\kappa \theta_t dt + \sigma_\theta dB_t$ (Ornstein-Uhlenbeck)

**Terminal Cost**: $g(x, \theta) = \frac{x^2}{2}$

### Perturbation Solution

For small coupling $\alpha$, expand:
```math
u^\theta(t,x) = u_0(t,x) + \alpha \theta u_1(t,x) + \mathcal{O}(\alpha^2)
```
```math
m^\theta(t,x) = m_0(t,x) + \alpha \theta m_1(t,x) + \mathcal{O}(\alpha^2)
```

where $(u_0, m_0)$ solve the **deterministic LQ-MFG** (setting $\alpha=0$).

**First-Order Corrections** $(u_1, m_1)$ solve linearized equations around $(u_0, m_0)$.

### Analytical Solution (Deterministic Part)

For the deterministic LQ-MFG, the solution has the form:

**Value Function**:
```math
u_0(t, x) = A(t) x^2 + B(t)
```

where $A(t)$ solves the **Riccati ODE**:
```math
\frac{dA}{dt} = -1 + 2A^2, \quad A(T) = \frac{1}{2}
```

**Density**:
```math
m_0(t, x) = \frac{1}{\sqrt{2\pi \sigma_t^2}} \exp\left(-\frac{x^2}{2\sigma_t^2}\right)
```

where $\sigma_t^2$ evolves according to the mean field coupling.

This analytical solution serves as a benchmark for testing the Common Noise MFG solver.

---

## Applications

### 1. Financial Markets with Market Volatility

**Setup**:
- State $x$: Portfolio position
- Common Noise $\theta$: Market volatility (VIX index)
- Hamiltonian: $H = \frac{p^2}{2\theta} + c(x, m)$ (risk-adjusted control cost)

**Interpretation**: When volatility $\theta$ is high, agents reduce trading activity (higher control cost).

### 2. Epidemic Dynamics with Policy Changes

**Setup**:
- State $x$: Infection status
- Common Noise $\theta$: Policy stringency (lockdown level)
- Dynamics: $dx = (infection\_rate(\theta) - recovery\_rate) dt + \sigma dW$

**Interpretation**: Policy changes $\theta_t$ affect transmission rates for entire population.

### 3. Multi-Robot Coordination with Sensor Noise

**Setup**:
- State $x$: Robot position
- Common Noise $\theta$: Shared sensor measurement error
- Hamiltonian: Control cost depends on measurement uncertainty $\theta$

**Interpretation**: All robots observe same noisy environment measurement.

---

## Implementation in MFG_PDE

### Basic Usage

```python
from mfg_pde.core.stochastic import StochasticMFGProblem, OrnsteinUhlenbeckProcess
from mfg_pde.alg.numerical.stochastic import CommonNoiseMFGSolver

# Define common noise process (market volatility)
vix_process = OrnsteinUhlenbeckProcess(
    kappa=2.0,  # Mean reversion speed
    mu=20.0,    # Long-term mean volatility
    sigma=8.0   # Volatility of volatility
)

# Define conditional Hamiltonian
def market_hamiltonian(x, p, m, theta):
    """Hamiltonian depends on VIX level theta."""
    risk_premium = 0.5 * (theta / 20.0) * p**2
    congestion_cost = 0.1 * m
    state_cost = 0.5 * x**2
    return risk_premium + congestion_cost + state_cost

# Create stochastic problem
problem = StochasticMFGProblem(
    xmin=-2.0, xmax=2.0, Nx=101,
    T=1.0, Nt=51,
    noise_process=vix_process,
    conditional_hamiltonian=market_hamiltonian
)

# Solve with common noise
solver = CommonNoiseMFGSolver(
    problem,
    num_noise_samples=100,      # Monte Carlo samples
    variance_reduction=True,     # Use Sobol sequences
    parallel=True,               # Parallel solving
    num_workers=8                # Number of CPU cores
)

result = solver.solve(verbose=True)

# Access results
u_mean = result.u_mean  # Expected value function
m_mean = result.m_mean  # Expected density
u_std = result.u_std    # Uncertainty quantification
mc_error = result.mc_error_u  # Monte Carlo error estimate
```

### Advanced: Custom Conditional Solver

```python
from mfg_pde.factory import create_solver

# Define custom conditional solver
def my_conditional_solver_factory(conditional_problem):
    return create_solver(
        conditional_problem,
        solver_type="weno",  # High-order spatial discretization
        method="tvd_rk3",    # Time integration
        backend="jax",       # GPU acceleration
        device="gpu"
    )

# Use custom solver for each conditional MFG
solver = CommonNoiseMFGSolver(
    problem,
    num_noise_samples=100,
    conditional_solver_factory=my_conditional_solver_factory,
    parallel=True
)

result = solver.solve()
```

---

## Performance Benchmarks

### Expected Performance (1D Problems)

| Configuration | Grid Size | Noise Samples | Time (CPU) | Time (GPU) |
|--------------|-----------|---------------|------------|------------|
| Small        | 51 × 21   | K=20          | ~10s       | ~3s        |
| Medium       | 101 × 51  | K=50          | ~2min      | ~30s       |
| Large        | 201 × 101 | K=100         | ~15min     | ~3min      |

**Parallel Scaling**: Near-linear speedup up to $K$ workers (embarrassingly parallel).

**Variance Reduction**: QMC typically achieves 2-5× effective sample size reduction compared to standard MC.

---

## Best Practices

### 1. Choosing Number of Noise Samples

**Rule of Thumb**: Use $K \geq 50$ for production, $K \geq 100$ for publication.

**Adaptive Strategy**:
```python
# Start with small K
result_k20 = solver.solve(num_noise_samples=20)
result_k50 = solver.solve(num_noise_samples=50)

# Check convergence
error_decrease = result_k20.mc_error_u / result_k50.mc_error_u
if error_decrease < 1.3:  # Less than sqrt(50/20) ≈ 1.58
    print("Need more samples!")
```

### 2. Balancing MC and Spatial Errors

Match discretization error with MC error:
```python
# If discretization error ~ h^2 with h = L/Nx
# and MC error ~ 1/sqrt(K)
# Choose: Nx^2 ~ K for balanced error

K = 100
Nx = int(np.sqrt(K) * 10)  # ≈ 100 for K=100
```

### 3. Debugging Common Noise Problems

**Zero Noise Test**: Verify σ→0 recovers deterministic solution
```python
# Create problem with tiny noise
noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=1e-6)
# Solution should match deterministic MFG
```

**Single Path Test**: Test with K=1 to debug conditional solver
```python
solver = CommonNoiseMFGSolver(problem, num_noise_samples=1, seed=42)
result = solver.solve()
# Should successfully solve one conditional MFG
```

---

## Further Reading

### Theoretical Foundations

1. **Carmona & Delarue (2018)**:
   *Probabilistic Theory of Mean Field Games with Applications (Vols. I & II)*
   Comprehensive treatment of stochastic MFG theory

2. **Cardaliaguet et al. (2019)**:
   *The Master Equation and the Convergence Problem in Mean Field Games*
   Connection between N-player games and MFG limit

3. **Carmona, Fouque, & Sun (2015)**:
   *Mean Field Games and Systemic Risk*
   Financial applications with common noise

### Numerical Methods

4. **Laurière & Tangpi (2022)**:
   *Convergence of large population games to mean field games with interaction through the controls*
   Numerical approximation theory

5. **Ruthotto et al. (2020)**:
   *A machine learning framework for solving high-dimensional mean field game and mean field control problems*
   Neural network approaches for stochastic MFG

---

## Appendix: Mathematical Notation

| Symbol | Description |
|--------|-------------|
| $\theta_t$ | Common noise process |
| $u^\theta(t,x)$ | Conditional value function given noise path $\theta$ |
| $m^\theta(t,x)$ | Conditional density given noise path $\theta$ |
| $\bar{u}(t,x)$ | Expected value function: $\mathbb{E}_\theta[u^\theta]$ |
| $\bar{m}(t,x)$ | Expected density: $\mathbb{E}_\theta[m^\theta]$ |
| $K$ | Number of Monte Carlo samples |
| $H(x,p,m,\theta)$ | Conditional Hamiltonian |
| $\sigma$ | Idiosyncratic noise (agent-specific) |
| $\sigma_\theta$ | Common noise volatility |

---

**Document Version**: 1.0
**Last Updated**: October 2025
**Implementation**: `mfg_pde.alg.numerical.stochastic.CommonNoiseMFGSolver`
