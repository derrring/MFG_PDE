# Stochastic Processes and Functional Calculus for MFG

**Date**: 2025-10-05
**Status**: Foundation modules for Phase 2.2 (Stochastic MFG Extensions)
**Related Issue**: #68

---

## Overview

This document describes two foundational modules for stochastic Mean Field Games:

1. **Noise Processes** (`mfg_pde/core/stochastic/noise_processes.py`)
   - Common noise processes for stochastic MFG
   - Used to model external uncertainty affecting all agents

2. **Functional Calculus** (`mfg_pde/utils/functional_calculus.py`)
   - Functional derivative computation for Master Equation
   - Numerical methods for infinite-dimensional calculus

---

## Noise Processes

### Purpose

Common noise processes model external uncertainty that affects all agents simultaneously:
- **Finance**: Market volatility (VIX), interest rates
- **Epidemiology**: Random infection events, policy changes
- **Robotics**: Shared sensor measurements with noise

### Implemented Processes

#### 1. Ornstein-Uhlenbeck (OU) Process

**SDE**:
```
dθ_t = κ(μ - θ_t) dt + σ dW_t
```

**Properties**:
- Mean-reverting: Drifts toward long-term mean μ
- Speed of reversion: κ (kappa)
- Volatility: σ
- Stationary distribution: Normal(μ, σ²/(2κ))

**Applications**: Interest rates, commodity prices, temperature

```python
from mfg_pde.core.stochastic import OrnsteinUhlenbeckProcess

ou_process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.3)
noise_path = ou_process.sample_path(
    theta0=0.5,  # Initial value
    T=1.0,       # Time horizon
    Nt=100,      # Time steps
    seed=42
)
```

#### 2. Cox-Ingersoll-Ross (CIR) Process

**SDE**:
```
dθ_t = κ(μ - θ_t) dt + σ√θ_t dW_t
```

**Properties**:
- Mean-reverting with state-dependent volatility
- Always non-negative (if Feller condition holds: 2κμ ≥ σ²)
- Volatility proportional to √θ
- Stationary distribution: Gamma-like

**Applications**: Interest rates, variance processes (Heston model)

```python
from mfg_pde.core.stochastic import CoxIngersollRossProcess

cir_process = CoxIngersollRossProcess(kappa=1.5, mu=0.04, sigma=0.2)
noise_path = cir_process.sample_path(
    theta0=0.04,  # Initial value (must be > 0)
    T=1.0,
    Nt=100,
    seed=42
)
```

**Feller Condition**: If 2κμ < σ², the process may hit zero. The implementation warns but still computes.

#### 3. Geometric Brownian Motion (GBM)

**SDE**:
```
dθ_t = μθ_t dt + σθ_t dW_t
```

**Properties**:
- Exponential growth with drift μ
- Volatility proportional to current value
- Always positive (if θ₀ > 0)
- Log-normal distribution

**Applications**: Stock prices, asset returns, multiplicative noise

```python
from mfg_pde.core.stochastic import GeometricBrownianMotion

gbm_process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
noise_path = gbm_process.sample_path(
    theta0=1.0,  # Initial value (must be > 0)
    T=1.0,
    Nt=100,
    seed=42
)
```

#### 4. Jump Diffusion Process

**SDE**:
```
dθ_t = μ dt + σ dW_t + J_t dN_t
```

where:
- μ: Drift
- σ: Diffusion coefficient
- J_t: Jump size (Normal(μ_jump, σ_jump²))
- dN_t: Poisson process with intensity λ

**Properties**:
- Combines continuous Brownian motion with discrete jumps
- Jump times: Poisson(λ)
- Jump sizes: Normal(μ_jump, σ_jump²)

**Applications**: Stock crashes, epidemic outbreaks, rare events

```python
from mfg_pde.core.stochastic import JumpDiffusionProcess

jump_process = JumpDiffusionProcess(
    mu=0.1,           # Drift
    sigma=0.2,        # Diffusion
    jump_intensity=5.0,     # Average 5 jumps per unit time
    jump_mean=-0.05,        # Negative jumps (crashes)
    jump_std=0.1
)
noise_path = jump_process.sample_path(
    theta0=0.0,
    T=1.0,
    Nt=100,
    seed=42
)
```

### Base Class: `NoiseProcess`

All processes inherit from the abstract base class:

```python
class NoiseProcess(ABC):
    @abstractmethod
    def drift(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Drift term μ(θ, t)"""
        pass

    @abstractmethod
    def diffusion(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Diffusion term σ(θ, t)"""
        pass

    def sample_path(
        self,
        theta0: float,
        T: float,
        Nt: int,
        seed: int | None = None
    ) -> np.ndarray:
        """Sample a path using Euler-Maruyama method"""
        # Implementation provided by base class
```

### Tests

**Coverage**: 32 unit tests (all passing ✅)

Tests cover:
- Initialization and parameter validation
- Drift and diffusion formulas
- Sample path properties (shape, initial conditions, reproducibility)
- Process-specific properties (mean reversion, positivity, jumps)

---

## Functional Calculus

### Purpose

Functional derivatives are needed for the **Master Equation** formulation of MFG:

```
Master Equation:
∂U/∂t + H(x, ∇_x U, δU/δm, m) = 0
```

where:
- U: [0,T] × ℝ^d × P(ℝ^d) → ℝ (value function on measure space)
- δU/δm: Functional derivative (linear derivative)

### Functional Derivative Definition

For a functional U[m], the functional derivative δU/δm at measure m in direction y is:

```
δU/δm[m](x,y) = lim_{ε→0} (U[m + εδ_y] - U[m]) / ε
```

where δ_y is the Dirac measure at y.

### Implemented Methods

#### 1. Finite Difference Functional Derivative

**Method**: Approximate functional derivative using finite differences on measure perturbations

```python
from mfg_pde.utils.functional_calculus import FiniteDifferenceFunctionalDerivative

# Define functional U[m]
def my_functional(measure):
    """U[m] = ∫ m(x)² dx (quadratic functional)"""
    return np.sum(measure**2)

# Create derivative operator
deriv_op = FiniteDifferenceFunctionalDerivative(
    epsilon=1e-4,          # Perturbation size
    method="central"       # central, forward, or backward difference
)

# Compute derivative
measure = np.array([0.1, 0.3, 0.4, 0.2])
derivative = deriv_op(my_functional, measure, y_index=1)
# Returns δU/δm[m](x=1) ≈ 2·m(1) = 0.6
```

**Schemes**:
- **Forward**: (U[m + εδ_y] - U[m]) / ε
- **Backward**: (U[m] - U[m - εδ_y]) / ε
- **Central**: (U[m + εδ_y] - U[m - εδ_y]) / (2ε) (more accurate)

#### 2. Particle Approximation Functional Derivative

**Method**: Represent measure as particle system and compute derivative

```python
from mfg_pde.utils.functional_calculus import (
    ParticleApproximationFunctionalDerivative,
    create_particle_measure
)

# Create particle measure
particles, weights = create_particle_measure(
    domain_bounds=(0, 1),
    num_particles=100,
    method="sobol"  # sobol, uniform, or random
)

# Create derivative operator
deriv_op = ParticleApproximationFunctionalDerivative(
    particles=particles,
    weights=weights
)

# Compute derivative
derivative = deriv_op(my_functional, weights, y_index=50)
```

**Particle Measures**:
- **Uniform grid**: Evenly spaced particles
- **Random sampling**: Monte Carlo particles
- **Quasi-Monte Carlo**: Sobol sequence for better coverage

### Validation and Testing

#### Verify Accuracy

```python
from mfg_pde.utils.functional_calculus import verify_functional_derivative_accuracy

def functional(m):
    return np.sum(m**2)

def analytical_derivative(m, y_index):
    return 2 * m[y_index]

errors = verify_functional_derivative_accuracy(
    domain_bounds=(0, 1),
    num_particles=50,
    functional=functional,
    analytical_derivative=analytical_derivative
)

print(f"Max error: {errors['max_error']:.6e}")
print(f"Mean error: {errors['mean_error']:.6e}")
```

### Tests

**Coverage**: 14 unit tests (all passing ✅)

Tests cover:
- Linear and quadratic functionals
- Finite difference schemes (forward, backward, central)
- Second-order derivatives
- Particle measure creation (uniform, random, Sobol)
- Accuracy verification against analytical derivatives

---

## Integration with MFG

### Common Noise MFG

**System**:
```
HJB: ∂u/∂t + H(x, ∇u, m^θ, θ_t) = 0
FP:  ∂m^θ/∂t - div(m^θ ∇_p H) - Δm^θ = 0
Noise: θ_t ~ NoiseProcess
```

**Usage** (future Common Noise MFG Solver):
```python
from mfg_pde.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfg_pde.alg.numerical.stochastic import CommonNoiseMFGSolver

# Define problem with common noise
noise_process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.0, sigma=0.3)
problem = StochasticMFGProblem(
    ...
    noise_process=noise_process
)

# Solve with Monte Carlo over noise realizations
solver = CommonNoiseMFGSolver(
    problem=problem,
    num_noise_samples=100,
    variance_reduction=True  # Use quasi-Monte Carlo
)

result = solver.solve()
```

### Master Equation

**System**:
```
Master Equation: ∂U/∂t + H(x, ∇_x U, δU/δm, m) = 0
```

**Usage** (future Master Equation Solver):
```python
from mfg_pde.utils.functional_calculus import FiniteDifferenceFunctionalDerivative
from mfg_pde.alg.numerical.stochastic import MasterEquationSolver

# Solve Master Equation
solver = MasterEquationSolver(
    problem=problem,
    derivative_method=FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
)

result = solver.solve()
```

---

## Mathematical References

1. **Carmona, R., & Delarue, F. (2018)**. *Probabilistic Theory of Mean Field Games*.
   - Chapter 5: Common noise and conditional McKean-Vlasov equations
   - Chapter 6: Master equation and its properties

2. **Cardaliaguet, P., Delarue, F., Lasry, J.-M., & Lions, P.-L. (2019)**.
   *The Master Equation and the Convergence Problem in Mean Field Games*. Princeton University Press.
   - Rigorous treatment of functional derivatives in MFG
   - Convergence of N-player games to Master Equation

3. **Lions, P.-L. (Collège de France Lectures, 2007-2011)**.
   Mean Field Games course notes.
   - Original development of Master Equation theory

---

## File Locations

```
mfg_pde/
├── core/
│   └── stochastic/
│       ├── __init__.py
│       └── noise_processes.py           # 531 lines
└── utils/
    └── functional_calculus.py           # 532 lines

tests/
└── unit/
    ├── test_noise_processes.py          # 381 lines, 32 tests
    └── test_functional_calculus.py      # 283 lines, 14 tests
```

**Total**: 1,727 lines of production code + tests

---

## Next Steps (Phase 2.2 Continuation)

With these foundation modules in place, the next components for Phase 2.2 are:

1. **StochasticMFGProblem class** - Problem definition with common noise
2. **CommonNoiseMFGSolver** - Monte Carlo solver for conditional MFG
3. **MasterEquationSolver** - Solver using functional calculus
4. **Integration tests** - End-to-end validation with analytical solutions

**Current Status**: Foundation complete ✅ (Week 1-2 equivalent from Issue #68 plan)

---

**Document Status**: ✅ COMPLETE
**Code Status**: ✅ TESTED (46 unit tests passing)
**Integration Status**: Ready for Phase 2.2 solver development
