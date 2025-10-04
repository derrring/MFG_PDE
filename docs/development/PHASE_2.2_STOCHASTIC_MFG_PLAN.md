# Phase 2.2: Stochastic MFG Extensions Implementation Plan

**Created**: 2025-10-04
**Priority**: HIGH - Research differentiation and theoretical advancement
**Timeline**: 12 weeks (Q1 2026 target)
**Dependencies**: Phase 1 (Neural) ✅ Complete, Phase 2.1 (3D WENO) ✅ Complete

---

## 🎯 Executive Summary

Implement advanced stochastic formulations for Mean Field Games, enabling modeling of uncertain environments and common noise scenarios. This positions MFG_PDE at the research frontier by supporting both Common Noise MFG and Master Equation formulations.

**Strategic Impact**:
- **Research Leadership**: First comprehensive framework for stochastic MFG
- **Practical Applications**: Finance (market uncertainty), epidemiology (random events), robotics (sensor noise)
- **Theoretical Advancement**: Master equation implementation enables cutting-edge research

---

## 📚 Mathematical Background

### Common Noise MFG Framework

**System of Equations**:
```
HJB with Common Noise:
∂u/∂t + H(x, ∇u, m_t^θ, θ_t) + σ_u(θ_t) · ∇u · dW_t = 0

Fokker-Planck with Conditional Dynamics:
∂m^θ/∂t - div(m^θ ∇_p H(x, ∇u, m^θ, θ)) - Δm^θ + σ_m(θ_t) · noise_terms = 0

Common Noise Process:
dθ_t = μ(θ_t) dt + σ_θ dB_t
```

**Key Features**:
- **Common Information**: All agents observe θ_t (e.g., market index, epidemic level)
- **Conditional Strategies**: Agent strategies depend on noise realization
- **Stochastic Environment**: External randomness affects population dynamics

**Applications**:
- **Finance**: Market uncertainty (VIX, interest rates)
- **Epidemiology**: Random infection events, policy changes
- **Robotics**: Shared sensor measurements with noise

### Master Equation Formulation

**Functional PDE**:
```
Master Equation:
∂U/∂t + H(x, ∇_x U, δU/δm, m) + σ^2/2 Δ_x U + ∫ L[δU/δm] m(dy) = 0

Functional Derivative (Linear Derivative):
δU/δm[m](x,y) = lim_{ε→0} (U[m + εδ_y] - U[m]) / ε

Lift Operator:
L[δU/δm](x,y) = -div_y(m(y) ∇_p H(y, δU/δm(x,y)))
```

**Key Concepts**:
- **Value Function on Measure Space**: U: [0,T] × ℝ^d × P(ℝ^d) → ℝ
- **Functional Calculus**: First-order (δU/δm) and second-order (δ²U/δm²) derivatives
- **Infinite-Dimensional PDE**: Evolution on probability measure space

**Mathematical Challenges**:
- Efficient computation of functional derivatives
- Numerical approximation on measure space
- Stability and convergence analysis

---

## 🏗️ Implementation Architecture

### Module Organization

```
mfg_pde/alg/
├── numerical/
│   ├── stochastic/                    # NEW: Stochastic MFG methods
│   │   ├── __init__.py
│   │   ├── common_noise_solver.py    # Common Noise MFG solver
│   │   ├── master_equation_solver.py  # Master equation solver
│   │   └── stochastic_fp_solver.py   # Stochastic Fokker-Planck
│   └── ...
├── neural/
│   └── stochastic/                    # NEW: Neural stochastic methods
│       ├── __init__.py
│       └── stochastic_pinn.py        # PINNs for stochastic MFG
└── ...

mfg_pde/core/
├── stochastic_problem.py              # NEW: Stochastic MFG problem class
└── noise_processes.py                 # NEW: Common noise processes

mfg_pde/utils/
├── functional_calculus.py             # NEW: Functional derivative computation
└── monte_carlo.py                     # ENHANCED: Stochastic integration
```

### Class Hierarchy

```python
# Stochastic MFG Problem Specification
class StochasticMFGProblem(MFGProblem):
    """Base class for stochastic MFG problems."""

    def common_noise_process(self) -> NoiseProcess:
        """Define common noise process θ_t."""

    def conditional_hamiltonian(self, x, p, m_theta, theta):
        """Hamiltonian conditioned on common noise."""

    def noise_coupling(self, theta):
        """Noise coupling terms σ(θ)."""

# Common Noise Process
class NoiseProcess(Protocol):
    """Protocol for common noise processes."""

    def drift(self, theta: Array, t: float) -> Array:
        """Drift term μ(θ,t)."""

    def diffusion(self, theta: Array, t: float) -> Array:
        """Diffusion term σ_θ(θ,t)."""

    def sample_path(self, T: float, Nt: int) -> Array:
        """Generate sample path of θ_t."""
```

---

## 🔬 Implementation Components

### 1. Common Noise MFG Solver

**Algorithm**: Monte Carlo over noise realizations

```python
class CommonNoiseMFGSolver(BaseMFGSolver):
    """
    Solve Common Noise MFG via Monte Carlo over noise realizations.

    Algorithm:
    1. Generate K sample paths of common noise θ_t
    2. For each path k:
       a. Solve conditional HJB: u^k(t,x,θ^k_t)
       b. Solve conditional FP: m^k(t,x,θ^k_t)
    3. Aggregate solutions via Monte Carlo average
    """

    def __init__(
        self,
        problem: StochasticMFGProblem,
        num_noise_samples: int = 100,
        hjb_solver: str = "weno",
        fp_solver: str = "upwind",
        variance_reduction: bool = True,
    ):
        self.problem = problem
        self.K = num_noise_samples
        self.hjb_solver = create_solver(hjb_solver)
        self.fp_solver = create_solver(fp_solver)
        self.variance_reduction = variance_reduction

    def solve(self) -> StochasticMFGResult:
        """
        Solve stochastic MFG with common noise.

        Returns:
            Solution with u^θ, m^θ for each noise realization
        """
        # 1. Sample common noise paths
        noise_paths = self._sample_noise_paths()

        # 2. Solve conditional MFG for each path
        conditional_solutions = []
        for k in range(self.K):
            theta_k = noise_paths[k]

            # Create conditional problem
            conditional_problem = self._create_conditional_problem(theta_k)

            # Solve conditional MFG
            u_k, m_k = self._solve_conditional_mfg(conditional_problem)
            conditional_solutions.append((u_k, m_k, theta_k))

        # 3. Aggregate via Monte Carlo
        result = self._aggregate_solutions(conditional_solutions)

        return result

    def _sample_noise_paths(self) -> List[Array]:
        """Sample K paths of common noise θ_t."""
        noise_process = self.problem.common_noise_process()

        if self.variance_reduction:
            # Use quasi-Monte Carlo for variance reduction
            paths = self._quasi_monte_carlo_paths(noise_process)
        else:
            # Standard Monte Carlo sampling
            paths = [noise_process.sample_path(self.problem.T, self.problem.Nt)
                    for _ in range(self.K)]

        return paths

    def _solve_conditional_mfg(self, problem) -> Tuple[Array, Array]:
        """Solve conditional MFG for given noise path."""
        # Fixed-point iteration for conditional equilibrium
        m = problem.rho0  # Initial density

        for iteration in range(self.max_iterations):
            # Solve conditional HJB backward
            u = self.hjb_solver.solve_backward(problem, m)

            # Solve conditional FP forward
            m_new = self.fp_solver.solve_forward(problem, u)

            # Check convergence
            if self._check_convergence(m, m_new):
                break

            m = m_new

        return u, m
```

**Key Technical Challenges**:
- **Variance Reduction**: Quasi-Monte Carlo, control variates
- **Computational Cost**: K independent MFG solves (parallelizable)
- **Convergence**: Monte Carlo error + MFG fixed-point error

**Performance Optimization**:
- Parallel solve across noise realizations (embarrassingly parallel)
- GPU acceleration for each conditional MFG
- Adaptive variance reduction based on error estimates

---

### 2. Master Equation Solver

**Algorithm**: Finite-dimensional approximation via particles

```python
class MasterEquationSolver(BaseMFGSolver):
    """
    Solve Master Equation via finite-dimensional approximation.

    Mathematical Approach:
    1. Approximate measure space P(ℝ^d) with N-particle system
    2. Compute functional derivatives via finite differences
    3. Solve resulting high-dimensional PDE
    """

    def __init__(
        self,
        problem: MFGProblem,
        num_particles: int = 50,
        derivative_method: str = "finite_difference",
        solver_type: str = "neural",
    ):
        self.problem = problem
        self.N = num_particles  # Dimension of approximation
        self.derivative_method = derivative_method
        self.solver_type = solver_type

    def solve(self) -> MasterEquationResult:
        """
        Solve Master Equation U(t, x, m).

        Returns:
            Value function U and functional derivatives δU/δm
        """
        # 1. Discretize measure space
        particle_grid = self._create_particle_grid()

        # 2. Approximate functional derivatives
        derivative_operator = self._build_derivative_operator()

        # 3. Solve high-dimensional PDE
        if self.solver_type == "neural":
            # Use PINNs for high-dimensional master equation
            result = self._solve_neural(particle_grid, derivative_operator)
        else:
            # Use finite difference for low dimensions
            result = self._solve_finite_difference(particle_grid, derivative_operator)

        return result

    def _build_derivative_operator(self) -> FunctionalDerivative:
        """
        Build functional derivative operator δU/δm.

        Finite Difference Approximation:
        δU/δm[m](x,y) ≈ (U[m + εδ_y] - U[m]) / ε
        """
        if self.derivative_method == "finite_difference":
            return FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
        elif self.derivative_method == "automatic":
            # Use automatic differentiation (JAX/PyTorch)
            return AutomaticFunctionalDerivative()
        else:
            raise ValueError(f"Unknown derivative method: {self.derivative_method}")

    def _solve_neural(self, particle_grid, derivative_op) -> MasterEquationResult:
        """
        Solve master equation using PINNs.

        Network Architecture:
        U_θ: (t, x, m) → ℝ where m ∈ ℝ^{N×d}
        """
        from mfg_pde.neural.stochastic import MasterEquationPINN

        # Create neural network for U(t, x, m)
        network = MasterEquationPINN(
            spatial_dim=self.problem.spatial_dim,
            particle_dim=self.N,
            hidden_layers=[128, 128, 128],
        )

        # Train network to satisfy master equation
        result = network.train(
            derivative_operator=derivative_op,
            problem=self.problem,
            num_epochs=10000,
        )

        return result
```

**Key Technical Challenges**:
- **Curse of Dimensionality**: Measure space approximation requires high dimensions
- **Functional Derivatives**: Efficient computation of δU/δm
- **Numerical Stability**: High-dimensional PDE stability

**Research Innovation**:
- **Neural Approximation**: PINNs naturally handle high-dimensional spaces
- **Particle Representation**: Finite-dimensional measure approximation
- **Automatic Differentiation**: Efficient functional derivative computation

---

### 3. Noise Process Library

**Common Noise Processes**:

```python
class OrnsteinUhlenbeckProcess(NoiseProcess):
    """
    Ornstein-Uhlenbeck mean-reverting process.

    dθ_t = κ(μ - θ_t) dt + σ dB_t

    Applications: Interest rates, volatility (VIX)
    """

    def __init__(self, kappa: float, mu: float, sigma: float):
        self.kappa = kappa  # Mean reversion speed
        self.mu = mu        # Long-term mean
        self.sigma = sigma  # Volatility

class CoxIngersollRossProcess(NoiseProcess):
    """
    Cox-Ingersoll-Ross process (always positive).

    dθ_t = κ(μ - θ_t) dt + σ√θ_t dB_t

    Applications: Interest rates, epidemic intensity
    """

class GeometricBrownianMotion(NoiseProcess):
    """
    Geometric Brownian Motion (stock prices).

    dθ_t = μ θ_t dt + σ θ_t dB_t

    Applications: Market indices, asset prices
    """

class JumpDiffusionProcess(NoiseProcess):
    """
    Jump diffusion process (Merton model).

    dθ_t = μ dt + σ dB_t + J_t dN_t

    Applications: Market crashes, sudden events
    """
```

---

## 🧪 Testing Strategy

### Unit Tests

```python
# tests/unit/test_stochastic_processes.py
def test_ornstein_uhlenbeck_mean_reversion():
    """Verify OU process converges to long-term mean."""
    process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)
    path = process.sample_path(T=10.0, Nt=1000)

    # Check long-term convergence
    assert np.abs(path[-100:].mean() - 0.5) < 0.1

def test_functional_derivative_finite_difference():
    """Verify functional derivative accuracy."""
    # Simple test functional: U[m](x) = ∫ m(y) K(x,y) dy
    derivative_op = FiniteDifferenceFunctionalDerivative(epsilon=1e-6)

    # Analytical derivative: δU/δm(x,y) = K(x,y)
    # Test against known solution
    ...
```

### Integration Tests

```python
# tests/integration/test_common_noise_mfg.py
def test_common_noise_convergence():
    """Test common noise MFG solver convergence."""
    # Create simple problem with analytical solution
    problem = create_linear_quadratic_common_noise_problem()

    solver = CommonNoiseMFGSolver(
        problem,
        num_noise_samples=100,
        hjb_solver="weno",
        fp_solver="upwind",
    )

    result = solver.solve()

    # Compare with analytical solution
    error = compute_solution_error(result, analytical_solution)
    assert error < 1e-3

# tests/integration/test_master_equation.py
def test_master_equation_low_dimensional():
    """Test master equation solver on 1D problem."""
    problem = create_simple_master_equation_problem()

    solver = MasterEquationSolver(
        problem,
        num_particles=20,
        solver_type="finite_difference",
    )

    result = solver.solve()

    # Verify solution satisfies master equation PDE
    residual = compute_master_equation_residual(result)
    assert residual < 1e-4
```

---

## 📊 Validation and Benchmarks

### Analytical Test Cases

**1. Linear-Quadratic Common Noise MFG**:
- Quadratic Hamiltonian with Gaussian noise
- Closed-form solution available
- Validate numerical convergence rates

**2. Master Equation with Independent Agents**:
- Special case reduces to standard MFG
- Compare with existing HJB-FP solvers
- Verify functional derivative correctness

### Performance Benchmarks

**Common Noise MFG**:
- Weak convergence rate in K (num_noise_samples)
- Strong convergence rate in spatial discretization
- Parallel efficiency across noise realizations

**Master Equation**:
- Approximation error vs. num_particles
- Neural solver scaling to high dimensions (d > 5)
- Comparison with particle method approximations

---

## 📚 Documentation Plan

### Theoretical Documentation

**File**: `docs/theory/stochastic_mfg_mathematical_formulation.md`

**Contents**:
- Mathematical foundations of common noise MFG
- Master equation derivation and theory
- Functional calculus primer
- Convergence analysis and error estimates
- References to key papers (Carmona & Delarue, Cardaliaguet et al.)

### User Guide

**File**: `docs/user/stochastic_mfg_guide.md`

**Contents**:
- Introduction to stochastic MFG applications
- Common noise process library usage
- Solver selection guide (common noise vs master equation)
- Example problems (finance, epidemiology)
- Performance tuning recommendations

### Examples

```python
# examples/advanced/common_noise_portfolio_optimization.py
"""
Portfolio optimization under market uncertainty.

Common noise: VIX index (volatility)
Agents: Investors with risk aversion
Equilibrium: Nash equilibrium under common information
"""

# examples/advanced/master_equation_crowd_dynamics.py
"""
Crowd dynamics with master equation formulation.

Demonstrates functional derivative computation and
high-dimensional neural solver for master equation.
"""
```

---

## 🗓️ Implementation Timeline

### Week 1-2: Foundation (Infrastructure)
- [ ] Create `stochastic/` directory structure
- [ ] Implement `StochasticMFGProblem` base class
- [ ] Build noise process library (OU, CIR, GBM, Jump)
- [ ] Unit tests for noise processes

### Week 3-4: Common Noise Solver (Core Algorithm)
- [ ] Implement `CommonNoiseMFGSolver`
- [ ] Monte Carlo sampling infrastructure
- [ ] Variance reduction techniques (QMC, control variates)
- [ ] Parallel execution across noise realizations

### Week 5-6: Master Equation Foundation
- [ ] Functional derivative operators
- [ ] Finite difference approximation
- [ ] Automatic differentiation integration (JAX/PyTorch)
- [ ] Unit tests for functional calculus

### Week 7-8: Master Equation Solver
- [ ] Implement `MasterEquationSolver`
- [ ] Finite difference solver (low-d)
- [ ] Neural PINN solver (high-d)
- [ ] Particle representation of measures

### Week 9-10: Testing and Validation
- [ ] Analytical test cases (LQ-MFG, independent agents)
- [ ] Integration tests for both solvers
- [ ] Convergence rate verification
- [ ] Performance benchmarks

### Week 11-12: Documentation and Examples
- [ ] Theoretical documentation (mathematical formulation)
- [ ] User guide and tutorials
- [ ] Working examples (finance, epidemiology)
- [ ] API documentation and docstrings

---

## 🎯 Success Metrics

### Functional Requirements
- [ ] Common Noise MFG solver passes analytical test cases (error < 1e-3)
- [ ] Master Equation solver handles d=5 dimensional problems
- [ ] Noise process library supports 4+ standard processes
- [ ] Functional derivative computation accurate (error < 1e-6)

### Performance Requirements
- [ ] Common Noise: K=100 samples solves in <5 minutes (1D, GPU)
- [ ] Master Equation: N=50 particles solves in <10 minutes (1D, neural)
- [ ] Parallel efficiency >80% for common noise solver
- [ ] Neural master equation scales to d=10 dimensions

### Research Impact
- [ ] First comprehensive stochastic MFG framework in open source
- [ ] Enables new research in uncertain environments
- [ ] Publication-quality implementations with theoretical rigor
- [ ] Working examples demonstrate practical applications

---

## 🔗 Integration with Existing Framework

### Factory Integration

```python
# mfg_pde/factory/solver_factory.py
def create_solver(problem, **kwargs):
    """Create solver with automatic type detection."""

    if isinstance(problem, StochasticMFGProblem):
        if problem.has_common_noise():
            return CommonNoiseMFGSolver(problem, **kwargs)
        else:
            return MasterEquationSolver(problem, **kwargs)
    else:
        # Existing deterministic solver creation
        ...
```

### Configuration System

```python
# mfg_pde/config/solver_config.py
@dataclass
class CommonNoiseConfig(SolverConfig):
    """Configuration for common noise MFG solver."""
    num_noise_samples: int = 100
    variance_reduction: bool = True
    hjb_solver: str = "weno"
    fp_solver: str = "upwind"
    parallel: bool = True

@dataclass
class MasterEquationConfig(SolverConfig):
    """Configuration for master equation solver."""
    num_particles: int = 50
    derivative_method: str = "finite_difference"
    solver_type: str = "neural"
    network_depth: int = 4
```

---

## 🚀 Future Extensions

### Phase 2.2+: Advanced Features

**Second-Order Master Equation**:
- Second-order functional derivatives δ²U/δm²
- Lions derivative and intrinsic derivatives
- Enhanced stability for non-convex problems

**Mean Field Control**:
- Centralized control vs. decentralized (MFG)
- Optimal control of population distributions
- Applications in epidemic control, crowd management

**Time-Inconsistent MFG**:
- Non-exponential discounting
- Hyperbolic preferences
- Behavioral economics applications

**Multi-Population Stochastic MFG**:
- Heterogeneous agents with common noise
- Network-based common information
- Systemic risk in financial networks

---

## 📖 References

### Key Papers

1. **Carmona, R., & Delarue, F.** (2018). *Probabilistic Theory of Mean Field Games with Applications* (Vols. I & II). Springer.
   - Comprehensive treatment of stochastic MFG theory
   - Common noise formulation and master equation

2. **Cardaliaguet, P., Delarue, F., Lasry, J.-M., & Lions, P.-L.** (2019). *The Master Equation and the Convergence Problem in Mean Field Games*. Princeton University Press.
   - Master equation mathematical foundations
   - Convergence of N-player games to MFG

3. **Carmona, R., Fouque, J.-P., & Sun, L.-H.** (2015). *Mean Field Games and Systemic Risk*. Communications in Mathematical Sciences.
   - Financial applications with common noise
   - Systemic risk modeling

4. **Gangbo, W., & Święch, A.** (2015). *Existence of a solution to an equation arising from mean field games*. Journal of Differential Equations.
   - Master equation well-posedness theory
   - Functional derivative formulations

### Implementation References

- **DeepXDE**: Physics-informed neural networks for PDEs
- **JAX Functional Derivatives**: Automatic differentiation for functionals
- **Quasi-Monte Carlo**: Variance reduction techniques

---

## ✅ Approval Checklist

Before starting implementation:

- [ ] Review mathematical formulation with domain expert
- [ ] Validate timeline with current development capacity
- [ ] Ensure compatibility with existing neural/numerical paradigms
- [ ] Confirm testing strategy is comprehensive
- [ ] Verify documentation plan meets research standards

---

**Status**: 🔄 **PLANNING PHASE**
**Next Action**: Review and approval for implementation start
**Owner**: Development team
**Estimated Completion**: Q1 2026 (12 weeks from start)

---

**Related Documents**:
- `docs/development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` - Overall roadmap
- `docs/theory/mean_field_games_mathematical_formulation.md` - MFG theory foundation
- `docs/development/PHASE_1_NEURAL_IMPLEMENTATION_SUMMARY.md` - Neural paradigm reference
