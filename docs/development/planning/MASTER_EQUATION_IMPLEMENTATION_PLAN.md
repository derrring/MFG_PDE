# Master Equation Implementation Plan

**Document Version**: 1.0
**Created**: October 5, 2025
**Status**: Planning Phase
**Priority**: MEDIUM (Deferred from Phase 2.2)
**Estimated Effort**: 4-6 weeks

## 🎯 Executive Summary

This document outlines the implementation plan for Mean Field Game Master Equation solver, building on existing infrastructure from Phase 2.2 (Common Noise MFG). The Master Equation provides a powerful functional PDE framework for MFG analysis and enables cutting-edge research applications.

**Strategic Value**:
- **Research Leadership**: First open-source Master Equation MFG implementation
- **Theoretical Foundation**: Enables rigorous analysis of MFG convergence
- **High-Dimensional Capability**: Natural framework for neural methods

## 📚 Mathematical Foundation

### Master Equation Formulation

The Master Equation describes the value function as a functional of the population distribution:

```
∂U/∂t[m](t,x) + H(x, ∇ₓU[m](t,x), δU/δm[m](t,x,·), m(t,·)) = 0

U[m](T,x) = g(x, m(T,·))
```

Where:
- `U[m](t,x)`: Value function as functional of measure m
- `δU/δm[m](t,x,y)`: Functional derivative (Lion's derivative)
- `H`: Hamiltonian depending on functional derivative
- `g`: Terminal cost functional

**Key Challenge**: Efficient computation of functional derivatives `δU/δm`

### Functional Derivative

For smooth test functions φ:
```
∫ φ(y) δU/δm[m](t,x,y) dm(y) = lim_{ε→0} 1/ε [U[m + εδᵧ](t,x) - U[m](t,x)]
```

**Numerical Approaches**:
1. **Finite Difference**: Direct discretization
2. **Particle Approximation**: Empirical measures
3. **Neural Networks**: Automatic differentiation

## 🏗️ Implementation Architecture

### Module Structure (extends Phase 2.2)

```
mfg_pde/
├── alg/numerical/stochastic/
│   ├── common_noise_solver.py          ✅ EXISTS
│   ├── master_equation_solver.py       ⬜ TO IMPLEMENT
│   └── stochastic_fp_solver.py         ⬜ TO IMPLEMENT
├── alg/neural/stochastic/
│   └── master_equation_pinn.py         ⬜ TO IMPLEMENT
├── core/stochastic/
│   ├── noise_processes.py              ✅ EXISTS
│   └── stochastic_problem.py           ✅ EXISTS
├── utils/numerical/
│   └── functional_calculus.py          ✅ EXISTS (foundation)
└── tests/
    ├── unit/test_functional_calculus.py    ✅ EXISTS (14 tests)
    └── integration/test_master_equation.py ⬜ TO IMPLEMENT
```

### Existing Infrastructure ✅

**From Phase 2.2 Common Noise Implementation**:
- ✅ `StochasticMFGProblem` base class
- ✅ Noise process library (4 processes: OU, CIR, GBM, Jump)
- ✅ `CommonNoiseMFGSolver` (reference for structure)
- ✅ Functional calculus utilities:
  - `FiniteDifferenceFunctionalDerivative`
  - `ParticleApproximationFunctionalDerivative`
  - `create_particle_measure`
  - `verify_functional_derivative_accuracy`

**From Neural Paradigm (Phase 1)**:
- ✅ PINN framework (`mfg_pde.alg.neural.nn`)
- ✅ PyTorch integration
- ✅ Automatic differentiation infrastructure

## 📋 Implementation Tasks

### **Week 1-2: Foundation**

#### Task 1.1: Enhance Functional Calculus ⬜
**File**: `mfg_pde/utils/numerical/functional_calculus.py`

**Additions**:
```python
class MasterEquationFunctionalDerivative:
    """
    Specialized functional derivative for Master Equation.

    Combines finite difference and particle methods with
    adaptive accuracy control.
    """

    def compute_derivative(
        self,
        U_functional: Callable,  # U[m](t,x)
        m: NDArray,  # Current measure
        x: NDArray,  # State point
        y_points: NDArray,  # Evaluation points for δU/δm
    ) -> NDArray:
        """Compute δU/δm[m](t,x,y) at specified points."""
        pass

    def compute_hamiltonian_term(
        self,
        U_functional: Callable,
        m: NDArray,
        x: NDArray,
    ) -> float:
        """Compute H(x, ∇U, δU/δm, m) term."""
        pass
```

**Tests**: Extend `test_functional_calculus.py` with Master Equation cases

#### Task 1.2: Master Equation Problem Class ⬜
**File**: `mfg_pde/core/stochastic/stochastic_problem.py`

**Additions**:
```python
class MasterEquationProblem(StochasticMFGProblem):
    """
    Master Equation formulation of MFG.

    Attributes:
        hamiltonian_functional: H(x, p, δU/δm, m)
        terminal_cost_functional: g(x, m)
        particle_representation: If True, use particle methods
    """

    def has_master_equation(self) -> bool:
        """Check if problem uses Master Equation formulation."""
        return True
```

### **Week 3-4: Numerical Solver**

#### Task 2.1: Finite Difference Master Equation Solver ⬜
**File**: `mfg_pde/alg/numerical/stochastic/master_equation_solver.py`

**Core Implementation**:
```python
class MasterEquationSolver(BaseMFGSolver):
    """
    Master Equation solver using finite difference methods.

    Approach:
    1. Discretize measure space (grid or particles)
    2. Compute functional derivatives via finite differences
    3. Solve coupled system iteratively

    Suitable for: Low-dimensional problems (d ≤ 3)
    """

    def __init__(
        self,
        problem: MasterEquationProblem,
        num_particles: int = 50,
        derivative_method: str = "finite_difference",
        hjb_solver: str = "upwind",
        fp_solver: str = "implicit",
    ):
        pass

    def solve(self) -> SolverResult:
        """
        Solve Master Equation MFG system.

        Algorithm:
        1. Initialize measure m⁰
        2. For each iteration k:
           a. Compute U[m^k] solving backward HJB
           b. Compute functional derivative δU/δm
           c. Update measure m^{k+1} solving forward FP
           d. Check convergence
        """
        pass

    def _compute_functional_derivative(
        self,
        U: NDArray,
        m: NDArray,
    ) -> NDArray:
        """Compute δU/δm at all spatial-measure points."""
        pass
```

**Key Methods**:
- `_discretize_measure_space()`: Particle or grid representation
- `_solve_hjb_functional()`: Solve HJB with functional derivative
- `_solve_fp_with_feedback()`: FP with measure-dependent drift
- `_check_convergence()`: Functional norm convergence

#### Task 2.2: Particle Representation Methods ⬜
**File**: `mfg_pde/alg/numerical/stochastic/particle_methods.py`

**Implementation**:
```python
class ParticleMeasureRepresentation:
    """
    Represent population measure as empirical measure.

    m_N = (1/N) Σᵢ δ_{Xⁱ}

    Enables tractable functional derivative computation.
    """

    def __init__(self, num_particles: int, dimension: int):
        self.N = num_particles
        self.particles = None  # (N, dimension)
        self.weights = None    # (N,)

    def compute_empirical_derivative(
        self,
        U_values: NDArray,  # U at particle locations
        particle_index: int,
    ) -> float:
        """
        Compute δU/δm via particle perturbation.

        δU/δm ≈ N * [U[m_N^i] - U[m_N]]
        where m_N^i has weight shifted to particle i
        """
        pass
```

### **Week 5-6: Neural Solver (High-Dimensional)**

#### Task 3.1: PINN-Based Master Equation Solver ⬜
**File**: `mfg_pde/alg/neural/stochastic/master_equation_pinn.py`

**Architecture**:
```python
class MasterEquationPINN(nn.Module):
    """
    Physics-Informed Neural Network for Master Equation.

    Network Architecture:
    - Input: (t, x, m_representation)
    - Output: U[m](t,x)
    - Physics Loss: Master Equation residual

    Suitable for: High-dimensional problems (d > 5)
    """

    def __init__(
        self,
        dimension: int,
        measure_embedding_dim: int = 64,
        hidden_layers: list[int] = [256, 256, 256],
    ):
        super().__init__()

        # Measure encoder (e.g., DeepSets architecture)
        self.measure_encoder = MeasureEmbedding(measure_embedding_dim)

        # Value function network U[m](t,x)
        self.value_network = MLP(
            input_dim=1 + dimension + measure_embedding_dim,
            hidden_layers=hidden_layers,
            output_dim=1,
        )

    def forward(self, t, x, measure_particles):
        """
        Compute U[m](t,x).

        Args:
            t: Time (batch,)
            x: State (batch, dimension)
            measure_particles: Particle representation (batch, N, dimension)

        Returns:
            U values (batch,)
        """
        # Encode measure
        m_embed = self.measure_encoder(measure_particles)  # (batch, embed_dim)

        # Concatenate (t, x, m_embed)
        inputs = torch.cat([t.unsqueeze(1), x, m_embed], dim=1)

        # Compute U[m](t,x)
        return self.value_network(inputs).squeeze()

    def compute_functional_derivative(self, t, x, y, measure_particles):
        """
        Compute δU/δm[m](t,x,y) via automatic differentiation.

        Take derivative of U[m + εδ_y] with respect to ε.
        """
        pass


class MeasureEmbedding(nn.Module):
    """
    Permutation-invariant measure embedding (DeepSets).

    φ(m) = ρ(Σᵢ φ(xⁱ))
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.phi = MLP([dimension, 128, 128])
        self.rho = MLP([128, 128, output_dim])

    def forward(self, particles):
        """particles: (batch, N, dimension)"""
        embedded = self.phi(particles)  # (batch, N, 128)
        pooled = embedded.mean(dim=1)   # (batch, 128)
        return self.rho(pooled)         # (batch, output_dim)
```

#### Task 3.2: Training Infrastructure ⬜

```python
class MasterEquationPINNSolver(BaseMFGSolver):
    """
    Master Equation solver using PINNs.

    Handles high-dimensional problems via neural approximation.
    """

    def __init__(
        self,
        problem: MasterEquationProblem,
        network: MasterEquationPINN,
        num_training_samples: int = 10000,
        num_epochs: int = 5000,
        learning_rate: float = 1e-3,
    ):
        pass

    def solve(self) -> SolverResult:
        """
        Train PINN to solve Master Equation.

        Loss = L_master_eq + L_initial + L_boundary
        """
        pass

    def _compute_master_equation_residual(
        self,
        t_samples,
        x_samples,
        measure_samples,
    ):
        """
        Compute Master Equation PDE residual.

        Residual = ∂U/∂t + H(x, ∇U, δU/δm, m)
        """
        pass
```

### **Week 7-8: Testing & Validation**

#### Task 4.1: Analytical Test Cases ⬜
**File**: `tests/integration/test_master_equation.py`

**Test Suite**:
```python
class TestMasterEquationSolver:
    """Test Master Equation solver with analytical solutions."""

    def test_linear_quadratic_master_equation(self):
        """
        Test with LQ-MFG where Master Equation is tractable.

        For linear dynamics and quadratic costs, Master Equation
        has known structure.
        """
        pass

    def test_independent_agents_limit(self):
        """
        Test that Master Equation reduces to classical HJB
        when agents are independent (no coupling).
        """
        pass

    def test_functional_derivative_accuracy(self):
        """
        Verify functional derivative computation against
        analytical test functionals.
        """
        pass

    def test_particle_approximation_convergence(self):
        """
        Test convergence as number of particles N → ∞.
        """
        pass

    def test_neural_vs_finite_difference(self):
        """
        Compare PINN solution to finite difference for
        low-dimensional problem.
        """
        pass
```

#### Task 4.2: Performance Benchmarks ⬜

**Targets**:
- **1D Master Equation**: N=50 particles <10 min (finite difference)
- **2D Master Equation**: N=30 particles <30 min (finite difference)
- **5D Master Equation**: PINN convergence in <2 hours (GPU)
- **10D Master Equation**: PINN feasibility demonstration

### **Week 9-10: Documentation**

#### Task 5.1: Theoretical Documentation ⬜
**File**: `docs/theory/master_equation_mathematical_formulation.md`

**Content**:
- Master Equation derivation
- Functional derivative theory
- Convergence analysis
- Relationship to McKean-Vlasov limit
- Key theoretical results and references

#### Task 5.2: User Guide ⬜
**File**: `docs/user/master_equation_guide.md`

**Content**:
- Quick start examples
- Choosing between finite difference and PINN
- Particle number selection guidelines
- Troubleshooting common issues
- Performance optimization tips

#### Task 5.3: Working Examples ⬜
**Files**: `examples/advanced/master_equation_*.py`

**Examples**:
1. `master_equation_crowd_dynamics.py` - 2D crowd flow
2. `master_equation_portfolio_optimization.py` - Financial application
3. `master_equation_high_dimensional.py` - PINN for d=10

## 🎯 Success Metrics

### Functional Requirements
- ✅ Master Equation solver handles d=1,2,3 (finite difference)
- ✅ Master Equation solver handles d≥5 (PINN)
- ✅ Functional derivative accurate to 1e-6 (analytical tests)
- ✅ Particle approximation converges as N → ∞
- ✅ Independent agent limit reduces to classical HJB

### Performance Requirements
- ✅ 1D: N=50 particles <10 minutes
- ✅ 2D: N=30 particles <30 minutes
- ✅ 5D: PINN convergence <2 hours (GPU)
- ✅ 10D: PINN feasibility demonstration

### Research Impact
- ✅ First comprehensive Master Equation MFG framework
- ✅ Publication-quality implementation with theory
- ✅ Working examples demonstrate practical applications
- ✅ Enables cutting-edge MFG research

## 🔗 Integration

### Factory Integration
```python
def create_solver(problem, **kwargs):
    if isinstance(problem, MasterEquationProblem):
        if problem.dimension <= 3:
            return MasterEquationSolver(problem, **kwargs)
        else:
            return MasterEquationPINNSolver(problem, **kwargs)
```

### Configuration
```python
@dataclass
class MasterEquationConfig(SolverConfig):
    num_particles: int = 50
    derivative_method: str = "finite_difference"
    hjb_solver: str = "upwind"
    fp_solver: str = "implicit"
    convergence_tolerance: float = 1e-4
    max_iterations: int = 100
```

## 📖 Key References

1. **Cardaliaguet, P., et al. (2019)**
   "The Master Equation and the Convergence Problem in Mean Field Games"
   *Annals of Mathematics Studies*

2. **Carmona, R., & Delarue, F. (2018)**
   "Probabilistic Theory of Mean Field Games with Applications" (Vol. I & II)
   *Springer*

3. **Lions, P. L. (2011-2012)**
   "College de France Lecture Notes on Mean Field Games"

4. **Gangbo, W., & Swiech, A. (2015)**
   "Existence of a solution to an equation arising from the theory of Mean Field Games"
   *Journal of Differential Equations*

## 🚦 Decision Points

### Go/No-Go Criteria

**Proceed with Implementation if**:
- ✅ Phase 2.2 Common Noise MFG complete (current status: ✅)
- ✅ Functional calculus foundation validated (current status: ✅)
- ✅ Neural paradigm operational (current status: ✅)
- ✅ Resources available (4-6 weeks development time)

**Defer if**:
- ❌ Higher priority features needed (e.g., production HPC)
- ❌ Test suite health critical (Issue #76 mass conservation)
- ❌ Limited research demand for Master Equation

**Current Recommendation**: **DEFER**
- Mass conservation bugs (Issue #76) are higher priority
- Master Equation is research-grade feature with limited immediate demand
- Common Noise MFG covers most practical stochastic scenarios

## 📝 Related Documents

- **Phase 2.2 Tracking**: Issue #68
- **Functional Calculus Tests**: `tests/unit/test_functional_calculus.py`
- **Common Noise Solver**: `mfg_pde/alg/numerical/stochastic/common_noise_solver.py`
- **Roadmap**: `docs/development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md`

---

**Status**: 📋 Planning Complete - Implementation Deferred
**Next Action**: Prioritize Issue #76 (test failures) before Master Equation
**Est. Start**: Q2 2026 (after core algorithm fixes)
