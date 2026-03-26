# Numerical Methods Paradigm for Mean Field Games

**Status**: ✅ ACTIVE (Core paradigm for MFGarchon)
**Last Updated**: 2025-10-11
**Related Issues**: #105

---

## 📖 Overview

The **Numerical Methods Paradigm** provides the foundational computational framework for solving Mean Field Game systems through classical numerical techniques. This paradigm encompasses finite difference methods (FDM), finite element methods (FEM), spectral methods, and particle-based approaches optimized for 1D, 2D, and network-structured problems.

### Why Numerical Methods?

**Strengths**:
- ✅ **Mathematically rigorous**: Well-established convergence theory
- ✅ **Fast for low dimensions**: d ≤ 3 solved in seconds to minutes
- ✅ **Predictable behavior**: Known stability conditions and error bounds
- ✅ **Guaranteed accuracy**: Mesh refinement → controlled error reduction
- ✅ **Production ready**: Mature implementations with extensive validation

**Limitations**:
- ⚠️ **Curse of dimensionality**: d > 5 becomes computationally prohibitive
- ⚠️ **Fixed topology**: Requires predefined mesh or grid
- ⚠️ **Memory intensive**: O(N^d) storage for d-dimensional grids

### When to Use Numerical Methods

| Problem Type | Recommended Method | Expected Performance |
|:-------------|:-------------------|:---------------------|
| **1D LQ-MFG** | FDM (Finite Difference) | < 1 second (Nx=100) |
| **1D Congestion** | FDM or Semi-Lagrangian | 1-5 seconds (Nx=100) |
| **2D Traffic Flow** | FDM or WENO | 30-180 seconds (50×50 grid) |
| **Network MFG** | Graph FDM | 1-10 seconds (100 nodes) |
| **High-dimensional (d>5)** | ❌ Use Neural Paradigm | N/A |
| **Particle systems** | Particle methods | 5-60 seconds (10k particles) |

---

## 🏗️ Architecture

### Package Structure

```
mfgarchon/alg/numerical/
├── hjb_solvers/              # Hamilton-Jacobi-Bellman equation solvers
│   ├── base_hjb.py           # Abstract base class for HJB solvers
│   ├── hjb_fdm.py            # Finite difference method (Newton iteration)
│   ├── hjb_weno.py           # Weighted Essentially Non-Oscillatory scheme
│   ├── hjb_semi_lagrangian.py # Semi-Lagrangian method (Lax-Friedrichs)
│   ├── hjb_gfdm.py           # Generalized FDM for irregular domains
│   └── hjb_network.py        # Graph-based HJB for networks
│
├── fp_solvers/               # Fokker-Planck equation solvers
│   ├── base_fp.py            # Abstract base class for FP solvers
│   ├── fp_fdm.py             # Finite difference method (implicit scheme)
│   ├── fp_particle.py        # Particle-based density evolution (SDE)
│   └── fp_network.py         # Graph-based FP for networks
│
├── mfg_solvers/              # Coupled MFG system solvers
│   ├── base_mfg.py           # Abstract base class for MFG solvers
│   ├── fixed_point_iterator.py # Picard iteration with Anderson acceleration
│   ├── fixed_point_utils.py  # Convergence checking, initialization utilities
│   ├── hybrid_fp_particle_hjb_fdm.py # Hybrid particle-FDM method
│   ├── particle_collocation_solver.py # Particle collocation with QP
│   └── network_mfg_solver.py # Complete network MFG solver
│
├── stochastic/               # Stochastic MFG extensions
│   └── common_noise_solver.py # Common noise MFG solver
│
├── density_estimation.py     # KDE and density estimation utilities
└── particle_utils.py         # Particle simulation utilities
```

### Solver Hierarchy

```
BaseMFGSolver (Abstract)
    │
    ├── FixedPointIterator  ───┐ (Picard iteration coupling)
    │                          │
    │   Couples:               │
    │   ├─ BaseHJBSolver ──────┤
    │   │   ├── HJBFDMSolver   │  (Hamilton-Jacobi-Bellman)
    │   │   ├── HJBWENOSolver  │
    │   │   ├── HJBSemiLagrangianSolver
    │   │   ├── HJBGFDMSolver  │
    │   │   └── HJBNetworkSolver
    │   │                      │
    │   └─ BaseFPSolver ───────┤
    │       ├── FPFDMSolver    │  (Fokker-Planck)
    │       ├── FPParticleSolver
    │       └── FPNetworkSolver
    │                          │
    ├── HybridFPParticleHJBFDM ┘ (Hybrid particle-FDM)
    ├── ParticleCollocationSolver (Particle + QP collocation)
    ├── NetworkMFGSolver       (Complete network solver)
    └── CommonNoiseSolver      (Stochastic MFG)
```

---

## 🧮 Hamilton-Jacobi-Bellman (HJB) Solvers

### Mathematical Formulation

The HJB equation describes the optimal value function $u(t,x)$:

$$
-\frac{\partial u}{\partial t} + H(\nabla u, x, m) = 0, \quad t \in [0,T], \, x \in \Omega
$$

with terminal condition $u(T,x) = g(x)$ and Hamiltonian:

$$
H(p, x, m) = \frac{\sigma^2}{2} |p|^2 + f(x, m) + V(x)
$$

### Base Class: `BaseHJBSolver`

**File**: `mfgarchon/alg/numerical/hjb_solvers/base_hjb.py:23`

**Interface**:
```python
class BaseHJBSolver(BaseNumericalSolver):
    @abstractmethod
    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,  # (Nt, Nx)
        U_final_condition_at_T: np.ndarray,       # (Nx,)
        U_from_prev_picard: np.ndarray,           # (Nt, Nx)
    ) -> np.ndarray:                              # Returns (Nt, Nx)
        """Solve HJB backward in time given density M."""
```

### 1. Finite Difference Method (FDM)

**Implementation**: `mfgarchon/alg/numerical/hjb_solvers/hjb_fdm.py:15`

**Method**: Newton iteration with upwind finite differences

**Discretization**:
$$
\frac{u_i^n - u_i^{n+1}}{\Delta t} + H(p_i^n, x_i, m_i^n) = 0
$$

where $p_i^n$ is computed via upwind scheme:
$$
p_i^+ = \frac{u_{i+1}^n - u_i^n}{\Delta x}, \quad p_i^- = \frac{u_i^n - u_{i-1}^n}{\Delta x}
$$

**Features**:
- Newton iteration for nonlinear Hamiltonian
- Backend support (NumPy, PyTorch, JAX)
- Configurable tolerance and max iterations

**Usage Example**:
```python
from mfgarchon import ExampleMFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

problem = ExampleMFGProblem(T=1.0, Nx=100, Nt=100)
hjb_solver = HJBFDMSolver(
    problem,
    max_newton_iterations=30,
    newton_tolerance=1e-6,
    backend="numpy"
)

# Solve HJB given density M
U_solution = hjb_solver.solve_hjb_system(
    M_density_evolution=M,
    U_final_condition=g_terminal,
    U_from_prev_picard=U_prev
)
```

**Performance**:
- 1D (Nx=100): ~0.5s per Picard iteration
- 2D (50×50): ~5-10s per Picard iteration

### 2. WENO (Weighted Essentially Non-Oscillatory)

**Implementation**: `mfgarchon/alg/numerical/hjb_solvers/hjb_weno.py`

**Method**: High-order accurate scheme for discontinuous solutions

**Features**:
- 5th-order accuracy in smooth regions
- Non-oscillatory near shocks and discontinuities
- Adaptive stencil weighting

**When to Use**:
- Shocks or discontinuities in value function
- High-accuracy requirements (< 1e-8 error)
- Smooth nonlinear Hamiltonians

**Performance**: 2-3× slower than FDM but higher accuracy

### 3. Semi-Lagrangian Method

**Implementation**: `mfgarchon/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py`

**Method**: Lax-Friedrichs scheme with characteristic tracing

**Discretization**:
$$
u_i^n = \min_{a \in A} \left[ u(x_i - a \Delta t, t^{n+1}) + L(x_i, a, m_i^n) \Delta t \right]
$$

**Features**:
- No CFL condition restriction
- Natural handling of non-smooth Hamiltonians
- Interpolation-based (linear or cubic splines)

**When to Use**:
- Large time steps needed (fast dynamics)
- Non-smooth optimal controls
- Problems with multiple local minima

**Performance**: Similar to FDM for moderate time steps

### 4. Generalized FDM (GFDM)

**Implementation**: `mfgarchon/alg/numerical/hjb_solvers/hjb_gfdm.py`

**Method**: Meshless FDM for irregular domains

**Features**:
- No structured grid required
- Adaptive node placement
- Handles complex geometries

**When to Use**:
- Irregular domains (obstacles, curved boundaries)
- Adaptive mesh refinement
- Non-Cartesian coordinate systems

**Performance**: 1.5-2× slower than standard FDM

### 5. Network HJB Solver

**Implementation**: `mfgarchon/alg/numerical/hjb_solvers/hjb_network.py`

**Method**: Graph-based HJB on networks

**Discretization**: For each node $v$ and edge $(v, w)$:
$$
-\frac{\partial u_v}{\partial t} + \min_{w \in N(v)} \left[ c(v,w) + \frac{1}{2\Delta t}(u_v - u_w)^2 \right] = 0
$$

**Features**:
- Graph adjacency-based stencil
- Weighted edge costs
- Handles directed/undirected graphs

**When to Use**:
- Traffic networks, infrastructure problems
- Discrete state spaces
- Multi-agent coordination on graphs

**Performance**: O(|E| × Nt) complexity

---

## 🌊 Fokker-Planck (FP) Solvers

### Mathematical Formulation

The FP equation describes the density evolution $m(t,x)$:

$$
\frac{\partial m}{\partial t} - \nabla \cdot (m \nabla H_p(\nabla u, x, m)) - \frac{\sigma^2}{2} \Delta m = 0, \quad t \in [0,T]
$$

with initial condition $m(0, x) = m_0(x)$ and mass conservation $\int_\Omega m(t,x) dx = 1$.

### Base Class: `BaseFPSolver`

**File**: `mfgarchon/alg/numerical/fp_solvers/base_fp.py:12`

**Interface**:
```python
class BaseFPSolver(ABC):
    @abstractmethod
    def solve_fp_system(
        self,
        m_initial_condition: np.ndarray,     # (Nx,)
        U_solution_for_drift: np.ndarray,    # (Nt, Nx)
    ) -> np.ndarray:                         # Returns (Nt, Nx)
        """Solve FP forward in time given value function U."""
```

### 1. Finite Difference Method (FDM)

**Implementation**: `mfgarchon/alg/numerical/fp_solvers/fp_fdm.py:15`

**Method**: Implicit Euler with upwind advection

**Discretization**:
$$
\frac{m_i^{n+1} - m_i^n}{\Delta t} = \frac{\sigma^2}{2} \frac{m_{i+1}^{n+1} - 2m_i^{n+1} + m_{i-1}^{n+1}}{\Delta x^2} - \nabla \cdot (m^{n+1} v_i^n)
$$

where drift $v_i^n = -\nabla H_p(\nabla u_i^n, x_i, m_i^n)$.

**Features**:
- Implicit scheme (unconditionally stable)
- Sparse linear system (tridiagonal for 1D)
- Mass conservation enforcement
- Boundary conditions: periodic, Dirichlet, no-flux

**Usage Example**:
```python
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.geometry import BoundaryConditions

problem = ExampleMFGProblem(T=1.0, Nx=100, Nt=100)
fp_solver = FPFDMSolver(
    problem,
    boundary_conditions=BoundaryConditions(type="no_flux")
)

# Solve FP given value function U
M_solution = fp_solver.solve_fp_system(
    m_initial_condition=m0,
    U_solution_for_drift=U
)
```

**Performance**:
- 1D (Nx=100): ~0.1s (sparse solver)
- 2D (50×50): ~1-2s per solve

### 2. Particle Method

**Implementation**: `mfgarchon/alg/numerical/fp_solvers/fp_particle.py`

**Method**: Monte Carlo SDE simulation

**Algorithm**:
1. Sample N particles from $m_0(x)$
2. Simulate SDEs: $dX_t = -\nabla H_p dt + \sigma dW_t$
3. Estimate density via KDE: $m(t,x) \approx \frac{1}{N} \sum_{i=1}^N K_h(x - X_i^t)$

**Features**:
- No curse of dimensionality (scales with N, not d)
- Natural handling of irregular domains
- GPU acceleration for large N

**When to Use**:
- High-dimensional problems (d > 3)
- Complex geometries
- Stochastic systems with high variance

**Performance**:
- 10k particles: ~5-10s
- 100k particles (GPU): ~20-30s

### 3. Network FP Solver

**Implementation**: `mfgarchon/alg/numerical/fp_solvers/fp_network.py`

**Method**: Graph-based FP on networks

**Discretization**: For each node $v$:
$$
\frac{dm_v}{dt} = \sum_{w \in N(v)} \left[ m_w \Phi(u_w, u_v) - m_v \Phi(u_v, u_w) \right]
$$

where $\Phi$ is the flux function determined by optimal control.

**Features**:
- Graph adjacency-based stencil
- Mass conservation on graphs
- Supports weighted edges

**Performance**: O(|E| × Nt) complexity

---

## 🔄 Coupled MFG Solvers

### Base Class: `BaseMFGSolver`

**File**: `mfgarchon/alg/numerical/mfg_solvers/base_mfg.py:15`

**Interface**:
```python
class BaseMFGSolver(ABC):
    @abstractmethod
    def solve(
        self,
        max_iterations: int,
        tolerance: float = 1e-5,
        **kwargs
    ) -> SolverResult:
        """Solve coupled MFG system."""
```

### 1. Fixed-Point Iterator (Picard Iteration)

**Implementation**: `mfgarchon/alg/numerical/mfg_solvers/fixed_point_iterator.py:34`

**Algorithm**:
```
Initialize: U⁰, M⁰
For k = 0, 1, 2, ...
    1. Solve HJB: U^{k+1} = HJB_solve(M^k)
    2. Solve FP:  M^{k+1} = FP_solve(U^{k+1})
    3. Damp:      U^{k+1} ← θ U^{k+1} + (1-θ) U^k
                  M^{k+1} ← θ M^{k+1} + (1-θ) M^k
    4. Check:     ||U^{k+1} - U^k|| < ε and ||M^{k+1} - M^k|| < ε
Until convergence or max_iterations
```

**Features**:
- **Config-based management**: Modern `MFGSolverConfig` support
- **Anderson acceleration**: Faster convergence (optional)
- **Backend support**: NumPy, PyTorch, JAX
- **Warm start**: Initialize from previous solution
- **Structured output**: `SolverResult` object (tuple-compatible for legacy)

**Usage Example**:
```python
from mfgarchon import ExampleMFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.mfg_solvers import FixedPointIterator
from mfgarchon.config import create_fast_config

# Create problem and solvers
problem = ExampleMFGProblem(T=1.0, Nx=100, Nt=100)
hjb_solver = HJBFDMSolver(problem)
fp_solver = FPFDMSolver(problem)

# Create MFG solver with config
config = create_fast_config(problem)
mfg_solver = FixedPointIterator(
    problem,
    hjb_solver,
    fp_solver,
    config=config,
    use_anderson=True,    # Enable Anderson acceleration
    anderson_depth=5,
    backend="numpy"
)

# Solve
result = mfg_solver.solve()

# Access results
U, M = result.U, result.M
print(f"Converged in {result.iterations} iterations")
print(f"Final error: {result.error_u:.2e}")
```

**Convergence**:
- Linear convergence: O(θ^k)
- Anderson acceleration: Superlinear for smooth problems
- Typical: 10-50 iterations for ε=1e-6

**Performance**:
- 1D (Nx=100): 5-15 seconds total
- 2D (50×50): 60-180 seconds total

### 2. Hybrid FP-Particle HJB-FDM

**Implementation**: `mfgarchon/alg/numerical/mfg_solvers/hybrid_fp_particle_hjb_fdm.py`

**Method**: Particle-based FP + grid-based HJB

**Algorithm**:
1. Initialize N particles for density representation
2. **FP step**: Simulate particles forward via SDE
3. **KDE**: Estimate density $m(t,x)$ on grid via kernel density estimation
4. **HJB step**: Solve HJB on grid given $m(t,x)$
5. Repeat until convergence

**Advantages**:
- ✅ No curse of dimensionality for FP
- ✅ Accurate HJB solution on grid
- ✅ Handles high-dimensional state spaces (d > 3)

**When to Use**:
- High-dimensional problems (3 < d ≤ 10)
- Complex density dynamics
- When HJB structure is low-dimensional

**Performance**:
- d=5, N=10k particles: ~60-120s
- Scales with N rather than grid size

### 3. Particle Collocation Solver

**Implementation**: `mfgarchon/alg/numerical/mfg_solvers/particle_collocation_solver.py`

**Method**: Particle representation + Quadratic Programming (QP) collocation

**Algorithm**:
1. Represent density as weighted particles: $m \approx \sum_i w_i \delta(x - x_i)$
2. Solve QP for optimal weights satisfying PDE constraints
3. Update particle positions via optimal control
4. Iterate until convergence

**Features**:
- Mesh-free approach
- Exact mass conservation
- Handles irregular domains naturally

**Performance**: Slower than FDM but more flexible

### 4. Network MFG Solver

**Implementation**: `mfgarchon/alg/numerical/mfg_solvers/network_mfg_solver.py`

**Method**: Complete graph-based MFG solver

**Couples**: `HJBNetworkSolver` + `FPNetworkSolver`

**When to Use**:
- Traffic networks
- Infrastructure problems
- Discrete state spaces (finite graphs)

**Performance**: Fast for moderate graphs (< 1000 nodes)

### 5. Common Noise Solver

**Implementation**: `mfgarchon/alg/numerical/stochastic/common_noise_solver.py`

**Method**: Stochastic MFG with common noise

**Formulation**: Extended MFG system with common noise $W_t$:
$$
-du + H(\nabla u, x, m) dt = \alpha dW_t
$$

**Features**:
- Master equation formulation
- Conditional value functions
- Stochastic convergence analysis

**Performance**: Similar to standard fixed-point but with stochastic iterations

---

## 📊 Performance Comparison

### Benchmark Results (1D LQ-MFG, Nx=100, Nt=100)

| Solver Configuration | Total Time | Iterations | Accuracy |
|:--------------------|:-----------|:-----------|:---------|
| **FDM + FDM** | 5.2s | 15 | 1e-6 |
| **FDM + FDM (Anderson)** | 3.1s | 8 | 1e-6 |
| **WENO + FDM** | 12.4s | 18 | 1e-8 |
| **Semi-Lagrangian + FDM** | 6.8s | 12 | 1e-6 |
| **FDM + Particle (10k)** | 45s | 20 | 1e-5 |

### Scaling with Dimension

| Dimension | Grid Size | FDM Time | Particle Time (10k) |
|:----------|:----------|:---------|:--------------------|
| **d=1** | 100 | 5s | 45s |
| **d=2** | 50×50 | 120s | 60s |
| **d=3** | 30×30×30 | ~30 min | 80s |
| **d=5** | 20^5 | ❌ (Infeasible) | 150s |

**Key Insight**: Particle methods scale better for d > 3, while FDM dominates for d ≤ 2.

---

## 🎯 Usage Guidance

### Decision Tree: Which Solver to Choose?

```
START
  │
  ├─ Is d > 5? ───YES──→ Use Neural Paradigm (PINN/DGM)
  │      │
  │     NO
  │      │
  ├─ Is domain a network/graph? ───YES──→ Use Network solvers
  │      │
  │     NO
  │      │
  ├─ Is d ≤ 2? ───YES──→ Use FDM (fast, accurate)
  │      │
  │     NO (3 ≤ d ≤ 5)
  │      │
  ├─ Need very high accuracy? ───YES──→ Use WENO
  │      │
  │     NO
  │      │
  ├─ Have discontinuities? ───YES──→ Use Semi-Lagrangian
  │      │
  │     NO
  │      │
  └──→ Use Hybrid Particle-FDM
```

### Quick Start Examples

#### Example 1: Standard 1D MFG

```python
from mfgarchon import ExampleMFGProblem
from mfgarchon.factory import create_fast_solver

# Create problem
problem = ExampleMFGProblem(T=1.0, Nx=100, Nt=100, sigma=0.5)

# Create solver (auto-configured)
solver = create_fast_solver(problem)

# Solve
result = solver.solve()

# Visualize
from mfgarchon.visualization import plot_mfg_solution
plot_mfg_solution(result, problem)
```

#### Example 2: 2D Traffic Flow with WENO

```python
from mfgarchon import TrafficFlowProblem2D
from mfgarchon.alg.numerical.hjb_solvers import HJBWENOSolver
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.mfg_solvers import FixedPointIterator
from mfgarchon.config import create_balanced_config

# Create 2D problem
problem = TrafficFlowProblem2D(T=1.0, Nx=50, Ny=50, Nt=100)

# Configure solvers
config = create_balanced_config(problem)
hjb_solver = HJBWENOSolver(problem)
fp_solver = FPFDMSolver(problem)
mfg_solver = FixedPointIterator(problem, hjb_solver, fp_solver, config=config)

# Solve
result = mfg_solver.solve()
```

#### Example 3: Network MFG

```python
import networkx as nx
from mfgarchon.core.network_mfg_problem import NetworkMFGProblem
from mfgarchon.alg.numerical.mfg_solvers import NetworkMFGSolver

# Create network
G = nx.erdos_renyi_graph(100, 0.1)
problem = NetworkMFGProblem(graph=G, T=1.0, Nt=100)

# Solve
solver = NetworkMFGSolver(problem)
result = solver.solve(max_iterations=100, tolerance=1e-6)
```

#### Example 4: Particle-Based Method for d=5

```python
from mfgarchon.alg.numerical.mfg_solvers import HybridFPParticleHJBFDM

# Create high-dimensional problem
problem = HighDimMFGProblem(dim=5, T=1.0)

# Use hybrid particle-FDM
solver = HybridFPParticleHJBFDM(
    problem,
    num_particles=10000,
    kde_bandwidth=0.1
)

result = solver.solve(max_iterations=50, tolerance=1e-5)
```

---

## 🔧 Advanced Configuration

### Anderson Acceleration

Accelerates fixed-point convergence via extrapolation:

```python
from mfgarchon.alg.numerical.mfg_solvers import FixedPointIterator

solver = FixedPointIterator(
    problem, hjb_solver, fp_solver,
    use_anderson=True,
    anderson_depth=5,      # Memory depth (5-10 recommended)
    anderson_beta=1.0,     # Mixing parameter (0.5-1.0)
)
```

**Effect**: Reduces iterations by 30-50% for smooth problems.

### Backend Selection

Choose computational backend for performance:

```python
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

# NumPy (default, CPU)
solver = HJBFDMSolver(problem, backend="numpy")

# PyTorch (GPU support)
solver = HJBFDMSolver(problem, backend="torch")  # Auto-detects CUDA

# JAX (JIT compilation + GPU)
solver = HJBFDMSolver(problem, backend="jax")
```

**Performance Impact**:
- NumPy: Baseline
- PyTorch (CUDA): 5-10× speedup for large grids
- JAX: 3-8× speedup + lower memory

### Boundary Conditions

```python
from mfgarchon.geometry import BoundaryConditions

# Periodic (default for many problems)
bc = BoundaryConditions(type="periodic")

# Dirichlet (fixed values)
bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=1.0)

# No-flux (natural for density conservation)
bc = BoundaryConditions(type="no_flux")

fp_solver = FPFDMSolver(problem, boundary_conditions=bc)
```

---

## 📚 Mathematical Background

### Convergence Theory

**FDM Convergence**: Under smoothness assumptions, FDM converges at rate O(Δx² + Δt):
$$
\|u_h - u\|_\infty \leq C(\Delta x^2 + \Delta t)
$$

**Picard Iteration Convergence**: For contraction factor θ < 1:
$$
\|u^{k+1} - u^*\| \leq \theta \|u^k - u^*\|
$$

**Stability**: CFL condition for explicit schemes:
$$
\frac{\sigma^2 \Delta t}{\Delta x^2} \leq \frac{1}{2}
$$

(Implicit FDM used in MFGarchon removes this restriction.)

### Mass Conservation

FP solvers enforce mass conservation:
$$
\int_\Omega m(t, x) dx = \int_\Omega m_0(x) dx = 1, \quad \forall t \in [0,T]
$$

**Verification**:
```python
import numpy as np

# Check mass conservation
mass_error = np.abs(np.trapz(result.M, problem.x_grid, axis=1) - 1.0)
print(f"Max mass conservation error: {mass_error.max():.2e}")
```

---

## 🔗 Related Documentation

- **Neural Paradigm**: `docs/development/NEURAL_PARADIGM_OVERVIEW.md`
- **Config System**: `docs/development/CONFIG_SYSTEM_GUIDE.md`
- **Backend Strategies**: `docs/development/BACKEND_STRATEGIES_GUIDE.md`
- **Solver Result API**: `docs/development/SOLVER_RESULT_API.md`
- **Benchmarking**: `benchmarks/notebooks/performance_comparison.ipynb`

---

## 📖 References

1. **Lasry & Lions (2007)**: "Mean Field Games", Japanese Journal of Mathematics
2. **Achdou & Capuzzo-Dolcetta (2010)**: "Mean Field Games: Numerical Methods", SIAM Journal on Numerical Analysis
3. **Cardaliaguet et al. (2013)**: "The Master Equation and the Convergence Problem in Mean Field Games"
4. **Carmona & Delarue (2018)**: "Probabilistic Theory of Mean Field Games", Springer

---

**Maintained by**: MFGarchon Development Team
**Last Updated**: 2025-10-11
