# MFG_PDE: Research-Grade Mean Field Games Framework

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://github.com/derrring/MFG_PDE/actions/workflows/ci.yml/badge.svg)](https://github.com/derrring/MFG_PDE/actions/workflows/ci.yml)
[![Code Quality](https://github.com/derrring/MFG_PDE/actions/workflows/modern_quality.yml/badge.svg)](https://github.com/derrring/MFG_PDE/actions/workflows/modern_quality.yml)
[![Security](https://github.com/derrring/MFG_PDE/actions/workflows/security.yml/badge.svg)](https://github.com/derrring/MFG_PDE/actions/workflows/security.yml)
[![Release](https://img.shields.io/github/v/release/derrring/MFG_PDE)](https://github.com/derrring/MFG_PDE/releases)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

A modern Python framework for solving Mean Field Games with modular solver architecture, GPU acceleration, reinforcement learning, and state-of-the-art numerical methods.

**🎯 Research-Grade**: Factory API for full algorithm access
**🧩 Three Solver Tiers**: Basic FDM / Hybrid (DEFAULT) / Advanced (WENO, Semi-Lagrangian)
**⭐ Mass-Conserving**: ~10⁻¹⁵ error with hybrid particle-FDM methods
**⚡ GPU Acceleration**: Multi-backend system (PyTorch, JAX, Numba)
**🎮 RL for MFG**: Complete continuous control (DDPG, TD3, SAC)
**🌐 Network Support**: Also works on graphs and networks

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/derrring/MFG_PDE.git
cd MFG_PDE
pip install -e .

# For developers: Add development tools
pip install -e ".[dev]"
```

### Your First MFG Solution (3 lines)

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_standard_solver

problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)
solver = create_standard_solver(problem, "fixed_point")  # Default: mass-conserving hybrid
result = solver.solve()

# Check results
print(f"Converged: {result.converged}")
print(f"Mass error: {result.mass_conservation_error:.2e}")  # ~10⁻¹⁵
```

**That's it!** You've solved a Mean Field Games system with research-grade quality.

## 🌟 **Key Capabilities**

### **💾 Data Persistence & I/O**
```python
# Save/load solver results with HDF5
from mfg_pde.utils.io import save_solution, load_solution

save_solution(U, M, metadata, 'solution.h5', compression='gzip')
U, M, meta = load_solution('solution.h5')

# Or use SolverResult convenience methods
result.save_hdf5('result.h5')
loaded_result = SolverResult.load_hdf5('result.h5')
```

### **🌐 Multi-Dimensional Solvers**
```python
# 2D/3D problems with efficient sparse methods
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.utils import SparseMatrixBuilder

grid = TensorProductGrid(dimension=2, bounds=[(0, 10), (0, 10)], num_points=[51, 51])
builder = SparseMatrixBuilder(grid, matrix_format='csr')
L = builder.build_laplacian(boundary_conditions='neumann')
```

### **📊 Stochastic MFG**
```python
# Common noise MFG with variance reduction
from mfg_pde.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfg_pde.alg.numerical import CommonNoiseMFGSolver

noise = OrnsteinUhlenbeckProcess(kappa=2.0, theta=0.2, sigma=0.1)
problem = StochasticMFGProblem(noise_process=noise, ...)
solver = CommonNoiseMFGSolver(problem, num_realizations=50, use_quasi_mc=True)
result = solver.solve()
```

### **🎯 Three-Tier Solver Hierarchy**
```python
from mfg_pde.factory import create_basic_solver, create_standard_solver, create_accurate_solver

# Tier 1: Basic FDM (benchmark, ~1-10% mass error)
solver = create_basic_solver(problem, damping=0.6)

# Tier 2: Hybrid (**DEFAULT** - ~10⁻¹⁵ mass error)
solver = create_standard_solver(problem, "fixed_point")

# Tier 3: Advanced (WENO, Semi-Lagrangian, DGM)
solver = create_accurate_solver(problem, solver_type="weno")
```

## 🏗️ **Factory API - Primary Interface**

### **Standard Usage**

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_standard_solver

# Create problem
problem = ExampleMFGProblem(Nx=100, Nt=50, T=1.0)

# Solve with default (Tier 2: Hybrid, mass-conserving)
solver = create_standard_solver(problem, "fixed_point")
result = solver.solve()

# Verify mass conservation
import numpy as np
for t in range(problem.Nt + 1):
    mass = np.sum(result.M[t, :]) * problem.Dx
    print(f"t={t}: mass={mass:.15f}")  # Should be 1.000000000000000
```

### **Method Comparison**

```python
from mfg_pde.factory import create_basic_solver, create_standard_solver, create_accurate_solver

# Compare three solver tiers
solvers = {
    "Basic FDM": create_basic_solver(problem),
    "Hybrid (Standard)": create_standard_solver(problem, "fixed_point"),
    "WENO (Advanced)": create_accurate_solver(problem, solver_type="weno")
}

results = {name: solver.solve() for name, solver in solvers.items()}

# Compare mass conservation
for name, result in results.items():
    print(f"{name}: {result.mass_conservation_error:.2e}")
```

## 🧩 **Modular Solver Architecture**

### **Mix & Match: FP + HJB Solver Combinations**

MFG_PDE solves the coupled Mean Field Games system by combining **any Fokker-Planck (FP) solver** with **any Hamilton-Jacobi-Bellman (HJB) solver**:

```python
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver
from mfg_pde.alg.numerical.hjb_solvers import HJBWeno5Solver
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator

# High-order combination: Particles + WENO5
problem = ExampleMFGProblem(Nx=128, Nt=64, T=1.0)
fp_solver = FPParticleSolver(problem, num_particles=5000)
hjb_solver = HJBWeno5Solver(problem, cfl_number=0.3)
mfg_solver = FixedPointIterator(problem, hjb_solver, fp_solver)

result = mfg_solver.solve(max_iterations=50, tolerance=1e-6)
print(f"✅ Converged with {result.iterations} iterations")
```

### **🎯 Available Solver Combinations**

| FP Method | HJB Method | Best For | Tier |
|-----------|------------|----------|------|
| **Particles** | **FDM** | Robust + conservation | **Tier 2 (DEFAULT)** |
| **Particles** | **WENO Family** | High accuracy + non-oscillatory | Tier 3 |
| **FDM** | **FDM** | Simple benchmark | Tier 1 |
| **Network/Graph** | **Network** | Complex geometries | Tier 3 |

### **⭐ WENO Family HJB Solvers**

State-of-the-art **fifth-order WENO solver**:

```python
from mfg_pde.alg.numerical.hjb_solvers import HJBWenoSolver

# Unified WENO family solver - choose your variant
weno_solver = HJBWenoSolver(
    problem=problem,
    weno_variant="weno5",              # Options: weno5, weno-z, weno-m, weno-js
    cfl_number=0.3,                    # Stability control
    time_integration="tvd_rk3",        # High-order time stepping
    weno_epsilon=1e-6                  # Non-oscillatory parameter
)

# Or use factory API
from mfg_pde.factory import create_accurate_solver
solver = create_accurate_solver(problem, solver_type="weno")
```

**WENO Family Features:**
- ✅ **Fifth-order spatial accuracy** in smooth regions
- ✅ **Non-oscillatory** behavior near discontinuities
- ✅ **Multiple variants**: WENO5, WENO-Z, WENO-M, WENO-JS
- ✅ **Academic publication ready** with comprehensive benchmarking

## 🎮 **Reinforcement Learning for MFG**

Complete RL framework supporting both **discrete** and **continuous** action spaces:

### **Continuous Control Algorithms**

```python
from mfg_pde.alg.reinforcement.algorithms import (
    MeanFieldDDPG,    # Deterministic policies with OU noise
    MeanFieldTD3,     # Twin delayed DDPG (best performance)
    MeanFieldSAC,     # Soft Actor-Critic (most robust)
)

# Example: Continuous control with SAC
from mfg_pde.alg.reinforcement.environments import LQMFGEnv

env = LQMFGEnv(state_dim=2, action_dim=2, population_size=100)
algo = MeanFieldSAC(
    env=env,
    state_dim=2,
    action_dim=2,
    population_dim=100,
    action_bounds=(-1.0, 1.0)
)

# Train agent to find Nash equilibrium
stats = algo.train(num_episodes=500)
print(f"Final reward: {stats['episode_rewards'][-1]:.2f}")
```

### **📊 Validated Performance** (Continuous LQ-MFG Benchmark)

- **TD3**: -3.32 ± 0.21 (best)
- **SAC**: -3.50 ± 0.17 (robust)
- **DDPG**: -4.28 ± 1.06 (fast)

### **🌍 Continuous MFG Environment Library**

Five production-ready environments:

| Environment | Domain | State Dim | Action Dim | Key Features |
|-------------|--------|-----------|------------|--------------|
| **LQ-MFG** | Control Theory | 2 | 2 | Quadratic costs, analytical solution |
| **Crowd Navigation** | Robotics | 5 | 2 | 2D kinematics, collision avoidance |
| **Price Formation** | Finance | 4 | 2 | Market making, liquidity |
| **Resource Allocation** | Economics | 6 | 3 | Portfolio optimization |
| **Traffic Flow** | Transportation | 3 | 1 | Congestion dynamics |

```python
from mfg_pde.alg.reinforcement.environments import CrowdNavigationEnv

env = CrowdNavigationEnv(num_agents=100, domain_size=10.0)
algo = MeanFieldSAC(env, state_dim=5, action_dim=2)
stats = algo.train(num_episodes=1000)
```

**📊 113 Tests** - All environments fully validated.

## ⚡ **Multi-Backend Acceleration System**

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_standard_solver
from mfg_pde.backends import create_backend

# Automatic optimal backend selection
problem = ExampleMFGProblem(Nx=100, Nt=50, T=1.0)
solver = create_standard_solver(problem, backend="auto")  # Chooses best available
result = solver.solve()

# Manual backend selection
torch_backend = create_backend("torch_mps")   # Apple Silicon
jax_backend = create_backend("jax_gpu")       # NVIDIA CUDA
numba_backend = create_backend("numba")       # CPU optimization
```

## 🌐 **Network MFG**

```python
from mfg_pde import create_grid_mfg_problem
from mfg_pde.factory import create_standard_solver

# MFG on networks and graphs
problem = create_grid_mfg_problem(10, 10, T=1.0, Nt=50)
solver = create_standard_solver(problem, backend="auto")
result = solver.solve()
```

## Features

- **🎯 Factory API**: Full algorithm access for researchers (3-tier solver hierarchy)
- **⚡ Mass-Conserving**: ~10⁻¹⁵ error with hybrid particle-FDM methods (Tier 2 default)
- **🧩 Modular Architecture**: Mix & match any FP solver + any HJB solver
- **⭐ WENO5 Solver**: Fifth-order accuracy with non-oscillatory properties
- **🎮 Reinforcement Learning**: Complete RL framework (Q-Learning, Actor-Critic, DDPG, TD3, SAC)
- **⚡ Multi-Backend System**: PyTorch (neural), JAX (math), Numba (CPU) + auto-selection
- **📊 Interactive Plots**: Built-in visualization with Plotly and Matplotlib
- **🚀 Performance**: Optimized for both small examples and large-scale problems
- **🌐 Network Support**: Solves MFG problems on graphs and networks
- **📚 Academic Ready**: Publication-quality benchmarking and documentation

## Requirements

- **Python**: 3.12+
- **NumPy**: 2.0+ (for best performance)
- **SciPy**, **Matplotlib** (core scientific stack)

**Optional but recommended:**
- **JAX**: GPU acceleration
- **PyTorch**: Neural network methods and RL
- **igraph/networkit**: Fast network backends
- **Plotly**: Interactive visualizations

## Documentation

### **For Users (Researchers & Practitioners)**
- **[Quick Start](docs/user/quickstart.md)** - Factory API tutorial (5 minutes)
- **[Solver Selection Guide](docs/user/SOLVER_SELECTION_GUIDE.md)** - Choosing solver tiers
- **[Examples](examples/)** - Working examples and tutorials
- **[Basic Examples Guide](examples/basic/README.md)** - 11 examples with learning paths

### **For Developers (Core Contributors)**
- **[Developer Guide](docs/development/)** - Extending the framework
- **[API Design](docs/development/PROGRESSIVE_DISCLOSURE_API_DESIGN.md)** - Two-level architecture
- **[Paradigm Overviews](docs/development/)** - Implementation guides for all paradigms
  - [Optimization Paradigm](docs/development/OPTIMIZATION_PARADIGM_OVERVIEW.md)
  - [RL Paradigm](docs/development/REINFORCEMENT_LEARNING_PARADIGM_OVERVIEW.md)
  - [Neural Paradigm](docs/development/NEURAL_PARADIGM_OVERVIEW.md)
- **[Theory](docs/theory/)** - Mathematical background
- **[Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Navigation hub for all docs

## Examples

- **[examples/basic/](examples/basic/)** - Simple getting started examples (11 examples, see [README](examples/basic/README.md))
- **[examples/advanced/](examples/advanced/)** - Complex workflows, GPU acceleration, WENO benchmarking
- **[examples/notebooks/](examples/notebooks/)** - Jupyter notebook tutorials

**Featured Examples**:
- `examples/basic/el_farol_bar_demo.py` - Classic coordination game (discrete states)
- `examples/basic/santa_fe_bar_demo.py` - Preference evolution formulation
- `examples/basic/towel_beach_demo.py` - Spatial competition with phase transitions
- `examples/advanced/weno_solver_demo.py` - Unified WENO family demonstration
- `examples/advanced/continuous_control_comparison.py` - RL continuous control comparison

## Testing

```bash
# Run tests
pytest tests/

# For developers
pytest tests/ -v
pre-commit run --all-files
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{mfg_pde2025,
  title={MFG_PDE: Research-Grade Mean Field Games Framework},
  author={derrring},
  year={2025},
  url={https://github.com/derrring/MFG_PDE}
}
```

## License

[Add appropriate license information]
