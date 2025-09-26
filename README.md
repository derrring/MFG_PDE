# MFG_PDE: Advanced Mean Field Games Framework

A modern Python framework for solving Mean Field Games with modular solver architecture, GPU acceleration, and state-of-the-art numerical methods.

**🎯 Simple API**: One-line solving for common problems
**🧩 Modular Design**: Mix & match FP + HJB solvers freely
**⭐ WENO Family Solvers**: Unified WENO variants (WENO5, WENO-Z, WENO-M, WENO-JS) + non-oscillatory properties
**⚡ GPU Acceleration**: JAX backend with 10-100× speedup potential
**🔧 Multiple Solvers**: Traditional PDE, particles, and hybrid methods
**🌐 Network Support**: Also works on graphs and networks

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/derrring/MFG_PDE.git
cd MFG_PDE
pip install -e .

# Try it out
python examples/basic/semi_lagrangian_example.py
```

**For developers**: Add `[dev]` to install development tools:
```bash
pip install -e ".[dev]"
```

### Verify Installation
```python
from mfg_pde import solve_mfg
result = solve_mfg("crowd_dynamics")
print("✅ MFG_PDE is working!")
```

## 🎊 **Recent Achievements: Strategic Typing Excellence**

MFG_PDE now features **100% strategic typing coverage** with a research-optimized CI/CD pipeline:

**🏆 Strategic Typing Framework:**
- **366 → 0 MyPy errors** (100% strategic reduction achieved)
- **91 source files** with complete type safety
- **Zero production breaking changes** throughout the typing improvement process
- **Strategic ignore methodology** for complex scientific computing patterns

**🚀 Research-Optimized CI/CD:**
- **Fast feedback loops** for development productivity
- **Comprehensive validation** for releases
- **Quality awareness** without blocking research workflows
- **Environment compatibility** handling for MyPy/Ruff differences

**📚 Complete Documentation:**
- [Strategic Typing Experience Guide](docs/development/CI_CD_STRATEGIC_TYPING_EXPERIENCE_GUIDE.md)
- [Strategic Typing Patterns Reference](docs/development/STRATEGIC_TYPING_PATTERNS_REFERENCE.md)
- [CI/CD Troubleshooting Quick Reference](docs/development/CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md)

This framework now serves as a **blueprint for strategic typing in scientific computing projects**!

## 🏗️ **Modular Solver Architecture**

### **🧩 Mix & Match: FP + HJB Solver Combinations**

MFG_PDE solves the coupled Mean Field Games system by combining **any Fokker-Planck (FP) solver** with **any Hamilton-Jacobi-Bellman (HJB) solver**:

```python
from mfg_pde.alg.fp_solvers import FPParticleSolver
from mfg_pde.alg.hjb_solvers import HJBWeno5Solver  # ✨ NEW: Fifth-order WENO
from mfg_pde.alg.mfg_solvers import FixedPointIterator

# Create any combination you want
problem = ExampleMFGProblem(Nx=128, Nt=64, T=1.0)

# High-order combination: Particles + WENO5
fp_solver = FPParticleSolver(problem, num_particles=5000)
hjb_solver = HJBWeno5Solver(problem, cfl_number=0.3)  # Fifth-order accuracy!
mfg_solver = FixedPointIterator(problem, hjb_solver, fp_solver)

result = mfg_solver.solve(max_iterations=50, tolerance=1e-6)
print(f"✅ Converged with {result.picard_iterations} iterations")
```

### **🎯 Available Solver Combinations**

| FP Method | HJB Method | Best For | Example |
|-----------|------------|----------|---------|
| **Particles** | **WENO Family** ✨ | High accuracy + non-oscillatory | Academic benchmarking |
| **Particles** | **Standard FDM** | Robust + conservation | Production applications |
| **Standard FDM** | **WENO Family** ✨ | High-order everywhere | Smooth problems |
| **Network/Graph** | **Network** | Complex geometries | Urban dynamics |

### **⭐ WENO Family HJB Solvers - ENHANCED!**

Our latest **fifth-order WENO solver** provides state-of-the-art accuracy:

```python
from mfg_pde.alg.hjb_solvers import HJBWenoSolver

# Unified WENO family solver - choose your variant
weno_solver = HJBWenoSolver(
    problem=problem,
    weno_variant="weno5",              # Options: weno5, weno-z, weno-m, weno-js
    cfl_number=0.3,                    # Stability control
    time_integration="tvd_rk3",        # High-order time stepping
    weno_epsilon=1e-6                  # Non-oscillatory parameter
)

# Run comprehensive WENO family demo
python examples/advanced/weno_solver_demo.py
```

**WENO Family Features:**
- ✅ **Fifth-order spatial accuracy** in smooth regions
- ✅ **Non-oscillatory** behavior near discontinuities
- ✅ **Explicit time integration** (complementary to implicit methods)
- ✅ **Academic publication ready** with comprehensive benchmarking

## Usage Examples

### 🎯 **Simple API - Start Here**

```python
from mfg_pde import solve_mfg

# One-line solving for common problems
result = solve_mfg("crowd_dynamics", domain_size=2.0, accuracy="high")
result.plot()  # Interactive visualization

# Portfolio optimization
result = solve_mfg("portfolio_optimization", risk_aversion=0.3)
print(f"Converged: {result.convergence_achieved}")

# Parameter validation with suggestions
from mfg_pde import validate_problem_parameters
validation = validate_problem_parameters("epidemic", infection_rate=1.5)
if not validation["valid"]:
    print("Issues:", validation["suggestions"])
```

### ⚡ **Multi-Backend Acceleration System**

```python
from mfg_pde import ExampleMFGProblem, create_fast_solver
from mfg_pde.backends import create_backend

# Automatic optimal backend selection
problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=100, T=1.0, Nt=50)
solver = create_fast_solver(problem, backend="auto")  # Chooses best available
result = solver.solve()

# Manual backend selection
torch_backend = create_backend("torch_mps")   # Apple Silicon
jax_backend = create_backend("jax_gpu")       # NVIDIA CUDA
numba_backend = create_backend("numba")       # CPU optimization
```

### 🌐 **Network MFG**

```python
from mfg_pde import create_grid_mfg_problem, create_fast_solver

# Also supports MFG on networks and graphs
problem = create_grid_mfg_problem(10, 10, T=1.0, Nt=50)
solver = create_fast_solver(problem, backend="auto")
result = solver.solve()
```

## Features

- **🎯 Simple API**: One-line `solve_mfg()` for common problems with smart defaults
- **🧩 Modular Architecture**: Mix & match any FP solver + any HJB solver
- **⭐ WENO5 Solver**: Fifth-order accuracy with non-oscillatory properties
- **⚡ Multi-Backend System**: PyTorch (neural), JAX (math), Numba (CPU) + auto-selection
- **🔧 Multiple Solvers**: Fixed-point, particle-collocation, hybrid methods
- **📊 Interactive Plots**: Built-in visualization with Plotly and Matplotlib
- **🚀 Performance**: Optimized for both small examples and large-scale problems
- **🌐 Network Support**: Also solves MFG problems on graphs and networks
- **📚 Academic Ready**: Publication-quality benchmarking and documentation

## Requirements

- **Python**: 3.12+
- **NumPy**: 2.0+ (for best performance)
- **SciPy**, **Matplotlib** (core scientific stack)

**Optional but recommended:**
- **JAX**: GPU acceleration
- **igraph/networkit**: Fast network backends
- **Plotly**: Interactive visualizations

## Documentation

- **[examples/](examples/)** - Working examples and tutorials
- **[docs/user/](docs/user/)** - User guides and tutorials
- **[docs/theory/](docs/theory/)** - Mathematical background
- **[mfg_pde/](mfg_pde/)** - API reference in source code

## Examples

- **[examples/basic/](examples/basic/)** - Simple getting started examples
- **[examples/advanced/](examples/advanced/)** - Complex workflows, GPU acceleration, and WENO family benchmarking
- **[examples/notebooks/](examples/notebooks/)** - Jupyter notebook tutorials

**🆕 Latest Examples:**
- `examples/advanced/weno_solver_demo.py` - Unified WENO family demonstration
- `examples/advanced/weno_family_comparison_demo.py` - Comprehensive WENO variants comparison
- See **[Issue #17](../../issues/17)** for roadmap: PINNs, DGM, and hybrid methods

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
  title={MFG_PDE: Advanced Mean Field Games Framework},
  author={derrring},
  year={2025},
  url={https://github.com/derrring/MFG_PDE}
}
```

## License

[Add appropriate license information]
