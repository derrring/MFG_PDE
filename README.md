# MFG_PDE: Advanced Mean Field Games Framework

A modern Python framework for solving Mean Field Games with GPU acceleration, simple APIs, and advanced numerical methods.

**🎯 Simple API**: One-line solving for common problems
**⚡ GPU Acceleration**: JAX backend with 10-100× speedup potential
**🔧 Multiple Solvers**: Traditional PDE and modern particle methods
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

### ⚡ **GPU Acceleration**

```python
from mfg_pde import ExampleMFGProblem, create_fast_solver

# Traditional MFG with automatic GPU acceleration
problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=100, T=1.0, Nt=50)
solver = create_fast_solver(problem, backend="jax")  # Uses GPU if available
result = solver.solve()
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
- **⚡ GPU Acceleration**: JAX backend with 10-100× speedup potential
- **🔧 Multiple Solvers**: Fixed-point, particle-collocation, adaptive methods
- **📊 Interactive Plots**: Built-in visualization with Plotly and Matplotlib
- **🚀 Performance**: Optimized for both small examples and large-scale problems
- **🌐 Network Support**: Also solves MFG problems on graphs and networks

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
- **[examples/advanced/](examples/advanced/)** - Complex workflows and GPU acceleration
- **[examples/notebooks/](examples/notebooks/)** - Jupyter notebook tutorials

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
