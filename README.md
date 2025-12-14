# MFG_PDE: Mean Field Games Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://github.com/derrring/MFG_PDE/actions/workflows/ci.yml/badge.svg)](https://github.com/derrring/MFG_PDE/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/derrring/MFG_PDE/graph/badge.svg?token=HGZFRSF5V6)](https://codecov.io/github/derrring/MFG_PDE)
[![Release](https://img.shields.io/github/v/release/derrring/MFG_PDE)](https://github.com/derrring/MFG_PDE/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/derrring/MFG_PDE/blob/main/LICENSE)

A Python framework for solving Mean Field Games with modern numerical methods, GPU acceleration, and reinforcement learning.

> **v0.16.7** - BaseSolver convergence integration, ConvergenceConfig support

---

## Quick Start

### Installation

```bash
pip install mfg-pde
```

Or install from source:
```bash
git clone https://github.com/derrring/MFG_PDE.git
cd MFG_PDE
pip install -e .
```

### Your First MFG Solution

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Create geometry (recommended)
domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])

# Create and solve
problem = MFGProblem(geometry=domain, T=1.0, Nt=20)
result = problem.solve()
```

**That's it.** Check convergence with `result.converged` and access solutions via `result.U` and `result.M`.

---

## Key Features

- **🎯 Simple API** - From problem definition to solution in 2 lines
- **⚡ Production-Ready** - 10⁻¹⁵ mass conservation error, 98.4% test pass rate
- **🧩 Modular** - Mix and match HJB + FP solvers (FDM, Particles, WENO, Neural)
- **🌐 Multi-Dimensional** - 1D/2D/3D/nD support with automatic dimensional splitting
- **🔀 Dual Geometry** - Separate discretizations for HJB and FP (multi-resolution, FEM meshes)
- **🎮 Reinforcement Learning** - Complete RL framework (DDPG, TD3, SAC)
- **⚡ GPU Acceleration** - PyTorch, JAX, Numba backends
- **🛠️ Essential Utilities** - Particle interpolation, SDF, QP caching, convergence monitoring

---

## Documentation

**Getting Started**:
- [Getting Started Tutorial](docs/tutorials/01_getting_started.md) - 30 minutes to first solve
- [Configuration Patterns](docs/tutorials/02_configuration_patterns.md) - Three ways to configure
- [Examples](examples/) - Working code examples

**Utilities & Guides**:
- [Configuration System](docs/user/configuration_system.md) - Pydantic + OmegaConf dual architecture
- [Particle Interpolation](docs/user/particle_interpolation.md) - Grid ↔ Particles
- [SDF Utilities](docs/user/sdf_utilities.md) - Geometry and obstacles

**For Developers**:
- [Developer Guide](docs/development/) - Extending the framework
- [API Documentation](docs/) - Complete API reference

---

## Examples

### Solve Any MFG Problem

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Create geometry
domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[101])

# Create and solve MFG problem
problem = MFGProblem(geometry=domain, T=1.0, Nt=50)
result = problem.solve()

print(f"Converged: {result.converged} in {result.iterations} iterations")
```

### Configuration Options

```python
# Default settings
result = problem.solve()

# Custom parameters
result = problem.solve(
    max_iterations=200,
    tolerance=1e-8,
    verbose=True
)
```

### Essential Utilities

```python
# Particle interpolation
from mfg_pde.utils import interpolate_grid_to_particles
u_particles = interpolate_grid_to_particles(u_grid, (0, 1), particles)

# Signed distance functions
from mfg_pde.utils import sdf_sphere, sdf_box, sdf_union
obstacles = sdf_union(
    sdf_sphere(points, center=[0.3, 0.5], radius=0.1),
    sdf_box(points, bounds=[[0.6, 0.8], [0.4, 0.6]])
)

# QP caching (2-5× speedup for GFDM)
from mfg_pde.utils import QPSolver, QPCache
solver = QPSolver(backend="osqp", cache=QPCache(max_size=1000))
```

### Dual Geometry (v1.0+)

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Multi-resolution: fine HJB + coarse FP (4-15× speedup)
hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[101, 101])
fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[26, 26])

problem = MFGProblem(
    hjb_geometry=hjb_grid,  # Fine for accuracy
    fp_geometry=fp_grid,     # Coarse for speed
    T=1.0, Nt=100, sigma=0.1
)

# Projections handled automatically
result = solve_mfg(problem, config="fast")
```

```python
from mfg_pde.geometry import Mesh2D, TensorProductGrid

# Complex domains: FEM mesh + regular grid
mesh = Mesh2D(
    domain_type="rectangle",
    bounds=(0.0, 1.0, 0.0, 1.0),
    holes=[{"type": "circle", "center": (0.5, 0.5), "radius": 0.2}],
    mesh_size=0.05
)
mesh.generate_mesh()

problem = MFGProblem(
    hjb_geometry=TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[51, 51]),
    fp_geometry=mesh,  # Handles obstacles naturally
    T=1.0, Nt=50, sigma=0.1
)
```

---

## Numerical Methods

### HJB Solvers
- **Finite Difference (FDM)** - Standard grid-based discretization
- **GFDM with Monotonicity** - Generalized FDM with QP-based monotone scheme enforcement, direct Hamiltonian gradient constraints
- **Semi-Lagrangian** - Adaptive time-stepping with CFL monitoring
- **WENO** - High-order shock-capturing schemes for non-smooth solutions
- **Neural (DGM, PINN)** - Deep learning approaches for high dimensions

### Fokker-Planck Solvers
- **FDM** - Conservative finite difference schemes
- **Particle Methods** - Monte Carlo, kernel density estimation

### Operator Infrastructure
- **RBF Operators** - Radial basis function differential operators for meshless methods
- **GFDM Operators** - Generalized finite difference with polynomial basis
- **JAX Autodiff** - Automatic Jacobian computation for O(1) Newton iteration

### Geometry & Boundaries
- **SDF Utilities** - Signed distance functions with CSG operations (union, intersection, difference)
- **Boundary Normals** - SDF-based normal computation and projection
- **Dual Geometry** - Separate HJB/FP discretizations, multi-resolution support

See [Changelog](CHANGELOG.md) for version history.

---

## Optional Features

Install additional capabilities as needed:

```bash
pip install mfg-pde[neural]          # PyTorch-based neural operators, PINNs, DGM
pip install mfg-pde[reinforcement]   # RL algorithms (DDPG, TD3, SAC)
pip install mfg-pde[gpu]             # CUDA support, JAX GPU
pip install mfg-pde[performance]     # JAX backend, Numba JIT
pip install mfg-pde[all]             # Everything
```

---

## Requirements

- Python 3.12+
- NumPy, SciPy, Matplotlib (installed automatically)

Optional: PyTorch, JAX, igraph, plotly (for advanced features)

---

## Citation

If you use MFG_PDE in your research:

```bibtex
@software{mfg_pde2025,
  title={MFG\_PDE: A Research-Grade Framework for Mean Field Games},
  author={Wang, Jeremy Jiongyi},
  year={2025},
  version={0.16.7},
  url={https://github.com/derrring/MFG_PDE}
}
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

**Copyright** (c) 2025 Jeremy Jiongyi Wang
