# MFGArchon: Mean Field Games Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://github.com/derrring/mfgarchon/actions/workflows/ci.yml/badge.svg)](https://github.com/derrring/mfgarchon/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/derrring/mfgarchon/graph/badge.svg?token=HGZFRSF5V6)](https://codecov.io/github/derrring/mfgarchon)
[![Release](https://img.shields.io/github/v/release/derrring/mfgarchon)](https://github.com/derrring/mfgarchon/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/derrring/mfgarchon/blob/main/LICENSE)

A Python framework for solving Mean Field Game systems using modern numerical methods, GPU acceleration, and reinforcement learning.

---

## Quick Start

### Installation

```bash
git clone https://github.com/derrring/mfgarchon.git
cd mfgarchon
pip install -e .
```

### Your First MFG Solution

```python
from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc

# Create geometry with boundary conditions
domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51],
                           boundary_conditions=neumann_bc(dimension=1))

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

**Tutorials**:
- [Getting Started](examples/tutorials/01_getting_started.md) - First solve in 30 minutes
- [Configuration Patterns](examples/tutorials/02_configuration_patterns.md) - Three ways to configure
- [Examples](examples/) - Working code examples

**Guides**:
- [Configuration System](docs/user/configuration_system.md) - Pydantic + OmegaConf dual architecture
- [Particle Interpolation](docs/user/particle_interpolation.md) - Grid-particle transfer
- [SDF Utilities](docs/user/sdf_utilities.md) - Geometry and obstacles
- [Developer Guide](docs/development/) - Extending the framework

---

## Examples

### Custom Solver Parameters

```python
result = problem.solve(
    max_iterations=200,
    tolerance=1e-8,
    verbose=True
)

print(f"Converged: {result.converged} in {result.iterations} iterations")
```

### Dual Geometry

```python
from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# Multi-resolution: fine HJB + coarse FP
hjb_grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[101, 101],
                              boundary_conditions=no_flux_bc(dimension=2))
fp_grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[26, 26],
                             boundary_conditions=no_flux_bc(dimension=2))

problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_grid,
    T=1.0, Nt=100, diffusion=0.1
)

# Projections handled automatically
result = problem.solve()
```

### Utilities

```python
# Particle interpolation
from mfgarchon.utils import interpolate_grid_to_particles
u_particles = interpolate_grid_to_particles(u_grid, (0, 1), particles)

# Signed distance functions
from mfgarchon.utils import sdf_sphere, sdf_box, sdf_union
obstacles = sdf_union(
    sdf_sphere(points, center=[0.3, 0.5], radius=0.1),
    sdf_box(points, bounds=[[0.6, 0.8], [0.4, 0.6]])
)

# QP caching (2-5x speedup for GFDM)
from mfgarchon.utils import QPSolver, QPCache
solver = QPSolver(backend="osqp", cache=QPCache(max_size=1000))
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

## Optional Dependencies

Additional capabilities can be enabled by installing optional packages:

- **Neural solvers** (PINNs, DGM): PyTorch
- **Reinforcement learning** (DDPG, TD3, SAC): Stable-Baselines3
- **GPU acceleration**: JAX with CUDA, PyTorch CUDA
- **Performance**: JAX backend, Numba JIT
- **Mesh generation**: Gmsh (for Mesh2D/Mesh3D geometry)

---

## Requirements

- Python 3.12+
- NumPy, SciPy, Matplotlib (installed automatically)

Optional: PyTorch, JAX, igraph, plotly (for advanced features)

---

## Citation

If you use MFGArchon in your research, please cite it. You can use GitHub's
"Cite this repository" button in the sidebar, or use the following BibTeX:

```bibtex
@software{mfgarchon,
  title={MFGArchon: A Research-Grade Framework for Mean Field Games},
  author={Wang, Jiongyi},
  year={2025},
  url={https://github.com/derrring/mfgarchon}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable citation metadata.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

**Copyright** (c) 2025-2026 Jeremy Jiongyi Wang
