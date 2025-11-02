# MFG_PDE: Mean Field Games Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://github.com/derrring/MFG_PDE/actions/workflows/ci.yml/badge.svg)](https://github.com/derrring/MFG_PDE/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/derrring/MFG_PDE/graph/badge.svg?token=HGZFRSF5V6)](https://codecov.io/github/derrring/MFG_PDE)
[![Release](https://img.shields.io/github/v/release/derrring/MFG_PDE)](https://github.com/derrring/MFG_PDE/releases)
[![License](https://img.shields.io/github/license/derrring/MFG_PDE)](https://github.com/derrring/MFG_PDE/blob/main/LICENSE)

A Python framework for solving Mean Field Games with modern numerical methods, GPU acceleration, and reinforcement learning.

> **✨ v0.9.0 Released** - [What's New](https://github.com/derrring/MFG_PDE/releases/tag/v0.9.0): Unified API, Essential Utilities, 2,300+ lines of documentation

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

### Your First MFG Solution (2 lines)

```python
from mfg_pde import solve_mfg, ExampleMFGProblem

result = solve_mfg(ExampleMFGProblem(), config="fast")
```

**That's it.** Check convergence with `result.converged` and visualize with `result.plot()`.

---

## Key Features

- **🎯 Simple API** - From problem definition to solution in 2 lines
- **⚡ Production-Ready** - 10⁻¹⁵ mass conservation error, 98.4% test pass rate
- **🧩 Modular** - Mix and match HJB + FP solvers (FDM, Particles, WENO, Neural)
- **🌐 Multi-Dimensional** - 1D/2D/3D/nD support with automatic dimensional splitting
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
- [Particle Interpolation](docs/user_guides/particle_interpolation.md) - Grid ↔ Particles
- [SDF Utilities](docs/user_guides/sdf_utilities.md) - Geometry and obstacles
- [Migration Guide](docs/migration/PHASE_3_MIGRATION_GUIDE.md) - Upgrading from v0.8.x

**For Developers**:
- [Developer Guide](docs/development/) - Extending the framework
- [API Documentation](docs/) - Complete API reference

---

## Examples

### Solve Any MFG Problem

```python
from mfg_pde import solve_mfg, create_lq_problem

# Linear-Quadratic MFG
problem = create_lq_problem(T=1.0, Nt=50, Nx=100)
result = solve_mfg(problem, config="accurate")

print(f"Converged: {result.converged} in {result.iterations} iterations")
```

### Three Configuration Patterns

```python
# Pattern 1: Preset strings (quickest)
result = solve_mfg(problem, config="fast")

# Pattern 2: Builder API (flexible)
from mfg_pde.config import ConfigBuilder
config = ConfigBuilder().picard(tolerance=1e-8).build()
result = solve_mfg(problem, config=config)

# Pattern 3: YAML files (reproducible)
result = solve_mfg(problem, config="experiments/config.yaml")
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

---

## What's New in v0.9.0

**Unified Architecture**:
- Single `solve_mfg()` interface replacing multiple solver creation patterns
- Three configuration patterns: Presets, Builder API, YAML files
- Simplified from 10+ lines to 2 lines for typical use cases

**Essential Utilities** (saves ~610 lines per project):
- Particle interpolation (grid ↔ particles, 1D/2D/3D)
- Signed distance functions (primitives, CSG operations, smooth blending)
- QP solver caching (2-5× GFDM speedup)
- Convergence monitoring (plotting, stagnation detection)

**Documentation**:
- 2,300+ lines of new user-facing documentation
- Migration guide for upgrading from v0.8.x
- Complete tutorials and user guides

See [Release Notes](https://github.com/derrring/MFG_PDE/releases/tag/v0.9.0) for full details.

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
  version={0.9.0},
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
