# MFG_PDE Tutorials

A structured 5-step learning path for getting started with Mean Field Games in Python.

## Format

Each tutorial is available in two formats:
- **Jupyter Notebook** (`.ipynb`) - **Recommended** for interactive learning with inline explanations
- **Python Script** (`.py`) - For command-line execution and integration into workflows

## Learning Path

Complete these tutorials in order to build a solid foundation:

### 01. [Hello MFG](./01_hello_mfg.ipynb) | [script](./01_hello_mfg.py)
**Difficulty**: Beginner
**Time**: 10 minutes

Your first Mean Field Game using the simplest possible setup.

**You'll learn**:
- How to create an MFG problem with `MFGProblem`
- How to solve it using `solve_mfg()`
- How to inspect solutions (U, M, convergence)
- How to check mass conservation

**Mathematical problem**: Linear-Quadratic MFG on [0,1]

---

### 02. [Custom Hamiltonian](./02_custom_hamiltonian.ipynb) | [script](./02_custom_hamiltonian.py)
**Difficulty**: Beginner
**Time**: 15 minutes

Learn to define your own MFG problems from scratch.

**You'll learn**:
- How to subclass `MFGProblem`
- How to define custom Hamiltonians H(x, p, m)
- How to specify terminal costs and initial density
- How to model congestion effects

**Mathematical problem**: Crowd evacuation with congestion penalties

---

### 03. [2D Geometry](./03_2d_geometry.ipynb) | [script](./03_2d_geometry.py)
**Difficulty**: Intermediate
**Time**: 20 minutes

Move from 1D to 2D spatial domains.

**You'll learn**:
- How to use `MFGProblem` for nD problems with `spatial_bounds` and `spatial_discretization`
- How to work with 2D gradients p = [px, py]
- How to compute 2D integrals for mass conservation
- How to visualize 2D density evolution

**Mathematical problem**: 2D target attraction (agents navigate to center)

---

### 04. [Particle Methods](./04_particle_methods.ipynb) | [script](./04_particle_methods.py)
**Difficulty**: Intermediate
**Time**: 25 minutes

Explore alternative solver backends using particle-based methods.

**You'll learn**:
- The difference between grid-based (FDM) and particle-based solvers
- How to configure particle methods with `ConfigBuilder`
- How particle count affects accuracy (bias-variance tradeoff)
- When to use particles vs grids

**Mathematical problem**: 1D Linear-Quadratic MFG (comparing solvers)

---

### 05. [ConfigBuilder System](./05_config_system.ipynb) | [script](./05_config_system.py)
**Difficulty**: Intermediate
**Time**: 20 minutes

Master the configuration API for advanced solver control.

**You'll learn**:
- How to build solver configurations with `ConfigBuilder`
- How to choose between solver backends (FDM, GFDM, particles)
- How to configure coupling methods (Picard, Policy Iteration)
- How to enable acceleration (JAX, GPU)
- Best practices for configuration selection

**Mathematical problem**: Comparing different solver configurations

---

## Quick Start

```bash
# Run all tutorials
python examples/tutorials/01_hello_mfg.py
python examples/tutorials/02_custom_hamiltonian.py
python examples/tutorials/03_2d_geometry.py
python examples/tutorials/04_particle_methods.py
python examples/tutorials/05_config_system.py
```

Each tutorial:
- Is self-contained and runnable
- Produces visualizations (saved to `examples/outputs/tutorials/`)
- Includes detailed explanations and mathematical background
- Takes 10-25 minutes to complete

## Prerequisites

**Required**:
- Python 3.9+
- NumPy, SciPy
- MFG_PDE installed (`pip install -e .`)

**Optional** (for visualizations):
- Matplotlib

**Optional** (for acceleration):
- JAX (for GPU/TPU acceleration)

## Output Files

Tutorial visualizations are saved to:
```
examples/outputs/tutorials/
├── 01_hello_mfg.png
├── 02_custom_hamiltonian.png
├── 03_2d_geometry.png
├── 04_particle_methods.png
└── 05_config_system.png
```

## After Completing Tutorials

### Explore Examples

**Basic examples** (single-concept demos):
- `examples/basic/core_infrastructure/` - Core API usage
- `examples/basic/geometry/` - Domain and geometry demos
- `examples/basic/solvers/` - Solver comparisons
- `examples/basic/utilities/` - Tools and utilities

**Advanced examples** (multi-feature demos):
- `examples/advanced/applications/` - Real-world applications
- `examples/advanced/solvers_advanced/` - Advanced numerical methods
- `examples/advanced/machine_learning/` - RL and neural network solvers

### Read Documentation

- `docs/` - Mathematical theory and API reference
- `README.md` - Project overview
- `CLAUDE.md` - Development conventions

### Join the Community

- GitHub Issues: https://github.com/derrring/MFG_PDE/issues
- Discussions: https://github.com/derrring/MFG_PDE/discussions

## Troubleshooting

**Problem**: ImportError when running tutorials
**Solution**: Install MFG_PDE: `pip install -e .` from repo root

**Problem**: No visualizations generated
**Solution**: Install matplotlib: `pip install matplotlib`

**Problem**: Solver doesn't converge
**Solution**: Increase `max_iterations` or loosen `tolerance` in ConfigBuilder

**Problem**: "JAX not available" warning
**Solution**: Optional - install JAX for acceleration: `pip install jax jaxlib`

## Contributing

Found an issue or have a suggestion? Please open an issue on GitHub!

## License

Same as MFG_PDE (see repository root LICENSE file)
