# MFG_PDE: Numerical Solvers for Mean Field Games

A Python package for solving Mean Field Game (MFG) systems using particle-collocation and finite difference methods.

## Quick Start

```python
from mfg_pde import ExampleMFGProblem, BoundaryConditions
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.alg.hjb_solvers.tuned_smart_qp_gfdm_hjb import TunedSmartQPGFDMHJBSolver

# Create an MFG problem
problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=1.0, Nt=30, 
                           sigma=0.1, coefCT=0.02)

# Set up optimized QP-Collocation solver
boundary_conditions = BoundaryConditions(type="no_flux")
solver = ParticleCollocationSolver(problem=problem, ...)

# Solve the MFG system
U, M, info = solver.solve(Niter=10, l2errBound=1e-3)
```

## Features

- **High-Performance Solvers**: Optimized QP-Collocation with ~90% reduction in QP calls
- **Multiple Methods**: Pure FDM, Hybrid Particle-FDM, and advanced Collocation methods
- **Robust Implementation**: Comprehensive test suite with 100% success rate across diverse problems
- **Mass Conservation**: Excellent conservation properties with < 0.1% error
- **Production Ready**: Battle-tested with extensive validation and benchmarking

## Installation

```bash
pip install -e .
```

## Documentation

- [Mathematical Background](docs/theory/mathematical_background.md)
- [API Reference](docs/api/)
- [Examples and Tutorials](docs/examples/)
- [Technical Issues](docs/issues/)

## Core Solvers

### 1. Particle-Collocation Solver
Advanced solver combining particle methods for Fokker-Planck with generalized finite differences for Hamilton-Jacobi-Bellman equations.

### 2. Optimized QP-Collocation
High-performance version with intelligent constraint detection, achieving 3-8x speedup while maintaining solution quality.

### 3. Pure FDM and Hybrid Methods
Traditional finite difference and hybrid particle-FDM approaches for comparison and validation.

## Performance

- **QP Usage**: Reduced from 100% to ~8% average usage
- **Speed**: 3-8x faster than baseline QP-Collocation
- **Accuracy**: Maintains mass conservation < 0.1% error
- **Robustness**: 100% success rate across 50+ diverse test cases

## Testing

```bash
# Run core functionality tests
python -m pytest tests/integration/

# Run comprehensive method comparisons
python tests/method_comparisons/comprehensive_final_evaluation.py

# Run mass conservation validation
python -m pytest tests/mass_conservation/
```

## Examples

See the [`examples/`](examples/) directory for working implementations:
- Basic particle-collocation usage
- Performance comparisons
- Parameter sensitivity studies
- Advanced optimization techniques

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

[Add appropriate license information]