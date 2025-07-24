# MFG_PDE Examples

This directory contains working examples demonstrating the usage of different solvers in the MFG_PDE package.

## Quick Start Examples

### 1. Basic Particle-Collocation Solver
**File**: [`basic_particle_collocation.py`](../../examples/particle_collocation_no_flux_bc.py)

The recommended approach for most applications. Combines particle methods for Fokker-Planck with generalized finite differences for Hamilton-Jacobi-Bellman.

```python
from mfg_pde import ExampleMFGProblem, BoundaryConditions
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver

# Create problem
problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=1.0, Nt=30)
boundary_conditions = BoundaryConditions(type="no_flux")

# Set up solver
solver = ParticleCollocationSolver(problem=problem, ...)
U, M, info = solver.solve(Niter=10, l2errBound=1e-3)
```

### 2. High-Performance QP-Collocation
**File**: [`optimized_qp_collocation.py`](optimized_qp_collocation.py) (to be created)

For demanding applications requiring maximum performance. Achieves 3-8x speedup with intelligent constraint detection.

### 3. Pure FDM Solver
**File**: [`pure_fdm.py`](../../examples/damped_fixed_point_pure_fdm.py)

Traditional finite difference method for reference and comparison.

### 4. Hybrid Particle-FDM
**File**: [`hybrid_method.py`](../../examples/hybrid_particle_fdm.py)

Combines particle Fokker-Planck with FDM Hamilton-Jacobi-Bellman.

## Performance Comparisons

### Three-Method Comparison
**File**: [`compare_all_methods.py`](../../examples/compare_all_no_flux_bc.py)

Comprehensive comparison of Pure FDM, Hybrid, and Particle-Collocation methods with performance metrics.

## Problem Configuration

All examples use no-flux boundary conditions and the standard MFG test problem:

```python
problem = ExampleMFGProblem(
    xmin=0.0, xmax=1.0,  # Spatial domain
    Nx=20,               # Spatial grid points
    T=1.0,               # Time horizon
    Nt=30,               # Time steps
    sigma=0.1,           # Volatility
    coefCT=0.02          # Control cost
)
```

## Expected Results

- **Mass Conservation**: < 0.1% error for all methods
- **Convergence**: Typically 3-6 fixed-point iterations
- **Performance**: QP-Collocation fastest, Pure FDM most stable
- **Accuracy**: All methods achieve similar solution quality

## Troubleshooting

- If solvers fail to converge, try reducing `l2errBound` or increasing `Niter`
- For stability issues with large problems, reduce time step (`Nt`) or spatial resolution (`Nx`)
- Memory issues: Consider using Pure FDM for very large problems

## Advanced Usage

See the [`tests/method_comparisons/`](../../tests/method_comparisons/) directory for:
- Robustness testing across parameter ranges
- Mass conservation validation
- Performance benchmarking
- Statistical analysis tools