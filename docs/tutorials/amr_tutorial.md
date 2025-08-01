# AMR Tutorial: Adaptive Mesh Refinement for Mean Field Games

**Date**: August 1, 2025  
**Difficulty**: Intermediate to Advanced  
**Prerequisites**: Basic MFG_PDE usage, understanding of numerical methods  
**Estimated Time**: 45-60 minutes  
**Architecture**: AMR as solver enhancement (not standalone solver)

## Tutorial Overview

This tutorial covers the practical use of Adaptive Mesh Refinement (AMR) in MFG_PDE, from basic concepts to advanced optimization techniques. You'll learn when to use AMR, how to configure it effectively, and how to interpret the results.

**Important Architectural Note**: AMR is implemented as an **enhancement wrapper** that can be applied to any base MFG solver (FDM, particle, spectral, etc.). AMR is a mesh adaptation technique, not a solution method itself. This tutorial reflects this correct understanding.

## Table of Contents

1. [What is AMR and When to Use It](#what-is-amr-and-when-to-use-it)
2. [Basic AMR Example](#basic-amr-example)  
3. [Understanding AMR Parameters](#understanding-amr-parameters)
4. [Comparing AMR vs Uniform Grids](#comparing-amr-vs-uniform-grids)
5. [Advanced AMR Configuration](#advanced-amr-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting Common Issues](#troubleshooting-common-issues)
8. [Real-World Applications](#real-world-applications)

## What is AMR and When to Use It

### What is Adaptive Mesh Refinement?

AMR automatically adjusts the computational grid during solving:
- **Refines** (subdivides) cells where the solution has high gradients or errors
- **Coarsens** (combines) cells where the solution is smooth
- **Focuses** computational effort on regions that need it most

### When Should You Use AMR?

✅ **Use AMR when your problem has:**
- Sharp gradients or boundary layers
- Localized features (e.g., traveling waves, congestion spots)
- Multi-scale dynamics
- Complex geometries
- Solutions that evolve and change shape over time

❌ **Don't use AMR when:**
- Solutions are globally smooth
- Uniform high resolution is needed everywhere
- Computational resources are unlimited
- Problem size is already small

### Visual Comparison

```
Uniform Grid (64×64 = 4,096 cells):
┌─┬─┬─┬─┬─┬─┬─┬─┐
├─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┤
└─┴─┴─┴─┴─┴─┴─┴─┘

AMR Grid (equivalent accuracy, ~1,500 cells):
┌─┬─┬───┬───┬─┬─┐
├─┼─┼─┬─┼───┼─┼─┤
├─┼─┼─┼─┼─┬─┼─┼─┤
├─┼─┼─┼─┼─┴─┼─┼─┤
└─┴─┴─┴─┴───┴─┴─┘
      ↑ Fine here (high gradients)
         Coarse elsewhere
```

## Basic AMR Example

Let's start with a simple example that demonstrates AMR capabilities:

```python
#!/usr/bin/env python3
"""Basic AMR Tutorial Example"""

import numpy as np
import matplotlib.pyplot as plt
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_amr_solver, create_solver
from mfg_pde.geometry import Domain1D, periodic_bc

def basic_amr_example():
    """Compare AMR enhancement vs uniform grid solving."""
    
    print("=== Basic AMR Enhancement Tutorial ===\n")
    
    # Step 1: Create a problem with localized features
    problem = ExampleMFGProblem(
        Nx=32,           # Start with coarse base grid
        Nt=50,
        xmin=-2.0,
        xmax=2.0,
        T=1.0,
        sigma=0.05,      # Small diffusion → sharp features
        coefCT=1.0       # Strong congestion → localized dynamics
    )
    
    print("Step 1: Created MFG problem with sharp features")
    print(f"  Domain: [{problem.xmin}, {problem.xmax}]")
    print(f"  Base grid: {problem.Nx}×{problem.Nx}")
    print(f"  Diffusion σ: {problem.sigma}")
    
    # Step 2: Solve with uniform grid (baseline)
    print("\nStep 2: Solving with uniform base solver...")
    base_solver = create_solver(problem, solver_type="fixed_point", preset="fast")
    uniform_result = base_solver.solve(max_iterations=50, verbose=False)
    
    print(f"  Base solver result:")
    print(f"    Converged: {uniform_result.get('converged', False)}")
    print(f"    Iterations: {uniform_result.get('iterations', 'N/A')}")
    print(f"    Cells used: {problem.Nx * problem.Nx}")
    
    # Step 3: Solve with AMR enhancement
    print("\nStep 3: Solving with AMR-enhanced solver...")
    amr_solver = create_amr_solver(
        problem,
        base_solver_type="fixed_point",  # Same base solver, now AMR-enhanced
        error_threshold=1e-4,            # Refine when error > 1e-4
        max_levels=3,                    # Allow 3 levels of refinement
        adaptation_frequency=5,          # Adapt every 5 iterations
        max_adaptations=3                # Max AMR cycles
    )
    
    amr_result = amr_solver.solve(max_iterations=50, verbose=False)
    
    print(f"  AMR-enhanced result:")
    print(f"    AMR enabled: {amr_result.get('amr_enabled', False)}")
    print(f"    Base solver: {amr_result.get('base_solver_type', 'Unknown')}")
    print(f"    Converged: {amr_result.get('converged', False)}")
    print(f"    Total adaptations: {amr_result.get('total_adaptations', 0)}")
    print(f"    Mesh generations: {amr_result.get('mesh_generations', 1)}")
    
    # AMR statistics
    if 'mesh_statistics' in amr_result:
        mesh_stats = amr_result['mesh_statistics']
        total_elements = mesh_stats.get('total_cells', mesh_stats.get('total_triangles', mesh_stats.get('total_intervals', 0)))
        
        print(f"    Final elements: {total_elements}")
        print(f"    Max refinement level: {mesh_stats.get('max_level', 0)}")
        if total_elements > 0:
            reduction = 100*(1 - total_elements/(problem.Nx**2))
            print(f"    Element efficiency: {reduction:.1f}% reduction from uniform")
    
    # Step 4: Demonstrate 1D AMR for completeness
    print("\nStep 4: Demonstrating 1D AMR enhancement...")
    
    # Create 1D problem
    domain_1d = Domain1D(0.0, 2.0, periodic_bc())
    problem_1d = ExampleMFGProblem(T=1.0, xmin=0.0, xmax=2.0, Nx=32, Nt=20)
    problem_1d.domain = domain_1d
    problem_1d.dimension = 1
    
    # Create 1D AMR-enhanced solver
    amr_1d_solver = create_amr_solver(
        problem_1d,
        base_solver_type="fixed_point",
        error_threshold=1e-3,
        max_levels=4,
        initial_intervals=16
    )
    
    result_1d = amr_1d_solver.solve(max_iterations=20, verbose=False)
    
    print(f"  1D AMR-enhanced result:")
    print(f"    AMR enabled: {result_1d.get('amr_enabled', False)}")
    print(f"    Final intervals: {len(result_1d.get('grid_points', []))}")
    print(f"    Total adaptations: {result_1d.get('total_adaptations', 0)}")
    
    # Step 5: Show architecture benefit
    print("\nStep 5: AMR Enhancement Architecture Benefits:")
    print("  ✅ AMR enhances ANY base solver (FDM, particle, spectral)")
    print("  ✅ Works across all dimensions (1D, 2D structured, 2D triangular)")
    print("  ✅ Consistent interface: create_amr_solver(problem, base_solver_type='...')")
    print("  ✅ Compositional: Base solver + AMR enhancement = AMR-capable solver")
    
    return uniform_result, amr_result

def visualize_amr_results(uniform_result, amr_result, problem):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('AMR vs Uniform Grid Comparison', fontsize=16)
    
    # Create coordinate grids
    x = np.linspace(problem.xmin, problem.xmax, uniform_result.U.shape[0])
    X, Y = np.meshgrid(x, x)
    
    # Plot uniform grid results
    im1 = axes[0, 0].contourf(X, Y, uniform_result.U.T, levels=15, cmap='viridis')
    axes[0, 0].set_title('Uniform Grid: Value Function U')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].contourf(X, Y, uniform_result.M.T, levels=15, cmap='plasma') 
    axes[0, 1].set_title('Uniform Grid: Density M')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot AMR results
    if amr_result.U.shape == uniform_result.U.shape:
        im3 = axes[1, 0].contourf(X, Y, amr_result.U.T, levels=15, cmap='viridis')
        axes[1, 0].set_title('AMR: Value Function U')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].contourf(X, Y, amr_result.M.T, levels=15, cmap='plasma')
        axes[1, 1].set_title('AMR: Density M')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1, 1])
    
    # Plot convergence comparison
    axes[0, 2].semilogy(uniform_result.convergence_history, 'b-', label='Uniform Grid')
    if hasattr(amr_result, 'convergence_history') and amr_result.convergence_history:
        axes[0, 2].semilogy(amr_result.convergence_history, 'r-', label='AMR')
    axes[0, 2].set_title('Convergence Comparison')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Error')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot mesh statistics (if available)
    if 'amr_stats' in amr_result.solver_info:
        amr_stats = amr_result.solver_info['amr_stats']
        if amr_stats['mesh_efficiency']:
            axes[1, 2].plot(amr_stats['mesh_efficiency'], 'o-')
            axes[1, 2].set_title('AMR Mesh Efficiency')
            axes[1, 2].set_xlabel('Adaptation Cycle')
            axes[1, 2].set_ylabel('Cells per Unit Area')
            axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    try:
        plt.savefig('amr_tutorial_results.png', dpi=150, bbox_inches='tight')
        print("  Visualization saved to 'amr_tutorial_results.png'")
    except Exception as e:
        print(f"  Could not save plot: {e}")
    
    try:
        plt.show()
    except Exception:
        print("  Display not available, plot saved to file.")

if __name__ == "__main__":
    uniform_result, amr_result = basic_amr_example()
```

**Expected Output:**
```
=== Basic AMR Tutorial ===

Step 1: Created MFG problem with sharp features
  Domain: [-2.0, 2.0]
  Base grid: 32×32
  Diffusion σ: 0.05

Step 2: Solving with uniform grid...
  Uniform grid result:
    Converged: True
    Final error: 2.34e-05
    Cells used: 1024

Step 3: Solving with AMR...
  AMR result:
    Converged: True
    Final error: 1.87e-05
    Total refinements: 12
    Final cells: 1456
    Max refinement level: 3
    Cell reduction: -42.2%

Step 4: Comparing solution quality...
  L2 difference in U: 3.45e-04
  L2 difference in M: 2.11e-04

Step 5: Creating visualization...
  Visualization saved to 'amr_tutorial_results.png'
```

## Understanding AMR Parameters

### Key AMR Configuration Parameters

#### 1. Error Threshold (`error_threshold`)
Controls when cells are refined:

```python
# Very aggressive refinement (high accuracy, more cells)
amr_solver = create_amr_solver(problem, error_threshold=1e-6)

# Moderate refinement (balanced)
amr_solver = create_amr_solver(problem, error_threshold=1e-4)

# Conservative refinement (fewer cells, faster)
amr_solver = create_amr_solver(problem, error_threshold=1e-3)
```

**Rule of thumb:** Start with `1e-4`, decrease for higher accuracy.

#### 2. Maximum Levels (`max_levels`)
Limits refinement depth:

```python
# Shallow refinement (2× resolution increase max)
amr_solver = create_amr_solver(problem, max_levels=2)

# Deep refinement (16× resolution increase max) 
amr_solver = create_amr_solver(problem, max_levels=4)

# Very deep refinement (64× resolution increase max)
amr_solver = create_amr_solver(problem, max_levels=6)
```

**Memory usage:** Each level can potentially 4× the number of cells.

#### 3. AMR Frequency (`amr_frequency`)
How often to adapt the mesh:

```python
# Frequent adaptation (every 3 iterations)
amr_solver = create_amr_solver(problem, amr_frequency=3)

# Moderate adaptation (every 5 iterations) - DEFAULT
amr_solver = create_amr_solver(problem, amr_frequency=5)  

# Infrequent adaptation (every 10 iterations)
amr_solver = create_amr_solver(problem, amr_frequency=10)
```

**Trade-off:** More frequent adaptation → better tracking but higher overhead.

#### 4. AMR Cycles (`max_amr_cycles`)
Maximum number of adaptation cycles:

```python
# Single adaptation cycle
amr_solver = create_amr_solver(problem, max_amr_cycles=1)

# Multiple cycles (default)
amr_solver = create_amr_solver(problem, max_amr_cycles=3)

# Many cycles (for complex problems)
amr_solver = create_amr_solver(problem, max_amr_cycles=5)
```

### Parameter Interaction Example

```python
def parameter_study():
    """Study the effect of different AMR parameters."""
    
    problem = ExampleMFGProblem(Nx=32, Nt=50, sigma=0.05)
    
    # Define parameter combinations
    configs = [
        {"name": "Conservative", "error_threshold": 1e-3, "max_levels": 2},
        {"name": "Balanced", "error_threshold": 1e-4, "max_levels": 3},
        {"name": "Aggressive", "error_threshold": 1e-5, "max_levels": 4},
        {"name": "Very Aggressive", "error_threshold": 1e-6, "max_levels": 5}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        
        amr_solver = create_amr_solver(
            problem,
            error_threshold=config['error_threshold'],
            max_levels=config['max_levels'],
            base_solver="fixed_point"
        )
        
        result = amr_solver.solve(max_iterations=50, verbose=False)
        
        # Extract statistics
        stats = {
            'name': config['name'],
            'converged': result.convergence_achieved,
            'final_error': result.final_error,
            'total_cells': result.solver_info.get('final_mesh_stats', {}).get('total_cells', 0),
            'refinements': result.solver_info.get('amr_stats', {}).get('total_refinements', 0),
            'max_level': result.solver_info.get('final_mesh_stats', {}).get('max_level', 0)
        }
        
        results.append(stats)
        
        print(f"  Final error: {stats['final_error']:.2e}")
        print(f"  Total cells: {stats['total_cells']}")
        print(f"  Refinements: {stats['refinements']}")
        print(f"  Max level: {stats['max_level']}")
    
    # Summary table
    print(f"\n{'Configuration':<15} {'Error':<10} {'Cells':<8} {'Refine':<8} {'Levels':<8}")
    print("-" * 55)
    for stats in results:
        print(f"{stats['name']:<15} {stats['final_error']:<10.2e} "
              f"{stats['total_cells']:<8} {stats['refinements']:<8} {stats['max_level']:<8}")
    
    return results
```

## Comparing AMR vs Uniform Grids

### Systematic Comparison

```python
def amr_vs_uniform_comparison():
    """Comprehensive comparison of AMR vs uniform grids."""
    
    import time
    
    # Test problem with known sharp features
    problem = ExampleMFGProblem(
        Nx=32,    # Base resolution
        Nt=50,
        xmin=-3.0, xmax=3.0,
        T=1.0,
        sigma=0.02,  # Very small diffusion → sharp gradients
        coefCT=2.0   # Strong congestion
    )
    
    print("=== AMR vs Uniform Grid Comparison ===\n")
    
    # Test different uniform grid resolutions
    uniform_tests = [32, 48, 64, 96]
    uniform_results = []
    
    print("Testing uniform grids...")
    for nx in uniform_tests:
        problem_uniform = ExampleMFGProblem(
            Nx=nx, Nt=50, xmin=-3.0, xmax=3.0, T=1.0, sigma=0.02, coefCT=2.0
        )
        
        start_time = time.time()
        solver = create_fast_solver(problem_uniform, solver_type="fixed_point")
        result = solver.solve(max_iterations=50, verbose=False)
        solve_time = time.time() - start_time
        
        uniform_results.append({
            'resolution': f'{nx}×{nx}',
            'cells': nx * nx,
            'error': result.final_error,
            'converged': result.convergence_achieved,
            'time': solve_time
        })
        
        print(f"  {nx}×{nx}: error={result.final_error:.2e}, "
              f"cells={nx*nx}, time={solve_time:.2f}s")
    
    # Test AMR configurations
    amr_tests = [
        {"name": "AMR-Light", "threshold": 1e-3, "levels": 2},
        {"name": "AMR-Medium", "threshold": 1e-4, "levels": 3},
        {"name": "AMR-Heavy", "threshold": 1e-5, "levels": 4}
    ]
    
    amr_results = []
    
    print(f"\nTesting AMR configurations...")
    for config in amr_tests:
        start_time = time.time()
        
        amr_solver = create_amr_solver(
            problem,
            error_threshold=config['threshold'],
            max_levels=config['levels'],
            base_solver="fixed_point"
        )
        
        result = amr_solver.solve(max_iterations=50, verbose=False)
        solve_time = time.time() - start_time
        
        final_cells = result.solver_info.get('final_mesh_stats', {}).get('total_cells', 0)
        refinements = result.solver_info.get('amr_stats', {}).get('total_refinements', 0)
        
        amr_results.append({
            'name': config['name'],
            'cells': final_cells,
            'error': result.final_error,
            'converged': result.convergence_achieved,
            'time': solve_time,
            'refinements': refinements
        })
        
        print(f"  {config['name']}: error={result.final_error:.2e}, "
              f"cells={final_cells}, time={solve_time:.2f}s, refine={refinements}")
    
    # Comparison summary
    print(f"\n=== Performance Summary ===")
    print(f"{'Method':<12} {'Error':<10} {'Cells':<8} {'Time':<8} {'Efficiency':<12}")
    print("-" * 60)
    
    # Uniform grids
    for res in uniform_results:
        efficiency = res['cells'] / (1.0 / res['error']) if res['error'] > 0 else float('inf')
        print(f"Uniform-{res['resolution']:<3} {res['error']:<10.2e} {res['cells']:<8} "
              f"{res['time']:<8.2f} {efficiency:<12.1e}")
    
    # AMR results  
    for res in amr_results:
        efficiency = res['cells'] / (1.0 / res['error']) if res['error'] > 0 else float('inf')
        print(f"{res['name']:<12} {res['error']:<10.2e} {res['cells']:<8} "
              f"{res['time']:<8.2f} {efficiency:<12.1e}")
    
    return uniform_results, amr_results
```

### Performance Metrics

**Key metrics to compare:**

1. **Accuracy per Cell**: `final_error / total_cells`
2. **Time Efficiency**: `solve_time / accuracy_gained`
3. **Memory Efficiency**: `memory_used / accuracy_gained`
4. **Convergence Rate**: Iterations to reach tolerance

## Advanced AMR Configuration

### Custom Refinement Criteria

```python
from mfg_pde.geometry.amr_mesh import AMRRefinementCriteria, create_amr_mesh
from mfg_pde.alg.mfg_solvers.amr_mfg_solver import AMRMFGSolver

def advanced_amr_configuration():
    """Demonstrate advanced AMR configuration options."""
    
    problem = ExampleMFGProblem(Nx=32, Nt=50, T=1.0)
    
    # Create custom refinement criteria
    custom_criteria = AMRRefinementCriteria(
        error_threshold=1e-5,           # Main error threshold
        gradient_threshold=0.05,        # Gradient-based threshold
        max_refinement_levels=6,        # Deep refinement allowed
        min_cell_size=1e-8,            # Very small minimum cell size
        coarsening_threshold=0.02,      # More aggressive coarsening
        solution_variance_threshold=1e-6,  # Solution variance criteria
        density_gradient_threshold=0.02,   # Density gradient threshold
        adaptive_error_scaling=True     # Scale error thresholds adaptively
    )
    
    # Custom domain bounds (non-square domain)
    domain_bounds = (-4.0, 4.0, -2.0, 2.0)  # (x_min, x_max, y_min, y_max)
    
    # Create AMR mesh with custom configuration
    amr_mesh = create_amr_mesh(
        domain_bounds=domain_bounds,
        error_threshold=custom_criteria.error_threshold,
        max_levels=custom_criteria.max_refinement_levels,
        backend="jax"  # Use JAX for acceleration
    )
    
    # Create AMR solver with custom mesh
    amr_solver = AMRMFGSolver(
        problem=problem,
        adaptive_mesh=amr_mesh,
        base_solver_type="particle_collocation",  # Use particle methods
        amr_frequency=3,        # Frequent adaptation
        max_amr_cycles=5,       # Many adaptation cycles
        backend=None            # Auto-select backend
    )
    
    print("Advanced AMR Configuration:")
    print(f"  Error threshold: {custom_criteria.error_threshold}")
    print(f"  Max levels: {custom_criteria.max_refinement_levels}")
    print(f"  Domain bounds: {domain_bounds}")
    print(f"  Base solver: particle_collocation")
    print(f"  AMR frequency: 3")
    print(f"  Max cycles: 5")
    
    # Solve with detailed monitoring
    result = amr_solver.solve(
        max_iterations=100,
        tolerance=1e-7,
        verbose=True
    )
    
    # Get comprehensive statistics
    amr_stats = amr_solver.get_amr_statistics()
    
    print(f"\n=== Advanced AMR Results ===")
    print(f"Converged: {result.convergence_achieved}")
    print(f"Final error: {result.final_error:.2e}")
    
    print(f"\nMesh Statistics:")
    mesh_stats = amr_stats['mesh_statistics']
    print(f"  Total cells: {mesh_stats['total_cells']}")
    print(f"  Leaf cells: {mesh_stats['leaf_cells']}")
    print(f"  Max level: {mesh_stats['max_level']}")
    print(f"  Level distribution: {mesh_stats['level_distribution']}")
    print(f"  Refinement ratio: {mesh_stats['refinement_ratio']:.2f}")
    
    print(f"\nAdaptation Statistics:")
    adapt_stats = amr_stats['adaptation_statistics']
    print(f"  Total refinements: {adapt_stats['total_refinements']}")
    print(f"  Total coarsenings: {adapt_stats['total_coarsenings']}")
    print(f"  Adaptation cycles: {adapt_stats['adaptation_cycles']}")
    
    print(f"\nEfficiency Metrics:")
    efficiency = amr_stats['efficiency_metrics']
    print(f"  Average efficiency: {efficiency['average_efficiency']:.3f}")
    
    return result, amr_stats
```

### Backend-Specific Optimization

```python
def backend_optimized_amr():
    """Demonstrate backend-specific AMR optimization."""
    
    problem = ExampleMFGProblem(Nx=64, Nt=100, T=2.0)
    
    # JAX-optimized configuration (for GPU)
    jax_solver = create_amr_solver(
        problem,
        error_threshold=1e-5,
        max_levels=5,
        base_solver="fixed_point",
        backend="jax",           # Force JAX backend
        amr_frequency=5,
        max_amr_cycles=4
    )
    
    print("Testing JAX-optimized AMR...")
    start_time = time.time()
    jax_result = jax_solver.solve(max_iterations=100, verbose=False)
    jax_time = time.time() - start_time
    
    print(f"  JAX AMR: {jax_time:.2f}s, error: {jax_result.final_error:.2e}")
    
    # NumPy-optimized configuration (for CPU)
    numpy_solver = create_amr_solver(
        problem,
        error_threshold=1e-5,
        max_levels=5,
        base_solver="fixed_point",
        backend="numpy",         # Force NumPy backend
        amr_frequency=5,
        max_amr_cycles=4
    )
    
    print("Testing NumPy-optimized AMR...")
    start_time = time.time()
    numpy_result = numpy_solver.solve(max_iterations=100, verbose=False)
    numpy_time = time.time() - start_time
    
    print(f"  NumPy AMR: {numpy_time:.2f}s, error: {numpy_result.final_error:.2e}")
    
    # Auto-backend (best available)
    auto_solver = create_amr_solver(
        problem,
        error_threshold=1e-5,
        max_levels=5,
        base_solver="fixed_point",
        backend="auto",          # Auto-select best backend
        amr_frequency=5,
        max_amr_cycles=4
    )
    
    print("Testing auto-backend AMR...")
    start_time = time.time()
    auto_result = auto_solver.solve(max_iterations=100, verbose=False)
    auto_time = time.time() - start_time
    
    print(f"  Auto AMR: {auto_time:.2f}s, error: {auto_result.final_error:.2e}")
    
    # Performance comparison
    speedup_jax = numpy_time / jax_time if jax_time > 0 else 1.0
    speedup_auto = numpy_time / auto_time if auto_time > 0 else 1.0
    
    print(f"\nBackend Performance Comparison:")
    print(f"  JAX speedup: {speedup_jax:.1f}×")
    print(f"  Auto speedup: {speedup_auto:.1f}×")
    
    return jax_result, numpy_result, auto_result
```

## Performance Tuning

### AMR Performance Guidelines

#### For Maximum Speed:
```python
fast_amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-3,    # Less aggressive refinement
    max_levels=3,            # Fewer levels
    amr_frequency=10,        # Less frequent adaptation
    max_amr_cycles=2,        # Fewer cycles
    base_solver="fixed_point",  # Fastest base solver
    backend="jax"            # GPU acceleration
)
```

#### For Maximum Accuracy:
```python
accurate_amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-6,    # Very aggressive refinement
    max_levels=6,            # Many levels allowed
    amr_frequency=3,         # Frequent adaptation
    max_amr_cycles=5,        # Many cycles
    base_solver="particle_collocation",  # Most accurate base solver
    backend="auto"           # Best available backend
)
```

#### For Balanced Performance:
```python
balanced_amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-4,    # Moderate refinement
    max_levels=4,            # Reasonable levels
    amr_frequency=5,         # Default frequency
    max_amr_cycles=3,        # Default cycles
    base_solver="fixed_point",
    backend="auto"
)
```

### Performance Monitoring

```python
def monitor_amr_performance():
    """Monitor and analyze AMR performance characteristics."""
    
    problem = ExampleMFGProblem(Nx=32, Nt=50, T=1.0)
    
    # Create AMR solver with monitoring
    amr_solver = create_amr_solver(
        problem,
        error_threshold=1e-4,
        max_levels=4,
        base_solver="fixed_point"
    )
    
    # Solve with timing
    import time
    start_time = time.time()
    result = amr_solver.solve(max_iterations=100, tolerance=1e-6, verbose=True)
    total_time = time.time() - start_time
    
    # Analyze performance
    amr_stats = amr_solver.get_amr_statistics()
    
    print(f"\n=== Performance Analysis ===")
    print(f"Total solve time: {total_time:.2f}s")
    print(f"Iterations to convergence: {len(result.convergence_history)}")
    print(f"Average time per iteration: {total_time/len(result.convergence_history):.3f}s")
    
    # Mesh efficiency analysis
    mesh_stats = amr_stats['mesh_statistics']
    base_cells = problem.Nx * problem.Nx
    final_cells = mesh_stats['total_cells']
    
    print(f"\nMesh Efficiency:")
    print(f"  Base grid cells: {base_cells}")
    print(f"  Final AMR cells: {final_cells}")
    print(f"  Cell ratio: {final_cells/base_cells:.2f}")
    print(f"  Memory factor: {final_cells/base_cells:.2f}×")
    
    # Adaptation efficiency
    adapt_stats = amr_stats['adaptation_statistics']
    total_adaptations = adapt_stats['total_refinements'] + adapt_stats['total_coarsenings']
    
    print(f"\nAdaptation Efficiency:")
    print(f"  Total adaptations: {total_adaptations}")
    print(f"  Refinement/coarsening ratio: {adapt_stats['total_refinements']/(adapt_stats['total_coarsenings']+1):.2f}")
    print(f"  Adaptation cycles: {adapt_stats['adaptation_cycles']}")
    
    # Accuracy efficiency
    accuracy_per_cell = 1.0 / (result.final_error * final_cells)
    
    print(f"\nAccuracy Efficiency:")
    print(f"  Final error: {result.final_error:.2e}")
    print(f"  Accuracy per cell: {accuracy_per_cell:.2e}")
    
    return result, amr_stats
```

## Troubleshooting Common Issues

### Issue 1: AMR Not Refining

**Symptoms:**
- No refinements occur
- Final mesh same as initial mesh
- `total_refinements` = 0

**Possible Causes & Solutions:**

```python
# Problem: Error threshold too high
amr_solver = create_amr_solver(problem, error_threshold=1e-2)  # Too high!
# Solution: Lower the threshold
amr_solver = create_amr_solver(problem, error_threshold=1e-4)  # Better

# Problem: Max levels reached immediately
amr_solver = create_amr_solver(problem, max_levels=1)  # Too restrictive!
# Solution: Allow more levels
amr_solver = create_amr_solver(problem, max_levels=4)  # Better

# Problem: Solution too smooth
problem = ExampleMFGProblem(sigma=1.0)  # High diffusion = smooth
# Solution: Create sharper features
problem = ExampleMFGProblem(sigma=0.05)  # Low diffusion = sharp
```

### Issue 2: Excessive Refinement

**Symptoms:**
- Memory usage grows very large
- Many refinement levels
- Solver becomes very slow

**Solutions:**

```python
# Solution 1: Increase error threshold
amr_solver = create_amr_solver(problem, error_threshold=1e-3)  # Less aggressive

# Solution 2: Limit refinement levels
amr_solver = create_amr_solver(problem, max_levels=3)  # Restrict depth

# Solution 3: Reduce adaptation frequency
amr_solver = create_amr_solver(problem, amr_frequency=10)  # Less frequent

# Solution 4: Set minimum cell size
from mfg_pde.geometry.amr_mesh import AMRRefinementCriteria
criteria = AMRRefinementCriteria(min_cell_size=1e-5)  # Prevent tiny cells
```

### Issue 3: AMR Slower than Uniform Grid

**Symptoms:**
- AMR takes longer than equivalent uniform grid
- High overhead from adaptation

**Solutions:**

```python
# Solution 1: Reduce adaptation frequency
amr_solver = create_amr_solver(problem, amr_frequency=20)

# Solution 2: Use fewer AMR cycles
amr_solver = create_amr_solver(problem, max_amr_cycles=1)

# Solution 3: Use JAX backend for acceleration
amr_solver = create_amr_solver(problem, backend="jax")

# Solution 4: Coarser error threshold
amr_solver = create_amr_solver(problem, error_threshold=1e-3)
```

### Issue 4: Solution Quality Degradation

**Symptoms:**
- AMR solution less accurate than uniform grid
- Conservation errors
- Interpolation artifacts

**Solutions:**

```python
# Solution 1: Decrease error threshold
amr_solver = create_amr_solver(problem, error_threshold=1e-5)

# Solution 2: Use more accurate base solver
amr_solver = create_amr_solver(problem, base_solver="particle_collocation")

# Solution 3: More frequent adaptation
amr_solver = create_amr_solver(problem, amr_frequency=3)

# Solution 4: Check mass conservation
result = amr_solver.solve()
initial_mass = np.sum(problem.m0) if hasattr(problem, 'm0') else 1.0
final_mass = np.sum(result.M)
mass_error = abs(final_mass - initial_mass) / initial_mass
print(f"Mass conservation error: {mass_error:.2e}")
```

### Debugging AMR Behavior

```python
def debug_amr_behavior(problem, **amr_kwargs):
    """Debug AMR behavior step by step."""
    
    amr_solver = create_amr_solver(problem, **amr_kwargs)
    
    print("=== AMR Debugging ===")
    print(f"Configuration:")
    for key, value in amr_kwargs.items():
        print(f"  {key}: {value}")
    
    # Solve with detailed monitoring
    result = amr_solver.solve(max_iterations=20, verbose=True)
    
    # Get detailed statistics
    amr_stats = amr_solver.get_amr_statistics()
    
    print(f"\n=== Debug Results ===")
    
    # Check if adaptation occurred
    adapt_stats = amr_stats['adaptation_statistics']
    if adapt_stats['total_refinements'] == 0:
        print("⚠️  WARNING: No refinements occurred!")
        print("   - Check if error_threshold is too high")
        print("   - Check if max_levels is too low")
        print("   - Verify problem has sharp features")
    else:
        print(f"✓ Refinements occurred: {adapt_stats['total_refinements']}")
    
    # Check mesh efficiency
    mesh_stats = amr_stats['mesh_statistics']
    base_cells = problem.Nx * problem.Nx
    efficiency_ratio = mesh_stats['total_cells'] / base_cells
    
    if efficiency_ratio > 2.0:
        print(f"⚠️  WARNING: High cell ratio {efficiency_ratio:.1f}×")
        print("   - Consider increasing error_threshold")
        print("   - Consider reducing max_levels")
    elif efficiency_ratio < 0.5:
        print(f"⚠️  WARNING: Low cell ratio {efficiency_ratio:.1f}×")  
        print("   - Problem may be too smooth for AMR")
        print("   - Consider uniform grid instead")
    else:
        print(f"✓ Good cell ratio: {efficiency_ratio:.1f}×")
    
    # Check convergence
    if result.convergence_achieved:
        print(f"✓ Converged to {result.final_error:.2e}")
    else:
        print(f"⚠️  Did not converge (error: {result.final_error:.2e})")
        print("   - Increase max_iterations")
        print("   - Check solver stability")
    
    return result, amr_stats
```

## Real-World Applications

### Application 1: Traffic Flow with Congestion

```python
def traffic_flow_amr_example():
    """Traffic flow problem with localized congestion."""
    
    # Problem with sharp congestion near city center
    class TrafficFlowProblem(ExampleMFGProblem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.city_center = 0.0  # Congestion at x=0
            
        def congestion_cost(self, x, m):
            # Higher congestion near city center
            distance_factor = np.exp(-2 * (x - self.city_center)**2)
            return self.coefCT * m * (1 + 5 * distance_factor)
    
    problem = TrafficFlowProblem(
        Nx=48,
        Nt=60,
        xmin=-5.0, xmax=5.0,
        T=2.0,
        sigma=0.1,
        coefCT=1.0
    )
    
    print("=== Traffic Flow with AMR ===")
    
    # AMR should automatically refine near city center
    amr_solver = create_amr_solver(
        problem,
        error_threshold=1e-4,
        max_levels=4,
        base_solver="fixed_point",
        amr_frequency=5
    )
    
    result = amr_solver.solve(max_iterations=80, verbose=True)
    
    # Analyze refinement pattern
    amr_stats = amr_solver.get_amr_statistics()
    mesh_stats = amr_stats['mesh_statistics']
    
    print(f"\nTraffic Flow Results:")
    print(f"  Final error: {result.final_error:.2e}")
    print(f"  Total cells: {mesh_stats['total_cells']}")
    print(f"  Max refinement level: {mesh_stats['max_level']}")
    print(f"  Refinements: {amr_stats['adaptation_statistics']['total_refinements']}")
    
    # Expected: High refinement near x=0 (city center)
    
    return result
```

### Application 2: Financial Market Dynamics

```python
def financial_market_amr_example():
    """Financial market with sharp price movements."""
    
    class FinancialMarketProblem(ExampleMFGProblem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.volatility_shock = True
            
        def running_cost(self, x, u, m):
            # Higher costs during volatility shocks
            if self.volatility_shock and abs(x) > 1.5:
                return 2.0 * (u**2 + self.coefCT * m**2)
            else:
                return u**2 + self.coefCT * m**2
    
    problem = FinancialMarketProblem(
        Nx=40,
        Nt=80,
        xmin=-3.0, xmax=3.0,
        T=1.5,
        sigma=0.05,  # Low base volatility
        coefCT=0.8
    )
    
    print("=== Financial Market with AMR ===")
    
    # AMR should refine in high-volatility regions
    amr_solver = create_amr_solver(
        problem,
        error_threshold=1e-5,    # High accuracy for financial data
        max_levels=5,
        base_solver="particle_collocation",  # Better for discontinuities
        amr_frequency=4,
        backend="jax"            # Fast computation needed
    )
    
    result = amr_solver.solve(max_iterations=100, tolerance=1e-7, verbose=True)
    
    print(f"\nFinancial Market Results:")
    print(f"  Convergence: {'Yes' if result.convergence_achieved else 'No'}")
    print(f"  Final error: {result.final_error:.2e}")
    
    if 'amr_stats' in result.solver_info:
        amr_stats = result.solver_info['amr_stats']
        print(f"  Total refinements: {amr_stats['total_refinements']}")
        print(f"  Adaptation cycles: {amr_stats['adaptation_cycles']}")
    
    return result
```

### Application 3: Epidemic Spread Modeling

```python
def epidemic_spread_amr_example():
    """Epidemic spread with localized outbreaks."""
    
    class EpidemicSpreadProblem(ExampleMFGProblem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.outbreak_centers = [-1.5, 1.5]  # Multiple outbreak locations
            
        def initial_density(self, x):
            # Initial infections concentrated at outbreak centers
            density = np.zeros_like(x)
            for center in self.outbreak_centers:
                density += 0.3 * np.exp(-5 * (x - center)**2)
            return density + 0.1  # Background population
    
    problem = EpidemicSpreadProblem(
        Nx=50,
        Nt=70,
        xmin=-4.0, xmax=4.0,
        T=3.0,
        sigma=0.2,   # Moderate spread rate
        coefCT=1.5   # Strong interaction effects
    )
    
    print("=== Epidemic Spread with AMR ===")
    
    # AMR should track spreading infection fronts
    amr_solver = create_amr_solver(
        problem,
        error_threshold=1e-4,
        max_levels=4,
        base_solver="fixed_point",
        amr_frequency=6,         # Track evolving fronts
        max_amr_cycles=4         # Allow mesh evolution
    )
    
    result = amr_solver.solve(max_iterations=90, verbose=True)
    
    print(f"\nEpidemic Spread Results:")
    print(f"  Final error: {result.final_error:.2e}")
    
    if 'final_mesh_stats' in result.solver_info:
        mesh_stats = result.solver_info['final_mesh_stats']
        print(f"  Final mesh cells: {mesh_stats['total_cells']}")
        print(f"  Max refinement level: {mesh_stats['max_level']}")
        
        # Analyze level distribution (should show refinement at outbreak sites)
        level_dist = mesh_stats['level_distribution']
        print(f"  Level distribution: {level_dist}")
    
    return result
```

## Summary and Best Practices

### When to Use AMR

✅ **Use AMR for:**
- Problems with sharp gradients or boundary layers
- Localized phenomena (congestion, shocks, fronts)
- Multi-scale dynamics
- Resource-constrained high-accuracy simulations
- Research requiring adaptive resolution

❌ **Don't use AMR for:**
- Globally smooth solutions
- Small problems where uniform grids are sufficient
- When uniform high resolution is required everywhere
- Time-critical applications where setup overhead matters

### AMR Configuration Guidelines

| Problem Type | Error Threshold | Max Levels | AMR Frequency | Base Solver |
|--------------|----------------|------------|---------------|-------------|
| **Smooth** | 1e-3 | 2-3 | 10 | fixed_point |
| **Moderate Features** | 1e-4 | 3-4 | 5 | fixed_point |
| **Sharp Features** | 1e-5 | 4-5 | 3-5 | particle_collocation |
| **Very Sharp/Discontinuous** | 1e-6 | 5-6 | 3 | particle_collocation |

### Performance Optimization Tips

1. **Start Conservative**: Begin with higher error thresholds and fewer levels
2. **Monitor Memory**: Watch for excessive refinement
3. **Use JAX**: Enable JAX backend for GPU acceleration
4. **Batch Testing**: Test multiple configurations to find optimal parameters
5. **Profile Performance**: Monitor adaptation overhead vs accuracy gains

### Debugging Checklist

- [ ] Check if refinements are occurring (`total_refinements > 0`)
- [ ] Verify reasonable cell count (not too high/low)
- [ ] Ensure convergence is achieved
- [ ] Check mass conservation (for density problems)
- [ ] Monitor adaptation overhead vs solve time
- [ ] Validate solution quality vs uniform grid

---

This completes the comprehensive AMR tutorial. The adaptive mesh refinement system in MFG_PDE provides powerful capabilities for efficiently solving Mean Field Games with complex solution features, offering both automatic adaptation and fine-grained control for advanced users.