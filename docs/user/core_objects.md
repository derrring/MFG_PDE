# Core Objects Guide - Research API

**For MFG research with custom mathematical formulations**

This is the primary API tier for research users who need to define custom Hamiltonians, geometries, boundary conditions, or cost functionals. Since MFG users typically understand PDE systems and variational formulations, this tier provides the mathematical flexibility needed for research problems.

## Architecture Overview

The MFG_PDE core objects follow a clean pipeline:

```python
MFGProblem → FixedPointSolver → MFGResult
```

1. **MFGProblem**: Defines the mathematical problem
2. **FixedPointSolver**: Configurable solver with hooks
3. **MFGResult**: Rich result object with analysis methods

## Creating Problems

### Using Problem Builders

```python
from mfg_pde import create_mfg_problem

# Create problems with explicit control
problem = create_mfg_problem("crowd_dynamics",
                           domain=(0, 5),        # Custom domain
                           time_horizon=2.0)     # Custom time horizon

# Portfolio optimization with custom parameters
problem = create_mfg_problem("portfolio_optimization",
                           domain=(0, 10),
                           time_horizon=1.0,
                           risk_aversion=0.3,
                           drift=0.05)
```

### Custom Problem Definition

```python
from mfg_pde.core import MFGProblem
import numpy as np

class CustomCrowdProblem(MFGProblem):
    def __init__(self, domain_bounds, time_horizon):
        self.domain_bounds = domain_bounds
        self.time_horizon = time_horizon

    def get_domain_bounds(self):
        return self.domain_bounds

    def get_time_horizon(self):
        return self.time_horizon

    def evaluate_hamiltonian(self, x, p, m, t):
        # Custom Hamiltonian: H(x,p,m,t)
        kinetic = 0.5 * p**2
        congestion = 0.2 * m * np.log(1 + m)  # Nonlinear congestion
        exit_cost = (x - self.domain_bounds[1])**2
        return kinetic + congestion + exit_cost

    def get_initial_density(self):
        # Custom initial distribution
        x = np.linspace(*self.domain_bounds, 101)
        density = np.exp(-10 * (x - 0.2)**2)  # Gaussian at entrance
        return density / np.trapz(density, x)

    def get_terminal_value(self):
        # Custom terminal cost
        x = np.linspace(*self.domain_bounds, 101)
        return 0.5 * (x - self.domain_bounds[1])**2

# Use your custom problem
problem = CustomCrowdProblem((0, 5), 2.0)
```

## Configuring Solvers

### Basic Solver Setup

```python
from mfg_pde.solvers import FixedPointSolver

# Create solver with custom settings
solver = FixedPointSolver(
    max_iterations=500,        # Maximum iterations
    tolerance=1e-6,           # Convergence tolerance
    damping=0.8,              # Damping parameter
    backend="auto"            # Choose backend (auto/numpy/torch/jax)
)
```

### Using Configuration Presets

```python
from mfg_pde.config import (
    fast_config, accurate_config, research_config,
    crowd_dynamics_config, financial_config
)

# Use problem-specific presets
config = crowd_dynamics_config()
solver = FixedPointSolver.from_config(config)

# Or accuracy-focused presets
config = accurate_config()
solver = FixedPointSolver.from_config(config)

# Chain configuration modifications
config = (fast_config()
         .with_tolerance(1e-5)
         .with_max_iterations(300)
         .with_damping(0.9))
solver = FixedPointSolver.from_config(config)
```

### Advanced Configuration

```python
from mfg_pde.config import SolverConfig

# Create completely custom configuration
config = SolverConfig(
    max_iterations=1000,
    tolerance=1e-8,
    damping=0.7,
    adaptive_damping=True,
    backend_preference=["torch", "jax", "numpy"],
    memory_limit="4GB",
    parallel_workers=4,
    checkpoint_frequency=50,
    early_stopping=True
)

solver = FixedPointSolver.from_config(config)
```

## Solving with Control

### Basic Solving

```python
# Solve with default settings
result = solver.solve(problem)

# Solve with progress monitoring
result = solver.solve(problem, verbose=True)

# Solve with custom stopping criteria
result = solver.solve(problem,
                     max_time=300,           # 5 minute timeout
                     target_accuracy=1e-7)   # Override tolerance
```

### Method Chaining

```python
from mfg_pde.solvers import FixedPointSolver

# Fluent interface for clean code
result = (FixedPointSolver()
         .with_tolerance(1e-6)
         .with_max_iterations(400)
         .with_damping(0.8)
         .solve(problem))
```

### Solver State Access

```python
# Access intermediate states during solving
solver = FixedPointSolver(max_iterations=100)

for state in solver.solve_iteratively(problem):
    print(f"Iteration {state.iteration}: residual = {state.residual:.2e}")

    # Custom stopping criteria
    if state.residual < 1e-8:
        break

    # Save intermediate results
    if state.iteration % 20 == 0:
        state.save_checkpoint(f"checkpoint_{state.iteration}.pkl")

final_result = state.to_result()
```

## Working with Results

### Rich Result Analysis

```python
result = solver.solve(problem)

# Solution data
u = result.value_function       # u(t,x)
m = result.density             # m(t,x)
t_grid = result.time_grid      # Time points
x_grid = result.spatial_grid   # Spatial points

# Convergence information
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final residual: {result.final_residual}")
print(f"Solve time: {result.solve_time:.2f}s")

# Physical quantities
print(f"Total mass: {result.total_mass}")
print(f"Mass conservation error: {result.mass_conservation_error}")
print(f"Energy: {result.total_energy}")
```

### Advanced Analysis

```python
# Compute derived quantities
velocity_field = result.compute_velocity_field()
optimal_trajectory = result.compute_optimal_trajectory(x_start=0.0)
hamiltonian_values = result.evaluate_hamiltonian()

# Sensitivity analysis
sensitivity = result.compute_parameter_sensitivity(['crowd_size', 'exit_attraction'])

# Export data
result.export_to_csv("results.csv")
result.export_to_hdf5("results.h5")
result.export_to_matlab("results.mat")
```

### Comparison and Benchmarking

```python
from mfg_pde.analysis import compare_results, benchmark_solvers

# Compare multiple solutions
configs = [fast_config(), accurate_config(), research_config()]
results = [FixedPointSolver.from_config(c).solve(problem) for c in configs]

comparison = compare_results(results, metrics=['accuracy', 'speed', 'memory'])
comparison.plot_comparison()

# Benchmark different solvers
benchmark = benchmark_solvers(
    problem=problem,
    solvers=['fixed_point', 'newton', 'anderson'],
    configs=[fast_config(), accurate_config()]
)
benchmark.print_summary()
```

## Problem Modification

### Runtime Parameter Changes

```python
# Create base problem
problem = create_mfg_problem("crowd_dynamics", domain=(0, 5), crowd_size=100)

# Solve with different parameters
for crowd_size in [50, 100, 200, 500]:
    modified_problem = problem.with_parameters(crowd_size=crowd_size)
    result = solver.solve(modified_problem)
    print(f"Crowd size {crowd_size}: {result.evacuation_time:.2f}s")
```

### Domain and Time Modifications

```python
# Change domain size
problem_large = problem.with_domain((0, 10))

# Change time horizon
problem_long = problem.with_time_horizon(5.0)

# Change both
problem_modified = (problem
                   .with_domain((0, 8))
                   .with_time_horizon(3.0)
                   .with_parameters(crowd_size=300))
```

## Error Handling and Debugging

### Robust Solving

```python
from mfg_pde.exceptions import ConvergenceError, ConfigurationError

try:
    result = solver.solve(problem)
except ConvergenceError as e:
    print(f"Failed to converge: {e}")
    print(f"Final residual: {e.final_residual}")
    print(f"Iterations completed: {e.iterations}")

    # Try with more relaxed settings
    relaxed_solver = solver.with_tolerance(1e-4).with_max_iterations(1000)
    result = relaxed_solver.solve(problem)

except ConfigurationError as e:
    print(f"Configuration problem: {e}")
    # Fix configuration and retry
```

### Diagnostic Tools

```python
from mfg_pde.diagnostics import diagnose_problem, analyze_convergence

# Analyze problem characteristics
diagnosis = diagnose_problem(problem)
print(f"Problem difficulty: {diagnosis.difficulty}")
print(f"Recommended tolerance: {diagnosis.recommended_tolerance}")
print(f"Estimated iterations: {diagnosis.estimated_iterations}")

# Analyze convergence behavior
convergence_analysis = analyze_convergence(result)
convergence_analysis.plot_convergence_history()
convergence_analysis.identify_convergence_issues()
```

## Integration with Simple API

You can mix the simple and core object APIs:

```python
from mfg_pde import solve_mfg
from mfg_pde.solvers import FixedPointSolver

# Start with simple API
result_simple = solve_mfg("crowd_dynamics", crowd_size=200)

# Extract the problem for further analysis
problem = result_simple.problem

# Use core objects for detailed control
custom_solver = (FixedPointSolver()
                .with_tolerance(1e-8)
                .with_adaptive_damping(True))

result_detailed = custom_solver.solve(problem)

# Compare results
print(f"Simple API: {result_simple.iterations} iterations")
print(f"Custom solver: {result_detailed.iterations} iterations")
```

## Performance Optimization

### Backend Selection

```python
# Try different computational backends
backends = ['numpy', 'torch', 'jax', 'numba']
results = {}

for backend in backends:
    solver = FixedPointSolver(backend=backend)
    start_time = time.time()
    result = solver.solve(problem)
    solve_time = time.time() - start_time

    results[backend] = {
        'time': solve_time,
        'iterations': result.iterations,
        'memory': result.peak_memory_usage
    }

# Find fastest backend
fastest = min(results.items(), key=lambda x: x[1]['time'])
print(f"Fastest backend: {fastest[0]} ({fastest[1]['time']:.2f}s)")
```

### Parallel Processing

```python
from mfg_pde.parallel import ParameterSweep

# Parallel parameter exploration
sweep = ParameterSweep(
    base_problem=problem,
    parameter_ranges={
        'crowd_size': [50, 100, 200, 500],
        'exit_attraction': [0.5, 1.0, 1.5, 2.0]
    },
    solver_config=fast_config(),
    n_workers=4
)

results = sweep.execute()
sweep.plot_parameter_space()
```

## What's Next?

- **Need algorithm-level control?** → See [Advanced Hooks Guide](advanced_hooks.md)
- **Want to implement custom solvers?** → Check [Developer Guide](../development/custom_solvers.md)
- **Performance issues?** → Read [Performance Optimization Guide](performance.md)
- **Complex geometries?** → Explore [Geometry and Domains Guide](geometry.md)
