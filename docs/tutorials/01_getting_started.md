# Getting Started with MFG_PDE

**Tutorial Level**: Beginner
**Estimated Time**: 30 minutes
**Prerequisites**: Basic Python knowledge, familiarity with PDEs (helpful but not required)

---

## What You'll Learn

In this tutorial, you'll learn:
1. How to install MFG_PDE
2. How to solve your first Mean Field Game
3. How to visualize the results
4. The three core concepts: Problems, Configs, and Results

By the end, you'll have solved a complete MFG problem in just a few lines of code.

---

## Installation

### Via pip (Recommended)

```bash
pip install mfg-pde
```

### From Source

```bash
git clone https://github.com/anthropics/mfg-pde.git
cd mfg-pde
pip install -e .
```

### Verify Installation

```python
import mfg_pde
print(mfg_pde.__version__)  # Should print: 0.9.0 (or later)
```

---

## Your First MFG Problem

Let's solve a simple Mean Field Game in just 3 lines of code:

```python
from mfg_pde import solve_mfg, ExampleMFGProblem

problem = ExampleMFGProblem()
result = solve_mfg(problem, preset="fast")
```

That's it! You've just solved an MFG problem. Let's break down what happened:

1. **Import**: We imported `solve_mfg` (the solver) and `ExampleMFGProblem` (a pre-configured problem)
2. **Problem**: We created a default MFG problem instance
3. **Solve**: We solved it using the "fast" preset configuration

---

## Understanding the Results

The `result` object contains everything you need:

```python
# Check if the solver converged
print(f"Converged: {result.converged}")
# Output: Converged: True

# Number of iterations
print(f"Iterations: {result.iterations}")
# Output: Iterations: 15

# The solutions
U = result.U  # Hamilton-Jacobi-Bellman solution (value function)
M = result.M  # Fokker-Planck solution (agent distribution)

print(f"U shape: {U.shape}")  # (51, 51) - time × space grid
print(f"M shape: {M.shape}")  # (51, 51) - time × space grid

# Convergence history
print(f"Final U error: {result.error_history_U[-1]:.2e}")
# Output: Final U error: 9.87e-07

# Execution time
print(f"Solve time: {result.execution_time:.3f}s")
# Output: Solve time: 1.234s
```

---

## Visualizing the Results

MFG_PDE includes built-in visualization tools:

```python
from mfg_pde.visualization import plot_results

# Create interactive plots
plot_results(result, problem)
```

This creates 4 plots:
1. **Value Function $U(t,x)$**: How valuable it is to be at position $x$ at time $t$
2. **Agent Distribution $m(t,x)$**: Where agents are located at each time
3. **Convergence History**: How errors decreased over iterations
4. **Final State**: Distribution of agents at final time

---

## The Three Core Concepts

### 1. Problems: What to Solve

A **Problem** defines the Mean Field Game mathematically:

```python
from mfg_pde import ExampleMFGProblem

problem = ExampleMFGProblem(
    nx=100,        # Spatial grid points
    nt=100,        # Time grid points
    T=1.0,         # Time horizon
    # The rest uses sensible defaults
)
```

**What's in a problem?**
- **Domain**: Spatial region and time horizon
- **Hamiltonian**: Running cost function $H(x, p, m)$
- **Coupling**: How agents interact through mean field $m$
- **Initial Distribution**: Where agents start ($m_0$)
- **Terminal Cost**: Cost at final time ($g(x)$)

The `ExampleMFGProblem` provides reasonable defaults for all of these, so you can start quickly.

---

### 2. Configs: How to Solve

A **Config** specifies the solver algorithms and parameters:

```python
from mfg_pde import solve_mfg

# Option 1: Use a preset (simplest)
result = solve_mfg(problem, preset="fast")      # Speed over accuracy
result = solve_mfg(problem, preset="balanced")  # Good balance (default)
result = solve_mfg(problem, preset="accurate")  # High accuracy

# Option 2: Use a config object (more control)
from mfg_pde.config import presets
config = presets.accurate_solver()
result = solve_mfg(problem, config=config)

# Option 3: Build custom config (maximum control)
from mfg_pde.config import ConfigBuilder
config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=3)
    .solver_fp(method="fdm")
    .picard(max_iterations=100, tolerance=1e-8)
    .build()
)
result = solve_mfg(problem, config=config)
```

**When to use each?**
- **Presets**: Quick prototyping, testing, most use cases
- **Config objects**: When you need to reuse configurations
- **Builder**: When you need full control over solver parameters

---

### 3. Results: What You Get

A **Result** contains the solutions and solver information:

```python
result = solve_mfg(problem, preset="fast")

# Solutions (numpy arrays)
U = result.U  # HJB solution: (nt+1, nx+1) array
M = result.M  # FP solution: (nt+1, nx+1) array

# Convergence info
result.converged           # bool: Did it converge?
result.iterations          # int: How many iterations?
result.error_history_U     # list[float]: U error per iteration
result.error_history_M     # list[float]: M error per iteration

# Timing
result.execution_time      # float: Total solve time (seconds)

# Metadata
result.config              # The configuration used
result.problem_name        # Problem identifier
```

---

## Complete Example: Linear-Quadratic MFG

Here's a complete example solving a Linear-Quadratic Mean Field Game:

```python
"""
Complete LQ-MFG Example
-----------------------
Solves a Linear-Quadratic Mean Field Game where agents:
- Move in 1D space [-2, 2]
- Have quadratic running cost
- Start with Gaussian distribution
- Interact through mean-field coupling
"""

from mfg_pde import solve_mfg, ExampleMFGProblem
from mfg_pde.visualization import plot_results
import matplotlib.pyplot as plt

# Step 1: Define the problem
problem = ExampleMFGProblem(
    nx=100,   # 100 spatial points
    nt=100,   # 100 time steps
    T=1.0,    # Time horizon: [0, 1]
)

print("Problem setup:")
print(f"  Spatial domain: [-1, 1] with {problem.domain.nx} points")
print(f"  Time horizon: [0, {problem.time_horizon}] with {problem.domain.nt} steps")

# Step 2: Solve with accurate preset
print("\nSolving MFG problem...")
result = solve_mfg(problem, preset="accurate")

# Step 3: Check results
print("\nResults:")
print(f"  Converged: {result.converged}")
print(f"  Iterations: {result.iterations}")
print(f"  Final U error: {result.error_history_U[-1]:.2e}")
print(f"  Final M error: {result.error_history_M[-1]:.2e}")
print(f"  Solve time: {result.execution_time:.3f}s")

# Step 4: Visualize
print("\nGenerating plots...")
plot_results(result, problem)
plt.show()

print("\n✓ Complete! Check the plots to see:")
print("  1. Value function U(t,x) - optimal cost-to-go")
print("  2. Agent distribution m(t,x) - where agents are located")
print("  3. Convergence - how quickly the solver converged")
```

**Expected output**:
```
Problem setup:
  Spatial domain: [-1, 1] with 100 points
  Time horizon: [0, 1.0] with 100 steps

Solving MFG problem...

Results:
  Converged: True
  Iterations: 23
  Final U error: 8.45e-09
  Final M error: 7.23e-09
  Solve time: 2.145s

Generating plots...

✓ Complete! Check the plots to see:
  1. Value function U(t,x) - optimal cost-to-go
  2. Agent distribution m(t,x) - where agents are located
  3. Convergence - how quickly the solver converged
```

---

## Common Adjustments

### Faster Solves

```python
# Use fewer grid points
problem = ExampleMFGProblem(nx=50, nt=50)

# Use fast preset
result = solve_mfg(problem, preset="fast")
```

### More Accurate Solutions

```python
# Use more grid points
problem = ExampleMFGProblem(nx=200, nt=200)

# Use accurate preset
result = solve_mfg(problem, preset="accurate")
```

### 2D Problems

```python
from mfg_pde.geometry import Domain2D

# Create 2D domain
domain = Domain2D(xmin=-1, xmax=1, ymin=-1, ymax=1, nx=50, ny=50, nt=50)

problem = ExampleMFGProblem(domain=domain, T=1.0)
result = solve_mfg(problem, preset="fast")
```

---

## Troubleshooting

### Issue: "Solver did not converge"

**Symptom**: `result.converged == False`

**Solutions**:
1. Increase max iterations:
   ```python
   from mfg_pde.config import ConfigBuilder
   config = ConfigBuilder().picard(max_iterations=200).build()
   result = solve_mfg(problem, config=config)
   ```

2. Decrease tolerance:
   ```python
   config = ConfigBuilder().picard(tolerance=1e-4).build()
   result = solve_mfg(problem, config=config)
   ```

3. Use smaller grid (fewer points, easier to converge):
   ```python
   problem = ExampleMFGProblem(nx=50, nt=50)
   ```

---

### Issue: "Solve is too slow"

**Symptom**: Takes minutes instead of seconds

**Solutions**:
1. Use fast preset:
   ```python
   result = solve_mfg(problem, preset="fast")
   ```

2. Reduce grid size:
   ```python
   problem = ExampleMFGProblem(nx=50, nt=50)  # Smaller grid
   ```

3. Use GPU backend (if available):
   ```python
   from mfg_pde.config import ConfigBuilder
   config = ConfigBuilder().backend(backend_type="pytorch", device="cuda").build()
   result = solve_mfg(problem, config=config)
   ```

---

### Issue: "Results look wrong"

**Symptom**: Solution doesn't match expectations

**Check**:
1. Did solver converge?
   ```python
   if not result.converged:
       print("Warning: Solver did not converge!")
       print(f"Final error: {result.error_history_U[-1]:.2e}")
   ```

2. Check problem setup:
   ```python
   print(f"Domain: {problem.domain}")
   print(f"Time horizon: {problem.time_horizon}")
   ```

3. Visualize convergence:
   ```python
   import matplotlib.pyplot as plt
   plt.semilogy(result.error_history_U, label='U error')
   plt.semilogy(result.error_history_M, label='M error')
   plt.legend()
   plt.show()
   ```

---

## Next Steps

Now that you've solved your first MFG problem, here's what to explore next:

### Beginner

- **Tutorial 2: Configuration Patterns** - Learn the three ways to configure solvers
- **Examples**: `examples/basic/` - Simple, well-commented examples
- **Visualization Guide**: `docs/user_guides/visualization.md`

### Intermediate

- **Tutorial 3: Custom Problems** - Define your own Hamiltonians and couplings
- **Tutorial 4: Advanced Topics** - Network MFG, stochastic formulations, 2D problems
- **Migration Guide**: `docs/migration/PHASE_3_MIGRATION_GUIDE.md` (if upgrading from v0.8.x)

### Advanced

- **API Reference**: `docs/api/` - Complete API documentation
- **Research Guide**: `docs/development/AI_INTERACTION_DESIGN.md` - Research-grade usage
- **Contributing**: `CONTRIBUTING.md` - Contribute to MFG_PDE

---

## Summary

**You've learned**:
- ✅ How to install MFG_PDE
- ✅ How to solve an MFG problem in 3 lines: `problem = ExampleMFGProblem()` → `result = solve_mfg(problem, preset="fast")`
- ✅ How to access and visualize results
- ✅ The three core concepts: Problems (what), Configs (how), Results (output)
- ✅ Common troubleshooting techniques

**Key takeaway**: MFG_PDE makes solving Mean Field Games **simple**. The unified `solve_mfg()` interface handles all the complexity for you.

---

## Quick Reference Card

```python
# Minimal example
from mfg_pde import solve_mfg, ExampleMFGProblem
result = solve_mfg(ExampleMFGProblem(), preset="fast")

# Custom grid size
problem = ExampleMFGProblem(nx=100, nt=100, T=1.0)
result = solve_mfg(problem, preset="accurate")

# Check results
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
U, M = result.U, result.M  # Solutions

# Visualize
from mfg_pde.visualization import plot_results
plot_results(result, problem)

# Custom config
from mfg_pde.config import ConfigBuilder
config = ConfigBuilder().picard(max_iterations=100).build()
result = solve_mfg(problem, config=config)
```

---

**Tutorial Version**: 1.0
**Last Updated**: 2025-11-03
**MFG_PDE Version**: v0.9.0+
**Next Tutorial**: [02_configuration_patterns.md](02_configuration_patterns.md)
