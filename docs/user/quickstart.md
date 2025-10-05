# Factory API Quickstart

**Get started with MFG_PDE in 5 minutes**

This guide shows you the **factory API** - the primary interface for MFG_PDE users.

---

## Prerequisites

MFG_PDE assumes you understand:
- **Mean Field Games**: HJB-FP coupled systems, Nash equilibria
- **Numerical PDEs**: Finite difference methods, stability
- **Python**: NumPy and basic scientific computing

If you need background, see the [Theory Guide](../theory/) and [Notebooks](../examples/notebooks/).

---

## Installation

```bash
pip install mfg-pde
```

---

## Your First MFG Solution (2 minutes)

### Step 1: Import and Create Problem

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver

# Create a standard MFG problem
problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)
```

### Step 2: Create Solver

```python
# Use default solver (Tier 2: Hybrid, mass-conserving)
solver = create_fast_solver(problem, "fixed_point")
```

### Step 3: Solve

```python
# Solve the HJB-FP system
result = solver.solve()
```

### Step 4: Access Results

```python
# Value function and density
print(result.U)  # u(t,x) - shape (Nt+1, Nx+1)
print(result.M)  # m(t,x) - shape (Nt+1, Nx+1)

# Check convergence
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Mass error: {result.mass_conservation_error:.2e}")
```

**That's it!** You've solved a Mean Field Games problem with research-grade quality.

---

## Three Solver Tiers

MFG_PDE provides three solver tiers based on **quality**:

| Tier | Name | Mass Error | Convergence | Use Case |
|:-----|:-----|:-----------|:------------|:---------|
| **Tier 1** | Basic FDM | ~1-10% | 50-100 iter | Benchmark only |
| **Tier 2** | Hybrid | ~10⁻¹⁵ | 10-20 iter | **DEFAULT** (production) |
| **Tier 3** | Advanced | Varies | Varies | Specialized (WENO, Semi-Lagrangian) |

### Tier 1: Basic FDM (Benchmark)

```python
from mfg_pde.factory import create_basic_solver

# Basic FDM - for benchmarking only (poor mass conservation)
solver = create_basic_solver(problem, damping=0.6)
result = solver.solve()
```

⚠️ **Not recommended for production** - use for comparison only.

### Tier 2: Hybrid (DEFAULT)

```python
from mfg_pde.factory import create_fast_solver

# Hybrid (HJB-FDM + FP-Particle) - RECOMMENDED
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()
```

✅ **Recommended** - excellent mass conservation, fast convergence, robust.

### Tier 3: Advanced Methods

```python
from mfg_pde.factory import create_accurate_solver

# High-order WENO
solver_weno = create_accurate_solver(problem, solver_type="weno")

# Semi-Lagrangian
solver_sl = create_accurate_solver(problem, solver_type="semi_lagrangian")

# Deep Galerkin Method (DGM)
solver_dgm = create_accurate_solver(problem, solver_type="dgm")
```

Use for specialized requirements (high-order accuracy, large time steps, high dimensions).

---

## Common Workflows

### Workflow 1: Standard Research Problem

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver

# Define problem
problem = ExampleMFGProblem(Nx=100, Nt=50, T=1.0)

# Solve with default
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()

# Verify mass conservation
import numpy as np
for t in range(problem.Nt + 1):
    mass = np.sum(result.M[t, :]) * problem.Dx
    print(f"t={t}: mass={mass:.15f}")  # Should be 1.000000000000000
```

### Workflow 2: Method Comparison

```python
from mfg_pde.factory import create_basic_solver, create_fast_solver, create_accurate_solver

# Compare three solver tiers
solvers = {
    "Basic FDM": create_basic_solver(problem),
    "Hybrid (Standard)": create_fast_solver(problem, "fixed_point"),
    "WENO (Advanced)": create_accurate_solver(problem, solver_type="weno")
}

results = {}
for name, solver in solvers.items():
    results[name] = solver.solve()
    print(f"{name}:")
    print(f"  Iterations: {results[name].iterations}")
    print(f"  Mass error: {results[name].mass_conservation_error:.2e}")
```

### Workflow 3: Custom Configuration

```python
from mfg_pde.factory import create_fast_solver

# Fine-tune solver parameters
solver = create_fast_solver(
    problem,
    "fixed_point",
    max_iterations=200,     # Increase max iterations
    tolerance=1e-8,         # Tighter convergence
    damping=0.7             # Adjust damping factor
)

result = solver.solve()
```

---

## Next Steps

### **Learn More**

1. **[Solver Selection Guide](SOLVER_SELECTION_GUIDE.md)** - When to use each tier
2. **[Factory API Reference](factory_api_reference.md)** - All `create_*_solver()` functions
3. **[Custom Problems](custom_problems.md)** - Define your own MFG formulations

### **Examples**

Browse working examples:
- **[Basic Examples](../examples/basic/)** - Single-concept demonstrations
- **[Advanced Examples](../examples/advanced/)** - Research-grade problems
- **[Notebooks](../examples/notebooks/)** - Interactive tutorials

### **Custom MFG Problems**

Define your own Hamiltonian:

```python
from mfg_pde.core import BaseMFGProblem
import numpy as np

class MyMFGProblem(BaseMFGProblem):
    def __init__(self, Nx, Nt, T):
        super().__init__(Nx, Nt, T)

    def evaluate_hamiltonian(self, x, p, m, t):
        """
        Define H(x, p, m, t).

        Args:
            x: spatial position
            p: momentum (gradient of value function)
            m: density
            t: time

        Returns:
            H(x, p, m, t): scalar
        """
        # Example: kinetic energy + congestion cost
        kinetic = 0.5 * p**2
        congestion = 0.2 * m * np.log(1 + m)
        return kinetic + congestion

# Use with factory API
problem = MyMFGProblem(Nx=50, Nt=20, T=1.0)
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()
```

See [Custom Problems Guide](custom_problems.md) for details.

---

## Common Questions

### **Q: Which solver tier should I use?**

**A**: Use **Tier 2 (Hybrid)** by default - it's the best balance of accuracy and speed.

```python
solver = create_fast_solver(problem, "fixed_point")  # Start here
```

Only use Tier 1 for benchmarking or Tier 3 for specialized needs.

### **Q: How do I know if my solution converged?**

**A**: Check the result object:

```python
result = solver.solve()

if result.converged:
    print(f"✅ Converged in {result.iterations} iterations")
else:
    print(f"⚠️ Did not converge (max iterations reached)")
    print(f"Final residual: {result.final_residual:.2e}")
```

### **Q: What if convergence is too slow?**

**A**: Try adjusting solver parameters:

```python
# Option 1: Increase max iterations
solver = create_fast_solver(problem, "fixed_point", max_iterations=200)

# Option 2: Relax tolerance
solver = create_fast_solver(problem, "fixed_point", tolerance=1e-5)

# Option 3: Adjust damping (for Tier 1 only)
solver = create_basic_solver(problem, damping=0.7)
```

### **Q: How do I visualize results?**

**A**: Use matplotlib or built-in plotting:

```python
import matplotlib.pyplot as plt

# Plot final density
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(result.U, aspect='auto', cmap='viridis')
plt.title('Value Function u(t,x)')
plt.colorbar()

plt.subplot(122)
plt.imshow(result.M, aspect='auto', cmap='hot')
plt.title('Density m(t,x)')
plt.colorbar()

plt.tight_layout()
plt.show()
```

---

## Summary

**Key Takeaways**:

1. **Factory API is primary** - Use `create_*_solver()` functions
2. **Tier 2 is default** - `create_fast_solver()` for most uses
3. **Three lines to solve**:
   ```python
   problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)
   solver = create_fast_solver(problem, "fixed_point")
   result = solver.solve()
   ```
4. **Check convergence** - Always verify `result.converged`
5. **Compare methods** - Use different tiers for benchmarking

**Next**: [Solver Selection Guide](SOLVER_SELECTION_GUIDE.md) for choosing the right tier.

---

**Need help?**
- **[GitHub Discussions](https://github.com/derrring/MFG_PDE/discussions)** - Ask questions
- **[GitHub Issues](https://github.com/derrring/MFG_PDE/issues)** - Report bugs
