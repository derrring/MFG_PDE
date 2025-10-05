# MFG Solver Selection Guide

**For Users**: Choosing the right solver for your application
**Last Updated**: 2025-10-05

## Quick Start

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver

# For most applications - use the default
problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()
```

**That's it!** The default solver (Tier 2: Hybrid) works well for 90% of use cases.

## Solver Hierarchy

### Three Tiers for Different Needs

```
📊 Tier 1: Basic (Benchmark)    → Simple, fast, approximate
✅ Tier 2: Standard (DEFAULT)    → Reliable, accurate, recommended
🔬 Tier 3: Advanced (Research)   → Specialized, high-performance
```

## Tier 1: Basic FDM (Benchmark)

### When to Use
- Quick testing and prototyping
- Benchmarking advanced methods
- Educational purposes
- Simple comparison baseline

### Characteristics
- **Method**: HJB-FDM + FP-FDM (Upwind + Damped Fixed Point)
- **Quality**: Poor (1-10% mass error)
- **Speed**: Fast
- **Stability**: Requires damping tuning

### Usage
```python
from mfg_pde.factory import create_basic_solver

# Tier 1: Basic benchmark
solver = create_basic_solver(
    problem,
    damping=0.6,         # Damping factor (0.5-0.7 recommended)
    max_iterations=100,
    tolerance=1e-5
)
result = solver.solve()
```

### Limitations
⚠️ **Not recommended for production**:
- Poor mass conservation (~1-10% error)
- May not converge (requires 50-100 iterations)
- Sensitive to damping parameter
- Use only for benchmarking

## Tier 2: Standard Hybrid (DEFAULT) ✅

### When to Use
- **Production applications** (recommended)
- Research with reliable baseline
- Publications and reports
- Any case where accuracy matters

### Characteristics
- **Method**: HJB-FDM + FP-Particle (Hybrid)
- **Quality**: Good (10⁻¹⁵ mass error)
- **Speed**: Moderate
- **Stability**: Robust, no tuning needed

### Usage
```python
from mfg_pde.factory import create_fast_solver

# Tier 2: Standard (DEFAULT) - just use this!
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()
```

### Advantages
✅ Perfect mass conservation (10⁻¹⁵ error)
✅ Fast convergence (10-20 iterations)
✅ No parameter tuning
✅ Reliable and robust
✅ **Recommended default**

## Tier 3: Advanced Solvers (Research)

### When to Use
- Specialized problem requirements
- High-dimensional problems (d > 3)
- Extreme accuracy needs
- Performance optimization (GPU)

### Available Methods

#### Semi-Lagrangian
```python
from mfg_pde.factory import create_semi_lagrangian_solver

solver = create_semi_lagrangian_solver(
    problem,
    interpolation_method="cubic",
    use_jax=True  # GPU acceleration
)
```
**Best for**: Smooth solutions, large time steps

#### WENO (High-Order)
```python
from mfg_pde.factory import create_accurate_solver

solver = create_accurate_solver(
    problem,
    solver_type="weno",
    weno_order=5
)
```
**Best for**: Shock capturing, discontinuous solutions

#### Deep Galerkin (Neural Network)
```python
solver = create_accurate_solver(
    problem,
    solver_type="dgm",
    hidden_layers=[64, 64, 64],
    use_gpu=True
)
```
**Best for**: High-dimensional problems (d ≥ 4)

## Decision Tree

```
Start here
    ↓
Do you need high accuracy for production? → YES → Use Tier 2 (create_fast_solver) ✅
    ↓ NO

Are you benchmarking a new method? → YES → Compare against Tier 1 & Tier 2
    ↓ NO

Do you have specialized requirements? → YES → Use Tier 3 (create_accurate_solver)
    ↓ NO

When in doubt → Use Tier 2 (create_fast_solver) ✅
```

## Comparison Table

| Feature | Tier 1: Basic | Tier 2: Standard ✅ | Tier 3: Advanced |
|---------|---------------|---------------------|------------------|
| **Mass conservation** | ~1-10% | ~10⁻¹⁵ | Varies |
| **Convergence** | 50-100 iter | 10-20 iter | Varies |
| **Speed** | Fast | Moderate | Varies |
| **Stability** | Needs tuning | Robust | Depends |
| **Use case** | Benchmark | **Production** | Research |
| **Recommended** | ❌ No | ✅ **Yes** | ⚙️ Specialized |

## Example: Complete Workflow

### Step 1: Start with Standard (Tier 2)
```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver

problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)

# Use the default - works great!
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()

# Check mass conservation
import numpy as np
for t in range(problem.Nt + 1):
    mass = np.sum(result.M[t, :]) * problem.Dx
    print(f"t={t}: mass={mass:.10f}")  # Should be 1.0000000000 ✓
```

### Step 2: Benchmark (if needed)
```python
from mfg_pde.factory import create_basic_solver

# Compare with basic FDM
solver_basic = create_basic_solver(problem)
result_basic = solver_basic.solve()

# See the difference in mass conservation
mass_basic = np.sum(result_basic.M[-1, :]) * problem.Dx
mass_standard = np.sum(result.M[-1, :]) * problem.Dx

print(f"Basic FDM:  {mass_basic:.4f} (error: {abs(mass_basic-1.0):.2%})")
print(f"Standard:   {mass_standard:.10f} (error: {abs(mass_standard-1.0):.10f})")
```

### Step 3: Advanced (if specialized)
```python
from mfg_pde.factory import create_accurate_solver

# For high-dimensional or specialized needs
solver_advanced = create_accurate_solver(
    problem,
    solver_type="semi_lagrangian"  # or "weno", "dgm"
)
result_advanced = solver_advanced.solve()
```

## Configuration Presets

Each tier has configuration presets:

```python
# Fast preset (fewer iterations, looser tolerance)
solver = create_fast_solver(problem, "fixed_point")

# Accurate preset (more iterations, tighter tolerance)
from mfg_pde.factory import create_accurate_solver
solver = create_accurate_solver(problem, solver_type="fixed_point")

# Research preset (maximum accuracy)
from mfg_pde.factory import create_research_solver
solver = create_research_solver(problem, solver_type="fixed_point")
```

## Troubleshooting

### "Solver doesn't converge"
1. Check you're using Tier 2 (Standard): `create_fast_solver()`
2. If using Tier 1 (Basic FDM), increase iterations or adjust damping
3. Try `create_accurate_solver()` for tighter tolerance

### "Mass conservation error too large"
1. Use Tier 2 (Standard) - it has perfect mass conservation
2. Avoid Tier 1 (Basic FDM) for production
3. Check boundary conditions are correct

### "Too slow"
1. Tier 2 (Standard) is already optimized
2. For speed, consider Tier 1 (Basic) but accept lower accuracy
3. For high-dimensional, try Tier 3 (DGM with GPU)

## Summary

**For 90% of users**: Just use Tier 2 (Standard)
```python
from mfg_pde.factory import create_fast_solver
solver = create_fast_solver(problem, "fixed_point")
```

**For benchmarking**: Use Tier 1 (Basic) as comparison
```python
from mfg_pde.factory import create_basic_solver
solver_basic = create_basic_solver(problem)
```

**For specialized needs**: Explore Tier 3 (Advanced)
```python
from mfg_pde.factory import create_accurate_solver
solver_advanced = create_accurate_solver(problem, solver_type="weno")
```

---

**Bottom line**: When in doubt, use `create_fast_solver()` ✅

It's fast enough, accurate enough, and just works.
