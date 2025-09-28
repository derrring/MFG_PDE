# Quick Start Guide - Simple API

**Get started with MFG_PDE in under 5 minutes**

This guide covers the simple API for built-in problem types. For custom mathematical formulations (Hamiltonians, geometries, costs), you'll need the [Core Objects Guide](core_objects.md).

## Installation

```bash
pip install mfg-pde
```

## One-Line Solutions

Solve Mean Field Games with a single function call:

```python
from mfg_pde import solve_mfg

# Crowd dynamics simulation
result = solve_mfg("crowd_dynamics")
result.plot_solution()
```

That's it! You've solved a crowd evacuation problem and visualized the results.

## Common Problems

### Crowd Dynamics
Model pedestrian flows, evacuations, and crowd behavior:

```python
# Basic crowd evacuation
result = solve_mfg("crowd_dynamics")

# Larger crowd in bigger space
result = solve_mfg("crowd_dynamics",
                   domain_size=5.0,
                   crowd_size=500)

# High accuracy for research
result = solve_mfg("crowd_dynamics",
                   accuracy="high",
                   verbose=True)
```

### Portfolio Optimization
Financial portfolio optimization (Merton problem):

```python
# Basic portfolio optimization
result = solve_mfg("portfolio_optimization")

# Custom risk aversion
result = solve_mfg("portfolio_optimization",
                   risk_aversion=0.3,
                   time_horizon=2.0)
```

### Traffic Flow
Traffic dynamics with congestion:

```python
# Basic traffic simulation
result = solve_mfg("traffic_flow")

# High-speed scenario
result = solve_mfg("traffic_flow",
                   speed_limit=2.0,
                   domain_size=10.0)
```

### Epidemic Models
Epidemic spreading with control:

```python
# Basic epidemic model
result = solve_mfg("epidemic")

# High transmission rate
result = solve_mfg("epidemic",
                   infection_rate=0.7,
                   time_horizon=5.0)
```

## Understanding Results

All results have the same simple interface:

```python
result = solve_mfg("crowd_dynamics")

# Visualize the solution
result.plot_solution()           # Interactive plot

# Access numerical data
density = result.m               # Final density m(T,x)
value = result.u                 # Value function u(T,x)
x_grid = result.x_grid           # Spatial grid
t_grid = result.t_grid           # Time grid

# Check solution quality
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Total time: {result.total_time:.2f}s")
```

## Accuracy Levels

Choose the right balance of speed vs accuracy:

```python
# Fast (for quick prototyping)
result = solve_mfg("crowd_dynamics", accuracy="fast")

# Balanced (default - good for most uses)
result = solve_mfg("crowd_dynamics", accuracy="balanced")

# High (for research and publication)
result = solve_mfg("crowd_dynamics", accuracy="high")

# Research (with detailed debugging)
result = solve_mfg("crowd_dynamics", accuracy="research", verbose=True)
```

## Parameter Validation

Get help with parameter choices:

```python
from mfg_pde import validate_problem_parameters, suggest_problem_setup

# Check if parameters are valid
validation = validate_problem_parameters("crowd_dynamics", crowd_size=-10)
if not validation['valid']:
    print("Issues:", validation['issues'])
    print("Suggestions:", validation['suggestions'])

# Get suggested parameters
setup = suggest_problem_setup("crowd_dynamics")
print(f"Suggested domain size: {setup['domain_size']}")
print(f"Suggested parameters: {setup['parameters']}")
```

## Working Examples

### **Built-in Examples (Tier 1)**

```python
from mfg_pde import load_example

# These examples work immediately with current API:
result = load_example("simple_crowd")       # ✅ Small crowd evacuation
result = load_example("portfolio_basic")    # ✅ Basic portfolio optimization
result = load_example("traffic_light")      # ✅ Traffic light problem
result = load_example("epidemic_basic")     # ✅ Simple epidemic model

# All return MFGResult objects with .plot_solution() method
result.plot_solution()  # Interactive visualization
```

### **Available Example Files**

**✅ Tier 1 Examples (`examples/basic/` - Simple API):**
- `simple_api_example.py` - Complete Tier 1 API showcase
- `el_farol_bar_example.py` - Economic coordination problem
- `towel_beach_example.py` - User-friendly beach scenario
- `visualization_example.py` - Mathematical plotting demonstration

**✅ Tier 2 & 3 Examples (`examples/advanced/` - Advanced APIs):**
- `new_api_core_objects_example.py` - Tier 2 Core Objects showcase
- `new_api_hooks_example.py` - Tier 3 Advanced Hooks showcase
- `pinn_mfg_example.py` - Physics-informed neural networks
- Plus 8+ other advanced research examples

**🎯 Directory Structure:**
- **`examples/basic/`** = **Tier 1 only** (`solve_mfg()` function)
- **`examples/advanced/`** = **Tier 2 & 3** (Core Objects + Hooks)
- All examples ready to run with `python filename.py`

## Getting Help

```python
from mfg_pde import get_available_problems

# See all available problem types
problems = get_available_problems()
for name, info in problems.items():
    print(f"{name}: {info['description']}")
    print(f"  Typical domain size: {info['typical_domain_size']}")
    print(f"  Recommended accuracy: {info['recommended_accuracy']}")
```

## What's Next?

### **Ready for research problems?**
Most MFG users need custom mathematical formulations:

- **Custom Hamiltonians** H(x,p,m,t) → [Core Objects Guide](core_objects.md)
- **Custom geometries** and boundary conditions → [Core Objects Guide](core_objects.md)
- **Custom costs** and terminal conditions → [Core Objects Guide](core_objects.md)
- **Non-standard problem types** → [Core Objects Guide](core_objects.md)

### **Advanced users:**
- **Algorithm development** → [Advanced Hooks Guide](advanced_hooks.md)
- **Migrating from old API** → [Migration Guide](migration.md)
- **More examples** → [Examples Gallery](../examples/)

### **Remember:**
The simple API covers ~60% of use cases (teaching, benchmarking, standard problems). For research with custom mathematical components, Tier 2 (Core Objects) is your starting point.

## Complete Example

```python
from mfg_pde import solve_mfg

# Solve a crowd evacuation problem
result = solve_mfg("crowd_dynamics",
                   domain_size=3.0,      # 3-meter corridor
                   crowd_size=200,       # 200 people
                   time_horizon=1.5,     # 1.5 seconds
                   accuracy="high",      # High accuracy
                   verbose=True)         # Show progress

# Visualize results
result.plot_solution()

# Check solution quality
print(f"✅ Converged in {result.iterations} iterations")
print(f"📊 Convergence status: {result.converged}")
print(f"⏱️ Total solve time: {result.total_time:.2f}s")
```

**That's it!** You're now solving sophisticated Mean Field Games with just a few lines of code.
