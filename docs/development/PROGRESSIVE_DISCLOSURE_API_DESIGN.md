# Two-Level API Design for MFG_PDE

**Date**: 2025-10-05 (Revised for research-focused audience)
**Status**: Implementation Complete
**Philosophy**: Research-grade by default, extensible for developers

---

## Executive Summary

MFG_PDE provides a **two-level API** designed for academic researchers and industrial practitioners who understand MFG systems:

- **Level 1 (95%)**: Users (Researchers & Practitioners) - Full algorithm access via factory
- **Level 2 (5%)**: Developers (Core Contributors) - Infrastructure modification

**Design Principle**: Assume users understand Mean Field Games. No "dumbed down" API—start with full research capabilities.

---

## Two User Levels

### 📚 Level 1: Users - Researchers & Practitioners (95%)
**Who**: PhD students, postdocs, professors, industrial researchers
**Assumption**: Understand MFG theory (HJB-FP systems, Nash equilibria)
**Needs**: Access all algorithms, compare methods, benchmark, custom problems
**Access**: Full algorithm API via factory

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import (
    create_basic_solver,    # Tier 1: Basic FDM (benchmark)
    create_fast_solver,     # Tier 2: Hybrid (standard, DEFAULT)
    create_accurate_solver  # Tier 3: Advanced (WENO, DGM, etc.)
)

# Standard usage - DEFAULT is Tier 2 (good quality, mass-conserving)
problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()

# Compare different algorithms for research
solver_fdm = create_basic_solver(problem)        # Benchmark
solver_weno = create_accurate_solver(problem, solver_type="weno")  # High-order

# Access results
print(result.U)  # Value function u(t,x)
print(result.M)  # Density m(t,x)
```

**Key Features for Users**:
- **Algorithm selection**: Choose from 3 solver tiers based on quality needs
- **Method comparison**: Benchmark multiple algorithms easily
- **Custom problems**: Define custom Hamiltonians, boundary conditions, geometries
- **Configuration control**: Full access to solver parameters (tolerance, max iterations, damping)

### 🔧 Level 2: Developers - Core Contributors (5%)
**Who**: Package maintainers, algorithm developers, infrastructure contributors
**Needs**: Add new numerical methods, modify core solvers, extend framework
**Access**: Full infrastructure API (base classes, factory registration)

```python
from mfg_pde.alg.numerical.hjb_solvers import BaseHJBSolver
from mfg_pde.alg.numerical.fp_solvers import BaseFPSolver
from mfg_pde.factory import SolverFactory

# Implement new solver from scratch
class MyCustomHJBSolver(BaseHJBSolver):
    def solve_hjb_system(self, M, final_u, U_prev):
        # Custom HJB implementation
        pass

class MyCustomFPSolver(BaseFPSolver):
    def solve_fp_system(self, m_init, U):
        # Custom FP implementation
        pass

# Register with factory for all users
SolverFactory.register_solver("my_custom", MyCustomHJBSolver, MyCustomFPSolver)
```

**Key Features for Developers**:
- **Base class hierarchy**: Extend `BaseHJBSolver`, `BaseFPSolver`, `BaseMFGSolver`
- **Factory integration**: Register new solvers for seamless user access
- **Infrastructure modification**: Add backends, geometry types, boundary conditions

---

## Mapping: User Levels to Solver Tiers

**Important**: User levels and solver tiers are **orthogonal concepts**:

| User Level | API Access | Typical Tier Usage | Purpose |
|------------|-----------|-------------------|----|
| **Users (Researchers)** | Factory API | All tiers | Research, benchmark, production |
| **Developers** | Core API | Create new tiers | Extend framework |

### Solver Tiers (Algorithm Quality)

- **Tier 1**: Basic FDM (poor quality, ~1-10% mass error, benchmark only)
- **Tier 2**: Hybrid (good quality, ~10⁻¹⁵ mass error, **DEFAULT**)
- **Tier 3**: Advanced (specialized methods: WENO, Semi-Lagrangian, DGM)

**Example**: A researcher (Level 1) uses all three tiers for comparison, defaults to Tier 2 for production.

---

## Implementation: Two-Level API Architecture

### Level 1: Factory API (Users - Researchers & Practitioners)

**Entry point**: Algorithm selection and comparison

```python
from mfg_pde.factory import (
    create_basic_solver,    # Tier 1: Benchmark
    create_fast_solver,     # Tier 2: Standard (default)
    create_accurate_solver  # Tier 3: Advanced
)

# Full control over algorithm choice
solver = create_basic_solver(problem, damping=0.6)
solver = create_fast_solver(problem, "fixed_point")
solver = create_accurate_solver(problem, solver_type="weno")
```

**Usage patterns**:
```python
# Pattern 1: Method comparison
solvers = {
    "FDM": create_basic_solver(problem),
    "Hybrid": create_fast_solver(problem, "fixed_point"),
    "WENO": create_accurate_solver(problem, solver_type="weno"),
    "Semi-Lagrangian": create_accurate_solver(problem, solver_type="semi_lagrangian")
}

results = {name: solver.solve() for name, solver in solvers.items()}

# Pattern 2: Benchmarking
from mfg_pde.benchmarks import BenchmarkSuite
suite = BenchmarkSuite(problem)
suite.add_solver("Basic FDM", create_basic_solver(problem))
suite.add_solver("Standard", create_fast_solver(problem, "fixed_point"))
suite.run_comparison()
```

### Level 2: Core API (Developers)

**Entry point**: Base classes and infrastructure

```python
from mfg_pde.alg.numerical.hjb_solvers import BaseHJBSolver
from mfg_pde.alg.numerical.fp_solvers import BaseFPSolver
from mfg_pde.alg.numerical.mfg_solvers import BaseMFGSolver

# Direct access to infrastructure
class MyNewSolver(BaseMFGSolver):
    def solve(self):
        # Modify core algorithm
        pass

# Extend factory system
from mfg_pde.factory.solver_factory import SolverFactory
SolverFactory.register("my_solver", MyNewSolver)
```

---

## Usage Examples by User Level

### User Example (Researcher)
```python
"""
Academic user: Comparing methods for a paper
"""
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_basic_solver, create_fast_solver, create_accurate_solver
import matplotlib.pyplot as plt

problem = ExampleMFGProblem(Nx=100, Nt=50, T=1.0)

# Test different solvers
methods = {
    "FDM (Tier 1)": create_basic_solver(problem),
    "Hybrid (Tier 2)": create_fast_solver(problem, "fixed_point"),
    "WENO (Tier 3)": create_accurate_solver(problem, solver_type="weno"),
    "Semi-Lag (Tier 3)": create_accurate_solver(problem, solver_type="semi_lagrangian")
}

# Run and compare
results = {}
for name, solver in methods.items():
    results[name] = solver.solve()
    print(f"{name}: {results[name].iterations} iterations, "
          f"mass error: {results[name].mass_conservation_error:.2e}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for (name, result), ax in zip(results.items(), axes.flat):
    ax.imshow(result.M, aspect='auto')
    ax.set_title(name)
plt.savefig("solver_comparison.png")
```

### Developer Example (Core Contributor)
```python
"""
Developer: Adding a new Semi-Implicit solver
"""
from mfg_pde.alg.numerical.hjb_solvers import BaseHJBSolver
from mfg_pde.alg.numerical.fp_solvers import BaseFPSolver
import numpy as np

class SemiImplicitHJBSolver(BaseHJBSolver):
    """New semi-implicit HJB solver."""

    def __init__(self, problem, theta=0.5):
        super().__init__(problem)
        self.theta = theta  # Crank-Nicolson parameter
        self.hjb_method_name = "SemiImplicit"

    def solve_hjb_system(self, M, final_u, U_prev):
        # Implement semi-implicit scheme
        # theta = 0.5 → Crank-Nicolson
        # theta = 1.0 → Fully implicit
        Nx, Nt = self.problem.Nx + 1, self.problem.Nt + 1
        U = np.zeros((Nt, Nx))
        U[-1, :] = final_u

        # Backward solve with semi-implicit scheme
        for k in range(Nt - 2, -1, -1):
            # Semi-implicit update
            U[k, :] = self._semi_implicit_step(U[k+1, :], M[k, :], U_prev[k, :])

        return U

    def _semi_implicit_step(self, u_next, m_current, u_prev):
        # Implementation details...
        pass

# Register with factory
from mfg_pde.factory.solver_factory import SolverFactory

def create_semi_implicit_solver(problem, **kwargs):
    from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    hjb_solver = SemiImplicitHJBSolver(problem)
    fp_solver = FPParticleSolver(problem, num_particles=5000)
    # ... create full solver
    return solver

# Now users can use it
from mfg_pde.factory import create_semi_implicit_solver
solver = create_semi_implicit_solver(problem)
```

---

## Documentation Structure

```
docs/
├── user/                          # For Users (Researchers & Practitioners)
│   ├── quickstart.md             # 5-minute tutorial with factory API
│   ├── SOLVER_SELECTION_GUIDE.md # Choosing solver tiers (Basic/Standard/Advanced)
│   ├── factory_reference.md      # All create_*_solver() functions
│   ├── solver_comparison.md      # Benchmarking guide
│   ├── algorithm_details.md      # Mathematical details of each method
│   └── examples/                  # Research examples
│
└── development/                   # For Developers (Core Contributors)
    ├── CORE_API_REFERENCE.md     # Base classes, extension points
    ├── adding_new_solvers.md     # How to extend framework
    ├── infrastructure.md          # Architecture details
    └── factory_registration.md    # How to register new solvers
```

---

## Summary Table

### User Level Mapping

| User Level | What They See | Entry Point | Solver Tiers Used |
|------------|---------------|-------------|-------------------|
| **📚 Users (Researchers)** | Factory API | `create_*_solver()` | All tiers (default: Tier 2) |
| **🔧 Developers** | Core API | Base classes | Create new tiers |

### Access Pattern

```python
# Users - Researchers & Practitioners (95% of users)
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver, create_accurate_solver

problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)

# Default: Use Tier 2 (Standard, mass-conserving)
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()

# Research: Compare multiple methods
solver_weno = create_accurate_solver(problem, solver_type="weno")

# Developers - Core Contributors (5% of users)
from mfg_pde.alg.numerical.hjb_solvers import BaseHJBSolver

class MyNewSolver(BaseHJBSolver):  # Extend framework
    def solve_hjb_system(self, M, final_u, U_prev):
        # Custom implementation
        pass
```

This creates a **research-grade design**: full capabilities by default, extensible for contributors.

---

## Implementation Status

- ✅ Factory API (Users) - complete
- ✅ Solver tiers 1-3 - complete
- ✅ Core API (Developers) - complete
- 📝 Documentation structure - needs update for two-level design

## Next Steps

1. ~~Remove `solve_mfg()` simple API~~ - **ELIMINATED** (researchers don't need it)
2. Update documentation for two user levels (Users vs Developers)
3. Emphasize factory API as the **primary entry point**
4. Create research-focused examples

---

**Last Updated**: 2025-10-05
**Consolidated from**: `api_architecture.md`, `USER_LEVEL_API_DESIGN.md`
**Revised**: Eliminated "basic user" level - package assumes MFG knowledge
**Philosophy**: Research-grade by default, extensible for contributors
