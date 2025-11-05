# Two-Level API Design for MFG_PDE

**Date**: 2025-11-03 (Updated with factory vs modular clarification)
**Status**: Implementation Complete
**Philosophy**: Research-grade by default, extensible for developers

---

## Executive Summary

MFG_PDE provides a **two-level API** designed for academic researchers and industrial practitioners who understand MFG systems:

- **Level 1 (95%)**: Users (Researchers & Practitioners) - Modular composition + Domain templates
- **Level 2 (5%)**: Developers (Core Contributors) - Infrastructure modification

**Design Principle**: Assume users understand Mean Field Games. No "dumbed down" API—start with full research capabilities.

**Key Distinction**:
- **Modular Approach** (Recommended): Explicit composition with full control
- **Domain Templates** (Future): Domain-specific patterns for mature applications (crowd motion, epidemic, finance)

---

## Two User Levels

### 📚 Level 1: Users - Researchers & Practitioners (95%)
**Who**: PhD students, postdocs, professors, industrial researchers
**Assumption**: Understand MFG theory (HJB-FP systems, Nash equilibria)
**Needs**: Access all algorithms, compare methods, benchmark, custom problems
**Access**: Modular Approach (Recommended) + Domain Templates (Future)

#### Modular Approach (Recommended)

For research, custom algorithms, and fine-grained control:

```python
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver, HJBGFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver, FPFDMSolver
from mfg_pde.alg.numerical.coupling import FixedPointIterator

# Compose custom solver configuration
hjb_solver = HJBGFDMSolver(
    problem,
    collocation_points=custom_points,
    max_newton_iterations=50
)
fp_solver = FPParticleSolver(
    problem,
    num_particles=10000,
    kde_bandwidth=0.1
)

# Create MFG solver with explicit composition
mfg_solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    damping_factor=0.7,
    use_anderson=True
)

result = mfg_solver.solve()
```

**Use modular approach when**:
- Research and experimentation (primary use case)
- Testing novel solver combinations
- Need fine control over solver parameters
- Custom Hamiltonian or geometry requires specific solvers
- Investigating algorithmic behavior
- Learning the framework internals

**Advantages**:
- Clear what you're getting (no hidden defaults)
- Full control over all parameters
- Easy to modify and experiment
- Explicit composition aids understanding

#### Domain Templates (Future)

For mature application domains with established best practices:

```python
from mfg_pde.templates import (
    create_crowd_motion_solver,   # Crowd evacuation, pedestrian dynamics
    create_epidemic_solver,        # SIR/SEIR models, disease spread
    create_finance_solver,         # Portfolio optimization, trading
    create_traffic_solver          # Traffic flow, congestion pricing
)

# Templates encode domain knowledge
problem = CrowdMotionProblem(...)  # Domain-specific problem class
solver = create_crowd_motion_solver(
    problem,
    num_particles=5000,
    use_anisotropy=True,      # Domain-specific option
    exit_absorption=True       # Typical for evacuation scenarios
)
result = solver.solve()
```

**Use domain templates when**:
- Working in well-established application domains
- Want domain-specific best practices
- Need production-ready defaults for specific fields
- Prefer domain terminology over algorithmic details

**Key Features for Users**:
- **Modular Approach**: Explicit composition with full control (recommended)
- **Domain Templates**: Domain-specific patterns for mature applications (future)
- **Method selection**: Choose from FDM, WENO, GFDM, DGM, particle methods
- **Algorithm comparison**: Benchmark multiple methods with explicit composition
- **Custom problems**: Define custom Hamiltonians, boundary conditions, geometries
- **Configuration control**: Full access to all solver parameters

### 🔧 Level 2: Developers - Core Contributors (5%)
**Who**: Package maintainers, algorithm developers, infrastructure contributors
**Needs**: Add new numerical methods, modify core solvers, extend framework
**Access**: Full infrastructure API (base classes, factory registration)

```python
from mfg_pde.alg.numerical.hjb_solvers import BaseHJBSolver
from mfg_pde.alg.numerical.fp_solvers import BaseFPSolver
from mfg_pde.alg.numerical.coupling import BaseMFGSolver
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

# Register with factory for all users (optional - enables factory mode)
SolverFactory.register_solver("my_custom", MyCustomHJBSolver, MyCustomFPSolver)
```

**Key Features for Developers**:
- **Base class hierarchy**: Extend `BaseHJBSolver`, `BaseFPSolver`, `BaseMFGSolver`
- **Factory integration**: Register new solvers for factory mode access
- **Infrastructure modification**: Add backends, geometry types, boundary conditions
- **Coupling algorithms**: Implement new MFG coupling methods in `alg/numerical/coupling/`

---

## Mapping: User Levels to Access Patterns

**Important**: User levels and access patterns are **independent concepts**:

| User Level | Access Pattern | API Entry Point | Purpose |
|------------|---------------|-----------------|---------|
| **Users (Researchers)** | Modular Approach | Direct solver imports | Research, custom configurations (recommended) |
| **Users (Researchers)** | Domain Templates | `create_[domain]_solver()` | Mature domains with best practices (future) |
| **Developers** | Core API | Base classes | Extend framework |

### Access Patterns (Level 1 Users)

1. **Modular Approach** (Recommended)
   - Explicit composition with full control
   - Clear what you're getting (no guessing)
   - Entry: `from mfg_pde.alg.numerical.{hjb_solvers,fp_solvers,coupling}`
   - Direct access to: HJBFDMSolver, HJBWENOSolver, HJBGFDMSolver, FPParticleSolver, FPFDMSolver, etc.
   - **Use for**: Research, custom problems, learning the framework

2. **Domain-Specific Templates** (Future)
   - Templates that understand problem domains
   - Entry: `from mfg_pde.templates import create_crowd_motion_solver`
   - Domain knowledge baked in (crowd motion, epidemic, finance, traffic)
   - **Use for**: Mature application domains with established best practices

### Legacy: Generic Factory Functions (Not Recommended)

Generic factory functions exist but lack domain knowledge:

```python
# These exist but don't understand your problem's needs
from mfg_pde.factory import create_fast_solver, create_accurate_solver

solver = create_fast_solver(problem)  # Generic defaults, may not suit your problem
```

**Why not recommended**:
- Names like "fast" or "accurate" are arbitrary and context-dependent
- No domain knowledge about your specific problem
- Quality depends on configuration (Nx, Nt, tolerance), not function name
- **Use modular approach instead** for clarity and control

**Example**: A researcher uses modular approach to compose custom GFDM+Particle solver, choosing parameters based on their specific problem requirements.

---

## Modular Approach vs Domain Templates

Both are available to **Level 1 users** (researchers & practitioners). Choose based on your use case:

### When to Use Modular Approach (Recommended)

```python
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver
from mfg_pde.alg.numerical.coupling import FixedPointIterator

# Explicit composition - you know exactly what you're getting
hjb = HJBGFDMSolver(problem, collocation_points=custom_points)
fp = FPParticleSolver(problem, num_particles=10000)
solver = FixedPointIterator(problem, hjb, fp, damping_factor=0.7)
```

**Use cases**:
- ✅ Research and experimentation
- ✅ Custom problems requiring specific methods
- ✅ Learning the framework
- ✅ When you need full control over configuration
- ✅ Prototyping new methods

### When to Use Domain Templates (Future)

```python
from mfg_pde.templates import create_crowd_motion_solver

# Domain template knows crowd motion requirements
solver = create_crowd_motion_solver(
    problem,
    num_particles=5000,
    use_anisotropy=True,
    exit_attraction=True
)
```

**Use cases**:
- ✅ Mature application domains (crowd motion, epidemic, finance)
- ✅ When domain best practices are established
- ✅ Production deployment in known domains
- ✅ Teaching domain-specific applications

### Relationship

**Domain templates use modular approach internally**:

```python
# What a domain template does:
def create_crowd_motion_solver(problem, num_particles=5000):
    # Encode domain expertise
    hjb = HJBFDMSolver(problem, max_newton_iterations=30)
    fp = FPParticleSolver(
        problem,
        num_particles=num_particles,
        normalize_kde_output=True  # Crowd motion needs this
    )
    return FixedPointIterator(
        problem, hjb, fp,
        damping_factor=0.5,  # Works well for crowd problems
        use_anderson=True
    )
```

Templates encode **domain knowledge**. For research or custom needs, use modular approach directly for full control.

---

## Implementation: Two-Level API Architecture

### Level 1: Modular Approach (Users - Recommended)

**Entry point**: Direct solver imports with explicit composition

```python
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver
from mfg_pde.alg.numerical.coupling import FixedPointIterator

# Create individual solvers with custom configuration
hjb_solver = HJBGFDMSolver(
    problem,
    collocation_points=my_custom_points,
    max_newton_iterations=50,
    newton_tolerance=1e-8
)

fp_solver = FPParticleSolver(
    problem,
    num_particles=10000,
    kde_bandwidth=0.1,
    normalize_kde_output=True
)

# Compose MFG solver with full control
mfg_solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    damping_factor=0.7,
    use_anderson=True,
    anderson_depth=10
)

# Solve
U, M, info = mfg_solver.solve(max_iterations=50, tolerance=1e-6)
```

**Usage patterns**:
```python
# Pattern 1: Custom solver combinations
from mfg_pde.alg.numerical.hjb_solvers import HJBWENOSolver, HJBGFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver, FPDGMSolver

# Test different combinations
combinations = [
    (HJBWENOSolver(problem), FPParticleSolver(problem, num_particles=5000)),
    (HJBGFDMSolver(problem), FPDGMSolver(problem)),
]

for hjb, fp in combinations:
    solver = FixedPointIterator(problem, hjb, fp)
    result = solver.solve()
    # Analyze results...

# Pattern 2: Parameter sweeps
for damping in [0.3, 0.5, 0.7]:
    for num_particles in [3000, 5000, 10000]:
        hjb = HJBFDMSolver(problem)
        fp = FPParticleSolver(problem, num_particles=num_particles)
        solver = FixedPointIterator(problem, hjb, fp, damping_factor=damping)
        # Run experiments...
```

### Level 2: Core API (Developers)

**Entry point**: Base classes and infrastructure

```python
from mfg_pde.alg.numerical.hjb_solvers import BaseHJBSolver
from mfg_pde.alg.numerical.fp_solvers import BaseFPSolver
from mfg_pde.alg.numerical.coupling import BaseMFGSolver

# Direct access to infrastructure
class MyNewSolver(BaseMFGSolver):
    def solve(self):
        # Modify core algorithm
        pass

# Extend factory system (optional)
from mfg_pde.factory.solver_factory import SolverFactory
SolverFactory.register("my_solver", MyNewSolver)
```

---

## Usage Examples by User Level

### User Example 1: Modular Approach (Recommended)
```python
"""
Researcher: Method comparison using modular composition
"""
from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver, HJBWENOSolver, HJBGFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver, FPFDMSolver
from mfg_pde.alg.numerical.coupling import FixedPointIterator
import matplotlib.pyplot as plt

problem = ExampleMFGProblem(Nx=100, Nt=50, T=1.0)

# Create explicit solver combinations
methods = {
    "FDM-only": (HJBFDMSolver(problem), FPFDMSolver(problem)),
    "Hybrid (FDM+Particle)": (HJBFDMSolver(problem), FPParticleSolver(problem, num_particles=5000)),
    "WENO+Particle": (HJBWENOSolver(problem), FPParticleSolver(problem, num_particles=5000)),
    "GFDM+Particle": (HJBGFDMSolver(problem), FPParticleSolver(problem, num_particles=5000)),
}

# Run and compare
results = {}
for name, (hjb, fp) in methods.items():
    solver = FixedPointIterator(problem, hjb, fp, damping_factor=0.6)
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

### User Example 2: Domain Template (Future)
```python
"""
Researcher: Using domain-specific template for crowd motion
"""
from mfg_pde import ExampleMFGProblem
from mfg_pde.templates import create_crowd_motion_solver  # Future

problem = ExampleMFGProblem(Nx=100, Nt=50, T=1.0)

# Template encodes crowd motion domain knowledge
solver = create_crowd_motion_solver(
    problem,
    num_particles=5000,
    use_anisotropy=True,  # Domain-specific option
    exit_absorption=True   # Typical for evacuation
)

U, M, info = solver.solve()
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
│   ├── quickstart.md             # 5-minute tutorial with modular approach
│   ├── SOLVER_SELECTION_GUIDE.md # Choosing methods (HJB solvers, FP solvers, coupling)
│   ├── modular_reference.md      # Solver classes and composition patterns
│   ├── solver_comparison.md      # Benchmarking guide
│   ├── algorithm_details.md      # Mathematical details of each method
│   ├── domain_templates.md       # Domain-specific templates (future)
│   └── examples/                  # Research examples
│
└── development/                   # For Developers (Core Contributors)
    ├── CORE_API_REFERENCE.md     # Base classes, extension points
    ├── adding_new_solvers.md     # How to extend framework
    ├── infrastructure.md          # Architecture details
    └── template_creation.md       # How to create domain templates
```

---

## Summary Table

### User Level and Access Pattern Mapping

| User Level | Access Pattern | Entry Point | Use Case | Control Level |
|------------|---------------|-------------|----------|---------------|
| **📚 Users (Researchers)** | Modular Approach | Direct solver imports | Research, custom configurations, method comparison | Full control |
| **📚 Users (Researchers)** | Domain Templates | `create_[domain]_solver()` | Mature domains with best practices | Domain-specific |
| **🔧 Developers** | Core API | Base classes | Extend framework, new methods | Infrastructure |

**Note**: Generic factory functions (`create_fast_solver`, `create_accurate_solver`) are legacy and not recommended. Use modular approach for clarity.

### Quick Reference

```python
# LEVEL 1: Modular Approach (Recommended)
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver
from mfg_pde.alg.numerical.coupling import FixedPointIterator

problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)

# Explicit composition with full control
hjb = HJBGFDMSolver(problem, collocation_points=custom_points)
fp = FPParticleSolver(problem, num_particles=10000)
solver = FixedPointIterator(problem, hjb, fp, damping_factor=0.7)
U, M, info = solver.solve()

# LEVEL 2: Developers (Core contributors)
from mfg_pde.alg.numerical.hjb_solvers import BaseHJBSolver

class MyNewSolver(BaseHJBSolver):  # Extend framework
    def solve_hjb_system(self, M, final_u, U_prev):
        # Custom implementation
        pass
```

This creates a **research-grade design**: full capabilities by default (Level 1), extensible for contributors (Level 2).

**Key Insights**:
- **Modular approach is recommended** - explicit composition with full control
- **Domain templates** (future) encode domain-specific best practices
- **Generic factories are legacy** - arbitrary names like "fast" or "accurate" lack domain knowledge
- Quality depends on **configuration** (Nx, Nt, tolerance, damping), not function names

---

## Implementation Status

- ✅ Modular approach (Users) - complete and recommended
- ✅ Core API (Developers) - complete
- ⚠️ Generic factory functions - present but legacy/not recommended
- 📝 Domain templates - future implementation
- 📝 Documentation - needs alignment with modular-first philosophy

## Next Steps

1. ✅ ~~Remove `solve_mfg()` simple API~~ - **ELIMINATED** (researchers don't need it)
2. ✅ ~~Eliminate "Solver Tiers" terminology~~ - **COMPLETED** (2025-11-03)
3. 📝 Update user documentation to emphasize modular approach
4. 📝 Implement domain-specific templates (crowd motion, epidemic, finance, traffic)
5. 📝 Add deprecation warnings to generic factory functions
6. 📝 Create migration guide from generic factories to modular approach

---

**Last Updated**: 2025-11-03
**Consolidated from**: `api_architecture.md`, `USER_LEVEL_API_DESIGN.md`
**Major Revisions**:
- 2025-10-05: Eliminated "basic user" level - package assumes MFG knowledge
- 2025-11-03: Eliminated "Solver Tiers", prioritized modular approach over generic factories
**Philosophy**: Research-grade by default, modular composition recommended, domain templates for mature applications
