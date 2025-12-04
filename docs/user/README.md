# MFG_PDE User Documentation

**Research-grade Mean Field Games solver for academic and industrial applications**

---

## 🚀 **Get Started in 5 Minutes**

```python
from mfg_pde import MFGProblem
from mfg_pde.factory import create_standard_solver

# Create problem
problem = MFGProblem(Nx=50, Nt=20, T=1.0)

# Solve with standard solver (mass-conserving, robust)
solver = create_standard_solver(problem, "fixed_point")
result = solver.solve()

# Access results
print(result.U)  # Value function u(t,x)
print(result.M)  # Density m(t,x)
```

**That's it!** You've solved a Mean Field Games system with research-grade quality.

---

## 📚 **Two-Level API Design**

MFG_PDE is designed for users who **understand Mean Field Games** (HJB-FP systems, Nash equilibria).

| **API Level** | **Target Users** | **Entry Point** | **What You Get** |
|---------------|------------------|-----------------|------------------|
| **Level 1: Users** | 95% - Researchers & Practitioners | Factory API | Full algorithm access, benchmarking |
| **Level 2: Developers** | 5% - Core Contributors | Base classes | Infrastructure extension |

### **📚 Level 1: Users - Researchers & Practitioners (95%)**

**Who**: PhD students, postdocs, professors, industrial researchers
**Assumption**: Understand HJB-FP systems, Nash equilibria, numerical PDEs
**Entry Point**: Factory API (`create_*_solver()`)

**What you get**:
- **Algorithm selection**: Choose from 3 solver tiers (Basic/Standard/Accurate)
- **Method comparison**: Benchmark FDM, Hybrid, and custom configurations
- **Custom problems**: Define your own Hamiltonians, geometries, boundary conditions
- **Full configuration**: Control tolerance, iterations, damping, backends

**Get started**: [Factory API Quickstart](quickstart.md)

```python
from mfg_pde.factory import (
    create_basic_solver,    # Tier 1: Basic FDM (benchmark)
    create_standard_solver, # Tier 2: Hybrid (DEFAULT - mass-conserving)
    create_accurate_solver  # Tier 3: Accurate configuration
)

# Standard usage (DEFAULT)
solver = create_standard_solver(problem, "fixed_point")
result = solver.solve()

# Accurate configuration
solver_accurate = create_accurate_solver(problem, "fixed_point", max_iterations=200)
result_accurate = solver_accurate.solve()
```

### **🔧 Level 2: Developers - Core Contributors (5%)**

**Who**: Package maintainers, algorithm developers
**Entry Point**: Base classes

**What you get**:
- **Extend base classes**: `BaseHJBSolver`, `BaseFPSolver`, `BaseMFGSolver`
- **Register new solvers**: Integrate into factory system
- **Modify infrastructure**: Add backends, geometries, boundary conditions

**Get started**: Extend the base solver classes below

```python
from mfg_pde.alg.numerical.hjb_solvers import BaseHJBSolver

class MyCustomSolver(BaseHJBSolver):
    def solve_hjb_system(self, M, final_u, U_prev):
        # Your implementation
        pass
```

---

## 📖 **Documentation Guide**

### **For Users (Researchers & Practitioners)**

#### **Start Here**
1. **[Factory API Quickstart](quickstart.md)** - 5-minute tutorial
2. **[Core Objects Guide](core_objects.md)** - Understanding Problem, Solver, Result classes

#### **Examples**
- **[Basic Examples](../../examples/basic/)** - Single-concept demonstrations
- **[Advanced Examples](../../examples/advanced/)** - Research-grade problems
- **[Notebooks](../../examples/notebooks/)** - Interactive tutorials

### **For Developers (Core Contributors)**

See **[Development Documentation](../development/)** for:
- Consistency guide and code standards
- Strategic development roadmap
- Type system documentation

---

## 🎯 **Choose Your Path**

### **Standard Research Use (95% of users)**
→ Start with [Factory API Quickstart](quickstart.md)

You need factory API if you want to:
- ✅ Solve MFG problems with standard methods
- ✅ Compare different numerical algorithms
- ✅ Benchmark solver performance
- ✅ Define custom Hamiltonians H(x,p,m,t)
- ✅ Specify custom geometries and boundary conditions

### **Package Development (5% of users)**
→ Read [Development Documentation](../development/)

You need developer API if you want to:
- ✅ Implement new numerical methods
- ✅ Add new solver algorithms
- ✅ Modify core infrastructure
- ✅ Contribute to the package

---

## ⚡ **Quick Examples**

### **Example 1: Standard Workflow**
```python
from mfg_pde import MFGProblem
from mfg_pde.factory import create_standard_solver

# Define problem
problem = MFGProblem(Nx=100, Nt=50, T=1.0)

# Solve with default (Tier 2: Hybrid, mass-conserving)
solver = create_standard_solver(problem, "fixed_point")
result = solver.solve()

# Check convergence
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Mass error: {result.mass_conservation_error:.2e}")
```

### **Example 2: Method Comparison**
```python
from mfg_pde.factory import create_basic_solver, create_standard_solver, create_accurate_solver

# Compare three solver tiers
solvers = {
    "Basic FDM": create_basic_solver(problem),
    "Standard (Hybrid)": create_standard_solver(problem, "fixed_point"),
    "Accurate": create_accurate_solver(problem, "fixed_point", max_iterations=200)
}

results = {name: solver.solve() for name, solver in solvers.items()}

# Compare mass conservation
for name, result in results.items():
    print(f"{name}: {result.mass_conservation_error:.2e}")
```

### **Example 3: Custom Problem**
```python
from mfg_pde.core import BaseMFGProblem
import numpy as np

class CustomCrowdProblem(BaseMFGProblem):
    def evaluate_hamiltonian(self, x, p, m, t):
        # Custom H(x, p, m, t)
        kinetic = 0.5 * p**2
        congestion = 0.2 * m * np.log(1 + m)
        return kinetic + congestion

# Use with factory API
problem = CustomCrowdProblem(Nx=50, Nt=20, T=1.0)
solver = create_standard_solver(problem, "fixed_point")
result = solver.solve()
```

---

## 💡 **Key Features**

✅ **Research-Grade**: Publication-quality solvers with rigorous validation
✅ **Algorithm Access**: Full control over numerical methods (FDM, Particle, WENO, Semi-Lagrangian)
✅ **Mass-Conserving**: Default solver achieves ~10⁻¹⁵ mass conservation error
✅ **Fast**: Multi-backend acceleration (PyTorch, JAX, Numba)
✅ **Benchmarking**: Easy comparison of multiple algorithms
✅ **Extensible**: Clean architecture for adding new methods
✅ **Well-Tested**: Comprehensive test suite and validation
✅ **Documented**: Complete API reference and examples

---

## 🔄 **Migration from Old API**

Already using MFG_PDE? The new API is backward compatible. See [quickstart](quickstart.md) for the latest patterns.

---

## 🤝 **Community and Support**

- **💬 Discussions**: [GitHub Discussions](https://github.com/derrring/MFG_PDE/discussions)
- **🐛 Issues**: [GitHub Issues](https://github.com/derrring/MFG_PDE/issues)

---

## 📋 **What You Should Know**

MFG_PDE assumes you understand:
- **Mean Field Games**: HJB-FP coupled systems, Nash equilibria
- **Numerical PDEs**: Finite difference methods, stability, convergence
- **Python**: Basic programming and scientific computing (NumPy)

If you need mathematical background, see:
- **[Theory Guide](../theory/)** - Mathematical formulations
- **[Notebooks](../../examples/notebooks/)** - Interactive tutorials with explanations

---

**Ready to get started?** → **[Factory API Quickstart](quickstart.md)**
