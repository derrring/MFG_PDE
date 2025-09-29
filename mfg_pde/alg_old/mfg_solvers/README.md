# MFG Solvers: Orchestration Layer for Mean Field Games

This directory contains the **orchestration layer** that combines Fokker-Planck (FP) and Hamilton-Jacobi-Bellman (HJB) solvers to solve complete Mean Field Games systems.

## 🏗️ **Architecture Overview**

Mean Field Games involve solving a **coupled system** of two PDEs:

```
HJB Equation (backward):  ∂u/∂t + H(x,∇u,m) - (σ²/2)Δu = 0
FP Equation (forward):    ∂m/∂t - div(m∇H_p) - (σ²/2)Δm = 0
```

**MFG Solvers** orchestrate the coupling through **Picard iteration**:

```python
for iteration in range(max_picard_iterations):
    # 1. Solve FP equation: u_old → m_new
    m_new = fp_solver.solve_fp_system(u_old)

    # 2. Solve HJB equation: m_new → u_new
    u_new = hjb_solver.solve_hjb_system(m_new, u_final, u_old)

    # 3. Check convergence
    if converged(u_new, u_old, m_new, m_old):
        return MFGResult(u_new, m_new)
```

## 🧩 **Design Patterns**

### **Pattern 1: Flexible Composition** (Recommended)
```python
from mfg_pde.alg.mfg_solvers import FixedPointIterator
from mfg_pde.alg.fp_solvers import FPParticleSolver
from mfg_pde.alg.hjb_solvers import HJBWenoSolver

# Mix any FP solver + any HJB solver
fp_solver = FPParticleSolver(problem, num_particles=5000)
hjb_solver = HJBWenoSolver(problem, weno_variant="weno5", cfl_number=0.3)

mfg_solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver
)
```

### **Pattern 2: Specialized Combinations**
```python
from mfg_pde.alg.mfg_solvers import ParticleCollocationSolver

# Pre-optimized combination: Particles + GFDM Collocation
solver = ParticleCollocationSolver(
    problem=problem,
    collocation_points=points,
    num_particles=5000
)
```

### **Pattern 3: Hybrid Methods**
```python
from mfg_pde.alg.mfg_solvers import HybridFPParticleHJBFDM

# Specific hybrid: Particle FP + FDM HJB
solver = HybridFPParticleHJBFDM(
    mfg_problem=problem,
    num_particles=5000,
    hjb_newton_iterations=30
)
```

## 📁 **Available MFG Solvers**

### **Core Solvers**

| Solver | FP Method | HJB Method | Architecture | Best For |
|--------|-----------|------------|--------------|----------|
| [`FixedPointIterator`](damped_fixed_point_iterator.py) | **Any** | **Any** | Flexible composition | Research & benchmarking |
| [`ConfigAwareFixedPointIterator`](config_aware_fixed_point_iterator.py) | **Configurable** | **Configurable** | Config-driven | Production workflows |

### **Specialized Combinations**

| Solver | FP Method | HJB Method | Optimized For |
|--------|-----------|------------|---------------|
| [`ParticleCollocationSolver`](particle_collocation_solver.py) | Particle | GFDM Collocation | Meshfree methods |
| [`HybridFPParticleHJBFDM`](hybrid_fp_particle_hjb_fdm.py) | Particle | Standard FDM | Robust hybrid approach |
| [`EnhancedParticleCollocationSolver`](enhanced_particle_collocation_solver.py) | Particle | Enhanced GFDM | Advanced particle methods |
| [`AdaptiveParticleCollocationSolver`](adaptive_particle_collocation_solver.py) | Adaptive Particle | GFDM | Dynamic refinement |

### **Advanced Solvers**

| Solver | Description | Features |
|--------|-------------|----------|
| [`AMRMFGSolver`](amr_mfg_solver.py) | Adaptive Mesh Refinement | Multi-resolution grids |
| [`LagrangianNetworkSolver`](lagrangian_network_solver.py) | Network-based | Graph/network problems |

## 🚀 **Quick Start Examples**

### **Example 1: High-Order Combination**
```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.fp_solvers import FPParticleSolver
from mfg_pde.alg.hjb_solvers import HJBWenoSolver
from mfg_pde.alg.mfg_solvers import FixedPointIterator

# Create problem
problem = ExampleMFGProblem(Nx=128, Nt=64, T=1.0)

# High-accuracy combination
fp_solver = FPParticleSolver(problem, num_particles=10000)
hjb_solver = HJBWenoSolver(problem, weno_variant="weno5", cfl_number=0.3)  # Fifth-order!

mfg_solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    thetaUM=0.5  # Damping parameter
)

# Solve
result = mfg_solver.solve(max_iterations=50, tolerance=1e-6)
print(f"✅ Converged in {result.picard_iterations} iterations")
```

### **Example 2: Robust Production Combination**
```python
from mfg_pde.alg.mfg_solvers import HybridFPParticleHJBFDM

# Pre-optimized hybrid solver
solver = HybridFPParticleHJBFDM(
    mfg_problem=problem,
    num_particles=5000,
    kde_bandwidth="scott",
    hjb_newton_iterations=30,
    hjb_newton_tolerance=1e-7
)

result = solver.solve(max_iterations=50, tolerance=1e-4)
```

### **Example 3: Meshfree Approach**
```python
import numpy as np
from mfg_pde.alg.mfg_solvers import ParticleCollocationSolver

# Create collocation points
x_points = np.linspace(0, 1, 100)
collocation_points = x_points.reshape(-1, 1)

solver = ParticleCollocationSolver(
    problem=problem,
    collocation_points=collocation_points,
    num_particles=5000,
    delta=0.1,  # Neighborhood radius
    taylor_order=2
)

result = solver.solve(max_iterations=40, tolerance=1e-5)
```

## 🎯 **Choosing the Right Solver**

### **For Research & Benchmarking**
✅ **Use**: `FixedPointIterator` with explicit FP/HJB solver choices
- Maximum flexibility for comparing methods
- Easy to swap solvers for academic studies
- Clear separation of concerns

### **For Production Applications**
✅ **Use**: Specialized solvers like `HybridFPParticleHJBFDM`
- Pre-optimized parameter combinations
- Robust convergence properties
- Performance-tuned implementations

### **For Complex Geometries**
✅ **Use**: `ParticleCollocationSolver` or `AdaptiveParticleCollocationSolver`
- Meshfree spatial discretization
- Natural handling of irregular domains
- Adaptive refinement capabilities

### **For High-Order Accuracy**
✅ **Use**: `FixedPointIterator` with `HJBWenoSolver` family
- Choose from WENO5, WENO-Z, WENO-M, WENO-JS variants
- Fifth-order spatial accuracy with enhanced properties
- Non-oscillatory reconstruction
- Ideal for smooth problems and discontinuous solutions

## 🔧 **Common Solver Combinations**

### **Academic Benchmarking**
```python
combinations = {
    "standard": (FPFDMSolver, HJBFDMSolver),
    "high_order": (FPFDMSolver, HJBWenoSolver),
    "particle_robust": (FPParticleSolver, HJBFDMSolver),
    "particle_accurate": (FPParticleSolver, HJBWenoSolver),
    "meshfree": (FPParticleSolver, HJBGFDMSolver)
}
```

### **Problem-Specific Recommendations**
| Problem Type | Recommended Combination | Reason |
|--------------|-------------------------|--------|
| **Smooth solutions** | FP: Standard FDM + HJB: WENO Family | High-order accuracy with variant selection |
| **Discontinuous solutions** | FP: Particles + HJB: WENO-Z/WENO-M | Enhanced non-oscillatory handling |
| **Complex geometry** | FP: Particles + HJB: GFDM | Meshfree flexibility |
| **Large scale** | FP: Standard FDM + HJB: Standard FDM | Computational efficiency |
| **High precision** | FP: Particles + HJB: WENO Family | Maximum accuracy with optimal variant |

## 📊 **Performance Considerations**

### **Computational Complexity**
| Solver Type | Time Complexity | Memory Usage | Parallelization |
|-------------|-----------------|--------------|-----------------|
| **FDM + FDM** | O(N log N) | O(N) | Excellent |
| **Particle + FDM** | O(P + N log N) | O(P + N) | Good |
| **Particle + WENO Family** | O(P + N) | O(P + N) | Excellent |
| **Particle + GFDM** | O(P × M) | O(P + M) | Moderate |

Where: N = grid points, P = particles, M = collocation points

### **Convergence Properties**
- **Standard FDM**: Robust, predictable convergence
- **WENO Family**: Adaptive convergence with variant-specific optimizations
- **Particle methods**: Excellent mass conservation
- **Hybrid approaches**: Best of both worlds

## 🔮 **Future Roadmap**

See **[Issue #17](https://github.com/derrring/MFG_PDE/issues/17)** for the comprehensive roadmap including:

### **Phase 1: High-Order Extensions** (Q1-Q2 2025)
- **WENO-Z** enhanced weights
- **2D WENO** multi-dimensional extensions
- **Discontinuous Galerkin** methods

### **Phase 2: Neural Network Methods** (Q2-Q3 2025)
- **Physics-Informed Neural Networks (PINNs)**
- **Deep Galerkin Method (DGM)**
- **Neural Operator** approaches

### **Phase 3: Advanced Hybrid Methods** (Q3-Q4 2025)
- **Adaptive method selection**
- **Multi-scale approaches**
- **Domain decomposition**

## 🧪 **Development Guidelines**

### **Creating New MFG Solvers**

1. **Inherit from `MFGSolver`**:
```python
from mfg_pde.alg.base_mfg_solver import MFGSolver

class MyCustomMFGSolver(MFGSolver):
    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        # Initialize component solvers
        self.fp_solver = SomeFPSolver(problem)
        self.hjb_solver = SomeHJBSolver(problem)
```

2. **Implement required methods**:
```python
def solve(self, max_iterations, tolerance=1e-5, **kwargs):
    # Picard iteration logic
    pass

def get_results(self):
    return self.U, self.M
```

3. **Follow naming conventions**:
```python
# Pattern: {FP_Method}{HJB_Method}MFGSolver
class ParticleWeno5MFGSolver(MFGSolver):
    pass
```

### **Testing New Solvers**

```python
# Always test with standard benchmark problems
from mfg_pde import ExampleMFGProblem

problem = ExampleMFGProblem(Nx=64, Nt=32, T=1.0)
solver = MyCustomMFGSolver(problem)
result = solver.solve(max_iterations=50, tolerance=1e-6)

# Verify convergence and conservation properties
assert result.convergence_achieved
assert abs(np.trapz(result.M_solution[-1, :]) * problem.Dx - 1.0) < 1e-6
```

## 📚 **References**

- **Base Classes**: [`../base_mfg_solver.py`](../base_mfg_solver.py)
- **FP Solvers**: [`../fp_solvers/`](../fp_solvers/)
- **HJB Solvers**: [`../hjb_solvers/`](../hjb_solvers/)
- **Examples**: [`../../../examples/advanced/`](../../../examples/advanced/)
- **Documentation**: [`../../../docs/theory/`](../../../docs/theory/)

## 🤝 **Contributing**

When adding new MFG solvers:

1. **Document the combination**: Clearly specify which FP + HJB methods are used
2. **Include benchmarks**: Compare against existing solvers
3. **Add examples**: Provide working usage examples
4. **Update this README**: Add your solver to the tables above

The modular architecture makes it easy to experiment with new combinations and contribute to the state-of-the-art in Mean Field Games research!
