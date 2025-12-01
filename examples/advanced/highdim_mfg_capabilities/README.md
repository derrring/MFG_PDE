# High-Dimensional MFG Capabilities

> **⚠️ DEPRECATED (v0.14.0)**: `GridBasedMFGProblem`, `HighDimMFGProblem`, and `HybridMFGSolver` have been removed. Use `MFGProblem` with `spatial_bounds` and `spatial_discretization` parameters for nD problems on tensor product grids.

This directory demonstrates the extended high-dimensional capabilities of the MFG_PDE package, including 2D, 3D, and nD Mean Field Games.

## 🎯 **Key Features Implemented**

### **Multi-Dimensional Infrastructure (v0.14.0+)**
- **`MFGProblem`**: Unified class supporting 1D, 2D, 3D, and nD problems
- **`spatial_bounds`**: List of tuples defining domain bounds per dimension
- **`spatial_discretization`**: List of grid points per dimension
- **`Domain3D`**: (Future) Complete 3D geometry with tetrahedral mesh generation

### **Solver Capabilities**
- **Damped Fixed Point**: Multi-dimensional ready (using existing `FixedPointIterator`)
- **Hybrid Methods**: Adaptive strategy switching with two-phase fixed-point approach
- **Complex Geometry**: Full integration with Gmsh → Meshio → PyVista pipeline

## 📁 **File Structure**

```
highdim_mfg_capabilities/
├── README.md                          # This documentation
├── demo_2d_complex_geometry.py        # 2D MFG on complex domains
├── demo_3d_box_domain.py              # 3D MFG on simple box
├── demo_3d_sphere_crowds.py           # 3D crowd dynamics on sphere
├── demo_hybrid_solver.py              # Multi-strategy hybrid solving
├── benchmark_performance.py           # Performance comparison 2D vs 3D
└── validation/
    ├── test_convergence.py            # Convergence validation
    └── compare_with_analytical.py     # Analytical solution comparison
```

## 🚀 **Quick Start Examples**

### **2D Grid-Based MFG (Modern API)**
```python
from mfg_pde import MFGProblem
from mfg_pde.factory import create_basic_solver

# 2D problem on tensor product grid
problem = MFGProblem(
    spatial_bounds=[(0, 2), (0, 1)],  # 2D domain
    spatial_discretization=[64, 32],   # 64x32 grid
    T=1.0,
    Nt=100,
    sigma=0.1,
)

# Solve MFG problem
solver = create_basic_solver(problem)
result = solver.solve()
```

### **3D Grid-Based MFG (Modern API)**
```python
from mfg_pde import MFGProblem
from mfg_pde.factory import create_basic_solver

# 3D box domain with regular grid
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1), (0, 1)],  # 3D unit cube
    spatial_discretization=[32, 32, 32],       # 32³ grid points
    T=1.0,
    Nt=50,
    sigma=0.1,
)

# Solve with factory
solver = create_basic_solver(problem, damping=0.6, max_iterations=30)
result = solver.solve()
```

### **High-Performance nD Problems**
```python
from mfg_pde import MFGProblem

# Any-dimensional problem via spatial_bounds
# Example: 4D problem
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],  # 4D hypercube
    spatial_discretization=[16, 16, 16, 16],          # 16^4 grid
    T=1.0,
    Nt=50,
    sigma=0.05,
)
```

## 🧮 **Mathematical Framework**

### **High-Dimensional HJB-FP System**
The framework solves the coupled system in d dimensions:

**Hamilton-Jacobi-Bellman (HJB)**:
```
-∂u/∂t + H(x, ∇u, m, t) = 0    in Ω × (0,T)
u(x,T) = g(x)                   in Ω
```

**Fokker-Planck (FP)**:
```
∂m/∂t - div(m ∇_p H) - σ²/2 Δm = 0    in Ω × (0,T)
m(x,0) = m₀(x)                         in Ω
```

### **Supported Hamiltonians**
- **Quadratic**: `H = ½|p|² + V(x) + γm|p|²`
- **Anisotropic**: `H = ½p^T A(x) p + f(x,m,t)`
- **Obstacle**: `H = max(½|p|² - c(x), ψ(x))`
- **Custom**: User-defined through `MFGComponents`

## 🔧 **Solver Architecture**

### **Damped Fixed Point (Recommended for Testing)**
- **Current Implementation**: `FixedPointIterator` (aliased as `DampedFixedPointIterator`)
- **Multi-dimensional**: ✅ Already supports arbitrary dimensions
- **Damping Parameter**: `damping_factor` for stability control
- **Best for**: Initial testing, stable problems, rapid prototyping

### **Particle Collocation**
- **Implementation**: `ParticleCollocationSolver` and variants
- **Multi-dimensional**: ✅ Native support via `(N_points, d)` collocation arrays
- **Best for**: Complex geometries, high accuracy requirements
- **Adaptive**: `AdaptiveParticleCollocationSolver` for dynamic refinement

### **Hybrid Multi-Strategy**
- **Phase 1**: Damped fixed point for rapid initial convergence
- **Phase 2**: Particle collocation for accuracy refinement
- **Phase 3**: Adaptive strategy based on convergence behavior
- **Best for**: Production-quality results, unknown problem difficulty

## 📊 **Performance Characteristics**

### **Computational Complexity**
| Dimension | Grid Points | Damped FP | Particle Collocation | Memory Usage |
|-----------|-------------|-----------|---------------------|--------------|
| 2D        | 32×32       | ~0.5s     | ~2s                 | ~50MB        |
| 2D        | 64×64       | ~2s       | ~8s                 | ~200MB       |
| 3D        | 16×16×16    | ~3s       | ~15s                | ~300MB       |
| 3D        | 32×32×32    | ~25s      | ~120s               | ~2GB         |

### **Scaling Recommendations**
- **2D Problems**: Up to 128×128 grids practical
- **3D Problems**: Up to 64×64×64 grids on modern hardware
- **Complex Geometry**: Prefer particle collocation for irregular domains
- **Memory Optimization**: Use sparse storage for large 3D problems

## 🎨 **Visualization Features**

### **2D Visualization**
- **Matplotlib**: Contour plots, streamlines, vector fields
- **Triangle-based**: Proper interpolation on irregular meshes
- **Animation**: Time evolution of density and value functions

### **3D Visualization**
- **PyVista**: Interactive 3D visualization with VTK backend
- **Volume Rendering**: Density and value function 3D rendering
- **Slice Plots**: Cross-sectional analysis
- **Mesh Quality**: Element quality visualization

## 🧪 **Validation and Testing**

### **Convergence Studies**
- Grid refinement analysis for spatial convergence
- Time step refinement for temporal accuracy
- Comparison between solver methods

### **Analytical Benchmarks**
- 2D Gaussian density evolution
- 3D sphere packing problems
- Linear-quadratic MFG with known solutions

### **Performance Benchmarks**
- Scaling analysis across dimensions
- Memory usage profiling
- Solver comparison (fixed point vs particle collocation)

## 🔮 **Advanced Features**

### **Adaptive Mesh Refinement (AMR)**
- **2D**: Triangular AMR fully implemented
- **3D**: Framework ready, needs tetrahedral AMR completion
- **Error Estimation**: Gradient-based and residual-based refinement

### **Complex Boundary Conditions**
- **Dirichlet**: Fixed values on boundaries
- **Neumann**: Fixed flux conditions
- **Robin**: Mixed boundary conditions
- **Periodic**: For torus-like domains

### **GPU Acceleration (Future)**
- JAX backend integration ready
- CUDA support through PyTorch compatibility
- Memory-efficient sparse operations

## 📝 **Usage Guidelines**

### **For Quick Testing**
1. Use `GridBasedMFGProblem` for simple rectangular domains
2. Start with damped fixed point solver (fastest)
3. Use moderate grid sizes (32×32 for 2D, 16×16×16 for 3D)

### **For Production Quality**
1. Use `HighDimMFGProblem` with custom geometry
2. Apply `HybridMFGSolver` for robust convergence
3. Validate with convergence studies

### **For Research**
1. Implement custom Hamiltonians via `MFGComponents`
2. Use particle collocation for maximum accuracy
3. Apply AMR for complex solution features

## 🚨 **Known Limitations and Future Work**

### **Current Limitations**
- **3D AMR**: Tetrahedral adaptive refinement not yet implemented
- **Memory Usage**: Large 3D problems require optimization
- **GPU Support**: Currently CPU-only (JAX integration planned)

### **Planned Extensions**
- **4D+ Support**: Extension to very high dimensions
- **Parallel Solvers**: MPI-based parallel implementation
- **Machine Learning**: Neural network Hamiltonian approximation
- **Stochastic MFG**: Noise-driven systems

## 📖 **Further Reading**

- **Mathematical Theory**: Cardaliaguet et al. "Notes on Mean Field Games"
- **Computational Methods**: Achdou & Capuzzo-Dolcetta "Mean Field Games: Numerical Methods"
- **Implementation Details**: See `docs/theory/` for mathematical formulations
- **API Reference**: See `docs/api/` for complete class documentation

---

**Package Integration**: These capabilities are fully integrated with the existing MFG_PDE infrastructure, maintaining backward compatibility while providing powerful extensions for high-dimensional applications.
