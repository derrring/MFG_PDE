# High-Dimensional MFG Capabilities

This directory demonstrates the extended high-dimensional capabilities of the MFG_PDE package, including 2D, 3D, and nD Mean Field Games with advanced solver methods.

## 🎯 **Key Features Implemented**

### **New High-Dimensional Infrastructure**
- **`Domain3D`**: Complete 3D geometry with tetrahedral mesh generation
- **`HighDimMFGProblem`**: Abstract base for multi-dimensional MFG problems
- **`GridBasedMFGProblem`**: Simplified interface for rectangular domains
- **`HybridMFGSolver`**: Multi-strategy solver combining damped fixed point and particle collocation

### **Solver Capabilities**
- **Damped Fixed Point**: Already multi-dimensional ready (using existing `FixedPointIterator`)
- **Particle Collocation**: Native support for arbitrary dimensions
- **Hybrid Methods**: Adaptive strategy switching for robust convergence
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

### **2D Complex Geometry MFG**
```python
from mfg_pde.geometry import Domain2D
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem

# Create 2D domain with holes
geometry = Domain2D(
    domain_type="rectangle",
    bounds=(0, 2, 0, 1),
    holes=[
        {"type": "circle", "center": (0.5, 0.5), "radius": 0.2},
        {"type": "circle", "center": (1.5, 0.5), "radius": 0.15}
    ]
)

# Solve MFG problem
problem = CustomMFG2D(geometry, time_domain=(1.0, 100))
result = problem.solve_with_damped_fixed_point()
```

### **3D Crowd Dynamics**
```python
from mfg_pde.geometry import Domain3D
from mfg_pde.core.highdim_mfg_problem import HybridMFGSolver

# Create 3D sphere domain
geometry = Domain3D(
    domain_type="sphere",
    center=(0, 0, 0),
    radius=1.0,
    mesh_size=0.1
)

# Hybrid solver approach
problem = CrowdDynamics3D(geometry)
solver = HybridMFGSolver(problem)
result = solver.solve(strategy="adaptive")
```

### **High-Performance Grid-Based**
```python
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem

# 3D box domain with regular grid
problem = GridBasedMFGProblem(
    domain_bounds=(0, 1, 0, 1, 0, 1),  # 3D unit cube
    grid_resolution=(32, 32, 32),      # 32³ grid points
    time_domain=(1.0, 50)
)

# Fast solution with optimized damping
result = problem.solve_with_damped_fixed_point(
    damping_factor=0.6,
    max_iterations=30
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
- **Damping Parameter**: `thetaUM` for stability control
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
