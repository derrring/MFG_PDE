# MFG_PDE Examples

This directory provides guidance for using MFG_PDE examples and demonstrations.

## 📁 Example Locations

Examples are organized by complexity in the main repository:

- **[Basic Examples](../../examples/basic/)** - Simple demonstrations and getting started
- **[Advanced Examples](../../examples/advanced/)** - Complex workflows and research tools  
- **[Notebooks](../../examples/notebooks/)** - Interactive Jupyter demonstrations

## 🚀 Quick Start Patterns

### Modern Factory Pattern (Recommended)
```python
from mfg_pde import ExampleMFGProblem, create_fast_solver

# Create problem
problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=1.0, Nt=30)

# Create solver with intelligent defaults
solver = create_fast_solver(problem, solver_type="adaptive_particle")
result = solver.solve()

print(f"Converged: {result.convergence_achieved}")
print(f"Execution time: {result.execution_time:.2f}s")
```

### Network MFG (NEW 2025)
```python
from mfg_pde import create_grid_mfg_problem, create_fast_solver

# Create network MFG problem
problem = create_grid_mfg_problem(10, 10, T=1.0, Nt=50)
solver = create_fast_solver(problem, backend="auto")
result = solver.solve()

print(f"Network nodes: {problem.num_nodes}")
```

### GPU Acceleration (JAX Backend)
```python
from mfg_pde import ExampleMFGProblem, create_fast_solver
from mfg_pde.backends import create_backend

problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=100, T=1.0, Nt=50)

# Use JAX backend for GPU acceleration
jax_backend = create_backend("jax", device="gpu", jit_compile=True)
solver = create_fast_solver(problem, backend=jax_backend)
result = solver.solve()
```

## 📊 Available Examples

### Basic Examples
- **[El Farol Bar](../../examples/basic/el_farol_simple_working.py)** - Classic MFG application
- **[Towel Beach Problem](../../examples/basic/towel_beach_problem.py)** - Spatial competition model
- **[Mesh Pipeline Demo](../../examples/basic/mesh_pipeline_demo.py)** - Geometry system demonstration

### Advanced Examples  
- **[Network MFG Comparison](../../examples/advanced/network_mfg_comparison.py)** - Backend performance comparison
- **[JAX Acceleration Demo](../../examples/advanced/jax_acceleration_demo.py)** - GPU benchmarking
- **[Enhanced Network Visualization](../../examples/advanced/enhanced_network_visualization_demo.py)** - 3D network plots
- **[Interactive Research Notebook](../../examples/advanced/interactive_research_notebook_example.py)** - Professional workflow

### Notebooks
- **[Working Demo](../../examples/notebooks/working_demo/)** - Interactive Jupyter demonstration

## 🎯 Performance Expectations

### Traditional MFG
- **Mass Conservation**: <0.1% error across all solver types
- **GPU Speedup**: 10-100× with JAX backend on compatible hardware
- **CPU Performance**: 3-8× faster with modern optimizations
- **Memory Usage**: <1GB for standard problems (Nx=100, Nt=100)

### Network MFG
- **Backend Performance**: 10-50× speedup with igraph/networkit vs networkx
- **Scalability**: Linear scaling up to 10,000+ nodes
- **Convergence**: Typically 5-15 iterations for grid networks

## 🔧 Troubleshooting

### Common Issues
- **Installation**: Run `pip install -e .` from repository root
- **GPU Issues**: Check JAX installation with `from jax import numpy as jnp`
- **Network Backend**: Install optional dependencies: `pip install igraph networkit`

### Performance Tips
- Use `backend="auto"` for automatic optimal backend selection
- Start with basic examples before advanced workflows
- Monitor memory usage with built-in performance tracking

## 📚 Documentation Links

- **[User Documentation](../user/)** - Complete tutorials and guides
- **[Theory Documentation](../theory/)** - Mathematical background
- **[API Reference](../api/)** - Function and class documentation
- **[Development Documentation](../development/)** - Technical implementation details

---

**Note**: All examples use modern factory patterns and current API conventions. For legacy code patterns, see archived documentation.
