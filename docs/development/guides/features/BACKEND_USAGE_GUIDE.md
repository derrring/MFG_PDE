# MFG_PDE Backend Usage Guide

## 🚀 **Quick Start: Choose Your Backend**

MFG_PDE automatically selects the best available backend, but you can also manually choose based on your hardware and problem type.

### **One-Line Backend Selection**
```python
from mfg_pde.backends import create_backend

# Automatic optimal selection
backend = create_backend("auto")

# Manual selection
backend = create_backend("torch_mps")  # Apple Silicon
backend = create_backend("jax_gpu")    # NVIDIA GPU with JAX
backend = create_backend("numba")      # CPU optimization
```

## 🎯 **Backend Recommendations by Use Case**

### **🧠 Neural Methods (PINN, Neural Operators)**
```python
# BEST: PyTorch with automatic device selection
from mfg_pde.backends import create_backend
from mfg_pde.alg.neural_solvers import PINNSolver

backend = create_backend("torch")  # Auto-selects CUDA/MPS/CPU
solver = PINNSolver(problem, backend=backend)
result = solver.solve()
```
**Why PyTorch**: Native neural network support, automatic differentiation, optimal tensor operations.

### **📊 Mathematical Kernels (Pure Computation)**
```python
# BEST: JAX for mathematical operations
from mfg_pde.utils.acceleration.jax_utils import compute_hamiltonian
from mfg_pde.alg.mfg_solvers import JAXMFGSolver

solver = JAXMFGSolver(problem, use_gpu=True, jit_compile=True)
result = solver.solve()  # Automatic XLA compilation
```
**Why JAX**: Pure functional code, automatic vectorization, excellent for gradients.

### **🌳 Adaptive Mesh Refinement (AMR)**
```python
# BEST: Numba for imperative algorithms
from mfg_pde.geometry import TriangularAMR
from mfg_pde.utils.performance_optimization import create_performance_optimizer

amr = TriangularAMR(use_numba=True)  # Numba-accelerated refinement
optimizer = create_performance_optimizer()
backend = optimizer["acceleration_backend"]
backend.set_backend("numba")
```
**Why Numba**: Excellent for loops, conditionals, tree traversal algorithms.

### **💾 Large-Scale Problems (Memory Constrained)**
```python
# BEST: Sparse operations with Numba parallelization
from mfg_pde.utils.performance_optimization import SparseMatrixOptimizer

optimizer = SparseMatrixOptimizer()
# Automatically uses Numba if available for parallel operations
laplacian = optimizer.create_laplacian_3d(nx=200, ny=200, nz=200)
solution = optimizer.solve_sparse_system(laplacian, rhs, method="cg")
```

## 🔧 **Backend Configuration Examples**

### **PyTorch Backend Configuration**
```python
from mfg_pde.backends import TorchBackend

# Apple Silicon (M1/M2/M3)
backend = TorchBackend(
    device="mps",              # Metal Performance Shaders
    precision="float32",       # Optimal for MPS
    memory_efficient=True      # Reduce memory fragmentation
)

# NVIDIA CUDA
backend = TorchBackend(
    device="cuda",
    precision="float16",       # Mixed precision for speed
    memory_efficient=False     # More memory for speed
)

# CPU (development/debugging)
backend = TorchBackend(
    device="cpu",
    precision="float64",       # High precision for debugging
    num_threads=4              # Control CPU threading
)
```

### **JAX Backend Configuration**
```python
from mfg_pde.alg.mfg_solvers import JAXMFGSolver

# GPU configuration
solver = JAXMFGSolver(
    problem,
    use_gpu=True,              # Use GPU if available
    jit_compile=True,          # JIT compilation for speed
    finite_diff_order=4,       # Higher-order accuracy
    adaptive_timestep=True     # Dynamic time stepping
)

# CPU configuration (high precision)
solver = JAXMFGSolver(
    problem,
    use_gpu=False,
    jit_compile=True,
    finite_diff_order=6,       # Maximum accuracy
    method="newton"            # Advanced solver method
)
```

### **Numba Optimization Setup**
```python
from mfg_pde.utils.performance_optimization import AccelerationBackend

backend = AccelerationBackend()
backend.set_backend("numba")

# Optimize a custom function
@backend.optimize_function  # Automatically applies numba.jit
def custom_finite_difference(u, dx):
    # Your imperative algorithm here
    result = np.zeros_like(u)
    for i in range(1, len(u)-1):
        result[i] = (u[i+1] - 2*u[i] + u[i-1]) / (dx*dx)
    return result
```

## 🏗️ **Hardware-Specific Optimization**

### **Apple Silicon (M1/M2/M3) Setup**
```python
# Optimal configuration for Mac
import mfg_pde

# Check what's available
from mfg_pde.backends import get_available_backends
available = get_available_backends()
print("Available backends:", available)
# Expected: {'torch_mps': True, 'jax': True, 'numba': True}

# Automatic optimal selection for Mac
backend = create_backend("auto")  # Usually selects torch_mps
print(f"Selected: {backend.name}")  # "torch_mps"
```

### **NVIDIA CUDA Setup**
```python
# Optimal for CUDA systems
backend = create_backend("auto")  # Prefers CUDA if available

# Manual CUDA configuration
if available.get("torch_cuda", False):
    torch_backend = create_backend("torch_cuda")

if available.get("jax_gpu", False):
    jax_backend = create_backend("jax_gpu")

# Use both for different problem components
neural_solver = PINNSolver(problem, backend=torch_backend)
kernel_solver = JAXMFGSolver(problem, use_gpu=True)
```

### **CPU-Only Optimization**
```python
# Optimal for CPU-only systems
from mfg_pde.utils.performance_optimization import ParallelizationHelper

# Use all available CPU cores
parallelization = ParallelizationHelper()
optimal_chunks = parallelization.estimate_optimal_chunks(
    total_size=problem.spatial_points,
    memory_limit_mb=1000  # Adjust based on available RAM
)

# Configure Numba for multiprocessing
backend = create_backend("numba")
# Numba automatically uses multiple cores for suitable operations
```

## 📊 **Performance Monitoring**

### **Built-in Performance Analysis**
```python
from mfg_pde.utils.performance_optimization import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.monitor_operation("backend_solve", backend_type="torch_mps"):
    result = solver.solve()

# Get detailed performance metrics
summary = monitor.get_summary()
print(f"Execution time: {summary['total_time']:.2f}s")
print(f"Peak memory: {summary['peak_memory']:.1f}MB")
```

### **Backend Benchmarking**
```python
# Compare all available backends
from mfg_pde.utils.performance_optimization import optimize_mfg_problem_performance

performance_analysis = optimize_mfg_problem_performance({
    "spatial_points": 10000,
    "time_steps": 100,
    "dimension": 2
})

print("Recommendations:")
for rec in performance_analysis["recommendations"]:
    print(f"  • {rec}")
```

## 🔄 **Migration from Old Structure (Phase 2+)**

### **Updating Existing Code**
If you have existing code using the old `mfg_pde.accelerated` imports:

```python
# OLD (still works with deprecation warnings)
from mfg_pde.accelerated import JAXMFGSolver
from mfg_pde.accelerated.jax_utils import compute_hamiltonian

# NEW (recommended)
from mfg_pde.alg.mfg_solvers import JAXMFGSolver
from mfg_pde.utils.acceleration.jax_utils import compute_hamiltonian
```

### **Suppress Deprecation Warnings (Temporarily)**
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Your existing code continues to work unchanged
from mfg_pde.accelerated import JAXMFGSolver
```

### **Migration Script Template**
```python
# find_and_replace.py - Update your codebase
import os
import re

def migrate_imports(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Update imports
    replacements = {
        "from mfg_pde.accelerated import JAXMFGSolver":
            "from mfg_pde.alg.mfg_solvers import JAXMFGSolver",
        "from mfg_pde.accelerated.jax_utils import":
            "from mfg_pde.utils.acceleration.jax_utils import"
    }

    for old, new in replacements.items():
        content = re.sub(old, new, content)

    with open(file_path, 'w') as f:
        f.write(content)

# Apply to your Python files
for root, dirs, files in os.walk("your_project/"):
    for file in files:
        if file.endswith(".py"):
            migrate_imports(os.path.join(root, file))
```

## 🎯 **Common Usage Patterns**

### **Automatic Backend Selection (Recommended)**
```python
# Let MFG_PDE choose the best backend
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_solver

problem = ExampleMFGProblem(Nx=100, Nt=50)
solver = create_solver(problem, method="auto", backend="auto")
result = solver.solve()
```

### **Mixed Backend Strategy**
```python
# Use different backends for different components
torch_backend = create_backend("torch_mps")  # For neural components
jax_backend = create_backend("jax")          # For pure math
numba_backend = create_backend("numba")      # For AMR

# Neural solver for learning components
neural_solver = PINNSolver(problem, backend=torch_backend)

# Mathematical solver for PDE kernels
math_solver = JAXMFGSolver(problem, use_gpu=True)

# AMR for mesh adaptation
amr = TriangularAMR(use_numba=True)
```

### **Development vs Production**
```python
# Development: Maximum compatibility and debugging
if os.getenv("MFG_DEVELOPMENT", "false").lower() == "true":
    backend = create_backend("numpy")  # Predictable, debuggable
    precision = "float64"              # High precision
else:
    # Production: Maximum performance
    backend = create_backend("auto")   # Optimal hardware utilization
    precision = "float32"              # Speed over precision
```

## 🔍 **Debugging Backend Issues**

### **Backend Detection Problems**
```python
from mfg_pde.backends import get_available_backends

available = get_available_backends()
print("Backend availability:")
for name, status in available.items():
    print(f"  {name}: {'✓' if status else '✗'}")

# Force backend for testing
try:
    backend = create_backend("torch_mps")
    print("✓ torch_mps working")
except Exception as e:
    print(f"✗ torch_mps failed: {e}")
```

### **Performance Debugging**
```python
# Enable verbose performance logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check GPU memory usage
if backend.name.endswith("gpu"):
    print(f"GPU memory: {backend.get_memory_usage()}")

# Profile specific operations
from mfg_pde.utils.performance_optimization import profile_jax_function

if backend.name.startswith("jax"):
    metrics = profile_jax_function(compute_hamiltonian, u_grad, density)
    print(f"Compile time: {metrics['compile_time']:.3f}s")
    print(f"Execute time: {metrics['mean_exec_time']:.3f}s")
```

## 📚 **Additional Resources**

### **Backend-Specific Documentation**
- **PyTorch**: See `docs/development/PYTORCH_PINN_INTEGRATION_ANALYSIS.md`
- **JAX**: See `mfg_pde/utils/acceleration/jax_utils.py` docstrings
- **Numba**: See `mfg_pde/utils/performance_optimization.py` examples

### **Hardware Setup Guides**
- **Apple Silicon**: Ensure Metal Performance Shaders enabled
- **NVIDIA CUDA**: Install appropriate CUDA toolkit version
- **CPU Optimization**: Configure OpenMP threads and BLAS libraries

### **Performance Benchmarks**
```python
# Run comprehensive benchmarks
python -m mfg_pde.benchmarks.backend_comparison --all-backends --problem-sizes large
```

---

**Quick Reference:**
- 🧠 **Neural methods** → `torch` backend
- 📊 **Pure math** → `jax` backend
- 🌳 **AMR/loops** → `numba` backend
- 💾 **Large problems** → `sparse` + `numba`
- 🔄 **Unsure** → `"auto"` selection
