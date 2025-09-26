# MFG_PDE Acceleration Architecture Complete Guide

## 🏗️ **Unified Backend System Overview**

MFG_PDE implements a sophisticated multi-backend acceleration system optimized for different computational patterns in Mean Field Games. This document provides the complete technical architecture after Phase 1 (PyTorch backend) and Phase 2 (JAX reorganization) completion.

## 🎯 **Backend Priority Hierarchy**

The system automatically selects the optimal backend based on hardware availability and computational requirements:

```
Hardware Detection Priority:
CUDA > Apple Silicon MPS > JAX GPU > Numba CPU > NumPy
```

### **Backend Roles & Specializations**

| Backend | Primary Role | Optimal Use Cases | Hardware Target |
|---------|--------------|-------------------|-----------------|
| **PyTorch** | Deep learning, tensor ops | PINN solvers, neural methods | CUDA, MPS, CPU |
| **JAX** | Pure functional, differentiable | Mathematical kernels, gradients | GPU, TPU, CPU |
| **Numba** | Imperative optimization | AMR, tight loops, conditionals | CPU (multi-core) |
| **NumPy** | Baseline reference | Development, compatibility | CPU (single-core) |

## 🚀 **PyTorch Backend (Phase 1 Complete)**

### **Implementation Details**
- **Location**: `mfg_pde/backends/torch_backend.py` (588 lines)
- **Apple Silicon Support**: Full Metal Performance Shaders (MPS) integration
- **Auto-device Selection**: CUDA > MPS > CPU with intelligent fallback
- **Neural Solver Integration**: Enhanced PINN base class with MPS detection

### **Key Features**
```python
# Automatic device selection with Apple Silicon support
def _select_device(self, device_spec: str) -> torch_device:
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch_device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch_device("mps")  # Apple Silicon acceleration
        else:
            return torch_device("cpu")
```

### **Performance Characteristics**
- **Memory Management**: Efficient tensor operations with automatic memory optimization
- **Mixed Precision**: Support for float16/float32 based on hardware capabilities
- **Batch Processing**: Vectorized operations for multiple problem instances

## 🔄 **JAX Backend (Phase 2 Reorganized)**

### **New Directory Structure** ✅
```
mfg_pde/
├── utils/acceleration/           # NEW: Centralized acceleration utilities
│   ├── __init__.py              # Acceleration detection and management
│   └── jax_utils.py             # Pure JAX numerical utilities
├── alg/mfg_solvers/             # ENHANCED: Complete solver collection
│   ├── jax_mfg_solver.py        # MOVED: JAX MFG solver in proper location
│   └── __init__.py              # Conditional JAX solver imports
└── accelerated/                 # DEPRECATED: Backward compatibility only
    ├── __init__.py              # Deprecation warnings and redirects
    ├── jax_mfg_solver.py        # Backward compatibility wrapper
    └── jax_utils.py             # Backward compatibility wrapper
```

### **Migration Guide** 📋
```python
# NEW (Phase 2+)
from mfg_pde.utils.acceleration.jax_utils import compute_hamiltonian
from mfg_pde.alg.mfg_solvers import JAXMFGSolver

# OLD (still works with deprecation warnings)
from mfg_pde.accelerated.jax_utils import compute_hamiltonian
from mfg_pde.accelerated import JAXMFGSolver
```

### **JAX Specializations**
- **Pure Functional Kernels**: Automatic differentiation for sensitivity analysis
- **JIT Compilation**: Hardware-optimized code generation
- **Vectorization**: SIMD operations for batch processing
- **XLA Integration**: Cross-platform accelerated linear algebra

## ⚡ **Numba Backend (Complementary Layer)**

### **Strategic Role**
Numba serves as our **"imperative code accelerator"** - handling computational patterns that don't fit JAX's functional paradigm:

### **Numba Use Cases**
1. **Adaptive Mesh Refinement (AMR)**
   ```python
   @numba.jit(nopython=True, cache=True)
   def refine_mesh_imperative(mesh_data, error_indicators):
       # Tree traversal with complex conditionals
       # Optimal for Numba's imperative optimization
   ```

2. **Parallel Sparse Operations**
   ```python
   def parallel_matvec(matrix, vectors):
       if NUMBA_AVAILABLE:
           return _numba_parallel_matvec(matrix, vectors)  # Multi-core CPU
       else:
           return np.array([matrix @ v for v in vectors])  # Fallback
   ```

3. **Performance-Critical Loops**
   - Finite difference stencils with boundary conditions
   - Time-stepping with adaptive step size
   - Mesh connectivity algorithms

### **Numba Integration Pattern**
```python
# Location: mfg_pde/utils/performance_optimization.py
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class AccelerationBackend:
    def optimize_function(self, func: Callable, backend: str) -> Callable:
        if backend == "numba" and NUMBA_AVAILABLE:
            return numba.jit(func, nopython=True)
        elif backend.startswith("jax") and JAX_AVAILABLE:
            return jax.jit(func)
        else:
            return func  # Unoptimized fallback
```

### **Hybrid JAX + Numba Architecture**
```python
# JAX: Pure functional computations (gradients, vectorized math)
@jax.jit
def compute_error_indicators_jax(U, M, dx, dy):
    # Automatic differentiation for error estimation
    return jnp.sqrt(jnp.sum((jnp.gradient(U, dx, dy))**2))

# Numba: Imperative bottlenecks (tree traversal, conditionals)
@numba.jit(nopython=True, cache=True)
def refine_mesh_numba(mesh_indices, error_flags):
    # Fast imperative mesh refinement with complex logic
    for i in range(len(mesh_indices)):
        if error_flags[i] > threshold:
            # Complex conditional refinement logic
            pass
```

## 🔧 **Backend Factory System**

### **Unified Backend Creation**
```python
from mfg_pde.backends import create_backend, get_available_backends

# Automatic optimal backend selection
backend = create_backend("auto")
print(f"Selected backend: {backend.name}")  # torch_mps, jax_gpu, etc.

# Available backends detection
available = get_available_backends()
# {'numpy': True, 'torch': True, 'torch_cuda': False, 'torch_mps': True,
#  'jax': True, 'jax_gpu': False, 'numba': True}
```

### **Backend Registry Architecture**
```python
class BackendRegistry:
    _backends = {
        "torch": TorchBackend,
        "torch_cuda": partial(TorchBackend, device="cuda"),
        "torch_mps": partial(TorchBackend, device="mps"),
        "jax": JAXBackend,
        "jax_gpu": partial(JAXBackend, device="gpu"),
        "numba": NumbaBackend,
        "numpy": NumpyBackend,
    }
```

## 📊 **Performance Characteristics**

### **Benchmark Results** (Representative)
| Problem Size | NumPy | Numba | JAX CPU | JAX GPU | PyTorch MPS |
|--------------|-------|-------|---------|---------|-------------|
| Small (20×10) | 0.1s | 0.05s | 0.08s | 0.12s | 0.09s |
| Medium (100×50) | 2.5s | 0.8s | 0.6s | 0.3s | 0.4s |
| Large (500×250) | 45s | 12s | 8s | 2.1s | 3.2s |

### **Hardware-Specific Optimizations**

**Apple Silicon (M1/M2/M3):**
- PyTorch MPS backend for tensor operations
- JAX CPU with optimized BLAS on ARM
- Numba with multiprocessing for CPU-bound tasks

**NVIDIA CUDA:**
- PyTorch CUDA for deep learning components
- JAX GPU for mathematical kernels
- Automatic mixed precision where available

**CPU-Only Systems:**
- Numba for tight loops and imperative code
- JAX CPU for functional mathematical operations
- NumPy as universal fallback

## 🛡️ **Backward Compatibility Strategy**

### **Deprecation Management**
Phase 2 reorganization maintains **100% backward compatibility**:

```python
# All old import paths continue working with warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mfg_pde.accelerated import JAXMFGSolver  # Still works
from mfg_pde.accelerated.jax_utils import compute_hamiltonian  # Still works
```

### **Migration Timeline**
- **Phase 2**: New structure available, old paths deprecated
- **Future Phase**: Remove deprecated paths after adoption
- **Gradual Migration**: Users can migrate at their own pace

## 🔄 **Integration with MFG Solvers**

### **Solver-Backend Mapping**
```python
# Solver factory automatically selects optimal backend
from mfg_pde.factory import create_solver

solver = create_solver(
    problem=mfg_problem,
    method="particle_collocation",
    backend="auto"  # Selects torch_mps, jax_gpu, etc.
)
```

### **Backend-Specific Optimizations**
- **PyTorch Solvers**: Neural network components, automatic differentiation
- **JAX Solvers**: Functional kernels, vectorized operations
- **Numba Solvers**: Imperative algorithms, mesh adaptation
- **Hybrid Approaches**: Combine backends for optimal performance

## 📈 **Performance Monitoring**

### **Built-in Profiling**
```python
from mfg_pde.utils.performance_optimization import PerformanceMonitor

monitor = PerformanceMonitor()
with monitor.monitor_operation("mfg_solve"):
    result = solver.solve()

print(monitor.get_summary())
# Shows backend selection, timing, memory usage
```

### **Backend Benchmarking**
```python
from mfg_pde.backends import benchmark_backends

results = benchmark_backends(problem_sizes=[(100, 50), (200, 100)])
# Automatic comparison across all available backends
```

## 🚀 **Future Extensions**

### **Potential Phase 3+ Enhancements**
1. **Intel oneAPI Integration**: DPC++ backend for Intel hardware
2. **AMD ROCm Support**: HIP backend for AMD GPUs
3. **Distributed Computing**: MPI backend for cluster computing
4. **WebGL Backend**: Browser-based visualization and computation
5. **Quantum Computing**: Qiskit integration for hybrid algorithms

### **Architecture Extensibility**
The backend system is designed for easy extension:
```python
class NewAcceleratorBackend(BaseBackend):
    def solve_hjb(self, problem):
        # Custom implementation
        pass

    def solve_fp(self, problem):
        # Custom implementation
        pass
```

## 📋 **Development Guidelines**

### **Adding New Backends**
1. Inherit from `BaseBackend`
2. Implement required interface methods
3. Add detection logic to `get_available_backends()`
4. Register in backend factory
5. Add comprehensive tests

### **Backend Selection Logic**
```python
def select_optimal_backend(problem_characteristics):
    if problem_characteristics.has_neural_components:
        return "torch_auto"  # PyTorch for neural methods
    elif problem_characteristics.is_purely_functional:
        return "jax_auto"    # JAX for mathematical kernels
    elif problem_characteristics.has_imperative_logic:
        return "numba"       # Numba for conditional algorithms
    else:
        return "numpy"       # Safe fallback
```

## 📚 **References and Documentation**

### **Key Files**
- `mfg_pde/backends/torch_backend.py`: Complete PyTorch implementation
- `mfg_pde/utils/acceleration/`: JAX utilities and detection
- `mfg_pde/alg/mfg_solvers/jax_mfg_solver.py`: JAX solver implementation
- `mfg_pde/utils/performance_optimization.py`: Numba integration
- `examples/advanced/jax_numba_hybrid_performance.py`: Hybrid architecture demo

### **Performance Documentation**
- Backend selection benchmarks in `benchmarks/backend_comparison/`
- Hardware-specific optimization guides in `docs/hardware/`
- Profiling and optimization tutorials in `docs/performance/`

---

**Status**: ✅ **COMPLETE** - Comprehensive multi-backend acceleration system
**Last Updated**: Phase 2 completion (2025-01-XX)
**Next Phase**: Enhanced documentation and optional cleanup (Phase 3)
