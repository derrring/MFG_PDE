# Phase 3: Tiered Backend Strategy - torch > jax > numpy

**Date**: 2025-10-04
**Status**: Implementation Plan Approved
**Architecture**: Pluggable backend system with intelligent auto-selection

---

## Executive Summary

Implement a **tiered backend system** where the best available acceleration framework is automatically selected based on user's installation, while maintaining universal compatibility.

**Selection Priority**: `torch > jax > numpy`

**Key Insight**: This is **not** about mixing frameworks—each computation runs on a **single, consistent backend** chosen at instantiation time.

---

## User Personas & Use Cases

### Persona 1: RL Researcher 🎯
**Profile**: Uses MFG for RL algorithms, already has PyTorch installed

**Experience**:
```python
from mfg_pde.alg.rl import ActorCriticSolver
from mfg_pde.alg.numerical import HJBFDMSolver

# Both auto-select PyTorch (same GPU, same device)
actor_critic = ActorCriticSolver(problem)  # backend="torch" (auto)
hjb = HJBFDMSolver(problem)  # backend="torch" (auto)

# Shared GPU memory, zero overhead
result = actor_critic.solve()
value = hjb.solve()
```

**Benefits**:
- ✅ Seamless integration (RL + numerical on same backend)
- ✅ Shared GPU memory pool
- ✅ Consistent device management
- ✅ No PyTorch/JAX data transfer overhead

### Persona 2: Scientific Computing User 🔬
**Profile**: Uses MFG for PDE solving, prefers JAX, no RL needed

**Experience**:
```python
from mfg_pde.alg.numerical import HJBFDMSolver, FPParticleSolver

# Auto-selects JAX (PyTorch not installed)
hjb = HJBFDMSolver(problem)  # backend="jax" (auto)
fp = FPParticleSolver(problem)  # backend="jax" (auto)

# JIT-compiled, GPU-accelerated (if available)
result = hjb.solve()
```

**Benefits**:
- ✅ No PyTorch dependency overhead
- ✅ JAX's superior JIT compilation for PDEs
- ✅ Functional programming style (NumPy-like)
- ✅ Automatic CPU/GPU selection

### Persona 3: Pedagogical User 📚
**Profile**: Learning MFG, no acceleration needed

**Experience**:
```python
from mfg_pde.alg.numerical import HJBFDMSolver

# Auto-selects NumPy (neither torch nor jax installed)
hjb = HJBFDMSolver(problem)  # backend="numpy" (auto)

# Works everywhere, minimal dependencies
result = hjb.solve()
```

**Benefits**:
- ✅ Universal compatibility
- ✅ Minimal installation
- ✅ Easy debugging (pure Python)
- ✅ Educational clarity

---

## Architecture Design

### Backend Factory Pattern

```python
# mfg_pde/backends/__init__.py

from typing import Optional
import logging

logger = logging.getLogger(__name__)

def create_backend(backend_type: Optional[str] = None) -> BaseBackend:
    """
    Create backend with intelligent auto-selection.

    Priority: torch > jax > numpy

    Args:
        backend_type: Explicit backend choice ("torch", "jax", "numpy")
                     If None, auto-selects best available

    Returns:
        Backend instance

    Example:
        >>> backend = create_backend()  # Auto-selects
        >>> print(backend.name)  # "torch" if available

        >>> backend = create_backend("jax")  # Force JAX
    """
    if backend_type is None:
        # Auto-selection: torch > jax > numpy
        if _torch_available():
            backend_type = "torch"
            logger.info("Auto-selected PyTorch backend (RL infrastructure detected)")
        elif _jax_available():
            backend_type = "jax"
            logger.info("Auto-selected JAX backend (PyTorch not available)")
        else:
            backend_type = "numpy"
            logger.info("Using NumPy backend (no acceleration available)")

    # Create backend instance
    if backend_type == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend()
    elif backend_type == "jax":
        from .jax_backend import JAXBackend
        return JAXBackend()
    elif backend_type == "numpy":
        from .numpy_backend import NumpyBackend
        return NumpyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


def _torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _jax_available() -> bool:
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        return False
```

### Abstract Base Backend

```python
# mfg_pde/backends/base_backend.py

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

class BaseBackend(ABC):
    """Abstract base class for computational backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name: 'torch', 'jax', or 'numpy'."""
        pass

    @property
    @abstractmethod
    def array_module(self) -> Any:
        """Array module (torch, jax.numpy, or numpy)."""
        pass

    # Core array operations
    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create array of zeros."""
        pass

    @abstractmethod
    def ones(self, shape, dtype=None):
        """Create array of ones."""
        pass

    @abstractmethod
    def array(self, data, dtype=None):
        """Create array from data."""
        pass

    @abstractmethod
    def to_numpy(self, array) -> np.ndarray:
        """Convert backend array to NumPy."""
        pass

    @abstractmethod
    def from_numpy(self, array: np.ndarray):
        """Convert NumPy array to backend."""
        pass

    # KDE operations
    @abstractmethod
    def gaussian_kde(self, x_grid, particles, bandwidth):
        """
        Gaussian kernel density estimation.

        Critical bottleneck for FP particle solvers.
        Each backend provides optimized implementation.
        """
        pass

    # Sparse linear algebra
    @abstractmethod
    def solve_tridiagonal(self, J_L, J_D, J_U, rhs):
        """
        Solve tridiagonal system.

        Critical for HJB Newton iteration.
        """
        pass

    # Gradient operations
    @abstractmethod
    def gradient(self, array, spacing):
        """Compute gradient (numerical derivative)."""
        pass
```

### Concrete Implementations

**PyTorch Backend** (Priority 1):
```python
# mfg_pde/backends/torch_backend.py

import torch
import numpy as np
from .base_backend import BaseBackend

class TorchBackend(BaseBackend):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def name(self) -> str:
        return "torch"

    @property
    def array_module(self):
        return torch

    def zeros(self, shape, dtype=None):
        dtype = dtype or torch.float64
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def gaussian_kde(self, x_grid, particles, bandwidth):
        """PyTorch-optimized KDE with GPU support."""
        x_grid_t = torch.tensor(x_grid, device=self.device, dtype=torch.float64)
        particles_t = torch.tensor(particles, device=self.device, dtype=torch.float64)

        # Vectorized kernel evaluation (GPU-parallel)
        distances = (x_grid_t[:, None] - particles_t[None, :]) / bandwidth
        weights = torch.exp(-0.5 * distances**2)
        density = weights.sum(dim=1) / (len(particles) * bandwidth * np.sqrt(2*np.pi))

        return density.cpu().numpy()

    def solve_tridiagonal(self, J_L, J_D, J_U, rhs):
        """Tridiagonal solve using PyTorch sparse."""
        # Convert to torch tensors
        Nx = len(J_D)

        # Build tridiagonal matrix (dense for small Nx < 1000)
        if Nx < 1000:
            # Dense solve (faster for small systems)
            J_dense = torch.diag(torch.tensor(J_D, device=self.device))
            if Nx > 1:
                J_dense += torch.diag(torch.tensor(J_L[:-1], device=self.device), diagonal=-1)
                J_dense += torch.diag(torch.tensor(J_U[1:], device=self.device), diagonal=1)

            rhs_t = torch.tensor(rhs, device=self.device)
            solution = torch.linalg.solve(J_dense, rhs_t)
            return solution.cpu().numpy()
        else:
            # Sparse solve for large systems
            # TODO: Implement torch.sparse version
            raise NotImplementedError("Sparse solve for Nx > 1000 not implemented")
```

**JAX Backend** (Priority 2):
```python
# mfg_pde/backends/jax_backend.py

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from .base_backend import BaseBackend

class JAXBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "jax"

    @property
    def array_module(self):
        return jnp

    def gaussian_kde(self, x_grid, particles, bandwidth):
        """JAX-optimized KDE with JIT compilation."""

        @jit
        def kernel(x):
            distances = (x - particles) / bandwidth
            weights = jnp.exp(-0.5 * distances**2)
            return jnp.sum(weights) / (len(particles) * bandwidth * jnp.sqrt(2*jnp.pi))

        # vmap over grid points (parallel evaluation)
        density = vmap(kernel)(jnp.array(x_grid))
        return np.array(density)

    def solve_tridiagonal(self, J_L, J_D, J_U, rhs):
        """Tridiagonal solve using JAX."""
        # JAX implementation (dense or iterative solver)
        # ...
        pass
```

**NumPy Backend** (Fallback):
```python
# mfg_pde/backends/numpy_backend.py

import numpy as np
from scipy.stats import gaussian_kde
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from .base_backend import BaseBackend

class NumpyBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "numpy"

    @property
    def array_module(self):
        return np

    def gaussian_kde(self, x_grid, particles, bandwidth):
        """Reference implementation using scipy."""
        kde = gaussian_kde(particles, bw_method=bandwidth)
        return kde(x_grid)

    def solve_tridiagonal(self, J_L, J_D, J_U, rhs):
        """Reference implementation using scipy.sparse."""
        Nx = len(J_D)
        J_L_rolled = np.roll(J_L, -1)
        J_U_rolled = np.roll(J_U, 1)
        Jac = spdiags([J_L_rolled, J_D, J_U_rolled], [-1, 0, 1], Nx, Nx, format='csr')
        return spsolve(Jac, rhs)
```

---

## Dependency Management

### pyproject.toml Configuration

```toml
[project]
name = "mfg_pde"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
    "matplotlib>=3.3",
]

[project.optional-dependencies]
# Tier 1: PyTorch (RL + numerical acceleration)
torch = [
    "torch>=2.0",
]

# Tier 2: JAX (numerical acceleration, lighter than torch)
jax = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
]

# GPU support for JAX
jax-gpu = [
    "jax[cuda12]>=0.4.20",
]

# Everything (both backends)
all = [
    "torch>=2.0",
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
]

# Development
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "ruff>=0.1.0",
]
```

### Installation Scenarios

```bash
# Minimal (NumPy only, universal compatibility)
pip install mfg_pde

# RL users (PyTorch, RL + numerical on same backend)
pip install "mfg_pde[torch]"

# Scientific computing (JAX, no PyTorch overhead)
pip install "mfg_pde[jax]"

# JAX with GPU support
pip install "mfg_pde[jax-gpu]"

# Maximum flexibility (both torch and jax)
pip install "mfg_pde[all]"

# Development (includes testing tools)
pip install "mfg_pde[dev,all]"
```

---

## Testing Strategy

### 1. Numerical Consistency Tests

```python
# tests/backends/test_backend_consistency.py

import pytest
import numpy as np
from mfg_pde.backends import create_backend
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

@pytest.mark.parametrize("backend_type", ["numpy", "torch", "jax"])
def test_kde_consistency(backend_type):
    """Verify all backends produce same KDE results."""
    # Skip if backend not available
    backend = create_backend(backend_type)

    particles = np.random.randn(1000) * 0.2 + 0.5
    x_grid = np.linspace(0, 1, 100)
    bandwidth = 0.1

    # Reference (NumPy/scipy)
    backend_ref = create_backend("numpy")
    density_ref = backend_ref.gaussian_kde(x_grid, particles, bandwidth)

    # Test backend
    density_test = backend.gaussian_kde(x_grid, particles, bandwidth)

    # Verify consistency (< 1e-5 error)
    error = np.max(np.abs(density_ref - density_test))
    assert error < 1e-5, f"{backend_type} KDE error {error:.2e} exceeds tolerance"


@pytest.mark.parametrize("backend_type", ["numpy", "torch", "jax"])
def test_mfg_solve_consistency(backend_type):
    """Verify full MFG solve produces consistent results."""
    problem = MFGProblem(Nx=50, Nt=20, T=0.5)

    # Reference solve
    solver_ref = FPParticleSolver(problem, backend="numpy")
    result_ref = solver_ref.solve()

    # Backend solve
    solver_test = FPParticleSolver(problem, backend=backend_type)
    result_test = solver_test.solve()

    # Compare final density
    M_error = np.max(np.abs(result_ref["M"] - result_test["M"]))
    assert M_error < 1e-4, f"{backend_type} MFG solve error {M_error:.2e}"
```

### 2. Performance Benchmarks

```python
# benchmarks/benchmark_backends.py

import pytest
from mfg_pde.backends import create_backend
from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfg_pde.core.mfg_problem import MFGProblem

def benchmark_kde(benchmark, backend_type):
    """Benchmark KDE performance."""
    backend = create_backend(backend_type)
    particles = np.random.randn(5000)
    x_grid = np.linspace(0, 1, 500)
    bandwidth = 0.1

    result = benchmark(backend.gaussian_kde, x_grid, particles, bandwidth)
    return result


@pytest.mark.benchmark(group="mfg-solve")
@pytest.mark.parametrize("backend_type", ["numpy", "torch", "jax"])
def test_mfg_solve_performance(benchmark, backend_type):
    """Benchmark full MFG solve."""
    problem = MFGProblem(Nx=200, Nt=50)
    solver = FPParticleSolver(problem, backend=backend_type, num_particles=5000)

    result = benchmark(solver.solve, max_iterations=10)

    # Report speedup
    # (pytest-benchmark will show comparison automatically)
```

### 3. Backend Availability Tests

```python
# tests/backends/test_backend_factory.py

def test_auto_selection():
    """Test auto-selection logic."""
    backend = create_backend()  # Auto-select

    # Should select best available
    assert backend.name in ["torch", "jax", "numpy"]

    # If torch available, should select torch
    try:
        import torch
        assert backend.name == "torch"
    except ImportError:
        pass


def test_explicit_selection():
    """Test explicit backend selection."""
    backend_numpy = create_backend("numpy")
    assert backend_numpy.name == "numpy"

    # Try torch (skip if not available)
    try:
        backend_torch = create_backend("torch")
        assert backend_torch.name == "torch"
    except ImportError:
        pytest.skip("PyTorch not available")
```

---

## Implementation Timeline

### Week 1-2: PyTorch Backend (Priority 1)
- ✅ Implement `TorchBackend` class
- ✅ PyTorch KDE with GPU support
- ✅ PyTorch tridiagonal solver
- ✅ Integration with existing RL infrastructure
- ✅ Tests for numerical accuracy

### Week 3-4: JAX Backend (Priority 2)
- ✅ Implement `JAXBackend` class
- ✅ JAX KDE with JIT/vmap
- ✅ JAX tridiagonal solver
- ✅ Tests for numerical accuracy
- ✅ Consistency tests vs PyTorch/NumPy

### Week 5: Integration & Testing
- ✅ Backend factory auto-selection
- ✅ Cross-backend consistency tests
- ✅ Performance benchmarks (all backends)
- ✅ Documentation and examples

### Week 6: Polish & Release
- ✅ Edge case handling
- ✅ Logging and diagnostics
- ✅ User documentation
- ✅ Migration guide

---

## Critical Implementation Details

### 1. Data Type Consistency

**Challenge**: PyTorch defaults to `float32`, JAX/NumPy to `float64`

**Solution**: Enforce `float64` for numerical accuracy
```python
# In TorchBackend
def zeros(self, shape, dtype=None):
    dtype = dtype or torch.float64  # Always float64
    return torch.zeros(shape, dtype=dtype, device=self.device)
```

### 2. Device Management (PyTorch)

**Challenge**: RL solvers may already have device preference

**Solution**: Share device with RL infrastructure
```python
# In TorchBackend.__init__
def __init__(self, device=None):
    if device is None:
        # Auto-detect: GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.device = device
```

### 3. JIT Compilation (JAX)

**Challenge**: Static shapes required for JIT

**Solution**: Separate JIT and non-JIT paths
```python
# In JAXBackend
def gaussian_kde(self, x_grid, particles, bandwidth):
    if len(particles) > 100:  # JIT worth it
        return self._gaussian_kde_jit(x_grid, particles, bandwidth)
    else:  # Overhead too large
        return self._gaussian_kde_eager(x_grid, particles, bandwidth)
```

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ All three backends (torch, jax, numpy) implemented
- ✅ Auto-selection works (torch > jax > numpy)
- ✅ Numerical consistency (< 1e-5 error between backends)
- ✅ PyTorch integrates with existing RL infrastructure
- ✅ JAX provides alternative for non-RL users

### Performance Targets
- ✅ PyTorch: 5-20x speedup on GPU vs NumPy
- ✅ JAX: 5-20x speedup on GPU vs NumPy
- ✅ PyTorch and JAX: Similar performance (within 2x)

### User Experience
- ✅ Zero configuration needed (auto-selection just works)
- ✅ Clear logging about backend selection
- ✅ Easy explicit override (`backend="..."`)
- ✅ Comprehensive documentation

---

## Migration Guide for Users

### Existing Code (Phase 2)
```python
# Old: backend parameter existed but no acceleration
solver = FPParticleSolver(problem, backend="numpy")
```

### New Code (Phase 3)
```python
# New: auto-selects best backend
solver = FPParticleSolver(problem)
# → Uses torch if available (RL users)
# → Falls back to jax if torch unavailable
# → Falls back to numpy if neither available

# Explicit choice still works
solver = FPParticleSolver(problem, backend="jax")
```

**No breaking changes**: All existing code continues to work!

---

## Conclusion

**Tiered backend strategy (torch > jax > numpy) provides**:
- ✅ **Maximum performance** for users with PyTorch (RL integration)
- ✅ **Alternative acceleration** for users without PyTorch (JAX)
- ✅ **Universal compatibility** for all users (NumPy fallback)
- ✅ **Intelligent auto-selection** (works out of the box)
- ✅ **Professional architecture** (factory pattern, dependency injection)

**Status**: Ready for implementation
**Timeline**: 6 weeks for complete dual backend system
**Risk**: Low (well-established patterns, clear testing strategy)

---

**Author**: MFG_PDE Development Team
**Date**: 2025-10-04
**Approved**: Ready to proceed
