# Backend-Aware Auto-Switching Architecture

**Date**: 2025-10-08
**Status**: 🏗️ Design Phase
**Motivation**: Replace hard-coded if-else with elegant, intelligent backend selection

## Design Philosophy

**Core Principle**: *The solver should not know about backend implementation details. Backend selection should be automatic, intelligent, and based on runtime characteristics.*

**Design Patterns**:
1. **Strategy Pattern**: Separate algorithm implementations as strategies
2. **Factory Pattern**: Auto-select strategy based on context
3. **Capability-Based Selection**: Query backend capabilities, not backend type
4. **Runtime Profiling**: Learn from actual performance, not assumptions

---

## Current Problems

### Track A (FDM Solvers) - Boundary Conversion
```python
# mfg_pde/backends/compat.py
def ensure_numpy(arr, backend):
    """Convert backend tensor to numpy at boundaries."""
    if backend is not None:
        return backend.to_numpy(arr)
    return arr
```

**Approach**: Single implementation with automatic conversion
**Pros**: Simple, backend-agnostic algorithm code
**Cons**: Many small transfers, not optimal for GPU

### Track B (Particle Solver) - Hard-Coded Selection
```python
def solve_fp_system(self, m_initial, U_drift):
    if self.backend is not None:  # ❌ Hard-coded!
        return self._solve_fp_system_gpu(...)
    else:
        return self._solve_fp_system_cpu(...)
```

**Approach**: Separate CPU and GPU implementations
**Pros**: Optimal GPU pipeline (no transfers)
**Cons**: Hard-coded selection, solver knows too much

### What's Wrong?
1. **Solver knows about backends**: Violates separation of concerns
2. **Binary choice**: Can't handle hybrid strategies
3. **No intelligence**: Doesn't adapt to problem size
4. **Not extensible**: Adding JAX/TPU requires modifying solver code

---

## Proposed Architecture

### Layer 1: Backend Capability Protocol

**File**: `mfg_pde/backends/backend_protocol.py`

```python
from typing import Protocol, Dict, Any

class BackendCapability(Protocol):
    """Protocol defining backend capabilities for auto-selection."""

    def has_capability(self, capability: str) -> bool:
        """Check if backend supports a specific capability.

        Capabilities:
        - "parallel_kde": Efficient GPU kernel density estimation
        - "parallel_interpolation": Fast parallel interpolation
        - "low_latency": Low kernel launch overhead (<10μs)
        - "high_bandwidth": High memory bandwidth (>100 GB/s)
        - "unified_memory": CPU/GPU share memory
        """
        ...

    def get_performance_hints(self) -> Dict[str, Any]:
        """Return performance characteristics.

        Returns:
            {
                "kernel_overhead_us": 50,  # Kernel launch overhead (μs)
                "memory_bandwidth_gb": 200,  # Memory bandwidth (GB/s)
                "device_type": "mps",  # "cuda", "mps", "cpu"
                "optimal_problem_size": (50000, 100, 50),  # (N, Nx, Nt)
            }
        """
        ...
```

### Layer 2: Strategy Implementations

**File**: `mfg_pde/alg/numerical/fp_solvers/particle_strategies.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class ParticleStrategy(ABC):
    """Abstract base class for particle solver strategies."""

    @abstractmethod
    def solve(
        self,
        m_initial: np.ndarray,
        U_drift: np.ndarray,
        problem: "MFGProblem",
        num_particles: int,
        **kwargs
    ) -> np.ndarray:
        """Solve FP system using specific strategy."""
        pass

    @abstractmethod
    def estimate_cost(
        self,
        problem_size: tuple[int, int, int]  # (N, Nx, Nt)
    ) -> float:
        """Estimate computational cost (seconds) for problem size."""
        pass


class CPUParticleStrategy(ParticleStrategy):
    """CPU-based particle solver using NumPy + scipy."""

    def solve(self, m_initial, U_drift, problem, num_particles, **kwargs):
        # Existing _solve_fp_system_cpu implementation
        pass

    def estimate_cost(self, problem_size):
        N, Nx, Nt = problem_size
        # Empirical cost model from benchmarks
        kde_cost = Nt * N * Nx * 1e-8  # KDE dominates
        other_cost = Nt * N * 1e-9
        return kde_cost + other_cost


class GPUParticleStrategy(ParticleStrategy):
    """GPU-based particle solver with internal KDE."""

    def __init__(self, backend):
        self.backend = backend

    def solve(self, m_initial, U_drift, problem, num_particles, **kwargs):
        # Existing _solve_fp_system_gpu implementation
        pass

    def estimate_cost(self, problem_size):
        N, Nx, Nt = problem_size
        hints = self.backend.get_performance_hints()

        # Account for kernel overhead
        kernel_overhead = hints["kernel_overhead_us"] * 1e-6
        num_kernels = Nt * 5  # ~5 kernels per iteration
        overhead_cost = num_kernels * kernel_overhead

        # Compute cost (GPU is faster per operation)
        kde_cost = Nt * N * Nx * 2e-9  # ~5x faster than CPU
        other_cost = Nt * N * 5e-10

        return overhead_cost + kde_cost + other_cost


class HybridParticleStrategy(ParticleStrategy):
    """Hybrid CPU/GPU strategy for intermediate problems."""

    def __init__(self, backend):
        self.backend = backend

    def solve(self, m_initial, U_drift, problem, num_particles, **kwargs):
        # Use GPU for KDE, CPU for other ops
        # Best of both worlds for medium problems
        pass

    def estimate_cost(self, problem_size):
        # Combine CPU and GPU cost models
        pass
```

### Layer 3: Intelligent Strategy Selector

**File**: `mfg_pde/alg/numerical/fp_solvers/strategy_selector.py`

```python
from typing import Optional, Dict, Any
import numpy as np

class StrategySelector:
    """Intelligent strategy selection based on backend capabilities and problem size."""

    def __init__(self):
        self.performance_cache: Dict[str, float] = {}
        self.enable_profiling = True

    def select_strategy(
        self,
        backend: Optional["BaseBackend"],
        problem_size: tuple[int, int, int],  # (N, Nx, Nt)
        strategy_hint: Optional[str] = None
    ) -> ParticleStrategy:
        """Select optimal strategy based on context.

        Parameters
        ----------
        backend : BaseBackend, optional
            Backend to use for GPU strategies
        problem_size : tuple
            (num_particles, grid_size, time_steps)
        strategy_hint : str, optional
            User override: "cpu", "gpu", "hybrid", "auto"

        Returns
        -------
        ParticleStrategy
            Optimal strategy for the given context
        """
        # User override
        if strategy_hint in ("cpu", "gpu", "hybrid"):
            return self._create_strategy(strategy_hint, backend)

        # No backend available
        if backend is None:
            return CPUParticleStrategy()

        # Auto-selection based on capabilities and problem size
        return self._auto_select(backend, problem_size)

    def _auto_select(
        self,
        backend: "BaseBackend",
        problem_size: tuple[int, int, int]
    ) -> ParticleStrategy:
        """Automatically select best strategy."""
        N, Nx, Nt = problem_size

        # Create candidate strategies
        strategies = {
            "cpu": CPUParticleStrategy(),
        }

        # Add GPU strategy if backend supports parallel KDE
        if backend.has_capability("parallel_kde"):
            strategies["gpu"] = GPUParticleStrategy(backend)

        # Add hybrid if both available and problem is medium-sized
        if "gpu" in strategies and 10000 <= N <= 100000:
            strategies["hybrid"] = HybridParticleStrategy(backend)

        # Estimate cost for each strategy
        costs = {
            name: strategy.estimate_cost(problem_size)
            for name, strategy in strategies.items()
        }

        # Select minimum cost strategy
        best_name = min(costs, key=costs.get)
        best_strategy = strategies[best_name]

        # Log selection (optional)
        if self.enable_profiling:
            self._log_selection(best_name, costs, problem_size)

        return best_strategy

    def _create_strategy(self, name: str, backend) -> ParticleStrategy:
        """Factory method for strategy creation."""
        if name == "cpu":
            return CPUParticleStrategy()
        elif name == "gpu":
            if backend is None:
                raise ValueError("GPU strategy requires backend")
            return GPUParticleStrategy(backend)
        elif name == "hybrid":
            if backend is None:
                raise ValueError("Hybrid strategy requires backend")
            return HybridParticleStrategy(backend)
        else:
            raise ValueError(f"Unknown strategy: {name}")

    def _log_selection(self, selected: str, costs: Dict, problem_size):
        """Log strategy selection for debugging."""
        N, Nx, Nt = problem_size
        print(f"[StrategySelector] Problem size: N={N}, Nx={Nx}, Nt={Nt}")
        print(f"[StrategySelector] Estimated costs: {costs}")
        print(f"[StrategySelector] Selected: {selected}")
```

### Layer 4: Refactored Solver

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`

```python
class FPParticleSolver(BaseFPSolver):
    def __init__(
        self,
        problem: MFGProblem,
        num_particles: int = 5000,
        kde_bandwidth: Any = "scott",
        normalize_kde_output: bool = True,
        boundary_conditions: BoundaryConditions | None = None,
        backend: str | None = None,
        strategy: str = "auto",  # New parameter
    ) -> None:
        super().__init__(problem)
        self.fp_method_name = "Particle"
        self.num_particles = num_particles
        self.kde_bandwidth = kde_bandwidth
        self.normalize_kde_output = normalize_kde_output
        self.M_particles_trajectory: np.ndarray | None = None

        # Initialize backend
        from mfg_pde.backends import create_backend
        if backend is not None:
            self.backend = create_backend(backend)
        else:
            self.backend = None

        # Boundary conditions
        if boundary_conditions is None:
            self.boundary_conditions = BoundaryConditions(type="periodic")
        else:
            self.boundary_conditions = boundary_conditions

        # Strategy selection (NEW!)
        from .strategy_selector import StrategySelector
        self.selector = StrategySelector()
        self.strategy_hint = strategy  # "auto", "cpu", "gpu", "hybrid"

    def solve_fp_system(
        self,
        m_initial_condition: np.ndarray,
        U_solution_for_drift: np.ndarray
    ) -> np.ndarray:
        """Solve FP system using automatically selected strategy.

        This method delegates to the optimal strategy based on:
        - Backend capabilities
        - Problem size (num_particles, Nx, Nt)
        - Runtime profiling (if enabled)

        The solver is agnostic to implementation details.
        """
        # Determine problem size
        problem_size = (
            self.num_particles,
            self.problem.Nx + 1,
            self.problem.Nt + 1
        )

        # Select optimal strategy
        strategy = self.selector.select_strategy(
            self.backend,
            problem_size,
            strategy_hint=self.strategy_hint
        )

        # Delegate to strategy (clean separation!)
        return strategy.solve(
            m_initial_condition,
            U_solution_for_drift,
            self.problem,
            self.num_particles,
            kde_bandwidth=self.kde_bandwidth,
            normalize_kde_output=self.normalize_kde_output,
            boundary_conditions=self.boundary_conditions,
            backend=self.backend
        )
```

---

## Backend Capability Implementation

### PyTorch Backend

```python
class TorchBackend(BaseBackend):
    def has_capability(self, capability: str) -> bool:
        capabilities = {
            "parallel_kde": True,
            "parallel_interpolation": True,
            "low_latency": self.device.type == "cuda",  # MPS has higher latency
            "high_bandwidth": self.device.type == "cuda",
            "unified_memory": self.device.type == "mps",
        }
        return capabilities.get(capability, False)

    def get_performance_hints(self) -> Dict[str, Any]:
        if self.device.type == "mps":
            return {
                "kernel_overhead_us": 50,
                "memory_bandwidth_gb": 200,
                "device_type": "mps",
                "optimal_problem_size": (50000, 100, 50),
            }
        elif self.device.type == "cuda":
            return {
                "kernel_overhead_us": 5,
                "memory_bandwidth_gb": 900,
                "device_type": "cuda",
                "optimal_problem_size": (10000, 100, 50),
            }
        else:  # CPU
            return {
                "kernel_overhead_us": 0,
                "memory_bandwidth_gb": 50,
                "device_type": "cpu",
                "optimal_problem_size": (5000, 50, 20),
            }
```

### NumPy Backend (Fallback)

```python
class NumpyBackend(BaseBackend):
    def has_capability(self, capability: str) -> bool:
        # NumPy doesn't have GPU capabilities
        return False

    def get_performance_hints(self) -> Dict[str, Any]:
        return {
            "kernel_overhead_us": 0,
            "memory_bandwidth_gb": 50,
            "device_type": "cpu",
            "optimal_problem_size": (5000, 50, 20),
        }
```

---

## Usage Examples

### Automatic Strategy Selection

```python
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver

problem = MFGProblem(Nx=50, Nt=50, ...)
solver = FPParticleSolver(
    problem,
    num_particles=50000,
    backend="torch",  # or "jax", "numpy"
    strategy="auto"  # Automatic selection (default)
)

M = solver.solve_fp_system(m_initial, U_drift)
# Automatically selects GPU strategy (N=50k is optimal for MPS)
```

### Manual Override

```python
# Force CPU even with GPU available (for debugging)
solver = FPParticleSolver(
    problem,
    num_particles=50000,
    backend="torch",
    strategy="cpu"  # Override: always use CPU
)
```

### Problem-Size Adaptation

```python
# Small problem - auto-selects CPU
solver_small = FPParticleSolver(
    problem,
    num_particles=5000,  # Too small for GPU
    backend="torch",
    strategy="auto"
)
# -> Automatically selects CPUParticleStrategy

# Large problem - auto-selects GPU
solver_large = FPParticleSolver(
    problem,
    num_particles=100000,  # Large enough for GPU
    backend="torch",
    strategy="auto"
)
# -> Automatically selects GPUParticleStrategy
```

---

## Runtime Profiling (Phase 2)

### Adaptive Strategy Selection

```python
class AdaptiveStrategySelector(StrategySelector):
    """Learn from runtime performance and adapt strategy selection."""

    def __init__(self):
        super().__init__()
        self.performance_history: Dict[str, List[float]] = {}

    def select_strategy(self, backend, problem_size, strategy_hint=None):
        # First time: use cost estimates
        if not self.performance_history:
            return super().select_strategy(backend, problem_size, strategy_hint)

        # Use historical performance data
        return self._select_from_history(backend, problem_size)

    def record_performance(self, strategy_name: str, problem_size, elapsed_time: float):
        """Record actual performance for learning."""
        key = f"{strategy_name}_{problem_size}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        self.performance_history[key].append(elapsed_time)

    def _select_from_history(self, backend, problem_size):
        # Use actual timing data to select strategy
        # Can implement machine learning here
        pass
```

---

## Benefits of This Architecture

### 1. Separation of Concerns ✅
- Solver doesn't know about backend details
- Strategy implementations are independent
- Easy to test each component separately

### 2. Extensibility ✅
```python
# Adding JAX backend requires ZERO changes to solver
class JAXParticleStrategy(ParticleStrategy):
    def solve(self, ...):
        # JAX-specific implementation
        pass

# Automatically available via selector
```

### 3. Intelligence ✅
- Problem-size-aware selection
- Backend-capability-based selection
- Runtime profiling and adaptation

### 4. User Control ✅
```python
# Automatic (default)
solver = FPParticleSolver(..., strategy="auto")

# Manual override when needed
solver = FPParticleSolver(..., strategy="cpu")
```

### 5. Performance Optimization ✅
- Estimates cost before running
- Learns from actual performance
- Adapts to hardware characteristics

---

## Implementation Plan

### Phase 1: Core Architecture (Current)
1. ✅ Design document (this file)
2. Implement `BackendCapability` protocol
3. Create `ParticleStrategy` base class
4. Implement `CPUParticleStrategy` (extract from current code)
5. Implement `GPUParticleStrategy` (extract from current code)
6. Implement `StrategySelector` with cost estimation

### Phase 2: Integration
7. Refactor `FPParticleSolver` to use strategy pattern
8. Update backends to implement capability protocol
9. Write tests for strategy selection
10. Validate numerical correctness

### Phase 3: Advanced Features
11. Implement `HybridParticleStrategy`
12. Add runtime profiling (`AdaptiveStrategySelector`)
13. Benchmark and tune cost models
14. Document usage and best practices

---

## Testing Strategy

### Unit Tests
```python
def test_strategy_selection_small_problem():
    """Small problems should select CPU."""
    selector = StrategySelector()
    backend = TorchBackend(device="mps")

    strategy = selector.select_strategy(
        backend,
        problem_size=(5000, 50, 50)  # N=5k (small)
    )

    assert isinstance(strategy, CPUParticleStrategy)

def test_strategy_selection_large_problem():
    """Large problems should select GPU."""
    selector = StrategySelector()
    backend = TorchBackend(device="mps")

    strategy = selector.select_strategy(
        backend,
        problem_size=(100000, 50, 50)  # N=100k (large)
    )

    assert isinstance(strategy, GPUParticleStrategy)
```

### Integration Tests
```python
def test_auto_switching_produces_correct_results():
    """Verify all strategies produce same numerical results."""
    problem = MFGProblem(...)

    # CPU strategy
    solver_cpu = FPParticleSolver(problem, strategy="cpu")
    M_cpu = solver_cpu.solve_fp_system(m_initial, U_drift)

    # GPU strategy
    solver_gpu = FPParticleSolver(problem, backend="torch", strategy="gpu")
    M_gpu = solver_gpu.solve_fp_system(m_initial, U_drift)

    # Auto strategy
    solver_auto = FPParticleSolver(problem, backend="torch", strategy="auto")
    M_auto = solver_auto.solve_fp_system(m_initial, U_drift)

    # All should be statistically equivalent
    np.testing.assert_allclose(M_cpu, M_gpu, rtol=0.3)
    np.testing.assert_allclose(M_auto, M_gpu, rtol=0.3)
```

---

## Conclusion

This architecture provides **elegant, intelligent backend selection** that:

1. **Separates concerns**: Solver agnostic to backend details
2. **Enables extensibility**: New backends/strategies drop in seamlessly
3. **Optimizes performance**: Intelligent selection based on problem context
4. **Maintains correctness**: All strategies produce equivalent results
5. **Provides control**: Users can override when needed

**Next Steps**: Implement Phase 1 components and integrate into existing solver.

---

**End of Design Document**
