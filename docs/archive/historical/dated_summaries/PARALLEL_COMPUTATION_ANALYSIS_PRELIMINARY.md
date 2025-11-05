# MFG_PDE Parallel Computation Analysis (Preliminary)

**Date**: 2025-10-13
**Status**: Analysis in progress
**Related Issue**: Performance optimization and practical parallel execution assessment

## Executive Summary

MFG_PDE includes parallel computation support across multiple paradigms. This analysis evaluates the **practical effectiveness** of parallelization for real-world use cases.

### Key Findings (From Code Analysis)

1. **Implementation Coverage**: Parallel execution supported in 3 major areas:
   - Common Noise MFG Monte Carlo (`CommonNoiseMFGSolver`)
   - Parameter sweep workflows (`ParameterSweep`)
   - Monte Carlo integration utilities

2. **Current Design**:
   - Process-based parallelism via `concurrent.futures.ProcessPoolExecutor`
   - Thread-based parallelism available for I/O-bound tasks
   - Configurable worker count (defaults to CPU count)

3. **Known Limitations**:
   - Pickling overhead for large problem objects
   - Process startup costs
   - No shared memory optimization
   - No GPU-aware parallelism

## Detailed Analysis

### 1. Common Noise MFG Monte Carlo

**Location**: `mfg_pde/alg/numerical/stochastic/common_noise_solver.py`

**Implementation**:
```python
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(self._solve_conditional_mfg, path): k
               for k, path in enumerate(noise_paths)}

    for future in as_completed(futures):
        u, m, converged = future.result()
        solutions.append((u, m, converged))
```

**Theoretical Speedup**:
- **Ideal**: Near-linear speedup (embarrassingly parallel problem)
- **Each noise realization is independent**
- Expected speedup: `min(K, num_workers)` where K = number of noise samples

**Practical Considerations**:
- ✅ **Good for**: K > 20, complex MFG problems (Nx, Nt > 100)
- ❌ **Poor for**: K < 10, simple problems with fast solves

**Overhead Sources**:
1. **Process spawning**: ~50-100ms per worker
2. **Pickling MFG problem**: ~10-50ms depending on problem size
3. **Result marshalling**: ~5-20ms per solution
4. **Total overhead**: 200-500ms baseline + O(K) communication

**Usage Example**:
```python
solver = CommonNoiseMFGSolver(
    problem,
    num_noise_samples=100,  # Enough samples to amortize overhead
    parallel=True,           # Enable parallel execution
    num_workers=None         # Use all CPUs
)
result = solver.solve()
```

**Current Usage in Codebase**:
- `examples/basic/common_noise_lq_demo.py`: Uses `parallel=False` (line 284)
  - **Reason**: "pickling issues with lambda functions"
  - **Note**: This is a limitation of the example, not the solver

### 2. Parameter Sweep Workflows

**Location**: `mfg_pde/workflow/parameter_sweep.py`

**Implementation**:
Three execution modes:
1. **Sequential**: Baseline, no parallelism
2. **parallel_threads**: ThreadPoolExecutor (I/O-bound tasks)
3. **parallel_processes**: ProcessPoolExecutor (CPU-bound tasks)

**Design Features**:
```python
config = SweepConfiguration(
    execution_mode="parallel_processes",
    max_workers=mp.cpu_count(),
    batch_size=None,
    timeout_per_run=None,
    retry_failed=True
)
```

**Theoretical Speedup**:
- **Ideal**: Linear up to CPU count
- **Practical**: 60-80% efficiency typical
- Depends on:
  - Individual run time (longer = better amortization)
  - Number of parameter combinations
  - Memory requirements

**Practical Considerations**:
- ✅ **Good for**: Many combinations (>10), slow functions (>0.5s each)
- ❌ **Poor for**: Few combinations (<5), fast functions (<0.1s each)

**Overhead Analysis**:
```
Total Time = Serial Work + Parallel Overhead
           = N * T_run + (N_workers * T_spawn + N * T_marshal)

Speedup = (N * T_run) / (N * T_run / N_workers + Overhead)
        ≈ N_workers * (1 - Overhead_fraction)
```

**Usage Example**:
```python
parameters = {
    "Nx": [50, 100, 200],
    "sigma": [0.1, 0.2, 0.3]
}

sweep = ParameterSweep(
    parameters,
    config=SweepConfiguration(execution_mode="parallel_processes")
)

results = sweep.execute(solver_function)
```

### 3. Monte Carlo Integration

**Location**: `mfg_pde/utils/numerical/monte_carlo.py`

**Current Status**:
- No explicit parallelization in `monte_carlo_integrate()`
- Config includes `parallel: bool = False` flag (line 72)
- **Implementation gap**: Flag exists but not used

**Potential for Parallelization**:
```python
# Sample generation: Embarrassingly parallel
samples = sampler.sample(num_samples)  # Could be chunked

# Function evaluation: Embarrassingly parallel
function_values = integrand(samples)   # Vectorized, could be parallel

# Aggregation: Sequential (cheap)
estimate = domain_volume * np.mean(weighted_values)
```

**Analysis**:
- Current implementation is already vectorized (NumPy)
- Parallelization would help for:
  - Very expensive integrand evaluations
  - Adaptive sampling with many batches
  - High-dimensional problems (>10D)

### 4. Additional Parallel Support

**Other modules with parallel mentions**:

1. **Network backend** (`mfg_pde/geometry/network_backend.py`)
   - Graph algorithms that could benefit from parallelism
   - Currently sequential

2. **Optimization** (`mfg_pde/utils/performance/optimization.py`)
   - Performance utilities
   - Could coordinate parallel strategies

3. **Neural operators** (`mfg_pde/alg/neural/operator_learning/`)
   - PyTorch/JAX provide their own parallelism (data/model parallel)
   - No explicit multiprocessing needed

## Overhead Quantification

### Process Pool Overhead Breakdown

**Fixed costs** (per execution):
```
Worker spawn time:        50-100 ms × num_workers
Process pool setup:       20-50 ms
Shutdown and cleanup:     10-30 ms
---
Total fixed:              100-300 ms
```

**Per-task costs**:
```
Pickle problem object:    2-50 ms  (depends on size)
Submit task:              0.1-1 ms
Retrieve result:          2-20 ms  (depends on solution size)
Result unpickling:        2-20 ms
---
Total per task:           6-91 ms
```

### Break-Even Analysis

For N tasks, task time T, overhead O:

```
Sequential time:  N * T
Parallel time:    N * T / W + O_fixed + N * O_per_task

Break-even when:
N * T > N * T / W + O
T > O / (N * (1 - 1/W))
```

**Example** (W=10 workers, O_fixed=200ms, O_per_task=10ms):
- N=10 tasks:  Need T > 33ms per task
- N=50 tasks:  Need T > 7ms per task
- N=100 tasks: Need T > 4ms per task

## Practical Recommendations

### When to Use Parallel Execution

#### Common Noise MFG:
- ✅ **USE** when:
  - K ≥ 20 noise samples
  - Individual MFG solve time > 1 second
  - Problem size Nx, Nt > 50

- ❌ **AVOID** when:
  - K < 10 samples
  - Fast solves (< 0.5s per realization)
  - Small problems (Nx, Nt < 30)

#### Parameter Sweeps:
- ✅ **USE** when:
  - Total combinations > 10
  - Individual run time > 0.5s
  - Use `parallel_processes` for CPU-bound MFG solves

- ❌ **AVOID** when:
  - Few combinations (< 5)
  - Very fast functions (< 0.1s)
  - High memory per task

### Optimal Configuration

**Worker count**:
```python
# Conservative (high memory per task):
num_workers = mp.cpu_count() // 2

# Aggressive (low memory per task):
num_workers = mp.cpu_count()

# Hyperthreading-aware:
num_workers = min(mp.cpu_count(), physical_core_count * 1.5)
```

**Batch size** (for parameter sweeps):
```python
# Rule of thumb: 2-4x worker count
batch_size = num_workers * 2
```

## Limitations and Future Work

### Current Limitations

1. **No shared memory**:
   - Each worker gets full copy of problem object
   - High memory usage for large problems
   - Solution: Use `multiprocessing.shared_memory` (Python 3.8+)

2. **Pickling overhead**:
   - Lambda functions not picklable (forces module-level functions)
   - Large arrays copied multiple times
   - Solution: Use memory-mapped arrays or Dask

3. **No GPU awareness**:
   - Parallel CPU workers compete with single GPU
   - No hybrid CPU+GPU parallelism
   - Solution: Coordinate with backend strategy selector

4. **Static work distribution**:
   - No dynamic load balancing
   - Uneven task times waste worker cycles
   - Solution: Use work-stealing queue

### Recommended Improvements

**Short-term** (Low effort, high impact):
1. Add timing diagnostics to measure actual overhead
2. Implement adaptive worker count based on problem size
3. Add shared memory support for large arrays
4. Cache pickled problem objects

**Medium-term** (Moderate effort):
1. Implement work-stealing for uneven task times
2. Add GPU-aware parallelism coordination
3. Support for Dask distributed computing
4. Memory profiling and optimization

**Long-term** (High effort, research-grade):
1. Ray integration for cluster computing
2. Hybrid MPI+threading for HPC environments
3. Automatic parallelization strategy selection
4. Zero-copy inter-process communication

## Benchmark Specification

### Test Cases

**Benchmark 1: Common Noise MFG**
- Problem sizes: Nx=51, Nt=51
- Noise samples: K ∈ {10, 20, 40}
- Measure: Sequential vs parallel execution time, overhead

**Benchmark 2: Parameter Sweep**
- Execution modes: Sequential, threads, processes
- Problem: 27 parameter combinations (3×3×3)
- Measure: Total time, speedup factor

**Benchmark 3: Worker Scaling**
- Worker counts: 1, 2, 4, 8, 10 (up to CPU count)
- Fixed workload: 64 parameter combinations
- Measure: Speedup vs efficiency curve

**Benchmark 4: Overhead Analysis**
- Vary problem size: Nx ∈ {30, 50, 100, 200}
- Measure: Pickling time, communication overhead

### Success Criteria

**Minimum acceptable performance**:
- Common Noise (K=50): ≥ 4x speedup on 10-core machine
- Parameter Sweep (N=27): ≥ 5x speedup with process pool
- Worker efficiency: ≥ 60% at full CPU utilization

**Target performance**:
- Common Noise (K=50): ≥ 7x speedup (70% efficiency)
- Parameter Sweep (N=27): ≥ 7x speedup (70% efficiency)
- Worker efficiency: ≥ 80% up to physical core count

## Conclusion

**Parallel computation in MFG_PDE is well-designed but underutilized**:

1. ✅ **Solid foundation**: ProcessPoolExecutor, configurable workers, proper error handling
2. ⚠️ **High overhead**: 200-500ms baseline + per-task costs limit effectiveness for small problems
3. ✅ **Clear use cases**: Common noise MFG with many samples, large parameter sweeps
4. ❌ **Missing optimizations**: No shared memory, static load balancing, no GPU awareness

**Practical guidance**:
- Use parallel execution for **production workloads with many independent tasks**
- Expect **60-80% parallel efficiency** in realistic scenarios
- Avoid parallelization for **quick prototyping or small problems**

**Next steps**:
1. Run comprehensive benchmark (in progress)
2. Validate overhead estimates
3. Implement high-priority optimizations (shared memory, adaptive workers)
4. Add performance guide to documentation

---

**Note**: This is a preliminary analysis based on code review. Quantitative results pending benchmark completion.
