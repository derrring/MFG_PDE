# AMR Evaluation Benchmarks

This directory contains comprehensive benchmarking tools for evaluating **Adaptive Mesh Refinement (AMR)** effectiveness in MFG solvers.

## 🎯 **Purpose**

Evaluate the performance, accuracy, and efficiency gains of AMR-enhanced MFG solvers compared to uniform grid approaches across realistic problem scenarios.

## 📊 **Benchmark Components**

### **Core AMR Evaluation Modules**

1. **`amr_performance_benchmark.py`** - Performance comparison suite
   - Solve time analysis: AMR vs uniform grids
   - Memory usage tracking and optimization
   - Mesh efficiency metrics (elements used vs baseline)
   - Convergence behavior across problem scales

2. **`amr_accuracy_benchmark.py`** - Solution accuracy validation
   - Manufactured solution testing with analytical forms
   - L2/H1 error convergence rate analysis
   - Sharp feature handling effectiveness
   - Reference solution accuracy comparisons

3. **`amr_gpu_profiler.py`** - GPU/CPU performance profiling
   - JAX compilation overhead analysis
   - Memory transfer benchmarking (CPU↔GPU)
   - Backend performance comparison
   - AMR operations breakdown timing

4. **`amr_memory_profiler.py`** - Memory usage analysis
   - Memory scaling with problem size
   - Memory leak detection across multiple solves
   - Component breakdown (mesh/solution/overhead)
   - Memory efficiency calculations

5. **`real_world_problems.py`** - Realistic MFG problem collection
   - **Traffic Flow**: Highway bottlenecks and variable capacity
   - **Financial Markets**: Volatility clustering and liquidity shocks
   - **Crowd Dynamics**: Emergency evacuation with obstacles
   - **Energy Trading**: Renewable intermittency and demand peaks
   - **Epidemic Spread**: Population heterogeneity and policy boundaries

6. **`comprehensive_benchmark.py`** - Master benchmark orchestrator
   - Coordinates all AMR evaluation components
   - Generates unified executive reports
   - Real-world problem validation suite
   - Machine-readable JSON result summaries

## 🚀 **Quick Start**

### Run Complete AMR Evaluation
```bash
cd benchmarks/amr_evaluation/
python comprehensive_benchmark.py
```

### Run Individual AMR Benchmarks
```bash
# Performance comparison
python amr_performance_benchmark.py

# Accuracy analysis
python amr_accuracy_benchmark.py

# GPU/CPU profiling
python amr_gpu_profiler.py

# Memory usage analysis
python amr_memory_profiler.py

# Real-world problem testing
python real_world_problems.py
```

## 📈 **Key Evaluation Metrics**

### Performance Metrics
- **Speedup Ratio**: AMR solve time vs uniform grid time
- **Mesh Efficiency**: Elements used vs uniform grid (target: <0.7 for good AMR)
- **Memory Usage**: Peak and average memory consumption scaling
- **Convergence Rate**: Iterations to convergence comparison

### Accuracy Metrics
- **L2 Error**: Solution accuracy vs reference solutions
- **H1 Error**: Gradient accuracy measurement
- **Convergence Order**: Error reduction rate with refinement
- **Feature Preservation**: Sharp gradient and discontinuity handling

### Resource Utilization
- **GPU Acceleration**: JAX compilation overhead and compute efficiency
- **Memory Scaling**: Sub-linear growth validation
- **Memory Leaks**: Stability across repeated solves
- **Component Efficiency**: Mesh/solution/overhead breakdown

## 🏗️ **Real-World Problem Validation**

### Why These Problems Test AMR Effectiveness

Each problem is specifically designed to highlight AMR advantages:

```python
from real_world_problems import RealWorldMFGProblems

problems = RealWorldMFGProblems()

# Traffic Flow - Sharp density changes at bottlenecks
traffic = problems.create_problem('traffic_flow')

# Financial Markets - Localized volatility clustering  
finance = problems.create_problem('financial_market')

# Crowd Dynamics - Sharp gradients around obstacles
crowd = problems.create_problem('crowd_dynamics')

# Energy Trading - Price spikes requiring fine resolution
energy = problems.create_problem('energy_trading')

# Epidemic Spread - Sharp transitions at policy boundaries
epidemic = problems.create_problem('epidemic_spread')
```

## 📊 **Output Structure**

```
comprehensive_benchmark_results/
├── comprehensive_benchmark_report.md    # Executive summary
├── benchmark_summary.json              # Machine-readable metrics
├── performance/
│   ├── benchmark_results.json
│   └── performance_report.md
├── accuracy/
│   ├── accuracy_report.md
│   └── convergence_analysis.png
├── gpu_profiling/
│   ├── gpu_profiling_results.json
│   └── gpu_performance_analysis.png
└── memory_profiling/
    ├── memory_analysis_results.json
    ├── memory_analysis_report.md
    └── memory_analysis.png
```

## 🎯 **Expected AMR Benefits**

### Performance Improvements
- **10-100x speedup** for problems with localized features
- **50-90% memory reduction** through adaptive element allocation
- **Better convergence** on problems with sharp gradients

### Quality Improvements  
- **Higher accuracy** with fewer total elements
- **Better feature resolution** in critical regions
- **Maintained solution quality** with computational savings

## 🔧 **Configuration**

### AMR Parameters
```python
amr_config = {
    'error_threshold': 1e-4,      # Refinement threshold
    'max_levels': 5,              # Maximum refinement levels
    'initial_intervals': 32,      # Starting grid size
    'refinement_factor': 2        # Subdivision factor
}
```

### Problem Scaling
```python
problem_sizes = [32, 64, 128, 256]  # Grid sizes to benchmark
```

## 📚 **Usage Examples**

### Custom AMR Problem Evaluation
```python
from comprehensive_benchmark import ComprehensiveAMRBenchmark
from mfg_pde import ExampleMFGProblem

# Create custom problem with sharp features
def create_sharp_problem():
    return ExampleMFGProblem(
        T=1.0, Nx=64, sigma=0.01,  # Low diffusion = sharp features
        coefCT=3.0  # High congestion = localized dynamics
    )

# Evaluate AMR effectiveness
benchmark = ComprehensiveAMRBenchmark()
results = benchmark.performance_suite.benchmark_problem(
    create_sharp_problem,
    "Sharp_Features_Test",
    [
        {'name': 'Uniform', 'amr_enabled': False},
        {'name': 'AMR', 'amr_enabled': True, 'error_threshold': 1e-4}
    ]
)
```

## 🚨 **Dependencies**

### Required
- `numpy`, `matplotlib`, `psutil` - Basic scientific computing
- `mfg_pde` - Core MFG solving capabilities

### Optional (Enhanced Features)
- `jax` - GPU acceleration and profiling
- `plotly` - Interactive visualization
- `scipy` - Advanced numerical methods

## 🎖️ **Success Criteria**

### AMR Effectiveness Indicators
- **Mesh Efficiency < 0.7**: AMR uses significantly fewer elements
- **Speedup Ratio > 2.0**: AMR solves faster than uniform grid
- **Accuracy Improvement > 1.5x**: Better solution quality
- **Memory Efficiency > 0.6**: Good memory utilization
- **No Memory Leaks**: Stable repeated execution

### Problem Coverage
- ✅ 1D problems with sharp features
- ✅ 2D structured problems 
- ✅ 2D triangular mesh problems
- ✅ Real-world application scenarios
- ✅ GPU/CPU performance validation

---

**This AMR evaluation suite provides comprehensive validation that adaptive mesh refinement delivers significant computational advantages for MFG problems with localized features and sharp gradients.**
