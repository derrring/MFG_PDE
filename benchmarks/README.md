# MFG_PDE Benchmarking Framework

This directory contains comprehensive benchmarking suites for evaluating different aspects of MFG solver performance, organized by benchmark category for clarity and extensibility.

## 📁 **Benchmark Categories**

### 🚀 **[AMR Evaluation](amr_evaluation/)** - *Primary Focus*
**Adaptive Mesh Refinement effectiveness evaluation**
- **Purpose**: Measure AMR performance gains vs uniform grids
- **Scope**: Performance, accuracy, memory usage, GPU profiling
- **Problems**: 5 real-world MFG scenarios designed for AMR validation
- **Key Metrics**: Speedup ratio, mesh efficiency, accuracy improvement

**Quick Start:**
```bash
cd amr_evaluation/
python comprehensive_benchmark.py  # Complete AMR evaluation suite
```

### 🧮 **[Solver Comparisons](solver_comparisons/)**
**Base MFG solution method comparisons**
- **Purpose**: Compare FDM, Hybrid, and QP-Collocation approaches  
- **Scope**: Raw solver performance, mass conservation, stability
- **Focus**: Establish baselines for AMR enhancement
- **Key Metrics**: Execution time, solution quality, resource usage

**Quick Start:**
```bash
cd solver_comparisons/
python comprehensive_final_evaluation.py  # Three-method comparison
```

### 🔮 **Future Benchmark Categories**
The framework is designed for easy extension. Potential future additions:

#### **🌐 Multi-Scale Methods** (`multi_scale_evaluation/`)
- **Scope**: Homogenization, multi-grid, spectral methods
- **Focus**: Large-scale problem handling and efficiency

#### **🤖 ML-Enhanced Solvers** (`ml_solver_evaluation/`)  
- **Scope**: PINN, neural operators, reinforcement learning integration
- **Focus**: AI-enhanced MFG solving effectiveness

#### **⚡ High-Performance Computing** (`hpc_benchmarks/`)
- **Scope**: MPI, distributed computing, cloud scaling
- **Focus**: Massively parallel MFG problem solving

#### **🔧 Numerical Stability** (`stability_analysis/`)
- **Scope**: Convergence analysis, parameter sensitivity, robustness
- **Focus**: Solution reliability across problem parameters

## 🎯 **Choosing the Right Benchmark**

### For AMR Performance Evaluation
**Use when**: Evaluating adaptive mesh refinement effectiveness
```bash
cd amr_evaluation/
python comprehensive_benchmark.py
```

### For Base Solver Selection  
**Use when**: Choosing optimal solver for your problem type
```bash
cd solver_comparisons/
python comprehensive_final_evaluation.py
```

### For New Method Development
**Use when**: Adding new solver methods to the framework
```bash
cd solver_comparisons/
python fixed_method_comparison.py  # Standardized validation
```

## 📊 **Benchmark Output Structure**

Each benchmark category produces organized results:

```
benchmarks/
├── amr_evaluation/
│   └── comprehensive_benchmark_results/
│       ├── comprehensive_benchmark_report.md     # Executive summary
│       ├── benchmark_summary.json               # Machine-readable
│       ├── performance/                         # Timing analysis
│       ├── accuracy/                           # Error convergence
│       ├── gpu_profiling/                      # GPU/CPU breakdown
│       └── memory_profiling/                   # Memory usage
│
└── solver_comparisons/
    └── evaluation_results/
        ├── method_comparison_report.md          # Solver comparison
        ├── performance_baseline.json           # Raw benchmarks
        └── optimization_analysis/              # QP improvements
```

## 🔧 **Framework Architecture**

### Extensible Design Principles

1. **Category Separation**: Each benchmark type in its own directory
2. **Standardized Interfaces**: Common API patterns across categories  
3. **Modular Components**: Individual benchmark modules can run independently
4. **Unified Reporting**: Consistent output formats and visualization
5. **Future-Proof Structure**: Easy addition of new benchmark categories

### Adding New Benchmark Categories

```bash
# Create new category directory
mkdir benchmarks/new_category/

# Follow established patterns
# - README.md with category description
# - Individual benchmark modules  
# - Comprehensive orchestrator script
# - Results documentation
```

## 💡 **Usage Examples**

### Evaluate AMR on Custom Problem
```python
from amr_evaluation.comprehensive_benchmark import ComprehensiveAMRBenchmark
from mfg_pde import ExampleMFGProblem

# Create problem with sharp features
def create_sharp_problem():
    return ExampleMFGProblem(
        T=1.0, Nx=64, sigma=0.01,  # Low diffusion = sharp features
        coefCT=3.0  # High congestion = localized dynamics
    )

# Evaluate AMR effectiveness
benchmark = ComprehensiveAMRBenchmark()
results = benchmark.performance_suite.benchmark_problem(
    create_sharp_problem, "Sharp_Features_Test",
    [
        {'name': 'Uniform', 'amr_enabled': False},
        {'name': 'AMR', 'amr_enabled': True, 'error_threshold': 1e-4}
    ]
)
```

### Compare Base Solver Methods
```python
from solver_comparisons.comprehensive_final_evaluation import comprehensive_three_method_evaluation

# Run complete method comparison
results = comprehensive_three_method_evaluation()
# Results include FDM, Hybrid, and QP-Collocation performance
```

### Validate New Solver Implementation
```python
from solver_comparisons.fixed_method_comparison import run_standardized_comparison

# Test new solver against established baselines
validation_results = run_standardized_comparison(new_solver_class)
```

## 🔧 **Dependencies**

### Required
- `numpy`, `matplotlib`, `psutil` - Scientific computing and system monitoring
- `mfg_pde` - Core MFG solving capabilities

### Optional (Enhanced Features)
- `jax` - GPU acceleration and AMR profiling
- `plotly` - Interactive visualization
- `scipy` - Advanced numerical methods

## 🚨 **Common Issues & Solutions**

### Setup Issues
- **Import Errors**: Run `pip install -e .` from repository root
- **JAX/GPU Issues**: Ensure proper JAX installation with GPU support
- **Memory Errors**: Reduce problem sizes for limited RAM systems

### Performance Issues
- **Slow First Runs**: JAX compilation time (normal, improves on subsequent runs)
- **Inconsistent Results**: Close other applications, set random seeds
- **Memory Growth**: Run garbage collection between benchmark iterations

## 📞 **Support & Development**

### Getting Help
- **Benchmark Results**: Check generated reports for detailed analysis
- **Console Output**: Review diagnostic information for troubleshooting
- **Documentation**: Consult individual README files in each category
- **Issues**: Report problems in GitHub issues with benchmark output

### Contributing New Benchmarks
1. **Choose Category**: AMR evaluation, solver comparison, or create new category
2. **Follow Patterns**: Use existing benchmark modules as templates
3. **Document Thoroughly**: Include README and result interpretation guides
4. **Test Completely**: Validate on different systems and problem sizes

### Development Guidelines
- **Modular Design**: Each benchmark should run independently
- **Standardized Output**: Follow established reporting formats
- **Error Handling**: Graceful failure with informative messages
- **Performance Aware**: Minimize benchmark overhead

---

## 🎯 **Summary**

This **organized benchmarking framework** provides:

- **🚀 AMR Evaluation**: Comprehensive assessment of adaptive mesh refinement effectiveness
- **🧮 Solver Comparisons**: Baseline performance analysis of different MFG solution methods  
- **🔮 Future Extensibility**: Clean structure for adding new benchmark categories
- **💡 Practical Usage**: Real-world examples and standardized validation tools

**The framework scales from individual solver validation to comprehensive performance analysis, supporting both research and production MFG applications.**

*Organized benchmark structure implemented: August 1, 2025*
