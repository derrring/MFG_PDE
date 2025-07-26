# Benchmarks

This directory contains performance benchmarks, method comparisons, and computational analysis tools for the MFG_PDE framework.

## 📊 Contents

### 🔬 [Method Comparisons](method_comparisons/)
Comprehensive comparisons between different numerical methods:

- **[comprehensive_final_evaluation.py](method_comparisons/comprehensive_final_evaluation.py)** - Complete method evaluation suite
- **[fixed_method_comparison.py](method_comparisons/fixed_method_comparison.py)** - Standardized comparison framework
- **[qp_optimization_success_summary.py](method_comparisons/qp_optimization_success_summary.py)** - QP solver performance analysis

### 📈 Analysis Reports
- **[comprehensive_evaluation_results.md](method_comparisons/comprehensive_evaluation_results.md)** - Detailed evaluation results
- **[optimization_implementation_summary.md](method_comparisons/optimization_implementation_summary.md)** - Implementation insights
- **[three_method_analysis.md](method_comparisons/three_method_analysis.md)** - Multi-method comparison
- **[three_method_summary.md](method_comparisons/three_method_summary.md)** - Summary of findings

## 🎯 Purpose

### **Performance Analysis**
- Compare computational efficiency of different solvers
- Analyze convergence rates and stability
- Measure memory usage and scalability

### **Method Validation**
- Verify numerical accuracy across methods
- Test robustness under different conditions
- Validate theoretical convergence properties

### **Research Support**
- Provide reproducible performance baselines
- Support algorithm development decisions
- Generate publication-quality comparisons

## 🚀 Running Benchmarks

### Individual Method Comparison
```bash
python benchmarks/method_comparisons/comprehensive_final_evaluation.py
```

### Quick Performance Check
```bash
python benchmarks/method_comparisons/fixed_method_comparison.py
```

### QP Solver Analysis
```bash
python benchmarks/method_comparisons/qp_optimization_success_summary.py
```

## 📊 Benchmark Results

### Method Performance Summary
The benchmarks compare these main approaches:
- **HJB-GFDM Tuned QP**: High accuracy, moderate speed
- **Particle Collocation**: Good balance, mass conservation
- **Semi-Lagrangian HJB**: Fast execution, memory efficient

### Key Metrics
- **Accuracy**: L2 error vs analytical solutions
- **Speed**: Runtime for standard problem sizes
- **Stability**: Convergence under various conditions
- **Memory**: Peak memory usage patterns

## 🔧 Adding New Benchmarks

### Benchmark Structure
```python
def benchmark_new_method():
    """
    Standard benchmark template
    """
    # Setup problem
    problem = create_standard_problem()
    
    # Run method
    start_time = time.time()
    solution = solve_with_new_method(problem)
    runtime = time.time() - start_time
    
    # Analyze results
    accuracy = compute_error(solution, analytical)
    
    return {
        'runtime': runtime,
        'accuracy': accuracy,
        'memory_peak': get_peak_memory()
    }
```

### Integration Guidelines
1. Use standardized problem configurations
2. Include error analysis and timing
3. Document method parameters
4. Provide comparison context

## 📈 Benchmark Categories

| Category | Purpose | Files | Update Frequency |
|----------|---------|-------|------------------|
| **Method Comparisons** | Algorithm performance | 3 scripts | With new methods |
| **Analysis Reports** | Results documentation | 4 reports | After benchmarks |
| **Validation Tests** | Accuracy verification | Integrated | Continuous |

## 🔍 Understanding Results

### Interpreting Outputs
- **Lower error values** = Higher accuracy
- **Shorter runtimes** = Better performance  
- **Stable convergence** = Robust method
- **Linear scaling** = Good efficiency

### Common Patterns
- **QP methods**: High accuracy, slower execution
- **Particle methods**: Balanced performance
- **FDM methods**: Fast but less accurate

## 📞 Support

- **Methodology**: See [../docs/theory/](../docs/theory/) for mathematical background
- **Implementation**: Check [../docs/development/](../docs/development/) for coding standards
- **Issues**: Report benchmark problems in GitHub issues

---

*Benchmarks last updated: 2025-07-26*