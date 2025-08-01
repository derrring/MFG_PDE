# MFG Solver Comparison Benchmarks

This directory contains comprehensive comparisons of different **base MFG solution methods** to evaluate their relative strengths, computational efficiency, and solution quality.

## 🎯 **Purpose**

Compare and validate different MFG solving approaches to:
- Establish performance baselines for each method
- Understand method-specific strengths and limitations  
- Guide solver selection for different problem types
- Inform AMR enhancement strategy (see `../amr_evaluation/`)

## 🧮 **Solver Methods Evaluated**

### **1. Pure Finite Difference Method (FDM)**
- **Approach**: Classical FDM for both HJB and Fokker-Planck equations
- **Strengths**: Fast, well-established, simple implementation
- **Limitations**: Fixed resolution, no adaptivity

### **2. Hybrid Particle-FDM**  
- **Approach**: Particle methods for Fokker-Planck + FDM for HJB
- **Strengths**: Balanced performance, good mass conservation
- **Use Case**: General-purpose MFG solving

### **3. QP-Constrained Collocation**
- **Approach**: Constrained optimization with monotonicity preservation
- **Strengths**: Highest accuracy, superior boundary handling
- **Optimization**: Achieved 3.7% QP usage rate (97% acceleration)

## 📊 **Benchmark Components**

### Analysis Scripts
- **`comprehensive_final_evaluation.py`** - Complete three-method evaluation with standardized test problems
- **`fixed_method_comparison.py`** - Standardized comparison framework for new solver validation
- **`qp_optimization_success_summary.py`** - QP solver performance analysis and optimization results

### Documentation & Results
- **`three_method_summary.md`** - Executive summary of FDM vs Hybrid vs QP-Collocation comparison
- **`three_method_analysis.md`** - Detailed technical analysis of method characteristics and performance
- **`comprehensive_evaluation_results.md`** - Complete evaluation results with numerical data
- **`optimization_implementation_summary.md`** - QP optimization implementation details and achievements

## 🚀 **Running Solver Comparisons**

### Complete Method Evaluation
```bash
cd benchmarks/solver_comparisons/
python comprehensive_final_evaluation.py
```

### Quick Solver Validation
```bash
python fixed_method_comparison.py
```

### QP Optimization Analysis
```bash
python qp_optimization_success_summary.py
```

## 📈 **Key Performance Findings**

### Base Method Performance Summary
Based on comprehensive evaluation of three core MFG solution approaches:

**1. Pure FDM Method**
- ✅ Fastest execution for simple problems
- ✅ Well-established numerical methods
- ⚠️ Limited accuracy for complex features
- ⚠️ No adaptive capabilities

**2. Hybrid Particle-FDM**
- ✅ Balanced performance and accuracy
- ✅ Good mass conservation properties
- ✅ Stable coupling between particle/FDM components
- ✅ Moderate computational cost

**3. QP-Constrained Collocation**
- ✅ Highest solution quality and accuracy
- ✅ Superior mass conservation (98% → 1.3% improvement)
- ✅ Monotonicity preservation via QP constraints
- ✅ Excellent boundary condition handling
- ⚠️ Higher computational cost (but optimized to 3.7% QP usage)

### Impact on Current AMR Implementation
These findings directly informed the current AMR enhancement architecture:

1. **Base Solver Selection**: All three methods are now available as base solvers for AMR enhancement
2. **AMR Integration**: The AMR system can enhance any of these base methods
3. **Performance Baseline**: Current AMR benchmarks build on these established performance baselines
4. **Quality Metrics**: Mass conservation and boundary handling insights carry forward to AMR validation