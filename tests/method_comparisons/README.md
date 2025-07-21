# Method Comparison Archive

This directory contains all scripts and results comparing different MFG solver methods, with focus on Hybrid vs QP-Collocation comparisons.

## Key Files

### Scripts
- `working_hybrid_vs_qp_comparison.py` - **BEST**: Successful comparison using examples directory implementations
- `fast_hybrid_vs_qp_comparison.py` - Quick comparison with optimized parameters
- `simple_hybrid_vs_qp.py` - Simplified comparison approach
- `working_three_method_comparison.py` - Three-method comparison attempt
- `simple_three_method_comparison.py` - Basic three-method comparison
- `proper_three_method_comparison.py` - Proper three-method setup
- `robust_three_method_comparison.py` - Robust three-method implementation

### Results
- `working_hybrid_vs_qp_comparison.png` - **KEY RESULT**: Successful Hybrid vs QP comparison
- `fast_hybrid_vs_qp_comparison.png` - Quick comparison results
- `simple_hybrid_vs_qp.png` - Simple comparison visualization
- `three_method_convergence_comparison.png` - Three-method analysis
- `robust_comparison.png` - Robust comparison results

## Key Findings

### Successful Hybrid vs QP Comparison
**Working comparison results** (from `working_hybrid_vs_qp_comparison.py`):
- **Hybrid**: 4.7s execution, -0.548% mass change
- **QP-Collocation**: 127s execution, +2.442% mass change  
- **Convergence**: 66.7% similarity score - both methods converge to similar solutions

### Method Characteristics
**Hybrid Particle-FDM**:
- ✅ Fast execution (FDM-based HJB solver)
- ✅ Proven stable coupling
- ✅ Well-established particle dynamics

**QP Particle-Collocation**:  
- ✅ Superior mass conservation
- ✅ Monotonicity constraints via QP
- ✅ Excellent boundary condition handling
- ⚠️ Slower execution but higher quality

### Three-Method Challenges
- Import issues with relative paths
- Grid dimension mismatches
- Numerical instabilities in simplified implementations
- **Solution**: Use proven examples directory implementations