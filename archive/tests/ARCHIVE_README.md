# Tests Archive

This directory contains organized archives of all test files, analysis scripts, and results generated during the comprehensive study of MFG solver methods, with particular focus on the QP-collocation method and the "90-degree cliff" phenomenon.

## Directory Structure

### `cliff_analysis/`
**90-degree cliff phenomenon investigation**
- Scripts for demonstrating and analyzing sudden mass loss
- Stability threshold determination
- Theoretical analysis of failure mechanisms
- **Key finding**: Cliff occurs due to cascade failure (clustering → bandwidth collapse → matrix singularity → particle explosion)

### `mass_conservation/`  
**Mass conservation studies and long-time behavior**
- T=1 to T=10 mass conservation demonstrations
- Extended time horizon analysis
- QP method mass conservation validation
- **Key finding**: Excellent conservation for T≤2, cliff behavior for T≥5

### `method_comparisons/`
**Hybrid vs QP-Collocation method comparisons**
- Successful working comparisons using examples directory
- Three-method comparison attempts and challenges
- Performance and accuracy analysis
- **Key finding**: Both methods converge to similar solutions, QP superior for mass conservation

### `stability_analysis/`
**Numerical stability analysis and parameter tuning**
- Stability tests and diagnostic scripts
- Parameter optimization studies
- Implementation debugging and validation
- **Key finding**: Parameter sensitivity critical for long-time stability

## Key Results Summary

### Successful Demonstrations
- ✅ **T=2 Extended Simulation**: +1.59% mass change, excellent stability
- ✅ **Hybrid vs QP Comparison**: 66.7% similarity, both methods work well
- ✅ **QP Mass Conservation**: Superior boundary handling and monotonicity

### Critical Discoveries
- ⚠️ **Stability Threshold**: Between T=2 (stable) and T=5 (cliff)
- ⚠️ **Cliff Mechanism**: Sudden catastrophic failure, not gradual degradation
- ⚠️ **Parameter Sensitivity**: Small changes dramatically affect long-time behavior

### Method Validation
- **QP-Collocation**: Best for mass conservation, excellent boundary handling
- **Hybrid Particle-FDM**: Fast execution, proven stability for moderate T
- **Examples Directory**: Most reliable implementations

## Documentation
- `docs/issues/90_degree_cliff_analysis.md` - Complete technical analysis
- Individual README files in each subdirectory
- Original documentation preserved alongside results

## Usage Notes
- Start with `method_comparisons/working_hybrid_vs_qp_comparison.py` for reliable comparison
- Use `mass_conservation/qp_extended_mass_conservation.py` for stable T=2 demonstration  
- Refer to `cliff_analysis/` for understanding long-time instability
- Check `stability_analysis/` for parameter tuning guidance

---
**Archive Created**: 2025-01-20  
**Total Files Archived**: 40+ scripts, 20+ result images  
**Analysis Period**: Comprehensive MFG solver investigation  
**Status**: Complete - All test materials organized and documented
