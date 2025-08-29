# Cliff Analysis Archive

This directory contains all scripts and results related to analyzing the "90-degree cliff" phenomenon in QP-collocation long-time simulations.

## Key Files

### Scripts
- `qp_cliff_demonstration.py` - Demonstrates cliff vs stable behavior comparison
- `qp_stability_threshold_test.py` - Systematic study to find stability threshold  
- `qp_stable_long_time_series.py` - Tests T=3,5,10 progression using stable parameters
- `stability_analysis_demo.py` - Theoretical analysis of stability loss mechanisms

### Results
- `qp_cliff_demonstration.png` - Visualization of cliff behavior
- `qp_stable_long_time_series.png` - Long-time stability progression analysis

## Key Findings

**Stability Threshold**: Between T=2 (stable) and T=5 (cliff behavior)

**Root Causes**:
1. Particle clustering → Bandwidth collapse
2. Kernel matrix singularity → QP solver failure  
3. Eigenvalue instability → Exponential error growth
4. Cascade failure → Sudden mass loss cliff

**Why Sudden**: Exponential error growth and critical threshold effects, not gradual degradation.

## Related Documentation
- `docs/issues/90_degree_cliff_analysis.md` - Complete technical analysis
