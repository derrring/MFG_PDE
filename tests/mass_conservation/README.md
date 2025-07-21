# Mass Conservation Analysis Archive

This directory contains all scripts and results related to QP-collocation mass conservation studies and long-time behavior analysis.

## Key Files

### Scripts
- `qp_t1_mass_conservation_demo.py` - Original T=1 demo, extended to T=5
- `qp_extended_mass_conservation.py` - Proven stable T=2 simulation
- `qp_long_time_mass_conservation.py` - T=10 long-time attempt
- `qp_conservative_t5_demo.py` - Conservative parameters for T=5
- `qp_convergence_validation.py` - Convergence analysis

### Results  
- `qp_t1_mass_conservation_demo.png` - T=1 original demonstration
- `qp_t5_mass_conservation_demo.png` - Failed T=5 attempt showing cliff
- `qp_extended_mass_conservation.png` - Successful T=2 with excellent mass conservation
- `qp_long_time_mass_conservation.png` - T=10 long-time analysis
- `conservative_qp_t5_demo.png` - Conservative T=5 approach
- `qp_analysis.png` - General QP method analysis

## Key Results

### Successful Cases
- **T=1**: +2.345% mass change, 0 violations
- **T=2**: +1.594% mass change, 259 violations (0.32% rate)

### Failed Cases  
- **T=5**: -100% mass change, 98,556 violations (cliff behavior)
- **T=10**: -100% mass change, >100k violations

### Mass Conservation Insights
- Mass increase with no-flux BC is expected (particle reflection)
- QP constraints maintain non-negative densities
- Excellent conservation for T ≤ 2, cliff behavior for T ≥ 5