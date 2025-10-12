# Advanced Examples

This directory contains advanced demonstrations and test scripts for the MFG_PDE package.

## Contents

### Demonstrations

**Multi-Population and Networks**:
- `network_mfg_comparison_example.py` - Comparison of network MFG solution methods

**Crowd Dynamics**:
- `anisotropic_crowd_dynamics_2d/` - 2D anisotropic crowd dynamics examples
  - `room_evacuation_two_doors.py` - Room evacuation with two exit doors
  - `numerical_demo.py` - Numerical validation demonstration

**Hybrid Methods**:
- `hybrid_fp_particle_hjb_fdm_demo.py` - Hybrid FP particle + HJB FDM solver

### Test and Verification Scripts

**QP Particle-Collocation Verification**:
- `test_m_matrix_verification.py` - Comprehensive M-matrix property verification test
  - Tests QP-constrained particle-collocation method
  - Measures empirical M-matrix satisfaction rates
  - Generates visualization plots
  - **Usage**: `python examples/advanced/test_m_matrix_verification.py`
  - **Output**: PNG plots in `examples/outputs/advanced/`

- `debug_fd_weights.py` - Debug utility for finite difference weight extraction
  - Tests FD weight computation from Taylor matrices
  - Helps diagnose weight extraction issues
  - **Usage**: `python examples/advanced/debug_fd_weights.py`

## Output Directory

Generated outputs are saved to `examples/outputs/advanced/` (gitignored by default).

## Running Examples

All examples can be run directly:

```bash
# Run M-matrix verification test
python examples/advanced/test_m_matrix_verification.py

# Run hybrid method demo
python examples/advanced/hybrid_fp_particle_hjb_fdm_demo.py

# Run network comparison
python examples/advanced/network_mfg_comparison_example.py
```

## Dependencies

Some examples require optional dependencies:
- PyTorch for neural network methods
- Plotly for interactive visualizations
- NetworkX for network MFG problems

See `requirements-optional.txt` for full list of optional dependencies.

## Documentation

For theoretical background on QP particle-collocation methods, see:
- `docs/theory/` - Mathematical formulations
- `docs/development/private/` - Implementation notes (if you have access)

## Contributing

When adding new advanced examples:
1. Include comprehensive docstrings
2. Add entry to this README
3. Use `examples/outputs/advanced/` for generated files
4. Follow the naming convention: `<feature>_<method>_<description>.py`
