# PINN Adaptive Features Archive

This directory contains advanced adaptive training features that were extracted from the duplicate `mfg_pde/alg/neural/pinn/` directory during consolidation.

## Archived Features

### `adaptive_training.py`
Advanced adaptive training strategies for Physics-Informed Neural Networks including:

- **Physics-Guided Sampling**: Adaptive point generation based on residual analysis
- **Curriculum Learning**: Progressive difficulty in training
- **Multi-Scale Training**: Hierarchical resolution progression
- **Loss Balancing**: Dynamic weight adjustment for loss components
- **Residual-Based Refinement**: Targeted sampling in high-error regions

## Integration Status

These features were archived during PINN consolidation (September 2025) when we chose to keep `pinn_solvers/` as the primary PINN implementation due to:

1. More comprehensive solver coverage (HJB, FP, MFG)
2. Current integration with main neural module
3. Production-ready implementation

## Future Development

These adaptive training strategies could be integrated into `pinn_solvers/` in future development to enhance PINN training capabilities. Key integration points:

- Extend `PINNBase` with adaptive sampling methods
- Add curriculum learning to training managers
- Implement multi-scale training workflows
- Integrate residual-based refinement

## Archive Date
September 30, 2025

## Reason for Archival
Directory consolidation - removed duplicate `pinn/` directory while preserving valuable adaptive training algorithms for future integration.
