# Operator Architecture Design

**Status**: ACTIVE
**Created**: 2026-01-24
**Issue**: Operator module reorganization

## Overview

The `mfgarchon/operators/` module provides reusable mathematical operators for PDE solving. This document defines what belongs in `operators/` vs other locations.

## Definition: What is an "Operator"?

An **operator** in this context is:
1. A mathematical map: functions → functions (or arrays → arrays)
2. Has well-defined algebraic properties (linearity, composition)
3. **Reusable** across different solvers/contexts
4. Can be represented as LinearOperator (matrix or matrix-free)

An **algorithm** (does NOT belong in `operators/`):
1. A specific procedure tied to a solver strategy
2. Combines operators with iteration/timestepping
3. Context-dependent (e.g., particle positions, mesh topology)

## Module Structure

```
mfgarchon/operators/
├── differential/          # Grid-based differential operators
│   ├── gradient.py        # PartialDerivOperator
│   ├── laplacian.py       # LaplacianOperator
│   ├── divergence.py      # DivergenceOperator
│   ├── advection.py       # AdvectionOperator
│   └── interface_jump.py  # InterfaceJumpOperator
├── interpolation/         # Grid ↔ point operators
│   ├── interpolation.py   # InterpolationOperator
│   └── projection.py      # GeometryProjector, ProjectionRegistry
├── stencils/              # Low-level FD building blocks
│   └── finite_difference.py
├── reconstruction/        # High-order reconstruction (WENO, ENO)
│   └── weno.py
└── spectral/              # FUTURE: Fourier-based operators
```

## Operator Family Classification

| Family | LinearOperator? | In `operators/`? | Location |
|--------|----------------|------------------|----------|
| **FDM** (grid) | ✅ Sparse | ✅ YES | `operators/differential/` |
| **Spectral/Fourier** | ✅ Dense (FFT) | ✅ YES (future) | `operators/spectral/` |
| **GFDM/Meshfree** | ⚠️ Point-dependent | ❌ NO | `utils/numerical/gfdm_*` |
| **FEM** | ✅ Sparse | 🟡 Future | Requires mesh infra |
| **SPH/Particle** | ❌ Non-linear | ❌ NO | `alg/numerical/` |

## Rationale by Family

### FDM (Grid-Based) → `operators/`
- Fixed stencil coefficients
- Sparse matrix representation
- Reusable across HJB, FP, heat equation, etc.
- Example: `LaplacianOperator` works for any elliptic problem

### Spectral/Fourier → `operators/` (future)
- FFT-based differentiation: ∂u/∂x = F⁻¹(ik · F(u))
- Infinite-order accurate for smooth functions
- O(N log N) complexity
- Natural for periodic BCs
- Composes well (spectral Laplacian = D_x @ D_x)

### GFDM/Meshfree → `utils/numerical/`
- Stencil weights depend on **local point cloud geometry**
- Each point has its own operator (row in matrix)
- Requires neighbor search, Taylor expansion fitting
- Tightly coupled to collocation strategy
- Not a standalone reusable operator

### FEM → Future consideration
- Requires mesh topology (elements, connectivity)
- Basis functions (P1, P2, etc.)
- Quadrature rules
- Would go in `operators/fem/` if added

### SPH/Particle → `alg/numerical/`
- Inherently **non-linear** and **time-dependent**
- Kernel width may adapt
- Neighbor lists change every timestep
- Not a fixed linear operator

## Conceptual Hierarchy

```
Stencils (fixed coefficients)
    ↓
Reconstruction (adaptive weighting: WENO, ENO)
    ↓
Differential Operators (LinearOperator interface)
    ↓
Solvers (combine operators with iteration/timestepping)
```

## Migration Notes

### Deprecated Locations
- `mfgarchon.geometry.operators` → use `mfgarchon.operators`
- `mfgarchon.utils.numerical.tensor_calculus` → use `mfgarchon.operators.stencils`

### Backward Compatibility
Old imports work with deprecation warnings until v0.20.0.

## References
- Issue #595: Operator refactoring
- Issue #606: WENO5 integration
- `docs/development/DEPRECATION_LIFECYCLE_POLICY.md`
