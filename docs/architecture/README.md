# MFG_PDE Architecture Documentation

**Last Updated**: 2025-12-14
**Current Version**: v0.16.7
**Status**: Architecture stabilized, active development continues

---

## Overview

This directory contains architecture documentation for MFG_PDE. Historical audit documents and completed phase plans have been archived to `docs/archive/historical/`.

## Current Architecture

### Core Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Problem Classes** | `mfg_pde/core/` | MFGProblem, MFGComponents |
| **HJB Solvers** | `mfg_pde/alg/numerical/hjb_solvers/` | FDM, GFDM, Semi-Lagrangian, WENO |
| **FP Solvers** | `mfg_pde/alg/numerical/fp_solvers/` | FDM, Particle methods |
| **Coupling** | `mfg_pde/alg/numerical/coupling/` | Fixed-point iteration |
| **Geometry** | `mfg_pde/geometry/` | Grids, meshes, boundaries |
| **Configuration** | `mfg_pde/config/` | Pydantic + OmegaConf dual system |
| **Convergence** | `mfg_pde/utils/convergence/` | Protocol-based checkers |

### Solver Ecosystem

**HJB Solvers** (7):
- `HJBFDMSolver` - Finite Difference (1D/nD)
- `HJBGFDMSolver` - Generalized FDM (meshfree)
- `HJBSemiLagrangianSolver` - Semi-Lagrangian method
- `HJBWenoSolver` - WENO scheme
- Neural solvers (DGM, PINN)

**FP Solvers** (4):
- `FPFDMSolver` - Finite Difference
- `FPParticleSolver` - Particle methods
- Network solvers

**Coupling Methods**:
- Fixed-point (Picard) iteration
- Policy iteration (planned)

### Recent Architecture Changes (v0.16.x)

1. **Convergence Module Consolidation** (v0.16.0)
   - Moved from `mfg_pde/alg/numerical/convergence/` to `mfg_pde/utils/convergence/`
   - Protocol-based checkers: `HJBConvergenceChecker`, `FPConvergenceChecker`, `MFGConvergenceChecker`
   - `ConvergenceConfig` for unified configuration

2. **BaseSolver Integration** (v0.16.7)
   - `BaseSolver` accepts `ConvergenceConfig`
   - Hooks pattern for iteration callbacks
   - Extensible via `_create_convergence_checker()` hook

3. **AMR Removal** (v0.16.x)
   - Adaptive Mesh Refinement removed (not production-ready)
   - Archived to `docs/archive/amr_removed_2025-12/`

---

## Current Documents

| Document | Purpose |
|----------|---------|
| `dimension_agnostic_solvers.md` | Overview of nD solver approaches |
| `proposals/DIMENSION_AGNOSTIC_FDM_ANALYSIS.md` | FDM extension analysis |
| `proposals/MFG_PDE_ARCHITECTURE_REFACTOR_PROPOSAL.md` | Long-term refactoring plan |

## Archived Documents

Historical phase completion summaries, verification checklists, and old proposals have been moved to `docs/archive/historical/`. These include:
- Phase 2 FDM completion summary
- Phase 2/3 verification checklists
- Old issue analyses
- Short-term plans from 2025-10

---

## Open Design Discussions

### Issue #476: Coupling Module Design

Active discussion on coupling module architecture:
- Result standardization (`MFGResult = HJBResult + FPResult`)
- Module structure (`solvers/mfg/` vs `alg/numerical/coupling/`)
- Convergence integration

See: https://github.com/derrring/MFG_PDE/issues/476

---

## Development Guidelines

### Adding New Solvers

1. Inherit from appropriate base class (`BaseSolver`, `BaseHJBSolver`, `BaseFPSolver`)
2. Implement required abstract methods
3. Override `_create_convergence_checker()` for custom convergence
4. Add to factory system
5. Write tests

### Configuration Pattern

```python
from mfg_pde.utils.convergence import ConvergenceConfig

config = ConvergenceConfig(
    tolerance=1e-6,
    max_iterations=100,
    require_both=True  # Require both U and M convergence
)

solver = MySolver(convergence=config)
result = solver.solve(problem)
```

---

## Related Documentation

- `docs/development/guides/` - Developer guides
- `docs/user/` - User documentation
- `docs/theory/` - Mathematical background
- `CLAUDE.md` - Development conventions

---

**Contact**: See GitHub issues for active discussions
