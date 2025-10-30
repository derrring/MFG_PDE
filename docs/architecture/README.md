# MFG_PDE Architecture Documentation

**Last Updated**: 2025-10-30
**Status**: Experimental Beta - Refactoring Planned

This directory contains comprehensive architecture audit and refactoring proposals for MFG_PDE based on intensive research usage over 3 weeks.

---

## Status: Experimental Beta

Both MFG_PDE and mfg-research are under active development. This documentation supports a planned architectural refactoring based on real-world research pain points.

**GitHub Issue**: [#200 - Architecture Refactoring: Unified Problem Classes and Multi-Dimensional Solver Support](https://github.com/derrring/MFG_PDE/issues/200)

**Issue Summary**: See `ARCHITECTURE_ISSUE_CREATED.md` for details on the GitHub issue creation and current status.

---

## Quick Start

### For Maintainers

**30-Second Overview**:
1. Read: `audit/AUDIT_ENRICHMENT_SUMMARY.md` (5 min)
2. Scan: `navigation/ARCHITECTURE_DOCUMENTATION_INDEX.md` (2 min)
3. Immediate action: Fix Bug #15 and Anderson (2 weeks)

**Full Understanding** (3-4 hours):
1. Executive Summary: `audit/AUDIT_ENRICHMENT_SUMMARY.md` (15 min)
2. Full Audit: `audit/ARCHITECTURE_AUDIT_ENRICHMENT.md` (2 hours)
3. Refactoring Plan: `proposals/MFG_PDE_ARCHITECTURE_REFACTOR_PROPOSAL.md` (1 hour)
4. Evidence samples from `evidence/` directory (30 min)

### For Researchers

**Understanding Current State**:
- Problem class confusion? See `audit/MFG_PDE_ARCHITECTURE_AUDIT.md` Section 1
- FDM limitations? See `evidence/FDM_SOLVER_LIMITATION_ANALYSIS.md`
- Integration challenges? See `audit/AUDIT_ENRICHMENT_SUMMARY.md` "Integration Efficiency"

---

## Key Findings Summary

### Critical Statistics

**Research Data Analyzed**:
- 251 Python files in MFG_PDE codebase
- 181 markdown documentation files from research
- 94 Python test/experiment files
- 3 weeks of intensive research sessions
- 2 GitHub issues filed
- 3 critical bugs discovered

**Quantitative Impact**:
- **45 hours** lost to architecture workarounds (3 weeks)
- **3,285 lines** of duplicate/adapter code written
- **7.6x integration overhead** (38h actual vs 5h expected)
- **1 bug per week** discovery rate
- **780 hours/year** projected annual cost

### Architecture Issues Documented

**48 total issues** across 8 categories:
- 6 CRITICAL severity issues
- 15 HIGH severity issues
- 18 MEDIUM severity issues
- 9 LOW severity issues

**Blocked Features**:
1. Pure FDM solver for 2D problems (PERMANENTLY BLOCKED)
2. GPU acceleration via backends (not wired)
3. Hybrid FDM-HJB + Particle-FP (type mismatch)
4. ~~Anderson-accelerated MFG~~ - UNBLOCKED (2025-10-30)
5. ~~QP-constrained particle collocation~~ - UNBLOCKED (2025-10-30)

---

## Repository Component Inventory

### Core Problem Classes (5)

1. **MFGProblem** - 1D standard MFG (28,296 bytes)
2. **HighDimMFGProblem** - Abstract base for d>=2 (18,490 bytes)
3. **GridBasedMFGProblem** - Regular grids 2D/3D (in highdim_mfg_problem.py)
4. **NetworkMFGProblem** - Graph-based MFG (22,479 bytes)
5. **VariationalMFGProblem** - Lagrangian formulation (23,242 bytes)

**Compatibility**: Only **4 out of 25** HJB×FP combinations work natively

### HJB Solvers (7)

1. **BaseHJBSolver** - Abstract base class
2. **HJBFDMSolver** - Finite Difference Method (1D ONLY)
3. **HJBGFDMSolver** - Generalized FDM (meshfree, nD support)
4. **HJBSemiLagrangianSolver** - Semi-Lagrangian method
5. **HJBWenoSolver** - WENO scheme
6. **NetworkHJBSolver** - Network/graph HJB
7. **NetworkPolicyIterationHJBSolver** - Policy iteration variant

**Dimension Support**:
- 1D: All solvers
- 2D/3D: GFDM, Semi-Lagrangian, WENO (partial)
- Network: NetworkHJBSolver only

### FP Solvers (4)

1. **BaseFPSolver** - Abstract base class
2. **FPFDMSolver** - Finite Difference Method (1D ONLY)
3. **FPParticleSolver** - Particle-based method (nD support)
4. **FPNetworkSolver** - Network/graph FP

**Dimension Support**:
- 1D: FDM, Particle, Network
- 2D/3D: Particle only

### MFG Coupled Solvers (6)

Located in `mfg_pde/alg/numerical/mfg_solvers/`:
1. **BaseMFGSolver** - Base class for coupled MFG
2. **FixedPointIterator** - Picard iteration infrastructure
3. **FixedPointUtils** - Utilities for convergence
4. **HybridFPParticleHJBFDM** - Hybrid solver (1D only)
5. **NetworkMFGSolver** - Network MFG solver
6. **VariationalMFGSolver** - Variational formulation

### Neural Solvers (29 files)

**Categories**:
- **DGM** (Deep Galerkin Method): 6 files
- **PINN** (Physics-Informed Neural Networks): 4 files
- **Operator Learning** (DeepONet, FNO): 5 files
- **NN Architectures**: 5 files
- **Core Neural**: 4 files
- **Stochastic Neural**: Small subsystem

**Key Components**:
- Loss functions, training loops, sampling strategies
- Variance reduction techniques
- Network architectures (feedforward, residual, modified MLP)

### Optimization Methods (11 files)

Located in `mfg_pde/alg/optimization/`:
- **Variational Methods**: 2 files
- **Primal-Dual**: Subdirectory
- **Augmented Lagrangian**: Subdirectory
- **Optimal Transport**: 3 files

### Reinforcement Learning (45 files)

Located in `mfg_pde/alg/reinforcement/`:
- Comprehensive RL integration
- Multiple algorithm implementations
- MFG-RL bridges

### Backends (9 files)

1. **BaseBackend** - Abstract backend interface
2. **NumPyBackend** - NumPy implementation
3. **JAXBackend** - JAX acceleration
4. **TorchBackend** - PyTorch integration
5. **NumbaBackend** - Numba JIT
6. **ArrayWrapper** - Backend-agnostic arrays
7. **SolverWrapper** - Solver backend integration
8. **Compat** - Compatibility layer

**Status**: Exists but not wired through factory system

### Factory System (5 files)

1. **SolverFactory** - Main solver creation (17 functions)
2. **PydanticSolverFactory** - Type-validated creation
3. **GeneralMFGFactory** - Problem creation from configs
4. **BackendFactory** - Backend creation (not integrated)

**Key Functions**:
- `create_solver()` - Main entry point
- `create_fast_solver()` - Quick setup
- `create_accurate_solver()` - High-precision
- `create_research_solver()` - Research-oriented
- `create_semi_lagrangian_solver()` - Specific method
- 12 more specialized factories

### Configuration System (7 files)

**Three Competing Systems**:
1. **Legacy**: `solver_config.py` (deprecated warnings)
2. **Pydantic**: `pydantic_config.py` (type validation)
3. **OmegaConf**: `omegaconf_manager.py` (YAML-based)

**Additional**:
- `modern_config.py` - Newer interface
- `structured_schemas.py` - Schema definitions
- `array_validation.py` - Array type checking

**Issue**: Unclear which to use, migration path undocumented

### Geometry System (23 files)

**Core**:
- `BaseGeometry` - Abstract base
- `Domain1D`, `Domain2D`, `Domain3D` - Standard domains
- `TensorProductGrid` - Cartesian grids
- `SimpleGrid` - Basic grid implementation

**Advanced**:
- AMR (Adaptive Mesh Refinement): 4 files (1D, 2D quadtree, 2D triangular, 3D tetrahedral)
- Implicit geometry: 5 files (CSG operations, hyperrectangle, hypersphere)
- Network geometry: 2 files
- Boundary conditions: 4 files (1D, 2D, 3D, manager)

**Total**: 18 geometry files + 5 boundary files

### Utilities (40 files)

**Categories**:
- **Numerical**: Anderson acceleration, autodiff, convergence, integration, MCMC (10 files)
- **Acceleration**: JAX utils, Torch utils (2 files)
- **Logging**: Logger, decorators, analysis (4 files)
- **IO**: HDF5 utilities (2 files)
- **Data**: Polars integration, validation (3 files)
- **Neural**: Normalization (2 files)
- **Notebooks**: Pydantic integration, reporting (3 files)
- **Performance**: Monitoring, optimization (3 files)
- **Core Utils**: aux_func, exceptions, dependencies, CLI (11 files)

**Key Missing** (discovered by research):
- Particle interpolation utilities
- SDF (signed distance function) helpers
- Convergence monitoring utilities
- QP result caching

### Examples (60+ files)

Located in `examples/`:
- Basic examples
- Advanced examples (30+ subdirectories)
- Specific demonstrations (anisotropic crowd, epidemic, portfolio, etc.)

**Status**: Many work, some broken, unclear which are canonical

---

## Critical Bugs Discovered

### Bug #13: Gradient Key Mismatch
**File**: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py`
**Symptom**: Hamiltonian receives wrong gradient format
**Root Cause**: Used `'grad_u_x'` instead of `'dpx'` (2 characters difference)
**Impact**: 3 days debugging, entire MFG system broken
**Status**: FIXED
**Evidence**: 15 documents in `docs/archived_bug_investigations/BUG_13_INDEX.md`

### Bug #14: GFDM Gradient Sign Error
**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:453`
**Symptom**: Gradients approximately negated
**Root Cause**: `b = u_center - u_neighbors` (should be reversed)
**Impact**: Agents move away from goals, MFG doesn't converge
**Status**: FIXED, merged to MFG_PDE, GitHub issue filed
**Evidence**: `experiments/maze_navigation/archives/bugs/bug14_gfdm_sign/`

### Bug #15: QP Sigma Type Error
**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:818`
**Symptom**: `TypeError: 'method' and 'int'` in QP monotonicity check
**Root Cause**: Code expects numeric `sigma`, particle methods use callable `sigma(x)`
**Impact**: Cannot use QP constraints without workaround
**Status**: FIXED (2025-10-30) - Comprehensive regression tests passing
**Evidence**: `experiments/maze_navigation/archives/bugs/bug15_qp_sigma/`, `tests/test_bug15_sigma_fix.py`

### Anderson Accelerator 1D Limitation
**File**: `mfg_pde/utils/numerical/anderson_acceleration.py`
**Symptom**: `IndexError` when passing 2D arrays
**Root Cause**: `np.column_stack` assumes 1D input
**Impact**: Requires 25 lines boilerplate per usage
**Status**: FIXED (2025-10-30) - Multi-dimensional support with flatten/reshape pattern
**Evidence**: `experiments/maze_navigation/ANDERSON_ISSUE_POSTED.md`, `tests/test_anderson_multidim.py`

---

## Refactoring Timeline

**Immediate (2 weeks)**:
1. ~~Fix Bug #15 (QP sigma API)~~ - COMPLETE (2025-10-30)
2. ~~Fix Anderson multi-dimensional~~ - COMPLETE (2025-10-30)
3. ~~Standardize gradient notation~~ - COMPLETE (Phase 2 & 3)

**Short-term (3 months)**:
1. Implement 2D/3D FDM solvers - 4-6 weeks
2. Add missing utilities - 4 weeks
3. Quick wins (return format, monitor) - 1 week

**Long-term (6-9 months)**:
1. Unified problem class - 8-10 weeks
2. Configuration simplification - 2-3 weeks
3. Backend integration - 2 weeks

**Total Estimated**: 22-37 weeks (5.5 to 9 months)

---

## Organization

### Directory Structure

```
docs/architecture/
├── README.md                      # This file
├── audit/                         # Complete audits with evidence
│   ├── AUDIT_ENRICHMENT_SUMMARY.md           # 5-min executive summary
│   ├── ARCHITECTURE_AUDIT_ENRICHMENT.md      # 45-page full audit
│   ├── MFG_PDE_ARCHITECTURE_AUDIT.md         # Original audit
│   └── ARCHITECTURE_AUDIT_SUMMARY.md         # Original summary
├── proposals/                     # Refactoring proposals
│   └── MFG_PDE_ARCHITECTURE_REFACTOR_PROPOSAL.md
├── evidence/                      # Supporting evidence
│   ├── FDM_SOLVER_LIMITATION_ANALYSIS.md
│   └── research_findings/         # Symlinks to research artifacts
└── navigation/                    # Documentation indexes
    └── ARCHITECTURE_DOCUMENTATION_INDEX.md
```

### Reading Paths

**Path 1: Executive (15 minutes)**
1. This README
2. `audit/AUDIT_ENRICHMENT_SUMMARY.md`
3. `proposals/` key recommendations

**Path 2: Technical (3-4 hours)**
1. `audit/ARCHITECTURE_AUDIT_ENRICHMENT.md` (full 45 pages)
2. `evidence/FDM_SOLVER_LIMITATION_ANALYSIS.md`
3. `proposals/MFG_PDE_ARCHITECTURE_REFACTOR_PROPOSAL.md`

**Path 3: Debugging (as needed)**
1. `navigation/ARCHITECTURE_DOCUMENTATION_INDEX.md` (find relevant section)
2. Jump to specific evidence file
3. Trace to source code

---

## Evidence Quality

**Documentation Trail**:
- 181 markdown files analyzed
- 15 documents for Bug #13
- 10 documents for Bug #14
- 7 documents for Bug #15
- 99 archived documents in maze_navigation alone
- Complete session logs for 3 weeks

**Code Evidence**:
- 251 Python files in MFG_PDE codebase inventoried
- 94 test/experiment files analyzed
- Every claim backed by file:line references

**Quantification**:
- Time measurements (45 hours total)
- Code line counts (3,285 duplicate lines)
- Efficiency ratios (7.6x overhead)
- Bug discovery rate (1 per week)

---

## Impact on Research

**Projected Annual Cost**:
- 45 hours lost per 3 weeks
- 52 weeks/year → **780 hours/year** lost to workarounds
- That's **19.5 work-weeks** or **nearly 5 months**

**Research Blocked**:
- Cannot provide FDM baseline for papers
- Cannot easily compare solver methods
- GPU acceleration unavailable
- Constant debugging instead of research

**Papers Delayed**:
- Maze navigation: 2 weeks (Bug #13)
- QP validation: 1 week (Bug #15)
- Anisotropic crowd: 1 week (adaptive bug)

---

## Next Steps

### For Maintainers

**Week 1-2: Immediate Fixes**
1. Review Bug #15 fix in `audit/AUDIT_ENRICHMENT_SUMMARY.md`
2. Apply 5-line patch to `hjb_gfdm.py`
3. Fix Anderson accelerator (10-line patch)
4. Write regression tests

**Month 1-3: High-Priority Items**
1. Begin FDM 2D/3D implementation
2. Add missing utilities (interpolation, SDF)
3. Implement quick wins

**Month 4-9: Systematic Refactoring**
1. Follow 6-phase migration plan in proposals/
2. Maintain backward compatibility
3. Gradual rollout with deprecation warnings

### For Researchers

**Workarounds Available**:
- QP sigma: Use `SmartSigma` wrapper class
- Anderson 2D: Manual flatten/reshape
- FDM 2D: Use GFDM instead
- Particle interpolation: Custom utility functions

**Report Issues**:
- File GitHub issues with MFG_PDE repository
- Reference this architecture documentation
- Provide minimal reproducible examples

---

## Related Documentation

**In MFG_PDE**:
- `docs/DOCUMENTATION_POLICY.md` - Official documentation standards
- `CLAUDE.md` - Development preferences (if exists)

**In mfg-research**:
- `CLAUDE.md` - Research workflow and standards
- `experiments/maze_navigation/` - Extensive research artifacts
- `docs/archived_bug_investigations/` - Bug investigation archives

---

## Conclusion

**The architecture audit is evidence-based, quantified, and actionable.**

**Key Takeaways**:
1. Architecture issues are real and measurable (45 hours, 3,285 lines)
2. Three critical bugs discovered through research usage
3. Five features permanently or temporarily blocked
4. Clear refactoring path with 2-week immediate actions
5. Long-term vision maintains backward compatibility

**Status**: Ready for maintainer review and prioritization

---

**Last Updated**: 2025-10-30
**Audit Team**: Research usage over 3 weeks (2025-10-06 to 2025-10-30)
**Contact**: See GitHub issues #199, PR #197 for active discussions
