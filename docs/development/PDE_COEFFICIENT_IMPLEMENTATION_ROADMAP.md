# PDE Coefficient Implementation Roadmap

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🔄
**Last Updated**: 2025-11-19
**Branch**: `feature/drift-strategy-pattern`

## Purpose

This document provides **status tracking and task checklists** for implementing flexible drift and diffusion coefficients in MFG_PDE.

**For detailed technical specifications, algorithms, and implementation guides**, see:
- **`PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md`** - Complete implementation spec (1,380 lines)
  - Section 2: Array diffusion algorithms
  - Section 3: Callable evaluation with bootstrap strategy
  - Section 4: MFG coupling integration
  - Section 5: Testing strategy
  - Section 6: Performance analysis
  - Appendices: Math formulations, examples, limitations

---

## Phase 1: Foundation (✅ COMPLETED)

### Deliverables Status

| Feature | Status | Files | Commit |
|:--------|:-------|:------|:-------|
| Unified drift API | ✅ | FP solvers (base + FDM/particle/network) | `9dd182b` |
| Variable diffusion API | ✅ | FP/HJB solvers (all base classes) | `1c26f13` |
| Type protocols | ✅ | `types/pde_coefficients.py` | `dcf1a51` |
| HJB diffusion broadcasting | ✅ | `hjb_solvers/base_hjb.py` | `5cbd263` |
| Coupling compatibility | ✅ | `coupling/fixed_point_iterator.py` | `9dd182b` |

### Implementation Coverage

**FP Solvers**:
- FP-FDM 1D: ✅ Callable diffusion (Phase 2.1-2.2 complete)
- FP-Particle: ✅ Constant diffusion | ⏳ Array/callable diffusion (Phase 2)
- FP-Network: ✅ Variable diffusion

**HJB Solvers**:
- HJB-FDM 1D: ✅ Array diffusion | ⏳ Callable (Phase 2.2)
- HJB-FDM nD: ⏳ Phase 2.4
- Other HJB: ⏳ Phase 2 (API added, implementation pending)

**Key Commits**: 7 major commits
1. `9dd182b` - Unified drift+diffusion API in FP solvers
2. `1c26f13` - Added diffusion_field to HJB solvers
3. `dcf1a51` - Type protocols for state-dependent coefficients
4. `5cbd263` - Simplified HJB diffusion_field broadcasting
5. `36730de` - Array diffusion in FP-FDM solver (Phase 2.1)
6. `c82bfcf` - Callable diffusion in FP-FDM solver (Phase 2.2)
7. (pending) - MFG coupling integration (Phase 2.3)

---

## Phase 2: State-Dependent & nD (🔄 IN PROGRESS)

**Design Doc**: `PHASE_2_DESIGN_STATE_DEPENDENT_COEFFICIENTS.md` (see for algorithms)

### 2.1: Array Diffusion in FP Solvers (✅ COMPLETED)

**Priority**: High | **Effort**: 1 day | **Status**: ✅ Complete | **Commit**: `36730de`

**Completed Tasks**:
- [x] Remove NotImplementedError for array diffusion in `FPFDMSolver`
- [x] Add diffusion array indexing in matrix assembly (`fp_fdm.py:_solve_fp_1d()`)
- [x] Add 5 unit tests (spatially varying, spatiotemporal, advection, mass conservation, validation)
- [x] Update docstrings with array diffusion examples

**Test Coverage**: Spatial σ(x), spatiotemporal σ(t,x), combined with advection, error handling

### 2.2: Callable Evaluation - FP Side (✅ COMPLETED)

**Priority**: High | **Effort**: 1 day | **Status**: ✅ Complete | **Commit**: `c82bfcf`

**Completed Tasks (FP-FDM)**:
- [x] Add `_solve_fp_1d_with_callable()` method (bootstrap strategy)
- [x] Add `_validate_callable_output()` helper (shape, NaN/Inf checking)
- [x] Route callable to appropriate solver in `solve_fp_system()`
- [x] 6 unit tests: porous medium, crowd dynamics, combined, validation

**Test Coverage**: D(m) = σ²m, D(m) = D₀+D₁(1-m/m_max), scalar return, error handling

**Remaining (HJB-FDM)**: ⏳
- [ ] Add callable evaluation in `solve_hjb_system_backward()`
- [ ] Re-evaluate per Picard iteration with M_density
- [ ] Unit tests: state-dependent diffusion

### 2.3: MFG Coupling Integration (✅ COMPLETED)

**Priority**: High | **Effort**: 1 day | **Status**: ✅ Complete | **Commit**: (pending)

**File**: `coupling/fixed_point_iterator.py`

**Completed Tasks**:
- [x] Add `diffusion_field` and `drift_field` to `__init__()`
- [x] Add introspection-based parameter passing to solvers
- [x] Pass coefficients to HJB and FP solvers appropriately
- [x] Integration tests: MFG with callable and array coefficients

**Test Coverage**: Array diffusion working end-to-end in MFG, callable tests marked xfail pending HJB support

**Note**: Infrastructure complete. Array diffusion verified working in MFG. Callable diffusion requires HJB callable support (Phase 2.2 HJB side, remaining work)

### 2.4: Complete nD Support

**Priority**: High | **Effort**: 1-2 weeks | **Status**: ⏳ Deferred

**Modules Required**: See Design Doc Section 2.5

**Tasks**:
- [ ] Create `utils/numerical/operators_nd.py` (gradient_nd, laplacian_nd, divergence_nd)
- [ ] Implement `HJBFDMSolver._solve_hjb_nd()` with variable diffusion
- [ ] Create `FPFDMSolver._solve_fp_nd()`
- [ ] nD Hamiltonian evaluation
- [ ] nD boundary conditions
- [ ] Unit tests: 2D, 3D problems
- [ ] Performance benchmarks

### 2.5: Anisotropic Diffusion Tensors

**Priority**: Medium | **Effort**: 3-5 days | **Status**: ⏳ After nD

**Tasks**:
- [ ] Extend operators to handle diffusion tensors D(x)
- [ ] Implement `divergence_tensor_diffusion()`
- [ ] Update FP/HJB solvers to accept tensor diffusion_field
- [ ] Unit tests: diagonal tensors, cross-diffusion
- [ ] Example: anisotropic crowd dynamics

---

## Phase 3: Advanced Features (🔮 FUTURE)

**Features**:
- 3.1: Lévy Processes (Jump-Diffusion)
- 3.2: Common Noise (Extended MFG)
- 3.3: Fractional Diffusion

**Status**: Planning only

---

## Success Metrics

### Phase 2 Goals

- [ ] 90%+ test coverage for new features
- [ ] <10% performance overhead for callable evaluation
- [ ] nD solvers validated against analytical solutions
- [ ] Examples run successfully in CI
- [ ] Documentation complete and reviewed

### Performance Targets

- Callable evaluation: <2x slowdown vs arrays
- nD solvers: Scale as O(N^d) where d = dimension
- Memory: <3x overhead for nD vs 1D (per point)

---

## Quick Reference

### Current API

```python
# Array diffusion (Phase 2.1)
M = solver.solve_fp_system(m0, drift_field=U, diffusion_field=sigma_array)

# Callable coefficients (Phase 2.2)
def porous_medium(t, x, m):
    return 0.1 * m

M = solver.solve_fp_system(m0, diffusion_field=porous_medium)

# MFG with callable (Phase 2.3)
coupling = FixedPointIterator(problem, hjb, fp, diffusion_field=porous_medium)
result = coupling.solve()
```

### Type Protocols

See `mfg_pde/types/pde_coefficients.py`:
- `DriftCallable`: α(t, x, m) -> drift vector
- `DiffusionCallable`: D(t, x, m) -> diffusion coefficient/tensor

---

**Next Action**: Complete Phase 2.2 HJB side (callable support in HJB solvers) OR proceed to Phase 2.4 (nD support)
**Questions**: GitHub issues or Design Doc
