# Incremental Evolution Plan: Dimension-Agnostic MFG Architecture

**Issue**: #245 - Radical Architecture Renovation for v1.0
**Status**: Phase 1 - Protocol Foundation (IN PROGRESS)
**Approach**: Incremental Evolution (safer than branch rebase)
**Last Updated**: 2025-11-09

---

## Executive Summary

The `feature/unified-nd-mfg-problem` branch created excellent dimension-agnostic architecture but diverged from main (4 days, 358 files changed). Instead of painful rebase, we're extracting good ideas and applying them incrementally to current main, preserving recent work (AMR GeometryProtocol, naming conventions, documentation cleanup).

**Key Decision**: Evolve current `mfg_problem.py` incrementally rather than replace it wholesale.

---

## Phase 1: Protocol Foundation ✅ IN PROGRESS

**Goal**: Establish type-safe interface for dimension-agnostic solvers

### ✅ Completed

1. **Created `mfg_pde/core/base_problem.py`**
   - `MFGProblemProtocol` defining solver interface
   - Runtime checkable with `isinstance(problem, MFGProblemProtocol)`
   - Documents required attributes and methods

2. **Exported from `mfg_pde/core/__init__.py`**
   - Available as `from mfg_pde.core import MFGProblemProtocol`

### 🔄 Next Steps (Phase 1 Completion)

3. **Add missing Protocol-required properties to `MFGProblem`**
   - [ ] `grid_spacing: list[float]` property
   - [ ] Ensure `dimension` is always set (not just nD mode)
   - [ ] Ensure `grid_shape` is always a tuple

4. **Verify Protocol Compliance**
   - [ ] Create simple test: `assert isinstance(MFGProblem(...), MFGProblemProtocol)`
   - [ ] Test with 1D, 2D, 3D problems

5. **Update Type Hints in Solvers**
   - [ ] Change `problem: MFGProblem` → `problem: MFGProblemProtocol` in:
     - `alg/numerical/hjb_solvers/base_hjb.py`
     - `alg/numerical/fp_solvers/base_fp.py`
     - `alg/numerical/coupling/fixed_point_iterator.py`

---

## Phase 2: Dimension-Agnostic Properties (Week 1)

**Goal**: Ensure all MFGProblem modes expose consistent dimension-agnostic properties

### Tasks

1. **Unify `grid_spacing` across all modes**
   ```python
   @property
   def grid_spacing(self) -> list[float]:
       """Grid spacing [Δx₀, Δx₁, ...] for each dimension."""
       if hasattr(self, "dx"):  # 1D mode
           return [self.dx]
       elif hasattr(self, "_grid_spacing"):  # nD mode
           return self._grid_spacing
       elif hasattr(self, "geometry"):  # Geometry mode
           return self.geometry.get_grid_spacing()
       # ... handle network mode
   ```

2. **Unify `dimension` property**
   - Currently only set in nD mode
   - Should be 1 for 1D mode, extracted from geometry/network

3. **Unify `grid_shape` property**
   - Return tuple `(N₀, N₁, ...)` for all modes
   - 1D: `(Nx,)`
   - 2D: `(Nx, Ny)` → `(spatial_discretization[0], spatial_discretization[1])`

4. **Add property tests**
   ```python
   # tests/unit/test_core/test_protocol_compliance.py
   def test_1d_protocol_compliance():
       problem = MFGProblem(xmin=0, xmax=1, Nx=100, T=1, Nt=50, sigma=0.1)
       assert isinstance(problem, MFGProblemProtocol)
       assert problem.dimension == 1
       assert problem.grid_spacing == [0.01]
       assert problem.grid_shape == (100,)
   ```

---

## Phase 3: Solver Dimension-Agnosticism (Week 2)

**Goal**: Update solvers to use Protocol interface exclusively

### Substeps

1. **Audit Dimension Hardcoding**
   - Grep for `self.problem.Nx` in solvers → replace with `self.problem.spatial_discretization[0]`
   - Grep for `self.problem.dx` → replace with `self.problem.grid_spacing[0]`
   - Look for 1D/2D/3D specific branching → refactor to dimension-agnostic

2. **Refactor HJB Solvers**
   - `hjb_fdm.py`: Already uses dimensional splitting for nD
   - `hjb_semi_lagrangian.py`: Support arbitrary dimensions
   - `hjb_gfdm.py`: Already dimension-agnostic (recent work ✅)

3. **Refactor FP Solvers**
   - `fp_fdm.py`: Use dimensional splitting
   - `fp_particle.py`: Already mostly dimension-agnostic

4. **Update Coupling**
   - `fixed_point_iterator.py`: Use Protocol properties

---

## Phase 4: Deprecation & Compatibility (Week 2-3)

**Goal**: Gracefully deprecate old patterns, maintain backward compatibility

### Tasks

1. **Deprecate Scalar Parameters in 1D Mode**
   ```python
   # Emit warnings for:
   MFGProblem(xmin=0, xmax=1, Nx=100, ...)  # Legacy scalar
   # Suggest:
   MFGProblem(spatial_bounds=[(0, 1)], spatial_discretization=[100], ...)
   ```

2. **Create Migration Guide**
   - `docs/migration/v1.0_dimension_agnostic_api.md`
   - Show before/after examples
   - Explain benefits of new API

3. **Ensure 100% Backward Compatibility**
   - All existing examples continue to work
   - Deprecation warnings guide users to new API
   - Tests pass with both old and new API

---

## Phase 5: Documentation & Polish (Week 3)

**Goal**: Update documentation to reflect dimension-agnostic design

### Tasks

1. **Update Docstrings**
   - MFGProblem: Emphasize unified API for all dimensions
   - Solvers: Note they work for arbitrary dimensions

2. **Create Tutorial Notebooks**
   - `examples/tutorials/dimension_agnostic_mfg.ipynb`
   - Show same API for 1D, 2D, 3D
   - Demonstrate solver compatibility

3. **Update Main README**
   - Highlight dimension-agnostic capability
   - Show clean 1D/2D/3D examples

---

## Phase 6: Performance & Release (Week 3-4)

**Goal**: Ensure no regression, release v1.0

### Tasks

1. **Benchmark**
   - Verify no performance degradation
   - Test 1D, 2D, 3D, 4D problems

2. **CI Updates**
   - Add Protocol compliance tests to CI
   - Test multiple dimensions systematically

3. **Release**
   - v1.0.0-rc1 (release candidate)
   - Community testing period (1 week)
   - v1.0.0 final release

---

## Comparison: Branch vs Incremental

| Aspect | `feature/unified-nd` Branch | Incremental Evolution |
|:-------|:----------------------------|:----------------------|
| **Architecture** | Complete rewrite | Progressive refinement |
| **Rebase Complexity** | Massive conflicts (2000+ lines) | Clean, conflict-free |
| **Risk** | High (lose recent work) | Low (preserve everything) |
| **Testing** | Need to re-validate everything | Continuous validation |
| **Timeline** | 2-3 weeks (rebase + fixes) | 3-4 weeks (incremental) |
| **Compatibility** | Potentially breaks examples | Maintains 100% compat |

**Winner**: Incremental Evolution (safer, cleaner, more compatible)

---

## Current Status

**Phase 1 Progress**: 40% complete

- ✅ Protocol defined
- ✅ Protocol exported
- 🔄 Properties being added to MFGProblem
- ⏳ Solver type hints pending
- ⏳ Compliance tests pending

**Estimated Completion**: Week 1 (Protocol Foundation by 2025-11-15)

---

## Success Criteria

### Phase 1 Success
- [ ] `MFGProblem` satisfies `MFGProblemProtocol` for all modes (1D, nD, geometry, network)
- [ ] Solvers accept `MFGProblemProtocol` type hints
- [ ] Protocol compliance tests pass

### Overall Success
- [ ] Single `MFGProblem` API works for 1D, 2D, 3D, nD
- [ ] All solvers work dimension-agnostically
- [ ] 100% backward compatibility maintained
- [ ] Documentation updated
- [ ] v1.0.0 released

---

## References

- **Issue #245**: Original architectural renovation proposal
- **Branch**: `origin/feature/unified-nd-mfg-problem` (archived, ideas extracted)
- **Related**: Issue #243 (array notation standard)
- **CLAUDE.md**: Testing philosophy, incremental development approach
