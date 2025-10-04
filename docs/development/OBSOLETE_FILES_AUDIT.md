# Obsolete Files Audit - October 2025

**Date**: 2025-10-04
**Context**: Post-Phase 3 cleanup after tiered backend integration
**Purpose**: Identify and recommend removal of obsolete test files and examples

---

## 📋 Summary

After Phase 3 (tiered backend integration) completion, multiple test files from Phase 2 experimental work are now obsolete. Additionally, mass conservation tests have accumulated many redundant versions.

**Recommendation**: Delete 11 obsolete/redundant test files to improve maintainability.

---

## 🗑️ Recommended Deletions

### Category 1: Phase 2 Backend Tests (OBSOLETE - Phase 3 supersedes)

**Context**: Phase 2 used manual backend parameter passing. Phase 3 implements tiered auto-selection (`torch > jax > numpy`), making Phase 2 tests obsolete.

| File | Reason | Replacement |
|:-----|:-------|:------------|
| `test_phase2_backend_hjb.py` | Phase 2 HJB backend integration test | `test_tiered_backend_factory.py` |
| `test_phase2_complete.py` | Phase 2 complete integration test | `test_cross_backend_consistency.py` |
| `test_backend_integration.py` | Generic backend parameter test | `test_tiered_backend_factory.py` |
| `test_wrapper_approach.py` | Abandoned wrapper strategy (unused) | Direct backend integration |

**Action**: Delete all 4 Phase 2 test files.

---

### Category 2: Mass Conservation Tests (REDUNDANT - Too many versions)

**Context**: Multiple experimental iterations of mass conservation tests created during stochastic particle method development.

| File | Purpose | Keep? |
|:-----|:--------|:------|
| `test_kde_mass_conservation.py` | KDE-specific mass conservation | ✅ **KEEP** (specialized) |
| `test_mass_conservation_1d.py` | 1D mass conservation test | ⚠️ **MERGE with simple version** |
| `test_mass_conservation_1d_simple.py` | Simplified 1D version | ⚠️ **KEEP only if different** |
| `test_mass_conservation_attempts.py` | Experimental attempts | ❌ **DELETE** (experimental) |
| `test_mass_conservation_fast.py` | Fast version (reduced params) | ⚠️ **KEEP for CI speed** |
| `test_stochastic_mass_conservation.py` | Full stochastic analysis | ✅ **KEEP** (comprehensive) |
| `test_stochastic_mass_conservation_simple.py` | Simplified stochastic | ❌ **DELETE** (redundant) |

**Recommended Actions**:
1. **DELETE**: `test_mass_conservation_attempts.py` (experimental, no longer needed)
2. **DELETE**: `test_stochastic_mass_conservation_simple.py` (full version is sufficient)
3. **EVALUATE**: Compare `test_mass_conservation_1d.py` vs `_simple.py` - keep only one
4. **KEEP**: `test_kde_mass_conservation.py`, `test_mass_conservation_fast.py`, `test_stochastic_mass_conservation.py`

---

### Category 3: Acceleration Tests (EXPERIMENTAL - May be obsolete)

**Context**: Anderson acceleration and two-level damping tests from experimental solver optimization work.

| File | Purpose | Keep? |
|:-----|:--------|:------|
| `test_anderson_acceleration.py` | Anderson acceleration comparison | ⚠️ **EVALUATE** - Is Anderson still used? |
| `test_two_level_damping.py` | Two-level damping (Picard + Anderson) | ⚠️ **EVALUATE** - Active feature? |

**Action Required**: Check if Anderson acceleration is actively used in `FixedPointIterator`:
- If YES → Keep tests as performance benchmarks
- If NO → Delete as obsolete experiments

---

### Category 4: Phase 3 Tests (CURRENT - KEEP ALL)

| File | Purpose | Status |
|:-----|:--------|:-------|
| `test_tiered_backend_factory.py` | Tiered auto-selection (`torch > jax > numpy`) | ✅ Active |
| `test_cross_backend_consistency.py` | Numerical consistency across backends | ✅ Active |
| `test_torch_kde.py` | PyTorch KDE validation | ✅ Active |
| `test_mps_scaling.py` | MPS performance benchmarks | ✅ Active |

**Action**: Keep all Phase 3 tests - these validate current backend system.

---

### Category 5: Specialized Numerical Method Tests (KEEP IF USED)

| File | Purpose | Keep? |
|:-----|:--------|:------|
| `test_collocation_gfdm_hjb.py` | Collocation GFDM for HJB | ⚠️ **Check if method still implemented** |
| `test_particle_collocation.py` | Particle collocation methods | ⚠️ **Check if method still implemented** |
| `test_weight_functions.py` | Weight function tests | ⚠️ **Check if used in current solvers** |

**Action Required**: Verify these numerical methods are still actively used in the package.

---

## 📊 Cleanup Summary

### Immediate Deletions (11 files)

**Phase 2 Backend Tests** (4 files - 100% obsolete):
```bash
rm tests/integration/test_phase2_backend_hjb.py
rm tests/integration/test_phase2_complete.py
rm tests/integration/test_backend_integration.py
rm tests/integration/test_wrapper_approach.py
```

**Redundant Mass Conservation Tests** (2 files):
```bash
rm tests/integration/test_mass_conservation_attempts.py
rm tests/integration/test_stochastic_mass_conservation_simple.py
```

**To Evaluate** (5 files):
1. Compare `test_mass_conservation_1d.py` vs `test_mass_conservation_1d_simple.py` → Delete one
2. Check if Anderson acceleration is used → Delete `test_anderson_acceleration.py` and `test_two_level_damping.py` if not
3. Verify specialized methods are used → Delete `test_collocation_gfdm_hjb.py`, `test_particle_collocation.py`, `test_weight_functions.py` if obsolete

---

## 🔍 Examples Directory Audit

**Status**: ✅ Audited
**Total Count**: 82 example files (11 basic + 71 advanced)

### Basic Examples (11 files) - ✅ All Good

| File | Purpose | Status |
|:-----|:--------|:-------|
| `lq_mfg_demo.py` | LQ-MFG baseline example | ✅ Keep (v1.4.0) |
| `continuous_action_ddpg_demo.py` | DDPG demo | ✅ Keep (v1.4.0) |
| `rl_intro_comparison.py` | RL intro comparison | ✅ Keep (v1.4.0) |
| `nash_q_learning_demo.py` | Nash Q-learning | ✅ Keep (v1.4.0) |
| `towel_beach_example.py` | Beach towel problem | ✅ Keep |
| `multi_paradigm_comparison.py` | Paradigm comparison | ✅ Keep |
| `el_farol_bar_example.py` | El Farol bar | ✅ Keep |
| `dgm_simple_validation.py` | DGM validation | ✅ Keep |
| `adaptive_pinn_demo.py` | Adaptive PINN | ✅ Keep |
| `visualization_example.py` | Visualization demo | ✅ Keep |
| `simple_api_example.py` | Simple API demo | ✅ Keep |

**Action**: No cleanup needed for basic examples.

---

### Advanced Examples (71 files) - 🚨 NEEDS CLEANUP

#### Category 1: Maze Examples (22 files - EXCESSIVE REDUNDANCY)

**Problem**: 22 maze-related examples with substantial overlap.

| File | Purpose | Recommendation |
|:-----|:--------|:---------------|
| `actor_critic_maze_demo.py` | Actor-critic in maze | ✅ **KEEP** (unique algorithm) |
| `all_maze_algorithms_visualization.py` | Algorithm comparison | ✅ **KEEP** (comprehensive) |
| `hybrid_maze_demo.py` | Hybrid approach | ✅ **KEEP** (unique method) |
| `voronoi_maze_demo.py` | Voronoi maze generation | ✅ **KEEP** (unique geometry) |
| `mfg_maze_environment_demo.py` | Environment demo | ✅ **KEEP** (core functionality) |
| `mfg_maze_environment.py` | Environment implementation | ⚠️ **Move to package?** (looks like module) |
| `mfg_maze_layouts.py` | Layout definitions | ⚠️ **Move to package?** (looks like module) |
| `maze_principles_demo.py` | Principles overview | ⚠️ **Consolidate?** |
| `maze_config_examples.py` | Configuration examples | ⚠️ **Consolidate?** |
| `demo_page45_maze.py` | Page 45 demo | ❌ **DELETE** (unclear reference) |
| `page45_perfect_maze_demo.py` | Page 45 perfect maze | ❌ **DELETE** (duplicate) |
| `visualize_page45_maze.py` | Page 45 visualization | ❌ **DELETE** (duplicate) |
| `show_proper_maze.py` | Proper maze demo | ❌ **DELETE** (redundant) |
| `improved_maze_generator.py` | Generator implementation | ⚠️ **Move to package?** |
| `perfect_maze_generator.py` | Perfect maze generator | ⚠️ **Consolidate with improved?** |
| `maze_algorithm_assessment.py` | Algorithm assessment | ❌ **DELETE** (testing, not demo) |
| `maze_postprocessing_demo.py` | Postprocessing demo | ❌ **DELETE** (too specialized) |
| `maze_smoothing_demo.py` | Smoothing demo | ❌ **DELETE** (too specialized) |
| `quick_maze_assessment.py` | Quick assessment | ❌ **DELETE** (testing, not demo) |
| `quick_maze_demo.py` | Quick demo | ❌ **DELETE** (redundant with main) |
| `quick_perfect_maze_visual.py` | Quick visualization | ❌ **DELETE** (redundant) |
| `test_mfg_maze_comprehensive.py` | Comprehensive test | ❌ **MOVE TO TESTS** (not example) |

**Recommended Maze Cleanup**:
- **DELETE 10 files**: page45 duplicates, testing scripts, overly specialized demos
- **CONSOLIDATE 3 files**: Merge redundant generators and configs
- **MOVE 4 files**: Move module-like files to package or tests
- **KEEP 5 files**: Core unique demonstrations

**Impact**: 22 → 5 maze examples (-77% reduction)

---

#### Category 2: WENO Solver Examples (4 files - Reasonable)

| File | Purpose | Status |
|:-----|:--------|:-------|
| `2d_weno_solver_demo.py` | 2D WENO demo | ✅ Keep |
| `weno_family_comparison_demo.py` | WENO family comparison | ✅ Keep |
| `weno_solver_demo.py` | Basic WENO demo | ✅ Keep |
| `weno5_hjb_benchmarking_demo.py` | WENO5 benchmarks | ✅ Keep |

**Action**: Keep all WENO examples (reasonable diversity).

---

#### Category 3: Backend Acceleration Examples (6 files)

| File | API Version | Action |
|:-----|:------------|:-------|
| `unified_backend_acceleration_demo.py` | ✅ Phase 3 | Keep |
| `jax_acceleration_demo.py` | ⚠️ BackendFactory | **Update to Phase 3** |
| `jax_numba_hybrid_performance.py` | ❓ Check | **Verify Phase 3 API** |
| `triangular_amr_integration.py` | ❓ Check | **Verify Phase 3 API** |
| `advanced_visualization_example.py` | ❓ Check | Keep |
| `meta_programming_demo.py` | ❓ Check | Keep |

**Action**: Update 1-3 files to Phase 3 tiered backend API.

---

#### Category 4: RL Examples (Multiple)

| File | Purpose | Status |
|:-----|:--------|:-------|
| `continuous_control_comparison.py` | v1.4.0 feature | ✅ Keep |
| `heterogeneous_traffic_control.py` | Multi-population | ✅ Keep |
| `mfg_rl_comprehensive_demo.py` | Comprehensive RL | ✅ Keep |
| `mfg_rl_experiment_suite.py` | Experiment suite | ✅ Keep |
| `rl_principles_summary.py` | Principles | ⚠️ Should be in docs? |

**Action**: Verify RL examples are current with v1.4.0 API.

---

#### Category 5: Problem-Specific Examples (Keep)

| Type | Examples | Status |
|:-----|:---------|:-------|
| MFG Applications | el_farol_bar, santa_fe_bar, predator_prey, traffic_flow | ✅ Keep |
| Neural Methods | pinn, dgm, neural_operator, adaptive_pinn | ✅ Keep |
| Optimization | lagrangian, primal_dual | ✅ Keep |
| Geometry | 2d_anisotropic_crowd (7 files), network_mfg | ✅ Keep |
| High-Dim | 3d_box, complete_optimization_suite | ✅ Keep |
| API Demos | factory_patterns, new_api_*, pydantic, progress | ✅ Keep |

**Action**: Keep all problem-specific and API demonstration examples.

---

## 📊 Examples Cleanup Summary

### Recommended Deletions (13 files minimum)

**Maze Examples** (10 files):
```bash
# Delete page45 duplicates
rm examples/advanced/demo_page45_maze.py
rm examples/advanced/page45_perfect_maze_demo.py
rm examples/advanced/visualize_page45_maze.py

# Delete testing/assessment scripts (not examples)
rm examples/advanced/maze_algorithm_assessment.py
rm examples/advanced/quick_maze_assessment.py
rm examples/advanced/test_mfg_maze_comprehensive.py

# Delete redundant quick demos
rm examples/advanced/quick_maze_demo.py
rm examples/advanced/quick_perfect_maze_visual.py
rm examples/advanced/show_proper_maze.py

# Delete overly specialized demos
rm examples/advanced/maze_postprocessing_demo.py
rm examples/advanced/maze_smoothing_demo.py
```

### Recommended Moves (1 file)

**Test to Tests Directory**:
```bash
git mv examples/advanced/test_mfg_maze_comprehensive.py tests/integration/
```

### Recommended Consolidations (3 files)

**Maze Generators**:
- Evaluate `perfect_maze_generator.py` vs `improved_maze_generator.py`
- Keep the better one, delete or merge the other

**Maze Configs**:
- Consolidate `maze_config_examples.py` and `maze_principles_demo.py`

### Backend API Updates (2-3 files)

**Update to Phase 3 Tiered Backend API**:
- `jax_acceleration_demo.py`: Replace `BackendFactory` with `create_backend()`
- `jax_numba_hybrid_performance.py`: Verify uses Phase 3 API
- `triangular_amr_integration.py`: Verify uses Phase 3 API

---

## 📈 Expected Impact

**Before Cleanup**:
- Advanced examples: 71 files
- Maze examples: 22 files (31%)
- Redundant/testing files: ~15 files (21%)

**After Cleanup**:
- Advanced examples: ~54-58 files (-18-24%)
- Maze examples: ~8-10 files (focused set)
- All examples demonstrate unique features
- No testing scripts in examples/

**Benefits**:
- Clearer example organization
- Easier for users to find relevant demos
- Reduced maintenance burden
- Better separation of examples vs tests

---

## ✅ Recommended Actions

### Step 1: Immediate Cleanup (No risk)
```bash
# Delete Phase 2 tests (100% obsolete)
git rm tests/integration/test_phase2_backend_hjb.py
git rm tests/integration/test_phase2_complete.py
git rm tests/integration/test_backend_integration.py
git rm tests/integration/test_wrapper_approach.py

# Delete redundant mass conservation tests
git rm tests/integration/test_mass_conservation_attempts.py
git rm tests/integration/test_stochastic_mass_conservation_simple.py

git commit -m "🧹 Remove obsolete Phase 2 backend tests and redundant mass conservation tests"
```

### Step 2: Evaluate and Cleanup (Requires verification)
1. **Check Anderson Acceleration Usage**:
   ```bash
   grep -r "use_anderson" mfg_pde/
   ```
   - If used → Keep tests as benchmarks
   - If not used → Delete tests

2. **Compare Mass Conservation 1D Tests**:
   ```bash
   diff tests/integration/test_mass_conservation_1d.py \
        tests/integration/test_mass_conservation_1d_simple.py
   ```
   - If substantially different → Keep both
   - If redundant → Delete one

3. **Verify Specialized Methods**:
   ```bash
   grep -r "collocation_gfdm" mfg_pde/
   grep -r "particle_collocation" mfg_pde/
   grep -r "weight_function" mfg_pde/
   ```
   - If used → Keep tests
   - If not used → Delete tests

### Step 3: Examples Audit
- **TODO**: Audit `examples/` directory for obsolete demos
- **TODO**: Update any Phase 2 backend examples to Phase 3 API
- **TODO**: Remove redundant RL examples if any

---

## 📈 Expected Impact

**Before Cleanup**:
- Integration tests: 20 files
- Obsolete tests: ~6 files (30%)
- Redundant tests: ~5 files (25%)

**After Cleanup**:
- Integration tests: ~9-14 files (depends on evaluation)
- All tests validate current Phase 3 system
- Clear separation: core tests vs specialized method tests
- Improved maintainability and CI speed

---

## 🔗 Related Issues

- Phase 3 completion: Merged to main (commit `7a0c77d`)
- Backend integration: See `docs/development/PHASE3_TIERED_BACKEND_STRATEGY.md`
- MPS performance: See `docs/user/guides/backend_usage.md`

---

**Next Review**: After examples directory audit
