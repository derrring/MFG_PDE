# MFG_PDE Test Suite Analysis: CI Reduction Candidates

**Analysis Date**: 2025-11-04  
**Total Test Count**: 3,500 tests  
**Current CI (with -m "not slow")**: 3,443 tests  

---

## 1. TEST COUNT BY CATEGORY

### Overall Breakdown
```
Total Tests:                    3,500
├── Unit Tests (marked):        3,254
├── Integration Tests:            122
└── Root-Level Tests:             124
    ├── Slow Tests:                57
    └── Fast Tests:              3,443
```

### By Directory Structure

#### tests/unit/ (3,254 tests total)
| Category | Test Count | Purpose |
|:---------|:----------:|:--------|
| **test_geometry/** | 404 | Domain geometry, SDF, mesh generation |
| **test_utils/** | 307 | Numerical utilities, helpers |
| **test_config/** | 244 | Configuration schemas, parsing |
| **test_backends/** | 196 | Solver backends, implementations |
| **test_workflow/** | 161 | Workflow management, tracking |
| **test_types/** | 106 | Type checking, protocols |
| **test_visualization/** | 106 | Plotting, visualization backends |
| **test_factory/** | 80 | Factory functions, creation |
| **geometry/** | 83 | Geometry primitives (subset) |
| **test_benchmarks/** | 48 | Performance benchmarks |
| **test_core/** | 33 | Core problem classes |
| **test_hooks/** | 21 | Workflow hooks |
| **test_io/** | 14 | File I/O operations |
| **test_compat/** | 0 | Compatibility (skipped) |
| **utils/** | 36 | General utilities |
| **Root test_*.py** | 351 | Solvers, problems, environments |

#### tests/integration/ (122 tests)
Critical cross-component tests:
- Coupled HJB-FP solvers (2D, general)
- Mass conservation (1D, hybrid, KDE)
- Backend consistency
- Particle collocation MFG
- GFDM collocation
- Common noise MFG
- Network solvers

#### tests/ (124 root-level tests)
| File | Tests | Category |
|:-----|:-----:|:---------|
| test_anderson_multidim.py | ~8 | Advanced algorithm |
| test_bug15_sigma_fix.py | ~4 | Regression fix |
| test_dgm_foundation.py | ~20 | Deep Galerkin method |
| test_geometry_pipeline.py | ~15 | Geometry workflow |
| test_gradient_notation_standard.py | ~12 | Mathematical notation |
| test_gradient_utils.py | ~8 | Gradient utilities |
| test_hjb_gfdm_monotonicity.py | ~3 | GFDM properties |
| test_neural_operators.py | ~20 | Neural operators |
| test_structured_configs.py | ~15 | Config validation |

---

## 2. OPTIONAL DEPENDENCY TESTS

### PyTorch Tests (29 files, ~120+ tests)
**Impact**: 4% of total tests, requires optional torch installation

**Files to mark as `@pytest.mark.optional_torch`**:
```
tests/unit/test_mean_field_sac.py                    (~15 tests)
tests/unit/test_mean_field_ddpg.py                   (~12 tests)
tests/unit/test_mean_field_td3.py                    (~12 tests)
tests/unit/test_multi_population_config.py           (~10 tests)
tests/unit/test_multi_population_sac.py              (~12 tests)
tests/unit/test_multi_population_ddpg.py             (~10 tests)
tests/unit/test_multi_population_td3.py              (~10 tests)
tests/unit/test_multi_population_algorithms_basic.py (~8 tests)
tests/unit/test_multi_population_environment.py      (~8 tests)
tests/unit/test_multi_population_networks.py         (~6 tests)
tests/unit/test_density_estimation.py                (~4 torch-specific tests)
tests/unit/test_alg/test_neural/test_core/test_training.py
tests/unit/test_backends/test_backend_factory.py     (~5 torch tests)
tests/integration/test_particle_gpu_pipeline.py      (~4 tests)
tests/integration/test_torch_kde.py                  (~8 tests)
tests/integration/test_mps_scaling.py                (~6 tests)
tests/integration/test_cross_backend_consistency.py  (~4 torch variants)
tests/test_dgm_foundation.py                         (~8 tests)
tests/test_neural_operators.py                       (~15 tests)
```

**Reduction**: -120 tests, CI becomes 3,380 tests

### SciPy-Specific Tests (17 files)
Generally fast, lightweight dependencies. Recommend KEEP for CI (scipy is core dependency).

### Plotly/Bokeh Tests (2+ files)
**Files**:
- tests/unit/test_visualization/test_mathematical_plots.py (~20 tests)
- tests/unit/test_visualization/test_mfg_analytics.py (~15 tests)

**Impact**: Interactive visualization, not core functionality
**Reduction**: -35 tests

---

## 3. PERFORMANCE & SLOW TESTS

### Currently Marked as Slow (57 tests)
```
integration/test_common_noise_mfg.py              (~8 tests, parametrized)
integration/test_hjb_fdm_2d_validation.py         (~9 tests)
integration/test_hybrid_mass_conservation.py      (~3 tests)
integration/test_lq_common_noise_analytical.py    (~6 tests)
integration/test_mass_conservation_1d_simple.py   (~4 tests)
```

**Already excluded from CI** - no action needed.

### Benchmark Tests (48 tests in test_benchmarks/)
These are explicitly performance analysis tests:
- test_performance_tracker.py
- test_solver_performance.py
- test_backend_performance.py
- Geometry sampling/SDF benchmarks

**Recommendation**: Mark as `@pytest.mark.benchmark`, exclude from CI

**Reduction**: -48 tests

---

## 4. EXPERIMENTAL & ADVANCED FEATURE TESTS

### Experimental Algorithms
These are research/exploratory, not core infrastructure:

| Module | Tests | Notes |
|:-------|:-----:|:------|
| test_cellular_automata.py | ~12 | CA-based MFG solver |
| test_mcmc.py | ~20 | MCMC sampling |
| test_monte_carlo.py | ~15 | Monte Carlo integration |
| test_anderson_multidim.py | ~8 | Anderson acceleration |

**Total**: ~55 tests  
**Recommendation**: Mark as `@pytest.mark.experimental`, exclude from CI

**Reduction**: -55 tests

### Maze/Environment Tests
```
test_maze_*.py (4 files)                          ~35 tests
test_cellular_automata.py                         ~12 tests
test_*_mfg_env.py (LQ, crowd, traffic, etc.)     ~40 tests
test_*_env_*.py variants                          ~25 tests
```

**Total**: ~112 tests  
**Status**: Non-core, environment-specific
**Recommendation**: Mark as `@pytest.mark.environment`, partial exclusion (-60)

---

## 5. CORE FUNCTIONALITY TESTS (MUST KEEP)

### Solver Tests (Essential)
```
tests/unit/test_hjb_fdm_solver.py              (~15 tests)  ✓ FDM HJB solver
tests/unit/test_hjb_gfdm_solver.py             (~12 tests)  ✓ GFDM HJB solver
tests/unit/test_hjb_semi_lagrangian.py         (~20 tests)  ✓ Semi-Lagrangian
tests/unit/test_fp_fdm_solver.py               (~15 tests)  ✓ FDM FP solver
tests/unit/test_fp_particle_solver.py          (~18 tests)  ✓ Particle solver
tests/unit/test_fp_network_solver.py           (~12 tests)  ✓ Network solver
tests/integration/test_fdm_solvers_mfg_complete.py (~10 tests)  ✓ Integration
tests/integration/test_coupled_hjb_fp_2d.py    (~3 tests)   ✓ Coupled solvers
```
**Total**: ~105 tests - ESSENTIAL, KEEP ALL

### Problem Classes (Essential)
```
tests/unit/test_unified_mfg_problem.py          (~12 tests)  ✓ Problem definition
tests/unit/test_solve_mfg.py                    (~8 tests)   ✓ Solution wrapper
tests/unit/test_lq_mfg_env.py                   (~10 tests)  ✓ LQ problem
tests/integration/test_mass_conservation_*.py   (~25 tests)  ✓ Physical validation
```
**Total**: ~55 tests - ESSENTIAL, KEEP ALL

### Factory & Config (Essential)
```
tests/unit/test_factory/                        (~80 tests)  ✓ Factory functions
tests/unit/test_config/                         (~244 tests) ✓ Configuration
tests/unit/test_factory_patterns.py             (~8 tests)   ✓ Patterns
tests/unit/test_problem_factories.py            (~6 tests)   ✓ Problem creation
```
**Total**: ~338 tests - ESSENTIAL, KEEP ALL

### Type System & Validation (Essential)
```
tests/unit/test_types/                          (~106 tests) ✓ Type checking
tests/unit/test_core/                           (~33 tests)  ✓ Core types
tests/unit/test_exceptions.py                   (~8 tests)   ✓ Error handling
```
**Total**: ~147 tests - ESSENTIAL, KEEP ALL

### Geometry (Core)
```
tests/unit/test_geometry/                       (~404 tests) ✓ Domain geometry
tests/unit/geometry/                            (~83 tests)  ✓ Geometry ops
```
**Total**: ~487 tests - ESSENTIAL, KEEP ALL
*(Note: Includes advanced geometry benchmarks - could move benchmarks to separate tier)*

---

## 6. CANDIDATES FOR CI EXCLUSION

### Tier 1: High Priority (Immediate Exclusion)

| Category | Tests | Marker | Rationale |
|:---------|:-----:|:-------|:----------|
| PyTorch algorithms | 120 | `optional_torch` | Optional dependency, +30s CI time |
| Benchmark tests | 48 | `benchmark` | Performance analysis, not validation |
| Experimental algorithms | 55 | `experimental` | Research code, unstable |
| Advanced maze/env | 60 | `environment` | Non-core, problem-specific |
| **Subtotal** | **283** | | **Reduction to 3,217 tests** |

### Tier 2: Secondary (Conditional Exclusion)

| Category | Tests | Marker | Rationale |
|:---------|:-----:|:-------|:----------|
| Interactive visualization | 35 | `visualization` | Requires display, plotly optional |
| RL environment tests | 50 | `gymnasium` | Requires gymnasium, non-core |
| GPU-specific tests | 15 | `gpu` | GPU-only validation |
| **Subtotal** | **100** | | **Additional reduction to 3,117 tests** |

### Not Recommended for Exclusion
- **Geometry tests** (487): Essential domain validation
- **Config tests** (244): Core infrastructure
- **Backend tests** (196): Solver validation
- **Solver integration** (25+): Mathematical correctness
- **Type checking** (106): API stability

---

## 7. PYTEST MARKERS TO ADD

```ini
# In pytest.ini - add to markers section:

markers =
    # ... existing markers ...
    
    # Optional dependencies
    optional_torch: Tests requiring PyTorch (exclude from basic CI)
    optional_scipy: Tests requiring scipy (keep in full CI)
    optional_plotly: Tests requiring plotly/interactive viz
    
    # Performance & analysis
    benchmark: Performance measurement tests (slow, exclude from CI)
    performance: Performance validation (may be slow)
    
    # Feature maturity
    experimental: Experimental/research features (unstable API)
    research: Research algorithms not in stable API
    
    # Problem domains
    environment: MFG environment tests (gymnasium, envs)
    gymnasium: Tests requiring gymnasium library
    gpu: GPU-specific tests (CUDA, etc.)
    visualization: Interactive visualization tests
    
    # Architecture-specific
    neural_network: Tests for neural network components
    rl_algorithm: Tests for RL algorithms (SAC, DDPG, TD3, etc.)
```

---

## 8. CI CONFIGURATION RECOMMENDATIONS

### Option A: Aggressive Reduction (Recommended for Speed)
```bash
# Exclude optional dependencies and advanced features
pytest -m "not slow and not optional_torch and not benchmark and not experimental"
# Result: ~3,217 tests (92% of suite)
# Time reduction: ~20-25%
```

### Option B: Balanced Reduction
```bash
# Keep more advanced validation, exclude only PyTorch-heavy tests
pytest -m "not slow and not optional_torch and not benchmark"
# Result: ~3,272 tests (94% of suite)
# Time reduction: ~10-15%
```

### Option C: Full Suite (Current)
```bash
pytest -m "not slow"
# Result: 3,443 tests
# (Status: Current CI configuration)
```

### Recommended Strategy:
1. **Add markers** to test files (quick, 30-minute task)
2. **Start with Option A** for CI (most aggressive)
3. **Keep Option C** for local/pre-commit (complete validation)
4. **Run Option C nightly** for full coverage

---

## 9. DETAILED EXCLUSION BREAKDOWN

### PyTorch Exclusion (-120 tests)

**Files requiring `@pytest.mark.optional_torch`**:
```python
# In tests/unit/test_mean_field_sac.py
@pytest.mark.optional_torch
class TestMeanFieldSAC:
    ...

# In tests/unit/test_mean_field_ddpg.py
@pytest.mark.optional_torch
def test_ddpg_training():
    ...

# Similar for: TD3, multi-population variants
# tests/unit/test_multi_population_*.py (5 files)

# In tests/integration/test_particle_gpu_pipeline.py
@pytest.mark.optional_torch
def test_gpu_matches_cpu_numerically():
    ...
```

### Benchmark Exclusion (-48 tests)

**Files requiring `@pytest.mark.benchmark`**:
```python
# tests/unit/test_benchmarks/test_*.py
# - test_performance_tracker.py
# - test_solver_performance.py
# - test_backend_performance.py

# tests/unit/geometry/test_geometry_benchmarks.py
@pytest.mark.benchmark
def test_hyperrectangle_sampling_2d():
    ...

# tests/integration/test_mps_scaling.py
@pytest.mark.benchmark
@pytest.mark.optional_torch
def test_scaling_performance():
    ...
```

### Experimental Exclusion (-55 tests)

**Files requiring `@pytest.mark.experimental`**:
```python
# tests/unit/test_cellular_automata.py
@pytest.mark.experimental
class TestCellularAutomata:
    ...

# tests/unit/test_mcmc.py
@pytest.mark.experimental
def test_mcmc_sampling():
    ...

# tests/unit/test_monte_carlo.py
@pytest.mark.experimental
class TestMonteCarloIntegration:
    ...

# tests/test_anderson_multidim.py
@pytest.mark.experimental
def test_anderson_acceleration():
    ...
```

---

## 10. IMPLEMENTATION CHECKLIST

### Phase 1: Add Markers (1-2 hours)
- [ ] Update pytest.ini with new markers
- [ ] Tag PyTorch tests with `@pytest.mark.optional_torch` (15 files)
- [ ] Tag benchmark tests with `@pytest.mark.benchmark` (8 files)
- [ ] Tag experimental tests with `@pytest.mark.experimental` (5 files)
- [ ] Tag environment tests with `@pytest.mark.environment` (10 files)

### Phase 2: Update CI Configuration (30 mins)
- [ ] Modify `.github/workflows/tests.yml`:
  ```yaml
  # Basic CI: exclude optional/experimental
  - name: Run core tests
    run: pytest -m "not slow and not optional_torch and not benchmark"
  
  # Full CI (nightly or manual):
  - name: Run complete suite
    run: pytest -m "not slow"
  ```

### Phase 3: Validation (1-2 hours)
- [ ] Run with aggressive markers: verify 3,217 tests pass
- [ ] Check local dev workflow: still runs full suite
- [ ] Verify type checking pipeline
- [ ] Document in CONTRIBUTING.md

### Phase 4: Monitoring (ongoing)
- [ ] Track CI time reduction
- [ ] Monitor PyTorch test results in separate job
- [ ] Adjust markers if tests are misclassified

---

## 11. RISK ASSESSMENT

### Low Risk (Safe to Exclude)
- PyTorch tests (skipped if torch unavailable anyway)
- Benchmark tests (supplementary, not validation)
- Experimental algorithms (marked unstable)

### Medium Risk (Need Review)
- Environment tests (problem-specific, but useful for examples)
- Visualization tests (depends on display availability)

### High Risk (Must Keep)
- All solver integration tests
- All geometry tests
- All factory/config tests
- All type checking tests

---

## 12. ESTIMATED TIME SAVINGS

| Change | Tests | Time Savings | Notes |
|:-------|:-----:|:------------|:------|
| Current baseline | 3,443 | ~120-150s | Full suite excluding slow |
| Remove PyTorch | -120 | ~15-20s | Skip ~5-10 large RL tests |
| Remove Benchmarks | -48 | ~5-10s | Skip timing-heavy tests |
| Remove Experimental | -55 | ~8-12s | Skip research features |
| Remove Environments | -60 | ~10-15s | Skip maze/env setup |
| **Total Aggressive** | 3,217 | **~40-60s (40%)** | Still validates core |
| **Total Balanced** | 3,272 | **~20-35s (25%)** | More coverage |

---

## 13. FINAL RECOMMENDATION

**IMPLEMENT: Option A (Aggressive) with Phased Approach**

1. **Immediate** (Day 1):
   - Add markers to pytest.ini
   - Tag top 5 PyTorch files
   - Create CI branch with `-m "not optional_torch"`

2. **Short-term** (Week 1):
   - Complete all marker tagging
   - Validate no regressions
   - Update documentation

3. **Medium-term** (Week 2-3):
   - Implement separate "full suite" job (nightly)
   - Update developer docs
   - Monitor CI metrics

**Result**: 40-60s time savings (40% reduction) while maintaining core validation

---

**Report Generated**: 2025-11-04  
**Analysis by**: Claude Code (Haiku 4.5)  
**Next Steps**: Review recommendations, implement marking phase, update CI config
