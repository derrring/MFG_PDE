# Test Coverage Improvement Plan

**Current Coverage**: 37% (11,877 covered / 32,499 total lines)
**Target Coverage**: 50-60% (16,000-19,000 covered lines)
**Effort**: 80-110 hours over 4-6 weeks

## Executive Summary

This plan strategically improves MFG_PDE test coverage from **37% to 50-60%** by focusing on:
1. **Quick wins** (0% coverage utils) → 42%
2. **High-impact core** (backends, config, geometry) → 50%
3. **Comprehensive validation** (RL, factory, performance) → 60%

**Key Insight**: Most critical modules (core, factory, numerical algorithms) already have 65-80% coverage. The biggest gaps are in:
- **Utilities** (0% for progress, decorators)
- **Backends** (38% - critical for GPU support)
- **Config** (40% - Pydantic/OmegaConf integration)
- **Visualization** (14% - acceptable, low priority)
- **Workflow** (0% - experimental, deferred)

---

## Coverage Breakdown by Module Priority

### CRITICAL Priority (Avg: 67.3%)
| Module | Coverage | Lines | Status |
|:-------|:---------|:------|:-------|
| **alg/numerical** | 65% | 8,500 | 🔸 Needs improvement |
| **core** | 75% | 1,500 | ✅ Good |
| **factory** | 80% | 600 | ✅ Good |

### HIGH Priority (Avg: 57.1%)
| Module | Coverage | Lines | Status |
|:-------|:---------|:------|:-------|
| **backends** | 38% | 1,200 | ⚠️ Urgent |
| **config** | 40% | 800 | ⚠️ Urgent |
| **geometry** | 52% | 2,100 | 🔸 Needs work |
| **utils/solver_result** | 62% | 151 | 🔸 Almost there |
| **alg/reinforcement** | 70% | 3,200 | ✅ Good |
| **utils/sparse_operations** | 81% | 235 | ✅ Excellent |

### MEDIUM Priority (Avg: 16.6%)
| Module | Coverage | Lines | Status |
|:-------|:---------|:------|:-------|
| **utils/progress** | 0% | 156 | ⚠️ Zero coverage |
| **utils/solver_decorators** | 0% | 112 | ⚠️ Zero coverage |
| **utils/performance/optimization** | 23% | 409 | 🔸 Low |
| **utils/performance/monitoring** | 24% | 244 | 🔸 Low |

### LOW Priority (Deferred)
| Module | Coverage | Lines | Rationale |
|:-------|:---------|:------|:----------|
| **workflow** | 0% | 1,281 | Experimental features, defer to v2.0+ |
| **visualization** | 14% | 1,604 | Hard to test plotting, use manual validation |
| **meta** | 29% | 284 | Metaprogramming, low usage |

---

## Phase 1: Quick Wins (37% → 42%, 2-3 days)

**Goal**: Cover zero-coverage utility modules
**Effort**: 12-16 hours
**Impact**: +245 covered lines

### 1.1 Utils/Progress (0% → 80%)
**File**: `mfg_pde/utils/progress.py`
**Lines**: 156 → +125 covered
**Why**: Used by solvers, simple to test

**Tests to write**:
```python
# tests/unit/test_utils/test_progress.py

def test_tqdm_available():
    """Test progress bar when tqdm available."""
    from mfg_pde.utils.progress import ProgressBar
    pbar = ProgressBar(total=100, desc="Test")
    # Verify tqdm integration

def test_tqdm_unavailable():
    """Test fallback when tqdm not available."""
    # Mock tqdm import failure
    # Verify silent fallback

def test_progress_context_manager():
    """Test progress bar as context manager."""
    with ProgressBar(total=100) as pbar:
        for i in range(100):
            pbar.update(1)

def test_custom_formatters():
    """Test custom progress formatters."""
    # Test different bar_format options
```

### 1.2 Utils/Solver Decorators (0% → 70%)
**File**: `mfg_pde/utils/solver_decorators.py`
**Lines**: 112 → +78 covered
**Why**: Logging/timing decorators, straightforward

**Tests to write**:
```python
# tests/unit/test_utils/test_solver_decorators.py

def test_timed_solver_decorator():
    """Test @timed_solver decorator."""
    from mfg_pde.utils.solver_decorators import timed_solver

    @timed_solver
    def dummy_solver():
        import time
        time.sleep(0.1)
        return "result"

    result = dummy_solver()
    # Verify timing logged

def test_logged_solver_decorator():
    """Test @logged_solver decorator."""
    # Verify logging output

def test_decorator_combinations():
    """Test multiple decorators on same function."""
    @timed_solver
    @logged_solver
    def solver():
        pass
```

### 1.3 Utils/Solver Result (62% → 90%)
**File**: `mfg_pde/utils/solver_result.py`
**Lines**: 151 → +42 covered
**Why**: Critical API, already partially tested

**Tests to add**:
```python
# tests/unit/test_solver_result.py (expand existing)

def test_hdf5_save_load_roundtrip():
    """Test HDF5 save/load preserves data."""
    result = SolverResult(U=..., M=..., converged=True)
    result.save_hdf5("test.h5")
    loaded = SolverResult.load_hdf5("test.h5")
    assert np.allclose(result.U, loaded.U)

def test_comparison_methods():
    """Test result comparison."""
    result1 = SolverResult(...)
    result2 = SolverResult(...)
    diff = result1.compare_to(result2)
    # Verify difference metrics

def test_convergence_analysis():
    """Test convergence analysis methods."""
    result.analyze_convergence()
    # Verify analysis output

def test_export_functionality():
    """Test export to various formats."""
    result.export_summary(format='markdown')
    result.export_summary(format='json')
```

**Expected Result**: 37% → 42% (+245 lines, +5%)

---

## Phase 2: High-Impact Core (42% → 50%, 1-2 weeks)

**Goal**: Strengthen critical infrastructure
**Effort**: 40-60 hours
**Impact**: +1,897 covered lines

### 2.1 Backends (38% → 65%, +324 lines)
**Why**: Critical for GPU acceleration

**Priority files**:
1. `backends/torch_backend.py` - PyTorch operations
2. `backends/jax_backend.py` - JAX operations
3. `backends/strategies.py` - Backend selection

**Tests to write**:
```python
# tests/unit/test_backends/test_torch_backend.py (expand)

def test_gpu_detection():
    """Test GPU availability detection."""
    from mfg_pde.backends import TorchBackend
    backend = TorchBackend()
    # Verify correct device selection

def test_cpu_gpu_consistency():
    """Test CPU and GPU give same results."""
    # Run same operation on both
    # Verify numerical consistency

def test_tensor_operations():
    """Test all tensor operations."""
    # matmul, einsum, reduce_sum, etc.

def test_backend_switching():
    """Test switching between backends."""
    # Test transition NumPy → PyTorch → JAX

def test_missing_dependency_handling():
    """Test error handling when GPU unavailable."""
    # Mock torch.cuda.is_available() = False
    # Verify graceful fallback
```

```python
# tests/integration/test_backend_consistency.py (NEW)

def test_solver_cpu_vs_gpu():
    """Test solver gives same results on CPU/GPU."""
    problem = ExampleMFGProblem(Nx=50, Nt=20)

    # CPU solution
    solver_cpu = create_fast_solver(problem, backend="numpy")
    result_cpu = solver_cpu.solve()

    # GPU solution (if available)
    if torch.cuda.is_available():
        solver_gpu = create_fast_solver(problem, backend="torch_cuda")
        result_gpu = solver_gpu.solve()
        assert np.allclose(result_cpu.U, result_gpu.U, rtol=1e-5)
```

### 2.2 Config System (40% → 70%, +240 lines)
**Why**: Pydantic/OmegaConf validation is critical

**Tests to write**:
```python
# tests/unit/test_config/test_pydantic_models.py (expand)

def test_valid_config_loading():
    """Test loading valid configurations."""
    from mfg_pde.config import SolverConfig
    config = SolverConfig(solver_type="fixed_point", max_iterations=100)
    # Verify validation passes

def test_invalid_config_validation():
    """Test validation catches invalid configs."""
    with pytest.raises(ValidationError):
        SolverConfig(solver_type="invalid", max_iterations=-1)

def test_yaml_loading():
    """Test loading from YAML."""
    yaml_str = """
    solver_type: fixed_point
    max_iterations: 100
    tolerance: 1e-6
    """
    config = SolverConfig.from_yaml(yaml_str)

def test_parameter_migration():
    """Test old parameter names migrate correctly."""
    # Old: max_iter → New: max_iterations
    config = SolverConfig(max_iter=50)  # deprecated
    assert config.max_iterations == 50

def test_default_values():
    """Test default value handling."""
    config = SolverConfig(solver_type="fixed_point")
    assert config.max_iterations == 100  # default
```

### 2.3 Geometry (52% → 75%, +483 lines)
**Why**: Domain definitions are fundamental

**Tests to write**:
```python
# tests/unit/test_geometry/test_boundary_conditions.py (expand)

def test_neumann_bc_enforcement():
    """Test Neumann BC zero flux."""
    from mfg_pde.geometry import BoundaryConditions
    bc = BoundaryConditions(type="neumann")
    # Verify zero gradient at boundaries

def test_dirichlet_bc_enforcement():
    """Test Dirichlet BC fixed values."""
    bc = BoundaryConditions(type="dirichlet", value=1.0)
    # Verify boundary values enforced

def test_periodic_bc():
    """Test periodic boundary conditions."""
    # Verify left matches right

def test_mixed_bc():
    """Test mixed BCs (left Neumann, right Dirichlet)."""
```

```python
# tests/unit/test_geometry/test_multidim_domains.py (NEW)

def test_2d_domain_creation():
    """Test 2D rectangular domain."""
    from mfg_pde.geometry import Domain2D
    domain = Domain2D(xmin=0, xmax=1, ymin=0, ymax=1, Nx=50, Ny=50)
    assert domain.ndim == 2

def test_3d_domain_creation():
    """Test 3D cuboid domain."""
    domain = Domain3D(bounds=[(0,1), (0,1), (0,1)], points=[20, 20, 20])

def test_adaptive_mesh_refinement():
    """Test AMR creates finer mesh near features."""
    # Test mesh refinement logic
```

### 2.4 Numerical Algorithms (65% → 75%, +850 lines)
**Focus**: Edge cases and numerical properties

**Tests to write**:
```python
# tests/mathematical/test_numerical_properties.py (expand)

@pytest.mark.parametrize("sigma", [0.01, 0.1, 0.5, 1.0])
def test_solver_stability(sigma):
    """Test solver stability for various diffusion coefficients."""
    problem = ExampleMFGProblem(sigma=sigma)
    solver = create_fast_solver(problem)
    result = solver.solve()
    assert result.converged

def test_zero_diffusion_limit():
    """Test behavior as sigma → 0."""
    # Should approach deterministic optimal control

def test_large_timestep_stability():
    """Test stability with large timesteps."""
    problem = ExampleMFGProblem(Nt=5)  # Large Dt
    # Should either converge or raise warning

def test_mass_conservation_edge_cases():
    """Test mass conservation in edge cases."""
    # Zero initial mass
    # Delta function initial condition
    # Discontinuous initial condition
```

**Expected Result**: 42% → 50% (+1,897 lines, +8%)

---

## Phase 3: Comprehensive Validation (50% → 60%, 2-3 weeks)

**Goal**: Complete RL framework and optimize coverage
**Effort**: 25-35 hours
**Impact**: +805 covered lines

### 3.1 Reinforcement Learning (70% → 85%, +480 lines)

**Tests to write**:
```python
# tests/unit/test_reinforcement/test_continuous_control.py (expand)

def test_ddpg_policy_gradient():
    """Test DDPG computes correct policy gradients."""
    from mfg_pde.alg.reinforcement import MeanFieldDDPG
    # Verify gradient calculations

def test_td3_twin_critics():
    """Test TD3 uses minimum of twin Q-values."""
    # Verify twin critic architecture

def test_sac_entropy_regularization():
    """Test SAC entropy term."""
    # Verify entropy maximization

def test_replay_buffer_sampling():
    """Test replay buffer uniformly samples."""
    # Verify sampling distribution

def test_convergence_to_nash():
    """Test RL converges to Nash equilibrium for LQ-MFG."""
    # Compare to analytical solution
```

### 3.2 Factory Functions (80% → 95%, +90 lines)

**Tests to write**:
```python
# tests/unit/test_factory/test_solver_factory.py (expand)

def test_all_solver_combinations():
    """Test all valid FP+HJB combinations."""
    fp_types = ['particle', 'fdm', 'network']
    hjb_types = ['fdm', 'weno', 'semi_lagrangian', 'gfdm']

    for fp_type in fp_types:
        for hjb_type in hjb_types:
            if is_valid_combination(fp_type, hjb_type):
                solver = create_solver(fp_type=fp_type, hjb_type=hjb_type)
                assert solver is not None

def test_invalid_config_handling():
    """Test factory raises clear errors for invalid configs."""
    with pytest.raises(ValueError, match="Unknown solver type"):
        create_fast_solver(problem, solver_type="nonexistent")

def test_backward_compatibility():
    """Test old API still works with warnings."""
    with pytest.warns(DeprecationWarning):
        solver = create_solver(max_iter=100)  # old parameter name
```

### 3.3 Utils/Performance (24% → 60%, +235 lines)

**Tests to write**:
```python
# tests/unit/test_utils/test_performance_monitoring.py (NEW)

def test_memory_tracking():
    """Test memory monitoring."""
    from mfg_pde.utils.performance import MemoryMonitor
    monitor = MemoryMonitor(max_memory_gb=2.0)
    # Allocate large array
    # Verify monitor detects it

def test_cpu_profiling():
    """Test CPU time profiling."""
    from mfg_pde.utils.performance import profile_function

    @profile_function
    def expensive_operation():
        # Simulate computation
        pass

    expensive_operation()
    # Verify timing recorded

def test_optimization_suggestions():
    """Test performance optimization suggestions."""
    # Run solver with inefficient settings
    # Verify suggestions provided
```

**Expected Result**: 50% → 60% (+805 lines, +10%)

---

## Implementation Roadmap

### Week 1-2: Phase 1 (Quick Wins)
**Days 1-2**: Utils/Progress (0% → 80%)
- Create `tests/unit/test_utils/test_progress.py`
- Test tqdm integration and fallback
- Test context managers

**Days 3-4**: Utils/Solver Decorators (0% → 70%)
- Create `tests/unit/test_utils/test_solver_decorators.py`
- Test timing and logging decorators
- Test decorator combinations

**Day 5**: Utils/Solver Result (62% → 90%)
- Expand `tests/unit/test_solver_result.py`
- Add HDF5, comparison, analysis tests

**Milestone**: 37% → 42% coverage

### Week 3-4: Phase 2 Part 1 (Backends + Config)
**Days 6-9**: Backends (38% → 65%)
- Expand backend tests
- Add CPU/GPU consistency tests
- Test backend selection logic

**Days 10-13**: Config System (40% → 70%)
- Expand Pydantic validation tests
- Add YAML loading tests
- Test parameter migration

**Milestone**: 42% → 46% coverage

### Week 5-6: Phase 2 Part 2 (Geometry + Numerical)
**Days 14-17**: Geometry (52% → 75%)
- Expand boundary condition tests
- Add multi-dimensional domain tests
- Test AMR logic

**Days 18-21**: Numerical Algorithms (65% → 75%)
- Add edge case tests
- Test numerical stability
- Property-based testing

**Milestone**: 46% → 50% coverage

### Week 7-8: Phase 3 (RL + Factory + Performance)
**Days 22-26**: RL Framework (70% → 85%)
- Expand continuous control tests
- Test convergence properties

**Days 27-29**: Factory + Performance (80% → 90%, 24% → 60%)
- Test all factory combinations
- Add performance monitoring tests

**Day 30**: Final cleanup and documentation

**Final Milestone**: 50% → 60% coverage

---

## Tools & Infrastructure

### Testing Tools
```bash
# Primary
pytest                  # Test runner
pytest-cov              # Coverage measurement
pytest-xdist            # Parallel test execution

# Property-based testing
hypothesis              # Property-based testing for numerical code

# Mutation testing (optional)
mutmut                  # Mutation testing for critical algorithms

# Performance
pytest-benchmark        # Performance regression testing
```

### Coverage Configuration
```yaml
# codecov.yml
coverage:
  status:
    project:
      default:
        target: 50%        # Phase 2 target
        threshold: 1%      # Allow 1% fluctuation

    patch:
      default:
        target: 70%        # New code should have high coverage

ignore:
  - "mfg_pde/workflow/**"           # Experimental
  - "mfg_pde/visualization/**"      # Hard to test
  - "mfg_pde/meta/**"               # Low priority
  - "mfg_pde/_internal/**"          # Private
```

### Property-Based Testing Example
```python
# Using Hypothesis for numerical stability
from hypothesis import given, strategies as st

@given(
    sigma=st.floats(min_value=0.01, max_value=2.0),
    Nx=st.integers(min_value=20, max_value=100),
    Nt=st.integers(min_value=10, max_value=50)
)
def test_solver_always_converges(sigma, Nx, Nt):
    """Property: Solver should converge for all valid parameters."""
    problem = ExampleMFGProblem(sigma=sigma, Nx=Nx, Nt=Nt)
    solver = create_fast_solver(problem)
    result = solver.solve()
    assert result.converged, f"Failed to converge with sigma={sigma}, Nx={Nx}, Nt={Nt}"
```

---

## Success Metrics

### Coverage Targets
- ✅ **Phase 1**: 42% (+5% from baseline)
- ✅ **Phase 2**: 50% (+8% from Phase 1)
- ✅ **Phase 3**: 60% (+10% from Phase 2)

### Module-Specific Targets
- ✅ **Critical modules** (core, factory): ≥ 90%
- ✅ **High-priority** (backends, config, geometry, RL): ≥ 70%
- ✅ **Medium-priority** (utils, performance): ≥ 50%
- ✅ **Low-priority** (visualization, workflow): Current levels acceptable

### Quality Metrics
- ✅ No test failures or flakiness
- ✅ All new tests have clear documentation
- ✅ Tests complete in <15 minutes (with optimization)
- ✅ Property-based tests for numerical stability

---

## Risks & Mitigation

### Risk: Coverage increase but tests are brittle
**Mitigation**:
- Focus on behavior, not implementation
- Use property-based testing for numerical code
- Test edge cases, not just happy paths

### Risk: Tests slow down CI
**Mitigation**:
- Mark expensive tests with `@pytest.mark.slow`
- Use `pytest-xdist` for parallelization
- Maintain 10-15 minute CI target

### Risk: Hard-to-test visualization code
**Mitigation**:
- Focus on import and basic rendering tests
- Use manual testing + example gallery
- Accept lower coverage for plotting code

### Risk: Scope creep
**Mitigation**:
- Stick to phased approach
- Track progress with Codecov
- Celebrate incremental wins

---

## Measurement & Tracking

### Daily
- Run tests locally: `pytest tests/ --cov=mfg_pde`
- Check coverage delta: `pytest --cov=mfg_pde --cov-report=term`

### Weekly
- Review Codecov dashboard
- Update progress in Issue #124
- Adjust priorities based on insights

### Milestones
- **Week 2**: Phase 1 complete (42%)
- **Week 4**: Phase 2 Part 1 (46%)
- **Week 6**: Phase 2 complete (50%)
- **Week 8**: Phase 3 complete (60%)

---

## Conclusion

This plan provides a **systematic, phased approach** to improving test coverage from 37% to 50-60% over 4-6 weeks. By focusing on:
1. **Quick wins** (zero-coverage utilities)
2. **High-impact core** (backends, config, geometry)
3. **Comprehensive validation** (RL, factory, performance)

We achieve maximum impact with minimum effort. The plan respects that some modules (visualization, workflow) are hard to test or experimental, and maintains reasonable expectations for those areas.

**Next Steps**:
1. Review and approve this plan
2. Create test file structure
3. Begin Phase 1 with utils/progress tests
4. Track progress with Codecov
