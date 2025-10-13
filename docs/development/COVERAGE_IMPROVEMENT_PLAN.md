# Test Coverage Improvement Plan

**Date**: 2025-10-13
**Current Coverage**: 46% (14,797/32,500 lines)
**Target Coverage**: 60% (Phase 1), 75% (Phase 2)
**Status**: Planning

---

## Executive Summary

The MFG_PDE codebase currently has **46% test coverage** with 2,255 tests passing. While core functionality (geometry, solvers, workflows) has excellent coverage (80-100%), several important modules remain completely untested (0% coverage).

**Key Findings**:
- ✅ **Well-tested**: Core geometry (domains, grids), solver results, workflow decorators
- ⚠️ **Partially tested**: Neural solvers (12-14%), reinforcement learning (8-15%), visualization
- ❌ **Untested**: Hooks system, meta-programming, type system, CLI, benchmarking tools

**Strategy**: Focus on high-impact, production-critical modules first, then expand to experimental features.

---

## Coverage Breakdown by Category

### 1. Core MFG Functionality (✅ Excellent: 80-100%)

**Geometry & Domains**:
- `domain_1d.py` - **100%** (21/21 lines)
- `simple_grid.py` - **99%** (146/147 lines)
- `tensor_product_grid.py` - **96%** (90/94 lines)
- `domain_2d.py` - **84%** (208/247 lines)

**Solver Infrastructure**:
- `solver_result.py` - **97%** (342/351 lines)
- `solver_decorators.py` - **96%** (108/112 lines)
- `solver_config.py` - **99%** (188/190 lines)

**Backend System**:
- `numpy_backend.py` - **100%** (127/127 lines)
- `base_backend.py` - **100%** (33/33 lines)
- `array_wrapper.py` - **96%** (125/130 lines)

**Assessment**: Core MFG infrastructure is production-ready with excellent test coverage.

---

### 2. Numerical Algorithms (⚠️ Mixed: 60-95%)

**Well-Covered**:
- `hjb_fdm.py` - **95%** (36/38 lines)
- `particle_utils.py` - **94%** (85/90 lines)
- `network_mfg_solver.py` - **89%** (16/18 lines)

**Needs Attention**:
- `convergence.py` - **16%** (66/411 lines) ⚠️ HIGH PRIORITY
- `anderson_acceleration.py` - **68%** (53/78 lines)
- `monte_carlo.py` - **40%** (90/223 lines)

**Completely Untested (0%)**:
- `sinkhorn_solver.py` - 0% (213 lines)
- `wasserstein_solver.py` - 0% (167 lines)
- `mcmc.py` - 0% (282 lines)

---

### 3. Neural Network Solvers (❌ Poor: 8-14%)

**Current State**:
- `mfg_pinn_solver.py` - **12%** (38/317 lines)
- `fp_pinn_solver.py` - **14%** (31/222 lines)
- `core/utils.py` - **11%** (20/189 lines)

**Challenge**: Neural solvers require PyTorch/JAX dependencies and GPU testing infrastructure.

**Strategy**:
- Phase 1: Test core initialization and architecture validation
- Phase 2: Test forward pass with simple problems
- Phase 3: End-to-end training tests (optional, expensive)

---

### 4. Reinforcement Learning (❌ Poor: 8-15%)

**Current State**:
- `base_mfrl.py` - **0%** (112 lines)
- `multi_population_q_learning.py` - **0%** (165 lines)
- `continuous_action_maze_env.py` - **0%** (166 lines)
- `multi_population_maze_env.py` - **0%** (289 lines)
- `trainer.py` - **8%** (9/107 lines)
- `multi_sac.py` - **15%** (18/120 lines)

**Total Untested**: ~1,100 lines (0% coverage)

**Assessment**: RL module is experimental research code. Testing RL algorithms requires:
- Environment validation
- Policy/value network tests
- Training loop tests (expensive)
- Multi-agent coordination tests

---

### 5. Utilities & Infrastructure (⚠️ Mixed: 10-95%)

**Excellent Coverage**:
- `dependencies.py` - **95%** (52/55 lines)
- `hdf5_utils.py` - **86%** (130/152 lines)
- `sparse_operations.py` - **81%** (190/235 lines)
- `torch_utils.py` - **81%** (79/98 lines)

**Critical Gaps**:
- `cli.py` - **8%** (17/203 lines) ⚠️ HIGH PRIORITY
- `experiment_manager.py` - **10%** (17/178 lines) ⚠️ HIGH PRIORITY
- `convergence.py` - **16%** (66/411 lines) ⚠️ HIGH PRIORITY
- `logging/analysis.py` - **9%** (20/222 lines)
- `logging/decorators.py` - **11%** (20/183 lines)
- `memory_management.py` - **27%** (35/131 lines)

---

### 6. Experimental Features (❌ Untested: 0%)

**Complete Testing Gaps**:

**Hooks System** (0% - 968 lines untested):
- `hooks/composition.py` - 198 lines
- `hooks/debug.py` - 239 lines
- `hooks/visualization.py` - 233 lines
- `hooks/control_flow.py` - 138 lines
- `hooks/extensions.py` - 118 lines
- `hooks/base.py` - 11 lines
- `hooks/__init__.py` - 31 lines

**Meta-Programming** (0% - 746 lines untested):
- `meta/optimization_meta.py` - 230 lines
- `meta/type_system.py` - 201 lines
- `meta/mathematical_dsl.py` - 177 lines
- `meta/code_generation.py` - 138 lines

**Type System** (0% - 148 lines untested):
- `types/solver_types.py` - 62 lines
- `types/state.py` - 49 lines
- `types/arrays.py` - 28 lines
- `types/protocols.py` - 4 lines

**High-Dimensional Benchmarks** (0% - 499 lines untested):
- `benchmarks/highdim_benchmark_suite.py` - 499 lines

**Compatibility Layer** (0% - 154 lines untested):
- `compat/legacy_config.py` - 48 lines
- `compat/legacy_problems.py` - 46 lines
- `compat/legacy_solvers.py` - 42 lines
- `compat/__init__.py` - 18 lines

**Assessment**: Experimental features can remain untested initially, but should be tested before promotion to stable API.

---

## Coverage Improvement Strategy

### Phase 1: Critical Infrastructure (Target: 55% → 60%)

**Priority 1A: CLI & User-Facing Tools** (~200 lines, HIGH IMPACT)
```python
# mfg_pde/utils/cli.py (8% → 60%)
- Test command parsing and argument validation
- Test solver configuration from CLI
- Test output formatting and reporting
- Test error handling for invalid inputs

# Impact: Improves user experience reliability
# Effort: 2-3 days
# Tests needed: 15-20 test cases
```

**Priority 1B: Experiment Management** (~160 lines, HIGH IMPACT)
```python
# mfg_pde/utils/experiment_manager.py (10% → 70%)
- Test experiment creation and configuration
- Test result logging and persistence
- Test experiment comparison and analysis
- Test parallel experiment execution

# Impact: Critical for research workflows
# Effort: 2-3 days
# Tests needed: 20-25 test cases
```

**Priority 1C: Convergence Analysis** (~345 lines, HIGH IMPACT)
```python
# mfg_pde/utils/numerical/convergence.py (16% → 60%)
- Test convergence criteria validation
- Test error estimation methods
- Test adaptive step size selection
- Test convergence history tracking

# Impact: Affects solver reliability
# Effort: 3-4 days
# Tests needed: 25-30 test cases
```

**Total Phase 1**: ~705 lines to cover, estimated **1-2 weeks**

---

### Phase 2: Algorithm Completeness (Target: 60% → 70%)

**Priority 2A: Optimal Transport Solvers** (~380 lines, MEDIUM IMPACT)
```python
# mfg_pde/alg/optimization/optimal_transport/sinkhorn_solver.py (0% → 80%)
# mfg_pde/alg/optimization/optimal_transport/wasserstein_solver.py (0% → 80%)
- Test solver initialization and parameter validation
- Test convergence with known solutions
- Test numerical stability edge cases
- Test integration with MFG problems

# Impact: Expands algorithmic capabilities
# Effort: 3-4 days
# Tests needed: 30-40 test cases
```

**Priority 2B: Monte Carlo & MCMC** (~505 lines, MEDIUM IMPACT)
```python
# mfg_pde/utils/numerical/monte_carlo.py (40% → 80%)
# mfg_pde/utils/numerical/mcmc.py (0% → 70%)
- Test sampling accuracy with known distributions
- Test convergence diagnostics
- Test variance reduction techniques
- Test parallel sampling

# Impact: Improves stochastic solver reliability
# Effort: 4-5 days
# Tests needed: 35-45 test cases
```

**Priority 2C: Memory Management** (~96 lines, MEDIUM IMPACT)
```python
# mfg_pde/utils/memory_management.py (27% → 80%)
- Test memory monitoring accuracy
- Test memory limit enforcement
- Test cleanup and garbage collection
- Test cross-platform compatibility

# Impact: Prevents out-of-memory failures
# Effort: 1-2 days
# Tests needed: 10-15 test cases
```

**Total Phase 2**: ~981 lines to cover, estimated **2-3 weeks**

---

### Phase 3: Neural & RL Infrastructure (Target: 70% → 75%)

**Priority 3A: Neural Solver Core** (~350 lines, RESEARCH IMPACT)
```python
# Neural PINN solvers (12-14% → 60%)
- Test network architecture construction
- Test loss function computation
- Test optimizer configuration
- Test training checkpointing

# Impact: Enables neural MFG research
# Effort: 4-5 days
# Tests needed: 30-40 test cases
# Note: Requires PyTorch/JAX in test environment
```

**Priority 3B: Reinforcement Learning Environments** (~500 lines, RESEARCH IMPACT)
```python
# RL environments (0% → 70%)
- Test environment initialization and reset
- Test action/observation space validation
- Test reward computation
- Test multi-agent coordination

# Impact: Enables RL-based MFG research
# Effort: 5-6 days
# Tests needed: 40-50 test cases
```

**Priority 3C: Visualization Modules** (~350 lines, QUALITY OF LIFE)
```python
# Visualization modules (14-29% → 60%)
- Test plot generation without display
- Test data preparation and formatting
- Test export functionality
- Test error handling for missing data

# Impact: Improves result presentation quality
# Effort: 3-4 days
# Tests needed: 25-35 test cases
```

**Total Phase 3**: ~1,200 lines to cover, estimated **3-4 weeks**

---

### Phase 4: Experimental Features (Optional: 75% → 80%+)

**Hooks System** (~968 lines untested):
- Experimental feature for advanced users
- Testing hooks requires integration test infrastructure
- **Recommendation**: Test before promoting to stable API
- **Effort**: 2-3 weeks for comprehensive coverage

**Meta-Programming** (~746 lines untested):
- Advanced compile-time optimization features
- Requires specialized testing infrastructure
- **Recommendation**: Test critical paths only
- **Effort**: 1-2 weeks for essential coverage

**Type System** (~148 lines untested):
- Type protocols and abstract interfaces
- Tested indirectly through implementations
- **Recommendation**: Add mypy strict mode tests
- **Effort**: 2-3 days

**High-Dimensional Benchmarks** (~499 lines untested):
- Performance benchmarking suite
- Requires computational resources
- **Recommendation**: Test construction, not execution
- **Effort**: 3-4 days

---

## Testing Infrastructure Recommendations

### 1. Test Organization

```
tests/
├── unit/                          # Unit tests (fast, isolated)
│   ├── algorithms/
│   ├── backends/
│   ├── config/
│   ├── geometry/
│   └── utils/
├── integration/                   # Integration tests (slower)
│   ├── solver_pipelines/
│   ├── workflow_tests/
│   └── end_to_end/
├── benchmarks/                    # Performance benchmarks
│   └── algorithm_comparisons/
└── regression/                    # Regression test suite
    └── known_solutions/
```

### 2. Testing Best Practices

**Unit Test Template**:
```python
# tests/unit/utils/test_cli.py
import pytest
from mfg_pde.utils.cli import parse_args, create_solver_from_cli

class TestCLIParsing:
    def test_basic_argument_parsing(self):
        """Test CLI parses basic solver arguments correctly."""
        args = parse_args(['--solver', 'fixed_point', '--Nx', '100'])
        assert args.solver == 'fixed_point'
        assert args.Nx == 100

    def test_invalid_solver_type_raises_error(self):
        """Test CLI rejects invalid solver types."""
        with pytest.raises(ValueError, match="Invalid solver"):
            parse_args(['--solver', 'nonexistent'])

    def test_solver_creation_from_cli(self):
        """Test solver can be created from CLI arguments."""
        args = parse_args(['--solver', 'fixed_point'])
        solver = create_solver_from_cli(args)
        assert solver is not None
```

### 3. Coverage Measurement

**Pre-commit Hook**:
```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest --cov=mfg_pde --cov-report=term-missing --cov-fail-under=46
```

**CI/CD Integration**:
- Upload coverage to Codecov on every PR
- Require coverage not to decrease
- Generate coverage reports in HTML for review

### 4. Test Performance

**Current Metrics**:
- Total tests: 2,255
- Test duration: 222 seconds (~3.7 minutes)
- Average: 0.1 seconds per test

**Target Metrics** (after Phase 3):
- Total tests: 3,500-4,000
- Test duration: < 10 minutes
- Maintain fast feedback loop

---

## Implementation Roadmap

### Week 1-2: Phase 1A + 1B (CLI & Experiment Management)
- [ ] Create `tests/unit/utils/test_cli.py`
- [ ] Create `tests/unit/utils/test_experiment_manager.py`
- [ ] Achieve 60% coverage on CLI module
- [ ] Achieve 70% coverage on experiment manager
- [ ] Update CI/CD to track coverage trends

### Week 3-4: Phase 1C (Convergence Analysis)
- [ ] Create `tests/unit/numerical/test_convergence.py`
- [ ] Test all convergence criteria methods
- [ ] Test error estimation functions
- [ ] Test adaptive algorithms
- [ ] Achieve 60% coverage on convergence module

### Week 5-7: Phase 2A + 2B (Optimal Transport & Monte Carlo)
- [ ] Create `tests/unit/optimization/test_sinkhorn_solver.py`
- [ ] Create `tests/unit/optimization/test_wasserstein_solver.py`
- [ ] Create `tests/unit/numerical/test_monte_carlo.py`
- [ ] Create `tests/unit/numerical/test_mcmc.py`
- [ ] Achieve 70-80% coverage on these modules

### Week 8-9: Phase 2C + Phase 3A (Memory & Neural)
- [ ] Create `tests/unit/utils/test_memory_management.py`
- [ ] Create `tests/unit/neural/test_pinn_solvers.py`
- [ ] Test memory monitoring and limits
- [ ] Test neural network initialization
- [ ] Add PyTorch to test dependencies

### Week 10-12: Phase 3B + 3C (RL & Visualization)
- [ ] Create `tests/unit/reinforcement/test_environments.py`
- [ ] Create `tests/unit/visualization/test_plots.py`
- [ ] Test RL environments without training
- [ ] Test plot generation without display
- [ ] Achieve target 75% overall coverage

---

## Monitoring & Validation

### Coverage Tracking
```bash
# Weekly coverage report
coverage report --sort=cover --skip-covered > coverage_$(date +%Y%m%d).txt

# Coverage trend visualization
coverage html -d htmlcov/
# View: htmlcov/index.html
```

### Metrics to Track
1. **Overall coverage percentage**
2. **Coverage by module category**
3. **Number of untested files**
4. **Test execution time**
5. **Number of skipped tests**

### Success Criteria

**Phase 1 Complete**:
- ✅ Overall coverage ≥ 60%
- ✅ CLI coverage ≥ 60%
- ✅ Experiment manager coverage ≥ 70%
- ✅ Convergence analysis coverage ≥ 60%

**Phase 2 Complete**:
- ✅ Overall coverage ≥ 70%
- ✅ Optimal transport solvers ≥ 80%
- ✅ Monte Carlo/MCMC ≥ 70-80%
- ✅ Memory management ≥ 80%

**Phase 3 Complete**:
- ✅ Overall coverage ≥ 75%
- ✅ Neural solvers ≥ 60%
- ✅ RL environments ≥ 70%
- ✅ Visualization ≥ 60%

---

## Known Limitations

### 1. Neural Network Testing
**Challenge**: Neural solvers require GPU and long training times.
**Mitigation**: Test initialization and architecture only, skip training convergence tests.

### 2. Reinforcement Learning Testing
**Challenge**: RL training is stochastic and slow.
**Mitigation**: Test environment dynamics and policy evaluation, skip full training loops.

### 3. Visualization Testing
**Challenge**: Plot generation requires display backend.
**Mitigation**: Use non-interactive backends (Agg) and test data preparation only.

### 4. High-Dimensional Benchmarks
**Challenge**: Benchmarks require significant computational resources.
**Mitigation**: Test with small problem sizes, skip full benchmarking runs.

---

## Appendix: Quick Reference

### Coverage by Priority

**🔴 CRITICAL (Must fix)**:
- CLI: 8% → 60% (~190 lines)
- Experiment Manager: 10% → 70% (~160 lines)
- Convergence Analysis: 16% → 60% (~345 lines)

**🟡 IMPORTANT (Should fix)**:
- Optimal Transport: 0% → 80% (~380 lines)
- Monte Carlo/MCMC: 0-40% → 70-80% (~505 lines)
- Memory Management: 27% → 80% (~96 lines)

**🟢 NICE TO HAVE (Can defer)**:
- Neural Solvers: 12-14% → 60% (~350 lines)
- RL Environments: 0% → 70% (~500 lines)
- Visualization: 14-29% → 60% (~350 lines)

**⚪ OPTIONAL (Research features)**:
- Hooks: 0% → TBD (~968 lines)
- Meta-programming: 0% → TBD (~746 lines)
- Type system: 0% → TBD (~148 lines)

### Coverage Commands

```bash
# Run tests with coverage
pytest --cov=mfg_pde --cov-report=term-missing tests/

# Generate HTML report
pytest --cov=mfg_pde --cov-report=html tests/
open htmlcov/index.html

# Check specific module
pytest --cov=mfg_pde.utils.cli --cov-report=term tests/unit/utils/

# Upload to Codecov (in CI)
bash <(curl -s https://codecov.io/bash)
```

---

**Document Status**: Planning
**Target Start**: Week of 2025-10-14
**Estimated Completion**: 12 weeks for 75% coverage
**Owner**: Development team

---

*This plan prioritizes production-critical infrastructure first, then expands to research algorithms and experimental features. Coverage goals are achievable within 3 months with focused effort.*
