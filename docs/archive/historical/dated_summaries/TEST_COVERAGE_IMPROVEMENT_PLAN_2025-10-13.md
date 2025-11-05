# Test Coverage Improvement Plan

**Date**: 2025-10-13
**Current Overall Coverage**: 46% (17,704 / 32,500 statements missed)
**Goal**: Increase to 55-60% through targeted high-impact testing

## Executive Summary

This document outlines a strategic plan to improve MFG_PDE test coverage from 46% to 55-60% by targeting high-impact modules that are currently untested or undertested. The plan prioritizes:

1. **Core functionality** used by many other modules
2. **Production-ready code** (not experimental features)
3. **High statement count** modules for maximum impact
4. **Critical algorithms** that affect correctness

## Current Coverage Status by Module

### Well-Tested Modules (>80% coverage)
✅ **Geometry** (54% → improved from 16% in Phase 2.3)
- domain_1d.py: 100%
- boundary_conditions_1d.py: 99%
- simple_grid.py: 99%
- tensor_product_grid.py: 96%
- boundary_conditions_2d.py: 93%
- boundary_conditions_3d.py: 91%

✅ **Core Solvers** (well-tested in Phase 2.2a)
- HJB solvers: 51-95%
- FP solvers: 83-94%
- MFG solvers: 62-89%

✅ **High-Level Utils**
- solver_result.py: 97%
- solver_decorators.py: 96%
- dependencies.py: 95%

### Modules Requiring Tests (Priority Order)

## Priority 1: Core Infrastructure (High Impact, Low Coverage)

### 1.1 Common Noise MFG Solver (48% → Target: 75%)
**File**: `mfg_pde/alg/numerical/stochastic/common_noise_solver.py`
**Current**: 75/144 statements missed
**Impact**: ⭐⭐⭐⭐⭐ (Critical for stochastic MFG)

**Test Plan**:
- Sequential vs parallel Monte Carlo execution
- Noise path generation and sampling
- Conditional MFG problem solving
- Convergence analysis and error estimation
- Edge cases: K=1 sample, large K, parallel overhead

**Estimated tests**: 25-30

### 1.2 Reinforcement Learning Algorithms (79-95% → Maintain)
**Files**: `mfg_pde/alg/reinforcement/algorithms/`
**Current**: Already well-tested (91-95% for most)
**Action**: Add edge case tests for multi-population algorithms

**Estimated tests**: 10-15 additional

### 1.3 Parameter Sweep & Workflow (70-71% → Target: 85%)
**Files**:
- `parameter_sweep.py`: 80/265 statements missed
- `workflow_manager.py`: 101/344 statements missed

**Impact**: ⭐⭐⭐⭐ (Used by researchers for batch experiments)

**Test Plan**:
- Sequential, thread, and process execution modes
- Parameter combinations and grid search
- Result aggregation and export
- Error handling and partial failures
- Integration with experiment tracker

**Estimated tests**: 30-40

## Priority 2: Untested Core Modules (0-20% coverage)

### 2.1 Optimal Transport Solvers (0% → Target: 60%)
**Files**:
- `sinkhorn_solver.py`: 213 statements (0%)
- `wasserstein_solver.py`: 167 statements (0%)

**Impact**: ⭐⭐⭐ (Important for advanced MFG formulations)

**Why untested**: Likely not used in current examples/workflows
**Test Plan**:
- Sinkhorn algorithm convergence
- Wasserstein distance computation
- Entropic regularization effects
- Mass transportation constraints
- Performance benchmarks

**Estimated tests**: 35-45

### 2.2 Neural Network Solvers (10-22% → Target: 50%)
**Files**:
- `mfg_pde/alg/neural/pinn_solvers/`: ~900 statements (10-20% coverage)
- `mfg_pde/alg/neural/dgm/`: ~800 statements (20-47% coverage)

**Impact**: ⭐⭐⭐⭐ (Growing area of research)

**Test Plan**:
- PINN architecture initialization
- Loss function computation
- Training loop (short runs)
- Adaptive sampling
- Integration with PyTorch/JAX backends

**Estimated tests**: 50-60

### 2.3 Compatibility Layer (0% → Target: 70%)
**File**: `mfg_pde/alg/compatibility.py` (46 statements, 0%)

**Impact**: ⭐⭐ (Backward compatibility for old code)

**Test Plan**:
- Old API → new API translation
- Deprecation warnings
- Parameter mapping
- Error messages for removed features

**Estimated tests**: 15-20

## Priority 3: Utility Modules (Mixed Coverage)

### 3.1 Monte Carlo Integration (40% → Target: 70%)
**File**: `mfg_pde/utils/numerical/monte_carlo.py`
**Current**: 133/223 statements missed

**Impact**: ⭐⭐⭐⭐ (Used by stochastic solvers)

**Test Plan**:
- MC integration with different samplers
- Variance reduction techniques
- Parallel MC execution
- Convergence rate validation
- Integration with common_noise_solver

**Estimated tests**: 20-25

### 3.2 Performance Monitoring (24% → Target: 60%)
**File**: `mfg_pde/utils/performance/monitoring.py`
**Current**: 185/244 statements missed

**Impact**: ⭐⭐⭐ (Useful for optimization and profiling)

**Test Plan**:
- Memory tracking
- Timing and profiling
- Performance report generation
- Integration with workflow manager

**Estimated tests**: 20-25

### 3.3 Logging System (9-29% → Target: 50%)
**Files**: `mfg_pde/utils/logging/`
- logger.py: 182/255 missed (29%)
- decorators.py: 163/183 missed (11%)
- analysis.py: 202/222 missed (9%)

**Impact**: ⭐⭐⭐ (Better debugging and experiment tracking)

**Test Plan**:
- Logger configuration
- Log level filtering
- Decorator usage
- Log analysis and parsing
- Integration with research_logs

**Estimated tests**: 30-40

## Priority 4: Advanced Features (Low Priority)

### 4.1 Hooks System (0% → Optional)
**Module**: `mfg_pde/hooks/` (957 statements, 0%)
**Impact**: ⭐ (Advanced metaprogramming feature, not critical)
**Action**: Defer until hooks are used in production workflows

### 4.2 Meta Programming (0% → Optional)
**Module**: `mfg_pde/meta/` (746 statements, 0%)
**Impact**: ⭐ (Experimental code generation features)
**Action**: Defer until meta features stabilize

### 4.3 CLI Tools (8% → Optional)
**File**: `mfg_pde/utils/cli.py` (186/203 statements missed)
**Impact**: ⭐⭐ (Useful but not critical)
**Action**: Add basic CLI tests if time permits

## Implementation Phases

### Phase 2.4: Core Infrastructure Tests (~100 tests)
**Duration**: 2-3 days
**Modules**:
1. CommonNoiseSolver (25-30 tests)
2. ParameterSweep & Workflow (30-40 tests)
3. Monte Carlo Integration (20-25 tests)
4. RL algorithm edge cases (10-15 tests)

**Expected Coverage**: 46% → 50%

### Phase 2.5: Optimal Transport & Neural Solvers (~100 tests)
**Duration**: 3-4 days
**Modules**:
1. Sinkhorn & Wasserstein solvers (35-45 tests)
2. PINN solvers (30-35 tests)
3. DGM solvers (20-25 tests)
4. Compatibility layer (15-20 tests)

**Expected Coverage**: 50% → 54%

### Phase 2.6: Utility Modules (~80 tests)
**Duration**: 2-3 days
**Modules**:
1. Performance monitoring (20-25 tests)
2. Logging system (30-40 tests)
3. Convergence analysis (15-20 tests)
4. Other utils (15-20 tests)

**Expected Coverage**: 54% → 57%

### Phase 2.7: Polish & Documentation (~50 tests)
**Duration**: 1-2 days
**Activities**:
1. Fill remaining gaps in high-coverage modules
2. Add integration tests for workflows
3. Documentation updates
4. Coverage report generation

**Expected Coverage**: 57% → 60%

## Success Metrics

### Quantitative Targets
- **Overall Coverage**: 46% → 60% (+14 percentage points)
- **Statements Tested**: +4,550 statements
- **New Tests**: ~330 tests across all phases
- **Critical Module Coverage**: All Priority 1-2 modules >50%

### Qualitative Goals
- ✅ All production-ready algorithms have >50% coverage
- ✅ Stochastic MFG workflow fully tested
- ✅ Neural solver architecture validated
- ✅ Workflow automation tested end-to-end
- ✅ Performance monitoring operational

## Risk Mitigation

### Challenge 1: Neural Solver Dependencies
**Risk**: PyTorch/JAX optional dependencies
**Mitigation**: Use `pytest.importorskip` and conditional testing

### Challenge 2: Long-Running Tests
**Risk**: Neural network training takes time
**Mitigation**:
- Use small networks and few iterations
- Mark expensive tests with `@pytest.mark.slow`
- Run full training in integration tests only

### Challenge 3: Parallel Execution Tests
**Risk**: Flaky tests due to timing/parallelism
**Mitigation**:
- Test logic separately from parallel execution
- Use deterministic random seeds
- Increase timeouts for CI

### Challenge 4: Optimal Transport Convergence
**Risk**: Sinkhorn may not converge for all test cases
**Mitigation**:
- Test with well-conditioned problems
- Add explicit convergence checks
- Document known failure modes

## Documentation Plan

### Test Documentation
1. **Test README**: Overview of test organization
2. **Module Test Guides**: Best practices for each major module
3. **Coverage Reports**: Regular coverage analysis

### Code Documentation
1. **Docstring Updates**: Ensure all tested functions have examples
2. **Theory Docs**: Link tests to mathematical formulations
3. **Example Integration**: Show how to use tested features

## Relation to Strategic Roadmap

This plan aligns with **Phase 2** of the Strategic Development Roadmap:
- **Phase 2.1-2.3**: ✅ COMPLETED (solvers, config, geometry)
- **Phase 2.4-2.7**: 🔄 IN PROGRESS (this plan)
- **Phase 3**: Awaiting Phase 2 completion

By end of Phase 2.7, we will have:
- Comprehensive test coverage for all core features
- Solid foundation for Phase 3 (GPU acceleration, advanced features)
- High confidence in package stability and correctness

## Next Steps

1. ✅ **Immediate**: Create Phase 2.4 branch and begin CommonNoiseSolver tests
2. **Short-term**: Complete Phase 2.4 infrastructure tests (2-3 days)
3. **Medium-term**: Execute Phases 2.5-2.6 (5-7 days)
4. **Long-term**: Polish and reach 60% coverage target (1-2 days)

---

**Total Estimated Duration**: 10-14 days for 46% → 60% coverage improvement
**Priority**: HIGH - Blocks Phase 3 development
**Owner**: Development team
**Status**: 📋 PLANNING COMPLETE, READY TO EXECUTE
