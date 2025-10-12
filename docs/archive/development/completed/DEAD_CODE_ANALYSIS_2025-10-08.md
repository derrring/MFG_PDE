# Dead Code Analysis Report

**Date**: 2025-10-08
**Tool**: Vulture 2.14 (static analyzer)
**Scope**: `mfg_pde/` package (241 Python files)

## Executive Summary

Analyzed the codebase for unused code including:
- Unused variables
- Unused imports
- Unused functions and classes
- Unused attributes

**Confidence Levels**:
- **100% confidence**: Definitively unused (36 issues)
- **90% confidence**: Very likely unused (7 issues)
- **60% confidence**: Possibly unused, needs investigation (many)

---

## High-Confidence Dead Code (80%+)

### Category 1: Unused Variables (100% confidence)

**Exception Handling Variables** (common pattern):
- `mfg_pde/backends/base_backend.py:250` - `exc_tb`
- `mfg_pde/backends/solver_wrapper.py:169` - `exc_tb`
- `mfg_pde/utils/logging/logger.py:519` - `exc_tb`
- `mfg_pde/utils/progress.py:88, 168` - `exc_tb` (2 instances)

**Impact**: Low - These are exception traceback variables in `except` blocks
**Action**: Can be replaced with `_` to indicate intentionally unused

**Neural Network Training Variables**:
- `mfg_pde/alg/neural/operator_learning/deeponet.py:437` - `fidelity_weight`
- `mfg_pde/alg/neural/operator_learning/operator_training.py:442` - `steps_per_epoch`
- `mfg_pde/alg/neural/operator_learning/operator_training.py:557` - `cache_solutions`

**Impact**: Medium - May indicate incomplete features or unused config options
**Action**: Investigate if features are planned or can be removed

**Solver Implementation Variables**:
- `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py:190` - `U_prev_picard`
- `mfg_pde/alg/reinforcement/algorithms/mean_field_q_learning.py:410` - `episode_num`
- `mfg_pde/alg/reinforcement/core/base_mfrl.py:260` - `episode_num`
- `mfg_pde/factory/solver_factory.py:81` - `amr_config`
- `mfg_pde/meta/optimization_meta.py:222` - `memory_limit`
- `mfg_pde/utils/numerical/convergence.py:994` - `original_info`
- `mfg_pde/utils/performance/optimization.py:593` - `num_threads`
- `mfg_pde/utils/solver_decorators.py:115` - `monitor_convergence`
- `mfg_pde/workflow/parameter_sweep.py:563-565` - 3 variables

**Impact**: Medium - Code quality issue, may indicate refactoring needed
**Action**: Review each case for removal or use

**Config/Legacy Variables**:
- `mfg_pde/compat/legacy_config.py:104` - `session_name`
- `mfg_pde/config/omegaconf_manager.py:48, 50, 84, 88` - `file` (4 instances)
- `mfg_pde/config/omegaconf_manager.py:451` - `schema_name`
- `mfg_pde/visualization/legacy_plotting.py:90` - `prefix`

**Impact**: Low - Legacy code, may be for future use
**Action**: Clean up if legacy features deprecated

### Category 2: Unused Imports (90% confidence)

**JAX Utilities**:
- `mfg_pde/utils/acceleration/jax_utils.py:26` - `jacfwd`, `jacrev`

**Impact**: Low - Auto-differentiation functions not currently used
**Action**: Remove if not needed, keep if planned for future use

**Numba Backend**:
- `mfg_pde/backends/numba_backend.py:22` - `njit`

**Impact**: Medium - JIT compilation import unused
**Action**: Either use or remove numba backend if not working

**OmegaConf**:
- `mfg_pde/config/omegaconf_manager.py:64` - `ListConfig`

**Impact**: Low - Config utility not used
**Action**: Remove if not needed

**Visualization Libraries**:
- `mfg_pde/visualization/interactive_plots.py:27` - `offline` (plotly)
- `mfg_pde/visualization/interactive_plots.py:36` - `transform` (holoviews)
- `mfg_pde/visualization/interactive_plots.py:37` - `curdoc`, `push_notebook` (bokeh)
- `mfg_pde/visualization/interactive_plots.py:41` - `Inferno256`, `Plasma256` (bokeh palettes)

**Impact**: Low - Optional visualization features
**Action**: Review interactive plotting needs, remove if unused

---

## Patterns and Recommendations

### Pattern 1: Exception Handling
**Current**:
```python
except Exception as e:
    exc_type, exc_value, exc_tb = sys.exc_info()
    # Only use exc_type and exc_value
```

**Recommended**:
```python
except Exception as e:
    exc_type, exc_value, _ = sys.exc_info()
    # or
    exc_type, exc_value = type(e), e
```

### Pattern 2: Unpacking with Unused Values
**Current**:
```python
objective_function, n_initial, n_iterations = get_config()
# Only use first value
```

**Recommended**:
```python
objective_function, *_ = get_config()
# or
objective_function = get_config()[0]
```

### Pattern 3: Unused Config Parameters
Many unused variables are from config dictionaries:
- May indicate incomplete features
- May be reserved for future use
- May be obsolete from refactoring

**Action**: Document with comments if planned, remove if obsolete

---

## Summary Statistics

**High Confidence Issues (80%+)**:
- 36 unused variables (100% confidence)
- 7 unused imports (90% confidence)
- Total: **43 high-confidence dead code items**

**Estimated Impact**:
- **Critical**: 0 (no broken functionality)
- **Medium**: ~15 issues (code quality, potential refactoring needed)
- **Low**: ~28 issues (minor cleanup, style improvements)

---

## Recommended Actions

### Immediate (Low-Hanging Fruit)
1. Replace unused `exc_tb` with `_` in exception handlers (5 locations)
2. Remove unused visualization imports if features not needed (6 imports)
3. Remove unused config imports (ListConfig)

### Short-Term Investigation
1. Review neural network training variables - incomplete features?
2. Check solver variables - refactoring artifacts?
3. Audit config/legacy code for obsolete features

### Long-Term Consideration
1. Add vulture to CI pipeline for continuous monitoring
2. Create whitelist for intentionally unused code
3. Document "reserved for future use" variables

---

## Next Steps

1. **Prioritize**: Start with 100% confidence items
2. **Verify**: Check each case in context before removing
3. **Test**: Run tests after cleanup
4. **Document**: Add comments for "planned future use" variables

**Estimated Cleanup Time**: 2-4 hours for high-confidence issues
