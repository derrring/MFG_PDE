# Phase 3 Migration Guide

**Version**: v0.8.x → v0.9.0
**Date**: 2025-11-03
**Applies to**: Users upgrading from Phase 2 (v0.8.x) to Phase 3 (v0.9.0)

---

## Overview

Phase 3 (v0.9.0) introduces a **unified architecture** that dramatically simplifies the MFG_PDE API while maintaining full backward compatibility. The key changes:

1. **Unified `solve_mfg()` interface** - Single entry point for all MFG problems
2. **Unified `SolverConfig` system** - Three patterns: YAML, Builder, Presets
3. **Unified `MFGProblem` class** - Replaces 5+ specialized problem classes
4. **Factory integration** - Automatic solver selection and configuration

**Migration effort**: Most code requires **minimal or no changes**. Phase 3 is designed for **incremental adoption**.

**Performance impact**: Negligible (<0.3 ms overhead, 0.03% of typical solve time)

---

## Quick Start: Minimal Migration

### Before (Phase 2)
```python
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.config import create_fast_config

# Manual setup
config = create_fast_config()
hjb = HJBFDMSolver(problem, config)
fp = FPFDMSolver(problem, config)
solver = FixedPointIterator(problem, hjb, fp, config.picard)
result = solver.solve()
```

### After (Phase 3)
```python
from mfg_pde import solve_mfg, ExampleMFGProblem

# Unified interface
problem = ExampleMFGProblem()
result = solve_mfg(problem, preset="fast")
```

**Lines of code**: 10+ → 2

---

## Migration Paths

### Path 1: Zero-Change (Backward Compatible)

Phase 3 **maintains all Phase 2 APIs** with deprecation warnings. Your existing code will work without modification:

```python
# Phase 2 code - still works in Phase 3
from mfg_pde.config.solver_config import create_fast_config
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator

config = create_fast_config()  # Works, with deprecation warning
solver = FixedPointIterator(...)  # Works
result = solver.solve()  # Works
```

**Action required**: None immediately. Update at your convenience.

**Deprecation timeline**: Phase 2 APIs supported through v0.10.x (at least 6 months).

---

### Path 2: Incremental Adoption (Recommended)

Adopt Phase 3 APIs gradually, starting with the highest-impact changes:

#### Step 1: Adopt `solve_mfg()` (Highest Impact)

**Before**:
```python
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator
solver = FixedPointIterator(problem, hjb_solver, fp_solver, picard_config)
result = solver.solve()
```

**After**:
```python
from mfg_pde import solve_mfg
result = solve_mfg(problem, preset="fast")
```

**Benefits**: Simplest API, automatic solver selection, type safety

#### Step 2: Adopt Unified Config (High Impact)

**Before**:
```python
from mfg_pde.config.solver_config import create_accurate_config
config = create_accurate_config()
```

**After**:
```python
from mfg_pde.config import presets
config = presets.accurate_solver()
```

**Benefits**: Clearer presets, builder API, YAML support

#### Step 3: Adopt Unified MFGProblem (Medium Impact)

**Before**:
```python
from mfg_pde.core import LQMFGProblem, CongestionMFGProblem
problem = LQMFGProblem(...)  # Specialized class
```

**After**:
```python
from mfg_pde import ExampleMFGProblem
problem = ExampleMFGProblem(...)  # Unified class
```

**Benefits**: Single problem class, auto-detection of problem type

---

### Path 3: Full Migration (For New Projects)

Use Phase 3 APIs exclusively:

```python
from mfg_pde import solve_mfg, ExampleMFGProblem
from mfg_pde.config import presets

# Define problem
problem = ExampleMFGProblem(nx=100, nt=100, T=1.0)

# Solve with preset
result = solve_mfg(problem, preset="accurate")

# Or use builder for custom config
from mfg_pde.config import ConfigBuilder
config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=3)
    .solver_fp(method="particle", num_particles=5000)
    .picard(max_iterations=100, tolerance=1e-8)
    .backend(backend_type="pytorch", device="cuda")
    .build()
)
result = solve_mfg(problem, config=config)
```

---

## API Changes Reference

### 1. Problem Classes

#### Before (Phase 2): Multiple Specialized Classes

```python
from mfg_pde.core import (
    LQMFGProblem,           # Linear-quadratic
    CongestionMFGProblem,   # Congestion problems
    PotentialMFGProblem,    # Potential field
    NetworkMFGProblem,      # Network MFG
    StochasticMFGProblem,   # Stochastic formulation
)

# Different classes for different problem types
lq_problem = LQMFGProblem(alpha=1.0, beta=1.0, ...)
congestion_problem = CongestionMFGProblem(coupling_function=..., ...)
```

#### After (Phase 3): Unified MFGProblem

```python
from mfg_pde import ExampleMFGProblem

# Single unified class - auto-detects problem type
problem = ExampleMFGProblem(
    hamiltonian=...,
    coupling=...,
    initial_distribution=...,
    # Type automatically detected from components
)
```

**Migration**:
- Most specialized problem classes → `ExampleMFGProblem`
- Problem type detected automatically from Hamiltonian/coupling
- Network and Variational problems retain specialized classes

**Compatibility**: Old problem classes still work, with deprecation warnings

---

### 2. Configuration System

#### Before (Phase 2): Factory Functions

```python
from mfg_pde.config.solver_config import (
    create_fast_config,
    create_accurate_config,
    create_research_config,
)

# Factory functions return configs
config = create_fast_config()
config = create_accurate_config()

# Or manual construction
from mfg_pde.config.solver_config import MFGSolverConfig, PicardConfig
config = MFGSolverConfig(
    picard=PicardConfig(max_iterations=50, tolerance=1e-6),
    # ... many parameters
)
```

#### After (Phase 3): Presets + Builder + YAML

**Option 1: Presets** (Recommended for most cases)
```python
from mfg_pde.config import presets

config = presets.fast_solver()        # Speed-optimized
config = presets.accurate_solver()    # Accuracy-optimized
config = presets.research_solver()    # Comprehensive output
```

**Option 2: Builder API** (For programmatic configuration)
```python
from mfg_pde.config import ConfigBuilder

config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp(method="particle", num_particles=5000)
    .picard(max_iterations=100, tolerance=1e-8, anderson_memory=5)
    .backend(backend_type="pytorch", device="cuda")
    .logging(level="INFO", progress_bar=True)
    .build()
)
```

**Option 3: YAML Files** (For reproducibility)
```yaml
# config.yaml
hjb:
  method: fdm
  accuracy_order: 2

fp:
  method: particle
  particle_config:
    num_particles: 5000

picard:
  max_iterations: 100
  tolerance: 1e-8
```

```python
from mfg_pde.config import load_solver_config

config = load_solver_config("config.yaml")
```

**Migration**:
- `create_fast_config()` → `presets.fast_solver()`
- `create_accurate_config()` → `presets.accurate_solver()`
- `create_research_config()` → `presets.research_solver()`
- Manual construction → `ConfigBuilder()` or YAML

---

### 3. Solver Creation

#### Before (Phase 2): Manual Instantiation

```python
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator

# Manual component creation
hjb_solver = HJBFDMSolver(problem, config)
fp_solver = FPFDMSolver(problem, config)

# Manual solver assembly
solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    picard_config=config.picard,
)

# Solve
result = solver.solve()
```

#### After (Phase 3): Unified `solve_mfg()`

**Pattern 1: With Preset String** (Simplest)
```python
from mfg_pde import solve_mfg

result = solve_mfg(problem, preset="fast")
```

**Pattern 2: With Config Object** (More control)
```python
from mfg_pde import solve_mfg
from mfg_pde.config import presets

config = presets.accurate_solver()
result = solve_mfg(problem, config=config)
```

**Pattern 3: Inline Builder** (Maximum flexibility)
```python
from mfg_pde import solve_mfg
from mfg_pde.config import ConfigBuilder

result = solve_mfg(
    problem,
    config=ConfigBuilder()
    .solver_hjb(method="fdm")
    .picard(max_iterations=50)
    .build()
)
```

**Migration**:
- All manual solver creation → `solve_mfg(problem, ...)`
- Factory functions still available in `mfg_pde.factory` for advanced use

---

### 4. Results API

#### Before (Phase 2): Solver-Specific Results

```python
result = solver.solve()

# Access fields
U = result.U  # HJB solution
M = result.M  # FP solution
converged = result.converged
iterations = result.iterations
```

#### After (Phase 3): Unified SolverResult

```python
result = solve_mfg(problem, preset="fast")

# Same field access - unchanged
U = result.U
M = result.M
converged = result.converged
iterations = result.iterations

# Enhanced with additional metadata
execution_time = result.execution_time
error_history_U = result.error_history_U
error_history_M = result.error_history_M
```

**Migration**: No changes required - result API is backward compatible

---

## Common Migration Scenarios

### Scenario 1: Research Scripts

**Before**:
```python
# research_script.py (Phase 2)
from mfg_pde.core import LQMFGProblem
from mfg_pde.config.solver_config import create_research_config
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator
from mfg_pde.visualization import plot_results

# Setup
problem = LQMFGProblem(Nx=100, Nt=100, T=1.0, alpha=1.0, beta=1.0)
config = create_research_config()

# Solve
hjb = HJBFDMSolver(problem, config)
fp = FPFDMSolver(problem, config)
solver = FixedPointIterator(problem, hjb, fp, config.picard)
result = solver.solve()

# Visualize
plot_results(result, problem)
```

**After**:
```python
# research_script.py (Phase 3)
from mfg_pde import solve_mfg, ExampleMFGProblem
from mfg_pde.visualization import plot_results

# Setup
problem = ExampleMFGProblem(nx=100, nt=100, T=1.0)

# Solve
result = solve_mfg(problem, preset="research")

# Visualize
plot_results(result, problem)
```

**Savings**: ~10 lines → ~5 lines (50% reduction)

---

### Scenario 2: Parameter Sweeps

**Before**:
```python
# parameter_sweep.py (Phase 2)
import numpy as np
from mfg_pde.core import LQMFGProblem
from mfg_pde.config.solver_config import create_fast_config
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator

results = []
for alpha in np.linspace(0.1, 2.0, 20):
    problem = LQMFGProblem(Nx=50, Nt=50, T=1.0, alpha=alpha, beta=1.0)
    config = create_fast_config()

    hjb = HJBFDMSolver(problem, config)
    fp = FPFDMSolver(problem, config)
    solver = FixedPointIterator(problem, hjb, fp, config.picard)

    result = solver.solve()
    results.append((alpha, result))
```

**After**:
```python
# parameter_sweep.py (Phase 3)
import numpy as np
from mfg_pde import solve_mfg, ExampleMFGProblem

results = []
for alpha in np.linspace(0.1, 2.0, 20):
    # Construct problem with varying parameter
    problem = ExampleMFGProblem(nx=50, nt=50, T=1.0, alpha=alpha)

    # Solve with preset
    result = solve_mfg(problem, preset="fast")
    results.append((alpha, result))
```

**Benefits**: Cleaner code, faster iteration, easier to parallelize

---

### Scenario 3: Custom Hamiltonians

**Before**:
```python
# custom_hamiltonian.py (Phase 2)
from mfg_pde.core import CustomMFGProblem
from mfg_pde.core.base_components import Hamiltonian

def custom_H(t, x, p, m):
    """Custom Hamiltonian: H(p, m) = |p|^2 / (2m)"""
    return 0.5 * (p**2) / (m + 1e-8)

hamiltonian = Hamiltonian(function=custom_H)
problem = CustomMFGProblem(
    domain=domain,
    hamiltonian=hamiltonian,
    initial_distribution=m0,
    terminal_cost=g,
)
```

**After**:
```python
# custom_hamiltonian.py (Phase 3)
from mfg_pde import ExampleMFGProblem
from mfg_pde.core import Hamiltonian

def custom_H(t, x, p, m, backend_module):
    """Custom Hamiltonian: H(p, m) = |p|^2 / (2m)"""
    return 0.5 * backend_module.sum(p**2, axis=-1) / (m + 1e-8)

hamiltonian = Hamiltonian(function=custom_H, signature="standard")
problem = ExampleMFGProblem(
    hamiltonian=hamiltonian,
    initial_distribution=m0,
    terminal_cost=g,
)
```

**Changes**:
- Added `backend_module` parameter to Hamiltonian (for backend compatibility)
- Added `signature="standard"` to specify signature type
- `CustomMFGProblem` → `ExampleMFGProblem`

---

### Scenario 4: Configuration Management

**Before** (Phase 2 - Limited options):
```python
# config.py (Phase 2)
from mfg_pde.config.solver_config import (
    MFGSolverConfig,
    PicardConfig,
    HJBConfig,
    FPConfig,
)

# Manual construction only
config = MFGSolverConfig(
    hjb=HJBConfig(method="fdm", accuracy_order=2),
    fp=FPConfig(method="fdm"),
    picard=PicardConfig(max_iterations=50, tolerance=1e-6),
)
```

**After** (Phase 3 - Three patterns):

**Option 1: Presets** (Quick prototyping)
```python
from mfg_pde.config import presets
config = presets.fast_solver()
```

**Option 2: Builder** (Programmatic)
```python
from mfg_pde.config import ConfigBuilder
config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp(method="fdm")
    .picard(max_iterations=50, tolerance=1e-6)
    .build()
)
```

**Option 3: YAML** (Reproducibility)
```yaml
# experiments/baseline.yaml
hjb:
  method: fdm
  accuracy_order: 2
fp:
  method: fdm
picard:
  max_iterations: 50
  tolerance: 1e-6
```

```python
from mfg_pde.config import load_solver_config
config = load_solver_config("experiments/baseline.yaml")
```

**Benefits**: Choose pattern based on use case, all three patterns supported

---

## Troubleshooting

### Issue 1: ImportError for Old APIs

**Symptom**:
```python
ImportError: cannot import name 'LQMFGProblem' from 'mfg_pde.core'
```

**Cause**: Specialized problem classes have been consolidated into `ExampleMFGProblem`

**Solution**:
```python
# Before
from mfg_pde.core import LQMFGProblem
problem = LQMFGProblem(...)

# After
from mfg_pde import ExampleMFGProblem
problem = ExampleMFGProblem(...)
```

---

### Issue 2: DeprecationWarning Messages

**Symptom**:
```
DeprecationWarning: create_fast_config() is deprecated, use presets.fast_solver()
```

**Cause**: Using Phase 2 APIs in Phase 3

**Solution**: Update to Phase 3 APIs (see migration paths above)

**Note**: Deprecation warnings do not prevent code from running. Update at your convenience.

---

### Issue 3: Configuration Validation Errors

**Symptom**:
```python
ValidationError: Unknown configuration parameter 'convergence_tolerance'
```

**Cause**: Some config parameters renamed or restructured in Phase 3

**Solution**: Use Phase 3 config system with correct parameter names
```python
# Before (Phase 2)
config = MFGSolverConfig(convergence_tolerance=1e-6)

# After (Phase 3)
from mfg_pde.config import ConfigBuilder
config = ConfigBuilder().picard(tolerance=1e-6).build()
```

**Common Renames**:
- `convergence_tolerance` → `picard.tolerance`
- `max_iter` → `picard.max_iterations`
- `anderson_depth` → `picard.anderson_memory`

---

### Issue 4: Type Errors with Hamiltonian Signature

**Symptom**:
```python
TypeError: Hamiltonian function has incompatible signature
```

**Cause**: Phase 3 requires explicit signature specification for backend compatibility

**Solution**: Add `signature` parameter and `backend_module` to Hamiltonian
```python
# Before (Phase 2)
def H(t, x, p, m):
    return 0.5 * p**2

hamiltonian = Hamiltonian(function=H)

# After (Phase 3)
def H(t, x, p, m, backend_module):
    return 0.5 * backend_module.sum(p**2, axis=-1)

hamiltonian = Hamiltonian(function=H, signature="standard")
```

---

## Performance Considerations

### Phase 3 API Overhead

Based on comprehensive benchmarking (see `docs/development/PHASE_3_PERFORMANCE_VALIDATION.md`):

| Component | Overhead | Impact |
|:----------|:---------|:-------|
| Config creation | 7.6 µs | Negligible |
| Problem construction | 238 µs | One-time cost |
| Factory overhead | 0.05 ms | Negligible |
| **Total per solve_mfg()** | **~0.3 ms** | **0.03% of 1s solve** |

**Conclusion**: Phase 3 API overhead is **negligible** for all practical purposes.

### Optimization Tips

1. **Reuse Problem Instances**: For parameter sweeps, reuse `MFGProblem` instances when possible:
   ```python
   problem = ExampleMFGProblem(nx=100, nt=100)  # Once

   for param in parameter_sweep:
       result = solve_mfg(problem, preset="fast", **param)  # Reuse problem
   ```

2. **Choose Appropriate Preset**:
   - `fast`: Prototyping, testing (low accuracy)
   - `balanced`: Production (good accuracy, reasonable speed)
   - `accurate`: Publication-quality results (high accuracy, slower)

3. **Use YAML for Experiments**: Reproducibility without code changes:
   ```python
   config = load_solver_config(f"experiments/{experiment_name}.yaml")
   result = solve_mfg(problem, config=config)
   ```

---

## Backward Compatibility

### What Still Works (Phase 2 APIs)

✅ **Fully supported** with deprecation warnings:
- `create_fast_config()`, `create_accurate_config()`, `create_research_config()`
- `FixedPointIterator`, `HJBFDMSolver`, `FPFDMSolver` (manual instantiation)
- Specialized problem classes (`LQMFGProblem`, `CongestionMFGProblem`, etc.)
- Old result API (all fields unchanged)

✅ **Examples still work**: All Phase 2 examples run without modification

✅ **Tests still pass**: Phase 2 test suite passes in Phase 3

### What Changed (Breaking Changes)

❌ **Removed**:
- None - Phase 3 is fully backward compatible

⚠️ **Deprecated** (will be removed in v0.11.0):
- Factory functions in `mfg_pde.config.solver_config` (use `presets` instead)
- Specialized problem classes in `mfg_pde.core` (use `ExampleMFGProblem` instead)
- Manual solver instantiation patterns (use `solve_mfg()` instead)

### Deprecation Timeline

- **v0.9.0** (Current): Phase 2 APIs work with deprecation warnings
- **v0.10.x**: Phase 2 APIs still supported (6+ months)
- **v0.11.0**: Phase 2 APIs removed (estimated Q2 2026)

---

## Getting Help

### Documentation

- **Phase 3 Overview**: `docs/development/PHASE_3_INTEGRATION_VERIFICATION.md`
- **Performance Validation**: `docs/development/PHASE_3_PERFORMANCE_VALIDATION.md`
- **API Reference**: `docs/api/` (updated for Phase 3)
- **Tutorials**: `docs/tutorials/` (new Phase 3 tutorials)

### Examples

- **Phase 3 Examples**: `examples/basic/solve_mfg_demo.py`
- **Migration Examples**: This document (above)
- **All Examples**: `examples/basic/`, `examples/advanced/`

### Issues

- **GitHub Issues**: https://github.com/anthropics/mfg-pde/issues
- **Migration Questions**: Label with `migration` + `phase-3`

---

## Summary Checklist

Use this checklist to track your migration progress:

### Quick Wins (High Impact, Low Effort)

- [ ] Replace `solver.solve()` with `solve_mfg(problem, preset="...")`
- [ ] Replace `create_fast_config()` with `presets.fast_solver()`
- [ ] Replace specialized problem classes with `ExampleMFGProblem`

### Medium Priority (Medium Impact, Medium Effort)

- [ ] Adopt `ConfigBuilder` for programmatic configuration
- [ ] Convert experiment configs to YAML files
- [ ] Update Hamiltonian signatures (add `backend_module`, `signature`)

### Low Priority (Nice-to-Have)

- [ ] Update imports to use `from mfg_pde import ...` pattern
- [ ] Remove deprecation warnings from logs
- [ ] Adopt Phase 3 naming conventions throughout codebase

---

**Migration Guide Version**: 1.0
**Last Updated**: 2025-11-03
**Applies to**: MFG_PDE v0.9.0
**Next Review**: v0.10.0 release
