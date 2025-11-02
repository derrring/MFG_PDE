# Changelog

All notable changes to MFG_PDE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2025-11-03

### Phase 3 Complete: Unified Architecture

Major architecture refactoring completing Phase 3.1 (MFGProblem), Phase 3.2 (SolverConfig), and Phase 3.3 (Factory Integration).

### Added

**Phase 3.1: Unified Problem Class (PR #218)**
- Single `MFGProblem` class replacing 5+ specialized problem classes
- Flexible `MFGComponents` system for custom problem definitions
- Auto-detection of problem types (standard, network, variational, stochastic, highdim)
- `MFGProblemBuilder` for programmatic problem construction
- Full backward compatibility with deprecated specialized classes

**Phase 3.2: Unified Configuration System (PR #222)**
- New `SolverConfig` class unifying 3 competing config systems
- Three usage patterns:
  - YAML files for experiments and reproducibility
  - Builder API for programmatic configuration
  - Presets for common use cases
- Modular config components: `PicardConfig`, `HJBConfig`, `FPConfig`, `BackendConfig`, `LoggingConfig`
- Preset configurations: fast, accurate, research, production, domain-specific
- YAML I/O with validation
- Legacy config compatibility layer

**Phase 3.3: Factory Integration (PR #224)**
- Unified problem factories supporting all MFG types:
  - `create_mfg_problem()` - Main factory for any problem type
  - `create_standard_problem()` - Standard HJB-FP MFG
  - `create_network_problem()` - Network/Graph MFG
  - `create_variational_problem()` - Variational/Lagrangian MFG
  - `create_stochastic_problem()` - Stochastic MFG with common noise
  - `create_highdim_problem()` - High-dimensional MFG (d > 3)
  - `create_lq_problem()` - Linear-Quadratic MFG
  - `create_crowd_problem()` - Crowd dynamics MFG
- Updated `solve_mfg()` interface:
  - New `config` parameter accepting `SolverConfig` instances or preset names
  - Deprecated `method` parameter (still works with warning)
  - Automatic config resolution from strings
- Extended `MFGComponents` for all problem types (network, variational, stochastic, highdim)
- Dual-output factory support: unified MFGProblem (default) or legacy classes (deprecated)
- New examples: `factory_demo.py`, updated `solve_mfg_demo.py`
- Comprehensive documentation:
  - Phase 3.3 design documents (2,000+ lines)
  - Problem type taxonomy
  - Migration guides
  - Completion summary

### Changed

**API Improvements**
- Simplified problem creation with unified factories
- Consistent configuration across all solver types
- Three flexible configuration patterns (YAML, Builder, Presets)
- Clearer separation: problem (math) vs solver (algorithm)

**Code Quality**
- Reduced code duplication through unification
- Better type safety with modern Python typing (`@overload`)
- Improved documentation with comprehensive examples
- Cleaner package structure

### Deprecated

**Problem Classes** (to be removed in v2.0.0)
- `LQMFGProblem` → Use `create_lq_problem()` or `MFGProblem`
- `NetworkMFGProblem` → Use `create_network_problem()` or `MFGProblem`
- `VariationalMFGProblem` → Use `create_variational_problem()` or `MFGProblem`
- `StochasticMFGProblem` → Use `create_stochastic_problem()` or `MFGProblem`

**Config Functions** (to be removed in v2.0.0)
- `create_fast_config()` → Use `presets.fast_solver()`
- `create_accurate_config()` → Use `presets.accurate_solver()`
- `create_research_config()` → Use `presets.research_solver()`
- Old `MFGSolverConfig` → Use new `SolverConfig`

**API Parameters** (to be removed in v2.0.0)
- `solve_mfg(method=...)` → Use `solve_mfg(config=...)`

### Migration Guide

**Old API**:
```python
from mfg_pde.problems import LQMFGProblem
from mfg_pde.config import create_accurate_config
from mfg_pde import solve_mfg

problem = LQMFGProblem(...)
result = solve_mfg(problem, method="accurate")
```

**New API** (Recommended):
```python
from mfg_pde.factory import create_lq_problem
from mfg_pde import solve_mfg

problem = create_lq_problem(...)
result = solve_mfg(problem, config="accurate")
```

### Documentation

- Added comprehensive Phase 3 design documents
- Created migration guides for Phase 3.2 and 3.3
- Updated examples with new unified API
- Added problem type taxonomy
- Created Phase 3 completion summary

### Technical Details

**Total Changes**:
- ~8,000 lines added/modified
- 21 files changed
- 3 major PRs (#218, #222, #224)
- Full backward compatibility maintained

**Key Benefits**:
- Simpler, more consistent API
- Three flexible configuration patterns
- Better documentation and examples
- Easier to maintain and extend
- Better type safety
- Single source of truth

---

## [0.8.1] - 2025-10-08

### Fixed
- Full nD FP Solver implementation
- Semi-Lagrangian 2D solver
- Bug #8 resolution

---

## Historical Versions

Previous versions (< 0.8.1) were tracked in git history but not formally documented in CHANGELOG.

For detailed historical changes, see:
- Git commit history
- Closed issues and PRs
- Development documentation in `docs/development/`

---

**Note**: Starting with v0.9.0, all changes are documented in this CHANGELOG following semantic versioning and Keep a Changelog standards.
