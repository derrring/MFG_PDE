# Phase 3.2 Configuration System Migration Guide

**Target Version**: v0.9.0 (released TBD)
**Deprecation Timeline**: v0.9.0 → v1.0.0 → v2.0.0 (old systems removed)

## Overview

Phase 3.2 consolidates three competing configuration systems into a single unified system with clear, consistent APIs.

### What Changed

**Before (3 competing systems)**:
- Dataclass-based configs (`solver_config.py`)
- Pydantic-based configs (`pydantic_config.py`)
- Modern configs (`modern_config.py`)

**After (1 unified system)**:
- Core system: `SolverConfig` with Pydantic v2 validation
- Three usage patterns: YAML (recommended), Builder API, Presets
- Clear principle: Config for SOLVERS (how), not PROBLEMS (what)

### Key Principle

```python
# PROBLEM (mathematical definition) - Python code
from mfg_pde import MFGProblem

problem = MFGProblem(
    terminal_cost=g,
    hamiltonian=H,
    initial_density=rho_0,
    geometry=domain
)

# CONFIG (algorithmic choices) - YAML or Python
from mfg_pde.config import presets

config = presets.accurate_solver()  # How to solve
```

**Separation of Concerns**:
- `MFGProblem`: WHAT to solve (g, H, ρ₀, geometry)
- `SolverConfig`: HOW to solve (method, tolerance, backend)

---

## Migration Paths

### 1. From Dataclass Configs (`create_*_config`)

**Old Code** (deprecated):
```python
from mfg_pde.config import create_fast_config, create_accurate_config

config = create_fast_config()
# or
config = create_accurate_config()
```

**New Code** (recommended):
```python
from mfg_pde.config import presets

config = presets.fast_solver()
# or
config = presets.accurate_solver()
```

**Migration Table**:

| Old Function | New Preset | Notes |
|:------------|:-----------|:------|
| `create_fast_config()` | `presets.fast_solver()` | Speed-optimized |
| `create_accurate_config()` | `presets.accurate_solver()` | Accuracy-optimized |
| `create_research_config()` | `presets.research_solver()` | Full diagnostics |
| `create_default_config()` | `presets.default_solver()` | Alias for fast |

### 2. From Modern Configs (`*_config()` functions)

**Old Code** (deprecated):
```python
from mfg_pde.config import (
    fast_config,
    crowd_dynamics_config,
    traffic_config,
)

config = crowd_dynamics_config()
```

**New Code** (recommended):
```python
from mfg_pde.config import presets

config = presets.crowd_dynamics_solver(accuracy="medium")
```

**Migration Table**:

| Old Function | New Preset | Notes |
|:------------|:-----------|:------|
| `fast_config()` | `presets.fast_solver()` | - |
| `accurate_config()` | `presets.accurate_solver()` | - |
| `research_config()` | `presets.research_solver()` | - |
| `crowd_dynamics_config()` | `presets.crowd_dynamics_solver()` | Added accuracy param |
| `traffic_config()` | `presets.traffic_flow_solver()` | Added SL option |
| `epidemic_config()` | `presets.epidemic_solver()` | - |
| `financial_config()` | `presets.financial_solver()` | - |
| `large_scale_config()` | `presets.large_scale_solver()` | - |
| `production_config()` | `presets.production_solver()` | - |
| `educational_config()` | `presets.educational_solver()` | - |

### 3. From Pydantic Configs (direct instantiation)

**Old Code** (deprecated):
```python
from mfg_pde.config import (
    MFGSolverConfig,
    HJBConfig,
    FPConfig,
    PicardConfig
)

config = MFGSolverConfig(
    hjb=HJBConfig(method="fdm", accuracy_order=2),
    fp=FPConfig(method="particle", num_particles=5000),
    picard=PicardConfig(max_iterations=50)
)
```

**New Code** (recommended - use presets or builder):
```python
from mfg_pde.config import presets

# Option 1: Use preset
config = presets.fast_solver()

# Option 2: Use builder for custom config
from mfg_pde.config import ConfigBuilder

config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp_particle(num_particles=5000)
    .picard(max_iterations=50)
    .build()
)
```

**New Code** (if direct instantiation needed):
```python
from mfg_pde.config import (
    SolverConfig,  # New unified class
    HJBConfig,
    FPConfig,
    ParticleConfig,
    PicardConfig
)

config = SolverConfig(
    hjb=HJBConfig(method="fdm", accuracy_order=2),
    fp=FPConfig(
        method="particle",
        particle_config=ParticleConfig(num_particles=5000)
    ),
    picard=PicardConfig(max_iterations=50)
)
```

**Key Changes**:
- `MFGSolverConfig` → `SolverConfig`
- Method-specific configs now nested (e.g., `particle_config`, `gfdm_config`)
- Cleaner structure with explicit method configurations

---

## New Features

### Feature 1: YAML Configuration (Recommended)

**Create YAML file** (`experiments/baseline.yaml`):
```yaml
hjb:
  method: fdm
  accuracy_order: 2
  boundary_conditions: neumann

fp:
  method: particle
  particle_config:
    num_particles: 5000
    kde_bandwidth: auto
    normalization: initial_only
    mode: hybrid

picard:
  max_iterations: 50
  tolerance: 1.0e-6
  anderson_memory: 5
  verbose: true

backend:
  type: numpy
  device: cpu
  precision: float64

logging:
  level: INFO
  progress_bar: true
  save_intermediate: false
```

**Load and use**:
```python
from mfg_pde.config import load_solver_config

config = load_solver_config("experiments/baseline.yaml")
result = solve_mfg(problem, config=config)
```

**Benefits**:
- Reproducibility: Version control your experiments
- Clarity: Separate configuration from code
- Sharing: Easy to share experimental setups
- Iteration: Try different configs without code changes

### Feature 2: Builder API (Programmatic)

**Fluent, discoverable API**:
```python
from mfg_pde.config import ConfigBuilder

config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp_particle(
        num_particles=5000,
        kde_bandwidth="auto",
        normalization="initial_only"
    )
    .picard(max_iterations=50, tolerance=1e-6, anderson_memory=5)
    .backend(backend_type="numpy", device="cpu", precision="float64")
    .logging(level="INFO", progress_bar=True)
    .build()
)
```

**Method-specific builders**:
```python
# GFDM for HJB
config = (
    ConfigBuilder()
    .solver_hjb_gfdm(
        delta=0.1,
        stencil_size=25,
        qp_optimization_level="auto"
    )
    .solver_fp(method="fdm")
    .build()
)

# Semi-Lagrangian for HJB
config = (
    ConfigBuilder()
    .solver_hjb_semi_lagrangian(
        interpolation_method="cubic",
        rk_order=2
    )
    .solver_fp(method="fdm")
    .build()
)

# WENO for HJB
config = (
    ConfigBuilder()
    .solver_hjb_weno(
        weno_order=5,
        flux_splitting="lax_friedrichs"
    )
    .solver_fp(method="fdm")
    .build()
)
```

### Feature 3: Preset Library

**11 ready-to-use configurations**:

**General-Purpose**:
```python
from mfg_pde.config import presets

# Speed-optimized (prototyping)
config = presets.fast_solver()

# Accuracy-optimized (publication)
config = presets.accurate_solver()

# Research/debugging
config = presets.research_solver()

# Production deployments
config = presets.production_solver()
```

**Domain-Specific**:
```python
# Crowd dynamics
config = presets.crowd_dynamics_solver(accuracy="high")

# Traffic flow
config = presets.traffic_flow_solver(use_semi_lagrangian=True)

# Epidemic modeling
config = presets.epidemic_solver()

# Financial applications
config = presets.financial_solver()
```

**Computational**:
```python
# High-dimensional problems (d > 3)
config = presets.high_dimensional_solver(use_gpu=True)

# Large-scale problems
config = presets.large_scale_solver()

# Educational/teaching
config = presets.educational_solver()
```

---

## Advanced Migration Scenarios

### Scenario 1: Custom Configuration with GFDM

**Old Code**:
```python
from mfg_pde.config import MFGSolverConfig, HJBConfig, GFDMConfig

config = MFGSolverConfig(
    hjb=HJBConfig(
        method="gfdm",
        gfdm_delta=0.15,
        stencil_size=25
    )
)
```

**New Code**:
```python
from mfg_pde.config import ConfigBuilder

config = (
    ConfigBuilder()
    .solver_hjb_gfdm(
        delta=0.15,
        stencil_size=25,
        qp_optimization_level="auto"
    )
    .solver_fp(method="fdm")
    .build()
)
```

**Or with direct instantiation**:
```python
from mfg_pde.config import SolverConfig, HJBConfig, GFDMConfig

config = SolverConfig(
    hjb=HJBConfig(
        method="gfdm",
        gfdm_config=GFDMConfig(
            delta=0.15,
            stencil_size=25,
            qp_optimization_level="auto"
        )
    )
)
```

### Scenario 2: Particle Methods

**Particle solver** (sample particles, output to grid via KDE):
```python
config = (
    ConfigBuilder()
    .solver_fp_particle(
        num_particles=5000,
        normalization="initial_only"
    )
    .build()
)
```

**For meshfree density evolution on collocation points**, use `FPGFDMSolver`:
```python
from mfg_pde.alg.numerical.fp_solvers import FPGFDMSolver
import numpy as np

# Create collocation points
particles = np.random.uniform(0, 1, (1000, dim))

# Use FPGFDMSolver for GFDM-based density evolution
solver = FPGFDMSolver(problem, collocation_points=particles)
```

### Scenario 3: Backend Configuration

**Old Code** (scattered):
```python
config = create_fast_config()
# Backend settings were implicit or scattered
```

**New Code** (explicit):
```python
from mfg_pde.config import ConfigBuilder

# NumPy backend (CPU only)
config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp(method="fdm")
    .backend(backend_type="numpy", device="cpu", precision="float64")
    .build()
)

# JAX backend (GPU-capable)
config = (
    ConfigBuilder()
    .solver_hjb_gfdm(delta=0.1, stencil_size=25)
    .solver_fp_particle(num_particles=10000)
    .backend(backend_type="jax", device="gpu", precision="float32")
    .build()
)
```

**Validation**: NumPy + GPU raises clear error:
```python
from mfg_pde.config import BackendConfig

try:
    BackendConfig(type="numpy", device="gpu")
except ValueError as e:
    print(e)  # "NumPy backend does not support GPU device. Use JAX or PyTorch."
```

---

## Deprecation Timeline

### Phase 1: v0.9.0 (Current)
- **Status**: Old APIs work with `DeprecationWarning`
- **Action**: Migrate to new system at your convenience
- **Testing**: All old code continues to work

**Example Warning**:
```python
from mfg_pde.config import create_fast_config

config = create_fast_config()
# DeprecationWarning: create_fast_config() is deprecated and will be removed
# in v2.0.0. Use 'from mfg_pde.config import presets; config = presets.fast_solver()'
```

### Phase 2: v1.0.0 (+3 months)
- **Status**: Warnings become prominent
- **Action**: Should migrate by this point
- **Testing**: Old code still works but warnings are loud

### Phase 3: v2.0.0 (+6 months)
- **Status**: Old APIs removed
- **Action**: Must use new system
- **Breaking**: Old imports will fail

---

## Migration Checklist

### Step 1: Assess Your Codebase
```bash
# Find old config usage
grep -r "create_fast_config\|create_accurate_config" .
grep -r "fast_config\|accurate_config\|crowd_dynamics_config" .
grep -r "MFGSolverConfig" .
```

### Step 2: Choose Migration Strategy

**For Simple Cases** (recommended):
- Replace with `presets.{preset_name}_solver()`
- Quickest migration path

**For Custom Configs**:
- Option A: Create YAML file (best for experiments)
- Option B: Use ConfigBuilder (best for programmatic)
- Option C: Direct instantiation (most control)

### Step 3: Update Imports
```python
# Old
from mfg_pde.config import create_fast_config, MFGSolverConfig

# New
from mfg_pde.config import presets, SolverConfig, ConfigBuilder
```

### Step 4: Update Configuration Creation
```python
# Old
config = create_fast_config()

# New
config = presets.fast_solver()
```

### Step 5: Test
```python
# Verify configuration works
from mfg_pde import solve_mfg

result = solve_mfg(problem, config=config)
print(result.success)
```

### Step 6: Update YAML (if using)
```bash
# Save configuration for reproducibility
python -c "
from mfg_pde.config import presets
config = presets.accurate_solver()
config.to_yaml('experiments/baseline.yaml')
"
```

---

## Common Questions

### Q: Can I still use the old APIs?

**A**: Yes, until v2.0.0 (6+ months). You'll see deprecation warnings guiding you to the new API.

### Q: Which approach should I use?

**A**:
- **Experiments/Research**: YAML files (reproducibility)
- **Programmatic/Dynamic**: Builder API (flexibility)
- **Quick Start**: Presets (convenience)

### Q: How do I customize a preset?

**A**: Three options:

```python
from mfg_pde.config import presets

# Option 1: Modify after creation
config = presets.fast_solver()
config.picard.max_iterations = 100

# Option 2: Save to YAML, edit, reload
config.to_yaml("my_config.yaml")
# Edit YAML file...
from mfg_pde.config import load_solver_config
config = load_solver_config("my_config.yaml")

# Option 3: Use builder for full control
from mfg_pde.config import ConfigBuilder
config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)  # Custom settings
    .solver_fp_particle(num_particles=5000)
    .picard(max_iterations=100)  # Custom iterations
    .build()
)
```

### Q: What about backward compatibility?

**A**: Full backward compatibility maintained until v2.0.0:
- `MFGSolverConfig` still exported (alias to Pydantic version)
- All old config functions still work (with warnings)
- Gradual migration path over 6+ months

### Q: Do I need to change my problem definitions?

**A**: No! `MFGProblem` is unchanged. Phase 3.2 only affects solver configuration.

```python
# Problem definition unchanged
from mfg_pde import MFGProblem

problem = MFGProblem(
    terminal_cost=g,
    hamiltonian=H,
    initial_density=rho_0,
    geometry=domain
)

# Only configuration syntax changed
from mfg_pde.config import presets
config = presets.accurate_solver()  # New

result = solve_mfg(problem, config=config)  # Same
```

---

## Examples

### Example 1: Quick Migration (Dataclass → Preset)

**Before**:
```python
from mfg_pde.config import create_accurate_config
from mfg_pde import solve_mfg

config = create_accurate_config()
result = solve_mfg(problem, config=config)
```

**After**:
```python
from mfg_pde.config import presets
from mfg_pde import solve_mfg

config = presets.accurate_solver()
result = solve_mfg(problem, config=config)
```

### Example 2: Experiment Workflow (Modern → YAML)

**Before**:
```python
from mfg_pde.config import crowd_dynamics_config
from mfg_pde import solve_mfg

config = crowd_dynamics_config()
result = solve_mfg(problem, config=config)
```

**After** (two files):

**`experiment.yaml`**:
```yaml
hjb:
  method: fdm
  accuracy_order: 2

fp:
  method: particle
  particle_config:
    num_particles: 5000
    normalization: initial_only

picard:
  max_iterations: 50
  tolerance: 1.0e-6
  anderson_memory: 5

backend:
  type: numpy
  device: cpu
```

**`experiment.py`**:
```python
from mfg_pde.config import load_solver_config
from mfg_pde import solve_mfg

config = load_solver_config("experiment.yaml")
result = solve_mfg(problem, config=config)
```

### Example 3: High-Dimensional Problem (Pydantic → Builder)

**Before**:
```python
from mfg_pde.config import MFGSolverConfig, HJBConfig, FPConfig

config = MFGSolverConfig(
    hjb=HJBConfig(method="gfdm", gfdm_delta=0.1),
    fp=FPConfig(method="particle", num_particles=10000)
)
```

**After**:
```python
from mfg_pde.config import ConfigBuilder

config = (
    ConfigBuilder()
    .solver_hjb_gfdm(delta=0.1, stencil_size=25)
    .solver_fp_particle(num_particles=10000)
    .backend(backend_type="jax", device="gpu")
    .build()
)
```

**Or use preset**:
```python
from mfg_pde.config import presets

config = presets.high_dimensional_solver(use_gpu=True)
```

---

## Support

**Documentation**:
- API Reference: `docs/api/config.md`
- Design Document: `docs/development/PHASE_3_2_CONFIG_SIMPLIFICATION_DESIGN.md`

**Questions**:
- GitHub Issues: Tag with `phase-3.2` and `configuration`
- Discussions: GitHub Discussions for general questions

**Reporting Bugs**:
- Check if issue exists in new system vs old system
- Provide minimal reproducible example
- Include config YAML if applicable

---

**Last Updated**: 2025-11-03
**Phase**: 3.2 (Configuration Simplification)
**Status**: Complete - Ready for review
