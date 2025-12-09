# Configuration Patterns in MFG_PDE

**Tutorial Level**: Beginner-Intermediate
**Estimated Time**: 20 minutes
**Prerequisites**: [Getting Started Tutorial](01_getting_started.md)

---

## Three Ways to Configure Solvers

MFG_PDE offers three configuration patterns, each suited to different use cases:

1. **Presets** - Quick start with sensible defaults
2. **Builder API** - Programmatic configuration
3. **YAML Files** - Reproducible experiments

All three patterns create the same `SolverConfig` object and have **identical performance** (~7 µs overhead).

---

## Pattern 1: Presets (Recommended for Most Cases)

### Available Presets

```python
from mfg_pde.config import presets

# Fast: Speed-optimized (prototyping, testing)
config = presets.fast_solver()
# - Low accuracy order
# - No Anderson acceleration
# - Minimal logging

# Balanced: Good default (production)
config = presets.balanced_solver()
# - Medium accuracy
# - Some acceleration
# - Normal logging

# Accurate: High-precision (publications)
config = presets.accurate_solver()
# - High accuracy order
# - Anderson acceleration
# - Detailed logging
```

### Inline Usage

```python
from mfg_pde import solve_mfg, MFGProblem

# Simplest: Use preset string
result = solve_mfg(MFGProblem(), preset="fast")

# Or: Use preset object
from mfg_pde.config import presets
result = solve_mfg(MFGProblem(), config=presets.accurate_solver())
```

**When to use**: Quick prototyping, most production code, when defaults are good enough.

---

## Pattern 2: Builder API (Programmatic Control)

### Basic Builder Usage

```python
from mfg_pde.config import ConfigBuilder

config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp(method="fdm")
    .picard(max_iterations=50, tolerance=1e-6)
    .backend(backend_type="numpy")
    .build()
)
```

### Incremental Building

```python
builder = ConfigBuilder()

# HJB solver
builder.solver_hjb(
    method="fdm",           # Finite difference
    accuracy_order=3,       # Higher accuracy
)

# FP solver
builder.solver_fp(
    method="particle",
    num_particles=5000,
)

# Picard iteration
builder.picard(
    max_iterations=100,
    tolerance=1e-8,
    anderson_memory=5,      # Anderson acceleration
)

# Backend
builder.backend(
    backend_type="pytorch",
    device="cuda",
)

# Build config
config = builder.build()
```

### Domain-Specific Presets

```python
from mfg_pde.config import presets

# Crowd dynamics (high spatial resolution)
config = presets.crowd_dynamics_solver()

# Traffic flow (conservative schemes)
config = presets.traffic_flow_solver()

# Financial applications (high accuracy)
config = presets.financial_solver()
```

**When to use**: Dynamic configuration, parameter sweeps, when you need specific solver combinations.

---

## Pattern 3: YAML Files (Reproducibility)

### Basic YAML Configuration

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

backend:
  type: numpy
```

### Loading YAML Configs

```python
from mfg_pde.config import load_solver_config

config = load_solver_config("experiments/baseline.yaml")
result = solve_mfg(problem, config=config)
```

### Advanced YAML Features

```yaml
# experiments/high_accuracy.yaml
hjb:
  method: fdm
  accuracy_order: 3
  fdm_config:
    scheme: central
    time_stepping: implicit

fp:
  method: particle
  particle_config:
    num_particles: 10000
    resampling_threshold: 0.5

picard:
  max_iterations: 200
  tolerance: 1e-10
  anderson_memory: 5
  damping_factor: 0.8

backend:
  type: pytorch
  device: cuda
  precision: float64

logging:
  level: INFO
  progress_bar: true
  save_intermediate: true
  output_dir: "results/high_accuracy"
```

### Saving Configs

```python
from mfg_pde.config import save_solver_config, presets

config = presets.accurate_solver()
save_solver_config(config, "experiments/saved_config.yaml")
```

**When to use**: Research experiments, reproducibility, version control of configurations.

---

## Comparing the Three Patterns

| Feature | Presets | Builder | YAML |
|:--------|:--------|:--------|:-----|
| **Speed** | ✅ Fastest to write | ✅ Fast | ⚠️ Requires file |
| **Flexibility** | ⚠️ Limited | ✅ Full control | ✅ Full control |
| **Reproducibility** | ✅ Named presets | ⚠️ Code-dependent | ✅ Version controlled |
| **Type Safety** | ✅ Full | ✅ Full | ⚠️ Runtime validation |
| **Documentation** | ✅ Self-documenting | ⚠️ Needs comments | ✅ YAML comments |

---

## Common Configuration Patterns

### Parameter Sweep

```python
from mfg_pde.config import ConfigBuilder
import numpy as np

results = []
for tol in [1e-4, 1e-6, 1e-8]:
    config = ConfigBuilder().picard(tolerance=tol).build()
    result = solve_mfg(problem, config=config)
    results.append((tol, result))
```

### Experiment Management

```python
experiments = {
    "fast": "experiments/fast.yaml",
    "balanced": "experiments/balanced.yaml",
    "accurate": "experiments/accurate.yaml",
}

for name, config_file in experiments.items():
    config = load_solver_config(config_file)
    result = solve_mfg(problem, config=config)
    # Save results with experiment name
    save_results(result, f"results/{name}.pkl")
```

### Conditional Configuration

```python
from mfg_pde.config import ConfigBuilder

builder = ConfigBuilder()

if use_gpu:
    builder.backend(backend_type="pytorch", device="cuda")
else:
    builder.backend(backend_type="numpy")

if high_accuracy:
    builder.picard(tolerance=1e-10, max_iterations=200)
else:
    builder.picard(tolerance=1e-6, max_iterations=50)

config = builder.build()
```

---

## Next Steps

- **Tutorial 3**: Custom Problems and Hamiltonians
- **API Reference**: Complete configuration options
- **Examples**: See `examples/` for real-world usage

---

**Tutorial Version**: 1.0
**Last Updated**: 2025-11-03
