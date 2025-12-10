# Pydantic and OmegaConf Cooperation in MFG_PDE

**Status**: [EVALUATION]
**Created**: 2025-12-10
**Related Issues**: #28 (closed), #403 (in progress)

---

## 1. Executive Summary

MFG_PDE has two configuration systems that serve **different but complementary** purposes:

| System | Purpose | Primary Use Case |
|:-------|:--------|:-----------------|
| **Pydantic** (`core.py`) | Runtime validation, programmatic API | Solver configuration in code |
| **OmegaConf** (`structured_schemas.py`) | YAML file management, experiments | Research experiments, parameter sweeps |

**Key Principle**: These are not competing systems. They cooperate through clear interfaces.

---

## 2. Current Architecture

### 2.1 Pydantic Configuration (`mfg_pde/config/core.py`)

The **canonical** solver configuration system:

```python
from mfg_pde.config import MFGSolverConfig, HJBConfig, FPConfig, PicardConfig

config = MFGSolverConfig(
    hjb=HJBConfig(method="fdm", accuracy_order=2),
    fp=FPConfig(method="particle", num_particles=5000),
    picard=PicardConfig(max_iterations=50, tolerance=1e-6),
)

# Use with solver
result = problem.solve(config=config)
```

**Strengths**:
- Strong runtime validation via Pydantic v2
- IDE autocompletion and type checking
- Model validators for cross-field validation
- Direct integration with solver API
- YAML serialization via `.to_yaml()` / `.from_yaml()`

**Structure**:
```
MFGSolverConfig
├── hjb: HJBConfig
├── fp: FPConfig
├── picard: PicardConfig
├── backend: BackendConfig
└── logging: LoggingConfig
```

### 2.2 OmegaConf Configuration (`mfg_pde/config/structured_schemas.py`)

**Experiment-oriented** configuration for YAML-based workflows:

```python
from mfg_pde.config.omegaconf_manager import load_structured_mfg_config

# Load from YAML with interpolation
config = load_structured_mfg_config("experiment.yaml")
print(config.problem.T)  # Type-safe access
print(config.solver.max_iterations)
```

**Strengths**:
- YAML interpolation: `${experiment.output_dir}/plots`
- Configuration composition: Merge multiple YAML files
- Parameter sweeps: Automated grid search
- Type safety via structured configs (Issue #28 solution)
- Experiment management with metadata

**Structure**:
```
MFGConfig
├── problem: ProblemConfig
│   ├── domain: DomainConfig
│   ├── initial_condition: InitialConditionConfig
│   └── boundary_conditions: BoundaryConditionsConfig
├── solver: SolverConfig
│   ├── hjb: HJBConfig
│   └── fp: FPConfig
└── experiment: ExperimentConfig
    ├── logging: LoggingConfig
    └── visualization: VisualizationConfig
```

---

## 3. Cooperation Modes

### Mode 1: Pydantic-Only (Simple Scripts)

For simple usage where YAML complexity is unnecessary:

```python
from mfg_pde import MFGProblem
from mfg_pde.config import MFGSolverConfig, PicardConfig

# Direct Pydantic configuration
config = MFGSolverConfig(
    picard=PicardConfig(max_iterations=100, tolerance=1e-6)
)

problem = MFGProblem(...)
result = problem.solve(config=config)
```

**When to use**:
- Quick prototyping
- Single-run scripts
- When configuration doesn't need file persistence

### Mode 2: OmegaConf-Only (Pure YAML Workflows)

For experiment-driven research with YAML files:

```yaml
# experiments/crowd_motion.yaml
problem:
  name: crowd_evacuation
  T: 2.0
  Nx: 100
  domain:
    x_min: 0.0
    x_max: 10.0

solver:
  type: fixed_point
  max_iterations: 200
  tolerance: 1e-7

experiment:
  name: crowd_study
  output_dir: results/${experiment.name}
```

```python
from mfg_pde.config.omegaconf_manager import load_structured_mfg_config

config = load_structured_mfg_config("experiments/crowd_motion.yaml")
# Use config.problem, config.solver, etc.
```

**When to use**:
- Reproducible experiments
- Configuration version control
- Parameter sweeps
- Sharing configurations across team

### Mode 3: OmegaConf to Pydantic Bridge (Hybrid)

Load YAML via OmegaConf, convert to Pydantic for solver:

```python
from mfg_pde.config.omegaconf_manager import OmegaConfManager
from mfg_pde.config import MFGSolverConfig

manager = OmegaConfManager()
omega_config = manager.load_config("experiment.yaml")

# Convert to Pydantic for solver
pydantic_config = manager.create_pydantic_config(omega_config)

# Now use with solver
result = problem.solve(config=pydantic_config)
```

**When to use**:
- Need YAML features (interpolation, composition)
- But solver requires Pydantic validation
- Mixed workflow with both file-based and programmatic config

### Mode 4: Parameter Sweeps (Research)

OmegaConf excels at parameter sweeps:

```python
from mfg_pde.config.omegaconf_manager import OmegaConfManager

manager = OmegaConfManager()
base_config = manager.load_config("baseline.yaml")

# Create sweep configurations
sweep_params = {
    "problem.parameters.crowd_aversion": [0.5, 1.0, 1.5, 2.0],
    "solver.tolerance": [1e-4, 1e-6, 1e-8],
}

configs = manager.create_parameter_sweep(base_config, sweep_params)

# Run experiments
for cfg in configs:
    pydantic_cfg = manager.create_pydantic_config(cfg)
    result = problem.solve(config=pydantic_cfg)
    # Save results with cfg.experiment.current_params metadata
```

**When to use**:
- Systematic parameter studies
- Grid search optimization
- Reproducible research experiments

---

## 4. Schema Comparison

| Concept | Pydantic (`core.py`) | OmegaConf (`structured_schemas.py`) |
|:--------|:--------------------|:------------------------------------|
| **Solver iteration** | `PicardConfig` | `SolverConfig.max_iterations` |
| **HJB method** | `HJBConfig.method` | `SolverConfig.hjb.method` |
| **FP method** | `FPConfig.method` | `SolverConfig.fp.method` |
| **Backend** | `BackendConfig.type` | `SolverConfig.backend` |
| **Problem definition** | `MFGProblem` (code) | `ProblemConfig` (YAML) |
| **Experiment metadata** | N/A | `ExperimentConfig` |

**Note**: The schemas are intentionally different because they serve different purposes:
- Pydantic: HOW to solve (algorithmic choices)
- OmegaConf: WHAT to solve + HOW + experiment context

---

## 5. Future Integration Opportunities

### 5.1 Unified Schema (Not Recommended)

One could unify the schemas, but this would:
- Lose OmegaConf's interpolation features
- Add complexity to Pydantic validation
- Blur the separation of concerns

**Recommendation**: Keep separate schemas with clear bridge.

### 5.2 Bidirectional Conversion (Medium Priority)

Currently: OmegaConf -> Pydantic (via `create_pydantic_config`)

Could add: Pydantic -> OmegaConf for saving experiment results:

```python
# Hypothetical future API
def save_experiment_config(pydantic_config, omega_config, results, output_path):
    """Save complete experiment record."""
    omega_config.experiment.final_config = pydantic_to_omega(pydantic_config)
    omega_config.experiment.results = results.to_dict()
    OmegaConf.save(omega_config, output_path)
```

### 5.3 Hydra Integration (Low Priority)

Hydra builds on OmegaConf for CLI argument parsing:

```bash
# Hypothetical future CLI
python run_experiment.py solver.tolerance=1e-8 problem.Nx=200
```

This would require:
- Hydra dependency
- CLI entry points
- Config store setup

**Status**: Low priority until user demand emerges.

### 5.4 Config Builder Pattern (Implemented)

The `MFGSolverConfig` docstring mentions a builder pattern:

```python
# Currently in docstring as example
config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp(method="particle", num_particles=5000)
    .picard(max_iterations=50)
    .build()
)
```

**Status**: Example only, not implemented. Consider if user demand emerges.

---

## 6. Recommendations

### For Library Users

| Use Case | Recommended Approach |
|:---------|:--------------------|
| Quick script | Pydantic directly |
| Reproducible experiment | OmegaConf YAML |
| Parameter sweep | OmegaConf + bridge to Pydantic |
| Production deployment | Pydantic with validated config |

### For Library Developers

1. **Keep schemas separate**: Different purposes, different structures
2. **Maintain bridge**: `OmegaConfManager.create_pydantic_config()` is the key integration point
3. **Add bridges as needed**: Don't pre-optimize; add bidirectional conversion when needed
4. **Document clearly**: Users should understand which system to use when

### Migration Notes

- `MFGSolverConfig` (Pydantic) is the **canonical** config for solvers
- `structured_schemas.SolverConfig` (OmegaConf) is for **type-safe YAML** (Issue #28)
- They have similar names but different purposes - this is intentional
- Do not attempt to merge them without understanding both use cases

---

## 7. Code References

| File | Purpose |
|:-----|:--------|
| `mfg_pde/config/core.py` | Pydantic MFGSolverConfig (canonical) |
| `mfg_pde/config/mfg_methods.py` | HJBConfig, FPConfig, etc. |
| `mfg_pde/config/structured_schemas.py` | OmegaConf dataclass schemas |
| `mfg_pde/config/omegaconf_manager.py` | OmegaConf manager with Pydantic bridge |
| `mfg_pde/config/io.py` | YAML I/O for Pydantic configs |

---

## Appendix: Decision Tree

```
Need to configure MFG solver?
│
├─ Single script, quick test?
│  └─ Use Pydantic directly: MFGSolverConfig(...)
│
├─ Reproducible experiment with YAML?
│  └─ Use OmegaConf: load_structured_mfg_config("config.yaml")
│     └─ Need Pydantic validation?
│        └─ Bridge: manager.create_pydantic_config(omega_config)
│
├─ Parameter sweep?
│  └─ Use OmegaConf: manager.create_parameter_sweep(...)
│     └─ Convert each to Pydantic for solver
│
└─ Production deployment?
   └─ Use Pydantic with validated config from .from_yaml()
```

---

**Document Version**: 1.0
**Related**: `docs/development/DEPRECATION_PLAN_v0.16.md`, Issue #28
