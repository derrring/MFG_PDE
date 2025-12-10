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

## 8. Known Trade-offs and Risks

### 8.1 The Maintenance Tax (DRY Violation)

**Trade-off**: Dual schemas violate DRY (Don't Repeat Yourself) principle.

**Risk**: If adding a new parameter (e.g., `viscosity_coefficient`), developer must update:
1. Pydantic model in `core.py`
2. OmegaConf dataclass in `structured_schemas.py`
3. Bridge mapping in `create_pydantic_config()`

If schemas diverge (e.g., different default values), silent bugs can occur.

**Mitigation**:
- Strict code review for config changes
- CI tests to verify schema consistency (planned)
- Clear documentation of which fields exist in which schema

### 8.2 Cognitive Load

**Trade-off**: Multiple entry points increase learning curve.

**Risk**: New users may be confused about which system to use:
- "Should I edit `config.py` or `experiment.yaml`?"
- "Why are there two `SolverConfig` classes?"

**Mitigation**:
- Decision tree (Appendix) guides users
- Clear naming: Pydantic for "solver config", OmegaConf for "experiment config"
- Consistent documentation patterns

### 8.3 Runtime Conversion Overhead

**Trade-off**: Mode 3/4 require config conversion at runtime.

**Risk**: Negligible for PDE solvers (ms vs minutes), but problematic for high-frequency calls.

**Mitigation**:
- Convert once at experiment start, not in inner loops
- Cache converted configs when running sweeps

---

## 9. Improvement Roadmap

### Phase 1: Current State
- Manual dual schema maintenance
- Manual bridging via `create_pydantic_config()`
- Requires developer discipline

### Phase 2: Recommended Improvements (Medium Priority)

#### 9.2.1 Generic Bridge Function

Replace hardcoded mapping with generic adapter:

```python
from typing import TypeVar, Type
from pydantic import BaseModel
from omegaconf import DictConfig

T = TypeVar("T", bound=BaseModel)

def bridge(omega_cfg: DictConfig, pydantic_cls: Type[T]) -> T:
    """Generic OmegaConf to Pydantic conversion with field mapping."""
    # Auto-map matching fields
    # Log warnings for unmatched fields
    # Handle nested configs recursively
    ...
```

**Value**: Open-Closed Principle compliance - new configs don't require bridge modifications.

#### 9.2.2 Effective Config Snapshot (Critical for Reproducibility)

**Problem**: Input YAML contains interpolations (`${...}`). Saving only input YAML makes reproduction difficult.

**Solution**: Mandatory snapshot before solver execution:

```python
def save_effective_config(pydantic_config, output_path):
    """Save resolved config with all defaults filled."""
    # No interpolations, no missing fields
    # Complete snapshot of actual execution parameters
    config_dict = pydantic_config.model_dump(mode="json")
    with open(output_path / "resolved_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
```

**Rationale**:
- Original YAML: Records user intent
- Resolved JSON: Records actual execution parameters (including defaults)

### Phase 3: Ideal State (Low Priority)

#### 9.3.1 SSOT Automation (Single Source of Truth)

Automated code generation to eliminate manual synchronization:

**Option A (Code First)**: Generate OmegaConf dataclasses from Pydantic models
```bash
# CI pipeline step
python scripts/generate_omegaconf_schemas.py --from pydantic --to structured_schemas.py
```

**Option B (Schema First)**: Define JSON Schema, generate both Pydantic and OmegaConf
```bash
# From canonical schema.json
python scripts/generate_configs.py --schema config_schema.json
```

#### 9.3.2 Schema Consistency Tests

```python
# tests/test_config_sync.py
def test_schema_field_consistency():
    """Ensure Pydantic and OmegaConf schemas have matching fields."""
    pydantic_fields = get_all_fields(MFGSolverConfig)
    omega_fields = get_all_fields(SolverConfig)

    # Check overlapping fields have compatible types and defaults
    for field in pydantic_fields & omega_fields:
        assert compatible_types(pydantic_fields[field], omega_fields[field])
```

---

## 10. Naming Strategy for Config Classes

### 10.1 Current Problem: Naming Conflicts

The dual-system architecture creates **equivocal naming** where identical class names exist in both systems:

| Class Name | Pydantic Location | OmegaConf Location | Conflict? |
|:-----------|:-----------------|:-------------------|:----------|
| `NewtonConfig` | `mfg_methods.py:27` | `structured_schemas.py:79` | **Yes** |
| `HJBConfig` | `mfg_methods.py:251` | `structured_schemas.py:88` | **Yes** |
| `FPConfig` | `mfg_methods.py:302` | `structured_schemas.py:98` | **Yes** |
| `LoggingConfig` | `core.py:29` | `structured_schemas.py:119` | **Yes** |
| `SolverConfig` | `core.py` (alias) | `structured_schemas.py:106` | **Yes** |
| `MFGSolverConfig` | `core.py:121` | - | No |
| `ProblemConfig` | - | `structured_schemas.py:64` | No (OmegaConf only) |

### 10.2 Why "Import-as Aliasing" is an Anti-Pattern

The initial approach of using `import as` aliases has serious problems:

#### A. The "Aliasing Tax"

This strategy shifts conflict resolution responsibility to users:

```python
# User must remember to alias EVERY TIME
from mfg_pde.config.structured_schemas import HJBConfig as YamlHJBConfig
```

**Problems**:
- Inconsistent aliases across codebase (Dev A: `YamlHJB`, Dev B: `HJB_Omega`)
- Easy to forget, leading to silent bugs
- Not enforceable by linters

#### B. IDE Auto-Import Nightmare

When typing `HJBConfig` and pressing Tab:
- IDE shows **two identical options**
- Wrong choice leads to confusing `ValidationError` at runtime
- No visual distinction during development

#### C. Semantic Mismatch

If two classes have **different semantics**, they should have **different names**.

### 10.3 Recommended Strategy: Schema Suffix

**Principle**: Pydantic classes are runtime **Configs**; OmegaConf classes are structure **Schemas**.

| Concept | Pydantic (keep as-is) | OmegaConf (rename) | Semantic |
|:--------|:---------------------|:-------------------|:---------|
| **Solver** | `MFGSolverConfig` | `SolverSchema` | Config = validated entity; Schema = YAML blueprint |
| **HJB method** | `HJBConfig` | `HJBSchema` | Same distinction |
| **FP method** | `FPConfig` | `FPSchema` | Same distinction |
| **Logging** | `LoggingConfig` | `LoggingSchema` | Same distinction |
| **Newton** | `NewtonConfig` | `NewtonSchema` | Same distinction |

#### Benefits

1. **Self-documenting**: Names explain their purpose without comments
2. **IDE-friendly**: Auto-import works correctly
3. **No aliasing**: Users never need `import as`
4. **Semantic clarity**: `*Schema` = external structure definition; `*Config` = internal validated entity

#### Clean Code Comparison

**Before (Confusing - requires aliasing)**:
```python
from mfg_pde.config import HJBConfig
from mfg_pde.config.structured_schemas import HJBConfig as YamlHJBConfig  # Must remember!

def bridge(omega_cfg: YamlHJBConfig) -> HJBConfig:
    ...
```

**After (Clear - self-documenting)**:
```python
from mfg_pde.config import HJBConfig
from mfg_pde.config.structured_schemas import HJBSchema  # No alias needed

def bridge(omega_cfg: HJBSchema) -> HJBConfig:
    ...
```

### 10.4 Migration Plan

Since renaming is a **breaking change**, implement with deprecation:

**Phase 1 (v0.16)**: Add `*Schema` aliases alongside old names
```python
# structured_schemas.py
@dataclass
class HJBSchema:  # New canonical name
    ...

HJBConfig = HJBSchema  # Deprecated alias, emit warning on use
```

**Phase 2 (v0.17)**: Emit `DeprecationWarning` for old names

**Phase 3 (v0.18)**: Remove old names

### 10.5 Complete Naming Convention

| Role | Naming Pattern | Example | Location |
|:-----|:---------------|:--------|:---------|
| **Runtime config (Pydantic)** | `*Config` | `HJBConfig`, `MFGSolverConfig` | `core.py`, `mfg_methods.py` |
| **YAML schema (OmegaConf)** | `*Schema` | `HJBSchema`, `SolverSchema` | `structured_schemas.py` |
| **Problem definition** | `*Config` | `ProblemConfig` | `structured_schemas.py` (OmegaConf-only, no conflict) |
| **Experiment metadata** | `*Config` | `ExperimentConfig` | `structured_schemas.py` (OmegaConf-only, no conflict) |

### 10.6 Import Guidelines (After Migration)

**For Solver Configuration** (programmatic API):
```python
# Canonical imports - use these for solver configuration
from mfg_pde.config import MFGSolverConfig, HJBConfig, FPConfig, PicardConfig
```

**For YAML Experiments** (OmegaConf):
```python
# Clean imports - no aliasing needed
from mfg_pde.config.structured_schemas import (
    SolverSchema,
    HJBSchema,
    FPSchema,
    ProblemConfig,  # OmegaConf-only, no conflict
)
```

**For Bridge Operations**:
```python
from mfg_pde.config import MFGSolverConfig  # Target (Pydantic)
from mfg_pde.config.omegaconf_manager import OmegaConfManager  # Bridge
from mfg_pde.config.structured_schemas import SolverSchema  # Source (OmegaConf)
```

### 10.7 Semantic Distinction Summary

| Suffix | System | Meaning | Validation | Use Case |
|:-------|:-------|:--------|:-----------|:---------|
| `*Config` | Pydantic | Runtime-validated entity | Full (types, ranges, cross-field) | Solver execution |
| `*Schema` | OmegaConf | YAML structure definition | Structural only | File loading, parameter sweeps |

**Key Insight**: OmegaConf `*Schema` classes are **simpler** (YAML-friendly DTOs), while Pydantic `*Config` classes are **richer** (full validation). The bridge function maps Schema to Config, adding validation in the process.

---

## 11. Summary

**Architecture Assessment**: This dual-system approach represents **best practice** for scientific computing configuration:

| Criterion | Assessment |
|:----------|:-----------|
| **Type Safety** | Excellent - double validation (YAML parse + runtime) |
| **Maintainability** | Medium-High - dual schema overhead mitigated by clear boundaries |
| **Extensibility** | Excellent - easy Hydra/MLflow integration path |
| **User Experience** | Good - layered entry points for different user types |

**Key Insight**: The "static vs dynamic" separation (Pydantic for strict runtime, OmegaConf for flexible YAML) avoids the "lowest common denominator" trap that plagues monolithic config systems.

**Evolution Path**:
1. **Now**: Manual maintenance with discipline
2. **Next**: Generic bridge + effective config snapshots
3. **Future**: Automated SSOT code generation

---

**Document Version**: 1.3
**Last Updated**: 2025-12-10
**Related**: `docs/development/DEPRECATION_PLAN_v0.16.md`, Issue #28
