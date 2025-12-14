# Pydantic and OmegaConf Cooperation in MFG_PDE

**Status**: [APPROVED / ARCHITECTURE_V2]
**Created**: 2025-12-10
**Last Updated**: 2025-12-10
**Related Issues**: #28, #403

---

## 1. Executive Summary

MFG_PDE employs a **Dual-System Configuration Architecture** to balance runtime safety with experimental flexibility.

| System | Role | Primary Responsibility |
|:-------|:-----|:-----------------------|
| **Pydantic** (`core.py`) | **The Gatekeeper** | Runtime validation, API safety, type strictness. |
| **OmegaConf** (`structured_schemas.py`) | **The Orchestrator** | YAML management, CLI overrides, parameter sweeps. |

**Core Philosophy**: Pydantic validates *HOW* the solver runs (math correctness); OmegaConf manages *WHAT* experiment is running (user intent)[^1].

---

## 2. Current Architecture

### 2.1 Pydantic Configuration (The Kernel)

*File: `mfg_pde/config/core.py`*

The **canonical** source of truth for solver instantiation. It enforces mathematical constraints (e.g., `tolerance > 0`).

```python
from mfg_pde.config import MFGSolverConfig, HJBConfig, FPConfig

# Strict validation happens here
config = MFGSolverConfig(
    hjb=HJBConfig(method="fdm", accuracy_order=2),
    fp=FPConfig(method="particle", num_particles=5000),
)
# result = problem.solve(config=config)
```

### 2.2 OmegaConf Configuration (The Interface)

*File: `mfg_pde/config/structured_schemas.py`*

The **experiment-oriented** layer. It handles YAML parsing, interpolation, and merging.

```python
from mfg_pde.config.omegaconf_manager import load_structured_mfg_config

# Supports interpolation: ${experiment.output_dir}, merging, and CLI overrides
config = load_structured_mfg_config("experiment.yaml")
```

---

## 3. Cooperation Modes

### Mode 1: Pydantic-Only (Dev / Scripts)

*Direct instantiation.* Best for unit tests and simple scripts where file persistence is not required.

### Mode 2: OmegaConf-Only (Pure YAML)

*Standard usage.* Users interact with YAML files. The system validates structure via OmegaConf Schemas.

### Mode 3: The Bridge (Hybrid / Production)

*The standard execution flow.* Load flexible YAML, convert to strict Pydantic for execution.

```python
# 1. Load Flexible Config (User Intent)
omega_cfg = manager.load_config("experiment.yaml")

# 2. Bridge: Convert to Strict Config (Execution Contract)
pydantic_cfg = manager.bridge_to_pydantic(omega_cfg, MFGSolverConfig)

# 3. Solve
problem.solve(config=pydantic_cfg)
```

### Mode 4: Parameter Sweeps (Research)

*High-throughput experiments.* OmegaConf generates variations; Pydantic validates each one.

**Requirement**: Every experiment run **MUST** save the `resolved_config.json` (Effective Config Snapshot).

```python
from pathlib import Path

# ... (sweep generation logic) ...

for i, omega_sub_cfg in enumerate(configs):
    # 1. Bridge to Pydantic (Validates math)
    pydantic_cfg = manager.bridge_to_pydantic(omega_sub_cfg, MFGSolverConfig)

    # 2. CRITICAL: Save Effective Snapshot
    # This records the EXACT parameters used (defaults filled, interpolation resolved)
    output_dir = Path(omega_sub_cfg.experiment.output_dir) / f"run_{i:04d}"
    manager.save_effective_config(pydantic_cfg, output_dir)

    # 3. Execute
    problem.solve(config=pydantic_cfg)
```

---

## 4. Schema Strategy

| Feature | Pydantic (`*Config`) | OmegaConf (`*Schema`) |
|:--------|:--------------------|:----------------------|
| **Structure** | Nested by logic | Hierarchical by domain |
| **Validation** | Strict (`AfterValidator`) | Structural (Types only) |
| **Interpolation** | No | Yes (`${...}`) |
| **Source** | Code / JSON | YAML / CLI |

**Note**: The schemas are **intentionally decoupled**.

- **Pydantic** schema is the "Kernel API".
- **OmegaConf** schema is the "User Interface".

---

## 5. Critical Implementation Details

### 5.1 The Generic Bridge (Adapter Pattern)

Instead of hardcoded mapping, we employ a **Generic Adapter** to reduce maintenance overhead. This allows the bridge to support new configs automatically.

```python
from typing import TypeVar, Type
from pydantic import BaseModel
from omegaconf import DictConfig, OmegaConf

T = TypeVar("T", bound=BaseModel)

def bridge_to_pydantic(omega_cfg: DictConfig, pydantic_cls: Type[T]) -> T:
    """
    Generic adapter that converts OmegaConf dicts to Pydantic models.
    """
    # 1. Resolve all interpolations (${...})
    OmegaConf.resolve(omega_cfg)

    # 2. Convert to container (primitive dict)
    container = OmegaConf.to_container(omega_cfg, resolve=True)

    # 3. Validation & Instantiation (Pydantic takes over)
    # Using model_validate for Pydantic V2
    return pydantic_cls.model_validate(container)
```

### 5.2 Serialization & Reproducibility (The Snapshot Rule)

A scientific workflow requires two records:

1. **Intent**: The input YAML (with variables).
2. **Execution**: The resolved JSON (actual values used).

**Rule**: Every experiment run **MUST** save the **Effective Config Snapshot**.

```python
import json
from pathlib import Path
from pydantic import BaseModel

def save_effective_config(pydantic_config: BaseModel, output_dir: Path) -> None:
    """Save resolved config with all defaults filled."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = pydantic_config.model_dump(mode="json")
    with open(output_dir / "resolved_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
```

---

## 6. Trade-offs and Mitigations

### 6.1 The "Double Maintenance" Tax

**Trade-off**: Adding a parameter requires updating both `core.py` and `structured_schemas.py`.

**Mitigation**: **Automated Consistency Tests**.

- We implement a pytest that inspects both classes via reflection.
- If `MFGSolverConfig` has field `alpha` but `SolverSchema` (OmegaConf) misses it, the test **fails**.

```python
# tests/test_config_sync.py
def test_schema_field_consistency():
    """Ensure Pydantic and OmegaConf schemas have matching fields."""
    pydantic_fields = get_all_fields(MFGSolverConfig)
    omega_fields = get_all_fields(SolverSchema)

    # Check overlapping fields have compatible types
    for field in pydantic_fields & omega_fields:
        assert compatible_types(pydantic_fields[field], omega_fields[field])
```

### 6.2 Runtime Overhead

**Trade-off**: Conversion takes time.

**Mitigation**: Conversion happens **once** at solver initialization (outer loop). It is strictly forbidden inside the time-stepping loop (inner loop).

---

## 7. Improvement Roadmap

### Phase 1: Stabilization (Completed)

- [x] Establish Pydantic V2 as the kernel Validator.
- [x] Establish OmegaConf as the Experiment Manager.

### Phase 2: Automation & Robustness (Current Focus)

- [ ] **Refactor**: Replace manual `create_pydantic_config` with **Generic Bridge** (Section 5.1).
- [ ] **Feature**: Implement mandatory **Effective Config Snapshot** (Section 5.2).
- [ ] **Renaming**: Execute the Global Rename Strategy (Section 8).

### Phase 3: Code Generation (Future)

- [ ] **Codegen**: Explore generating OmegaConf dataclasses *automatically* from Pydantic models using `datamodel-code-generator` to remove the maintenance tax.

---

## 8. Naming Strategy for Config Classes (STRICT VERSION)

### 8.1 The Core Problem: Ambiguity

We face a naming dilemma where concepts exist in two forms:

- **Runtime Logic**: Validated objects used by the solver (Pydantic).
- **Data Structure**: Static blueprints used for YAML loading (OmegaConf).

Using identical names or "import aliases" led to IDE confusion and cognitive load[^2].

### 8.2 The Strategy: Strict Semantic Suffixes

We enforce a **Universal Naming Convention** based on the layer where the class is defined. This rule applies **globally**.

- **Pydantic Layer (`core.py`)** → Suffix: **`*Config`**
    - Represents: *Active, validated entities.*
- **OmegaConf Layer (`structured_schemas.py`)** → Suffix: **`*Schema`**
    - Represents: *Passive, structural blueprints (DTOs).*

### 8.3 Naming Table (Strict)

| Concept | Pydantic Entity (Runtime) | OmegaConf Blueprint (YAML) | Status |
|:--------|:--------------------------|:---------------------------|:-------|
| **Solver Root** | `MFGSolverConfig` | `SolverSchema` | **Rename** |
| **HJB Method** | `HJBConfig` | `HJBSchema` | **Rename** |
| **FP Method** | `FPConfig` | `FPSchema` | **Rename** |
| **Newton** | `NewtonConfig` | `NewtonSchema` | **Rename** |
| **Logging** | `LoggingConfig` | `LoggingSchema` | **Rename** |
| **Experiment** | *(N/A)* | `ExperimentSchema` | **Rename** |
| **Problem** | *(N/A)* | `ProblemSchema` | **Rename** |
| **Domain** | *(N/A)* | `DomainSchema` | **Rename** |
| **Boundary Conditions** | *(N/A)* | `BoundaryConditionsSchema` | **Rename** |
| **Initial Condition** | *(N/A)* | `InitialConditionSchema` | **Rename** |
| **Visualization** | *(N/A)* | `VisualizationSchema` | **Rename** |
| **MFG Root** | *(N/A)* | `MFGSchema` | **Rename** |
| **Beach Problem** | *(N/A)* | `BeachProblemSchema` | **Rename** |

### 8.4 Migration & Deprecation Plan

To handle the renaming smoothly, we implement a module-level `__getattr__` hook in `structured_schemas.py` to warn users who are still importing the old names.

```python
# mfg_pde/config/structured_schemas.py
import warnings
from dataclasses import dataclass

# 1. Define New Canonical Names
@dataclass
class HJBSchema:
    method: str = "fdm"
    # ...

# 2. Implement Backward Compatibility Hook
def __getattr__(name: str):
    """
    Allows imports of old names (e.g., HJBConfig) but warns the user.
    """
    DEPRECATED_MAP = {
        "HJBConfig": "HJBSchema",
        "FPConfig": "FPSchema",
        "SolverConfig": "SolverSchema",
        "NewtonConfig": "NewtonSchema",
        "LoggingConfig": "LoggingSchema",
        "ProblemConfig": "ProblemSchema",
        "ExperimentConfig": "ExperimentSchema",
        "DomainConfig": "DomainSchema",
        "BoundaryConditionsConfig": "BoundaryConditionsSchema",
        "InitialConditionConfig": "InitialConditionSchema",
        "VisualizationConfig": "VisualizationSchema",
        "MFGConfig": "MFGSchema",
        "BeachProblemConfig": "BeachProblemSchema",
    }

    if name in DEPRECATED_MAP:
        new_name = DEPRECATED_MAP[name]
        warnings.warn(
            f"'{name}' is deprecated and will be removed in v0.18. "
            f"Please import '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return globals()[new_name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

**Migration Timeline**:
- **v0.16**: Rename classes, add `__getattr__` hook for backwards compatibility
- **v0.17**: Emit INFO-level warnings in addition to DeprecationWarning
- **v0.18**: Remove deprecated aliases

---

## 9. Conclusion

This architecture adopts a **"Loose Coupling, Tight Validation"** strategy.

- **Loose Coupling** allows researchers to organize YAMLs however they like.
- **Tight Validation** ensures the Math Kernel never receives invalid parameters.

By implementing the **Generic Bridge** and **Strict Naming Strategy** defined in this revision, `MFG_PDE` achieves production-grade robustness while maintaining research-grade flexibility.

---

## References

[^1]: Martin Fowler. *Patterns of Enterprise Application Architecture*. Addison-Wesley Professional, 2002. (Separation of Concerns).

[^2]: Robert C. Martin. *Clean Code: A Handbook of Agile Software Craftsmanship*. Prentice Hall, 2008. (Chapter 2: Meaningful Names).

---

**Document Version**: 2.0
**Related**: `docs/development/DEPRECATION_PLAN_v0.16.md`, Issue #28, Issue #403
