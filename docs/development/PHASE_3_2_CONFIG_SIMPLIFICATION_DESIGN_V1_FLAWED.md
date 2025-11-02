# Phase 3.2: Configuration System Simplification Design

**Date**: 2025-11-03
**Status**: Design Phase
**Timeline**: 2-3 weeks
**Dependencies**: Phase 3.1 (Unified MFGProblem)

---

## Executive Summary

Consolidate 3 competing configuration systems (dataclass, Pydantic, Modern builder) into a single unified YAML-based configuration with schema validation and builder patterns.

**Current State**:
- ✅ 3 different config systems (2,859 lines total)
- ⚠️ Users confused about which to use
- ⚠️ Duplication across systems
- ❌ No YAML file support (only programmatic)
- ❌ Limited usage (14 imports across codebase)

**Goal**: Single configuration system with YAML support, schema validation, and clean programmatic API.

---

## Current State Analysis

### Three Competing Systems

#### 1. Dataclass-based (`solver_config.py` - 414 lines)

**Purpose**: Original configuration system
**Pattern**: Plain Python dataclasses

```python
@dataclass
class DataclassMFGSolverConfig:
    hjb_config: HJBConfig
    fp_config: FPConfig
    picard_config: PicardConfig
```

**Pros**:
- Simple, minimal dependencies
- Type-checked

**Cons**:
- No validation
- No serialization support
- Verbose creation

#### 2. Pydantic-based (`pydantic_config.py` - 475 lines)

**Purpose**: Enhanced validation and serialization
**Pattern**: Pydantic BaseModel

```python
class MFGSolverConfig(BaseModel):
    hjb_config: HJBConfig
    fp_config: FPConfig
    picard_config: PicardConfig

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
```

**Pros**:
- Automatic validation
- JSON/dict serialization
- Good error messages

**Cons**:
- Heavy dependency (Pydantic)
- Still verbose for simple cases

#### 3. Modern Builder (`modern_config.py` - 410 lines)

**Purpose**: Fluent builder API
**Pattern**: Builder with presets

```python
config = (
    SolverConfig()
    .for_problem_type("crowd_dynamics")
    .with_accuracy("high")
    .with_backend("jax")
    .build()
)

# Or use presets
config = crowd_dynamics_config()
```

**Pros**:
- Concise API
- Domain-specific presets
- Discoverable

**Cons**:
- Another system to learn
- Limited adoption

### Additional Systems

#### 4. OmegaConf Manager (`omegaconf_manager.py` - 719 lines)

**Purpose**: Experiment management with YAML
**Optional**: Requires OmegaConf installation

```python
manager = OmegaConfManager("experiments/crowd_dynamics")
config = manager.load_experiment("baseline")
configs = manager.create_parameter_sweep({"sigma": [0.1, 0.2, 0.3]})
```

**Status**: Good design, but optional dependency and separate from main config system

#### 5. Array Validation (`array_validation.py` - 525 lines)

**Purpose**: Specialized array/tensor validation
**Use Case**: Advanced validation for neural network inputs

**Status**: Specialized, orthogonal to main config system

---

## Design Goals

### Primary Goals

1. **Single Entry Point**: One configuration system for all use cases
2. **YAML Support**: File-based configuration for experiments
3. **Schema Validation**: Automatic validation with clear error messages
4. **Backward Compatible**: Existing code continues to work
5. **Preset Library**: Domain-specific presets (crowd, traffic, epidemic, etc.)

### Design Principles

1. **Progressive Complexity**: Simple for simple cases, powerful for complex ones
2. **File-First**: YAML files as primary configuration method
3. **Type Safety**: Full type hints and validation
4. **Migration Path**: Gradual deprecation with clear warnings

---

## Proposed Unified Architecture

### Core Concept

**Single configuration system with 3 usage patterns**:

1. **YAML Files** (recommended) - For experiments and reproducibility
2. **Builder API** (programmatic) - For dynamic configuration
3. **Direct Construction** (advanced) - For fine control

### Pattern 1: YAML Files (Recommended)

```yaml
# experiments/crowd_dynamics/baseline.yaml
problem:
  type: mfg_problem
  dimension: 2
  spatial_bounds: [[0, 10], [0, 10]]
  spatial_discretization: [50, 50]
  time_domain: [1.0, 20]
  sigma: 0.1

solver:
  hjb:
    method: fdm
    accuracy_order: 2
    boundary_conditions: neumann

  fp:
    method: particle
    num_particles: 5000
    kde_bandwidth: auto

  picard:
    max_iterations: 50
    tolerance: 1.0e-6
    anderson_memory: 5

backend:
  type: numpy  # or jax, pytorch
  device: cpu

logging:
  level: INFO
  progress_bar: true
```

**Usage**:
```python
from mfg_pde.config import load_config

config = load_config("experiments/crowd_dynamics/baseline.yaml")
result = solve_mfg(config)
```

### Pattern 2: Builder API (Programmatic)

```python
from mfg_pde.config import ConfigBuilder

config = (
    ConfigBuilder()
    .problem(
        dimension=2,
        spatial_bounds=[[0, 10], [0, 10]],
        spatial_discretization=[50, 50],
        time_domain=(1.0, 20),
        sigma=0.1
    )
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp(method="particle", num_particles=5000)
    .picard(max_iterations=50, tolerance=1e-6)
    .backend("numpy")
    .build()
)
```

**Or use presets**:
```python
from mfg_pde.config import presets

config = presets.crowd_dynamics(
    dimension=2,
    resolution=50,
    accuracy="high"
)
```

### Pattern 3: Direct Construction (Advanced)

```python
from mfg_pde.config import MFGConfig, HJBConfig, FPConfig, PicardConfig

config = MFGConfig(
    problem=ProblemConfig(...),
    hjb=HJBConfig(...),
    fp=FPConfig(...),
    picard=PicardConfig(...),
    backend=BackendConfig(...)
)
```

---

## Implementation Plan

### Phase 3.2.1: Unified Config Classes (Week 1)

**Goal**: Single set of config classes with validation

**New Structure**:
```
mfg_pde/config/
├── __init__.py           # Public API
├── core.py               # Core config classes (MFGConfig, etc.)
├── schemas.py            # YAML schema definitions
├── builder.py            # Builder API
├── presets.py            # Domain-specific presets
├── validation.py         # Validation logic
├── io.py                 # YAML load/save
└── legacy.py             # Backward compatibility
```

**Core Config Classes** (`core.py`):
```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union

class ProblemConfig(BaseModel):
    """Configuration for MFG problem."""
    dimension: int = Field(gt=0, le=10)
    spatial_bounds: list[tuple[float, float]]
    spatial_discretization: list[int] | int
    time_domain: tuple[float, int]
    sigma: float = Field(gt=0)
    coef_CT: float = Field(default=0.5, ge=0)

    @field_validator('spatial_bounds')
    @classmethod
    def validate_bounds(cls, v, info):
        if len(v) != info.data['dimension']:
            raise ValueError(f"spatial_bounds must match dimension")
        return v

class HJBConfig(BaseModel):
    """Configuration for HJB solver."""
    method: Literal["fdm", "gfdm", "semi_lagrangian", "weno"] = "fdm"
    accuracy_order: Literal[1, 2, 3, 4, 5] = 2
    boundary_conditions: Literal["dirichlet", "neumann", "periodic"] = "neumann"
    newton_iterations: int = Field(default=10, ge=1)
    newton_tolerance: float = Field(default=1e-6, gt=0)

class FPConfig(BaseModel):
    """Configuration for FP solver."""
    method: Literal["fdm", "particle", "network"] = "fdm"
    num_particles: int | None = Field(default=None, gt=0)
    kde_bandwidth: float | Literal["auto"] = "auto"

class PicardConfig(BaseModel):
    """Configuration for Picard iteration."""
    max_iterations: int = Field(default=100, ge=1)
    tolerance: float = Field(default=1e-6, gt=0)
    anderson_memory: int = Field(default=0, ge=0)
    verbose: bool = True

class BackendConfig(BaseModel):
    """Configuration for computational backend."""
    type: Literal["numpy", "jax", "pytorch"] = "numpy"
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    precision: Literal["float32", "float64"] = "float64"

class MFGConfig(BaseModel):
    """Unified MFG configuration."""
    problem: ProblemConfig
    hjb: HJBConfig = Field(default_factory=HJBConfig)
    fp: FPConfig = Field(default_factory=FPConfig)
    picard: PicardConfig = Field(default_factory=PicardConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        ...

    @classmethod
    def from_yaml(cls, path: str) -> "MFGConfig":
        """Load configuration from YAML file."""
        ...
```

### Phase 3.2.2: YAML Support (Week 1-2)

**Goal**: File-based configuration with schema validation

**YAML Schema** (`schemas.py`):
```python
import yaml
from pathlib import Path

def load_config(path: str | Path) -> MFGConfig:
    """Load configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return MFGConfig.model_validate(data)

def save_config(config: MFGConfig, path: str | Path) -> None:
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
```

### Phase 3.2.3: Builder API (Week 2)

**Goal**: Fluent API for programmatic configuration

**Builder Implementation** (`builder.py`):
```python
class ConfigBuilder:
    """Fluent builder for MFG configurations."""

    def __init__(self):
        self._config = {}

    def problem(
        self,
        dimension: int,
        spatial_bounds: list[tuple[float, float]],
        **kwargs
    ) -> "ConfigBuilder":
        self._config["problem"] = {
            "dimension": dimension,
            "spatial_bounds": spatial_bounds,
            **kwargs
        }
        return self

    def solver_hjb(self, method: str, **kwargs) -> "ConfigBuilder":
        self._config["hjb"] = {"method": method, **kwargs}
        return self

    def solver_fp(self, method: str, **kwargs) -> "ConfigBuilder":
        self._config["fp"] = {"method": method, **kwargs}
        return self

    def picard(self, **kwargs) -> "ConfigBuilder":
        self._config["picard"] = kwargs
        return self

    def backend(self, type: str, **kwargs) -> "ConfigBuilder":
        self._config["backend"] = {"type": type, **kwargs}
        return self

    def build(self) -> MFGConfig:
        """Build the configuration."""
        return MFGConfig.model_validate(self._config)
```

### Phase 3.2.4: Preset Library (Week 2)

**Goal**: Domain-specific configuration presets

**Presets Implementation** (`presets.py`):
```python
def crowd_dynamics(
    dimension: int = 2,
    resolution: int = 50,
    accuracy: Literal["low", "medium", "high"] = "medium"
) -> MFGConfig:
    """Preset for crowd dynamics problems."""
    accuracy_map = {"low": 1, "medium": 2, "high": 3}

    return ConfigBuilder() \\
        .problem(
            dimension=dimension,
            spatial_bounds=[(0, 10)] * dimension,
            spatial_discretization=[resolution] * dimension,
            time_domain=(1.0, 20),
            sigma=0.1
        ) \\
        .solver_hjb(method="fdm", accuracy_order=accuracy_map[accuracy]) \\
        .solver_fp(method="particle", num_particles=5000) \\
        .picard(max_iterations=50, anderson_memory=5) \\
        .build()

def traffic_flow(**kwargs) -> MFGConfig:
    """Preset for traffic flow problems."""
    ...

def epidemic(**kwargs) -> MFGConfig:
    """Preset for epidemic models."""
    ...

def financial(**kwargs) -> MFGConfig:
    """Preset for financial applications."""
    ...
```

### Phase 3.2.5: Backward Compatibility (Week 2-3)

**Goal**: Deprecate old systems without breaking existing code

**Strategy**:
1. Keep old configs as aliases with deprecation warnings
2. Provide migration utilities
3. Update documentation with examples

**Legacy Support** (`legacy.py`):
```python
from warnings import warn

def create_fast_config(*args, **kwargs):
    """DEPRECATED: Use presets.fast_config() or ConfigBuilder."""
    warn(
        "create_fast_config is deprecated. Use presets.fast_config() instead",
        DeprecationWarning,
        stacklevel=2
    )
    from .presets import fast_config
    return fast_config(*args, **kwargs)

# Similar for other legacy functions
```

### Phase 3.2.6: Integration with solve_mfg() (Week 3)

**Goal**: Wire new config system into high-level interface

**Updated `solve_mfg()` signature**:
```python
def solve_mfg(
    problem: MFGProblem | None = None,
    config: MFGConfig | str | Path | None = None,
    **kwargs
) -> MFGResult:
    """
    Solve Mean Field Game problem.

    Parameters
    ----------
    problem : MFGProblem, optional
        Problem definition. If None, created from config.
    config : MFGConfig | str | Path, optional
        Configuration. Can be:
        - MFGConfig object
        - Path to YAML file
        - None (use kwargs)
    **kwargs
        Additional parameters (deprecated, use config instead)

    Returns
    -------
    MFGResult
        Solution containing U, M, and metadata
    """
    # Load config from file if string/Path
    if isinstance(config, (str, Path)):
        config = load_config(config)

    # Create config from kwargs if needed (deprecated path)
    if config is None and kwargs:
        warn("Passing solver parameters as kwargs is deprecated. Use config object or YAML file.")
        config = ConfigBuilder().from_dict(kwargs).build()

    # Use defaults if nothing provided
    if config is None:
        config = MFGConfig()

    # Create problem from config if not provided
    if problem is None:
        problem = MFGProblem.from_config(config.problem)

    # Create and run solver
    solver = create_solver_from_config(problem, config)
    return solver.solve()
```

---

## Migration Path

### Step 1: Introduce New System (Week 1-2)

- Implement new config classes
- Add YAML support
- Create builder API
- Comprehensive tests

### Step 2: Add Deprecation Warnings (Week 2)

- Wrap old config functions with warnings
- Provide migration utilities
- Update documentation

### Step 3: Update Examples (Week 2-3)

- Convert examples to use new system
- Add YAML-based examples
- Migration guide with examples

### Step 4: Gradual Phase-Out (Post-release)

**Timeline**:
- **v0.9.0** (current): Deprecation warnings added
- **v1.0.0** (+3 months): Warnings become prominent
- **v2.0.0** (+6 months): Old systems removed

---

## Testing Strategy

### Unit Tests

1. **Config Validation**: Test all validators
2. **YAML I/O**: Round-trip testing
3. **Builder API**: Test all methods
4. **Presets**: Validate all presets

### Integration Tests

1. **solve_mfg()**: Test with YAML files
2. **Backward Compatibility**: Ensure old code works
3. **Error Handling**: Test validation errors
4. **Migration**: Test conversion utilities

### Examples

1. **YAML-based**: Examples using config files
2. **Builder API**: Programmatic examples
3. **Presets**: Domain-specific examples

---

## Success Criteria

### Phase 3.2 Complete When:

1. ✅ Single unified config system implemented
2. ✅ YAML file support with schema validation
3. ✅ Builder API for programmatic configuration
4. ✅ Domain-specific preset library (5+ presets)
5. ✅ 100% backward compatibility maintained
6. ✅ All existing tests pass
7. ✅ Migration guide and examples available
8. ✅ Documentation updated

### Metrics

**Before**:
- 3 competing config systems (2,859 lines)
- No YAML support
- Limited usage (14 imports)
- Confusion about which system to use

**After**:
- 1 unified config system (~1,500 lines)
- Full YAML support with validation
- Clear recommended approach (YAML-first)
- Builder API for programmatic use
- Preset library for common scenarios

---

## Risk Assessment

### Low Risk
- ✅ Pydantic validation (well-tested library)
- ✅ YAML support (standard library)

### Medium Risk
- ⚠️ Migration complexity (3 systems → 1)
  * Mitigation: Gradual deprecation, compatibility layer
- ⚠️ User adoption
  * Mitigation: Clear documentation, compelling examples

### High Risk
- ❌ Breaking existing workflows
  * Mitigation: 100% backward compatibility, 6-month deprecation cycle

---

## Timeline & Milestones

### Week 1: Core Implementation
- Unified config classes with Pydantic
- YAML schema and I/O functions
- Basic validation

### Week 2: Builder & Presets
- Builder API implementation
- Preset library (5+ presets)
- Backward compatibility layer

### Week 3: Integration & Testing
- Integrate with solve_mfg()
- Comprehensive test suite
- Migration guide and examples
- Documentation updates

---

## Next Steps

1. **Review & Approve**: Get feedback on this design
2. **Create Issue**: Track Phase 3.2 work
3. **Create Branch**: `feature/phase3-config-simplification`
4. **Start Implementation**: Begin with core config classes

---

**Status**: Design Complete - Ready for Review
**Last Updated**: 2025-11-03
**Author**: Architecture Refactoring Team
