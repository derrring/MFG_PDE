# Phase 3.2: Configuration System Simplification Design (V2 - Corrected)

**Date**: 2025-11-03
**Version**: 2.0 (Corrected after design review)
**Status**: Design Phase
**Timeline**: 2-3 weeks
**Dependencies**: Phase 3.1 (Unified MFGProblem)

---

## Design Review Summary

**V1 Issues Found**:
1. ❌ `ProblemConfig` duplicated `MFGProblem` functionality (Phase 3.1 already unified problems)
2. ❌ Mixing problem definition (mathematical) with solver configuration (algorithmic)
3. ❌ `LoggingConfig` referenced but never defined
4. ❌ Pydantic validators used incorrect v1 API
5. ❌ Missing solver-specific configs (GFDM, Semi-Lagrangian, WENO, Particle modes)
6. ❌ Presets tried to create problems (impossible - problems need custom methods)

**V2 Fixes**:
- ✅ Rename `MFGConfig` → `SolverConfig` (clearer purpose)
- ✅ Remove `ProblemConfig` (problems are `MFGProblem` instances from Phase 3.1)
- ✅ YAML for solver settings only (problem definition stays in Python)
- ✅ Add comprehensive solver-specific configs
- ✅ Presets for solver configs, not problem definitions
- ✅ Proper Pydantic v2 validation

---

## Executive Summary

Consolidate 3 competing configuration systems into **single unified YAML-based SOLVER configuration**. Problems remain as `MFGProblem` instances (Phase 3.1), configurations specify **how to solve** them.

**Key Principle**: Clean separation of concerns
- **MFGProblem** (Python code): Mathematical problem definition (g, H, ρ₀, custom geometry)
- **SolverConfig** (YAML/Python): Algorithmic choices (methods, tolerances, backend)

---

## Correct Architecture

### Core Concept

**Configuration is for SOLVERS, not PROBLEMS**:
- Problems require custom Python code (terminal costs, Hamiltonians, geometries)
- Configurations specify algorithmic parameters (which method, tolerance, backend)
- YAML files configure solver behavior for a given problem

### Pattern 1: YAML Files (Recommended)

```yaml
# experiments/crowd_dynamics/baseline.yaml
solver:
  hjb:
    method: fdm  # or gfdm, semi_lagrangian, weno
    accuracy_order: 2
    boundary_conditions: neumann
    newton:
      max_iterations: 10
      tolerance: 1.0e-6

  fp:
    method: particle  # or fdm, network
    num_particles: 5000
    kde_bandwidth: auto
    normalization: initial_only

  picard:
    max_iterations: 50
    tolerance: 1.0e-6
    anderson_memory: 5
    verbose: true

backend:
  type: numpy  # or jax, pytorch
  device: cpu
  precision: float64

logging:
  level: INFO
  progress_bar: true
  save_intermediate: false
```

**Usage**:
```python
from mfg_pde import MFGProblem
from mfg_pde.config import load_solver_config
from mfg_pde import solve_mfg

# 1. Define problem in Python (requires custom methods)
class CrowdDynamicsProblem(MFGProblem):
    def g(self, x):
        return 0.5 * np.sum((x - 10.0)**2)

    def rho0(self, x):
        return np.exp(-10 * np.sum((x - 2.0)**2))

problem = CrowdDynamicsProblem(
    dimension=2,
    spatial_bounds=[[0, 10], [0, 10]],
    spatial_discretization=[50, 50],
    time_domain=(1.0, 20),
    sigma=0.1
)

# 2. Load solver config from YAML
config = load_solver_config("experiments/crowd_dynamics/baseline.yaml")

# 3. Solve
result = solve_mfg(problem, config=config)
```

### Pattern 2: Builder API (Programmatic)

```python
from mfg_pde.config import ConfigBuilder

config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp(method="particle", num_particles=5000)
    .picard(max_iterations=50, tolerance=1e-6, anderson_memory=5)
    .backend("numpy", device="cpu")
    .logging(level="INFO", progress_bar=True)
    .build()
)

result = solve_mfg(problem, config=config)
```

### Pattern 3: Presets (Solver Configs)

```python
from mfg_pde.config import presets

# Presets are SOLVER configs (not problem definitions!)
config = presets.accurate_solver()  # High accuracy settings
config = presets.fast_solver()      # Speed-optimized settings
config = presets.research_solver()  # Research defaults

# Domain-specific solver recommendations
config = presets.crowd_dynamics_solver(accuracy="high")
config = presets.traffic_flow_solver()
config = presets.epidemic_solver()

result = solve_mfg(problem, config=config)
```

---

## Implementation Plan

### Phase 3.2.1: Core Solver Config Classes (Week 1)

**File Structure**:
```
mfg_pde/config/
├── __init__.py           # Public API
├── core.py               # SolverConfig, PicardConfig, BackendConfig, LoggingConfig
├── hjb_configs.py        # HJBConfig + FDMHJBConfig, GFDMConfig, SLConfig, WENOConfig
├── fp_configs.py         # FPConfig + FDMFPConfig, ParticleConfig, NetworkConfig
├── builder.py            # ConfigBuilder for fluent API
├── presets.py            # Solver config presets
├── io.py                 # YAML load/save
└── legacy.py             # Backward compatibility
```

**Core Classes** (`core.py`):
```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal

class LoggingConfig(BaseModel):
    """Configuration for logging and progress reporting."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    progress_bar: bool = True
    save_intermediate: bool = False
    output_dir: str | None = None

class BackendConfig(BaseModel):
    """Configuration for computational backend."""
    type: Literal["numpy", "jax", "pytorch"] = "numpy"
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    precision: Literal["float32", "float64"] = "float64"

class PicardConfig(BaseModel):
    """Configuration for Picard iteration."""
    max_iterations: int = Field(default=100, ge=1)
    tolerance: float = Field(default=1e-6, gt=0)
    anderson_memory: int = Field(default=0, ge=0)
    verbose: bool = True

    @model_validator(mode='after')
    def validate_anderson(self) -> 'PicardConfig':
        if self.anderson_memory < 0:
            raise ValueError("anderson_memory must be non-negative")
        return self

class SolverConfig(BaseModel):
    """Unified solver configuration."""
    hjb: 'HJBConfig' = Field(default_factory=lambda: HJBConfig())
    fp: 'FPConfig' = Field(default_factory=lambda: FPConfig())
    picard: PicardConfig = Field(default_factory=PicardConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        from .io import save_solver_config
        save_solver_config(self, path)

    @classmethod
    def from_yaml(cls, path: str) -> "SolverConfig":
        """Load configuration from YAML file."""
        from .io import load_solver_config
        return load_solver_config(path)
```

**HJB Configs** (`hjb_configs.py`):
```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any

class NewtonConfig(BaseModel):
    """Newton iteration configuration."""
    max_iterations: int = Field(default=10, ge=1)
    tolerance: float = Field(default=1e-6, gt=0)
    relaxation: float = Field(default=1.0, gt=0, le=1.0)

class HJBConfig(BaseModel):
    """Base HJB solver configuration."""
    method: Literal["fdm", "gfdm", "semi_lagrangian", "weno"] = "fdm"
    accuracy_order: int = Field(default=2, ge=1, le=5)
    boundary_conditions: Literal["dirichlet", "neumann", "periodic"] = "neumann"
    newton: NewtonConfig = Field(default_factory=NewtonConfig)

    # Method-specific configs (populated based on method)
    fdm_config: 'FDMHJBConfig | None' = None
    gfdm_config: 'GFDMConfig | None' = None
    sl_config: 'SLConfig | None' = None
    weno_config: 'WENOConfig | None' = None

    @model_validator(mode='after')
    def validate_method_config(self) -> 'HJBConfig':
        """Ensure method-specific config is provided if method selected."""
        # Auto-populate default config if not provided
        if self.method == "gfdm" and self.gfdm_config is None:
            self.gfdm_config = GFDMConfig()
        elif self.method == "semi_lagrangian" and self.sl_config is None:
            self.sl_config = SLConfig()
        elif self.method == "weno" and self.weno_config is None:
            self.weno_config = WENOConfig()
        elif self.method == "fdm" and self.fdm_config is None:
            self.fdm_config = FDMHJBConfig()
        return self

class FDMHJBConfig(BaseModel):
    """Finite Difference Method specific configuration."""
    scheme: Literal["central", "upwind", "lax_friedrichs"] = "central"
    time_stepping: Literal["explicit", "implicit", "crank_nicolson"] = "implicit"

class GFDMConfig(BaseModel):
    """Generalized Finite Difference Method (meshfree/particle) configuration."""
    delta: float = Field(default=0.1, gt=0)
    stencil_size: int = Field(default=20, ge=5)
    qp_optimization_level: Literal["none", "auto", "always"] = "auto"
    monotonicity_check: bool = True
    adaptive_qp: bool = False

class SLConfig(BaseModel):
    """Semi-Lagrangian method configuration."""
    interpolation_method: Literal["linear", "cubic", "rbf"] = "cubic"
    rk_order: Literal[1, 2, 3, 4] = 2
    cfl_number: float = Field(default=0.5, gt=0, le=1.0)

class WENOConfig(BaseModel):
    """WENO scheme configuration."""
    weno_order: Literal[3, 5, 7] = 5
    flux_splitting: Literal["lax_friedrichs", "roe", "local_lax"] = "lax_friedrichs"
    epsilon: float = Field(default=1e-6, gt=0)
```

**FP Configs** (`fp_configs.py`):
```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal
import numpy as np

class FPConfig(BaseModel):
    """Base FP solver configuration."""
    method: Literal["fdm", "particle", "network"] = "fdm"

    # Method-specific configs
    fdm_config: 'FDMFPConfig | None' = None
    particle_config: 'ParticleConfig | None' = None
    network_config: 'NetworkConfig | None' = None

    @model_validator(mode='after')
    def validate_method_config(self) -> 'FPConfig':
        """Auto-populate method-specific config."""
        if self.method == "fdm" and self.fdm_config is None:
            self.fdm_config = FDMFPConfig()
        elif self.method == "particle" and self.particle_config is None:
            self.particle_config = ParticleConfig()
        elif self.method == "network" and self.network_config is None:
            self.network_config = NetworkConfig()
        return self

class FDMFPConfig(BaseModel):
    """FDM-based FP solver configuration."""
    scheme: Literal["upwind", "central", "lax_friedrichs"] = "upwind"
    time_stepping: Literal["explicit", "implicit"] = "implicit"

class ParticleConfig(BaseModel):
    """Particle-based FP solver configuration."""
    num_particles: int = Field(default=5000, gt=0)
    kde_bandwidth: float | Literal["auto"] = "auto"
    normalization: Literal["none", "initial_only", "all"] = "initial_only"
    mode: Literal["hybrid", "collocation"] = "hybrid"
    external_particles: np.ndarray | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

class NetworkConfig(BaseModel):
    """Network/graph-based FP solver configuration."""
    discretization_method: Literal["finite_volume", "finite_element"] = "finite_volume"
```

### Phase 3.2.2: YAML I/O (Week 1)

**YAML Support** (`io.py`):
```python
import yaml
from pathlib import Path
from .core import SolverConfig

def load_solver_config(path: str | Path) -> SolverConfig:
    """
    Load solver configuration from YAML file.

    Parameters
    ----------
    path : str | Path
        Path to YAML configuration file

    Returns
    -------
    SolverConfig
        Validated solver configuration

    Raises
    ------
    ValidationError
        If configuration is invalid
    FileNotFoundError
        If file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Validate and construct
    return SolverConfig.model_validate(data)

def save_solver_config(config: SolverConfig, path: str | Path) -> None:
    """
    Save solver configuration to YAML file.

    Parameters
    ----------
    config : SolverConfig
        Configuration to save
    path : str | Path
        Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(
            config.model_dump(exclude_none=True, mode='json'),
            f,
            default_flow_style=False,
            sort_keys=False
        )
```

### Phase 3.2.3: Builder API (Week 2)

**Builder Implementation** (`builder.py`):
```python
from .core import SolverConfig, PicardConfig, BackendConfig, LoggingConfig
from .hjb_configs import HJBConfig, GFDMConfig, SLConfig, WENOConfig
from .fp_configs import FPConfig, ParticleConfig

class ConfigBuilder:
    """Fluent builder for solver configurations."""

    def __init__(self):
        self._hjb = {}
        self._fp = {}
        self._picard = {}
        self._backend = {}
        self._logging = {}

    def solver_hjb(
        self,
        method: str = "fdm",
        accuracy_order: int = 2,
        **kwargs
    ) -> "ConfigBuilder":
        """Configure HJB solver."""
        self._hjb = {
            "method": method,
            "accuracy_order": accuracy_order,
            **kwargs
        }
        return self

    def solver_fp(
        self,
        method: str = "fdm",
        **kwargs
    ) -> "ConfigBuilder":
        """Configure FP solver."""
        self._fp = {
            "method": method,
            **kwargs
        }
        return self

    def picard(self, **kwargs) -> "ConfigBuilder":
        """Configure Picard iteration."""
        self._picard = kwargs
        return self

    def backend(self, type: str = "numpy", **kwargs) -> "ConfigBuilder":
        """Configure backend."""
        self._backend = {"type": type, **kwargs}
        return self

    def logging(self, **kwargs) -> "ConfigBuilder":
        """Configure logging."""
        self._logging = kwargs
        return self

    def build(self) -> SolverConfig:
        """Build the configuration."""
        config_dict = {}

        if self._hjb:
            config_dict["hjb"] = self._hjb
        if self._fp:
            config_dict["fp"] = self._fp
        if self._picard:
            config_dict["picard"] = self._picard
        if self._backend:
            config_dict["backend"] = self._backend
        if self._logging:
            config_dict["logging"] = self._logging

        return SolverConfig.model_validate(config_dict)
```

### Phase 3.2.4: Preset Library (Week 2)

**Presets for SOLVER Configs** (`presets.py`):
```python
from .builder import ConfigBuilder
from .core import SolverConfig

def fast_solver() -> SolverConfig:
    """Fast solver configuration (speed-optimized)."""
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=1)
        .solver_fp(method="fdm")
        .picard(max_iterations=30, tolerance=1e-4, anderson_memory=0)
        .backend("numpy")
        .logging(level="WARNING", progress_bar=False)
        .build()
    )

def accurate_solver() -> SolverConfig:
    """Accurate solver configuration (accuracy-optimized)."""
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=3)
        .solver_fp(method="fdm")
        .picard(max_iterations=100, tolerance=1e-8, anderson_memory=5)
        .backend("numpy", precision="float64")
        .logging(level="INFO", progress_bar=True)
        .build()
    )

def research_solver() -> SolverConfig:
    """Research solver configuration (comprehensive output)."""
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=2)
        .solver_fp(method="fdm")
        .picard(max_iterations=100, tolerance=1e-6, anderson_memory=5, verbose=True)
        .backend("numpy")
        .logging(level="DEBUG", progress_bar=True, save_intermediate=True)
        .build()
    )

# Domain-specific SOLVER recommendations (not problem definitions!)
def crowd_dynamics_solver(accuracy: str = "medium") -> SolverConfig:
    """Recommended solver settings for crowd dynamics problems."""
    accuracy_map = {"low": 1, "medium": 2, "high": 3}

    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=accuracy_map[accuracy])
        .solver_fp(method="particle", num_particles=5000, normalization="initial_only")
        .picard(max_iterations=50, anderson_memory=5)
        .backend("numpy")
        .build()
    )

def traffic_flow_solver() -> SolverConfig:
    """Recommended solver settings for traffic flow problems."""
    return (
        ConfigBuilder()
        .solver_hjb(method="semi_lagrangian", accuracy_order=2)
        .solver_fp(method="fdm")
        .picard(max_iterations=40, tolerance=1e-6)
        .backend("numpy")
        .build()
    )

def epidemic_solver() -> SolverConfig:
    """Recommended solver settings for epidemic models."""
    return (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=2)
        .solver_fp(method="fdm")
        .picard(max_iterations=60, tolerance=1e-7)
        .backend("numpy")
        .build()
    )

def high_dimensional_solver() -> SolverConfig:
    """Recommended solver settings for high-dimensional problems (d > 3)."""
    return (
        ConfigBuilder()
        .solver_hjb(method="gfdm")  # Meshfree for high-d
        .solver_fp(method="particle", num_particles=10000, mode="collocation")
        .picard(max_iterations=50, anderson_memory=5)
        .backend("jax", device="gpu")  # GPU acceleration
        .build()
    )
```

### Phase 3.2.5: Backward Compatibility (Week 2-3)

**Legacy Support** (`legacy.py`):
```python
from warnings import warn
from .presets import fast_solver, accurate_solver, research_solver

def create_fast_config(*args, **kwargs):
    """
    DEPRECATED: Use presets.fast_solver() instead.

    Will be removed in v2.0.0.
    """
    warn(
        "create_fast_config is deprecated and will be removed in v2.0.0. "
        "Use 'from mfg_pde.config import presets; config = presets.fast_solver()' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return fast_solver()

def create_accurate_config(*args, **kwargs):
    """DEPRECATED: Use presets.accurate_solver() instead."""
    warn(
        "create_accurate_config is deprecated. Use presets.accurate_solver() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return accurate_solver()

def create_research_config(*args, **kwargs):
    """DEPRECATED: Use presets.research_solver() instead."""
    warn(
        "create_research_config is deprecated. Use presets.research_solver() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return research_solver()

# Map old names to new
__all__ = [
    "create_fast_config",
    "create_accurate_config",
    "create_research_config",
]
```

### Phase 3.2.6: Integration with solve_mfg() (Week 3)

**Updated `solve_mfg()` signature**:
```python
from pathlib import Path
from .config import SolverConfig, load_solver_config

def solve_mfg(
    problem: MFGProblem,
    config: SolverConfig | str | Path | None = None,
    **kwargs
) -> MFGResult:
    """
    Solve Mean Field Game problem.

    Parameters
    ----------
    problem : MFGProblem
        Problem definition (from Phase 3.1 unified interface)
    config : SolverConfig | str | Path, optional
        Solver configuration. Can be:
        - SolverConfig object
        - Path to YAML config file
        - None (use default or kwargs)
    **kwargs
        Deprecated: Solver parameters as kwargs (use config instead)

    Returns
    -------
    MFGResult
        Solution containing U, M, convergence info, and metadata

    Examples
    --------
    >>> # With YAML config
    >>> problem = MyCrowdProblem(...)
    >>> result = solve_mfg(problem, config="configs/baseline.yaml")

    >>> # With preset
    >>> from mfg_pde.config import presets
    >>> result = solve_mfg(problem, config=presets.accurate_solver())

    >>> # With builder
    >>> from mfg_pde.config import ConfigBuilder
    >>> config = ConfigBuilder().solver_hjb(method="fdm").build()
    >>> result = solve_mfg(problem, config=config)
    """
    # Load config from file if string/Path
    if isinstance(config, (str, Path)):
        config = load_solver_config(config)

    # Create config from kwargs if provided (deprecated path)
    if config is None and kwargs:
        from warnings import warn
        warn(
            "Passing solver parameters as kwargs is deprecated and will be "
            "removed in v2.0.0. Use config parameter with SolverConfig object "
            "or YAML file path instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .config import ConfigBuilder
        config = ConfigBuilder().from_dict(kwargs).build()

    # Use defaults if nothing provided
    if config is None:
        from .config import presets
        config = presets.fast_solver()

    # Create and run solver
    solver = create_solver_from_config(problem, config)
    return solver.solve()
```

---

## Success Criteria

### Phase 3.2 Complete When:

1. ✅ Single unified `SolverConfig` system implemented
2. ✅ YAML file support with schema validation
3. ✅ Builder API for programmatic configuration
4. ✅ Preset library (5+ presets: fast, accurate, research, + domain-specific)
5. ✅ Comprehensive solver-specific configs (GFDM, SL, WENO, Particle modes)
6. ✅ 100% backward compatibility maintained
7. ✅ All existing tests pass
8. ✅ Migration guide and examples
9. ✅ Clear separation: MFGProblem (mathematical) vs SolverConfig (algorithmic)

### Metrics

**Before**:
- 3 competing config systems (2,859 lines)
- No YAML support
- Limited usage (14 imports)
- Confusion about config vs problem definition

**After**:
- 1 unified solver config system (~1,200 lines)
- Full YAML support with validation
- Clear separation: Problems in Python, configs in YAML
- Comprehensive solver-specific configurations
- 5+ preset solver configs

---

## Timeline & Milestones

### Week 1: Core Implementation
- Core solver config classes (SolverConfig, PicardConfig, BackendConfig, LoggingConfig)
- HJB solver configs (FDM, GFDM, SL, WENO)
- FP solver configs (FDM, Particle, Network)
- YAML I/O with validation

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

**Status**: Design Complete (V2 - Corrected)
**Last Updated**: 2025-11-03
**Design Review**: Passed (6 issues fixed from V1)
