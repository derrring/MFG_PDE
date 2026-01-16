# Factory Pattern Infrastructure Audit

**Document**: Infrastructure Survey for Factory Pattern Design
**Date**: 2026-01-16
**Purpose**: Ground factory design in MFG_PDE's actual codebase
**Related**: `FACTORY_PATTERN_DESIGN.md`, Issue #580

---

## Executive Summary

**Survey Scope**: Comprehensive analysis of MFG_PDE infrastructure to validate factory pattern design assumptions.

**Key Finding**: Infrastructure is **80% ready** for factory pattern implementation:
- ✅ **Ready**: Config system (Pydantic), solver hierarchy, geometry protocol, BC applicators
- ⚠️ **Partial**: Method enums (scattered), backend selection (particle-only)
- ❌ **Missing**: Solver registry, dynamic dispatch, geometry capabilities, fallback routing

**Critical Gap**: `problem.solve()` (line 1954-2026) is **hardcoded** to GFDM+Particle, contradicting design doc's assumption of intelligent selection. This is precisely why Issue #580 exists.

**Actionable Outcome**: Design doc needs revision to acknowledge current state and provide migration path from hardcoded to factory-based selection.

---

## Table of Contents

1. [Solver Class Hierarchy](#1-solver-class-hierarchy)
2. [Current Factory Implementations](#2-current-factory-implementations)
3. [problem.solve() Current State](#3-problemsolve-current-state)
4. [Config System Structure](#4-config-system-structure)
5. [Geometry Classes & Properties](#5-geometry-classes--properties)
6. [Existing Enums & Types](#6-existing-enums--types)
7. [Design Assumptions vs Reality](#7-design-assumptions-vs-reality)
8. [Recommendations for Design Doc](#8-recommendations-for-design-doc)

---

## 1. Solver Class Hierarchy

### Current Architecture

```
BaseMFGSolver (abstract)
├── validate_solution()
├── get_boundary_conditions()
└── BC resolution hierarchy (5-level fallback)

BaseNumericalSolver(BaseMFGSolver)
├── discretization_type: str

BaseHJBSolver(BaseNumericalSolver)
├── solve_hjb_system()
├── discretize()
├── Implementations:
│   ├── HJBFDMSolver
│   ├── HJBGFDMSolver
│   ├── HJBSemiLagrangianSolver
│   └── HJBWENOSolver

BaseFPSolver(BaseNumericalSolver)
├── solve_fp_system()
├── validate_solution()
├── Implementations:
│   ├── FPFDMSolver
│   ├── FPParticleSolver
│   ├── FPGFDMSolver
│   └── FPSemiLagrangianSolver
```

### Validator Pattern (Issue #543)

**Current practice**: Uses `try/except AttributeError` instead of `hasattr()`:

```python
# In BaseFPSolver._validate_problem_compatibility()
try:
    _ = problem.some_required_attribute
except AttributeError:
    raise ValueError("Problem missing required attribute")
```

**Implication for design doc**: Trait checking should follow this pattern, not `hasattr()`.

### Missing: Scheme Family Traits

**Design doc assumes**:
```python
class HJBFDMSolver(BaseHJBSolver):
    _scheme_family = SchemeFamily.FDM  # ← NOT IMPLEMENTED
```

**Reality**: No `_scheme_family` attribute exists on any solver.

**Action**: Add to implementation Step 1 (Infrastructure).

---

## 2. Current Factory Implementations

### Problem Factories (`factory/problem_factories.py`)

**Implemented** (Concern 1 - WHAT):
- `create_lq_problem()` ✅
- `create_crowd_problem()` ✅
- `create_network_problem()` ✅
- `create_variational_problem()` ✅
- `create_stochastic_problem()` ✅
- `create_highdim_problem()` ✅
- Plus generic `create_mfg_problem()`

**Status**: Fully functional, matches design doc Concern 1.

---

### Solver Factory (`factory/solver_factory.py`)

**Current signature**:
```python
def create_solver(
    problem: MFGProblem,
    solver_type: Literal["fixed_point"],  # ← ONLY option!
    hjb_solver: BaseHJBSolver | None,     # ← Required
    fp_solver: BaseFPSolver | None,       # ← Required
    config: MFGSolverConfig | None,
    **kwargs
) -> FixedPointIterator
```

**Critical limitation**: Requires pre-instantiated solvers, does NOT create them.

**Removed methods**:
- `create_fast_solver()` → NotImplementedError
- `create_accurate_solver()` → NotImplementedError
- `create_research_solver()` → NotImplementedError
- Particle collocation variants → Removed from core

**Config kwargs handling**:
```python
# Maps kwargs to MFGSolverConfig paths
"max_picard_iterations" → config.picard.max_iterations
"picard_tolerance"      → config.picard.tolerance
"max_newton_iterations" → config.hjb.newton.max_iterations
"newton_tolerance"      → config.hjb.newton.tolerance
"num_particles"         → config.fp.particle.num_particles
"delta"                 → config.hjb.gfdm.delta
```

**Gap vs design doc**: Design assumes `create_paired_solvers()` creates solvers from scheme enum. Current factory only assembles pre-created solvers.

---

### Backend Factory (`factory/backend_factory.py`)

**Intelligent selection** (partial):
```python
def create_optimal_backend(problem, prefer_gpu=False, precision="float64"):
    total_size = ∏(grid_shape) × Nt

    if total_size > 100_000:
        return JAXBackend()
    else:
        return NumPyBackend()
```

**Size categories**:
- Small: < 10,000 points
- Medium: 10,000-100,000 points
- Large: ≥ 100,000 points

**Status**: Intelligent backend selection EXISTS, can serve as template for solver selection.

---

## 3. problem.solve() Current State

**Location**: `mfg_pde/core/mfg_problem.py:1954-2026`

**Current implementation** (CRITICAL):

```python
def solve(
    self,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
    config: Any | None = None
) -> Any:
    """Solve this MFG problem."""

    # Default config creation
    if config is None:
        config = MFGSolverConfig()

    # Parameter updates
    config.picard.max_iterations = max_iterations
    config.picard.tolerance = tolerance

    # === HARDCODED SOLVER SELECTION ===
    hjb_solver = HJBGFDMSolver(
        self,
        collocation_points  # ← Assumes collocation mode
    )
    fp_solver = FPParticleSolver(self)
    # ===================================

    # Create coupled iterator
    solver = FixedPointIterator(
        problem=self,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        config=config
    )

    return solver.solve(verbose=verbose)
```

**Critical issues**:
1. ❌ **Always GFDM+Particle** regardless of problem characteristics
2. ❌ **No geometry adaptation** (dimension, convexity ignored)
3. ❌ **No fallback** if GFDM collocation fails
4. ❌ **Only 3 parameters exposed** (max_iterations, tolerance, verbose)
5. ❌ **No scheme selection** - user cannot choose FDM, SL, etc.

**This is precisely what Issue #580 aims to fix.**

---

## 4. Config System Structure

### Hierarchical Pydantic Architecture (PRODUCTION-READY)

```python
MFGSolverConfig (root)
├── hjb: HJBConfig
│   ├── method: Literal["fdm", "gfdm", "semi_lagrangian", "weno"]
│   ├── accuracy_order: int [1-5]
│   ├── boundary_conditions: Literal["dirichlet", "neumann", "periodic"]
│   ├── newton: NewtonConfig
│   │   ├── max_iterations: int
│   │   ├── tolerance: float
│   │   └── line_search: bool
│   ├── fdm: FDMConfig | None
│   │   ├── scheme: "central" | "upwind" | "lax_friedrichs"
│   │   └── time_stepping: "explicit" | "implicit" | "crank_nicolson"
│   ├── gfdm: GFDMConfig | None
│   │   ├── delta: float (support radius)
│   │   ├── stencil_size: int
│   │   ├── qp_optimization_level: "none" | "auto" | "always"
│   │   ├── monotonicity_check: bool
│   │   ├── adaptive_qp: bool
│   │   └── qp_threshold: float
│   ├── sl: SLConfig | None
│   └── weno: WENOConfig | None
├── fp: FPConfig
│   ├── method: Literal["fdm", "particle", "network"]
│   ├── fdm: FDMConfig | None
│   ├── particle: ParticleConfig | None
│   │   ├── num_particles: int
│   │   ├── kde_bandwidth: float | "auto"
│   │   ├── normalization: "none" | "initial_only" | "all"
│   │   └── mode: "hybrid" | "collocation"
│   └── network: NetworkConfig | None
├── picard: PicardConfig
│   ├── max_iterations: int = 100
│   ├── tolerance: float = 1e-6
│   ├── damping_factor: float = 1.0
│   ├── anderson_memory: int = 5
│   └── verbose: bool = True
├── backend: BackendConfig
│   ├── type: "numpy" | "jax" | "pytorch"
│   ├── device: "cpu" | "gpu" | "auto"
│   └── precision: "float32" | "float64"
└── logging: LoggingConfig
    ├── level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
    ├── progress_bar: bool = True
    ├── save_intermediate: bool = False
    └── output_dir: str | None = None
```

### Auto-Population Pattern

**Already implemented** with Pydantic validators:

```python
@model_validator(mode="after")
def populate_method_configs(self):
    """Auto-populate method-specific configs if None."""
    if self.method == "fdm" and self.fdm is None:
        self.fdm = FDMConfig()
    if self.method == "gfdm" and self.gfdm is None:
        self.gfdm = GFDMConfig()
    # ... etc
    return self
```

### Gap: No Routing Logic

**Config declares methods but doesn't select solvers**:

```python
# What EXISTS:
config.hjb.method = "fdm"  # Declares intent

# What's MISSING:
hjb_solver = SOLVER_REGISTRY[config.hjb.method](problem, config)
```

**Action**: Design doc's `create_paired_solvers()` must add this routing.

---

## 5. Geometry Classes & Properties

### GeometryProtocol (PRODUCTION-READY)

**Location**: `mfg_pde/geometry/protocol.py`

```python
@runtime_checkable
class GeometryProtocol(Protocol):
    # ✅ Available properties
    @property
    def dimension(self) -> int: ...

    @property
    def geometry_type(self) -> GeometryType: ...

    @property
    def num_spatial_points(self) -> int: ...

    # ✅ Available methods
    def get_spatial_grid(self) -> NDArray | list[NDArray]: ...
    def get_bounds(self) -> NDArray: ...
    def is_on_boundary(self, points: NDArray) -> NDArray[bool]: ...
    def get_boundary_normal(self, points: NDArray) -> NDArray: ...
    def project_to_boundary(self, points: NDArray) -> NDArray: ...
    def project_to_interior(self, points: NDArray) -> NDArray: ...
    def get_boundary_regions(self) -> dict[str, NDArray]: ...
```

### GeometryType Enum

```python
class GeometryType(Enum):
    CARTESIAN_GRID = "cartesian_grid"
    NETWORK = "network"
    MAZE = "maze"
    DOMAIN_2D = "domain_2d"
    DOMAIN_3D = "domain_3d"
    IMPLICIT = "implicit"
    CUSTOM = "custom"
```

### Missing Properties for Auto-Selection

**Design doc assumes** (from `_auto_select_scheme()` heuristics):

```python
# ❌ NOT AVAILABLE
geometry.is_structured()  # Would need implementation
geometry.is_convex()      # Would need implementation
geometry.has_holes        # Could infer from MAZE type
geometry.boundary_is_smooth  # Not available
```

**Workaround**: Use `geometry_type` as proxy:

```python
# Can infer from type
is_structured = geometry.geometry_type == GeometryType.CARTESIAN_GRID
has_obstacles = geometry.geometry_type in [GeometryType.MAZE, GeometryType.IMPLICIT]
```

**Recommendation**: Add explicit methods to GeometryProtocol or provide helper functions:

```python
# mfg_pde/geometry/utils.py (NEW)
def is_structured_grid(geometry: GeometryProtocol) -> bool:
    """Check if geometry is structured (FDM-compatible)."""
    return geometry.geometry_type in [
        GeometryType.CARTESIAN_GRID,
        GeometryType.DOMAIN_2D,  # If rectangular
        GeometryType.DOMAIN_3D,  # If cuboidal
    ]

def has_complex_boundaries(geometry: GeometryProtocol) -> bool:
    """Check if geometry has obstacles/non-convexity."""
    return geometry.geometry_type in [
        GeometryType.MAZE,
        GeometryType.IMPLICIT,
        GeometryType.CUSTOM,
    ]
```

---

## 6. Existing Enums & Types

### Scattered Method Definitions (NEEDS CONSOLIDATION)

**Currently in 4 different places**:

1. **HJBConfig.method**: `Literal["fdm", "gfdm", "semi_lagrangian", "weno"]`
2. **FPConfig.method**: `Literal["fdm", "particle", "network"]`
3. **Hardcoded in problem.solve()**: GFDM + Particle
4. **SolverFactory.solver_type**: `Literal["fixed_point"]`

**Design doc proposes**: Single `NumericalScheme` enum

**Recommendation**: Create consolidated enum with HJB-FP pairs:

```python
# mfg_pde/types/numerical_schemes.py (NEW)

from enum import Enum

class NumericalScheme(Enum):
    """
    Validated HJB-FP numerical scheme pairs.

    Each scheme guarantees adjoint pairing (Type A or B).
    """
    # Type A: Exact discrete adjoint
    FDM_UPWIND = ("fdm_upwind", "fdm", "fdm")
    SL_LINEAR = ("sl_linear", "semi_lagrangian", "semi_lagrangian")

    # Type B: Asymptotic adjoint
    GFDM = ("gfdm", "gfdm", "gfdm")

    # Future
    WENO = ("weno", "weno", "weno")
    FVM_GODUNOV = ("fvm_godunov", "fvm", "fvm")

    def __init__(self, name: str, hjb_method: str, fp_method: str):
        self.scheme_name = name
        self.hjb_method = hjb_method
        self.fp_method = fp_method
```

Then use in configs:
```python
# Update MFGSolverConfig to accept enum
config.scheme = NumericalScheme.FDM_UPWIND
# Auto-populate hjb.method and fp.method
```

---

## 7. Design Assumptions vs Reality

### Comparison Table

| Feature | Design Doc Assumes | Actual Implementation | Impact on Design |
|:--------|:-------------------|:---------------------|:-----------------|
| **Solver selection** | Intelligent (geometry-aware) | Hardcoded (GFDM+Particle) | ❌ **Must implement** |
| **Scheme enum** | Single `NumericalScheme` | Scattered `Literal[...]` | ⚠️ **Consolidate** |
| **Solver traits** | `_scheme_family` attribute | Not implemented | ❌ **Add to Step 1** |
| **Config threading** | Through `create_paired_solvers()` | Only in `create_solver()` | ✅ **Extend pattern** |
| **Geometry capabilities** | `is_convex()`, `is_structured()` | Only `geometry_type` | ⚠️ **Use proxy methods** |
| **Auto-selection** | `_auto_select_scheme()` | Not implemented | ❌ **Core feature** |
| **Fallback strategies** | Primary → Secondary → Tertiary | None (fail-fast) | ❌ **Add to Step 3** |
| **Solver registry** | `SOLVER_REGISTRY[method]` | Not implemented | ❌ **Add to Step 1** |
| **Duality validation** | `check_solver_duality()` | Not implemented | ❌ **Add to Step 2** |
| **Expert mode warnings** | On non-adjoint pairs | Not implemented | ❌ **Add to Step 3** |

### Critical Mismatches

**1. problem.solve() is NOT factory-based**

Design doc Section "Three-Mode Solving API" shows:
```python
def solve(self, scheme=None, hjb_solver=None, fp_solver=None, ...):
```

Reality (`mfg_problem.py:1954`):
```python
def solve(self, max_iterations=100, tolerance=1e-6, verbose=True, config=None):
    # Hardcoded: HJBGFDMSolver + FPParticleSolver
```

**Impact**: Entire Mode 1/2/3 API needs implementation from scratch.

---

**2. Config system has methods but no routing**

Design doc assumes:
```python
if config.hjb.method == "fdm":
    hjb = HJBFDMSolver(problem, config)
```

Reality: Config declares methods, but no code reads them to select solvers.

**Impact**: Need to implement routing logic in `create_paired_solvers()`.

---

**3. Geometry introspection limited**

Design doc assumes (`_auto_select_scheme()`):
```python
if not self.geometry.is_convex():
    return NumericalScheme.GFDM
```

Reality: No `is_convex()` method exists.

**Impact**: Must use `geometry_type` proxy or implement helper functions.

---

## 8. Recommendations for Design Doc

### Priority 1: Acknowledge Current State

**Add section**: "Current Implementation Status" before "Implementation Design"

**Content**:
```markdown
## Current Implementation Status

Before implementing Issue #580, understand MFG_PDE's current state:

### What Currently Exists ✅

1. **Config System** (Production-ready)
   - Hierarchical Pydantic models with validation
   - Method-specific configs auto-populated
   - Location: `mfg_pde/config/`

2. **Solver Hierarchy** (Production-ready)
   - Base classes: `BaseMFGSolver` → `BaseNumericalSolver` → `BaseHJBSolver`/`BaseFPSolver`
   - Concrete solvers: FDM, GFDM, Semi-Lagrangian, WENO, Particle
   - Location: `mfg_pde/alg/numerical/`

3. **Geometry Protocol** (Production-ready)
   - Runtime-checkable protocol with boundary methods
   - `GeometryType` enum for categorization
   - Location: `mfg_pde/geometry/protocol.py`

4. **Problem Factories** (Production-ready)
   - Application-specific: `create_lq_problem()`, `create_crowd_problem()`, etc.
   - Location: `mfg_pde/factory/problem_factories.py`

5. **Backend Selection** (Partial)
   - Intelligent JAX vs NumPy selection based on problem size
   - Template for solver selection logic
   - Location: `mfg_pde/factory/backend_factory.py`

### What's Missing ❌

1. **Solver Selection Logic**
   - Current: `problem.solve()` hardcodes GFDM+Particle (line 1954-2026)
   - Needed: Intelligent routing based on geometry and config

2. **Scheme Enum**
   - Current: Methods scattered across 4 `Literal[...]` definitions
   - Needed: Single `NumericalScheme` enum

3. **Solver Traits**
   - Current: No `_scheme_family` attribute
   - Needed: Trait system for duality checking

4. **Solver Registry**
   - Current: No mapping from method name to solver class
   - Needed: `SOLVER_REGISTRY[method]` → `SolverClass`

5. **Geometry Capabilities**
   - Current: Only `geometry_type` enum
   - Needed: `is_structured()`, `is_convex()` helpers or use type as proxy

6. **Duality Validation**
   - Current: No runtime checking of solver compatibility
   - Needed: `check_solver_duality(hjb, fp)` with warnings

### Migration Path

Issue #580 transforms:
```python
# BEFORE (Current - mfg_problem.py:1954)
def solve(self, max_iterations=100, tolerance=1e-6, verbose=True, config=None):
    hjb_solver = HJBGFDMSolver(self, collocation_points)  # Hardcoded
    fp_solver = FPParticleSolver(self)                    # Hardcoded
    ...

# AFTER (Issue #580 implementation)
def solve(self, scheme=None, hjb_solver=None, fp_solver=None, config=None, ...):
    if scheme is not None:
        hjb_solver, fp_solver = create_paired_solvers(scheme, self, config)
    elif hjb_solver is not None and fp_solver is not None:
        validate_manual_pairing(hjb_solver, fp_solver)
    else:
        scheme = self._auto_select_scheme()
        hjb_solver, fp_solver = create_paired_solvers(scheme, self, config)
    ...
```
```

---

### Priority 2: Revise Configuration Leakage Example

**Current design doc** (lines 610-643) shows:

```python
def create_paired_solvers(
    scheme: NumericalScheme,
    problem: MFGProblem,
    config: MFGSolverConfig | None = None,
):
    if config is None:
        config = MFGSolverConfig()

    if scheme == NumericalScheme.GFDM:
        hjb = HJBGFDMSolver(
            problem,
            upwind="exponential_downwind",
            kernel_bandwidth=config.hjb.gfdm.kernel_bandwidth,  # ✅
            stencil_size=config.hjb.gfdm.stencil_size,
        )
```

**Update to match actual GFDMConfig structure**:

```python
def create_paired_solvers(
    scheme: NumericalScheme,
    problem: MFGProblem,
    config: MFGSolverConfig | None = None,
):
    if config is None:
        config = MFGSolverConfig()

    if scheme == NumericalScheme.GFDM:
        # Ensure GFDM config populated
        if config.hjb.gfdm is None:
            config.hjb.gfdm = GFDMConfig()
        if config.fp.gfdm is None:
            config.fp.gfdm = GFDMConfig()

        hjb = HJBGFDMSolver(
            problem,
            upwind="exponential_downwind",
            delta=config.hjb.gfdm.delta,  # ← Actual GFDM param name
            stencil_size=config.hjb.gfdm.stencil_size,
            qp_optimization_level=config.hjb.gfdm.qp_optimization_level,
            monotonicity_check=config.hjb.gfdm.monotonicity_check,
        )

        fp = FPGFDMSolver(
            problem,
            upwind="exponential_upwind",
            delta=config.fp.gfdm.delta,
            stencil_size=config.fp.gfdm.stencil_size,
        )

        # Auto-inject renormalization for Type B
        if config.fp.gfdm.normalization != "none":
            from mfg_pde.alg.numerical.fp_solvers.renormalization import RenormalizationWrapper
            fp = RenormalizationWrapper(fp)

        return hjb, fp
```

---

### Priority 3: Update Geometry Auto-Selection

**Current design doc** (lines 749-816) assumes methods exist:

```python
if hasattr(self.geometry, 'is_structured'):
    if not self.geometry.is_structured():
        return NumericalScheme.GFDM
```

**Reality**: These methods don't exist. Use `geometry_type` instead:

```python
def _auto_select_scheme(self) -> NumericalScheme:
    """
    Auto-select scheme based on problem characteristics.

    Uses geometry_type as proxy for structural properties since
    is_structured() and is_convex() are not implemented.
    """
    from mfg_pde.types import NumericalScheme
    from mfg_pde.geometry.protocol import GeometryType

    # Priority 1: Complex geometries force GFDM
    if self.geometry.geometry_type in [
        GeometryType.MAZE,
        GeometryType.IMPLICIT,
        GeometryType.CUSTOM,
    ]:
        return NumericalScheme.GFDM

    # Priority 2: Dimension-based for structured grids
    if self.geometry.geometry_type == GeometryType.CARTESIAN_GRID:
        if self.dimension <= 2:
            return NumericalScheme.FDM_UPWIND
        elif self.dimension == 3:
            return NumericalScheme.SL_LINEAR
        else:
            raise ValueError(
                f"Auto-selection not available for {self.dimension}D problems.\n"
                f"Please explicitly specify scheme."
            )

    # Priority 3: Network geometries
    if self.geometry.geometry_type == GeometryType.NETWORK:
        return NumericalScheme.GFDM  # Only method supporting graphs

    # Default: Conservative choice for unknown types
    return NumericalScheme.GFDM
```

---

### Priority 4: Add "Migration from Hardcoded Solvers" Section

**Add before "Implementation Design"**:

```markdown
## Migration from Hardcoded Solvers

### Current Hardcoded State

Location: `mfg_pde/core/mfg_problem.py:1954-2026`

```python
def solve(self, max_iterations=100, tolerance=1e-6, verbose=True, config=None):
    # === PROBLEM: Hardcoded solver selection ===
    hjb_solver = HJBGFDMSolver(self, collocation_points)
    fp_solver = FPParticleSolver(self)
    # ============================================

    solver = FixedPointIterator(
        problem=self,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        config=config or MFGSolverConfig()
    )
    return solver.solve(verbose=verbose)
```

**Issues with current approach**:
1. Always uses GFDM+Particle regardless of problem type
2. No geometry adaptation (1D, 2D, 3D treated identically)
3. No fallback if GFDM collocation fails
4. Cannot leverage FDM for simple rectangular domains
5. Cannot use Semi-Lagrangian for higher dimensions

### Target State (After Issue #580)

```python
def solve(
    self,
    scheme: NumericalScheme | None = None,  # NEW: Mode 1 (Safe)
    hjb_solver: BaseHJBSolver | None = None,  # NEW: Mode 2 (Expert)
    fp_solver: BaseFPSolver | None = None,    # NEW: Mode 2 (Expert)
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
    config: MFGSolverConfig | None = None,
) -> SolverResult:
    """
    Solve MFG problem using three-mode API.

    Mode 1 (Safe): result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
    Mode 2 (Expert): result = problem.solve(hjb_solver=my_hjb, fp_solver=my_fp)
    Mode 3 (Auto): result = problem.solve()  # Intelligent selection
    """

    # === MODE DETECTION ===
    if scheme is not None:
        # Mode 1: Safe - validated pairing
        hjb_solver, fp_solver = self._create_paired_solvers(scheme, config)

    elif hjb_solver is not None and fp_solver is not None:
        # Mode 2: Expert - manual injection with validation
        self._validate_manual_pairing(hjb_solver, fp_solver)

    else:
        # Mode 3: Auto - intelligent selection
        scheme = self._auto_select_scheme()
        hjb_solver, fp_solver = self._create_paired_solvers(scheme, config)
        if verbose:
            print(f"Auto-selected: {scheme.value}")

    # === COMMON PATH ===
    solver = self._create_solver_iterator(hjb_solver, fp_solver, config)
    return solver.solve(max_iterations=max_iterations, tolerance=tolerance, verbose=verbose)
```

### Migration Strategy

**Phase 1: Add new parameters** (backward compatible)
- Add `scheme`, `hjb_solver`, `fp_solver` parameters
- Keep old behavior as default (Mode 3 with hardcoded GFDM+Particle)

**Phase 2: Implement routing** (still compatible)
- Implement `_create_paired_solvers()`, `_auto_select_scheme()`, `_validate_manual_pairing()`
- Default Mode 3 now uses intelligent selection
- Old hardcoded path deprecated with warning

**Phase 3: Remove hardcoded path** (breaking change for v1.0.0)
- Remove GFDM+Particle hardcoding
- Mode 3 becomes true auto-selection
- Update all examples to use explicit schemes
```

---

### Priority 5: Update Pre-Implementation Checklist

**Current checklist** (lines 1249-1256) needs grounding in reality:

**BEFORE**:
```markdown
- [ ] All solver classes have `_scheme_family` trait
- [ ] `MFGSolverConfig` supports scheme-specific parameters
- [ ] Geometry classes expose `is_structured()` and `is_convex()` methods
- [ ] Test infrastructure can validate adjoint properties
- [ ] Deprecation strategy agreed upon (v0.17.0 → v1.0.0)
```

**AFTER** (realistic):
```markdown
- [ ] **Create** `_scheme_family` trait on `BaseSolver` (NOT EXISTS)
- [ ] **Add** `_scheme_family = SchemeFamily.XXX` to all solver classes
- [ ] ✅ **Verify** `MFGSolverConfig` structure matches actual hierarchy (EXISTS)
- [ ] **Create** `NumericalScheme` enum consolidating scattered `Literal[...]`
- [ ] **Create** `SchemeFamily` enum for trait system
- [ ] **Implement** geometry helper functions (proxy for missing `is_structured()`)
  - [ ] `is_structured_grid(geometry)` using `geometry_type`
  - [ ] `has_complex_boundaries(geometry)` using `geometry_type`
- [ ] **Create** `SOLVER_REGISTRY` mapping method → SolverClass
- [ ] **Refactor** `problem.solve()` from hardcoded to factory-based
- [ ] **Implement** fallback strategy framework
- [ ] **Add** tests for Mode 1/2/3 solving patterns
- [ ] **Update** documentation with migration guide
- [ ] **Deprecate** hardcoded GFDM+Particle with clear warning
- [ ] **Plan** backward compatibility for v0.17.0 → v1.0.0
```

---

## Summary: Design Doc Polish Checklist

Based on infrastructure survey, update design doc with:

1. ✅ **Add** "Current Implementation Status" section (what exists vs missing)
2. ✅ **Add** "Migration from Hardcoded Solvers" section (before/after comparison)
3. ✅ **Update** Configuration Leakage example with actual `GFDMConfig` structure
4. ✅ **Update** Auto-selection heuristic to use `geometry_type` (not missing methods)
5. ✅ **Revise** Pre-Implementation Checklist with realistic requirements
6. ✅ **Reference** actual file locations (`mfg_problem.py:1954`, etc.)
7. ✅ **Acknowledge** Issue #543 validator pattern (try/except, not hasattr)
8. ✅ **Add** backend selection as template for solver selection
9. ✅ **Note** scattered method Literals need consolidation into enum
10. ✅ **Document** config auto-population pattern (already works, extend to routing)

---

**End of Infrastructure Audit**
