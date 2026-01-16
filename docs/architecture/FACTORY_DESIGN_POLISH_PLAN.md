# Factory Pattern Design - Polish Plan

**Document**: Update plan for FACTORY_PATTERN_DESIGN.md based on infrastructure audit
**Date**: 2026-01-16
**Source**: FACTORY_PATTERN_INFRASTRUCTURE_AUDIT.md findings
**Status**: Ready for implementation

---

## Executive Summary

Infrastructure audit revealed **5 critical gaps** between design assumptions and MFG_PDE reality. This document provides specific updates to incorporate into FACTORY_PATTERN_DESIGN.md.

**Priority**: High - Design doc assumes features that don't exist (intelligent selection, geometry introspection, scheme enums)

**Approach**: Add grounding sections + revise examples to match actual code structure

---

## Update 1: Add "Current Implementation Status" Section

**Location**: After "Relationship to 2-Level Principle", before "Motivation"

**Insert new section** (approximately line 96):

```markdown
## Current Implementation Status

**Purpose**: Ground factory design in MFG_PDE's actual codebase state.

### What Currently Exists ✅

#### 1. Config System (Production-Ready)
- **Location**: `mfg_pde/config/`
- **Structure**: Hierarchical Pydantic models with auto-validation
- **Status**: ✅ Ready for routing logic integration
- **Example**:
  ```python
  config = MFGSolverConfig()
  config.hjb.method = "fdm"  # Declares intent
  config.hjb.gfdm = GFDMConfig(delta=0.1, stencil_size=20)
  ```

#### 2. Solver Hierarchy (Production-Ready)
- **Location**: `mfg_pde/alg/numerical/`
- **Structure**: `BaseMFGSolver` → `BaseNumericalSolver` → `BaseHJBSolver`/`BaseFPSolver`
- **Concrete Implementations**:
  - HJB: `HJBFDMSolver`, `HJBGFDMSolver`, `HJBSemiLagrangianSolver`, `HJBWENOSolver`
  - FP: `FPFDMSolver`, `FPParticleSolver`, `FPGFDMSolver`, `FPSemiLagrangianSolver`
- **Status**: ✅ Ready for trait annotations

#### 3. Geometry Protocol (Production-Ready)
- **Location**: `mfg_pde/geometry/protocol.py`
- **Features**: Runtime-checkable protocol, `GeometryType` enum
- **Available Properties**: `dimension`, `geometry_type`, `num_spatial_points`
- **Available Methods**: `is_on_boundary()`, `get_boundary_normal()`, boundary projection
- **Missing**: `is_structured()`, `is_convex()` (use `geometry_type` as proxy)
- **Status**: ✅ Ready with workarounds for missing properties

#### 4. Problem Factories (Production-Ready)
- **Location**: `mfg_pde/factory/problem_factories.py`
- **Implemented**: `create_lq_problem()`, `create_crowd_problem()`, `create_network_problem()`, etc.
- **Status**: ✅ Fully functional (Concern 1: WHAT)

#### 5. Backend Selection (Partial Template)
- **Location**: `mfg_pde/factory/backend_factory.py`
- **Feature**: Intelligent JAX vs NumPy selection based on problem size
- **Status**: ⚠️ Template for solver selection logic

### What's Missing ❌

#### 1. Solver Selection Logic (CRITICAL)
- **Current**: `mfg_problem.py:1954-2026` hardcodes GFDM+Particle
- **Code**:
  ```python
  def solve(self, max_iterations=100, tolerance=1e-6, verbose=True, config=None):
      hjb_solver = HJBGFDMSolver(self, collocation_points)  # ← Hardcoded!
      fp_solver = FPParticleSolver(self)                    # ← Hardcoded!
  ```
- **Impact**: Users cannot choose FDM, SL, or any other scheme
- **This is exactly what Issue #580 fixes**

#### 2. Numerical Scheme Enum
- **Current**: Methods scattered across 4 `Literal[...]` definitions
  - `HJBConfig.method: Literal["fdm", "gfdm", "semi_lagrangian", "weno"]`
  - `FPConfig.method: Literal["fdm", "particle", "network"]`
  - Hardcoded in `problem.solve()`
  - `SolverFactory.solver_type: Literal["fixed_point"]`
- **Needed**: Single `NumericalScheme` enum as proposed in design

#### 3. Solver Family Traits
- **Current**: No `_scheme_family` attribute on any solver
- **Needed**: `_scheme_family: SchemeFamily` for duality checking
- **Implementation**: Add to Step 1 (Infrastructure)

#### 4. Solver Registry
- **Current**: No mapping from method name to solver class
- **Needed**: `SOLVER_REGISTRY[method] → SolverClass`
- **Implementation**: Add to Step 1 (Infrastructure)

#### 5. Geometry Capability Helpers
- **Current**: Only `geometry_type` enum available
- **Needed**: Helper functions to proxy `is_structured()`, `is_convex()`
- **Workaround**:
  ```python
  def is_structured_grid(geometry):
      return geometry.geometry_type == GeometryType.CARTESIAN_GRID

  def has_complex_boundaries(geometry):
      return geometry.geometry_type in [GeometryType.MAZE, GeometryType.IMPLICIT]
  ```

#### 6. Duality Validation
- **Current**: No runtime checking of solver compatibility
- **Needed**: `check_solver_duality(hjb, fp)` with warnings
- **Implementation**: Add to Step 2 (Validation Logic)

### Migration Context

**Before Issue #580** (Current State):
- User has NO control over solver selection
- Always GFDM+Particle regardless of problem type
- 1D problems use expensive GFDM instead of simple FDM

**After Issue #580** (Target State):
- Mode 1 (Safe): `problem.solve(scheme=NumericalScheme.FDM_UPWIND)`
- Mode 2 (Expert): `problem.solve(hjb_solver=my_hjb, fp_solver=my_fp)`
- Mode 3 (Auto): `problem.solve()` with intelligent selection

**Backward Compatibility Plan**:
- v0.16.x (Current): Hardcoded GFDM+Particle
- v0.17.0 (Transition): Add `scheme` parameter, deprecate hardcoded path with warning
- v1.0.0 (Breaking): Remove hardcoded path, Mode 3 uses true auto-selection
```

**Impact**: Users understand the baseline and migration path clearly.

---

## Update 2: Revise Configuration Leakage Example

**Location**: Section "Critical Implementation Considerations" → "1. Configuration Leakage Risk"

**Current code** (lines 627-640) shows:
```python
hjb = HJBGFDMSolver(
    problem,
    upwind="exponential_downwind",
    kernel_bandwidth=config.hjb.gfdm.kernel_bandwidth,  # ← Generic name
    stencil_size=config.hjb.gfdm.stencil_size,
)
```

**Replace with actual `GFDMConfig` structure** (from audit findings):

```python
def create_paired_solvers(
    scheme: NumericalScheme,
    problem: MFGProblem,
    config: MFGSolverConfig | None = None,
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """Create adjoint-paired solvers with configuration."""

    if config is None:
        config = MFGSolverConfig()

    if scheme == NumericalScheme.GFDM:
        # Auto-populate GFDM configs if missing (Pydantic pattern)
        if config.hjb.method != "gfdm":
            config.hjb.method = "gfdm"
        if config.hjb.gfdm is None:
            config.hjb.gfdm = GFDMConfig()

        if config.fp.method != "gfdm":
            config.fp.method = "gfdm"
        if config.fp.gfdm is None:
            config.fp.gfdm = GFDMConfig()

        # Create HJB solver with actual GFDM parameters
        hjb = HJBGFDMSolver(
            problem,
            upwind="exponential_downwind",
            delta=config.hjb.gfdm.delta,  # ← Actual param name (support radius)
            stencil_size=config.hjb.gfdm.stencil_size,
            qp_optimization_level=config.hjb.gfdm.qp_optimization_level,
            monotonicity_check=config.hjb.gfdm.monotonicity_check,
            adaptive_qp=config.hjb.gfdm.adaptive_qp,
        )

        # Create FP solver with actual GFDM parameters
        fp = FPGFDMSolver(
            problem,
            upwind="exponential_upwind",
            delta=config.fp.gfdm.delta,
            stencil_size=config.fp.gfdm.stencil_size,
        )

        # Auto-inject renormalization if configured (Type B requirement)
        if config.fp.particle.normalization != "none":
            from mfg_pde.alg.numerical.fp_solvers.renormalization import RenormalizationWrapper
            fp = RenormalizationWrapper(fp, method=config.fp.particle.normalization)

        return hjb, fp

    elif scheme == NumericalScheme.FDM_UPWIND:
        # FDM doesn't need complex config, simpler
        hjb = HJBFDMSolver(
            problem,
            scheme="upwind",
            time_stepping=config.hjb.fdm.time_stepping if config.hjb.fdm else "implicit",
        )
        fp = FPFDMSolver(
            problem,
            scheme="upwind",
            time_stepping=config.fp.fdm.time_stepping if config.fp.fdm else "implicit",
        )
        return hjb, fp

    # ... other schemes
```

**Why important**: Shows users the ACTUAL config parameters they can control.

---

## Update 3: Revise Auto-Selection Heuristic

**Location**: Section "Critical Implementation Considerations" → "3. Auto-Selection Heuristic Edge Cases"

**Current code** (lines 767-816) assumes:
```python
if hasattr(self.geometry, 'is_structured'):
    if not self.geometry.is_structured():
        return NumericalScheme.GFDM
```

**Replace with `geometry_type` proxy** (from audit findings):

```python
def _auto_select_scheme(self) -> NumericalScheme:
    """
    Auto-select scheme based on problem characteristics.

    Selection hierarchy (priority order):
    1. Complex geometries (MAZE, IMPLICIT) → GFDM (only flexible method)
    2. Network topologies → GFDM (only method supporting graphs)
    3. Structured grids (1D/2D) → FDM (most stable, exact adjoint)
    4. Structured grids (3D) → SL (better scaling than FDM)
    5. High dimension (4D+) → Error (require explicit choice)

    Uses geometry_type as proxy since is_structured() and is_convex()
    methods are not implemented in GeometryProtocol.
    """
    from mfg_pde.types import NumericalScheme
    from mfg_pde.geometry.protocol import GeometryType

    # Priority 1: Complex geometries force GFDM
    if self.geometry.geometry_type in [
        GeometryType.MAZE,       # Obstacles, non-convex
        GeometryType.IMPLICIT,   # Level-set boundaries
        GeometryType.CUSTOM,     # Unknown structure
    ]:
        return NumericalScheme.GFDM

    # Priority 2: Network geometries require GFDM
    if self.geometry.geometry_type == GeometryType.NETWORK:
        return NumericalScheme.GFDM

    # Priority 3: Structured grids - dimension-based selection
    if self.geometry.geometry_type in [
        GeometryType.CARTESIAN_GRID,
        GeometryType.DOMAIN_2D,  # If rectangular
        GeometryType.DOMAIN_3D,  # If cuboidal
    ]:
        if self.dimension <= 2:
            # Low dimension + structured → FDM (most stable, exact adjoint)
            return NumericalScheme.FDM_UPWIND

        elif self.dimension == 3:
            # 3D + structured → Semi-Lagrangian (better scaling)
            return NumericalScheme.SL_LINEAR

        else:
            # High dimension → Force explicit choice
            raise ValueError(
                f"Auto-selection not available for {self.dimension}D problems.\n"
                f"Please explicitly specify scheme:\n"
                f"  - NumericalScheme.SL_LINEAR (up to ~6D)\n"
                f"  - NumericalScheme.GFDM (flexible, any dimension)\n"
                f"High-dimensional MFG requires careful scheme selection."
            )

    # Default: Conservative fallback for unknown geometry types
    # (Better to use flexible method than fail)
    return NumericalScheme.GFDM


def _get_selection_rationale(self, scheme: NumericalScheme) -> str:
    """Get human-readable rationale for auto-selection."""
    if scheme == NumericalScheme.FDM_UPWIND:
        return f"{self.dimension}D structured grid (CARTESIAN_GRID)"

    elif scheme == NumericalScheme.SL_LINEAR:
        return f"3D structured grid (better scaling than FDM)"

    elif scheme == NumericalScheme.GFDM:
        geo_type = self.geometry.geometry_type
        if geo_type == GeometryType.MAZE:
            return "Complex geometry with obstacles"
        elif geo_type == GeometryType.IMPLICIT:
            return "Implicit boundary (level-set)"
        elif geo_type == GeometryType.NETWORK:
            return "Network/graph topology"
        elif geo_type == GeometryType.CUSTOM:
            return "Custom geometry (unknown structure)"
        else:
            return "Conservative fallback for unknown type"

    else:
        return "Unknown selection criteria"
```

**Key changes**:
1. Uses `geometry_type` enum instead of missing methods
2. Explicit handling of all `GeometryType` variants
3. Clear comments explaining proxy approach
4. Conservative fallback for unknown types

---

## Update 4: Add Solver Registry Example

**Location**: Section "Implementation Design" → after "Directory Structure"

**Add new subsection** (after line 851):

```markdown
### Solver Registry Pattern

**Purpose**: Map method names to solver classes for dynamic instantiation.

**Location**: `mfg_pde/factory/scheme_factory.py` (NEW)

```python
# Solver registry for dynamic instantiation
from typing import Type
from mfg_pde.alg.numerical.hjb_solvers import (
    BaseHJBSolver,
    HJBFDMSolver,
    HJBGFDMSolver,
    HJBSemiLagrangianSolver,
    HJBWENOSolver,
)
from mfg_pde.alg.numerical.fp_solvers import (
    BaseFPSolver,
    FPFDMSolver,
    FPGFDMSolver,
    FPSemiLagrangianSolver,
    FPParticleSolver,
)

# HJB solver registry
HJB_SOLVER_REGISTRY: dict[str, Type[BaseHJBSolver]] = {
    "fdm": HJBFDMSolver,
    "gfdm": HJBGFDMSolver,
    "semi_lagrangian": HJBSemiLagrangianSolver,
    "weno": HJBWENOSolver,
}

# FP solver registry
FP_SOLVER_REGISTRY: dict[str, Type[BaseFPSolver]] = {
    "fdm": FPFDMSolver,
    "gfdm": FPGFDMSolver,
    "semi_lagrangian": FPSemiLagrangianSolver,
    "particle": FPParticleSolver,
}

# Usage in create_paired_solvers()
def create_paired_solvers(
    scheme: NumericalScheme,
    problem: MFGProblem,
    config: MFGSolverConfig | None = None,
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """Create solvers using registry pattern."""

    if config is None:
        config = MFGSolverConfig()

    # Map scheme to method names
    hjb_method = scheme.hjb_method  # e.g., "fdm" from NumericalScheme.FDM_UPWIND
    fp_method = scheme.fp_method    # e.g., "fdm"

    # Get solver classes from registry
    HJBClass = HJB_SOLVER_REGISTRY[hjb_method]
    FPClass = FP_SOLVER_REGISTRY[fp_method]

    # Instantiate with config
    hjb = HJBClass(problem, config=config.hjb)
    fp = FPClass(problem, config=config.fp)

    # Auto-inject middleware for Type B schemes
    if scheme in [NumericalScheme.GFDM]:
        from mfg_pde.alg.numerical.fp_solvers.renormalization import RenormalizationWrapper
        fp = RenormalizationWrapper(fp)

    return hjb, fp
```

**Benefits**:
- Dynamic solver instantiation without hardcoding
- Extensible: Add new methods by updating registry
- Type-safe: Registry enforces `Type[BaseHJBSolver]`
```

---

## Update 5: Revise Pre-Implementation Checklist

**Location**: Section "Architectural Audit Results" → "Pre-Implementation Checklist"

**Current checklist** (lines 1249-1256) is aspirational, not grounded.

**Replace with realistic checklist**:

```markdown
### Pre-Implementation Checklist

**Verify current infrastructure before implementing Issue #580:**

#### Step 0: Infrastructure Verification ✓ (Already Done)

- [x] Config system structure mapped (see INFRASTRUCTURE_AUDIT.md)
- [x] Solver class hierarchy documented
- [x] Geometry protocol capabilities identified
- [x] Current `problem.solve()` behavior understood (hardcoded GFDM+Particle)
- [x] Config auto-population pattern validated (Pydantic `@model_validator`)

#### Step 1: Foundation (NEW Code Required)

- [ ] **Create** `NumericalScheme` enum consolidating scattered `Literal[...]`
  - [ ] Define HJB-FP pairs with `(name, hjb_method, fp_method)` structure
  - [ ] Location: `mfg_pde/types/numerical_schemes.py`

- [ ] **Create** `SchemeFamily` enum for trait system
  - [ ] Values: FDM, SL, FVM, GFDM, PINN, GENERIC
  - [ ] Location: `mfg_pde/alg/numerical/base_solver.py`

- [ ] **Add** `_scheme_family` trait to `BaseSolver`
  - [ ] Class variable with default `SchemeFamily.GENERIC`

- [ ] **Annotate** all solver classes with `_scheme_family`
  - [ ] `HJBFDMSolver._scheme_family = SchemeFamily.FDM`
  - [ ] `FPFDMSolver._scheme_family = SchemeFamily.FDM`
  - [ ] `HJBGFDMSolver._scheme_family = SchemeFamily.GFDM`
  - [ ] ... (8 solver classes total)

- [ ] **Create** solver registries
  - [ ] `HJB_SOLVER_REGISTRY: dict[str, Type[BaseHJBSolver]]`
  - [ ] `FP_SOLVER_REGISTRY: dict[str, Type[BaseFPSolver]]`
  - [ ] Location: `mfg_pde/factory/scheme_factory.py`

#### Step 2: Validation Logic (NEW Functions Required)

- [ ] **Implement** `check_solver_duality()` using `_scheme_family` traits
  - [ ] Return type: `Literal["exact", "asymptotic", "none"]`
  - [ ] Location: `mfg_pde/utils/adjoint_validation.py`

- [ ] **Implement** `create_paired_solvers()` with config threading
  - [ ] Use solver registries for instantiation
  - [ ] Auto-populate method configs (follow existing Pydantic pattern)
  - [ ] Auto-inject renormalization for Type B schemes
  - [ ] Location: `mfg_pde/factory/scheme_factory.py`

#### Step 3: Facade Integration (REFACTOR Existing Code)

- [ ] **Refactor** `problem.solve()` signature
  - [ ] Add `scheme: NumericalScheme | None` parameter
  - [ ] Add `hjb_solver: BaseHJBSolver | None` parameter
  - [ ] Add `fp_solver: BaseFPSolver | None` parameter
  - [ ] Keep existing parameters for backward compatibility

- [ ] **Implement** mode detection logic
  - [ ] Mode 1 (Safe): `if scheme is not None`
  - [ ] Mode 2 (Expert): `elif hjb_solver and fp_solver`
  - [ ] Mode 3 (Auto): `else` with `_auto_select_scheme()`

- [ ] **Implement** `_create_paired_solvers()` internal method
  - [ ] Calls `create_paired_solvers()` from factory

- [ ] **Implement** `_auto_select_scheme()` with geometry_type logic
  - [ ] Use audit findings (geometry_type proxy, not missing methods)

- [ ] **Implement** `_validate_manual_pairing()` with warnings
  - [ ] Calls `check_solver_duality()`
  - [ ] Emits detailed `UserWarning` for non-adjoint pairs

#### Step 4: Deprecation (Transition Strategy)

- [ ] **Add** deprecation warning to hardcoded GFDM+Particle path
  - [ ] Trigger when `scheme is None and hjb_solver is None`
  - [ ] Message: "Hardcoded GFDM+Particle is deprecated. Specify scheme or use auto-selection."

- [ ] **Mark** `create_solver()` as deprecated
  - [ ] Add `warnings.warn(DeprecationWarning)`
  - [ ] Update docstring with migration guide

- [ ] **Plan** removal timeline
  - [ ] v0.17.0: Deprecation warnings active
  - [ ] v0.18.0: Warning escalates to error in strict mode
  - [ ] v1.0.0: Hardcoded path removed entirely

#### Step 5: Testing & Documentation

- [ ] **Add** unit tests for `check_solver_duality()`
  - [ ] Test exact pairs (FDM-FDM, SL-SL)
  - [ ] Test asymptotic pairs (GFDM-GFDM)
  - [ ] Test non-pairs (SL-FDM)

- [ ] **Add** integration tests for `problem.solve()`
  - [ ] Test Mode 1 (Safe): All schemes in `NumericalScheme`
  - [ ] Test Mode 2 (Expert): Manual injection with warnings
  - [ ] Test Mode 3 (Auto): Each geometry type

- [ ] **Update** user documentation
  - [ ] Migration guide from v0.16.x to v0.17.0
  - [ ] Examples for all three modes
  - [ ] Mark internal factories as "do not use directly"

- [ ] **Run** CI/CD pipeline
  - [ ] All existing tests pass
  - [ ] No new type errors
  - [ ] Documentation builds successfully
```

**Key difference**: Checklist now explicitly states "CREATE" vs "VERIFY" and references actual file locations.

---

## Update 6: Add Backend Selection as Template

**Location**: Section "Current Factory Implementations" (or new subsection in "Implementation Design")

**Add new subsection showing pattern to follow**:

```markdown
### Learning from Backend Selection (Template)

**Observation**: MFG_PDE already implements intelligent selection for backends.

**Location**: `mfg_pde/factory/backend_factory.py`

**Pattern** (simplified):

```python
class BackendFactory:
    @staticmethod
    def create_optimal_backend(
        problem: MFGProblem,
        prefer_gpu: bool = False,
        precision: Literal["float32", "float64"] = "float64"
    ):
        """Intelligent backend selection based on problem size."""

        # Calculate problem size
        total_size = np.prod(problem.geometry.get_spatial_grid().shape) * problem.Nt

        # Size-based heuristic
        if total_size > 100_000:
            # Large problem → JAX (GPU-friendly)
            return JAXBackend(device="gpu" if prefer_gpu else "cpu", precision=precision)
        else:
            # Small/medium problem → NumPy (simpler)
            return NumPyBackend(precision=precision)
```

**Key lessons for solver selection**:

1. **Problem introspection**: Calculate size, check geometry type
2. **Heuristic-based routing**: Use thresholds (dimension, size, type)
3. **Fallback logic**: Default to safest option (NumPy for backends, FDM for solvers)
4. **Config override**: User can force backend with `config.backend.type`

**Apply to solver selection**:

```python
def _auto_select_scheme(self) -> NumericalScheme:
    """Follow backend_factory pattern for consistency."""

    # Introspection (like total_size calculation)
    geometry_type = self.geometry.geometry_type
    dimension = self.dimension

    # Heuristic routing (like size thresholds)
    if geometry_type in [GeometryType.MAZE, GeometryType.IMPLICIT]:
        return NumericalScheme.GFDM  # Complex geometry
    elif geometry_type == GeometryType.CARTESIAN_GRID:
        if dimension <= 2:
            return NumericalScheme.FDM_UPWIND  # Simple structured
        else:
            return NumericalScheme.SL_LINEAR  # High-dim structured
    else:
        return NumericalScheme.GFDM  # Conservative fallback
```

**Benefit**: Consistent API pattern across MFG_PDE (backend selection ↔ solver selection).
```

---

## Update 7: Add Reference to Validator Pattern

**Location**: Section "Implementation Design" → "Critical Implementation Considerations" → "2. Type Checking Brittleness Risk"

**After showing trait-based solution, add note**:

```markdown
### Validator Pattern Consistency (Issue #543)

**Note**: MFG_PDE migrated from `hasattr()` to `try/except AttributeError` (Issue #543).

**Current pattern** (in `BaseFPSolver._validate_problem_compatibility()`):

```python
# ✅ CORRECT (Issue #543 pattern)
try:
    required_attr = problem.some_attribute
except AttributeError:
    raise ValueError(f"Problem missing required attribute: some_attribute")

# ❌ DEPRECATED (old pattern)
if not hasattr(problem, 'some_attribute'):
    raise ValueError(...)
```

**Implication for `check_solver_duality()`**:

```python
def check_solver_duality(hjb_solver, fp_solver) -> AdjointType:
    """Check duality using _scheme_family trait."""

    # Follow Issue #543 validator pattern
    try:
        hjb_family = hjb_solver._scheme_family
    except AttributeError:
        hjb_family = SchemeFamily.GENERIC  # Default for custom solvers

    try:
        fp_family = fp_solver._scheme_family
    except AttributeError:
        fp_family = SchemeFamily.GENERIC

    # Duality logic...
```

**Consistency**: All validation logic should use try/except, not hasattr().
```

---

## Update 8: Add Table of Contents Link

**Location**: Table of Contents section (line 86)

**Add new entries**:

```markdown
## Table of Contents

1. [Motivation](#motivation)
2. [Relationship to 2-Level Principle](#relationship-to-2-level-principle)
3. [Current Implementation Status](#current-implementation-status)  ← NEW
4. [Three-Concern Separation](#three-concern-separation)
5. [Three-Mode Solving API](#three-mode-solving-api)
6. [Implementation Design](#implementation-design)
   - [Critical Implementation Considerations](#critical-implementation-considerations)
   - [Learning from Backend Selection](#learning-from-backend-selection)  ← NEW
   - [Solver Registry Pattern](#solver-registry-pattern)  ← NEW
7. [Anti-Confusion Strategy](#anti-confusion-strategy)
8. [Migration Guide](#migration-guide)
9. [Examples](#examples)
10. [Architectural Audit Results](#architectural-audit-results)
```

---

## Summary: Polish Action Items

**8 specific updates** to apply to FACTORY_PATTERN_DESIGN.md:

1. ✅ **Add** "Current Implementation Status" section (what exists vs missing)
2. ✅ **Revise** Configuration Leakage example (actual `GFDMConfig` params)
3. ✅ **Revise** Auto-selection heuristic (use `geometry_type`, not missing methods)
4. ✅ **Add** Solver Registry Pattern subsection
5. ✅ **Revise** Pre-Implementation Checklist (realistic, grounded)
6. ✅ **Add** Backend Selection template subsection
7. ✅ **Add** Validator Pattern consistency note (Issue #543 reference)
8. ✅ **Update** Table of Contents with new sections

**Expected outcome**: Design doc accurately reflects MFG_PDE's current infrastructure and provides clear migration path from hardcoded to factory-based solver selection.

---

**End of Polish Plan**
