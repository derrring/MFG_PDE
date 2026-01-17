# Issue #580 Implementation Plan: Adjoint-Aware Solver Pairing

**Document Status**: Active Implementation Roadmap
**Issue**: #580 (Adjoint-aware solver pairing for HJB-FP duality)
**Design Reference**: `FACTORY_PATTERN_DESIGN.md` v1.3
**Date Created**: 2026-01-17

---

## Executive Summary

**Goal**: Implement three-mode solving API with mathematical duality guarantees.

**Current State** (v0.16.x):
- `problem.solve()` hardcoded to GFDM+Particle (line 1954-2026)
- No user control over numerical scheme
- No validation of solver duality

**Target State** (v0.17.0):
```python
# Mode 1 (Safe): Validated scheme pairing
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

# Mode 2 (Expert): Manual injection with warnings
result = problem.solve(hjb_solver=my_hjb, fp_solver=my_fp)

# Mode 3 (Auto): Intelligent selection
result = problem.solve()  # Auto-selects based on geometry
```

**Implementation Risk**: HIGH
- Refactors core `problem.solve()` (breaks existing usage)
- Requires careful migration of all examples and tests
- Must preserve backward compatibility during deprecation period

---

## Related Issues and Dependencies

### Completed Dependencies ✅

| Issue | Title | Status | Relevance |
|:------|:------|:-------|:----------|
| #578 | Semi-Lagrangian FP solver | CLOSED | Provides FPSLSolver for SL-SL pairing |
| #543 | Eliminate hasattr() duck typing | CLOSED | Validator pattern (try/except AttributeError) |
| #542 | FDM BC handling | CLOSED | Fixed BC retrieval for FDM solvers |

### Open Dependencies ⚠️

| Issue | Title | Impact on #580 |
|:------|:------|:---------------|
| #574 | HJB BC consistency at reflecting boundaries | MEDIUM - Affects FDM auto-selection quality |
| #573 | Generalize FP drift for non-quadratic Hamiltonians | LOW - Future extension |
| #517 | BC Semantic Dispatch Factory | LOW - Orthogonal to solver pairing |
| #583 | HJB SL cubic interpolation NaN | MEDIUM - Affects SL_CUBIC scheme stability |

**Decision**: Implement #580 independently. Mark SL_CUBIC as experimental until #583 resolved.

---

## Implementation Phases (5 Phases)

### Phase 1: Infrastructure Foundation (2-3 days, LOW RISK)

**Goal**: Add enums and traits without touching existing solver logic.

#### 1.1 Create NumericalScheme Enum

**File**: `mfg_pde/types/schemes.py` (NEW)

```python
from enum import Enum

class NumericalScheme(Enum):
    """
    Numerical discretization schemes for MFG with duality guarantees.

    Discrete Duality (exact adjoint at matrix level):
    - FDM_UPWIND: First-order upwind finite differences
    - SL_LINEAR: Semi-Lagrangian with linear interpolation
    - SL_CUBIC: Semi-Lagrangian with cubic interpolation (experimental)

    Continuous Duality (asymptotic as h→0, requires renormalization):
    - GFDM: Meshfree generalized finite differences

    See docs/theory/adjoint_operators_mfg.md for mathematical foundation.
    """
    # Discrete duality schemes (Type A)
    FDM_UPWIND = "fdm_upwind"
    FDM_CENTERED = "fdm_centered"
    SL_LINEAR = "sl_linear"
    SL_CUBIC = "sl_cubic"  # Experimental (Issue #583)

    # Continuous duality schemes (Type B)
    GFDM = "gfdm"
```

**Export**: Add to `mfg_pde/types/__init__.py`

**Tests**:
- `tests/unit/types/test_schemes.py` - Enum values, string representation
- Verify no circular imports

**Acceptance Criteria**:
- [ ] Enum defined with docstrings
- [ ] Exported in `mfg_pde/__init__.py`
- [ ] Unit tests passing
- [ ] Type checker passes (mypy)

---

#### 1.2 Create SchemeFamily Enum

**File**: `mfg_pde/alg/numerical/base_solver.py` (MODIFY)

```python
from enum import Enum

class SchemeFamily(Enum):
    """
    Numerical scheme families for duality checking.

    Used as trait annotation on solver classes to enable refactoring-safe
    duality validation (avoids brittle string matching on class names).
    """
    FDM = "fdm"           # Finite Difference Methods
    SL = "semi_lagrangian"  # Semi-Lagrangian
    FVM = "fvm"           # Finite Volume (future)
    GFDM = "gfdm"         # Meshfree GFDM
    PINN = "pinn"         # Physics-Informed Neural Network (future)
    GENERIC = "generic"   # Unknown/custom solvers
```

**Acceptance Criteria**:
- [ ] Enum added to base_solver.py
- [ ] Imported in hjb_solvers/__init__.py and fp_solvers/__init__.py
- [ ] No breaking changes to existing solver classes

---

#### 1.3 Add _scheme_family Trait to Solvers

**Goal**: Annotate all solver classes with their scheme family.

**Pattern** (Issue #543 validator pattern):
```python
class HJBFDMSolver(BaseHJBSolver):
    """FDM solver for HJB equation."""

    # Trait annotation (class-level constant)
    _scheme_family: SchemeFamily = SchemeFamily.FDM
```

**Files to Modify** (10 total):

**HJB Solvers**:
1. `hjb_solvers/hjb_fdm.py` → `SchemeFamily.FDM`
2. `hjb_solvers/hjb_semi_lagrangian.py` → `SchemeFamily.SL`
3. `hjb_solvers/hjb_gfdm.py` → `SchemeFamily.GFDM`
4. `hjb_solvers/hjb_weno.py` → `SchemeFamily.FDM` (WENO is FDM variant)

**FP Solvers**:
5. `fp_solvers/fp_fdm.py` → `SchemeFamily.FDM`
6. `fp_solvers/fp_semi_lagrangian.py` → `SchemeFamily.SL`
7. `fp_solvers/fp_semi_lagrangian_adjoint.py` → `SchemeFamily.SL`
8. `fp_solvers/fp_gfdm.py` → `SchemeFamily.GFDM`
9. `fp_solvers/fp_particle.py` → `SchemeFamily.GFDM` (particle ≈ meshfree)

**Base Classes**:
10. `base_hjb.py`, `base_fp.py` → Add default `_scheme_family = SchemeFamily.GENERIC`

**Tests**:
- `tests/unit/alg/test_solver_traits.py` - Verify all solvers have trait
- Check trait inheritance works correctly

**Acceptance Criteria**:
- [ ] All production solver classes annotated
- [ ] Base classes have default GENERIC
- [ ] Trait retrieval tests passing
- [ ] No runtime errors in existing tests

---

### Phase 2: Validation and Factory (3-4 days, MEDIUM RISK)

**Goal**: Implement duality checking and paired solver creation.

#### 2.1 Implement Duality Validation

**File**: `mfg_pde/utils/adjoint_validation.py` (NEW)

```python
"""
Adjoint Validation Utilities

Provides runtime validation of solver duality relationships.
See docs/theory/adjoint_operators_mfg.md for mathematical foundation.
"""

from typing import Literal

from mfg_pde.alg.numerical.base_solver import SchemeFamily

AdjointType = Literal["exact", "asymptotic", "none"]


def check_solver_duality(hjb_solver, fp_solver) -> AdjointType:
    """
    Check if HJB and FP solvers form a dual pair.

    Returns:
        - "exact": Discrete transpose (L_FP = L_HJB^T exactly)
        - "asymptotic": Continuous duality (L_FP = L_HJB^T + O(h))
        - "none": No duality relationship

    Type A Pairs (Exact):
        - FDM-FDM: Same upwind/centered scheme
        - SL-SL: Linear/cubic interpolation with adjoint splatting
        - FVM-FVM: Godunov flux (future)

    Type B Pairs (Asymptotic):
        - GFDM-GFDM: Complementary upwind parameters
        - PINN-PINN: Shared network architecture (future)

    Non-Dual Pairs:
        - Any mixed combination (e.g., SL + FDM)

    Example:
        >>> hjb = HJBFDMSolver(problem)
        >>> fp = FPFDMSolver(problem)
        >>> check_solver_duality(hjb, fp)
        'exact'

        >>> hjb = HJBSLSolver(problem)
        >>> fp = FPFDMSolver(problem)  # WRONG!
        >>> check_solver_duality(hjb, fp)
        'none'
    """
    # Use getattr() with default for optional trait (Issue #543)
    hjb_family = getattr(hjb_solver, '_scheme_family', SchemeFamily.GENERIC)
    fp_family = getattr(fp_solver, '_scheme_family', SchemeFamily.GENERIC)

    # Type A: Same family (except GENERIC) → Exact duality
    if (hjb_family == fp_family and
        hjb_family != SchemeFamily.GENERIC):

        # Special handling for Type B schemes
        if hjb_family == SchemeFamily.GFDM:
            return "asymptotic"

        # Type A schemes (FDM, SL, FVM)
        return "exact"

    # No duality
    return "none"


def get_duality_description(adjoint_type: AdjointType) -> str:
    """Get human-readable description of duality type."""
    descriptions = {
        "exact": "Discrete adjoint (L_FP = L_HJB^T exactly)",
        "asymptotic": "Continuous duality (L_FP = L_HJB^T + O(h))",
        "none": "No duality relationship"
    }
    return descriptions[adjoint_type]
```

**Tests**:
- `tests/unit/utils/test_adjoint_validation.py`
  - Test all valid pairs (FDM-FDM, SL-SL, GFDM-GFDM)
  - Test invalid pairs (SL-FDM, FDM-GFDM)
  - Test unknown solvers (GENERIC)

**Acceptance Criteria**:
- [ ] Function implemented with comprehensive docstring
- [ ] Uses validator pattern (Issue #543)
- [ ] Unit tests cover all cases
- [ ] Exported in mfg_pde/__init__.py

---

#### 2.2 Create Scheme Factory

**File**: `mfg_pde/factory/scheme_factory.py` (NEW)

```python
"""
Numerical Scheme Factory

Creates adjoint-paired HJB-FP solver pairs with guaranteed mathematical duality.
"""

from mfg_pde.config import MFGSolverConfig, GFDMConfig
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.types import NumericalScheme
from mfg_pde.alg.numerical.hjb_solvers import (
    HJBFDMSolver, HJBSLSolver, HJBGFDMSolver
)
from mfg_pde.alg.numerical.fp_solvers import (
    FPFDMSolver, FPSLAdjointSolver, FPGFDMSolver
)
from mfg_pde.workflow import RenormalizationWrapper


def create_paired_solvers(
    scheme: NumericalScheme,
    problem: MFGProblem,
    config: MFGSolverConfig | None = None,
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create adjoint-paired HJB and FP solvers.

    This function guarantees mathematical duality between the returned solvers:
    - Type A (exact): Discrete operators are matrix transposes
    - Type B (asymptotic): Continuous duality as h→0

    Args:
        scheme: Numerical discretization scheme
        problem: MFG problem to solve
        config: Solver configuration (optional, uses defaults if None)

    Returns:
        (hjb_solver, fp_solver): Paired solvers with guaranteed duality

    Raises:
        ValueError: If scheme is unknown

    Example:
        >>> from mfg_pde import create_crowd_problem, NumericalScheme
        >>> from mfg_pde.factory import create_paired_solvers
        >>>
        >>> problem = create_crowd_problem(...)
        >>> hjb, fp = create_paired_solvers(NumericalScheme.FDM_UPWIND, problem)
        >>> # hjb and fp are now guaranteed to be dual
    """
    if config is None:
        config = MFGSolverConfig()

    # Dispatch to scheme-specific factory
    if scheme == NumericalScheme.FDM_UPWIND:
        return _create_fdm_pair(problem, config, upwind=True)

    elif scheme == NumericalScheme.FDM_CENTERED:
        return _create_fdm_pair(problem, config, upwind=False)

    elif scheme == NumericalScheme.SL_LINEAR:
        return _create_sl_pair(problem, config, interpolation="linear")

    elif scheme == NumericalScheme.SL_CUBIC:
        return _create_sl_pair(problem, config, interpolation="cubic")

    elif scheme == NumericalScheme.GFDM:
        return _create_gfdm_pair(problem, config)

    else:
        raise ValueError(
            f"Unknown scheme: {scheme}\n"
            f"Available schemes: {[s.value for s in NumericalScheme]}"
        )


def _create_fdm_pair(
    problem: MFGProblem,
    config: MFGSolverConfig,
    upwind: bool,
) -> tuple[HJBFDMSolver, FPFDMSolver]:
    """Create FDM adjoint pair."""
    hjb = HJBFDMSolver(
        problem,
        upwind=upwind,
        newton_config=config.hjb.newton,
    )

    fp = FPFDMSolver(
        problem,
        upwind=upwind,  # Must match HJB for duality!
        scheme=config.fp.fdm.scheme if config.fp.fdm else "upwind",
    )

    return hjb, fp


def _create_sl_pair(
    problem: MFGProblem,
    config: MFGSolverConfig,
    interpolation: str,
) -> tuple[HJBSLSolver, FPSLAdjointSolver]:
    """Create Semi-Lagrangian adjoint pair."""
    hjb = HJBSLSolver(
        problem,
        interpolation_method=interpolation,
        rk_order=config.hjb.sl.rk_order if config.hjb.sl else 2,
    )

    # Use adjoint FP solver (forward splatting, not backward interpolation)
    fp = FPSLAdjointSolver(
        problem,
        interpolation_method=interpolation,  # Must match HJB!
    )

    return hjb, fp


def _create_gfdm_pair(
    problem: MFGProblem,
    config: MFGSolverConfig,
) -> tuple[HJBGFDMSolver, BaseFPSolver]:
    """Create GFDM pair with mass renormalization."""
    # Auto-populate GFDM configs if missing (Pydantic pattern)
    if config.hjb.gfdm is None:
        config.hjb.gfdm = GFDMConfig()
    if config.fp.gfdm is None:
        config.fp.gfdm = GFDMConfig()

    # Create HJB solver
    hjb = HJBGFDMSolver(
        problem,
        upwind="exponential_downwind",
        delta=config.hjb.gfdm.delta,
        stencil_size=config.hjb.gfdm.stencil_size,
        qp_optimization_level=config.hjb.gfdm.qp_optimization_level,
        monotonicity_check=config.hjb.gfdm.monotonicity_check,
        adaptive_qp=config.hjb.gfdm.adaptive_qp,
    )

    # Create FP solver
    fp = FPGFDMSolver(
        problem,
        upwind="exponential_upwind",  # Complementary to HJB
        delta=config.fp.gfdm.delta,
        stencil_size=config.fp.gfdm.stencil_size,
        qp_optimization_level=config.fp.gfdm.qp_optimization_level,
        monotonicity_check=config.fp.gfdm.monotonicity_check,
        adaptive_qp=config.fp.gfdm.adaptive_qp,
    )

    # Wrap with automatic mass renormalization (GFDM requires this)
    fp = RenormalizationWrapper(fp)

    return hjb, fp
```

**Tests**:
- `tests/unit/factory/test_scheme_factory.py`
  - Test all schemes create correct solver types
  - Test config threading works
  - Test default configs applied correctly
  - Test GFDM gets RenormalizationWrapper

**Acceptance Criteria**:
- [ ] Factory function implemented
- [ ] All Phase 1 schemes supported (FDM, SL, GFDM)
- [ ] Config parameters thread through correctly
- [ ] Unit tests passing
- [ ] Exported in mfg_pde/factory/__init__.py

---

### Phase 3: Facade Integration (4-5 days, HIGH RISK)

**Goal**: Refactor `problem.solve()` with three-mode API.

**⚠️ CRITICAL**: This phase modifies core user-facing API. Requires extensive testing.

#### 3.1 Implement Auto-Selection Logic

**File**: `mfg_pde/core/mfg_problem.py` (MODIFY)

Add method to MFGProblem class:

```python
def _auto_select_scheme(self) -> NumericalScheme:
    """
    Auto-select numerical scheme based on problem characteristics.

    Selection hierarchy (priority order):
    1. Obstacles → GFDM (only scheme that handles them)
    2. Complex geometry → GFDM (avoid FDM staircase errors)
    3. Low dimension (1D/2D) + regular grid → FDM (most stable)
    4. Medium dimension (3D) → SL (better scaling)
    5. High dimension (4D+) → Error (explicit choice required)

    Returns:
        Recommended NumericalScheme

    Conservative philosophy: When in doubt, choose FDM (most stable).
    """
    from mfg_pde.types import NumericalScheme

    # Priority 1: Obstacles force GFDM
    if self.has_obstacles:
        return NumericalScheme.GFDM

    # Priority 2: Geometry complexity check
    if self.dimension <= 2:
        # Use geometry_type enum as proxy (is_structured()/is_convex() not yet implemented)
        geom_type = self.geometry.geometry_type

        # Complex geometries → GFDM (avoid FDM staircase errors)
        if geom_type in ["unstructured_mesh", "point_cloud"]:
            return NumericalScheme.GFDM

        # Non-rectangular domains → GFDM for smooth boundary treatment
        if geom_type in ["polygon", "implicit"]:
            return NumericalScheme.GFDM

        # Regular grid (structured_grid, cartesian_grid) → FDM (most stable)
        if geom_type in ["structured_grid", "cartesian_grid"]:
            return NumericalScheme.FDM_UPWIND

        # Unknown geometry type → Default to FDM (conservative choice)
        return NumericalScheme.FDM_UPWIND

    # Priority 3: Dimension-based selection
    elif self.dimension == 3:
        # 3D → Semi-Lagrangian (better scaling than FDM)
        return NumericalScheme.SL_LINEAR

    else:
        # High dimension (4D+) → Force explicit choice
        raise ValueError(
            f"Auto-selection not available for {self.dimension}D problems.\n"
            f"Please explicitly specify scheme:\n"
            f"  - NumericalScheme.SL_LINEAR (up to ~6D)\n"
            f"  - NumericalScheme.GFDM (flexible, any dimension)\n"
            f"High-dimensional MFG requires careful scheme selection."
        )


def _get_selection_rationale(self, scheme: NumericalScheme) -> str:
    """Get human-readable rationale for auto-selection."""
    if scheme == NumericalScheme.FDM_UPWIND:
        return f"{self.dimension}D {self.geometry.geometry_type}, structured domain"
    elif scheme == NumericalScheme.SL_LINEAR:
        return f"3D problem, Semi-Lagrangian scales better"
    elif scheme == NumericalScheme.GFDM:
        if self.has_obstacles:
            return "Complex geometry with obstacles"

        geom_type = self.geometry.geometry_type
        if geom_type in ["unstructured_mesh", "point_cloud"]:
            return f"Unstructured geometry ({geom_type})"
        elif geom_type in ["polygon", "implicit"]:
            return "Non-convex domain (avoid FDM staircase errors)"
        else:
            return "Complex geometry requiring flexible discretization"
    else:
        return "Unknown selection criteria"
```

**Tests**:
- `tests/unit/core/test_auto_selection.py`
  - Test 1D/2D structured → FDM_UPWIND
  - Test 2D unstructured → GFDM
  - Test 3D → SL_LINEAR
  - Test with obstacles → GFDM
  - Test 4D+ → ValueError

**Acceptance Criteria**:
- [ ] Auto-selection logic implemented
- [ ] Rationale function provides clear explanations
- [ ] Unit tests cover all branches
- [ ] Handles unknown geometry types gracefully

---

#### 3.2 Refactor problem.solve() Signature

**File**: `mfg_pde/core/mfg_problem.py` (MODIFY)

**Current signature** (line ~1954):
```python
def solve(
    self,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
    config: MFGSolverConfig | None = None,
) -> SolverResult:
```

**New signature**:
```python
def solve(
    self,
    scheme: NumericalScheme | None = None,  # NEW: Safe Mode
    hjb_solver: BaseHJBSolver | None = None,  # NEW: Expert Mode
    fp_solver: BaseFPSolver | None = None,    # NEW: Expert Mode
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
    config: MFGSolverConfig | None = None,
) -> SolverResult:
    """
    Solve this MFG problem using one of three modes.

    Mode 1 (Safe): Validated scheme with guaranteed duality
        >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

    Mode 2 (Expert): Manual solver injection with warnings
        >>> result = problem.solve(hjb_solver=my_hjb, fp_solver=my_fp)

    Mode 3 (Auto): Intelligent scheme selection
        >>> result = problem.solve()  # Auto-selects based on geometry

    Args:
        scheme: Numerical scheme (Mode 1). If provided, hjb_solver/fp_solver ignored.
        hjb_solver: Custom HJB solver (Mode 2). Requires fp_solver.
        fp_solver: Custom FP solver (Mode 2). Requires hjb_solver.
        max_iterations: Maximum Picard iterations
        tolerance: Convergence tolerance
        verbose: Print progress information
        config: Solver configuration (scheme-specific parameters)

    Returns:
        SolverResult with solution, convergence history, and metadata

    Raises:
        ValueError: If mode detection fails or invalid parameter combination

    Example:
        >>> from mfg_pde import create_crowd_problem, NumericalScheme
        >>> problem = create_crowd_problem(...)
        >>>
        >>> # Safe Mode (recommended)
        >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
        >>>
        >>> # Expert Mode (advanced)
        >>> hjb = HJBFDMSolver(problem)
        >>> fp = FPFDMSolver(problem)
        >>> result = problem.solve(hjb_solver=hjb, fp_solver=fp)
        >>>
        >>> # Auto Mode (exploratory)
        >>> result = problem.solve()  # Picks FDM for 1D/2D, SL for 3D
    """
    # Mode detection
    if scheme is not None:
        # === MODE 1: SAFE (Validated Scheme) ===
        if verbose:
            print(f"[Safe Mode] Using scheme: {scheme.value}")

        # Create paired solvers via factory
        from mfg_pde.factory import create_paired_solvers
        hjb_solver, fp_solver = create_paired_solvers(scheme, self, config)

    elif hjb_solver is not None and fp_solver is not None:
        # === MODE 2: EXPERT (Manual Injection) ===
        if verbose:
            print(f"[Expert Mode] Using manual solvers")

        # Validate pairing and warn if broken
        self._validate_manual_pairing(hjb_solver, fp_solver, verbose)

    elif hjb_solver is None and fp_solver is None:
        # === MODE 3: AUTO (Intelligent Selection) ===
        scheme = self._auto_select_scheme()
        rationale = self._get_selection_rationale(scheme)

        if verbose:
            print(f"[Auto Mode] Selected {scheme.value}")
            print(f"  Rationale: {rationale}")

        # Create paired solvers via factory
        from mfg_pde.factory import create_paired_solvers
        hjb_solver, fp_solver = create_paired_solvers(scheme, self, config)

    else:
        # Invalid parameter combination
        raise ValueError(
            "Invalid parameter combination.\n"
            "Use one of:\n"
            "  1. scheme=... (Safe Mode)\n"
            "  2. hjb_solver=... + fp_solver=... (Expert Mode)\n"
            "  3. No parameters (Auto Mode)\n"
            "Cannot mix scheme with manual solvers."
        )

    # Create coupling iterator and solve
    from mfg_pde.workflow import FixedPointIterator

    solver = FixedPointIterator(
        problem=self,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        config=config,
    )

    return solver.solve(
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose,
    )
```

**Acceptance Criteria**:
- [ ] New signature implemented
- [ ] Mode detection logic correct
- [ ] All three modes functional
- [ ] Backward compatible (no required params removed)
- [ ] Comprehensive docstring with examples

---

#### 3.3 Implement Manual Pairing Validation

**File**: `mfg_pde/core/mfg_problem.py` (MODIFY)

Add method:

```python
def _validate_manual_pairing(
    self,
    hjb_solver: BaseHJBSolver,
    fp_solver: BaseFPSolver,
    verbose: bool,
) -> None:
    """
    Validate manually-provided solver pairing and warn if broken.

    This is an educational function - it doesn't prevent mismatched solvers,
    but guides users toward correct pairings.
    """
    from mfg_pde.utils.adjoint_validation import (
        check_solver_duality, get_duality_description
    )

    duality = check_solver_duality(hjb_solver, fp_solver)

    if duality == "none":
        # Broken duality - issue warning
        import warnings

        hjb_type = type(hjb_solver).__name__
        fp_type = type(fp_solver).__name__

        warnings.warn(
            f"\n{'=' * 80}\n"
            f"⚠️  SOLVER DUALITY WARNING\n"
            f"{'=' * 80}\n\n"
            f"HJB solver: {hjb_type}\n"
            f"FP solver:  {fp_type}\n"
            f"Duality:    {get_duality_description(duality)}\n\n"
            f"This solver pairing does NOT satisfy HJB-FP duality.\n\n"
            f"Expected consequences:\n"
            f"  • Newton iteration may stagnate around 1e-4 error\n"
            f"  • Increased Picard iterations (2-3x typical count)\n"
            f"  • Potential mass conservation drift over time\n"
            f"  • Solution quality degradation in long-time horizons\n\n"
            f"For validated pairings with guaranteed convergence, use:\n"
            f"  problem.solve(scheme=NumericalScheme.<SCHEME>)\n\n"
            f"Available schemes: {[s.value for s in NumericalScheme]}\n"
            f"{'=' * 80}\n",
            UserWarning,
            stacklevel=3
        )
    elif verbose:
        # Valid pairing - just log
        print(f"  Duality: {get_duality_description(duality)} ✓")
```

**Tests**:
- `tests/unit/core/test_manual_validation.py`
  - Test valid pairs (no warning)
  - Test invalid pairs (warning issued)
  - Test warning message content

**Acceptance Criteria**:
- [ ] Validation implemented
- [ ] Educational warnings for broken pairings
- [ ] Tests verify warnings issued correctly
- [ ] stacklevel=3 correct for user code

---

### Phase 4: Testing and Validation (3-4 days, CRITICAL)

**Goal**: Comprehensive testing to prevent regressions.

#### 4.1 Unit Tests

**New Test Files**:
1. `tests/unit/types/test_schemes.py` - NumericalScheme enum
2. `tests/unit/alg/test_solver_traits.py` - _scheme_family traits
3. `tests/unit/utils/test_adjoint_validation.py` - check_solver_duality()
4. `tests/unit/factory/test_scheme_factory.py` - create_paired_solvers()
5. `tests/unit/core/test_auto_selection.py` - _auto_select_scheme()
6. `tests/unit/core/test_manual_validation.py` - _validate_manual_pairing()

**Test Coverage Goals**:
- Enums: 100%
- Traits: 100% (all solvers checked)
- Validation: 100% (all duality combinations)
- Factory: 100% (all schemes)
- Auto-selection: 100% (all geometry types)

---

#### 4.2 Integration Tests (New)

**File**: `tests/integration/test_three_mode_api.py`

```python
"""Integration tests for three-mode solving API."""

import pytest
from mfg_pde import create_lq_problem, NumericalScheme
from mfg_pde.alg.numerical import HJBFDMSolver, FPFDMSolver


class TestSafeMode:
    """Test Mode 1: Validated scheme pairing."""

    def test_fdm_upwind_1d(self):
        """Test FDM upwind on 1D LQ problem."""
        problem = create_lq_problem(dimension=1, nx=50, T=1.0)
        result = problem.solve(scheme=NumericalScheme.FDM_UPWIND, verbose=False)

        assert result.converged
        assert result.iterations < 50
        assert result.final_error < 1e-6

    def test_sl_linear_2d(self):
        """Test Semi-Lagrangian on 2D LQ problem."""
        problem = create_lq_problem(dimension=2, nx=30, T=1.0)
        result = problem.solve(scheme=NumericalScheme.SL_LINEAR, verbose=False)

        assert result.converged
        assert result.iterations < 100

    def test_gfdm_2d(self):
        """Test GFDM on 2D LQ problem."""
        problem = create_lq_problem(dimension=2, nx=30, T=1.0)
        result = problem.solve(scheme=NumericalScheme.GFDM, verbose=False)

        assert result.converged


class TestExpertMode:
    """Test Mode 2: Manual solver injection."""

    def test_valid_pairing_no_warning(self):
        """Valid pairing (FDM-FDM) should not warn."""
        problem = create_lq_problem(dimension=1, nx=50, T=1.0)

        hjb = HJBFDMSolver(problem, upwind=True)
        fp = FPFDMSolver(problem, upwind=True)

        with pytest.warns(None) as record:
            result = problem.solve(hjb_solver=hjb, fp_solver=fp, verbose=False)

        # No warnings should be issued
        assert len(record) == 0
        assert result.converged

    def test_invalid_pairing_warns(self):
        """Invalid pairing (SL-FDM) should warn."""
        problem = create_lq_problem(dimension=1, nx=50, T=1.0)

        from mfg_pde.alg.numerical import HJBSLSolver
        hjb = HJBSLSolver(problem)
        fp = FPFDMSolver(problem, upwind=True)

        with pytest.warns(UserWarning, match="SOLVER DUALITY WARNING"):
            result = problem.solve(hjb_solver=hjb, fp_solver=fp, verbose=False)

        # Should still solve (just warn)
        assert result.converged or result.iterations == 100


class TestAutoMode:
    """Test Mode 3: Intelligent selection."""

    def test_1d_selects_fdm(self):
        """1D problem should auto-select FDM."""
        problem = create_lq_problem(dimension=1, nx=50, T=1.0)
        result = problem.solve(verbose=False)  # Auto mode

        assert result.converged
        # Verify FDM was selected (check result metadata)
        assert "fdm" in result.metadata.get("scheme", "").lower()

    def test_3d_selects_sl(self):
        """3D problem should auto-select Semi-Lagrangian."""
        problem = create_lq_problem(dimension=3, nx=10, T=1.0)
        result = problem.solve(verbose=False)  # Auto mode

        assert result.converged
        assert "semi" in result.metadata.get("scheme", "").lower()


class TestInvalidCombinations:
    """Test error handling for invalid parameter combinations."""

    def test_scheme_and_manual_solvers_error(self):
        """Cannot mix scheme with manual solvers."""
        problem = create_lq_problem(dimension=1, nx=50, T=1.0)

        hjb = HJBFDMSolver(problem, upwind=True)
        fp = FPFDMSolver(problem, upwind=True)

        with pytest.raises(ValueError, match="Cannot mix scheme with manual solvers"):
            problem.solve(
                scheme=NumericalScheme.FDM_UPWIND,
                hjb_solver=hjb,
                fp_solver=fp,
                verbose=False
            )

    def test_only_hjb_error(self):
        """Expert mode requires both hjb_solver and fp_solver."""
        problem = create_lq_problem(dimension=1, nx=50, T=1.0)

        hjb = HJBFDMSolver(problem, upwind=True)

        with pytest.raises(ValueError, match="Invalid parameter combination"):
            problem.solve(hjb_solver=hjb, verbose=False)
```

**Acceptance Criteria**:
- [ ] All integration tests passing
- [ ] Test coverage ≥ 95% for new code
- [ ] All three modes validated
- [ ] Error cases handled correctly

---

#### 4.3 Convergence Tests (Mathematical Validation)

**File**: `tests/integration/test_adjoint_convergence.py`

```python
"""
Convergence tests to verify adjoint duality preservation.

These tests validate that paired solvers achieve expected convergence rates
and maintain mathematical properties (mass conservation, energy decay).
"""

import numpy as np
from mfg_pde import create_lq_problem, NumericalScheme


def test_fdm_convergence_rate():
    """FDM should achieve O(h^0.5-1) convergence."""
    # Test on 1D LQ with known solution
    errors = []
    grids = [25, 50, 100, 200]

    for nx in grids:
        problem = create_lq_problem(dimension=1, nx=nx, T=1.0)
        result = problem.solve(scheme=NumericalScheme.FDM_UPWIND, verbose=False)

        # Compute error vs analytical solution
        error = compute_lq_error(result, problem)
        errors.append(error)

    # Check convergence rate
    rates = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(2)
    assert np.mean(rates) > 0.5  # At least O(h^0.5)


def test_sl_convergence_rate():
    """Semi-Lagrangian should achieve O(h^2) convergence."""
    errors = []
    grids = [25, 50, 100]

    for nx in grids:
        problem = create_lq_problem(dimension=1, nx=nx, T=1.0)
        result = problem.solve(scheme=NumericalScheme.SL_LINEAR, verbose=False)

        error = compute_lq_error(result, problem)
        errors.append(error)

    # Check convergence rate
    rates = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(2)
    assert np.mean(rates) > 1.5  # Close to O(h^2)


def test_mass_conservation():
    """FP solver should conserve mass at each timestep."""
    problem = create_lq_problem(dimension=1, nx=100, T=1.0)
    result = problem.solve(scheme=NumericalScheme.FDM_UPWIND, verbose=False)

    # Check mass conservation (integral of m over domain = 1)
    dx = problem.domain.dx
    for t in range(problem.Nt):
        mass = np.sum(result.M[:, t]) * dx
        assert np.abs(mass - 1.0) < 1e-6, f"Mass not conserved at t={t}"
```

**Acceptance Criteria**:
- [ ] FDM convergence test passes
- [ ] SL convergence test passes
- [ ] Mass conservation test passes
- [ ] Tests run in CI pipeline

---

#### 4.4 Regression Tests (Existing Tests)

**Goal**: Ensure no existing functionality broken.

**Action Items**:
1. Run full test suite: `pytest tests/ -v`
2. Fix any failures caused by `problem.solve()` refactoring
3. Update tests that hardcoded GFDM+Particle assumption

**Expected Failures**:
- Tests that rely on `problem.solve()` always using GFDM
- Tests that assume specific solver types

**Fix Strategy**:
- Update to use Safe Mode with explicit scheme
- Or update to use Auto Mode and check metadata

**Acceptance Criteria**:
- [ ] All existing tests passing
- [ ] No new test failures introduced
- [ ] Deprecation warnings addressed

---

### Phase 5: Documentation and Migration (2-3 days, USER-FACING)

**Goal**: Update all user-facing documentation and examples.

#### 5.1 Update Examples

**Files to Update** (~15-20 examples):

**Basic Examples**:
1. `examples/basic/lq_game_1d.py`
2. `examples/basic/crowd_motion_2d.py`
3. `examples/basic/obstacle_avoidance.py`

**Pattern**:
```python
# OLD (hardcoded GFDM)
result = problem.solve(max_iterations=100, tolerance=1e-6)

# NEW (explicit scheme)
from mfg_pde.types import NumericalScheme
result = problem.solve(
    scheme=NumericalScheme.FDM_UPWIND,  # Or SL_LINEAR, GFDM
    max_iterations=100,
    tolerance=1e-6
)
```

**Advanced Examples**:
- Keep some examples showing Expert Mode (custom solvers)
- Add comments explaining when to use each mode

**Acceptance Criteria**:
- [ ] All examples updated
- [ ] All examples execute without errors
- [ ] Examples demonstrate all three modes
- [ ] Comments explain mode selection

---

#### 5.2 Deprecation Warnings

**File**: `mfg_pde/factory/solver_factory.py` (MODIFY)

```python
def create_solver(
    problem: MFGProblem,
    method: str = "gfdm",
    **kwargs
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create HJB and FP solvers (DEPRECATED).

    .. deprecated:: 0.17.0
        Use `problem.solve(scheme=...)` instead.
        This function will be removed in v1.0.0.

    Migration:
        >>> # OLD
        >>> hjb, fp = create_solver(problem, method="fdm")
        >>> solver = FixedPointIterator(problem, hjb, fp)
        >>> result = solver.solve()
        >>>
        >>> # NEW
        >>> from mfg_pde.types import NumericalScheme
        >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
    """
    import warnings
    warnings.warn(
        "create_solver() is deprecated since v0.17.0. "
        "Use problem.solve(scheme=...) instead. "
        "This function will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2
    )

    # Old implementation...
    ...
```

**Acceptance Criteria**:
- [ ] Deprecation warnings added to old factory functions
- [ ] Migration examples in docstrings
- [ ] Version timeline clearly stated (v0.17.0 → v1.0.0)

---

#### 5.3 User Guide

**File**: `docs/user/THREE_MODE_SOLVING_GUIDE.md` (NEW)

```markdown
# Three-Mode Solving API

**Quick Start**: Use `problem.solve(scheme=...)` for guaranteed duality.

## The Three Modes

### Mode 1: Safe (Recommended)

**Use when**: You want validated solver pairings with guaranteed duality.

```python
from mfg_pde import create_crowd_problem, NumericalScheme

problem = create_crowd_problem(...)

# Explicit scheme selection
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
```

**Available schemes**:
- `FDM_UPWIND`: First-order upwind (most stable, O(h^0.5-1))
- `FDM_CENTERED`: Second-order centered (smooth problems only)
- `SL_LINEAR`: Semi-Lagrangian linear (O(h^2), better scaling)
- `SL_CUBIC`: Semi-Lagrangian cubic (experimental, Issue #583)
- `GFDM`: Meshfree (complex geometry, obstacles)

### Mode 2: Expert (Advanced)

**Use when**: You need full control (custom solvers, research).

```python
from mfg_pde.alg.numerical import HJBFDMSolver, FPFDMSolver

problem = create_crowd_problem(...)

# Create custom solvers
hjb = HJBFDMSolver(problem, upwind=True, ...)
fp = FPFDMSolver(problem, upwind=True, ...)

# Manual injection (with duality warning if mismatched)
result = problem.solve(hjb_solver=hjb, fp_solver=fp)
```

**⚠️ Warning**: If solvers are not dual, you'll receive an educational warning.

### Mode 3: Auto (Exploratory)

**Use when**: Exploring new problems, don't know best scheme.

```python
problem = create_crowd_problem(...)

# Let MFG_PDE choose based on geometry
result = problem.solve()
```

**Auto-selection logic**:
- 1D/2D structured → FDM (most stable)
- 2D unstructured → GFDM (handles irregular grids)
- 3D → Semi-Lagrangian (better scaling)
- Obstacles → GFDM (only scheme supporting them)

## Which Mode Should I Use?

| Scenario | Recommended Mode |
|:---------|:----------------|
| Production code | Safe Mode |
| Research (standard methods) | Safe Mode |
| Research (custom methods) | Expert Mode |
| Initial exploration | Auto Mode |
| Benchmarking | Safe Mode (explicit control) |

## Mathematical Guarantees

### Type A (Exact Duality)

Schemes: FDM, Semi-Lagrangian

**Guarantee**: L_FP = (L_HJB)^T at matrix level

**Consequences**:
- Exact discrete Nash equilibrium
- Energy conservation by construction
- No artificial mass drift

### Type B (Continuous Duality)

Schemes: GFDM

**Guarantee**: L_FP = (L_HJB)^T + O(h)

**Consequences**:
- Approximate Nash equilibrium (error → 0 as h → 0)
- Requires mass renormalization (automatic)
- Small artificial drift (bounded)

## Migration from Old API

### Old Way (v0.16.x)
```python
# Hardcoded GFDM+Particle (no control)
result = problem.solve()
```

### New Way (v0.17.0+)
```python
# Explicit scheme control
from mfg_pde.types import NumericalScheme
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
```

See `DEPRECATION_MODERNIZATION_GUIDE.md` for complete migration guide.
```

**Acceptance Criteria**:
- [ ] User guide written
- [ ] Examples for all three modes
- [ ] Mathematical guarantees explained
- [ ] Decision matrix provided
- [ ] Migration guide included

---

#### 5.4 API Reference

**File**: `docs/api/core.md` (UPDATE)

Add section:

```markdown
## MFGProblem.solve()

**Signature**:
```python
def solve(
    self,
    scheme: NumericalScheme | None = None,
    hjb_solver: BaseHJBSolver | None = None,
    fp_solver: BaseFPSolver | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
    config: MFGSolverConfig | None = None,
) -> SolverResult
```

**Parameters**:
- `scheme` (NumericalScheme | None): Numerical discretization scheme (Safe Mode)
- `hjb_solver` (BaseHJBSolver | None): Custom HJB solver (Expert Mode)
- `fp_solver` (BaseFPSolver | None): Custom FP solver (Expert Mode)
- ... (other parameters)

**Returns**:
- `SolverResult`: Solution with convergence history

**Examples**: See user guide.
```

**Acceptance Criteria**:
- [ ] API reference updated
- [ ] All parameters documented
- [ ] Return types documented
- [ ] Cross-references to user guide

---

## Implementation Timeline

**Total Estimated Time**: 14-19 days (3-4 weeks)

| Phase | Duration | Risk | Dependencies |
|:------|:---------|:-----|:-------------|
| Phase 1: Infrastructure | 2-3 days | LOW | None |
| Phase 2: Factory | 3-4 days | MEDIUM | Phase 1 |
| Phase 3: Facade | 4-5 days | HIGH | Phase 1, 2 |
| Phase 4: Testing | 3-4 days | CRITICAL | Phase 1, 2, 3 |
| Phase 5: Docs | 2-3 days | USER-FACING | Phase 1, 2, 3, 4 |

**Parallel Work Opportunities**:
- Phase 1.1-1.3 can be done in parallel (3 separate files)
- Phase 2.1-2.2 can be done in parallel (2 separate files)
- Phase 4.1-4.3 can be done in parallel (separate test files)
- Phase 5.1-5.4 can be done in parallel (examples vs docs)

**Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

---

## Risk Mitigation

### High-Risk Areas

1. **problem.solve() Refactoring** (Phase 3)
   - **Risk**: Breaks all existing code
   - **Mitigation**:
     - Keep signature backward compatible (all new params optional)
     - Extensive testing before merge
     - Feature flag for gradual rollout
     - Deprecation period (v0.17.0 → v1.0.0)

2. **Auto-Selection Logic** (Phase 3.1)
   - **Risk**: Picks suboptimal scheme
   - **Mitigation**:
     - Conservative defaults (FDM when unsure)
     - Clear logging of selection rationale
     - Easy override via Safe Mode
     - Community feedback period

3. **Test Coverage** (Phase 4)
   - **Risk**: Missing edge cases
   - **Mitigation**:
     - ≥95% coverage requirement
     - Integration tests for all three modes
     - Convergence tests for mathematical validation
     - Regression tests for all examples

### Medium-Risk Areas

1. **Config Threading** (Phase 2.2)
   - **Risk**: Config parameters not passed correctly
   - **Mitigation**:
     - Unit tests for each scheme factory
     - Verify default configs applied
     - Test custom configs override defaults

2. **Duality Validation** (Phase 2.1)
   - **Risk**: False positives/negatives in duality checking
   - **Mitigation**:
     - Unit tests for all solver combinations
     - Test GENERIC fallback behavior
     - Verify trait inheritance

### Low-Risk Areas

1. **Enum Definitions** (Phase 1.1-1.2)
   - **Risk**: Minimal (just data definitions)
   - **Mitigation**: Standard enum patterns

2. **Trait Annotations** (Phase 1.3)
   - **Risk**: Low (non-breaking addition)
   - **Mitigation**: Smoke tests for all solvers

---

## Acceptance Criteria (Overall)

### Functionality ✅

- [ ] All three modes functional (Safe, Expert, Auto)
- [ ] All Phase 1 schemes supported (FDM, SL, GFDM)
- [ ] Duality validation working correctly
- [ ] Auto-selection logic tested on all geometry types
- [ ] Config threading verified for all schemes

### Testing ✅

- [ ] Unit test coverage ≥ 95% for new code
- [ ] Integration tests for all three modes
- [ ] Convergence tests pass
- [ ] All existing tests still passing
- [ ] No new test failures

### Documentation ✅

- [ ] User guide complete (THREE_MODE_SOLVING_GUIDE.md)
- [ ] API reference updated
- [ ] All examples migrated
- [ ] Migration guide written
- [ ] Deprecation warnings in place

### Code Quality ✅

- [ ] Type hints on all new functions
- [ ] Comprehensive docstrings
- [ ] Follows CONSISTENCY_GUIDE.md
- [ ] No mypy errors
- [ ] Ruff checks pass

### Release Readiness ✅

- [ ] CHANGELOG.md updated
- [ ] Version bumped to v0.17.0
- [ ] Deprecation timeline documented
- [ ] Breaking changes noted
- [ ] GitHub Issue #580 closed with reference to PR

---

## Follow-Up Work (Post-v0.17.0)

**Phase 2 Schemes** (Future):
- FVM schemes (Godunov, Lax-Friedrichs)
- DG schemes (high-order, unstructured)
- Graph-based schemes (finite state spaces)

**Improvements**:
- Solver Registry Pattern (optional, if plugin ecosystem develops)
- Geometry introspection (replace geometry_type workaround)
- Auto-selection heuristics refinement based on user feedback

**Deprecation Completion** (v1.0.0):
- Remove `create_solver()` (deprecated in v0.17.0)
- Remove old factory functions
- Update all documentation to new API only

---

## References

- **Issue #580**: RFC for adjoint-aware solver pairing
- **FACTORY_PATTERN_DESIGN.md**: Complete architectural design
- **adjoint_operators_mfg.md**: Mathematical foundation
- **Issue #578**: Semi-Lagrangian FP solver implementation
- **Issue #543**: Validator pattern (hasattr elimination)

---

**Status**: 📋 **READY FOR IMPLEMENTATION**
**Next Step**: Create feature branch `feature/issue-580-adjoint-pairing`
**Estimated Completion**: 3-4 weeks from start

---

**End of Implementation Plan**
