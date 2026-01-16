# MFG_PDE Factory Pattern Design

**Document**: Factory Pattern Design
**Status**: Active Design Reference (Issue #580 Implementation)
**Date**: 2026-01-16
**Related Issues**: #580 (Adjoint-aware solver pairing)

---

## Executive Summary

MFG_PDE uses a **three-concern factory organization** to separate concerns:
- **Concern 1 (WHAT)**: Application-domain problem configuration
- **Concern 2 (HOW)**: Numerical scheme selection with mathematical guarantees
- **Concern 3 (WHO)**: Solver assembly and iteration control

**Key principle**: `problem.solve()` is the ONLY user-facing entry point. Internal factories are implementation details.

---

## Relationship to 2-Level Principle

**Important**: This design preserves the 2-level user API established in `docs/archive/historical/API_SIMPLIFICATION_PROPOSAL.md` (2025-11-23).

### The 2-Level Principle (Existing Architecture)

> **Quote from API Simplification**: "2-level architecture: Factory vs Expert, no middle ground"

- **Level 1 (Factory)**: Pre-configured, batteries-included solutions
- **Level 2 (Expert)**: Direct control, full flexibility

This principle **rejected intermediate abstractions** like:
- ❌ `MFGProblemBuilder` (fluent API)
- ❌ `create_mfg_problem()` (thin wrapper around `MFGProblem`)
- ❌ Any "convenience" layers between factory and expert

### How This Design Preserves It ✅

**User-Facing API** (2 levels - PRESERVED):
```python
# Level 1: Factory mode (pre-configured)
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

# Level 2: Expert mode (full control)
result = problem.solve(hjb_solver=my_hjb, fp_solver=my_fp)
```

**Users see only TWO entry points** - no intermediate abstractions added!

**Internal Organization** (3 separable concerns - NOT visible to users):
```python
# Concern 1: WHAT to solve (problem configuration)
problem = create_crowd_problem(...)  # Application factory

# Concern 2: HOW to solve (scheme selection)
scheme = NumericalScheme.FDM_UPWIND  # Enum, not factory call

# Concern 3: WHO couples them (solver assembly - INTERNAL)
# User NEVER calls this - hidden behind problem.solve() facade
```

**Key Insight**: The three "concerns" are **internal implementation details** that compose behind the single `problem.solve()` entry point. They are NOT user-visible levels or abstractions.

### Comparison with Other Architecture "Layers"

To avoid confusion with other uses of "layer" terminology in MFG_PDE:

**Boundary Conditions Architecture** (different context):
- Layer 1: BC Specification (`geometry/boundary/conditions.py`)
- Layer 2: BC Application (`geometry/boundary/applicator_*.py`)
- Layer 3: Solver Integration (`alg/numerical/`)

These are **software architecture layers** (data → logic → integration).

**Factory Concerns** (this document):
- Concern 1: Problem Configuration (WHAT to solve)
- Concern 2: Numerical Scheme (HOW to discretize)
- Concern 3: Solver Assembly (WHO couples them)

These are **separable concerns** (orthogonal dimensions), not stacked layers.

**Terminology choice**: "Concern" (separation of concerns) avoids confusion with BC "layers" and clarifies these are orthogonal, composable dimensions rather than sequential abstraction layers.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Three-Concern Separation](#three-concern-separation)
3. [Three-Mode Solving API](#three-mode-solving-api)
4. [Implementation Design](#implementation-design)
5. [Anti-Confusion Strategy](#anti-confusion-strategy)
6. [Migration Guide](#migration-guide)
7. [Examples](#examples)

---

## Motivation

### Problem: Factory Pattern Confusion

Users encounter multiple factory functions and struggle to understand their relationships:

```python
# Confusing landscape (before clarification)
create_lq_problem()          # Concern 1: Problem type
create_crowd_problem()       # Concern 1: Problem type
create_solver()              # Concern 3: Solver assembly
create_fast_solver()         # Deprecated
NumericalScheme enum         # Concern 2: Scheme selection (NEW in #580)
create_paired_solvers()      # Concern 2: Internal (NEW in #580)
```

**User questions:**
- "Which one do I use?"
- "Do I need to combine them?"
- "What's the difference between problem factories and solver factories?"

### Solution: Clear Layer Separation with Single Entry Point

**Principle**: Hide factory complexity behind `problem.solve()` facade.

```python
# Clear workflow (one entry point)
problem = create_crowd_problem(...)  # Concern 1: WHAT to solve
result = problem.solve(scheme=...)   # Layers 2+3: HOW + WHO (internal)
```

---

## Three-Concern Separation

### Concern 1: WHAT to Solve (Problem Configuration)

**Location**: `mfg_pde/factory/problem_factories.py`
**Responsibility**: Configure MFGProblem for specific applications
**Output**: `MFGProblem` instance

#### Purpose
- Define physics/math: Hamiltonian, running costs, potentials
- Set domain: Geometry, boundary conditions
- Specify conditions: Initial density, terminal cost

#### Examples

```python
from mfg_pde.factory import create_lq_problem, create_crowd_problem

# Linear-Quadratic MFG
problem_lq = create_lq_problem(
    geometry=domain,
    terminal_cost=lambda x: |x - target|^2,
    running_cost_control=1.0,
    running_cost_congestion=0.5
)

# Crowd dynamics
problem_crowd = create_crowd_problem(
    geometry=room_geometry,
    target_location=exit_door,
    initial_density=uniform_crowd,
    congestion_sensitivity=2.0
)
```

#### User Mental Model
> "I'm solving a crowd evacuation problem"

#### Naming Convention
- Pattern: `create_<APPLICATION>_problem()`
- Examples: `create_lq_problem`, `create_traffic_problem`, `create_network_problem`

---

### Concern 2: HOW to Solve (Numerical Scheme Selection)

**Location**: `mfg_pde/factory/scheme_factory.py` (NEW in Issue #580)
**Responsibility**: Select discretization with guaranteed mathematical properties
**Output**: `(hjb_solver, fp_solver)` tuple with validated duality

#### Purpose
- Ensure discrete duality (Type A: exact, Type B: asymptotic)
- Auto-inject middleware (renormalization, mass conservation)
- Validate solver compatibility with problem geometry

#### Mathematical Foundation

See `docs/theory/adjoint_operators_mfg.md` for complete theory.

**Type A Schemes (Exact Discrete Transpose)**:
```
L_FP = L_HJB^T  (exactly at finite h)

Property: Quadratic Newton convergence to machine precision
```

**Type B Schemes (Asymptotic Adjoint)**:
```
L_FP = L_HJB^T + O(h)  (only as h → 0)

Property: May stagnate around 1e-4 without renormalization
```

#### Implementation

```python
# NEW: mfg_pde/factory/scheme_factory.py

from enum import Enum

class NumericalScheme(Enum):
    """Validated schemes with guaranteed duality."""

    # Type A: Exact discrete adjoint
    FDM_UPWIND = "fdm_upwind"         # O(h) accuracy, stable
    SL_LINEAR = "sl_linear"           # O(h²) accuracy, flexible
    FVM_GODUNOV = "fvm_godunov"       # Conservation, shocks

    # Type B: Asymptotic adjoint
    GFDM = "gfdm"                     # Complex geometry
    PINN = "pinn"                     # Very high dimension


def create_paired_solvers(
    scheme: NumericalScheme,
    problem: MFGProblem,
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create adjoint-paired solvers with guaranteed duality.

    This is an INTERNAL factory called by problem.solve().
    Users should NOT call this directly.

    Args:
        scheme: Validated numerical scheme
        problem: MFG problem to solve

    Returns:
        (hjb_solver, fp_solver) with guaranteed duality

    Type A pairs:
        - Both solvers from SAME numerical method
        - L_FP = L_HJB^T exactly
        - No middleware needed

    Type B pairs:
        - Both solvers from SAME primal method
        - Complementary parameters (e.g., upwind vs downwind)
        - AUTO-INJECT renormalization wrapper for mass conservation
    """
    from mfg_pde.alg.numerical.hjb_solvers import (
        HJBFDMSolver, HJBSemiLagrangianSolver, HJBGFDMSolver
    )
    from mfg_pde.alg.numerical.fp_solvers import (
        FPFDMSolver, FPSplattingSolver, FPGFDMSolver, RenormalizationWrapper
    )

    if scheme == NumericalScheme.FDM_UPWIND:
        # Type A: Exact discrete transpose
        hjb = HJBFDMSolver(problem, scheme="upwind")
        fp = FPFDMSolver(problem, scheme="upwind")
        return hjb, fp

    elif scheme == NumericalScheme.SL_LINEAR:
        # Type A: Exact discrete transpose via splatting
        hjb = HJBSemiLagrangianSolver(problem, interpolation="linear")
        fp = FPSplattingSolver(problem, interpolation="linear")
        return hjb, fp

    elif scheme == NumericalScheme.GFDM:
        # Type B: Asymptotic adjoint
        hjb = HJBGFDMSolver(problem, upwind="exponential_downwind")
        fp = FPGFDMSolver(problem, upwind="exponential_upwind")

        # AUTO-INJECT mass conservation middleware (Type B requirement)
        fp = RenormalizationWrapper(fp)

        return hjb, fp

    else:
        raise NotImplementedError(f"Scheme {scheme} not yet implemented")
```

#### User Mental Model
> "I want to use finite differences with guaranteed convergence"

#### Naming Convention
- Enum: `NumericalScheme.<SCHEME_NAME>`
- Factory: `create_paired_solvers(scheme, problem)` (internal only)
- **NO** application-specific variants (no `create_crowd_fdm_solver`)

---

### Concern 3: WHO Couples Them (Solver Assembly)

**Location**: `mfg_pde/factory/solver_factory.py`
**Responsibility**: Wire HJB + FP into fixed-point iterator
**Output**: `FixedPointIterator` instance

#### Purpose
- Assemble HJB and FP solvers into coupled iterator
- Configure iteration control (max_iterations, tolerance)
- Set up monitoring (progress bars, convergence plots)

#### Implementation (Deprecated)

```python
# mfg_pde/factory/solver_factory.py

def create_solver(
    problem: MFGProblem,
    hjb_solver: BaseHJBSolver,
    fp_solver: BaseFPSolver,
    config: MFGSolverConfig | None = None,
) -> FixedPointIterator:
    """
    Create MFG solver (DEPRECATED - use problem.solve() instead).

    This factory function is deprecated in favor of problem.solve(),
    which provides validation, warnings, and tier selection.

    OLD API (discouraged):
        solver = create_solver(problem, hjb, fp)
        result = solver.solve()

    NEW API (recommended):
        result = problem.solve(hjb_solver=hjb, fp_solver=fp)

    Rationale: problem.solve() checks duality and emits warnings
    for non-adjoint pairs.
    """
    import warnings
    warnings.warn(
        "create_solver() is deprecated. Use problem.solve() instead.\n"
        "Migration: problem.solve(hjb_solver=hjb, fp_solver=fp)",
        DeprecationWarning,
        stacklevel=2
    )

    from mfg_pde.alg.numerical.coupling import FixedPointIterator
    return FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        config=config or MFGSolverConfig(),
    )
```

#### User Mental Model
> "Run the coupled system until convergence"

---

## Three-Mode Solving API

### Facade Pattern: `problem.solve()`

**Key Design Decision**: `problem.solve()` is the ONLY user-facing entry point.

```python
# In mfg_pde/core/mfg_problem.py

def solve(
    self,
    # Mode 1: Safe Mode (validated scheme)
    scheme: NumericalScheme | None = None,

    # Mode 2: Expert Mode (manual injection)
    hjb_solver: BaseHJBSolver | None = None,
    fp_solver: BaseFPSolver | None = None,

    # Common parameters
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
    config: MFGSolverConfig | None = None,
) -> SolverResult:
    """
    Solve MFG problem using three-mode API.

    Three modes available:

    Mode 1 (Safe): Validated scheme pairing
        result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

    Mode 2 (Expert): Manual solver injection with warnings
        result = problem.solve(hjb_solver=..., fp_solver=...)

    Mode 3 (Auto): Problem-aware automatic selection
        result = problem.solve()

    Args:
        scheme: Numerical scheme for validated pairing (Mode 1)
        hjb_solver: Custom HJB solver (Mode 2 - expert mode)
        fp_solver: Custom FP solver (Mode 2 - expert mode)
        max_iterations: Maximum Picard iterations
        tolerance: Convergence tolerance
        verbose: Show progress information
        config: Advanced solver configuration

    Returns:
        SolverResult with U, M, convergence info

    Raises:
        ValueError: If parameters are ambiguous or incomplete

    Examples:
        # Mode 1: Safe mode with validated scheme
        result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

        # Mode 2: Expert mode with custom solvers
        hjb = HJBSemiLagrangianSolver(problem)
        fp = FPFDMSolver(problem)  # Non-adjoint pairing
        result = problem.solve(hjb_solver=hjb, fp_solver=fp)
        # Warning: Using non-adjoint pair - expect convergence issues

        # Mode 3: Auto mode
        result = problem.solve()  # Auto-selects FDM for 1D/2D grids
    """

    # === MODE DETECTION ===
    if scheme is not None:
        # Mode 1: Safe Mode - validated scheme pairing
        if hjb_solver is not None or fp_solver is not None:
            raise ValueError(
                "Ambiguous solve mode: specify EITHER 'scheme' (Mode 1) "
                "OR ('hjb_solver', 'fp_solver') (Mode 2), not both.\n"
                "See docs/architecture/FACTORY_PATTERN_DESIGN.md"
            )

        # Internal call to Concern 2 factory
        hjb_solver, fp_solver = self._create_paired_solvers(scheme)

    elif hjb_solver is not None and fp_solver is not None:
        # Mode 2: Expert Mode - validate and warn about non-adjoint pairs
        self._validate_manual_pairing(hjb_solver, fp_solver)

    elif hjb_solver is not None or fp_solver is not None:
        # Incomplete manual injection
        raise ValueError(
            "Incomplete manual injection: both hjb_solver AND fp_solver required.\n"
            "Provide both, or use scheme parameter for safe pairing."
        )

    else:
        # Mode 3: Auto Mode - problem-aware selection
        scheme = self._auto_select_scheme()
        hjb_solver, fp_solver = self._create_paired_solvers(scheme)

        if verbose:
            print(f"Auto-selected scheme: {scheme.value}")
            print(f"  Reason: {self._get_selection_rationale(scheme)}")

    # === COMMON PATH: Assemble and solve ===
    # Internal call to Concern 3 factory
    solver = self._create_solver_iterator(hjb_solver, fp_solver, config)

    return solver.solve(
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose
    )


# === INTERNAL LAYER 2 FACADE ===
def _create_paired_solvers(
    self,
    scheme: NumericalScheme
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """Internal: Call Layer 2 factory with validated scheme."""
    from mfg_pde.factory.scheme_factory import create_paired_solvers
    return create_paired_solvers(scheme, self)


# === INTERNAL LAYER 3 FACADE ===
def _create_solver_iterator(
    self,
    hjb_solver: BaseHJBSolver,
    fp_solver: BaseFPSolver,
    config: MFGSolverConfig | None,
) -> FixedPointIterator:
    """Internal: Call Layer 3 factory to assemble iterator."""
    from mfg_pde.alg.numerical.coupling import FixedPointIterator

    return FixedPointIterator(
        problem=self,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        config=config or MFGSolverConfig(),
    )


# === EXPERT MODE VALIDATION ===
def _validate_manual_pairing(
    self,
    hjb_solver: BaseHJBSolver,
    fp_solver: BaseFPSolver
) -> None:
    """Validate manually injected solvers and emit warnings."""
    from mfg_pde.utils.adjoint_validation import check_solver_duality

    duality_type = check_solver_duality(hjb_solver, fp_solver)

    if duality_type == "none":
        import warnings
        hjb_type = type(hjb_solver).__name__
        fp_type = type(fp_solver).__name__

        warnings.warn(
            f"\n{'=' * 80}\n"
            f"⚠️  EXPERT MODE WARNING - NON-ADJOINT SOLVER PAIRING\n"
            f"{'=' * 80}\n"
            f"You are using manually injected solvers:\n"
            f"  HJB: {hjb_type}\n"
            f"  FP:  {fp_type}\n\n"
            f"These solvers are NOT discrete adjoints of each other.\n\n"
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


# === AUTO-SELECTION LOGIC ===
def _auto_select_scheme(self) -> NumericalScheme:
    """
    Auto-select numerical scheme based on problem characteristics.

    Selection criteria:
    - Dimension: 1D/2D → FDM, 3D+ → SL or GFDM
    - Geometry: Regular grid → FDM/SL, Complex → GFDM
    - Obstacles: Present → GFDM (only scheme that handles them)

    Returns:
        Recommended NumericalScheme

    Conservative philosophy: When in doubt, choose FDM (most stable).
    """
    from mfg_pde.types import NumericalScheme

    # Priority 1: Obstacles force GFDM
    if self.has_obstacles:
        return NumericalScheme.GFDM

    # Priority 2: Complex geometry → GFDM
    if self.domain_type not in ["grid"]:
        return NumericalScheme.GFDM

    # Priority 3: Dimension-based selection
    if self.dimension <= 2:
        # Low dimension, regular grid → FDM (most stable)
        return NumericalScheme.FDM_UPWIND
    elif self.dimension == 3:
        # 3D → Semi-Lagrangian (better scaling than FDM)
        return NumericalScheme.SL_LINEAR
    else:
        # High dimension → GFDM (curse of dimensionality)
        return NumericalScheme.GFDM


def _get_selection_rationale(self, scheme: NumericalScheme) -> str:
    """Get human-readable rationale for auto-selection."""
    if scheme == NumericalScheme.FDM_UPWIND:
        return f"{self.dimension}D regular grid, no obstacles"
    elif scheme == NumericalScheme.SL_LINEAR:
        return f"3D problem, Semi-Lagrangian scales better"
    elif scheme == NumericalScheme.GFDM:
        if self.has_obstacles:
            return "Complex geometry with obstacles"
        else:
            return f"{self.dimension}D, GFDM for flexibility"
    else:
        return "Unknown selection criteria"
```

---

## Implementation Design

### Critical Implementation Considerations

Before implementing Issue #580, address these technical risks identified in architectural audit:

#### 1. Configuration Leakage Risk ⚠️

**Problem**: Current `create_paired_solvers()` signature doesn't pass `config`:

```python
# ❌ INCOMPLETE - Missing config parameter
def create_paired_solvers(
    scheme: NumericalScheme,
    problem: MFGProblem,
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    ...
```

**Risk**: Scheme-specific parameters (GFDM kernel bandwidth, stencil size, etc.) can't be passed through. Solvers fall back to hardcoded defaults, losing user control.

**Solution**: Add `config` parameter with proper threading:

```python
# ✅ CORRECT - Config threaded through factory chain
def create_paired_solvers(
    scheme: NumericalScheme,
    problem: MFGProblem,
    config: MFGSolverConfig | None = None,
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create adjoint-paired solvers with configuration.

    Args:
        scheme: Numerical scheme for pairing
        problem: MFG problem to solve
        config: Solver configuration (scheme-specific params, iteration control)
    """
    if config is None:
        config = MFGSolverConfig()

    if scheme == NumericalScheme.GFDM:
        hjb = HJBGFDMSolver(
            problem,
            upwind="exponential_downwind",
            kernel_bandwidth=config.hjb.gfdm.kernel_bandwidth,  # ← Config used
            stencil_size=config.hjb.gfdm.stencil_size,
        )
        fp = FPGFDMSolver(
            problem,
            upwind="exponential_upwind",
            kernel_bandwidth=config.fp.gfdm.kernel_bandwidth,
            stencil_size=config.fp.gfdm.stencil_size,
        )
        fp = RenormalizationWrapper(fp)
        return hjb, fp

    # ... other schemes
```

**Update `problem.solve()` facade**:
```python
def solve(self, scheme=None, hjb_solver=None, fp_solver=None, config=None, ...):
    # ...
    if scheme is not None:
        # Thread config through to factory
        hjb_solver, fp_solver = self._create_paired_solvers(scheme, config)
```

---

#### 2. Type Checking Brittleness Risk ⚠️

**Problem**: `check_solver_duality()` uses string matching:

```python
# ❌ FRAGILE - Breaks on refactoring
exact_pairs = {
    ("HJBFDMSolver", "FPFDMSolver"),  # ← String matching class names
    ...
}
if (hjb_type, fp_type) in exact_pairs:
    return "exact"
```

**Risk**: Silent failure if class names change during refactoring (e.g., `HJBFDMSolver` → `FDMSolverHJB`). Type checking becomes obsolete without any indication.

**Solution**: Trait-based validation (extensible and refactoring-safe):

```python
# NEW: mfg_pde/alg/numerical/base_solver.py

from enum import Enum

class SchemeFamily(Enum):
    """Numerical scheme families for duality checking."""
    FDM = "fdm"
    SL = "semi_lagrangian"
    FVM = "fvm"
    GFDM = "gfdm"
    PINN = "pinn"
    GENERIC = "generic"  # Unknown/custom


class BaseSolver:
    """Base class for all solvers with trait annotations."""

    # Trait: Declare scheme family (subclasses override)
    _scheme_family: SchemeFamily = SchemeFamily.GENERIC


# Example: FDM solvers declare family
class HJBFDMSolver(BaseHJBSolver):
    _scheme_family = SchemeFamily.FDM

class FPFDMSolver(BaseFPSolver):
    _scheme_family = SchemeFamily.FDM


# ✅ ROBUST - Survives refactoring
def check_solver_duality(hjb_solver, fp_solver) -> AdjointType:
    """Check duality using scheme family traits."""

    hjb_family = getattr(hjb_solver, '_scheme_family', SchemeFamily.GENERIC)
    fp_family = getattr(fp_solver, '_scheme_family', SchemeFamily.GENERIC)

    # Type A: Same family (except GENERIC)
    if (hjb_family == fp_family and
        hjb_family != SchemeFamily.GENERIC):

        # Special handling for Type B schemes
        if hjb_family == SchemeFamily.GFDM:
            return "asymptotic"

        # Type A schemes
        return "exact"

    # No duality
    return "none"
```

**Benefits**:
- ✅ Refactoring-safe: Class names can change freely
- ✅ Extensible: New schemes just set `_scheme_family`
- ✅ Self-documenting: Trait visible in class definition
- ✅ Plugin-friendly: Custom solvers declare their family

---

#### 3. Auto-Selection Heuristic Edge Cases ⚠️

**Problem**: `_auto_select_scheme()` uses dimension-only heuristic:

```python
# ❌ INCOMPLETE - Ignores geometry complexity
if self.dimension <= 2:
    return NumericalScheme.FDM_UPWIND  # Fails on non-convex polygons!
```

**Risk**: FDM auto-selected for 2D non-convex/unstructured domains where it produces poor results (staircase approximation, O(h) errors).

**Solution**: Check geometry characteristics:

```python
def _auto_select_scheme(self) -> NumericalScheme:
    """
    Auto-select scheme based on problem characteristics.

    Selection hierarchy (priority order):
    1. Obstacles → GFDM (only scheme that handles them)
    2. Complex geometry → GFDM (avoid FDM staircase errors)
    3. Low dimension (1D/2D) + regular grid → FDM (most stable)
    4. Medium dimension (3D) → SL (better scaling)
    5. High dimension (4D+) → Error (explicit choice required)
    """
    from mfg_pde.types import NumericalScheme

    # Priority 1: Obstacles force GFDM
    if self.has_obstacles:
        return NumericalScheme.GFDM

    # Priority 2: Geometry complexity check
    if self.dimension <= 2:
        # Check if geometry is FDM-compatible (structured, rectangular)
        if hasattr(self.geometry, 'is_structured'):
            if not self.geometry.is_structured():
                # Unstructured mesh → GFDM
                return NumericalScheme.GFDM

        # Check for non-convexity (FDM struggles with staircase boundaries)
        if hasattr(self.geometry, 'is_convex'):
            if not self.geometry.is_convex():
                # Non-convex polygon → GFDM for smooth boundary treatment
                return NumericalScheme.GFDM

        # Regular grid, no complexity issues → FDM (most stable)
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
        return f"{self.dimension}D regular grid, structured domain"
    elif scheme == NumericalScheme.SL_LINEAR:
        return f"3D problem, Semi-Lagrangian scales better"
    elif scheme == NumericalScheme.GFDM:
        if self.has_obstacles:
            return "Complex geometry with obstacles"
        elif hasattr(self.geometry, 'is_structured') and not self.geometry.is_structured():
            return "Unstructured mesh"
        elif hasattr(self.geometry, 'is_convex') and not self.geometry.is_convex():
            return "Non-convex domain (avoid FDM staircase errors)"
        else:
            return "Complex geometry requiring flexible discretization"
    else:
        return "Unknown selection criteria"
```

**Fallback strategy**: If geometry introspection unavailable (old Domain classes), be conservative:

```python
# Conservative fallback when geometry type unknown
if not hasattr(self.geometry, 'is_structured'):
    # Unknown geometry type → Prefer GFDM for safety
    if self.dimension <= 3:
        return NumericalScheme.GFDM
```

---

### Directory Structure

```
mfg_pde/
├── factory/
│   ├── __init__.py                # Public API exports
│   ├── problem_factories.py       # Concern 1: WHAT (applications)
│   ├── scheme_factory.py          # Concern 2: HOW (NEW in #580)
│   └── solver_factory.py          # Concern 3: WHO (deprecated)
├── core/
│   └── mfg_problem.py             # Facade: problem.solve()
├── utils/
│   └── adjoint_validation.py     # NEW: Duality checker
└── types/
    └── schemes.py                 # NEW: NumericalScheme enum

docs/
├── architecture/
│   ├── FACTORY_PATTERN_DESIGN.md  # This document
│   └── THREE_TIER_SOLVING.md      # User-facing guide
└── theory/
    └── adjoint_operators_mfg.md   # Mathematical foundation
```

### File: `mfg_pde/utils/adjoint_validation.py` (NEW)

```python
"""
Adjoint Validation Utilities

Provides runtime validation of solver duality relationships.
See docs/theory/adjoint_operators_mfg.md for mathematical foundation.
"""

from typing import Literal

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
        - SL-Splatting: Linear interpolation
        - FVM-FVM: Godunov flux

    Type B Pairs (Asymptotic):
        - GFDM-GFDM: Complementary upwind parameters
        - PINN-PINN: Shared network architecture

    Non-Dual Pairs:
        - Any mixed combination (e.g., SL + FDM)

    Example:
        >>> hjb = HJBFDMSolver(problem)
        >>> fp = FPFDMSolver(problem)
        >>> check_solver_duality(hjb, fp)
        'exact'

        >>> hjb = HJBSemiLagrangianSolver(problem)
        >>> fp = FPFDMSolver(problem)
        >>> check_solver_duality(hjb, fp)
        'none'  # Mixed scheme - no duality!
    """
    hjb_type = type(hjb_solver).__name__
    fp_type = type(fp_solver).__name__

    # Type A: Exact discrete duality
    exact_pairs = {
        ("HJBFDMSolver", "FPFDMSolver"),
        ("HJBSemiLagrangianSolver", "FPSplattingSolver"),
        ("HJBFVMSolver", "FPFVMSolver"),
    }

    if (hjb_type, fp_type) in exact_pairs:
        return "exact"

    # Type B: Asymptotic duality (same primal method)
    if hjb_type == "HJBGFDMSolver" and fp_type == "FPGFDMSolver":
        # Check if parameters are complementary
        # (In practice, factory always sets these correctly)
        return "asymptotic"

    if "PINN" in hjb_type and "PINN" in fp_type:
        # Assume locked parameters (factory enforces this)
        return "asymptotic"

    # No duality
    return "none"
```

---

## Anti-Confusion Strategy

### 1. Naming Convention

| Layer | Function Pattern | Purpose |
|:------|:----------------|:--------|
| Layer 1 | `create_<APP>_problem()` | Problem type (LQ, crowd, traffic) |
| Layer 2 | `NumericalScheme.<SCHEME>` | Scheme selection (enum, not factory call) |
| Layer 2 | `create_paired_solvers()` | **INTERNAL** - not for user |
| Layer 3 | `create_solver()` | **DEPRECATED** - use `problem.solve()` |

**Anti-patterns to avoid**:
- ❌ `create_crowd_fdm_solver()` - Mixes Layer 1 and Layer 2
- ❌ `create_fast_lq_solver()` - Mixes Layer 1 and optimization strategy
- ❌ Direct factory calls - Always use `problem.solve()` facade

### 2. Single Entry Point

**Public API**: `problem.solve()` only

```python
# ✅ CORRECT: Single entry point
problem = create_crowd_problem(...)
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

# ❌ WRONG: Direct factory calls (confusing)
problem = create_crowd_problem(...)
hjb, fp = create_paired_solvers(NumericalScheme.FDM_UPWIND, problem)
solver = create_solver(problem, hjb, fp)
result = solver.solve()
```

### 3. Clear Error Messages

```python
# User attempts wrong pattern
>>> from mfg_pde.factory import create_crowd_problem
>>> create_crowd_problem(scheme=NumericalScheme.FDM_UPWIND)

ValueError:
    create_crowd_problem() configures WHAT to solve (problem type).
    To specify HOW to solve (numerical scheme), use:

        problem = create_crowd_problem(
            geometry=...,
            target_location=...,
            initial_density=...
        )
        result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

    See docs/architecture/FACTORY_PATTERN_DESIGN.md for details.
```

### 4. Documentation Hierarchy

**User-facing docs** (`docs/user/`):
- Show ONLY `problem.solve()` API
- Mention factory functions ONLY for problem configuration (Layer 1)
- Mark Layer 2/3 factories as "Internal - do not use directly"

**Architecture docs** (`docs/architecture/`):
- Explain three-layer design for contributors
- Document factory implementations
- Provide migration guides for deprecated patterns

**API reference** (auto-generated):
- Mark `create_solver()` with `@deprecated` decorator
- Mark `create_paired_solvers()` with `@internal` tag
- Type hints guide users to correct usage

---

## Migration Guide

### From Direct Factory Calls → problem.solve()

**Before (OLD API - confusing)**:
```python
from mfg_pde import MFGProblem, create_solver
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver

problem = MFGProblem(...)
hjb = HJBFDMSolver(problem, scheme="upwind")
fp = FPFDMSolver(problem, scheme="upwind")
solver = create_solver(problem, hjb, fp)
result = solver.solve()
```

**After (NEW API - clear)**:
```python
from mfg_pde import MFGProblem
from mfg_pde.types import NumericalScheme

problem = MFGProblem(...)
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
```

### From Hardcoded solve() → Scheme Selection

**Before (Issue #580 problem)**:
```python
problem = MFGProblem(...)
result = problem.solve()  # ← Hardcoded GFDM+Particle, no choice!
```

**After (Issue #580 solution)**:
```python
problem = MFGProblem(...)

# Option 1: Explicit scheme
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

# Option 2: Auto-selection (new default)
result = problem.solve()  # ← Now auto-selects based on problem
```

### Deprecation Timeline

| Version | Status | Action |
|:--------|:-------|:-------|
| v0.16.x | Current | Factory functions work with DeprecationWarning |
| v0.17.0 | Transition | `create_solver()` raises warning by default |
| v1.0.0 | Breaking | `create_solver()` removed, only `problem.solve()` |

---

## Examples

### Example 1: Crowd Evacuation (Mode 1 - Safe Mode)

```python
from mfg_pde.factory import create_crowd_problem
from mfg_pde.types import NumericalScheme
from mfg_pde.geometry import Hyperrectangle

# Concern 1: Configure problem (WHAT)
room = Hyperrectangle(bounds=[[0, 10], [0, 10]])
exit_door = [10, 5]

problem = create_crowd_problem(
    geometry=room,
    target_location=exit_door,
    initial_density=lambda x: uniform_crowd(x),
    congestion_sensitivity=2.0,
    time_horizon=10.0
)

# Concerns 2+3: Solve with validated scheme (HOW + WHO - internal)
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final error: {result.error:.2e}")
```

### Example 2: LQ Problem with Auto-Selection (Mode 3)

```python
from mfg_pde.factory import create_lq_problem
from mfg_pde.geometry import TensorProductGrid

# Concern 1: Configure LQ problem
domain = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[51, 51])

problem = create_lq_problem(
    geometry=domain,
    terminal_cost=lambda x: |x - target|^2,
    initial_density=lambda x: gaussian(x, center, sigma),
    running_cost_control=1.0,
    running_cost_congestion=0.5
)

# Concerns 2+3: Auto-select scheme based on 2D regular grid
result = problem.solve(verbose=True)
# Output: Auto-selected scheme: fdm_upwind
#         Reason: 2D regular grid, no obstacles
```

### Example 3: Research Experiment (Mode 2 - Expert Mode)

```python
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver

# Research question: "What happens if I mix SL + FDM?"

problem = MFGProblem(...)

# Concern 2: Manual solver injection (deliberate mismatch)
hjb = HJBSemiLagrangianSolver(problem, interpolation="linear")
fp = FPFDMSolver(problem, scheme="upwind")  # ← Non-adjoint!

# Concern 3: Assemble and solve
result = problem.solve(hjb_solver=hjb, fp_solver=fp)

# WARNING emitted:
# ⚠️  EXPERT MODE WARNING - NON-ADJOINT SOLVER PAIRING
# You are using manually injected solvers:
#   HJB: HJBSemiLagrangianSolver
#   FP:  FPFDMSolver
#
# These solvers are NOT discrete adjoints of each other.
# Expected consequences:
#   • Newton iteration may stagnate around 1e-4 error
#   • Increased Picard iterations (2-3x typical count)
#   ...
```

### Example 4: Plugin Mode (Custom Solver)

```python
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver

# Implement custom research algorithm
class MyNovelHJBSolver(BaseHJBSolver):
    """My experimental HJB solver."""

    def solve_timestep(self, U_next, m_current, timestep_idx):
        # Novel algorithm here
        U_current = my_novel_method(U_next, m_current)
        return U_current

    def solve_system(self, U, M):
        # Backward time stepping
        for t in reversed(range(self.problem.Nt)):
            U[t] = self.solve_timestep(U[t+1], M[t], t)
        return U

# Use custom solver
problem = MFGProblem(...)
my_hjb = MyNovelHJBSolver(problem)
fp = FPFDMSolver(problem)

result = problem.solve(hjb_solver=my_hjb, fp_solver=fp)
# WARNING: Non-adjoint pairing detected
```

---

## Related Documentation

- `docs/theory/adjoint_operators_mfg.md` - Mathematical foundation of discrete duality
- `docs/user/SOLVING_GUIDE.md` - User-facing solving tutorial
- `docs/development/CONSISTENCY_GUIDE.md` - Code style and patterns
- GitHub Issue #580 - Adjoint-aware solver pairing RFC

---

## Architectural Audit Results

**Status**: ✅ **Approved (High Quality)** - Ready for implementation

**Audit Date**: 2026-01-16
**Auditor**: Expert Review (MFG Theory + Software Architecture)

### Overall Assessment

> "这份文档写得非常出色。它不仅完美地贯彻了我们之前讨论的"分层架构"理念，还巧妙地解决了一个非常棘手的矛盾：如何在保留"2-Level API"简单性的同时，引入"3-Concern"的内部复杂度。"

**Rating**: Approved as blueprint for Issue #580 implementation

### Key Strengths (Highlights)

1. **Separation of Concerns** ✅
   - WHAT/HOW/WHO decomposition avoids "combination explosion"
   - No `create_crowd_fdm_solver()` anti-patterns
   - Clear boundary: Users define problems, mathematicians define schemes

2. **2-Level Principle Preservation** ✅
   - Facade pattern (`problem.solve()`) hides internal complexity
   - Users see only Factory vs Expert modes
   - Concern 2 (scheme pairing) correctly hidden as internal detail

3. **Educational Error Messages** ✅
   - `ValueError` designed for teaching, not just reporting
   - Guides users to correct API patterns
   - Hallmark of mature open-source projects

### Technical Risks Addressed

All three risks identified in audit have been incorporated into "Critical Implementation Considerations":

1. **Configuration Leakage** → Addressed with `config` parameter threading
2. **Type Checking Brittleness** → Addressed with `SchemeFamily` trait system
3. **Auto-Selection Edge Cases** → Addressed with geometry introspection heuristics

### Implementation Sequence (Recommended)

Follow this order for Issue #580 implementation:

**Step 1: Infrastructure** (Foundation)
- Define `NumericalScheme` enum (`mfg_pde/types/schemes.py`)
- Add `SchemeFamily` enum (`mfg_pde/alg/numerical/base_solver.py`)
- Add `_scheme_family` trait to `BaseSolver` and all solver subclasses

**Step 2: Validation Logic** (Core)
- Implement `check_solver_duality()` using trait-based validation (`mfg_pde/utils/adjoint_validation.py`)
- Implement `create_paired_solvers()` with config threading (`mfg_pde/factory/scheme_factory.py`)

**Step 3: Facade Integration** (User API)
- Refactor `problem.solve()` to accept `scheme` parameter
- Implement mode detection (Safe/Expert/Auto)
- Implement `_auto_select_scheme()` with geometry checks
- Implement `_validate_manual_pairing()` with warnings

**Step 4: Deprecation** (Cleanup)
- Mark `create_solver()` as deprecated with `DeprecationWarning`
- Update docstrings with migration guides
- Add tests for all three modes

**Step 5: Documentation** (User-Facing)
- User guide showing only `problem.solve()` API
- Mark internal factories as "do not use directly"
- Add examples for all three modes

### Pre-Implementation Checklist

Before coding, ensure:
- [ ] All solver classes have `_scheme_family` trait
- [ ] `MFGSolverConfig` supports scheme-specific parameters
- [ ] Geometry classes expose `is_structured()` and `is_convex()` methods
- [ ] Test infrastructure can validate adjoint properties
- [ ] Deprecation strategy agreed upon (v0.17.0 → v1.0.0)

---

## Change History

| Date | Version | Changes |
|:-----|:--------|:--------|
| 2026-01-16 | 1.0 | Initial design document for Issue #580 implementation |
| 2026-01-16 | 1.1 | Terminology fix: Layer→Concern, Tier→Mode; Added 2-Level Principle section |
| 2026-01-16 | 1.2 | Added Critical Implementation Considerations from architectural audit |

---

**End of Document**
