# MFG_PDE API Style Guide

**Version**: 1.0
**Date**: 2025-11-11
**Related**: Issue #277 (API Consistency Audit)

This document establishes API design standards for MFG_PDE to ensure consistency, maintainability, and clarity.

---

## Table of Contents

1. [Naming Conventions](#naming-conventions)
2. [Parameter Design](#parameter-design)
3. [Return Values](#return-values)
4. [Deprecation Strategy](#deprecation-strategy)
5. [Type Hints](#type-hints)
6. [Examples](#examples)

---

## Naming Conventions

### Grid and Discretization

#### ✅ DO: Use Descriptive Names

```python
# Grid resolution
num_points: int | list[int]  # Number of grid points per dimension
num_points[0]  # Points in dimension 0
num_points[1]  # Points in dimension 1

# Grid spacing
spacing: float | list[float]  # Grid spacing per dimension
spacing[0]  # Spacing in dimension 0
spacing[1]  # Spacing in dimension 1

# Time discretization
num_time_steps: int  # Number of time steps
time_step_size: float  # dt = T / num_time_steps
```

#### ❌ DON'T: Use Single-Letter Abbreviated Names

```python
# Avoid
Nx, Ny, Nz  # Use num_points instead
dx, dy, dz  # Use spacing instead
Nt  # Use num_time_steps instead
dt  # Use time_step_size or TimeDomain.dt instead
```

**Exception**: Mathematical variables in limited scope are acceptable
```python
# OK in mathematical context (loop indices, temporary variables)
for i in range(Nx):
    x = xmin + i * dx  # Clear mathematical meaning in tight scope
```

### Domain and Geometry

#### ✅ DO: Use Clear, Explicit Names

```python
# Domain bounds
domain_bounds: tuple[float, ...] | DomainBounds
min_coords: np.ndarray  # Minimum coordinates per dimension
max_coords: np.ndarray  # Maximum coordinates per dimension

# Geometry objects
geometry: BaseGeometry
grid: TensorProductGrid
mesh: MeshData
```

#### ❌ DON'T: Use Ambiguous Abbreviations

```python
# Avoid
dom, geom, bnd  # Use full names for clarity
```

### Solution Variables

#### Mathematical Variables: Balance Clarity and Convention

**Core Variables** (keep concise for mathematical clarity):
```python
M: np.ndarray  # Density m(t,x) - standard MFG notation
U: np.ndarray  # Value function u(t,x) - standard MFG notation
m_initial: np.ndarray  # Initial density
u_final: np.ndarray  # Terminal condition
```

**Derived/Intermediate Variables** (use descriptive names):
```python
density_evolution: np.ndarray  # Full trajectory m(t,x)
value_trajectory: np.ndarray  # Full trajectory u(t,x)
gradient_u: np.ndarray  # ∇u
optimal_control: np.ndarray  # Optimal control α*
```

**Guideline**: Use `M`, `U` for primary solution variables in solver internals. Use descriptive names in user-facing APIs and documentation.

### Function and Method Names

#### ✅ DO: Use Verb-Noun Patterns

```python
# Actions
def compute_gradient(U: np.ndarray) -> np.ndarray: ...
def evaluate_hamiltonian(x, m, p, t) -> float: ...
def solve_hjb_system(...) -> np.ndarray: ...

# Queries
def get_collocation_points() -> np.ndarray: ...
def is_converged(error: float, tolerance: float) -> bool: ...

# Construction
def create_solver(problem, solver_type) -> BaseSolver: ...
def generate_mesh() -> MeshData: ...
```

#### ❌ DON'T: Use Ambiguous or Inconsistent Names

```python
# Avoid
def calc(...)  # Use compute_ or calculate_
def do_step(...)  # Use advance_timestep or similar
def proc(...)  # Use process_ or similar
```

---

## Parameter Design

### Boolean Parameters

#### When to Use Booleans

✅ **Good Cases** (single, independent toggle):
```python
def __init__(
    self,
    enable_logging: bool = True,  # Single on/off feature
    use_adaptive_timestep: bool = False,  # Independent from other params
):
```

❌ **Bad Cases** (mutually exclusive options):
```python
# WRONG: Boolean pairs
def __init__(
    self,
    use_method_a: bool = False,
    use_method_b: bool = True,
    use_method_c: bool = False,
):
```

#### When to Use Enums

✅ **Use Enum** for mutually exclusive options:
```python
from enum import Enum

class SolverMethod(str, Enum):
    """Solver method selection."""
    METHOD_A = "method_a"
    METHOD_B = "method_b"
    METHOD_C = "method_c"

def __init__(
    self,
    method: SolverMethod | str = SolverMethod.METHOD_B,
):
    # Convert string to enum if needed
    if isinstance(method, str):
        method = SolverMethod(method)
    self.method = method
```

**Benefits**:
- Type-safe: Catch typos at runtime
- Self-documenting: IDE autocomplete shows all options
- Extensible: Add new methods without boolean explosion
- Testable: Exhaustive testing of all enum values

### Enum Design Patterns

#### Pattern 1: String Enum (Recommended)

```python
class MyOption(str, Enum):
    """My option description."""
    OPTION_A = "option_a"  # Lowercase with underscores
    OPTION_B = "option_b"
    DEFAULT = "default"  # Explicit default value

# Usage:
MyOption.OPTION_A  # In code
MyOption("option_a")  # From string
MyOption.OPTION_A.value  # Get string value
```

**Rationale**: String enums serialize naturally (JSON, YAML, CLI args)

#### Pattern 2: Accept Both String and Enum

```python
def __init__(
    self,
    mode: MyOption | str = MyOption.DEFAULT,
):
    """
    Args:
        mode: Operating mode. Can be MyOption enum or string.
              Options: "option_a", "option_b", "default"
    """
    # Normalize to enum
    if isinstance(mode, str):
        try:
            mode = MyOption(mode)
        except ValueError:
            valid = [opt.value for opt in MyOption]
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {valid}"
            )
    self.mode = mode
```

**Benefits**:
- User flexibility: Strings for convenience, enums for type safety
- Clear errors: Invalid strings raise helpful messages
- IDE support: Enums enable autocomplete

---

## Return Values

### When to Return Tuples

✅ **Good Cases** (2-3 related values, obvious meaning):
```python
def get_bounds() -> tuple[np.ndarray, np.ndarray]:
    """
    Get domain bounds.

    Returns:
        (min_coords, max_coords) where each is shape (dimension,)
    """
    return self.min_coords, self.max_coords

# Unpacking is natural
min_coords, max_coords = geometry.get_bounds()
```

❌ **Bad Cases** (4+ values, non-obvious order, reused across codebase):
```python
# WRONG: Too many values, unclear order
def analyze() -> tuple[float, float, int, bool, str]:
    return error, tolerance, iterations, converged, message
```

### When to Use Dataclasses

✅ **Use Dataclass** for complex return values:
```python
from dataclasses import dataclass

@dataclass
class SolverResult:
    """Result from solver execution."""
    U: np.ndarray  # Value function
    M: np.ndarray  # Density
    converged: bool  # Convergence flag
    iterations: int  # Number of iterations
    error: float  # Final error
    error_history_U: list[float]  # Convergence history for U
    error_history_M: list[float]  # Convergence history for M
    message: str = ""  # Optional status message

    @property
    def success(self) -> bool:
        """Convenience property."""
        return self.converged

def solve(...) -> SolverResult:
    ...
    return SolverResult(
        U=U_final,
        M=M_final,
        converged=True,
        iterations=10,
        error=1e-8,
        error_history_U=errors_u,
        error_history_M=errors_m,
        message="Converged successfully"
    )

# Usage: Named access
result = solve(...)
if result.converged:
    print(f"Converged in {result.iterations} iterations")
    plt.semilogy(result.error_history_U)
```

**Benefits**:
- Named access: `result.iterations` vs `result[3]`
- Type hints: IDEs know field types
- Extensible: Add fields without breaking existing code
- Documentation: Dataclass fields self-document
- Properties: Can add computed properties

### Dataclass Design Patterns

#### Pattern 1: Core Data + Computed Properties

```python
@dataclass
class TimeDomain:
    """Time discretization parameters."""
    T_final: float  # Terminal time
    num_steps: int  # Number of time steps

    def __post_init__(self):
        """Validate inputs."""
        if self.T_final <= 0:
            raise ValueError("T_final must be positive")
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1")

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T_final / self.num_steps

    @property
    def time_grid(self) -> np.ndarray:
        """Time grid points."""
        return np.linspace(0, self.T_final, self.num_steps + 1)

    @classmethod
    def from_tuple(cls, t: tuple[float, int]) -> TimeDomain:
        """Support legacy tuple format."""
        return cls(T_final=t[0], num_steps=t[1])
```

#### Pattern 2: Accept Both Tuple and Dataclass

```python
def __init__(
    self,
    time_domain: TimeDomain | tuple[float, int] = (1.0, 100),
):
    """
    Args:
        time_domain: Time discretization. Can be TimeDomain or (T_final, num_steps) tuple.
    """
    # Normalize to dataclass
    if isinstance(time_domain, tuple):
        import warnings
        warnings.warn(
            "Passing time_domain as tuple is deprecated. Use TimeDomain dataclass.",
            DeprecationWarning,
            stacklevel=2,
        )
        time_domain = TimeDomain.from_tuple(time_domain)

    self.time_domain = time_domain
    self.T = time_domain.T_final  # Convenience alias
    self.Nt = time_domain.num_steps  # Convenience alias
```

---

## Deprecation Strategy

### Deprecation Lifecycle

**Version N (Deprecation Announced)**:
- Add new API
- Add deprecation warnings to old API
- Support both old and new simultaneously
- Update documentation to recommend new API

**Version N+1, N+2 (Deprecation Period)**:
- Maintain both APIs
- Warnings guide users to migration
- Collect user feedback

**Version N+3 or Major Version (Removal)**:
- Remove deprecated API
- Clean up compatibility code
- Update all examples and tests

### Deprecation Warning Template

```python
def __init__(
    self,
    # New parameter
    normalization: NormalizationType = NormalizationType.ALL,
    # Deprecated parameters
    normalize_output: bool | None = None,
    normalize_initial_only: bool | None = None,
):
    """
    Args:
        normalization: Normalization strategy (recommended).
        normalize_output: DEPRECATED. Use 'normalization' parameter instead.
        normalize_initial_only: DEPRECATED. Use 'normalization' parameter instead.
    """
    # Handle deprecated parameters
    if normalize_output is not None or normalize_initial_only is not None:
        import warnings
        warnings.warn(
            "Parameters 'normalize_output' and 'normalize_initial_only' are deprecated "
            "since v0.10.0 and will be removed in v2.0.0. "
            "Use 'normalization' parameter with NormalizationType enum instead.\n"
            "Examples:\n"
            "  - NormalizationType.ALL (default)\n"
            "  - NormalizationType.INITIAL_ONLY\n"
            "  - NormalizationType.NONE\n"
            "See docs/development/API_STYLE_GUIDE.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Map old parameters to new enum
        if normalize_output is False:
            normalization = NormalizationType.NONE
        elif normalize_initial_only is True:
            normalization = NormalizationType.INITIAL_ONLY
        else:
            normalization = NormalizationType.ALL

    # Convert string to enum if needed
    if isinstance(normalization, str):
        normalization = NormalizationType(normalization)

    self.normalization = normalization
```

**Key Elements**:
1. Clear deprecation message
2. Version information (when deprecated, when removed)
3. Migration instructions with examples
4. Reference to migration guide
5. Mapping logic for backward compatibility
6. Correct `stacklevel` (usually 2)

### Property-Based Deprecation

For attribute name changes:
```python
class MyClass:
    def __init__(self, num_points: int):
        self._num_points = num_points

    @property
    def num_points(self) -> int:
        """Number of grid points (recommended)."""
        return self._num_points

    @property
    def Nx(self) -> int:
        """DEPRECATED: Use num_points instead."""
        import warnings
        warnings.warn(
            "'Nx' property is deprecated since v0.10.0. Use 'num_points' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._num_points

    @Nx.setter
    def Nx(self, value: int) -> None:
        """DEPRECATED: Use num_points instead."""
        import warnings
        warnings.warn(
            "'Nx' property is deprecated since v0.10.0. Use 'num_points' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._num_points = value
```

---

## Type Hints

### Modern Type Hint Syntax

Use Python 3.10+ syntax with `from __future__ import annotations`:
```python
from __future__ import annotations

# ✅ Modern syntax (preferred)
def process(data: list[int]) -> dict[str, float]:
    ...

def optional_param(value: int | None = None) -> str | int:
    ...

# ❌ Old syntax (avoid)
from typing import List, Dict, Optional, Union

def process(data: List[int]) -> Dict[str, float]:  # Use list[int] instead
    ...

def optional_param(value: Optional[int] = None) -> Union[str, int]:  # Use | instead
    ...
```

### Union Types for Flexibility

Accept multiple formats for user convenience:
```python
def __init__(
    self,
    resolution: int | tuple[int, ...] | list[int],
    time_domain: TimeDomain | tuple[float, int],
    mode: MyMode | str = MyMode.DEFAULT,
):
    """Accept multiple formats, normalize internally."""
    ...
```

### Generic Type Hints

For backend-agnostic code:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Any

def compute(
    data: NDArray[np.floating] | Any,  # NumPy or backend array
) -> NDArray[np.floating] | Any:
    """Works with NumPy, JAX, PyTorch, etc."""
    ...
```

---

## Examples

### Example 1: Grid Class (Modern API)

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GridConfig:
    """Grid configuration parameters."""
    num_points: list[int]  # Grid points per dimension
    bounds: list[tuple[float, float]]  # (min, max) per dimension

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return len(self.num_points)

    @property
    def spacing(self) -> list[float]:
        """Grid spacing per dimension."""
        return [
            (bounds[1] - bounds[0]) / num_points
            for bounds, num_points in zip(self.bounds, self.num_points, strict=False)
        ]

    @classmethod
    def from_legacy(
        cls,
        bounds: tuple[float, ...],
        resolution: int | tuple[int, ...],
    ) -> GridConfig:
        """Support legacy tuple format."""
        # Parse bounds: (xmin, xmax, ymin, ymax, ...)
        if len(bounds) % 2 != 0:
            raise ValueError("Bounds must have even length")
        dim = len(bounds) // 2
        bounds_list = [(bounds[2*i], bounds[2*i+1]) for i in range(dim)]

        # Parse resolution
        if isinstance(resolution, int):
            num_points = [resolution] * dim
        else:
            num_points = list(resolution)

        return cls(num_points=num_points, bounds=bounds_list)


class ModernGrid:
    """Example of modern grid API."""

    def __init__(
        self,
        config: GridConfig,
        boundary_type: str = "periodic",
    ):
        """
        Initialize grid.

        Args:
            config: Grid configuration (use GridConfig dataclass)
            boundary_type: Boundary condition type
        """
        self.config = config
        self.boundary_type = boundary_type

    @classmethod
    def from_legacy(
        cls,
        bounds: tuple[float, ...],
        resolution: int | tuple[int, ...],
        **kwargs,
    ) -> ModernGrid:
        """
        Create grid from legacy tuple format.

        Deprecated: Use ModernGrid(GridConfig(...)) instead.
        """
        import warnings
        warnings.warn(
            "Creating grid from tuple bounds/resolution is deprecated. "
            "Use GridConfig dataclass instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        config = GridConfig.from_legacy(bounds, resolution)
        return cls(config, **kwargs)

    @property
    def num_points(self) -> list[int]:
        """Number of grid points per dimension."""
        return self.config.num_points

    @property
    def spacing(self) -> list[float]:
        """Grid spacing per dimension."""
        return self.config.spacing

    # Deprecated properties with warnings
    @property
    def Nx(self) -> int:
        """DEPRECATED: Use num_points[0] instead."""
        import warnings
        warnings.warn("'Nx' is deprecated. Use 'num_points[0]'.", DeprecationWarning, stacklevel=2)
        return self.num_points[0]

    @property
    def dx(self) -> float:
        """DEPRECATED: Use spacing[0] instead."""
        import warnings
        warnings.warn("'dx' is deprecated. Use 'spacing[0]'.", DeprecationWarning, stacklevel=2)
        return self.spacing[0]
```

### Example 2: Solver with Enum Options

```python
from enum import Enum

class NewtonSolverType(str, Enum):
    """Newton solver variant."""
    STANDARD = "standard"
    DAMPED = "damped"
    TRUST_REGION = "trust_region"

class ConvergenceCriterion(str, Enum):
    """Convergence criterion type."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    BOTH = "both"

class ModernSolver:
    """Example of modern solver API."""

    def __init__(
        self,
        problem,
        solver_type: NewtonSolverType | str = NewtonSolverType.STANDARD,
        convergence_criterion: ConvergenceCriterion | str = ConvergenceCriterion.RELATIVE,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        # Deprecated parameters
        use_damping: bool | None = None,
        check_relative_error: bool | None = None,
    ):
        """
        Initialize solver.

        Args:
            problem: MFG problem instance
            solver_type: Newton solver variant (recommended)
            convergence_criterion: Convergence criterion type (recommended)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            use_damping: DEPRECATED. Use solver_type='damped' instead.
            check_relative_error: DEPRECATED. Use convergence_criterion='relative' instead.
        """
        # Handle deprecated boolean flags
        if use_damping is not None:
            import warnings
            warnings.warn(
                "'use_damping' is deprecated. Use solver_type='damped' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if use_damping:
                solver_type = NewtonSolverType.DAMPED

        if check_relative_error is not None:
            import warnings
            warnings.warn(
                "'check_relative_error' is deprecated. Use convergence_criterion instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            convergence_criterion = ConvergenceCriterion.RELATIVE if check_relative_error else ConvergenceCriterion.ABSOLUTE

        # Normalize string to enum
        if isinstance(solver_type, str):
            solver_type = NewtonSolverType(solver_type)
        if isinstance(convergence_criterion, str):
            convergence_criterion = ConvergenceCriterion(convergence_criterion)

        self.problem = problem
        self.solver_type = solver_type
        self.convergence_criterion = convergence_criterion
        self.max_iterations = max_iterations
        self.tolerance = tolerance
```

---

## Testing Guidelines

### Test Enum Values Exhaustively

```python
def test_all_solver_types():
    """Test all solver type options."""
    problem = create_test_problem()

    for solver_type in NewtonSolverType:
        solver = ModernSolver(problem, solver_type=solver_type)
        assert solver.solver_type == solver_type
```

### Test Deprecation Warnings

```python
def test_deprecated_parameter_warning():
    """Verify deprecation warning for old parameters."""
    problem = create_test_problem()

    with pytest.warns(DeprecationWarning, match="use_damping.*deprecated"):
        solver = ModernSolver(problem, use_damping=True)

    # Verify correct mapping
    assert solver.solver_type == NewtonSolverType.DAMPED
```

### Test Both Tuple and Dataclass

```python
def test_time_domain_formats():
    """Test both tuple and dataclass time_domain formats."""
    problem_class = ModernProblem

    # Test tuple format (with deprecation warning)
    with pytest.warns(DeprecationWarning):
        problem1 = problem_class(time_domain=(1.0, 100))

    # Test dataclass format (no warning)
    time_domain = TimeDomain(T_final=1.0, num_steps=100)
    problem2 = problem_class(time_domain=time_domain)

    # Both should work identically
    assert problem1.T == problem2.T == 1.0
    assert problem1.Nt == problem2.Nt == 100
```

---

## Migration Checklist

When refactoring existing APIs:

- [ ] Identify all usage locations (grep, IDE search)
- [ ] Create new API (enum/dataclass)
- [ ] Add deprecation warnings to old API
- [ ] Implement mapping from old to new
- [ ] Update documentation (docstrings, examples, guides)
- [ ] Add tests for new API
- [ ] Add tests for deprecated API (verify warnings)
- [ ] Update all examples to use new API
- [ ] Add migration notes to CHANGELOG
- [ ] Set removal version in deprecation message

---

## References

- **API_CONSISTENCY_VIOLATIONS.md**: Comprehensive violation inventory
- **NAMING_CONVENTIONS.md**: Naming standards
- **CONSISTENCY_GUIDE.md**: Development consistency guide
- **Issue #277**: API Consistency Audit tracking

---

**Document Version**: 1.0
**Last Updated**: 2025-11-11
**Maintained By**: MFG_PDE Development Team
