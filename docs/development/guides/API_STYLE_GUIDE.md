# MFG_PDE API Style Guide

**Version**: 1.0
**Date**: 2025-10-10
**Status**: Official Reference

This guide establishes the API design standards for MFG_PDE, ensuring consistency, clarity, and usability across the entire codebase.

---

## Table of Contents

1. [Naming Conventions](#naming-conventions)
2. [Parameter Design](#parameter-design)
3. [Enum vs Boolean Guidelines](#enum-vs-boolean-guidelines)
4. [Return Type Standards](#return-type-standards)
5. [Deprecation Procedures](#deprecation-procedures)
6. [Type Hints and Annotations](#type-hints-and-annotations)
7. [Examples and Anti-Patterns](#examples-and-anti-patterns)

---

## 1. Naming Conventions

### Mathematical Entities: Uppercase

Use **uppercase** for mathematical entities, following standard mathematical notation:

**Grid Dimensions:**
```python
# ✅ GOOD: Uppercase for mathematical grid dimensions
Nx: int = 100  # Number of spatial grid points
Nt: int = 50   # Number of temporal grid points
Ny: int = 100  # For 2D problems
Nz: int = 100  # For 3D problems

# ❌ BAD: Lowercase for mathematical entities
nx: int = 100  # Inconsistent with mathematical notation
nt: int = 50
```

**Solution Arrays:**
```python
# ✅ GOOD: Uppercase for mathematical quantities
U: np.ndarray  # Value function u(t,x)
M: np.ndarray  # Density function m(t,x)
V: np.ndarray  # Velocity field v(t,x)

# ❌ BAD: Lowercase for mathematical quantities
u: np.ndarray  # Less clear
m: np.ndarray
```

**Mathematical Constants:**
```python
# ✅ GOOD: Uppercase for significant mathematical parameters
T: float  # Terminal time
L: float  # Domain length
Sigma: float  # Diffusion coefficient

# ❌ BAD: Lowercase for mathematical constants
t: float  # Could be confused with time variable
sigma: float  # Less visually distinct
```

### Metadata and Configuration: Lowercase

Use **lowercase** for metadata, configuration, and non-mathematical entities:

```python
# ✅ GOOD: Lowercase for metadata
iterations: int  # Number of iterations performed
converged: bool  # Convergence status
execution_time: float  # Runtime in seconds
max_iterations: int  # Configuration parameter
tolerance: float  # Convergence threshold

# ✅ GOOD: Lowercase for configuration
solver_type: str
backend_name: str
enable_logging: bool
```

### Rationale

- **Mathematical tradition**: Uppercase matches standard mathematical notation (N for cardinality, U for utility)
- **Visual distinction**: Uppercase entities stand out as mathematical quantities
- **Consistency**: Aligns with NumPy/SciPy conventions (N for matrix dimensions)

---

## 2. Parameter Design

### General Principles

1. **Explicit over implicit**: Prefer explicit parameter names over positional arguments
2. **Sensible defaults**: Provide defaults for optional parameters
3. **Type hints required**: All public API parameters must have type hints
4. **Mutually exclusive options**: Use enums, not multiple booleans

### Parameter Ordering

```python
def create_solver(
    # 1. Required mathematical parameters (no defaults)
    Nx: int,
    Nt: int,
    T: float,

    # 2. Optional mathematical parameters (with defaults)
    sigma: float = 1.0,
    lambda_param: float = 0.5,

    # 3. Configuration options (enums, with defaults)
    backend: AutoDiffBackend = AutoDiffBackend.NUMPY,
    normalization: NormalizationType = NormalizationType.LAYER,

    # 4. Behavioral flags (booleans, with defaults)
    verbose: bool = False,
    enable_validation: bool = True,
) -> Solver:
    """Create a solver with sensible parameter ordering."""
    ...
```

---

## 3. Enum vs Boolean Guidelines

### When to Use Enums

Use **enums** for mutually exclusive options (select one of many):

```python
from enum import Enum

# ✅ GOOD: Enum for mutually exclusive choices
class AutoDiffBackend(str, Enum):
    NUMPY = "numpy"
    JAX = "jax"
    PYTORCH = "pytorch"

class NormalizationType(str, Enum):
    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"

# Usage: Clear and type-safe
config = SolverConfig(
    backend=AutoDiffBackend.JAX,
    normalization=NormalizationType.BATCH,
)
```

**Benefits:**
- **Type safety**: Cannot accidentally specify invalid combinations
- **Self-documenting**: IDE autocomplete shows all valid options
- **Extensible**: Easy to add new options without breaking existing code
- **Validation**: Automatic validation of enum values

### When to Use Booleans

Use **booleans** for independent on/off flags (not mutually exclusive):

```python
# ✅ GOOD: Booleans for independent flags
def solve_mfg(
    verbose: bool = False,           # Independent: logging on/off
    enable_validation: bool = True,  # Independent: validation on/off
    save_history: bool = False,      # Independent: history tracking on/off
) -> Solution:
    """Independent boolean flags are acceptable."""
    ...
```

### Anti-Pattern: Boolean Proliferation

**❌ AVOID: Multiple mutually exclusive booleans**

```python
# ❌ BAD: Confusing - what if both are True?
class BadConfig:
    use_jax: bool = False
    use_pytorch: bool = False
    use_numpy: bool = True

# ❌ BAD: Confusing - what if both are True?
class BadDeepONetConfig:
    use_batch_norm: bool = False
    use_layer_norm: bool = True

# Problem: Need complex validation logic
if config.use_jax and config.use_pytorch:
    raise ValueError("Cannot use both JAX and PyTorch!")
```

**✅ SOLUTION: Use enum instead**

```python
# ✅ GOOD: Clear, type-safe, self-documenting
class GoodConfig:
    backend: AutoDiffBackend = AutoDiffBackend.NUMPY
    normalization: NormalizationType = NormalizationType.LAYER

# Usage is clear and unambiguous
config = GoodConfig(backend=AutoDiffBackend.JAX)
```

### Decision Tree: Enum vs Boolean

```
Is this parameter selecting ONE option from MANY mutually exclusive choices?
    ├─ YES → Use Enum
    │   Examples: backend selection, normalization type, solver method
    │
    └─ NO → Are these independent on/off flags?
        ├─ YES → Use Boolean(s)
        │   Examples: verbose, enable_validation, save_history
        │
        └─ NO → Reconsider your API design
            Maybe you need multiple parameters or a different structure
```

---

## 4. Return Type Standards

### Simple Returns: Tuples

For 2-3 simple values, tuples are acceptable:

```python
def get_grid_params() -> tuple[int, int, float]:
    """Return grid parameters as a simple tuple."""
    return Nx, Nt, dt

# Usage: Simple unpacking
Nx, Nt, dt = get_grid_params()
```

### Complex Returns: Dataclasses

For 4+ values or semantic meaning, use dataclasses:

```python
from dataclasses import dataclass

@dataclass
class SolverResult:
    """Result from solver execution."""

    # Solution arrays (uppercase for mathematical quantities)
    U: np.ndarray  # Value function
    M: np.ndarray  # Density function

    # Metadata (lowercase for non-mathematical)
    iterations: int
    converged: bool
    residual: float
    execution_time: float

def solve_mfg(...) -> SolverResult:
    """Return structured result."""
    return SolverResult(
        U=value_function,
        M=density,
        iterations=iter_count,
        converged=True,
        residual=1e-6,
        execution_time=elapsed_time,
    )

# Usage: Clear field access
result = solve_mfg(...)
print(f"Converged in {result.iterations} iterations")
print(f"Final residual: {result.residual}")
```

### Benefits of Dataclasses

- **Named fields**: `result.U` is clearer than `result[0]`
- **Type hints**: Each field has explicit type
- **Documentation**: Docstrings for fields
- **Extensibility**: Can add fields without breaking existing code
- **IDE support**: Autocomplete for field names

### When to Use What

| Return Complexity | Recommended Approach | Example |
|:------------------|:--------------------|:--------|
| Single value | Direct return | `return U` |
| 2-3 simple values | Tuple | `return U, M, dt` |
| 4+ values | Dataclass | `return SolverResult(...)` |
| Complex semantics | Dataclass | Even for 2-3 values if meaning is important |

---

## 5. Deprecation Procedures

### Why Deprecate Instead of Breaking Changes

- **User trust**: Breaking changes frustrate users
- **Migration path**: Users need time to update their code
- **Backward compatibility**: Existing code continues to work

### Standard Deprecation Timeline

**2-Version Cycle:**

1. **Version N**: Introduce new API, deprecate old API
   - Old API still works but emits deprecation warnings
   - New API is recommended and documented

2. **Version N+1**: Maintain both APIs
   - Old API still works with deprecation warnings
   - Users have full version cycle to migrate

3. **Version N+2**: Remove old API
   - Only new API remains
   - Breaking change is documented in release notes

### Deprecation Pattern: Parameters

```python
import warnings
from dataclasses import dataclass

@dataclass
class ModernConfig:
    # New API (recommended)
    Nx: int | None = None
    Nt: int | None = None
    backend: AutoDiffBackend = AutoDiffBackend.NUMPY

    # Deprecated API (for backward compatibility)
    nx: int | None = None  # DEPRECATED
    nt: int | None = None  # DEPRECATED
    use_jax: bool | None = None  # DEPRECATED

    def __post_init__(self) -> None:
        """Handle deprecated parameters with warnings."""
        # Handle deprecated lowercase grid parameters
        if self.nx is not None:
            warnings.warn(
                "Parameter 'nx' is deprecated, use 'Nx' (uppercase) instead",
                DeprecationWarning,
                stacklevel=2,  # Correct stack level for user code
            )
            if self.Nx is None:
                self.Nx = self.nx

        if self.nt is not None:
            warnings.warn(
                "Parameter 'nt' is deprecated, use 'Nt' (uppercase) instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.Nt is None:
                self.Nt = self.nt

        # Handle deprecated boolean backend parameter
        if self.use_jax is not None:
            warnings.warn(
                "Parameter 'use_jax' is deprecated, "
                "use 'backend=AutoDiffBackend.JAX' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.use_jax and self.backend == AutoDiffBackend.NUMPY:
                self.backend = AutoDiffBackend.JAX
```

### Deprecation Warning Format

```python
warnings.warn(
    "Parameter '{old_name}' is deprecated, use '{new_api}' instead",
    DeprecationWarning,
    stacklevel=2,  # Points to user's code, not library internals
)
```

### Documentation Requirements

1. **Docstring**: Mark parameter as deprecated
   ```python
   def function(
       Nx: int,
       nx: int | None = None,  # DEPRECATED: Use Nx instead
   ):
       """
       Function description.

       Args:
           Nx: Number of grid points (use this)
           nx: DEPRECATED - Use Nx instead
       """
   ```

2. **Changelog**: Document deprecation
   ```markdown
   ## Version 0.5.0

   ### Deprecated
   - `nx`, `nt` parameters: Use uppercase `Nx`, `Nt` instead
   - `use_jax` boolean: Use `backend=AutoDiffBackend.JAX` instead

   ### Migration Guide
   Old code:
   ```python
   config = Config(nx=100, nt=50, use_jax=True)
   ```

   New code:
   ```python
   config = Config(Nx=100, Nt=50, backend=AutoDiffBackend.JAX)
   ```
   ```

---

## 6. Type Hints and Annotations

### Required Type Hints

All public API functions must have complete type hints:

```python
# ✅ GOOD: Complete type hints
def solve_mfg(
    Nx: int,
    Nt: int,
    T: float,
    sigma: float = 1.0,
    backend: AutoDiffBackend = AutoDiffBackend.NUMPY,
) -> SolverResult:
    """Solve MFG problem."""
    ...

# ❌ BAD: Missing type hints
def solve_mfg(Nx, Nt, T, sigma=1.0):
    """Solve MFG problem."""
    ...
```

### Modern Python Type Syntax

Use modern `|` syntax for unions (Python 3.10+):

```python
# ✅ GOOD: Modern union syntax
def process(value: int | float | None = None) -> np.ndarray | None:
    """Modern type hints."""
    ...

# ❌ OUTDATED: Old Union syntax (still works but verbose)
from typing import Union, Optional
def process(value: Optional[Union[int, float]] = None) -> Optional[np.ndarray]:
    """Old style type hints."""
    ...
```

### Optional Parameters

```python
# ✅ GOOD: Use | None for optional parameters
def compute(
    required: int,
    optional: float | None = None,
) -> float:
    """Function with optional parameter."""
    if optional is None:
        optional = 1.0
    return required * optional
```

---

## 7. Examples and Anti-Patterns

### Example 1: Grid Parameter Naming

**❌ BEFORE (Inconsistent):**
```python
class BadConfig:
    nx: int = 100  # Lowercase
    nt: int = 50   # Lowercase
    Nx: int = 100  # Mixed usage
```

**✅ AFTER (Consistent):**
```python
class GoodConfig:
    Nx: int = 100  # Uppercase for mathematical
    Nt: int = 50   # Uppercase for mathematical

    # Deprecated for backward compatibility
    nx: int | None = None  # DEPRECATED
    nt: int | None = None  # DEPRECATED

    def __post_init__(self) -> None:
        """Handle deprecated parameters."""
        if self.nx is not None:
            warnings.warn("Use 'Nx' instead", DeprecationWarning)
            if self.Nx == 100:  # Default value
                self.Nx = self.nx
```

### Example 2: Backend Selection

**❌ BEFORE (Boolean Proliferation):**
```python
class BadConfig:
    use_numpy: bool = True
    use_jax: bool = False
    use_pytorch: bool = False

    def __post_init__(self):
        # Complex validation needed
        backends_enabled = sum([
            self.use_numpy,
            self.use_jax,
            self.use_pytorch
        ])
        if backends_enabled != 1:
            raise ValueError("Exactly one backend must be enabled!")
```

**✅ AFTER (Enum-Based):**
```python
from mfg_pde.utils.numerical.autodiff import AutoDiffBackend

class GoodConfig:
    backend: AutoDiffBackend = AutoDiffBackend.NUMPY

    # Deprecated for backward compatibility
    use_jax: bool | None = None  # DEPRECATED
    use_pytorch: bool | None = None  # DEPRECATED

    def __post_init__(self) -> None:
        """Handle deprecated parameters."""
        if self.use_jax is not None:
            warnings.warn(
                "Use 'backend=AutoDiffBackend.JAX' instead",
                DeprecationWarning,
            )
            if self.use_jax:
                self.backend = AutoDiffBackend.JAX
```

### Example 3: Normalization Type

**❌ BEFORE (Mutually Exclusive Booleans):**
```python
class BadDeepONetConfig:
    use_batch_norm: bool = False
    use_layer_norm: bool = True

    def validate(self):
        if self.use_batch_norm and self.use_layer_norm:
            raise ValueError("Cannot use both normalizations!")
```

**✅ AFTER (Enum-Based):**
```python
from mfg_pde.utils.neural import NormalizationType

class GoodDeepONetConfig:
    normalization: NormalizationType = NormalizationType.LAYER

    # Deprecated for backward compatibility
    use_batch_norm: bool | None = None  # DEPRECATED
    use_layer_norm: bool | None = None  # DEPRECATED
```

### Example 4: Return Types

**❌ BEFORE (Unnamed Tuple):**
```python
def solve_mfg(...) -> tuple:
    """Solve MFG problem."""
    # What does each position mean?
    return U, M, 42, True, 0.001, 2.5

# Usage: Unclear meaning
result = solve_mfg(...)
print(result[2])  # What is index 2?
```

**✅ AFTER (Dataclass):**
```python
@dataclass
class SolverResult:
    """Result from MFG solver."""
    U: np.ndarray  # Value function
    M: np.ndarray  # Density function
    iterations: int
    converged: bool
    residual: float
    execution_time: float

def solve_mfg(...) -> SolverResult:
    """Solve MFG problem."""
    return SolverResult(
        U=value_function,
        M=density,
        iterations=42,
        converged=True,
        residual=0.001,
        execution_time=2.5,
    )

# Usage: Clear and self-documenting
result = solve_mfg(...)
print(result.iterations)  # Clear meaning
if result.converged:
    print(f"Solution converged with residual {result.residual}")
```

---

## Summary: API Design Checklist

When designing or reviewing API, ask:

- [ ] **Naming**: Are mathematical entities uppercase? (Nx, Nt, U, M)
- [ ] **Naming**: Is metadata lowercase? (iterations, converged)
- [ ] **Parameters**: Are mutually exclusive options using enums?
- [ ] **Parameters**: Are independent flags using booleans?
- [ ] **Type Hints**: Are all parameters and returns fully typed?
- [ ] **Returns**: Are complex returns using dataclasses?
- [ ] **Deprecation**: Are old APIs deprecated with warnings?
- [ ] **Documentation**: Are deprecations documented in docstrings?
- [ ] **Backward Compatibility**: Does old code still work?

---

## Related Documentation

- **Consistency Guide**: `docs/development/guides/CONSISTENCY_GUIDE.md`
- **Type Checking Philosophy**: See `CLAUDE.md` section
- **Modern Python Typing**: See `CLAUDE.md` section on `@overload`, `isinstance()`, `cast()`

---

**Enforcement**: This guide is the official API design standard for MFG_PDE. All new code must follow these conventions. Existing code should be gradually migrated using the deprecation procedures outlined above.

**Exceptions**: Any exceptions to these rules must be documented and justified in code comments and design documents.
