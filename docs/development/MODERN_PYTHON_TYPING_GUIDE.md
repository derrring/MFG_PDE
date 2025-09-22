# Modern Python Typing Guide for Scientific Computing (Python 3.9+)

**Date**: 2025-09-20
**Context**: Best practices for type hints in scientific Python projects
**Target**: Python 3.9+ with modern syntax

## 🎯 **Executive Summary**

This guide synthesizes modern Python typing best practices specifically for scientific computing projects like MFG_PDE. It emphasizes clean, maintainable code using Python 3.9+ built-in collection types and Python 3.10+ union syntax.

## 📖 **Table of Contents**

1. [Syntax: Use Modern Built-ins and Operators](#1-syntax-use-modern-built-ins-and-operators)
2. [Clarity: Use TypeAlias for Complex Signatures](#2-clarity-use-typealias-for-complex-signatures)
3. [API Design: Layer Your Types for Your Audience](#3-api-design-layer-your-types-for-your-audience)
4. [Backward Compatibility: Use `__future__` Imports](#4-backward-compatibility-use-__future__-imports)
5. [Scientific Computing Specific Patterns](#5-scientific-computing-specific-patterns)
6. [Configuration and Tool Setup](#6-configuration-and-tool-setup)
7. [Migration Strategy for Existing Projects](#7-migration-strategy-for-existing-projects)

---

## 1. Syntax: Use Modern Built-ins and Operators

The most significant improvement in modern typing is moving away from the `typing` module for generic types. This creates cleaner code with fewer imports.

### **Modern Type Syntax (Python 3.9+)**

| Legacy Syntax (Pre-3.9) | Modern Syntax (3.9+) | Notes |
|-------------------------|----------------------|-------|
| `from typing import List`<br>`List[int]` | `list[int]` | Simpler, no import needed |
| `from typing import Dict`<br>`Dict[str, float]` | `dict[str, float]` | Also applies to `tuple`, `set`, etc. |
| `from typing import Union`<br>`Union[int, str]` | `int \| str` | Much more readable (Python 3.10+) |
| `from typing import Optional`<br>`Optional[bool]` | `bool \| None` | `Optional` is just union with `None` |
| `from typing import Tuple`<br>`Tuple[int, str]` | `tuple[int, str]` | Fixed-length tuples |

### **What Still Requires Import from `typing`**

```python
from typing import (
    Any, Callable, TypeAlias, Protocol, TypeVar,
    Literal, Final, ClassVar, overload
)

# Special constructs that don't have built-in equivalents
UserCallback: TypeAlias = Callable[[str, int], bool]
T = TypeVar('T')
```

### **Scientific Computing Example**

```python
# ✅ MODERN - Clean and readable
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

# Type aliases for scientific data
SolutionArray: TypeAlias = NDArray[np.float64]
GridPoints: TypeAlias = tuple[int, ...]
ParameterDict: TypeAlias = dict[str, float | int | str]

def solve_mfg(
    initial_u: SolutionArray,
    grid_resolution: GridPoints,
    parameters: ParameterDict,
    callback: Callable[[int, float], None] | None = None
) -> tuple[SolutionArray, SolutionArray]:
    """Modern typing for MFG solver."""
    pass

# ❌ LEGACY - Verbose and cluttered
from typing import Optional, Callable, Dict, Tuple, Union
def solve_mfg_old(
    initial_u: NDArray,
    grid_resolution: Tuple[int, ...],
    parameters: Dict[str, Union[float, int, str]],
    callback: Optional[Callable[[int, float], None]] = None
) -> Tuple[NDArray, NDArray]:
    pass
```

---

## 2. Clarity: Use TypeAlias for Complex Signatures

When type signatures become complex, create descriptive aliases. This dramatically improves readability and maintainability.

### **Scientific Computing Type Aliases**

```python
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

# ✅ EXCELLENT - Self-documenting scientific types
SpatialGrid: TypeAlias = NDArray[np.float64]  # x-coordinates
TemporalGrid: TypeAlias = NDArray[np.float64]  # t-coordinates
ValueFunction: TypeAlias = NDArray[np.float64]  # u(t,x)
DensityFunction: TypeAlias = NDArray[np.float64]  # m(t,x)

# Complex data structures
SolverState: TypeAlias = dict[str, ValueFunction | DensityFunction | float | int]
ConvergenceHistory: TypeAlias = list[float]
SolverMetadata: TypeAlias = dict[str,
    SpatialGrid | TemporalGrid | str | bool | dict[str, Any]
]

# Result type with rich structure
MFGResult: TypeAlias = dict[str,
    ValueFunction | DensityFunction | bool | int | ConvergenceHistory | SolverMetadata
]

# ✅ CLEAN - Function signature is now self-documenting
def fixed_point_iteration(
    initial_state: SolverState,
    metadata: SolverMetadata,
    max_iterations: int = 100
) -> MFGResult:
    """Clean function signature using type aliases."""
    pass

# ❌ UNREADABLE - Complex nested types in signature
def fixed_point_iteration_bad(
    initial_state: dict[str, NDArray[np.float64] | float | int],
    metadata: dict[str, NDArray[np.float64] | str | bool | dict[str, Any]],
    max_iterations: int = 100
) -> dict[str, NDArray[np.float64] | bool | int | list[float] | dict[str, Any]]:
    pass
```

### **Protocol-Based Type Design**

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MFGProblem(Protocol):
    """Protocol for Mean Field Game problems."""

    def get_domain_bounds(self) -> tuple[float, float]:
        """Return spatial domain bounds."""
        ...

    def get_time_horizon(self) -> float:
        """Return time horizon T."""
        ...

    def evaluate_hamiltonian(
        self, x: float, p: float, m: float, t: float
    ) -> float:
        """Evaluate Hamiltonian H(x, p, m, t)."""
        ...

# ✅ FLEXIBLE - Duck typing with type safety
def solve_problem(problem: MFGProblem) -> MFGResult:
    """Accepts any object implementing the MFGProblem protocol."""
    bounds = problem.get_domain_bounds()
    T = problem.get_time_horizon()
    # Implementation...
```

---

## 3. API Design: Layer Your Types for Your Audience

Structure your API to reveal complexity progressively. Not every user needs to see your full type system.

### **Three-Layer Type Architecture**

#### **Layer 1: Simple Facade (90% of users)**
```python
# mfg_pde/__init__.py - Simple string-based configuration
def solve_mfg(
    problem_type: str,
    domain_size: float = 1.0,
    time_horizon: float = 1.0,
    accuracy: str = "balanced"  # "fast" | "balanced" | "precise"
) -> dict[str, Any]:
    """Simple one-line solver for common use cases."""
    pass

def solve_crowd_dynamics(
    domain_bounds: tuple[float, float],
    num_agents: int,
    time_steps: int = 100
) -> dict[str, Any]:
    """Domain-specific simple interface."""
    pass
```

#### **Layer 2: Core Objects (Advanced users)**
```python
# mfg_pde/core.py - Typed objects with clean interfaces
class FixedPointSolver:
    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        damping_factor: float = 1.0
    ) -> None:
        pass

    def solve(self, problem: MFGProblem) -> MFGResult:
        """Solve with full control over parameters."""
        pass

class MFGSolverConfig:
    """Configuration object with validation."""
    tolerance: float
    max_iterations: int
    solver_type: Literal["fixed_point", "newton", "quasi_newton"]
```

#### **Layer 3: Advanced/Internal (Expert users)**
```python
# mfg_pde/types.py - Full internal type system
class SpatialTemporalState(NamedTuple):
    """Internal solver state with full type annotations."""
    u: ValueFunction  # u(t,x)
    m: DensityFunction  # m(t,x)
    iteration: int
    residual: float
    metadata: dict[str, Any]

# mfg_pde/hooks.py - Expert customization
class SolverHooks(Protocol):
    def on_iteration_start(self, state: SpatialTemporalState) -> None: ...
    def on_iteration_end(self, state: SpatialTemporalState) -> str | None: ...
    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None: ...
```

---

## 4. Backward Compatibility: Use `__future__` Imports

For libraries supporting older Python versions while using modern syntax:

```python
# Must be at the top of the file
from __future__ import annotations

# Now you can use modern syntax even in Python 3.8
def get_solution_data() -> list[tuple[str, NDArray]]:
    """Modern syntax with backward compatibility."""
    return [("u", u_array), ("m", m_array)]

# Type checking conditional imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .advanced_types import ComplexInternalType
```

**Important Notes:**
- `from __future__ import annotations` enables `list[int]`, `dict[str, float]` syntax
- The `|` union operator (`int | str`) still requires Python 3.10+
- For older versions, continue using `Union[int, str]`

---

## 5. Scientific Computing Specific Patterns

### **NumPy Array Typing**

```python
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

# ✅ RECOMMENDED - Simple and flexible
SolutionArray: TypeAlias = NDArray[np.float64]
SpatialGrid: TypeAlias = NDArray[np.float64]

# ✅ ACCEPTABLE - When you need specific shapes
Matrix2D: TypeAlias = NDArray[np.float64]  # Shape checking at runtime
Vector1D: TypeAlias = NDArray[np.float64]

# ❌ AVOID - Too specific, breaks across numpy versions
# NDArray[np.float64, Literal["N", "M"]]  # Shape annotations are experimental

def solve_system(
    u: SolutionArray,
    grid: SpatialGrid,
    dt: float,
    dx: float
) -> SolutionArray:
    """Clear scientific function signature."""
    # Runtime shape validation
    assert u.ndim == 2, f"Expected 2D array, got {u.ndim}D"
    assert grid.ndim == 1, f"Expected 1D grid, got {grid.ndim}D"
    return u  # Implementation...
```

### **Configuration and Parameters**

```python
from typing import TypeAlias, Literal
from dataclasses import dataclass

# Modern configuration patterns
SolverMethod: TypeAlias = Literal["picard", "newton", "quasi_newton"]
AccuracyLevel: TypeAlias = Literal["fast", "balanced", "precise", "research"]

@dataclass
class ModernSolverConfig:
    """Modern configuration with literal types."""
    method: SolverMethod = "picard"
    accuracy: AccuracyLevel = "balanced"
    tolerance: float = 1e-6
    max_iterations: int = 100

    # Optional parameters using modern union syntax
    custom_residual: Callable[[SolutionArray, SolutionArray], float] | None = None
    adaptive_damping: bool = False

    def __post_init__(self) -> None:
        """Validation with proper return type."""
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
```

### **Error Handling and Validation**

```python
from typing import TypeAlias

# Modern exception handling
ValidationResult: TypeAlias = dict[str, list[str] | bool | dict[str, float]]

def validate_solution(
    u: SolutionArray,
    m: DensityFunction,
    strict: bool = True
) -> ValidationResult:
    """Modern validation with clear return type."""
    result: ValidationResult = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "diagnostics": {}
    }

    # Type-safe validation logic
    if np.isnan(u).any():
        result["errors"].append("NaN values in value function")
        result["valid"] = False

    return result
```

---

## 6. Configuration and Tool Setup

### **pyproject.toml for Modern Typing**

```toml
[project]
name = "mfg-pde"
requires-python = ">=3.9"  # Enable modern typing

[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

# Scientific computing friendly settings
ignore_missing_imports = true  # For optional dependencies
no_implicit_optional = true

# Allow mathematical variable names
[tool.pylint.basic]
good-names = ["i", "j", "k", "x", "y", "z", "t", "u", "m", "T", "N", "nx", "nt"]

[tool.ruff]
target-version = "py39"
select = ["E", "F", "UP"]  # UP checks for upgradeable syntax

[tool.ruff.pyupgrade]
keep-runtime-typing = false  # Remove unnecessary typing imports
```

### **Pre-commit Hook for Modern Typing**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [numpy, types-requests]
```

---

## 7. Migration Strategy for Existing Projects

### **Step 1: Update Python Version Requirements**
```toml
# pyproject.toml
[project]
requires-python = ">=3.9"  # Enable modern syntax
```

### **Step 2: Add `__future__` Import Gradually**
```python
# Add to files as you modify them
from __future__ import annotations

# Now modernize types gradually
def process_data(items: list[str]) -> dict[str, int]:  # Modern syntax
    pass
```

### **Step 3: Remove Unnecessary Imports**
```python
# ❌ OLD - Remove these imports
from typing import List, Dict, Tuple, Optional, Union

# ✅ NEW - Keep only what you need
from typing import TypeAlias, Protocol, Callable, Any

# Modern usage
ProcessingResult: TypeAlias = dict[str, list[float] | bool]
```

### **Step 4: Create Type Aliases for Complex Types**
```python
# Before: Complex inline types
def complex_function(
    data: dict[str, list[tuple[NDArray[np.float64], dict[str, float | int]]]]
) -> list[dict[str, NDArray[np.float64] | bool]]:
    pass

# After: Clean type aliases
ExperimentData: TypeAlias = dict[str, list[tuple[SolutionArray, ParameterDict]]]
ProcessedResults: TypeAlias = list[dict[str, SolutionArray | bool]]

def complex_function(data: ExperimentData) -> ProcessedResults:
    pass
```

### **Step 5: Use Automated Tools**
```bash
# Use pyupgrade to modernize syntax automatically
pip install pyupgrade
pyupgrade --py39-plus **/*.py

# Use ruff to check for upgradeable patterns
ruff check --select UP .
```

---

## 🔧 **Practical Example: MFG_PDE Modernization**

### **Before (Legacy Typing)**
```python
from typing import Optional, List, Dict, Tuple, Union, Callable, Any
import numpy as np
from numpy.typing import NDArray

def solve_mfg_system(
    initial_u: NDArray[np.float64],
    initial_m: NDArray[np.float64],
    domain_bounds: Tuple[float, float],
    time_horizon: float,
    config: Optional[Dict[str, Union[float, int, str]]] = None,
    callback: Optional[Callable[[int, float], None]] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, Any]]:
    pass
```

### **After (Modern Typing)**
```python
from __future__ import annotations  # For backward compatibility
from typing import TypeAlias, Callable
import numpy as np
from numpy.typing import NDArray

# Clear type aliases
SolutionArray: TypeAlias = NDArray[np.float64]
DomainBounds: TypeAlias = tuple[float, float]
SolverConfig: TypeAlias = dict[str, float | int | str]
IterationCallback: TypeAlias = Callable[[int, float], None]
SolverResult: TypeAlias = tuple[SolutionArray, SolutionArray, dict[str, Any]]

def solve_mfg_system(
    initial_u: SolutionArray,
    initial_m: SolutionArray,
    domain_bounds: DomainBounds,
    time_horizon: float,
    config: SolverConfig | None = None,
    callback: IterationCallback | None = None
) -> SolverResult:
    """Modern, clean, and self-documenting."""
    pass
```

---

## 📋 **Quick Reference Checklist**

### **✅ Modern Typing Checklist**
- [ ] Use `list[T]`, `dict[K, V]`, `tuple[T, ...]` instead of `typing` imports
- [ ] Use `X | Y` instead of `Union[X, Y]` (Python 3.10+)
- [ ] Use `X | None` instead of `Optional[X]`
- [ ] Create `TypeAlias` for complex types
- [ ] Use `Protocol` for duck typing
- [ ] Use `Literal` for string/enum constraints
- [ ] Add `from __future__ import annotations` for compatibility
- [ ] Configure tools (mypy, ruff) for modern syntax

### **🧪 Scientific Computing Specific**
- [ ] Use simple `NDArray[np.float64]` for NumPy arrays
- [ ] Create domain-specific type aliases (`SolutionArray`, `GridPoints`)
- [ ] Use runtime shape validation instead of complex type annotations
- [ ] Layer APIs: Simple → Clean Objects → Advanced Hooks
- [ ] Allow mathematical variable names in linter config

### **⚡ Performance Tips**
- [ ] Use `ruff` instead of `black + isort + flake8` for speed
- [ ] Configure `mypy` with `--cache-dir` for faster repeat checks
- [ ] Use `pre-commit` for automated type checking

---

## 🎯 **Key Takeaways**

1. **Simplicity**: Modern typing is cleaner with fewer imports
2. **Clarity**: Use `TypeAlias` for complex types - readability matters
3. **Layering**: Progressive disclosure for different user sophistication levels
4. **Compatibility**: `__future__` imports enable modern syntax on older Python
5. **Tooling**: Automate modernization with `pyupgrade` and `ruff`
6. **Scientific Context**: Balance type safety with mathematical notation flexibility

**The goal is type safety that enhances rather than hinders scientific computing productivity.**

---

**Last Updated**: 2025-09-20
**Python Version**: 3.9+ (with 3.10+ union syntax notes)
**Status**: ✅ Production-ready guide for modern scientific Python typing
