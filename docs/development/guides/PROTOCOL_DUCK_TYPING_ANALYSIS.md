# Protocol Duck Typing: Design Pattern Analysis

## Executive Summary

**Question:** Is protocol duck typing a proper design pattern?

**Answer:** **Yes**, it is a formal design pattern known as **Structural Subtyping**[^1]. In Python, it serves as a bridge between the dynamic flexibility of traditional "duck typing" and the rigor of static type checking. However, it is not a universal replacement for inheritance. It is best applied to define **capabilities** (what an object *can do*), whereas Abstract Base Classes (ABCs) are best for defining **identities** and sharing **implementation** (what an object *is* and *how it works*).

---

## 1. Concept Definition

### The Core Distinction

* **Nominal Subtyping (ABCs)**: Relations are explicit. `class B(A)` means B is a subtype of A because you *said* so. This is the standard OOP inheritance model[^2].
* **Structural Subtyping (Protocols)**: Relations are implicit. B is a subtype of A because B has the methods defined in A, regardless of inheritance hierarchy. This is formalized in Python via PEP 544[^1].

### Syntax Example

```python
from typing import Protocol, runtime_checkable
import numpy as np

# A Protocol defines a 'shape' or 'capability'
@runtime_checkable
class ConstraintProtocol(Protocol):
    def project(self, u: np.ndarray) -> np.ndarray: ...
    def is_feasible(self, u: np.ndarray) -> bool: ...

# A concrete class satisfies the Protocol implicitly
class ObstacleConstraint:
    def __init__(self, obstacle):
        self.obstacle = obstacle

    def project(self, u):
        return np.maximum(u, self.obstacle)

    def is_feasible(self, u):
        return np.all(u >= self.obstacle)

# Usage
# True, even though ObstacleConstraint does not inherit from ConstraintProtocol
isinstance(ObstacleConstraint(0.0), ConstraintProtocol)
```

**Key Insight**: Protocols define **capabilities**, ABCs define **identities**.

---

## 2. Strategic Analysis: When to Use Protocols

Protocols excel when you need **flexibility** and **interoperability**, particularly in library design.

### ✅ Case 1: Interface Segregation (ISP)

The **Interface Segregation Principle** states that no client should be forced to depend on methods it does not use[^3]. ABCs often encourage monolithic interfaces (God Objects), whereas Protocols encourage granular capabilities.

| Feature | Monolithic ABC (Bad) | Granular Protocols (Good) |
|:--------|:---------------------|:--------------------------|
| **Requirement** | Implement *all* 10 abstract methods. | Implement only `SupportsLaplacian`. |
| **Coupling** | High (Client depends on full hierarchy). | Low (Client depends on single method). |
| **Flexibility** | Rigid taxonomy. | Mix-and-match traits. |

**Example in MFG_PDE**:

```python
# ✅ GOOD: Focused protocols (Interface Segregation)
class SupportsLaplacian(Protocol):
    def get_laplacian_operator(self) -> LinearOperator: ...

class SupportsGradient(Protocol):
    def get_gradient_operator(self) -> tuple[LinearOperator, ...]: ...

# Different solvers need different capabilities
def solve_poisson(geometry: SupportsLaplacian, f):
    """Only needs Laplacian, not full geometry interface."""
    L = geometry.get_laplacian_operator()
    return scipy.sparse.linalg.spsolve(L, f)

# ❌ BAD: Monolithic ABC forces all implementations
class GeometryABC(ABC):
    @abstractmethod
    def get_laplacian_operator(self): ...
    @abstractmethod
    def get_gradient_operator(self): ...
    @abstractmethod
    def get_divergence_operator(self): ...
    # ... 10+ more methods that not all geometries need
```

### ✅ Case 2: Third-Party Integration

You cannot retroactively make a class from `scipy` or `numpy` inherit from your project's `GeometryABC`. Protocols solve this by typing the *structure* of these external objects.

```python
# We can't change ThirdPartyShape, but we can type-hint it
class Drawable(Protocol):
    def draw(self) -> None: ...

def render(obj: Drawable):
    obj.draw()

# Works with any object having a .draw() method, even from external libs
render(external_lib.ThirdPartyShape())
```

**Real-world use**: Accepting `scipy.sparse` matrices, `numpy` arrays, or custom user objects that happen to have the right methods.

### ✅ Case 3: Gradual Typing of Legacy Code

For mature codebases like portions of `MFG_PDE`, refactoring deep inheritance hierarchies to add type hints is risky. Protocols allow you to define types for existing behaviors without changing the runtime code structure.

```python
# Legacy code (no types, no inheritance)
class OldSolver:
    def solve(self, problem):
        return result

# Add type hints via protocol (no refactoring needed)
class SolverLike(Protocol):
    def solve(self, problem: MFGProblem) -> Result: ...

# Type check without changing OldSolver
def run_simulation(solver: SolverLike):
    return solver.solve(problem)
```

---

## 3. Strategic Analysis: When to Avoid Protocols

Protocols are poor at enforcing **behavioral guarantees** and **code reuse**.

### ❌ Case 1: DRY Violations (Shared Implementation)

Protocols cannot provide default implementations (mostly). If you find yourself copy-pasting the `preprocess()` logic into five different solvers, you need an **Abstract Base Class**, not a Protocol.

> **Rule of Thumb:** Use Protocols for *interfaces*; use ABCs for *implementations*.

**Example**:

```python
# ❌ BAD: Protocol leads to duplication
class SolverProtocol(Protocol):
    def preprocess(self): ...
    def solve(self): ...
    def postprocess(self): ...

class Solver1:
    def preprocess(self):
        # Duplicated validation/setup logic
        self.validate_inputs()
        self.setup_matrices()
    def solve(self): ...

class Solver2:
    def preprocess(self):
        # Same duplication!
        self.validate_inputs()
        self.setup_matrices()
    def solve(self): ...

# ✅ GOOD: ABC provides shared implementation
class SolverBase(ABC):
    def preprocess(self):
        # Shared implementation (DRY)
        self.validate_inputs()
        self.setup_matrices()

    @abstractmethod
    def solve(self): ...  # Only abstract what varies
```

### ❌ Case 2: Invariants and Validation

Protocols check syntax (method names/signatures), not semantics. They cannot enforce that a boundary condition preserves mass, only that it *has* an `apply` method.

```python
# Protocol cannot stop this semantic error
class BrokenBC:
    def apply(self, field):
        return field * 0  # Syntactically correct, physically wrong (mass loss)
```

Use an ABC with a `final` template method[^4] to enforce such invariants:

```python
from abc import ABC, abstractmethod

class BoundaryConditionBase(ABC):
    def apply(self, field):
        """Template method enforcing conservation invariant."""
        result = self._apply_impl(field)
        assert self._check_conservation(result), "BC violates conservation"
        return result

    @abstractmethod
    def _apply_impl(self, field):
        """Subclass implements specific BC logic."""
        ...

    def _check_conservation(self, result):
        """Enforce physical invariant."""
        return abs(result.sum() - field.sum()) < 1e-10
```

**Template Method Pattern**[^4]: Base class controls the algorithm structure, subclasses fill in the details.

### ❌ Case 3: Discovery and Registries

If you need to find all available plugins (e.g., "List all available Solvers"), Protocols are hard to use because they don't maintain a registry of implementers. ABCs can easily track subclasses via `__init_subclass__` or metaclasses.

```python
# ❌ BAD: Protocol has no discovery mechanism
class ConstraintProtocol(Protocol):
    def project(self, u): ...

# No way to enumerate all implementations!

# ✅ GOOD: ABC with automatic registration
class ConstraintBase(ABC):
    _registry: list[type] = []

    def __init_subclass__(cls):
        """Auto-register all subclasses."""
        ConstraintBase._registry.append(cls)

    @abstractmethod
    def project(self, u): ...

# Can enumerate: ConstraintBase._registry
# Useful for factory patterns, plugin systems
```

---

## 4. Hybrid Approaches (Best Practices)

The most robust architectures often combine both patterns.

### Pattern: "Protocol for API, ABC for Implementation"

This allows you to accept *any* valid object in your API (flexibility) while providing a convenient starting point for your own implementations (reuse).

```python
# 1. The Public Contract (Protocol)
# Users can implement this however they want (duck typing)
class ConstraintProtocol(Protocol):
    def project(self, u: np.ndarray) -> np.ndarray: ...
    def is_feasible(self, u: np.ndarray) -> bool: ...

# 2. The Internal Skeleton (ABC)
# Provides shared helpers for our internal standard implementations
class ConstraintBase(ABC):
    @abstractmethod
    def project(self, u: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def is_feasible(self, u: np.ndarray) -> bool: ...

    def project_if_violated(self, u):
        """Shared utility method."""
        if not self.is_feasible(u):
            return self.project(u)
        return u

# 3. User chooses: duck-type the protocol or inherit the ABC
class CustomConstraint:  # Duck types protocol
    def project(self, u): ...
    def is_feasible(self, u): ...

class ObstacleConstraint(ConstraintBase):  # Inherits ABC utilities
    def project(self, u): return np.maximum(u, self.obstacle)
    def is_feasible(self, u): return np.all(u >= self.obstacle)
```

**Benefits**:
- **Protocol**: Flexible public API, third-party friendly
- **ABC**: DRY, shared utilities, optional inheritance

---

## 5. MFG_PDE Contextual Audit

Based on the current architecture of `MFG_PDE`, here is the specific guidance:

### ✅ Approved Uses (Keep as Protocols)

1. **`ConstraintProtocol`** (`mfg_pde/geometry/boundary/constraint_protocol.py`):
   - **Reasoning**: Constraints are often mathematical functions that don't share state. Users might pass simple lambdas or lightweight classes. Explicit inheritance is unnecessary friction here.
   - **Interface**: 3 methods (minimal)
   - **Shared logic**: None (projection is specific to each constraint type)
   - **Status**: ✅ Appropriate use of Protocol

2. **`Geometry Traits`** (Issue #590 - planned):
   - **Reasoning**: `SupportsLaplacian`, `SupportsGradient`, etc. Different geometries (grids vs. graphs vs. meshes) share little implementation but share interface requirements. This is a perfect use of Interface Segregation.
   - **Use case**: Poisson solver only needs `SupportsLaplacian`, not full geometry
   - **Status**: ✅ Appropriate use of Protocol

3. **`BoundaryHandler`** (`mfg_pde/geometry/boundary/handler_protocol.py`):
   - **Reasoning**: Minimal solver integration interface
   - **Interface**: 2-3 methods
   - **Status**: ✅ Appropriate (if kept minimal)

### ⚠️ Refactor Recommendations (Switch to ABC)

1. **`BCApplicatorProtocol`** (`mfg_pde/geometry/boundary/applicator_base.py`):
   - **Issue**: This component likely requires shared logic for ghost cell indexing and input validation.
   - **Current**: ~10+ methods, shared ghost cell computation logic
   - **Action**: Convert to `BCApplicatorBase(ABC)`. Use the **Template Method Pattern**[^4] to handle the heavy lifting of ghost-cell iteration in the base class, requiring subclasses only to implement the local value calculation.
   - **Benefit**: Eliminate duplicated ghost cell logic across FDM/FEM/GFDM applicators

**Proposed Refactoring**:

```python
# Current (Protocol - broad interface, no shared code)
class BCApplicatorProtocol(Protocol):
    def apply(self, field, bc, **kwargs): ...
    def validate_bc(self, bc): ...
    def create_ghost_buffer(self, field, bc): ...
    # ... many more methods

# Proposed (ABC - shared ghost cell logic)
class BCApplicatorBase(ABC):
    """Base class for boundary condition application."""

    def apply(self, field, bc, **kwargs):
        """Template method with shared validation/iteration logic."""
        # Shared validation
        self._validate_inputs(field, bc)

        # Shared ghost cell buffer creation
        ghost_buffer = self._create_ghost_buffer(field, bc)

        # Subclass-specific computation
        return self._compute_bc_values(field, bc, ghost_buffer)

    @abstractmethod
    def _compute_bc_values(self, field, bc, ghost_buffer):
        """Subclass implements BC-specific computation."""
        ...

    def _validate_inputs(self, field, bc):
        """Shared validation logic."""
        if field.ndim != self.expected_ndim:
            raise ValueError(f"Expected {self.expected_ndim}D field")

    def _create_ghost_buffer(self, field, bc):
        """Shared ghost cell indexing (complex logic, DRY)."""
        # ... shared implementation
```

---

## 6. Conclusion & Decision Matrix

To decide between a Protocol and an ABC, ask these three questions:

### Decision Framework

**Question 1**: Do I need to share code/logic between implementations?
- **Yes** → **ABC** (Inheritance for code reuse)
- **No** → Protocol

**Question 2**: Is this for a plugin system where I need to discover all types?
- **Yes** → **ABC** (Registration via `__init_subclass__`)
- **No** → Protocol

**Question 3**: Will I accept objects from third-party libraries (e.g., NumPy)?
- **Yes** → **Protocol** (Duck typing, no inheritance required)
- **No** → ABC

**Question 4**: Do I need to enforce behavioral invariants (e.g., conservation laws)?
- **Yes** → **ABC** (Template method with validation)
- **No** → Protocol

**Question 5**: Is the interface minimal (< 5 methods)?
- **Yes** → Protocol (likely appropriate)
- **No** → Consider ABC (broad interfaces often have shared logic)

### Summary Table

| Criterion | Protocol | ABC |
|:----------|:---------|:----|
| **Purpose** | Define capabilities | Define identity + provide implementation |
| **Coupling** | Minimal (structural) | Explicit (nominal) |
| **Code Reuse** | None | High (inheritance) |
| **Third-party** | ✅ Works | ❌ Requires inheritance |
| **Discovery** | ❌ No registry | ✅ `__init_subclass__` |
| **Invariants** | ❌ No enforcement | ✅ Template method |
| **Best for** | Small, focused interfaces | Shared implementation, frameworks |

---

## 7. Final Verdict

**Protocol Duck Typing is a legitimate and powerful design pattern for Python**, provided it is used to define **structural interfaces** rather than to organize **implementation hierarchies**.

**Key Principles**:
1. **Protocols** define **what** (capabilities, structure)
2. **ABCs** define **how** (behavior, implementation, identity)
3. **Hybrid approach** often optimal: Protocol for public API, ABC for internal utilities

**MFG_PDE Recommendation**:
- ✅ Keep: `ConstraintProtocol`, geometry traits, minimal handler interfaces
- ⚠️ Refactor: `BCApplicatorProtocol` → `BCApplicatorBase(ABC)` with template method
- 📋 Guideline: Default to Protocol for < 5 methods with no shared logic, ABC otherwise

---

## References

[^1]: Python Software Foundation. (2017). *PEP 544 – Protocols: Structural subtyping (static duck typing)*. Retrieved from https://peps.python.org/pep-0544/

[^2]: Python Software Foundation. (2007). *PEP 3119 – Introducing Abstract Base Classes*. Retrieved from https://peps.python.org/pep-3119/

[^3]: Martin, R. C. (2002). *Agile Software Development, Principles, Patterns, and Practices*. Prentice Hall. (Chapter on Interface Segregation Principle).

[^4]: Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. (See: Template Method Pattern).

---

**Document Metadata**:
- **Last Updated**: 2026-01-17
- **Context**: Design review during Issue #591 (Variational Inequality Constraints)
- **Status**: Authoritative architectural reference for MFG_PDE
- **Format**: Typora/Obsidian compatible with proper footnotes
