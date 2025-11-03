# Hamiltonian Abstraction Design

**Date**: 2025-11-03
**Status**: Design proposal - not yet implemented
**Goal**: Unified abstract interface for all Hamiltonian types

---

## Current State

### **Three Different Hamiltonian Representations**

1. **Problem-level method** (`MFGProblem.H()`) - Grid-based evaluation
2. **Custom functions** (`MFGComponents.hamiltonian_func`) - User-defined callables
3. **Default implementation** - Built-in quadratic Hamiltonian

**Issue**: No unified abstraction - different interfaces, hard to compose

---

## Proposed Design

### **Base Hamiltonian Abstract Class**

```python
# mfg_pde/core/hamiltonian.py (PROPOSED)

from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

class BaseHamiltonian(ABC):
    """
    Abstract base class for Hamiltonian functions in MFG systems.

    Standard signature: H(x, m, p, t) where:
    - x: spatial position (scalar or array)
    - m: density at position
    - p: momentum/gradient (∇u)
    - t: time

    All Hamiltonian implementations must provide evaluation and
    optionally derivatives for advanced solvers.
    """

    @abstractmethod
    def evaluate(
        self,
        x: float | NDArray,
        m: float | NDArray,
        p: float | NDArray,
        t: float = 0.0
    ) -> float | NDArray:
        """
        Evaluate Hamiltonian H(x, m, p, t).

        Args:
            x: Spatial position(s)
            m: Density value(s)
            p: Momentum/gradient value(s)
            t: Time (default 0.0)

        Returns:
            Hamiltonian value(s)
        """
        pass

    def derivative_m(
        self,
        x: float | NDArray,
        m: float | NDArray,
        p: float | NDArray,
        t: float = 0.0
    ) -> float | NDArray:
        """
        Derivative with respect to density: ∂H/∂m.

        Default: Numerical differentiation (can be overridden for efficiency)

        Args:
            x: Spatial position(s)
            m: Density value(s)
            p: Momentum/gradient value(s)
            t: Time

        Returns:
            ∂H/∂m value(s)
        """
        # Default: numerical differentiation
        eps = 1e-8
        H_plus = self.evaluate(x, m + eps, p, t)
        H_minus = self.evaluate(x, m - eps, p, t)
        return (H_plus - H_minus) / (2 * eps)

    def derivative_p(
        self,
        x: float | NDArray,
        m: float | NDArray,
        p: float | NDArray,
        t: float = 0.0
    ) -> float | NDArray:
        """
        Derivative with respect to momentum: ∂H/∂p.

        Default: Numerical differentiation (can be overridden for efficiency)

        Args:
            x: Spatial position(s)
            m: Density value(s)
            p: Momentum/gradient value(s)
            t: Time

        Returns:
            ∂H/∂p value(s)
        """
        # Default: numerical differentiation
        eps = 1e-8
        H_plus = self.evaluate(x, m, p + eps, t)
        H_minus = self.evaluate(x, m, p - eps, t)
        return (H_plus - H_minus) / (2 * eps)

    def __call__(self, x, m, p, t=0.0):
        """Allow H(x, m, p, t) syntax."""
        return self.evaluate(x, m, p, t)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
```

---

## Concrete Implementations

### **1. Quadratic Hamiltonian** (Default)

```python
class QuadraticHamiltonian(BaseHamiltonian):
    """
    Standard quadratic Hamiltonian: H = (1/2)|p|² + λm

    Common in:
    - LQ games
    - Crowd motion (with congestion)
    - Traffic flow
    """

    def __init__(self, congestion_coeff: float = 0.0):
        """
        Args:
            congestion_coeff: Coefficient λ for congestion term (default 0)
        """
        self.congestion_coeff = congestion_coeff

    def evaluate(self, x, m, p, t=0.0):
        return 0.5 * p**2 + self.congestion_coeff * m

    def derivative_m(self, x, m, p, t=0.0):
        return self.congestion_coeff

    def derivative_p(self, x, m, p, t=0.0):
        return p

    def __repr__(self):
        return f"QuadraticHamiltonian(λ={self.congestion_coeff})"
```

### **2. Power-Law Hamiltonian**

```python
class PowerLawHamiltonian(BaseHamiltonian):
    """
    Power-law Hamiltonian: H = (1/γ)|p|^γ + λm^β

    Used in:
    - Traffic flow (γ ≠ 2)
    - Crowd dynamics with different congestion models
    """

    def __init__(
        self,
        momentum_power: float = 2.0,
        congestion_power: float = 1.0,
        congestion_coeff: float = 0.0
    ):
        self.gamma = momentum_power
        self.beta = congestion_power
        self.lambda_c = congestion_coeff

    def evaluate(self, x, m, p, t=0.0):
        return (1.0 / self.gamma) * np.abs(p)**self.gamma + \
               self.lambda_c * m**self.beta

    def derivative_m(self, x, m, p, t=0.0):
        return self.lambda_c * self.beta * m**(self.beta - 1)

    def derivative_p(self, x, m, p, t=0.0):
        return np.sign(p) * np.abs(p)**(self.gamma - 1)

    def __repr__(self):
        return f"PowerLawHamiltonian(γ={self.gamma}, β={self.beta}, λ={self.lambda_c})"
```

### **3. Custom Callable Hamiltonian** (Adapter)

```python
class CallableHamiltonian(BaseHamiltonian):
    """
    Wrap arbitrary callable as Hamiltonian.

    Supports legacy signatures via HamiltonianAdapter.
    """

    def __init__(
        self,
        func: Callable,
        derivative_m_func: Callable | None = None,
        derivative_p_func: Callable | None = None,
        signature_hint: str | None = None
    ):
        """
        Args:
            func: Hamiltonian function
            derivative_m_func: Optional ∂H/∂m function (else numerical)
            derivative_p_func: Optional ∂H/∂p function (else numerical)
            signature_hint: Signature hint for HamiltonianAdapter
        """
        from mfg_pde.utils.hamiltonian_adapter import HamiltonianAdapter

        self.adapter = HamiltonianAdapter(func, signature_hint=signature_hint)
        self._derivative_m_func = derivative_m_func
        self._derivative_p_func = derivative_p_func

    def evaluate(self, x, m, p, t=0.0):
        return self.adapter(x, m, p, t)

    def derivative_m(self, x, m, p, t=0.0):
        if self._derivative_m_func is not None:
            return self._derivative_m_func(x, m, p, t)
        return super().derivative_m(x, m, p, t)

    def derivative_p(self, x, m, p, t=0.0):
        if self._derivative_p_func is not None:
            return self._derivative_p_func(x, m, p, t)
        return super().derivative_p(x, m, p, t)

    def __repr__(self):
        return f"CallableHamiltonian({self.adapter.func.__name__})"
```

### **4. Composite Hamiltonian**

```python
class CompositeHamiltonian(BaseHamiltonian):
    """
    Sum of multiple Hamiltonians: H = H₁ + H₂ + ... + Hₙ

    Useful for:
    - Decomposing complex physics
    - Adding perturbations
    - Combining kinetic + potential + coupling terms
    """

    def __init__(self, *hamiltonians: BaseHamiltonian):
        """
        Args:
            *hamiltonians: Hamiltonian components to sum
        """
        if not hamiltonians:
            raise ValueError("Must provide at least one Hamiltonian")
        self.hamiltonians = hamiltonians

    def evaluate(self, x, m, p, t=0.0):
        return sum(H.evaluate(x, m, p, t) for H in self.hamiltonians)

    def derivative_m(self, x, m, p, t=0.0):
        return sum(H.derivative_m(x, m, p, t) for H in self.hamiltonians)

    def derivative_p(self, x, m, p, t=0.0):
        return sum(H.derivative_p(x, m, p, t) for H in self.hamiltonians)

    def __repr__(self):
        components = " + ".join(repr(H) for H in self.hamiltonians)
        return f"CompositeHamiltonian({components})"
```

---

## Integration with MFGProblem

### **Updated MFGComponents**

```python
@dataclass
class MFGComponents:
    """Updated to use Hamiltonian abstraction."""

    # NEW: Accept Hamiltonian object OR legacy function
    hamiltonian: BaseHamiltonian | Callable | None = None

    # DEPRECATED: Legacy function interface (backward compatibility)
    hamiltonian_func: Callable | None = None
    hamiltonian_dm_func: Callable | None = None

    # ... other components ...
```

### **Updated MFGProblem.H()**

```python
class MFGProblem:
    def __init__(self, ..., components: MFGComponents | None = None):
        # ... initialization ...

        # Set up Hamiltonian
        if components is not None:
            if isinstance(components.hamiltonian, BaseHamiltonian):
                # NEW: Use Hamiltonian object directly
                self._hamiltonian = components.hamiltonian
            elif components.hamiltonian_func is not None:
                # LEGACY: Wrap callable
                self._hamiltonian = CallableHamiltonian(
                    components.hamiltonian_func,
                    components.hamiltonian_dm_func
                )
            else:
                # Default
                self._hamiltonian = QuadraticHamiltonian(self.coefCT)
        else:
            # Default problem
            self._hamiltonian = QuadraticHamiltonian(self.coefCT)

    def H(self, x_idx, m_at_x, derivs, ...):
        """Simplified - delegates to Hamiltonian object."""
        # Extract position and momentum
        x_position = self.xSpace[x_idx]
        p = derivs.get((1,), 0.0)
        t = self.tSpace[t_idx] if t_idx is not None else 0.0

        # Delegate to Hamiltonian object
        return self._hamiltonian.evaluate(x_position, m_at_x, p, t)
```

---

## Benefits

### **1. Composability**

```python
# Build complex Hamiltonians from simple pieces
kinetic = QuadraticHamiltonian(congestion_coeff=0.0)
congestion = PowerLawHamiltonian(momentum_power=2.0, congestion_power=2.0, congestion_coeff=5.0)

# Composite: H = (1/2)p² + 5m²
hamiltonian = CompositeHamiltonian(kinetic, congestion)
```

### **2. Reusability**

```python
# Define once, use in multiple problems
crowd_hamiltonian = PowerLawHamiltonian(momentum_power=2.0, congestion_power=2.0, congestion_coeff=10.0)

problem1 = MFGProblem(Nx=50, Nt=20, components=MFGComponents(hamiltonian=crowd_hamiltonian))
problem2 = MFGProblem(Nx=100, Nt=40, components=MFGComponents(hamiltonian=crowd_hamiltonian))
```

### **3. Type Safety**

```python
# Type checker knows it's a Hamiltonian
def analyze_hamiltonian(H: BaseHamiltonian):
    # Can call evaluate(), derivative_m(), etc.
    pass
```

### **4. Extensibility**

```python
# Users can create custom Hamiltonians
class MyCustomHamiltonian(BaseHamiltonian):
    def evaluate(self, x, m, p, t=0.0):
        # Custom physics
        return ...
```

---

## Migration Path

### **Phase 1: Implement Base Classes** (No breaking changes)
- Add `mfg_pde/core/hamiltonian.py` with base class
- Implement concrete classes (Quadratic, PowerLaw, Callable, Composite)
- Keep all existing interfaces working

### **Phase 2: Update MFGComponents** (Backward compatible)
- Add `hamiltonian: BaseHamiltonian` field
- Keep `hamiltonian_func` for backward compatibility
- Auto-wrap callables in `CallableHamiltonian`

### **Phase 3: Update Examples** (Demonstrate new API)
- Show Hamiltonian object usage in examples
- Add examples with composite Hamiltonians
- Document migration for users

### **Phase 4: Deprecation** (Future)
- Mark `hamiltonian_func` as deprecated
- Provide migration guide
- Eventually remove (major version bump)

---

## Relationship to Domain Templates

Domain templates can provide **pre-configured Hamiltonian objects**:

```python
# Domain template provides Hamiltonian with domain knowledge
def create_crowd_motion_solver(problem, congestion_level="medium"):
    # Map domain terminology to Hamiltonian parameters
    congestion_map = {"low": 2.0, "medium": 10.0, "high": 20.0}
    λ = congestion_map[congestion_level]

    # Create domain-specific Hamiltonian
    hamiltonian = PowerLawHamiltonian(
        momentum_power=2.0,
        congestion_power=2.0,
        congestion_coeff=λ
    )

    # Create problem with Hamiltonian
    components = MFGComponents(hamiltonian=hamiltonian)
    crowd_problem = MFGProblem(..., components=components)

    # Compose solver
    return FixedPointIterator(crowd_problem, ...)
```

This way templates **encode Hamiltonian knowledge** specific to each domain.

---

## Implementation Priority

**Priority**: Medium (after domain template design, before template implementation)

**Dependencies**:
- None (can be implemented independently)

**Enables**:
- Domain templates with reusable Hamiltonians
- Better Hamiltonian testing and validation
- User-defined Hamiltonian libraries
- Hamiltonian composition and decomposition

---

**Last Updated**: 2025-11-03
**Status**: Design proposal - implementation recommended before domain templates
