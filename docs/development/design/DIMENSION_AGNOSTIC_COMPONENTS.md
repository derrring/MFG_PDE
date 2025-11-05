# Dimension-Agnostic MFGComponents Design

**Date**: 2025-11-03
**Issue**: MFGComponents and Hamiltonian interfaces have 1D assumptions
**Goal**: Make environment configuration truly dimension-agnostic

---

## Current State Analysis

### **MFGProblem Dimensionality Support** ✅

MFGProblem **already supports** arbitrary dimensions:

```python
# 1D
problem = MFGProblem(xmin=0, xmax=1, Nx=100)

# 2D
problem = MFGProblem(spatial_bounds=[(0,1), (0,1)], spatial_discretization=[50, 50])

# 3D
problem = MFGProblem(spatial_bounds=[(0,1), (0,1), (0,1)], spatial_discretization=[30, 30, 30])
```

**Attributes**:
- `problem.dimension` - spatial dimension
- `problem.spatial_shape` - tuple of grid sizes per dimension
- `problem._grid` - TensorProductGrid for n-D (n ≥ 2)

### **MFGComponents** ✅

MFGComponents dataclass is **already dimension-agnostic**:

```python
@dataclass
class MFGComponents:
    hamiltonian_func: Callable | None = None  # No dimension restriction
    potential_func: Callable | None = None
    # ... all fields are dimension-agnostic
```

**No issue here** - it's just a container.

### **Problem: Hamiltonian Interface** ❌

The `MFGProblem.H()` method has **1D assumptions**:

```python
def H(
    self,
    x_idx: int,              # ❌ Single index - 1D specific
    m_at_x: float,           # ✅ OK - density is scalar
    derivs: dict[tuple, float] | None = None,  # ✅ OK - dimension-aware
    p_values: dict[str, float] | None = None,  # ❌ Legacy 1D
    t_idx: int | None = None,
    x_position: float | None = None,  # ❌ Scalar - should be array
    current_time: float | None = None,
) -> float:
```

**Issues**:
1. `x_idx: int` - In n-D, need tuple of indices `(i, j, k, ...)`
2. `x_position: float` - In n-D, need array `[x, y, z, ...]`
3. `p_values: dict[str, float]` - 1D-specific legacy format
4. Line 775: `self.xSpace[x_idx]` - Only works for 1D

---

## Proposed Solution

### **Option 1: Multi-Signature Overload** (Backward Compatible)

Keep current 1D signature, add n-D variant:

```python
from typing import overload

class MFGProblem:
    # 1D signature (backward compatible)
    @overload
    def H(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float],
        t_idx: int | None = None,
        x_position: float | None = None,
        current_time: float | None = None,
    ) -> float: ...

    # N-D signature (new)
    @overload
    def H(
        self,
        x_idx: tuple[int, ...],
        m_at_x: float,
        derivs: dict[tuple, float],
        t_idx: int | None = None,
        x_position: NDArray[np.float64] | None = None,
        current_time: float | None = None,
    ) -> float: ...

    def H(
        self,
        x_idx: int | tuple[int, ...],
        m_at_x: float,
        derivs: dict[tuple, float] | None = None,
        p_values: dict[str, float] | None = None,  # Legacy
        t_idx: int | None = None,
        x_position: float | NDArray[np.float64] | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Hamiltonian function H(x, m, ∇u, t) - dimension-agnostic.

        Args:
            x_idx: Grid index
                   - 1D: int (0 to Nx)
                   - N-D: tuple (i, j, k, ...)
            m_at_x: Density at grid point
            derivs: Derivatives in tuple notation (dimension-agnostic):
                    - 1D: {(0,): u, (1,): du/dx}
                    - 2D: {(0,0): u, (1,0): du/dx, (0,1): du/dy}
                    - 3D: {(0,0,0): u, (1,0,0): du/dx, (0,1,0): du/dy, (0,0,1): du/dz}
            x_position: Spatial position
                        - 1D: float
                        - N-D: array [x, y, z, ...]
        """
        # Auto-detect dimension from x_idx type
        if isinstance(x_idx, int):
            # 1D mode
            if x_position is None:
                x_position = self.xSpace[x_idx]
        else:
            # N-D mode
            if x_position is None:
                x_position = self._grid.get_point(x_idx)

        # Rest of implementation...
```

**Pros**:
- ✅ 100% backward compatible
- ✅ Type-safe for both 1D and n-D
- ✅ Auto-detects dimension from input type

**Cons**:
- Complexity in implementation
- Two code paths to maintain

### **Option 2: Unified Array-Based Interface** (Clean, Modern)

Use arrays for all dimensions (including 1D):

```python
def H(
    self,
    x_idx: tuple[int, ...],  # Always tuple, even for 1D: (i,)
    m_at_x: float,
    derivs: dict[tuple, float],
    t_idx: int | None = None,
    x_position: NDArray[np.float64] | None = None,  # Always array
    current_time: float | None = None,
) -> float:
    """
    Hamiltonian function H(x, m, ∇u, t) - dimension-agnostic.

    Args:
        x_idx: Grid index tuple (i,) for 1D, (i,j) for 2D, etc.
        m_at_x: Density at grid point
        derivs: Derivatives in tuple notation
        x_position: Spatial position array (1D array even for 1D problems)
    """
    # Compute position if not provided
    if x_position is None:
        if self.dimension == 1:
            x_position = np.array([self.xSpace[x_idx[0]]])
        else:
            x_position = self._grid.get_point(x_idx)

    # Use custom Hamiltonian if provided
    if self.is_custom and self.components is not None:
        if self.components.hamiltonian_func is not None:
            return self.components.hamiltonian_func(
                x_idx=x_idx,
                x_position=x_position,
                m_at_x=m_at_x,
                derivs=derivs,
                t_idx=t_idx,
                current_time=current_time,
                problem=self
            )

    # Default Hamiltonian (dimension-agnostic)
    return self._default_hamiltonian(derivs, m_at_x)

def _default_hamiltonian(self, derivs: dict[tuple, float], m: float) -> float:
    """
    Default quadratic Hamiltonian: H = (1/2)||∇u||² + λm

    Works for any dimension by computing gradient magnitude.
    """
    # Extract all first derivatives
    grad_components = []
    for key, value in derivs.items():
        if sum(key) == 1:  # First derivative
            grad_components.append(value)

    # Compute ||∇u||²
    grad_norm_sq = sum(p**2 for p in grad_components)

    return 0.5 * grad_norm_sq + self.coupling_coefficient * m
```

**Pros**:
- ✅ Clean, uniform interface
- ✅ Truly dimension-agnostic
- ✅ Easier to reason about

**Cons**:
- ❌ Breaking change for 1D users (need `(i,)` instead of `i`)
- Migration needed

### **Option 3: Adapter Pattern** (Best of Both Worlds)

Keep both interfaces, provide adapters:

```python
class MFGProblem:
    def H(
        self,
        x_idx: int | tuple[int, ...],
        m_at_x: float,
        derivs: dict[tuple, float] | None = None,
        **kwargs
    ) -> float:
        """Auto-adapting Hamiltonian interface."""
        # Normalize inputs
        if isinstance(x_idx, int):
            x_idx = (x_idx,)  # Convert to tuple

        x_position = kwargs.get('x_position')
        if x_position is None:
            x_position = self._get_position(x_idx)
        elif isinstance(x_position, (int, float)):
            x_position = np.array([x_position])  # Convert to array

        # Call dimension-agnostic implementation
        return self._H_impl(x_idx, m_at_x, derivs, x_position, **kwargs)

    def _get_position(self, x_idx: tuple[int, ...]) -> NDArray:
        """Get spatial position from index tuple."""
        if self.dimension == 1:
            return np.array([self.xSpace[x_idx[0]]])
        else:
            return self._grid.get_point(x_idx)

    def _H_impl(
        self,
        x_idx: tuple[int, ...],
        m_at_x: float,
        derivs: dict[tuple, float],
        x_position: NDArray,
        **kwargs
    ) -> float:
        """Internal dimension-agnostic Hamiltonian."""
        # Uniform implementation for all dimensions
        ...
```

**Pros**:
- ✅ Backward compatible (accepts both int and tuple)
- ✅ Clean internal implementation
- ✅ Auto-adapts legacy code

**Cons**:
- Slight runtime overhead for type checking

---

## Custom Hamiltonian Interface

### **Current Custom Hamiltonian Signature** ❌

```python
def my_hamiltonian(
    x_idx: int,              # 1D-specific
    x_position: float,       # 1D-specific
    m_at_x: float,
    derivs: dict[tuple, float],
    t_idx: int | None,
    current_time: float | None,
    problem: MFGProblem
) -> float:
    # User implementation
```

### **Proposed: Dimension-Agnostic Signature** ✅

```python
def my_hamiltonian(
    x_idx: tuple[int, ...],        # Works for any dimension
    x_position: NDArray,            # Position vector
    m_at_x: float,                  # Density (scalar)
    derivs: dict[tuple, float],     # Dimension-agnostic gradients
    t_idx: int | None,
    current_time: float | None,
    problem: MFGProblem
) -> float:
    """
    Custom Hamiltonian for arbitrary dimensions.

    Example for 2D:
        x_idx = (i, j)
        x_position = array([x, y])
        derivs = {(0,0): u, (1,0): du/dx, (0,1): du/dy}
    """
    # Extract dimension from problem
    dim = problem.dimension

    # Extract position components
    if dim == 1:
        x = x_position[0]
    elif dim == 2:
        x, y = x_position[0], x_position[1]
    elif dim == 3:
        x, y, z = x_position

    # Extract gradient components
    grad = [derivs.get(tuple(1 if i == j else 0 for i in range(dim)), 0.0)
            for j in range(dim)]

    # Compute Hamiltonian (dimension-agnostic)
    grad_norm_sq = sum(p**2 for p in grad)
    return 0.5 * grad_norm_sq + congestion_coeff * m_at_x**2
```

---

## Proposed Hamiltonian Abstraction (Revisited)

With dimension-agnostic interface, the base class becomes:

```python
class BaseHamiltonian(ABC):
    """Dimension-agnostic Hamiltonian base class."""

    @abstractmethod
    def evaluate(
        self,
        x: NDArray,              # Position vector (any dimension)
        m: float,                # Density (scalar)
        p: NDArray,              # Gradient vector (any dimension)
        t: float = 0.0
    ) -> float:
        """
        Evaluate H(x, m, ∇u, t).

        Args:
            x: Position array, shape (d,) for d-dimensional problem
            m: Density (scalar)
            p: Gradient array, shape (d,)
            t: Time (scalar)

        Returns:
            Hamiltonian value (scalar)
        """
        pass

    def derivative_m(self, x, m, p, t=0.0) -> float:
        """∂H/∂m - same signature."""
        eps = 1e-8
        return (self.evaluate(x, m + eps, p, t) -
                self.evaluate(x, m - eps, p, t)) / (2 * eps)

    def derivative_p(self, x, m, p, t=0.0) -> NDArray:
        """
        ∂H/∂p - returns gradient vector.

        Returns:
            Array of shape (d,) containing [∂H/∂p₁, ∂H/∂p₂, ..., ∂H/∂pₐ]
        """
        eps = 1e-8
        d = len(p)
        grad = np.zeros(d)
        for i in range(d):
            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            grad[i] = (self.evaluate(x, m, p_plus, t) -
                      self.evaluate(x, m, p_minus, t)) / (2 * eps)
        return grad
```

**Example: Quadratic Hamiltonian (Dimension-Agnostic)**

```python
class QuadraticHamiltonian(BaseHamiltonian):
    """H = (1/2)||p||² + λm - works for any dimension."""

    def __init__(self, congestion_coeff: float = 0.0):
        self.lambda_c = congestion_coeff

    def evaluate(self, x, m, p, t=0.0):
        return 0.5 * np.sum(p**2) + self.lambda_c * m

    def derivative_m(self, x, m, p, t=0.0):
        return self.lambda_c

    def derivative_p(self, x, m, p, t=0.0):
        return p  # ∂H/∂p = p for quadratic
```

Works automatically for 1D, 2D, 3D, ..., nD!

---

## Migration Path

### **Phase 1: Make Internal Implementation Dimension-Agnostic**
- Update `MFGProblem.H()` to normalize inputs (int → tuple)
- Update `_default_hamiltonian()` to work for any dimension
- **No breaking changes** - adapters handle legacy calls

### **Phase 2: Update Documentation**
- Show dimension-agnostic custom Hamiltonian examples
- Document signature: use tuples and arrays
- Provide migration guide for 1D-specific code

### **Phase 3: Deprecation Warnings**
- Warn when passing `int` instead of `tuple` for x_idx
- Warn when passing `float` instead of `array` for x_position
- Guide users to dimension-agnostic signatures

### **Phase 4: Implement Hamiltonian Abstraction**
- Add `BaseHamiltonian` with dimension-agnostic interface
- Implement `QuadraticHamiltonian`, `PowerLawHamiltonian`, etc.
- All work for arbitrary dimensions automatically

---

## Benefits

### **1. True Dimension Agnosticism**

```python
# Same environment works for any dimension
quadratic_env = MFGComponents(
    hamiltonian=QuadraticHamiltonian(congestion_coeff=5.0)
)

# 1D
problem_1d = MFGProblem(xmin=0, xmax=1, Nx=100, components=quadratic_env)

# 2D
problem_2d = MFGProblem(
    spatial_bounds=[(0,1), (0,1)],
    spatial_discretization=[50, 50],
    components=quadratic_env  # Same environment!
)
```

### **2. Reusable Environment Configurations**

```python
# Define once, use everywhere
crowd_environment = MFGComponents(
    hamiltonian=PowerLawHamiltonian(momentum_power=2.0, congestion_power=2.0, congestion_coeff=10.0),
    boundary_conditions=exit_boundaries
)

# Works in 1D corridor
corridor = MFGProblem(xmin=0, xmax=100, Nx=200, components=crowd_environment)

# Works in 2D room
room = MFGProblem(
    spatial_bounds=[(0, 20), (0, 20)],
    spatial_discretization=[100, 100],
    components=crowd_environment  # Same physics!
)
```

### **3. Simplified User Code**

Users don't need to think about dimension when defining environment physics:

```python
# Define environment physics (dimension-agnostic)
def my_hamiltonian(x_idx, x_position, m_at_x, derivs, **kwargs):
    dim = len(x_position)  # Auto-detect dimension
    grad = [derivs.get(tuple(1 if i==j else 0 for i in range(dim)), 0.0)
            for j in range(dim)]
    return 0.5 * sum(p**2 for p in grad) + 5.0 * m_at_x**2

# Works automatically for 1D, 2D, 3D, ...
components = MFGComponents(hamiltonian_func=my_hamiltonian)
```

---

## Recommendation

**Implement Option 3 (Adapter Pattern) with Hamiltonian Abstraction**:

1. ✅ Add input normalization in `MFGProblem.H()` (backward compatible)
2. ✅ Make internal implementation dimension-agnostic
3. ✅ Implement `BaseHamiltonian` with dimension-agnostic interface
4. ✅ Update documentation to show dimension-agnostic examples
5. ⚠️ Add deprecation warnings (future)
6. ❌ Remove 1D-specific code paths (major version)

This gives us:
- **Immediate**: Backward compatibility maintained
- **Medium-term**: Clean dimension-agnostic abstractions
- **Long-term**: Simplified, unified interface

---

**Last Updated**: 2025-11-03
**Priority**: High - fundamental to environment configuration concept
**Dependencies**: Should be done before/alongside Hamiltonian abstraction
