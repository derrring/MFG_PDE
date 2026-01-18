# Guide: Geometry Trait Protocols for Solver Developers

**Audience**: Solver developers, advanced users extending MFG_PDE
**Prerequisites**: Python protocols, linear algebra, numerical PDEs
**Version**: 1.0 (2026-01-18)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Trait Protocols?](#why-trait-protocols)
3. [Core Trait Protocols](#core-trait-protocols)
4. [Tutorial: Writing a Dimension-Agnostic Solver](#tutorial-writing-a-dimension-agnostic-solver)
5. [Operator Patterns](#operator-patterns)
6. [Best Practices](#best-practices)
7. [Advanced Topics](#advanced-topics)

---

## Introduction

MFG_PDE uses **trait protocols** to decouple solvers from geometry implementations. Solvers request **capabilities** (e.g., "can compute Laplacian"), and geometries provide **operators** (e.g., finite difference Laplacian matrix).

### What Are Trait Protocols?

Trait protocols are Python `Protocol` classes that define **interfaces** without inheritance:

```python
from typing import Protocol

class SupportsLaplacian(Protocol):
    """Geometry can compute Laplacian operator."""
    def get_laplacian_operator(self, order: int = 2, bc=None):
        ...
```

**Any geometry** implementing this method automatically satisfies `SupportsLaplacian`, no base class needed.

### Key Benefits

✅ **Dimension-agnostic solvers**: Write once, works in 1D, 2D, 3D
✅ **Discretization-agnostic**: FDM, FEM, GFDM, meshfree all compatible
✅ **Testable**: Mock geometries for unit tests
✅ **Clear contracts**: Explicit capabilities ("supports gradient" vs guessing)

---

## Why Trait Protocols?

### The Old Way: Duck Typing with hasattr()

**Anti-pattern** (pre-Issue #590):
```python
# Bad: Guessing capabilities
def solve_heat_equation(u0, geometry, dt, steps):
    if hasattr(geometry, "get_laplacian_operator"):
        laplacian = geometry.get_laplacian_operator()
    elif hasattr(geometry, "laplacian_matrix"):
        laplacian = geometry.laplacian_matrix
    elif hasattr(geometry, "get_laplace_operator"):  # Typo variation!
        laplacian = geometry.get_laplace_operator()
    else:
        raise TypeError("Geometry doesn't support Laplacian!")

    # ... solver logic ...
```

**Problems**:
- ❌ Fragile to typos and naming variations
- ❌ No static type checking
- ❌ Unclear what capabilities geometry has
- ❌ Runtime errors instead of design-time errors

### The New Way: Explicit Trait Protocols

**Correct pattern** (Issue #590+):
```python
from mfg_pde.geometry.protocols import SupportsLaplacian

def solve_heat_equation(
    u0: NDArray,
    geometry: SupportsLaplacian,  # Type hint: requires Laplacian
    dt: float,
    steps: int,
) -> NDArray:
    """
    Solve heat equation ∂u/∂t = α·∇²u.

    Geometry must implement SupportsLaplacian trait.
    """
    laplacian = geometry.get_laplacian_operator()  # Guaranteed to exist
    u = u0.copy()

    for _ in range(steps):
        u = u + dt * laplacian @ u.ravel()  # Forward Euler

    return u
```

**Benefits**:
- ✅ Type checker validates geometry capability **before runtime**
- ✅ IDE autocomplete knows `get_laplacian_operator()` exists
- ✅ Clear documentation: "requires SupportsLaplacian trait"
- ✅ No hasattr() checks needed

**★ Insight ─────────────────────────────────────**
Trait protocols move **capability checking from runtime to design time**. If a geometry doesn't support Laplacian, mypy/pyright will catch it when you write the code, not when a user runs it 3 hours into a simulation.
**─────────────────────────────────────────────────**

---

## Core Trait Protocols

MFG_PDE provides the following trait protocols for geometry capabilities:

### 1. SupportsLaplacian

**Purpose**: Geometry can compute Laplacian $\nabla^2 u = \sum_i \partial^2 u / \partial x_i^2$.

**Method Signature**:
```python
def get_laplacian_operator(
    self,
    order: int = 2,
    bc: BoundaryConditions | None = None,
) -> LinearOperator | Callable[[NDArray], NDArray]:
    ...
```

**Parameters**:
- `order`: Discretization order (2 = 2nd-order, 4 = 4th-order, etc.)
- `bc`: Boundary conditions to incorporate (Dirichlet, Neumann, Robin)

**Returns**: Operator $L$ where `L @ u` or `L(u)` computes $\nabla^2 u$

**Implementations**:
- `TensorProductGrid`: Finite difference Laplacian (5-point stencil in 2D, 7-point in 3D)
- `UnstructuredMesh`: FEM stiffness matrix
- `GraphGeometry`: Graph Laplacian matrix
- `ImplicitDomain`: Meshfree Laplacian (GFDM or RBF)

**Example**:
```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import neumann_bc

grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[51, 51])
bc = neumann_bc(dimension=2)

# Get Laplacian operator
laplacian = grid.get_laplacian_operator(order=2, bc=bc)

# Apply to a field
import numpy as np
u = np.random.rand(51, 51)
Lu = laplacian @ u.ravel()  # Shape: (51*51,)
Lu_reshaped = Lu.reshape(51, 51)  # Back to grid shape
```

### 2. SupportsGradient

**Purpose**: Geometry can compute spatial gradients $\nabla u = (\partial u / \partial x_1, \partial u / \partial x_2, ...)$.

**Method Signature**:
```python
def get_gradient_operator(
    self,
    direction: int | None = None,
    order: int = 2,
    scheme: Literal["centered", "forward", "backward", "upwind"] = "centered",
) -> LinearOperator | Callable | tuple[LinearOperator | Callable, ...]:
    ...
```

**Parameters**:
- `direction`: Specific direction (0=x, 1=y, 2=z). If `None`, return all directions.
- `order`: Discretization order (2, 4, 6, ...)
- `scheme`:
  - `"centered"`: Centered differences (most accurate, requires interior points)
  - `"forward"`: Forward differences (for right boundary or upwind with $v > 0$)
  - `"backward"`: Backward differences (for left boundary or upwind with $v < 0$)
  - `"upwind"`: Godunov upwind (for hyperbolic PDEs, Level Set)

**Returns**:
- If `direction=None`: Tuple `(∂/∂x, ∂/∂y, ∂/∂z, ...)` of operators
- If `direction` specified: Single operator for `∂/∂xᵢ`

**Example**:
```python
grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[51, 51])

# Get all gradient directions
grad_x, grad_y = grid.get_gradient_operator(scheme="centered", order=2)

# Compute gradient
u = np.random.rand(51, 51)
ux = grad_x @ u.ravel()  # ∂u/∂x
uy = grad_y @ u.ravel()  # ∂u/∂y

# Gradient magnitude for HJB Hamiltonian
grad_mag = np.sqrt(ux**2 + uy**2)  # |∇u|
```

### 3. SupportsDivergence

**Purpose**: Geometry can compute divergence $\text{div}(v) = \sum_i \partial v_i / \partial x_i$.

**Method Signature**:
```python
def get_divergence_operator(
    self,
    order: int = 2,
) -> LinearOperator | Callable[[NDArray], NDArray]:
    ...
```

**Parameters**:
- `order`: Discretization order (2, 4, 6, ...)

**Returns**: Operator that computes `div(v)` from vector field `v` of shape `(dimension, num_points)`

**Example**:
```python
grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[51, 51])
div_op = grid.get_divergence_operator(order=2)

# Vector field (e.g., velocity or flux)
v = np.random.rand(2, 51*51)  # (vx, vy) at each point
div_v = div_op @ v.ravel()  # Flatten (2, N) → (2*N,) before applying

# Or use callable form
div_v = div_op(v)  # Some operators accept (dimension, N) directly
```

**Use Cases**:
- Fokker-Planck equation: $\partial m/\partial t + \text{div}(m \cdot \alpha) = 0$
- Curvature computation: $\kappa = \nabla \cdot (\nabla\phi / |\nabla\phi|)$
- Mass conservation checks

### 4. SupportsAdvection

**Purpose**: Geometry can compute advection term $v \cdot \nabla u$ or $\text{div}(u \cdot v)$.

**Method Signature**:
```python
def get_advection_operator(
    self,
    velocity_field: NDArray,
    scheme: Literal["upwind", "centered", "weno", "lax_friedrichs"] = "upwind",
    conservative: bool = True,
) -> LinearOperator | Callable[[NDArray], NDArray]:
    ...
```

**Parameters**:
- `velocity_field`: Drift/velocity field, shape `(dimension, num_points)` or `(num_points,)` for 1D
- `scheme`:
  - `"upwind"`: 1st-order upwind (stable, dissipative)
  - `"centered"`: Centered differences (2nd-order, may oscillate)
  - `"weno"`: Weighted ENO (high-order, for shocks)
  - `"lax_friedrichs"`: Lax-Friedrichs flux (stable, diffusive)
- `conservative`: If `True`, compute $\text{div}(u \cdot v)$; if `False`, compute $v \cdot \nabla u$

**Returns**: Operator that applies advection to a field

**Example**:
```python
grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[51, 51])

# Optimal control drift: α = -∇u
grad_x, grad_y = grid.get_gradient_operator()
u = np.random.rand(51*51)
alpha = -np.stack([grad_x @ u, grad_y @ u], axis=0)  # Shape: (2, 51*51)

# Get advection operator (conservative form for FP equation)
adv_op = grid.get_advection_operator(
    velocity_field=alpha,
    scheme="upwind",
    conservative=True,
)

# Apply to density
m = np.random.rand(51*51)
div_m_alpha = adv_op @ m  # Transport term in FP equation
```

### 5. SupportsInterpolation

**Purpose**: Geometry can interpolate field values at arbitrary (non-grid) points.

**Method Signature**:
```python
def interpolate(
    self,
    field: NDArray,
    points: NDArray,
    method: Literal["linear", "cubic", "rbf"] = "linear",
) -> NDArray:
    ...
```

**Use Cases**:
- Semi-Lagrangian solvers (evaluate at characteristic foot points)
- Particle-in-cell methods
- Visualization at arbitrary resolution

**Example**:
```python
grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[51, 51])
u_grid = np.random.rand(51, 51)

# Interpolate at arbitrary points
query_points = np.array([[0.33, 0.67], [0.12, 0.88]])  # (2, 2) - 2 points in 2D
u_interp = grid.interpolate(u_grid, query_points, method="linear")
# u_interp.shape: (2,) - interpolated values at 2 points
```

---

## Tutorial: Writing a Dimension-Agnostic Solver

Let's write a **heat equation solver** that works in any dimension (1D, 2D, 3D, ...) using trait protocols.

### Problem: Heat Equation

```
∂u/∂t = α·∇²u    in Ω
u(0, x) = u₀(x)
BC: Neumann (∂u/∂n = 0)
```

### Step 1: Import Trait Protocols

```python
from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.boundary import BoundaryConditions
from mfg_pde.geometry.protocols import SupportsLaplacian

if TYPE_CHECKING:
    from numpy.typing import NDArray
```

### Step 2: Define Solver Function with Trait Type Hint

```python
def solve_heat_equation(
    u0: NDArray,
    geometry: SupportsLaplacian,  # Trait protocol type hint
    alpha: float,
    T: float,
    dt: float,
    bc: BoundaryConditions | None = None,
) -> NDArray:
    """
    Solve heat equation ∂u/∂t = α·∇²u using forward Euler.

    Args:
        u0: Initial condition (flattened array of shape (N,))
        geometry: Geometry supporting Laplacian operator
        alpha: Thermal diffusivity
        T: Final time
        dt: Time step (must satisfy CFL: α·dt/dx² < 0.5)
        bc: Boundary conditions (passed to Laplacian operator)

    Returns:
        Solution at time T (flattened array of shape (N,))

    Raises:
        ValueError: If CFL condition violated (warning if detected)

    Notes:
        Works in **any dimension** and with **any discretization** supporting
        Laplacian trait (FDM, FEM, GFDM, meshfree, graph Laplacian).
    """
    # Get Laplacian operator from geometry
    laplacian = geometry.get_laplacian_operator(order=2, bc=bc)

    # Time stepping
    Nt = int(T / dt)
    u = u0.copy()

    for n in range(Nt):
        # Forward Euler: u^{n+1} = u^n + dt·α·L[u^n]
        Lu = laplacian @ u  # Apply Laplacian (works for any geometry!)
        u = u + dt * alpha * Lu

    return u
```

**★ Insight ─────────────────────────────────────**
The type hint `geometry: SupportsLaplacian` does **two things**:
1. **Documents** that geometry must provide Laplacian capability
2. **Validates** at type-check time that the trait is satisfied

No runtime `hasattr()` checks needed! The contract is enforced by the type system.
**─────────────────────────────────────────────────**

### Step 3: Test with Different Geometries

**1D Cartesian Grid**:
```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import neumann_bc

# 1D grid
grid_1d = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[101])
u0_1d = np.sin(np.pi * grid_1d.coordinates[0])  # Initial condition

# Solve
u_final_1d = solve_heat_equation(
    u0=u0_1d,
    geometry=grid_1d,  # Satisfies SupportsLaplacian ✓
    alpha=0.01,
    T=1.0,
    dt=0.0001,
    bc=neumann_bc(dimension=1),
)

print(f"1D solution computed: shape {u_final_1d.shape}")
```

**2D Cartesian Grid**:
```python
# 2D grid
grid_2d = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[51, 51])
x, y = grid_2d.coordinates
X, Y = np.meshgrid(x, y, indexing='ij')
u0_2d = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Solve (same function, different dimension!)
u_final_2d = solve_heat_equation(
    u0=u0_2d.ravel(),  # Flatten to (N,)
    geometry=grid_2d,  # Satisfies SupportsLaplacian ✓
    alpha=0.01,
    T=1.0,
    dt=0.0001,
    bc=neumann_bc(dimension=2),
)

print(f"2D solution computed: shape {u_final_2d.shape}")
u_final_2d = u_final_2d.reshape(51, 51)  # Back to grid shape
```

**Graph Geometry** (bonus):
```python
from mfg_pde.geometry.graph import NetworkGeometry

# Graph (network topology)
graph = NetworkGeometry(topology="scale_free", n_nodes=100, m=3, seed=42)
u0_graph = np.random.rand(100)  # Initial condition on nodes

# Solve (same function, graph Laplacian!)
u_final_graph = solve_heat_equation(
    u0=u0_graph,
    geometry=graph,  # Satisfies SupportsLaplacian ✓ (graph Laplacian)
    alpha=0.01,
    T=1.0,
    dt=0.001,
    bc=None,  # Graphs may not have traditional BC
)

print(f"Graph solution computed: {u_final_graph.shape}")
```

**Result**: **One solver, four dimensions, three discretization types** (FDM 1D, FDM 2D, graph Laplacian). No code duplication!

### Step 4: Static Type Checking

Run mypy to validate trait usage:

```bash
$ mypy my_solver.py
```

**Mypy catches errors**:
```python
# Error: Geometry doesn't support Laplacian
class MyGeometry:
    def dimension(self) -> int:
        return 2

grid_broken = MyGeometry()
solve_heat_equation(u0, grid_broken, alpha=0.01, T=1.0, dt=0.001)
# mypy: Argument 2 has incompatible type "MyGeometry"; expected "SupportsLaplacian"
```

---

## Operator Patterns

### Pattern 1: Operator as Matrix (Explicit)

**When**: Small problems, need to inspect matrix structure

```python
laplacian = geometry.get_laplacian_operator(order=2)
# Returns sparse matrix (scipy.sparse.csr_matrix or similar)

u = np.random.rand(N)
Lu = laplacian @ u  # Matrix-vector product

# Inspect matrix
print(f"Laplacian matrix: {laplacian.shape}, nnz={laplacian.nnz}")
```

### Pattern 2: Operator as LinearOperator (Matrix-Free)

**When**: Large problems, iterative solvers, no need to store matrix

```python
laplacian = geometry.get_laplacian_operator(order=2)
# Returns scipy.sparse.linalg.LinearOperator

# Use with iterative solvers
from scipy.sparse.linalg import cg

b = np.random.rand(N)  # Right-hand side
u, info = cg(laplacian, b, tol=1e-6)  # Conjugate gradient
```

### Pattern 3: Operator as Callable

**When**: Geometry-specific optimizations (e.g., FFT for periodic BC)

```python
laplacian = geometry.get_laplacian_operator(order=2)
# Returns callable: laplacian(u) computes ∇²u

u = np.random.rand(N)
Lu = laplacian(u)  # Function call instead of @ operator
```

**Solver Pattern** (handles all three):
```python
def apply_operator(op, u):
    """Apply operator (matrix, LinearOperator, or callable)."""
    if callable(op):
        return op(u)
    else:
        return op @ u  # Works for both sparse matrix and LinearOperator
```

### Pattern 4: Combining Operators

**Example**: Compute $\nabla^2 u$ manually from gradient and divergence

```python
# Get operators
grad_x, grad_y = geometry.get_gradient_operator(scheme="centered")
div_op = geometry.get_divergence_operator()

# Compute ∇²u = ∇·(∇u)
u = np.random.rand(N)
ux = grad_x @ u
uy = grad_y @ u
grad_u = np.stack([ux, uy], axis=0)  # Shape: (2, N)
laplacian_u = div_op @ grad_u.ravel()  # div(grad(u))

# Compare with direct Laplacian
laplacian_op = geometry.get_laplacian_operator()
laplacian_u_direct = laplacian_op @ u

print(f"Error: {np.linalg.norm(laplacian_u - laplacian_u_direct)}")
# Should be small (numerical precision)
```

---

## Best Practices

### 1. Use Trait Type Hints for Public APIs

**DO**:
```python
def my_solver(geometry: SupportsLaplacian & SupportsGradient):
    """Solver requires Laplacian AND Gradient."""
    laplacian = geometry.get_laplacian_operator()
    grad_ops = geometry.get_gradient_operator()
    # ...
```

**Explanation**: Type intersection `&` requires geometry to satisfy **both** traits.

### 2. Check Traits at Runtime (Optional)

For defensive programming or clear error messages:

```python
def my_solver(geometry):
    """Solver for heat equation."""
    from mfg_pde.geometry.protocols import SupportsLaplacian

    if not isinstance(geometry, SupportsLaplacian):
        raise TypeError(
            f"Geometry {type(geometry).__name__} does not support Laplacian. "
            f"Implement get_laplacian_operator() method."
        )

    laplacian = geometry.get_laplacian_operator()
    # ...
```

**When to use**:
- ✅ Entry points to libraries (clear user-facing errors)
- ✅ Public APIs that others will call
- ❌ Internal functions (trust type hints)

### 3. Flatten Arrays for Operators

**Convention**: Operators expect **1D arrays** (flattened), not grid-shaped arrays.

```python
# Grid shape: (Nx, Ny)
u_grid = np.random.rand(Nx, Ny)

# Apply operator: flatten first
laplacian = geometry.get_laplacian_operator()
Lu_flat = laplacian @ u_grid.ravel()  # Flatten (Nx, Ny) → (Nx*Ny,)

# Reshape back to grid
Lu_grid = Lu_flat.reshape(Nx, Ny)
```

**Rationale**: Linear algebra operates on vectors (1D). Reshaping is cheap (`O(1)`, view not copy).

### 4. Document Required Traits

```python
def solve_mfg_system(problem, geometry):
    """
    Solve MFG system.

    Required Traits:
        - SupportsLaplacian: For diffusion term in HJB/FP
        - SupportsGradient: For Hamiltonian ∇u term
        - SupportsAdvection: For Fokker-Planck drift

    Args:
        problem: MFGProblem instance
        geometry: Geometry satisfying above traits
    """
    # Check traits (optional, for clarity)
    from mfg_pde.geometry.protocols import (
        SupportsAdvection,
        SupportsGradient,
        SupportsLaplacian,
    )

    assert isinstance(geometry, SupportsLaplacian), "Need Laplacian"
    assert isinstance(geometry, SupportsGradient), "Need Gradient"
    assert isinstance(geometry, SupportsAdvection), "Need Advection"

    # ... implementation ...
```

### 5. Use Scheme Parameters Appropriately

**Gradient Schemes**:
- **`"centered"`**: Use for smooth solutions, interior points (most accurate)
- **`"upwind"`**: Use for hyperbolic PDEs, Level Set evolution (stable)
- **`"forward"/"backward"`**: Use for boundary conditions or one-sided approximations

**Advection Schemes**:
- **`"upwind"`**: Default for FP equation (stable, 1st-order)
- **`"weno"`**: High-order shocks, discontinuities (requires fine grid)
- **`"centered"`**: Smooth solutions only (may oscillate)

**Conservative vs Non-Conservative**:
- **`conservative=True`**: Use for FP equation (preserves mass)
- **`conservative=False`**: Use for Hamilton-Jacobi (non-conservative form)

---

## Advanced Topics

### 1. Implementing a Custom Geometry with Traits

Suppose you want to create a custom geometry (e.g., hexagonal grid). Implement the required protocols:

```python
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

from mfg_pde.geometry.boundary import BoundaryConditions
from mfg_pde.geometry.protocol import GeometryProtocol, GeometryType
from mfg_pde.geometry.protocols import SupportsGradient, SupportsLaplacian


class HexagonalGrid:
    """Custom hexagonal grid geometry."""

    def __init__(self, size: int):
        self.size = size
        self.num_cells = size * size  # Simplified
        # ... hexagonal grid construction ...

    # ====== GeometryProtocol (mandatory) ======
    @property
    def dimension(self) -> int:
        return 2

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.CUSTOM

    @property
    def num_spatial_points(self) -> int:
        return self.num_cells

    # ... other GeometryProtocol methods ...

    # ====== SupportsLaplacian trait ======
    def get_laplacian_operator(
        self,
        order: int = 2,
        bc: BoundaryConditions | None = None,
    ) -> LinearOperator | Callable:
        """Hexagonal Laplacian operator."""
        # Build sparse matrix for hexagonal stencil (6 neighbors)
        # ... implementation details ...

        # Return sparse matrix
        L = csr_matrix((self.num_cells, self.num_cells))
        # ... fill L with hexagonal stencil ...
        return L

    # ====== SupportsGradient trait ======
    def get_gradient_operator(self, direction=None, order=2, scheme="centered"):
        """Gradient on hexagonal grid."""
        # ... hexagonal gradient stencil ...
        if direction is None:
            grad_x = ...  # 2D gradient in x-direction
            grad_y = ...  # 2D gradient in y-direction
            return (grad_x, grad_y)
        else:
            return ...  # Single direction

# Now usable with trait-based solvers!
hex_grid = HexagonalGrid(size=50)
u0 = np.random.rand(hex_grid.num_cells)

# Works with solve_heat_equation defined earlier
u_final = solve_heat_equation(
    u0=u0,
    geometry=hex_grid,  # Satisfies SupportsLaplacian ✓
    alpha=0.01,
    T=1.0,
    dt=0.001,
)
```

### 2. Trait Composition for Complex Solvers

**Example**: MFG solver requires **multiple traits**

```python
from typing import Protocol

from mfg_pde.geometry.protocols import (
    SupportsAdvection,
    SupportsGradient,
    SupportsLaplacian,
)


# Define composite trait
class MFGGeometry(SupportsLaplacian, SupportsGradient, SupportsAdvection, Protocol):
    """Geometry supporting all operators needed for MFG solvers."""
    pass


def solve_mfg(problem, geometry: MFGGeometry):
    """Solve MFG system."""
    # All operators guaranteed to exist
    laplacian = geometry.get_laplacian_operator()
    grad_ops = geometry.get_gradient_operator()
    # ... (advection operator created later with velocity field)

    # ... Picard iteration ...
```

### 3. Performance: Matrix-Free vs Explicit

**Rule of Thumb**:
- **Small problems** (N < 10,000): Explicit sparse matrix (faster for direct solves, easier to inspect)
- **Large problems** (N > 100,000): Matrix-free LinearOperator (memory-efficient, works with iterative solvers)
- **Structured grids**: Consider FFT-based operators for periodic BC (O(N log N) complexity)

**Example**: FFT-based Laplacian for periodic BC

```python
from scipy.fft import fft2, ifft2


def get_laplacian_operator_fft(grid):
    """FFT-based Laplacian for periodic BC (2D)."""
    Nx, Ny = grid.get_grid_shape()
    dx, dy = grid.spacing

    # Wavenumbers
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_squared = KX**2 + KY**2

    def laplacian_fft(u):
        """Apply Laplacian via FFT."""
        u_2d = u.reshape(Nx, Ny)
        u_hat = fft2(u_2d)
        laplacian_u_hat = -k_squared * u_hat  # Fourier space: ∇² → -k²
        laplacian_u = ifft2(laplacian_u_hat).real
        return laplacian_u.ravel()

    return laplacian_fft  # Return callable
```

---

## Summary and Recommendations

### Quick Reference: When to Use Which Trait

| Trait | PDE Term | Typical Use |
|:------|:---------|:------------|
| **SupportsLaplacian** | $\nabla^2 u$ | Heat equation, diffusion, HJB/FP with $\sigma > 0$ |
| **SupportsGradient** | $\nabla u$ | Hamiltonian $H(\nabla u)$, optimal control $\alpha = -\nabla u$ |
| **SupportsDivergence** | $\nabla \cdot v$ | FP conservation $\partial m/\partial t + \nabla \cdot (m\alpha) = 0$, curvature |
| **SupportsAdvection** | $v \cdot \nabla u$ or $\nabla \cdot (uv)$ | Transport, Fokker-Planck, Level Set evolution |
| **SupportsInterpolation** | Evaluate $u(\tilde{x})$ | Semi-Lagrangian, particle-in-cell, visualization |

### Solver Development Checklist

When writing a new solver:

1. ✅ **Identify required operators**: What PDE terms do you need?
2. ✅ **Add trait type hints**: `def my_solver(geometry: SupportsLaplacian & SupportsGradient)`
3. ✅ **Request operators via traits**: `laplacian = geometry.get_laplacian_operator()`
4. ✅ **Flatten arrays**: `u.ravel()` before applying operators
5. ✅ **Document trait requirements**: In docstring, list required traits
6. ✅ **Test with multiple geometries**: 1D grid, 2D grid, graph (if applicable)
7. ✅ **Run mypy**: Validate trait usage statically

### Migration Guide (Pre-Issue #590 → Post-Issue #590)

**Old pattern** (duck typing):
```python
# Before
if hasattr(geometry, "laplacian_matrix"):
    L = geometry.laplacian_matrix
elif hasattr(geometry, "get_laplacian"):
    L = geometry.get_laplacian()
```

**New pattern** (trait protocols):
```python
# After
from mfg_pde.geometry.protocols import SupportsLaplacian

def my_solver(geometry: SupportsLaplacian):
    L = geometry.get_laplacian_operator()  # Guaranteed to exist
```

### Further Reading

- **Theory**: `docs/theory/GEOMETRY_BC_ARCHITECTURE_DESIGN.md` (design rationale)
- **Examples**:
  - `examples/advanced/stefan_problem_1d.py` (SupportsGradient, SupportsDivergence for curvature)
  - `examples/advanced/capacity_constrained_mfg/` (SupportsLaplacian, SupportsAdvection)
- **API Reference**:
  - `mfg_pde/geometry/protocol.py` (GeometryProtocol)
  - `mfg_pde/geometry/protocols/operators.py` (Operator traits)
- **Testing**: `tests/unit/geometry/protocols/` (protocol unit tests)

---

**Last Updated**: 2026-01-18
**Author**: MFG_PDE Documentation Team
**Related Issues**: #590 (Geometry Traits), #594 (Documentation)
