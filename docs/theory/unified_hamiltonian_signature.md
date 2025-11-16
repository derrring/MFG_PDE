# Unified Hamiltonian Signature: Dimension-Agnostic MFG Design

**Status**: Production (v0.9.0+)
**Date**: November 2025
**Related Files**:
- `examples/advanced/arbitrary_nd_geometry_demo.py:156-181`
- `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`
- `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py`

## Mathematical Foundation

### Classical Hamiltonian in MFG

The Hamilton-Jacobi-Bellman (HJB) equation in Mean Field Games takes the form:

$$
-\frac{\partial u}{\partial t} + H(t, x, \nabla u, m) = 0
$$

where:
- $u(t,x) \in \mathbb{R}$ is the value function
- $m(t,x) \geq 0$ is the population density
- $H: \mathbb{R} \times \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R} \to \mathbb{R}$ is the Hamiltonian
- $\nabla u = (p_1, \ldots, p_d) \in \mathbb{R}^d$ is the gradient (momentum variable)

### The Dimension-Agnostic Challenge

Traditional implementations often hardcode dimension-specific logic:

```python
# ❌ BAD: Dimension-specific implementation
if d == 2:
    H = 0.5 * (p_x**2 + p_y**2) + 0.5 * m
elif d == 3:
    H = 0.5 * (p_x**2 + p_y**2 + p_z**2) + 0.5 * m
# ... etc for each dimension
```

This approach:
- Fails for $d \geq 4$ (high-dimensional state spaces)
- Requires code duplication
- Cannot handle arbitrary dimensions
- Limits extensibility

## The Unified Signature

### Design Principle

**Principle**: The Hamiltonian signature should work identically in 2D, 4D, 10D, 100D without modification.

### Type Signature

```python
def hamiltonian(
    t: float,
    x: np.ndarray,           # (d,) position vector
    derivs: tuple[np.ndarray, ...],  # Tuple of gradient arrays
    m: float                 # Scalar density
) -> float:
    """
    Dimension-agnostic Hamiltonian.

    Args:
        t: Time (scalar)
        x: Position vector in ℝ^d, shape (d,)
        derivs: Tuple of derivative arrays. For first-order MFG:
                derivs = (p,) where p is ∇u with shape (d,)
                For second-order MFG:
                derivs = (p, q) where p = ∇u, q = ∇²u
        m: Population density at (t, x) (scalar)

    Returns:
        Hamiltonian value H(t, x, ∇u, m) (scalar)
    """
```

### Key Innovation: Tuple Notation

The crucial design choice is using `derivs: tuple[np.ndarray, ...]` instead of individual arguments:

**Why Tuples?**
1. **Extensibility**: Supports first-order $(p)$, second-order $(p, q)$, and higher
2. **Dimension-agnostic**: `derivs[0]` has shape `(d,)` regardless of $d$
3. **Type-safe**: Modern Python typing with `tuple[np.ndarray, ...]`
4. **Uniform interface**: Same signature for all Hamiltonian types

**Comparison with Legacy Approaches**:

```python
# ❌ Legacy: String-key dictionary (runtime errors, no type safety)
def hamiltonian_legacy(t, x, gradients: dict, m):
    p = gradients['p_values']  # KeyError if wrong key!
    return 0.5 * np.sum(p**2) + 0.5 * m

# ✅ Modern: Tuple notation (compile-time checks, dimension-agnostic)
def hamiltonian_unified(t, x, derivs: tuple[np.ndarray, ...], m):
    p = derivs[0]  # Always works, type-checked
    return 0.5 * np.sum(p**2) + 0.5 * m
```

## Examples

### Example 1: Standard Quadratic Hamiltonian

The most common MFG Hamiltonian with crowd aversion:

$$
H(t, x, p, m) = \frac{1}{2}\|p\|^2 + \frac{\nu}{2} m
$$

**Implementation**:

```python
def quadratic_hamiltonian(
    t: float,
    x: np.ndarray,
    derivs: tuple[np.ndarray, ...],
    m: float
) -> float:
    """
    Standard quadratic Hamiltonian with crowd aversion.

    Works for ANY dimension d:
    - 2D: x ∈ ℝ², p ∈ ℝ²
    - 4D: x ∈ ℝ⁴, p ∈ ℝ⁴
    - 100D: x ∈ ℝ¹⁰⁰, p ∈ ℝ¹⁰⁰
    """
    p = derivs[0]  # Gradient ∇u, shape (d,)

    # Kinetic energy: (1/2)||p||²
    kinetic = 0.5 * np.sum(p**2)

    # Crowd aversion: (ν/2)m
    nu = 1.0  # Crowd aversion coefficient
    interaction = 0.5 * nu * m

    return kinetic + interaction
```

**Usage in different dimensions**:

```python
# 2D case
t = 0.5
x_2d = np.array([1.0, 2.0])
p_2d = np.array([0.5, -0.3])
m = 1.0

H_2d = quadratic_hamiltonian(t, x_2d, (p_2d,), m)
# H = 0.5 * (0.5² + 0.3²) + 0.5 * 1.0 = 0.67

# 4D case - SAME CODE
x_4d = np.array([1.0, 2.0, 3.0, 4.0])
p_4d = np.array([0.5, -0.3, 0.2, -0.1])

H_4d = quadratic_hamiltonian(t, x_4d, (p_4d,), m)
# H = 0.5 * (0.5² + 0.3² + 0.2² + 0.1²) + 0.5 * 1.0 = 0.695

# 100D case - STILL SAME CODE
x_100d = np.random.rand(100)
p_100d = np.random.rand(100)

H_100d = quadratic_hamiltonian(t, x_100d, (p_100d,), m)
# Works identically!
```

### Example 2: Anisotropic Diffusion

For problems with directional preferences (e.g., pedestrian motion):

$$
H(t, x, p, m) = \frac{1}{2} p^T D p + \nu m
$$

where $D \in \mathbb{R}^{d \times d}$ is a positive-definite diffusion matrix.

**Implementation**:

```python
def anisotropic_hamiltonian(
    t: float,
    x: np.ndarray,
    derivs: tuple[np.ndarray, ...],
    m: float,
    D: np.ndarray  # Diffusion matrix (d, d)
) -> float:
    """
    Anisotropic Hamiltonian with directional diffusion.

    Example: In 2D pedestrian flow, diffusion along corridors
    differs from perpendicular diffusion.
    """
    p = derivs[0]

    # Anisotropic kinetic: (1/2) p^T D p
    kinetic = 0.5 * p @ D @ p

    # Interaction
    nu = 1.0
    interaction = nu * m

    return kinetic + interaction
```

**Usage**:

```python
# 2D corridor: easier to move in x direction
D_2d = np.array([[1.0, 0.0],   # Easy in x
                 [0.0, 0.1]])   # Hard in y (perpendicular to corridor)

x = np.array([5.0, 3.0])
p = np.array([1.0, 0.5])
H = anisotropic_hamiltonian(0.0, x, (p,), 1.0, D_2d)

# 4D example: Configuration space (position + velocity)
D_4d = np.diag([1.0, 1.0, 0.5, 0.5])  # Position easier than velocity
x_4d = np.random.rand(4)
p_4d = np.random.rand(4)
H_4d = anisotropic_hamiltonian(0.0, x_4d, (p_4d,), 1.0, D_4d)
```

### Example 3: Position-Dependent Hamiltonian

For spatially heterogeneous costs (e.g., terrain with obstacles):

$$
H(t, x, p, m) = \frac{1}{2} c(x) \|p\|^2 + g(x) + \nu m
$$

**Implementation**:

```python
def spatial_hamiltonian(
    t: float,
    x: np.ndarray,
    derivs: tuple[np.ndarray, ...],
    m: float
) -> float:
    """
    Position-dependent Hamiltonian for heterogeneous domains.

    c(x): Running cost (e.g., difficult terrain)
    g(x): Potential (e.g., attractors/repellers)
    """
    p = derivs[0]

    # Running cost: increases near obstacles
    c_x = 1.0 + np.exp(-np.linalg.norm(x - np.array([5.0]*len(x)))**2)

    # Kinetic with running cost
    kinetic = 0.5 * c_x * np.sum(p**2)

    # Potential: attraction to origin
    potential = 0.1 * np.linalg.norm(x)**2

    # Crowd aversion
    interaction = 0.5 * m

    return kinetic + potential + interaction
```

### Example 4: Second-Order MFG

For problems involving acceleration (e.g., vehicle dynamics):

$$
H(t, x, p, q, m) = \frac{1}{2}\|p\|^2 + \frac{\alpha}{2}\text{tr}(q) + \nu m
$$

where $q = \nabla^2 u$ is the Hessian matrix.

**Implementation**:

```python
def second_order_hamiltonian(
    t: float,
    x: np.ndarray,
    derivs: tuple[np.ndarray, ...],
    m: float,
    alpha: float = 0.1
) -> float:
    """
    Second-order Hamiltonian with viscosity.

    derivs = (p, q) where:
    - p = ∇u (gradient), shape (d,)
    - q = ∇²u (Hessian), shape (d, d)
    """
    p = derivs[0]    # Gradient
    q = derivs[1]    # Hessian

    # First-order term
    kinetic = 0.5 * np.sum(p**2)

    # Second-order term (viscosity)
    viscosity = 0.5 * alpha * np.trace(q)

    # Interaction
    interaction = 0.5 * m

    return kinetic + viscosity + interaction
```

**Usage**:

```python
# 2D case with Hessian
x = np.array([1.0, 2.0])
p = np.array([0.5, -0.3])
q = np.array([[0.1, 0.0],
              [0.0, -0.2]])  # Hessian matrix

H = second_order_hamiltonian(0.0, x, (p, q), 1.0, alpha=0.1)
# H = 0.5*(0.5² + 0.3²) + 0.5*0.1*(0.1 - 0.2) + 0.5*1.0
```

## Integration with Solvers

### Particle-Collocation Methods

The unified signature is essential for particle methods in high dimensions:

```python
# From mfg_pde/alg/numerical/fp_solvers/fp_particle.py

def solve_mfg_particle(
    hamiltonian: Callable[[float, np.ndarray, tuple[np.ndarray, ...], float], float],
    domain: ImplicitDomain,  # Works in any dimension!
    ...
):
    """
    Particle-collocation solver for arbitrary dimensions.

    The Hamiltonian signature enables:
    1. Same solver code for 2D, 4D, 10D problems
    2. Type-safe gradient handling
    3. Extensibility to second-order problems
    """
    # Sample particles in d-dimensional domain
    particles = domain.sample_uniform(n_particles)  # (N, d)

    # Compute gradients (dimension-agnostic)
    grad_u = compute_gradient(u, particles)  # (N, d)

    # Evaluate Hamiltonian at each particle
    for i, (x, p) in enumerate(zip(particles, grad_u)):
        m = density[i]
        H[i] = hamiltonian(t, x, (p,), m)
        # Works for ANY dimension d!
```

### Semi-Lagrangian Methods

Grid-based methods also benefit from the unified signature:

```python
# From mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py

def solve_hjb_semilagrangian(
    hamiltonian: Callable[[float, np.ndarray, tuple[np.ndarray, ...], float], float],
    grid: TensorProductGrid,  # Supports d ≤ 3 efficiently
    ...
):
    """
    Semi-Lagrangian solver with unified Hamiltonian.
    """
    # For each grid point
    for idx in np.ndindex(grid.shape):
        x = grid.points[idx]

        # Compute gradient at grid point
        p = compute_gradient_fd(u, grid, idx)  # (d,)

        # Evaluate Hamiltonian
        m = density[idx]
        H_val = hamiltonian(t, x, (p,), m)

        # Update value function
        # ...
```

## Migration Guide

### From Legacy String-Key Format

**Old code** (pre-v0.9.0):

```python
def old_hamiltonian(t, x, gradients, m):
    p_x = gradients['p_x']
    p_y = gradients['p_y']
    return 0.5 * (p_x**2 + p_y**2) + 0.5 * m

# Usage
H = old_hamiltonian(t, x, {'p_x': 0.5, 'p_y': -0.3}, m)
```

**New code** (v0.9.0+):

```python
def new_hamiltonian(t, x, derivs: tuple[np.ndarray, ...], m):
    p = derivs[0]  # Works for ANY dimension
    return 0.5 * np.sum(p**2) + 0.5 * m

# Usage
H = new_hamiltonian(t, x, (np.array([0.5, -0.3]),), m)
```

**Migration steps**:

1. **Change signature**: Replace `gradients: dict` with `derivs: tuple[np.ndarray, ...]`
2. **Extract gradient**: Replace `p_x, p_y = gradients['p_x'], gradients['p_y']` with `p = derivs[0]`
3. **Use vectorized operations**: Replace `p_x**2 + p_y**2` with `np.sum(p**2)`
4. **Update call sites**: Replace `H(t, x, {'p_x': ...}, m)` with `H(t, x, (p,), m)`

### Backward Compatibility

For codebases with extensive legacy usage, create adapter:

```python
def legacy_adapter(hamiltonian_new):
    """Adapt new signature to old string-key format."""
    def hamiltonian_old(t, x, gradients: dict, m):
        # Convert dict to tuple
        d = len(x)
        if d == 2:
            p = np.array([gradients['p_x'], gradients['p_y']])
        elif d == 3:
            p = np.array([gradients['p_x'], gradients['p_y'], gradients['p_z']])
        else:
            raise ValueError(f"Unsupported dimension {d}")

        return hamiltonian_new(t, x, (p,), m)

    return hamiltonian_old

# Usage
new_ham = quadratic_hamiltonian
old_ham = legacy_adapter(new_ham)

# Old code keeps working
H = old_ham(t, x, {'p_x': 0.5, 'p_y': -0.3}, m)
```

## Performance Considerations

### Memory Efficiency

The tuple-based approach has minimal overhead:

```python
# Memory comparison (for d=100)
p = np.random.rand(100)  # 800 bytes (100 * 8 bytes/float64)

# Old: Dictionary with 100 keys
old_format = {f'p_{i}': p[i] for i in range(100)}
# ~15 KB (dict overhead + string keys + values)

# New: Single tuple
new_format = (p,)
# ~800 bytes (just the array!) + 56 bytes (tuple object)
# Total: ~856 bytes vs 15 KB

# Speedup: ~17.5x less memory
```

### Computational Cost

Vectorized operations enable SIMD optimizations:

```python
import time

d = 1000
p = np.random.rand(d)

# ❌ Element-wise (slow)
start = time.time()
for _ in range(10000):
    H = sum(p[i]**2 for i in range(d))
print(f"Element-wise: {time.time() - start:.3f}s")  # ~2.5s

# ✅ Vectorized (fast)
start = time.time()
for _ in range(10000):
    H = np.sum(p**2)
print(f"Vectorized: {time.time() - start:.3f}s")    # ~0.05s

# Speedup: ~50x faster
```

## Best Practices

### 1. Always Use Type Hints

```python
# ✅ GOOD: Type-checked
def hamiltonian(
    t: float,
    x: np.ndarray,
    derivs: tuple[np.ndarray, ...],
    m: float
) -> float:
    ...

# ❌ BAD: No type safety
def hamiltonian(t, x, derivs, m):
    ...
```

### 2. Document Dimension Expectations

```python
def hamiltonian(
    t: float,
    x: np.ndarray,           # (d,)
    derivs: tuple[np.ndarray, ...],  # (p,) where p has shape (d,)
    m: float
) -> float:
    """
    Works for d ∈ {2, 3, 4, ..., 100}.

    For d > 100, ensure numerical stability.
    """
    ...
```

### 3. Validate Inputs in Development

```python
def hamiltonian_safe(t, x, derivs, m):
    # Development mode: check shapes
    if __debug__:
        assert isinstance(derivs, tuple), "derivs must be tuple"
        assert len(derivs) >= 1, "Need at least gradient"
        p = derivs[0]
        assert p.shape == x.shape, f"Gradient shape {p.shape} != position shape {x.shape}"

    # Production: no overhead
    p = derivs[0]
    return 0.5 * np.sum(p**2) + 0.5 * m
```

### 4. Use Partial Application for Parameters

```python
from functools import partial

# Base Hamiltonian with parameters
def parametric_hamiltonian(t, x, derivs, m, nu, alpha):
    p = derivs[0]
    return 0.5 * np.sum(p**2) + nu * m + alpha * np.linalg.norm(x)**2

# Create specialized versions
crowd_hamiltonian = partial(parametric_hamiltonian, nu=1.0, alpha=0.1)
sparse_hamiltonian = partial(parametric_hamiltonian, nu=0.1, alpha=0.5)

# Use with solvers
solve_mfg(crowd_hamiltonian, ...)  # nu=1.0, alpha=0.1
solve_mfg(sparse_hamiltonian, ...)  # nu=0.1, alpha=0.5
```

## Summary

### Key Principles

1. **Tuple notation**: `derivs: tuple[np.ndarray, ...]` enables dimension-agnostic code
2. **Vectorization**: Use `np.sum(p**2)` not `p_x**2 + p_y**2`
3. **Type safety**: Modern Python typing prevents runtime errors
4. **Extensibility**: Same signature for first-order, second-order, higher-order problems

### Advantages

| Aspect | Legacy (String Keys) | Unified (Tuples) |
|:-------|:--------------------|:-----------------|
| **Dimension support** | Hard-coded (2D, 3D) | Arbitrary (2D → 100D) |
| **Type safety** | Runtime errors | Compile-time checks |
| **Memory** | ~17x overhead | Minimal |
| **Performance** | Element-wise loops | SIMD vectorization |
| **Extensibility** | Difficult | Natural |

### When to Use

- ✅ **Always** for new code
- ✅ **Particle-collocation** methods (essential for d ≥ 4)
- ✅ **High-dimensional** state spaces (position + velocity + orientation)
- ⚠️ **Legacy codebases**: Use adapters for gradual migration

### Related Documentation

- **Implicit Geometry**: `docs/theory/implicit_geometry_mathematical_formulation.md`
- **Particle Methods**: `docs/theory/particle_collocation_methods.md`
- **API Design**: `docs/development/guides/CONSISTENCY_GUIDE.md`
- **Examples**: `examples/advanced/arbitrary_nd_geometry_demo.py`

---

**Next Steps**: See `examples/advanced/arbitrary_nd_geometry_demo.py` for complete working examples in 2D, 4D, 10D, and 100D.
