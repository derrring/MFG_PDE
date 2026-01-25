# Corner Handling Implementation Status

**Issue**: #521
**Last Updated**: 2025-01-25
**Status**: Partially Implemented

## Overview

Corner handling in PDE solvers requires different strategies depending on the numerical method. This document tracks implementation status and outlines requirements for unimplemented methods.

## Implementation Status Summary

| Solver/Method | Strategy | Status | Location |
|:-------------|:---------|:-------|:---------|
| FDM (Grid) | Sequential (Implicit) | ✅ Done | `geometry/boundary/applicator_fdm.py` |
| Particles (SDE/Position-based) | Fold Reflection | ✅ Done | `utils/geo/boundary_reflection.py` |
| Particles (Velocity-based/Billiard) | Normal Average | ❌ TODO | — |
| GFDM/Meshfree | Fold Reflection | ✅ Done | `geometry/boundary/applicator_meshfree.py` |
| SDF/Level Set | Mollify | ✅ Done | `geometry/base.py` |
| FVM (Flux) | Zero/Ignore | ❌ TODO | — |
| Eikonal (FMM/FSM) | Upwind | ❌ TODO | — |
| Subgradient Methods | Subgradient Selection | ❌ TODO | — |

---

## Implemented Methods

### 1. FDM Sequential Update (✅ Done)

**Location**: `mfg_pde/geometry/boundary/applicator_fdm.py`

**Strategy**: Ghost cells filled by sequential dimension updates. Last dimension "wins" at corners.

```
Dimension 0 (X)     Dimension 1 (Y)     Final Result
    ?   ?   ?           ?   ?   ?           C   B   C
    ?   ·   ?    →      A   ·   A    →      A   ·   A
    ?   ?   ?           ?   ?   ?           C   B   C

Corner C = overwritten by Y-pass (last dimension)
```

**No configuration needed** - implicit in the algorithm.

---

### 2. Position-based Particle Reflection (✅ Done)

**Location**: `mfg_pde/utils/geo/boundary_reflection.py`

**Strategy**: Modular fold reflection per dimension. Diagonal reflection emerges naturally.

```python
def reflect_positions(positions, bounds):
    """Each dimension processed independently → diagonal at corners."""
    for d in range(ndim):
        shifted = result[:, d] - xmin
        period = 2 * Lx
        pos_in_period = shifted % period
        in_second_half = pos_in_period > Lx
        pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
        result[:, d] = xmin + pos_in_period
```

**Used by**: `fp_particle_bc.py`, `applicator_particle.py`, `applicator_meshfree.py`

---

### 3. SDF Mollification (✅ Done)

**Location**: `mfg_pde/geometry/base.py:530-547`

**Strategy**: Treat sharp corner as rounded. Normal points from corner vertex toward query point.

```python
elif corner_strategy == "mollify":
    corner_vertex = np.where(near_min[i], min_coords, max_coords)
    direction = points[i] - corner_vertex
    normals[i] = direction / np.linalg.norm(direction)
```

**Configuration**: `corner_strategy="mollify"` in `get_boundary_normal()`

---

## TODO: Unimplemented Methods

### 4. Velocity-based Particle Reflection (❌ TODO)

**Priority**: Medium
**Use Case**: Billiard dynamics, hard-sphere collisions, deterministic particle systems

**Challenge**: Zeno trap at corners when velocity reflects infinitely fast.

**Required Implementation**:

```python
# Proposed location: mfg_pde/utils/geo/velocity_reflection.py

def reflect_velocity(
    position: NDArray,
    velocity: NDArray,
    bounds: list[tuple[float, float]],
    corner_strategy: Literal["average", "priority", "mollify"] = "average",
) -> NDArray:
    """
    Reflect velocity at boundary using specified corner strategy.

    At corners, multiple normals exist. Strategy determines which to use:
    - "average": Use averaged normal (diagonal reflection)
    - "priority": Use first face's normal (dimension 0 priority)
    - "mollify": Use mollified normal (smooth transition)

    Args:
        position: Current position at boundary
        velocity: Incoming velocity vector
        bounds: Domain bounds [(xmin, xmax), ...]
        corner_strategy: How to handle corners

    Returns:
        Reflected velocity: v_new = v - 2(v·n)n
    """
    from mfg_pde.geometry.base import DomainGeometry

    geom = DomainGeometry(bounds)
    normal = geom.get_boundary_normal(position, corner_strategy=corner_strategy)

    # Specular reflection: v_new = v - 2(v·n)n
    v_dot_n = np.dot(velocity, normal)
    return velocity - 2 * v_dot_n * normal
```

**Anti-Zeno safeguard** (optional):

```python
def reflect_with_damping(velocity, normal, damping=0.99):
    """Slight energy loss prevents infinite corner bounces."""
    v_dot_n = np.dot(velocity, normal)
    return damping * (velocity - 2 * v_dot_n * normal)
```

**Tests needed**:
- Corner reflection produces diagonal bounce with `average`
- No infinite loops (Zeno trap) in corner
- Energy conservation (or controlled dissipation)

---

### 5. FVM Corner Flux (❌ TODO)

**Priority**: Low
**Use Case**: Finite Volume Methods for conservation laws

**Challenge**: Corner cells have zero face area in certain directions.

**Required Implementation**:

```python
# Proposed location: mfg_pde/alg/numerical/fvm/corner_flux.py

def compute_corner_flux(
    cell_values: NDArray,
    corner_index: tuple[int, ...],
    flux_function: Callable,
    strategy: Literal["zero", "average", "upwind"] = "zero",
) -> float:
    """
    Compute flux contribution at corner cells.

    Strategies:
    - "zero": No flux through corners (standard FVM)
    - "average": Average flux from adjacent faces
    - "upwind": Use upwind direction for flux

    For most FVM applications, "zero" is correct since corner
    face area is zero in the limit.
    """
    if strategy == "zero":
        return 0.0
    elif strategy == "average":
        # Average flux from adjacent faces
        adjacent_fluxes = get_adjacent_face_fluxes(cell_values, corner_index)
        return np.mean(adjacent_fluxes)
    elif strategy == "upwind":
        # Use flux from upwind direction
        return get_upwind_flux(cell_values, corner_index, flux_function)
```

**Note**: Most FVM schemes naturally handle corners by having zero face area. Explicit corner handling rarely needed.

---

### 6. Eikonal Solver Corner Update (❌ TODO)

**Priority**: Medium
**Use Case**: Fast Marching Method (FMM), Fast Sweeping Method (FSM)

**Challenge**: Wavefront propagation must respect causality at corners.

**Required Implementation**:

```python
# Proposed location: mfg_pde/alg/numerical/eikonal/corner_update.py

def eikonal_corner_update(
    T: NDArray,           # Travel time array
    corner_idx: tuple,    # Corner grid index
    speed: float,         # Local speed F(x)
    dx: tuple[float, ...],  # Grid spacing per dimension
) -> float:
    """
    Update travel time at corner using upwind scheme.

    Eikonal equation: |∇T| = 1/F

    At corners, use the minimum of:
    1. Update from each adjacent face separately
    2. Diagonal update using Godunov upwind

    Returns:
        Updated travel time at corner
    """
    candidates = []

    # Single-dimension updates (from each face)
    for d in range(len(corner_idx)):
        T_neighbor = get_upwind_neighbor(T, corner_idx, d)
        candidates.append(T_neighbor + dx[d] / speed)

    # Multi-dimension Godunov update
    T_godunov = godunov_hamiltonian_update(T, corner_idx, speed, dx)
    candidates.append(T_godunov)

    # Causality: take minimum (first arrival)
    return min(candidates)
```

**Reference**: Sethian, "Level Set Methods and Fast Marching Methods" (1999)

---

### 7. Subgradient Selection at Corners (❌ TODO)

**Priority**: Low
**Use Case**: Non-smooth optimization in HJB solvers, proximal methods

**Challenge**: At corners of value function, gradient is undefined. Need subgradient selection.

**Required Implementation**:

```python
# Proposed location: mfg_pde/alg/numerical/hjb_solvers/subgradient.py

def select_subgradient(
    value_function: NDArray,
    corner_point: NDArray,
    bounds: list[tuple[float, float]],
    strategy: Literal["min_norm", "steepest", "random"] = "min_norm",
) -> NDArray:
    """
    Select a subgradient at non-smooth point of value function.

    At corners where V is non-differentiable:
    ∂V(x) = conv{∇V from each adjacent smooth region}

    Strategies:
    - "min_norm": Minimum norm element of subdifferential
    - "steepest": Steepest descent direction
    - "random": Random element (for stochastic methods)

    Args:
        value_function: Discretized value function
        corner_point: Point where gradient undefined
        bounds: Domain bounds
        strategy: Subgradient selection rule

    Returns:
        Selected subgradient vector
    """
    # Compute all one-sided derivatives
    gradients = compute_onesided_gradients(value_function, corner_point)

    if strategy == "min_norm":
        # Project origin onto convex hull of gradients
        return minimum_norm_subgradient(gradients)
    elif strategy == "steepest":
        # Most negative directional derivative
        return steepest_descent_subgradient(gradients)
    elif strategy == "random":
        # Convex combination with random weights
        weights = np.random.dirichlet(np.ones(len(gradients)))
        return sum(w * g for w, g in zip(weights, gradients))
```

**Mathematical Background**:

For convex function $f$, subdifferential at $x$:
$$\partial f(x) = \{g : f(y) \geq f(x) + g^T(y-x) \; \forall y\}$$

At corner of domain, value function $V$ may have kinks. The subgradient selection affects:
- Convergence of policy iteration
- Stability of Newton methods
- Accuracy of optimal control

**Reference**: Clarke, "Optimization and Nonsmooth Analysis" (1990)

---

## Architecture Diagram

```
                    Corner Handling Architecture
                    ===========================

    ┌─────────────────────────────────────────────────────────────┐
    │                    geometry/base.py                         │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │ get_boundary_normal(corner_strategy=...)            │   │
    │  │   - "average": sum normals, normalize               │   │
    │  │   - "priority": first face normal                   │   │
    │  │   - "mollify": radial from corner vertex            │   │
    │  └─────────────────────────────────────────────────────┘   │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │ is_near_corner()                                    │   │
    │  │ get_boundary_faces_at_point()                       │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ utils/geo/  │    │ applicator_ │    │ applicator_ │
    │             │    │ fdm.py      │    │ meshfree.py │
    │ Position    │    │             │    │             │
    │ Reflection  │    │ Sequential  │    │ Uses        │
    │ (fold)      │    │ Update      │    │ utils/geo/  │
    └─────────────┘    └─────────────┘    └─────────────┘
           │
           │ TODO
           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    Future Extensions                        │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
    │  │ velocity_   │  │ fvm/corner_ │  │ eikonal/corner_     │ │
    │  │ reflection  │  │ flux.py     │  │ update.py           │ │
    │  │ .py         │  │             │  │                     │ │
    │  │ (Billiard)  │  │ (FVM)       │  │ (FMM/FSM)           │ │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘ │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │ hjb_solvers/subgradient.py                          │   │
    │  │ (Non-smooth optimization)                           │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

| Method | Priority | Effort | Dependency |
|:-------|:---------|:-------|:-----------|
| Velocity-based Reflection | Medium | Small | `geometry/base.py` |
| Eikonal Corner Update | Medium | Medium | New solver |
| Subgradient Selection | Low | Medium | HJB refactor |
| FVM Corner Flux | Low | Small | FVM solver |

**Recommended order**: Velocity reflection → Eikonal → Subgradient → FVM

---

## References

1. Sethian, J.A. (1999). "Level Set Methods and Fast Marching Methods"
2. Clarke, F.H. (1990). "Optimization and Nonsmooth Analysis"
3. LeVeque, R.J. (2002). "Finite Volume Methods for Hyperbolic Problems"
4. Beck, A. & Teboulle, M. (2003). "Mirror descent and nonlinear projected subgradient methods"
