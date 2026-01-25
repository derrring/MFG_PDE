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
| Particles (SDE/Position-based) | Fold Reflection | ✅ Done | `geometry/boundary/corner/position.py` |
| Particles (Velocity-based/Billiard) | Specular Reflection | ✅ Done | `geometry/boundary/corner/velocity.py` |
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

**Location**: `mfg_pde/geometry/boundary/corner/position.py`

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

### 4. Velocity-based Particle Reflection (✅ Done)

**Location**: `mfg_pde/geometry/boundary/corner/velocity.py`

**Strategy**: Specular reflection with corner_strategy parameter.

```python
from mfg_pde.geometry.boundary.corner import reflect_velocity

# Basic usage
v_new = reflect_velocity(position, velocity, bounds)

# With corner strategy
v_new = reflect_velocity(position, velocity, bounds, corner_strategy="average")

# With anti-Zeno damping
v_new = reflect_velocity(position, velocity, bounds, damping=0.9)
```

**Features**:
- Specular reflection: `v_new = v - 2(v·n)n`
- Corner strategies: `"average"`, `"priority"`, `"mollify"`
- Anti-Zeno damping parameter (default 1.0 = elastic)
- Only reflects if moving into boundary (`v·n > 0`)
- Batch processing support
- Low-level API: `reflect_velocity_with_normal()` for pre-computed normals

---

## TODO: Unimplemented Methods

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
    ┌───────────────────────────────────────────────────────────────┐
    │           geometry/boundary/corner/ (IMPLEMENTED)             │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
    │  │ position.py     │  │ velocity.py     │  │ strategies.py│  │
    │  │ - reflect_pos   │  │ - reflect_vel   │  │ - CornerStrat│  │
    │  │ - wrap_pos      │  │ - specular      │  │ - validate   │  │
    │  │ - absorb_pos    │  │ - damping       │  │              │  │
    │  └─────────────────┘  └─────────────────┘  └──────────────┘  │
    └───────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ applicator_ │    │ applicator_ │    │ applicator_ │
    │ particle.py │    │ fdm.py      │    │ meshfree.py │
    │ (uses       │    │ (Sequential │    │ (uses       │
    │  corner/)   │    │  Update)    │    │  corner/)   │
    └─────────────┘    └─────────────┘    └─────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │                    Future Extensions (TODO)                  │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ fvm/corner_ │  │ eikonal/    │  │ hjb_solvers/        │  │
    │  │ flux.py     │  │ corner_     │  │ subgradient.py      │  │
    │  │ (FVM)       │  │ update.py   │  │ (Non-smooth opt)    │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

| Method | Priority | Effort | Status |
|:-------|:---------|:-------|:-------|
| Position-based Reflection | High | Done | ✅ `corner/position.py` |
| Velocity-based Reflection | Medium | Done | ✅ `corner/velocity.py` |
| Eikonal Corner Update | Medium | Medium | ❌ TODO |
| Subgradient Selection | Low | Medium | ❌ TODO |
| FVM Corner Flux | Low | Small | ❌ TODO |

**Recommended order**: Eikonal → Subgradient → FVM

---

## References

1. Sethian, J.A. (1999). "Level Set Methods and Fast Marching Methods"
2. Clarke, F.H. (1990). "Optimization and Nonsmooth Analysis"
3. LeVeque, R.J. (2002). "Finite Volume Methods for Hyperbolic Problems"
4. Beck, A. & Teboulle, M. (2003). "Mirror descent and nonlinear projected subgradient methods"
