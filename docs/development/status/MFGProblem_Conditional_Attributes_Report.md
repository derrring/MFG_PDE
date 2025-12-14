# MFGProblem Conditional Attributes: Architecture Decision Report

**Issue**: [#417](https://github.com/derrring/MFG_PDE/issues/417)
**Implementation**: [#435](https://github.com/derrring/MFG_PDE/issues/435)
**Status**: ✅ IMPLEMENTED (Phases 1-6 Complete)
**Date**: 2025-12-11
**Author**: Claude Code Analysis
**Review**: Expert feedback received and incorporated

> **Note**: This ADR has been fully implemented. See PRs #434, #436-#441 for code changes.

---

## Executive Summary

`MFGProblem` is the central domain object in MFG_PDE, representing a Mean Field Game problem. Currently, it has ~30 public attributes whose availability depends on the initialization mode (1D legacy, n-D grid, geometry, network, mesh). This creates type safety issues and runtime errors.

Two architectural solutions are proposed:
- **Option A**: Domain-Specific Subclasses
- **Option B**: Explicit Domain Container

This report analyzes both options and recommends **Modified Option B: Unify around `geometry` attribute**.

---

## 1. Problem Statement

### 1.1 Current Architecture

`MFGProblem` (~1985 lines) has 4 init methods, but only **2 conceptual modes**:

| Init Method | Status | What It Does | `self.geometry` |
|-------------|--------|--------------|-----------------|
| `_init_1d_legacy` | **Deprecated** | Creates `TensorProductGrid` internally | ✅ Set |
| `_init_nd` | **Deprecated** | Creates `TensorProductGrid` internally | ✅ Set |
| `_init_geometry` | Current API | Uses provided geometry directly | ✅ Set |
| `_init_network` | **BUG** | Sets `geometry = None` | ❌ None |

**Key insight**: The "1D Legacy" and "n-D Grid" modes are **deprecated backward-compatibility shims** that already convert to `TensorProductGrid`. The only broken path is `_init_network`.

Legacy API entry points (deprecated, emit warnings):
- `MFGProblem(Nx=100, xmin=0, xmax=1, ...)` → `_init_1d_legacy` → `TensorProductGrid`
- `MFGProblem(spatial_bounds=[(0,1),(0,1)], ...)` → `_init_nd` → `TensorProductGrid`

Modern API entry points:
- `MFGProblem(geometry=TensorProductGrid(...), ...)` → `_init_geometry`
- `MFGProblem(geometry=UnstructuredMesh(...), ...)` → `_init_geometry`
- `MFGProblem(network=NetworkGraph(...), ...)` → `_init_network` (broken)

### 1.2 The Type Safety Problem

```python
# Current code - type checker cannot verify safety
problem = MFGProblem(network=my_network)
x = problem.xSpace[0]  # Runtime AttributeError - xSpace is None for network mode

# Desired - compile-time type safety
problem = MFGProblem(network=my_network)
x = problem.xSpace[0]  # Type checker warns: xSpace may be None
```

### 1.3 Attribute Inventory

**Always Set** (all modes):
- `T`, `Nt`, `dt` (time domain)
- `sigma`, `coupling_coefficient` (physics)
- `dimension`, `spatial_shape`, `domain_type`
- `m_init`, `u_fin` (initial/terminal conditions)

**Conditionally Set** (~20 attributes):

| Attribute | 1D | n-D | Geometry | Network | Mesh |
|-----------|:--:|:---:|:--------:|:-------:|:----:|
| `xmin`, `xmax`, `dx`, `Lx`, `Nx` | ✓ | ✗ | ✗ | ✗ | ✗ |
| `xSpace` | ✓ | ✗ | ✗ | ✗ | ✗ |
| `spatial_bounds` | ✓ | ✓ | ✓ | ✗ | ✗ |
| `spatial_discretization` | ✓ | ✓ | ✓ | ✗ | ✗ |
| `geometry` | ✓ | ✓ | ✓ | ✗ | ✓ |
| `network` | ✗ | ✗ | ✗ | ✓ | ✗ |
| `num_nodes`, `adjacency_matrix` | ✗ | ✗ | ✗ | ✓ | ✗ |
| `mesh_data`, `collocation_points` | ✗ | ✗ | ✗ | ✗ | ✓ |
| `obstacles` | ✗ | ✗ | ✓ | ✗ | ✗ |

---

## 2. Option A: Domain-Specific Subclasses

### 2.1 Design

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

class MFGProblem(ABC):
    """Abstract base for all MFG problems."""
    T: float
    Nt: int
    sigma: float

    @abstractmethod
    def get_spatial_points(self) -> NDArray: ...

class CartesianMFGProblem(MFGProblem):
    """MFG on tensor-product grids (1D, 2D, nD)."""
    bounds: tuple[tuple[float, float], ...]  # ((x1min, x1max), ...)
    discretization: tuple[int, ...]          # (N1, N2, ...)
    geometry: TensorProductGrid

    # Guaranteed non-None
    @property
    def spatial_bounds(self) -> tuple[tuple[float, float], ...]:
        return self.bounds

class NetworkMFGProblem(MFGProblem):
    """MFG on graph/network domains."""
    network: NetworkGraph
    num_nodes: int
    adjacency_matrix: NDArray

class MeshMFGProblem(MFGProblem):
    """MFG on unstructured meshes."""
    geometry: UnstructuredMesh
    mesh_data: MeshData
    collocation_points: NDArray
```

### 2.2 Pros

1. **Full type safety** - Each subclass has guaranteed non-None attributes
2. **Clear domain modeling** - Class hierarchy reflects mathematical structure
3. **IDE support** - Autocomplete shows only relevant attributes
4. **Extensibility** - New domain types = new subclasses

### 2.3 Cons

1. **Breaking change** - All existing code using `MFGProblem` must migrate
2. **Factory complexity** - Need factory to create correct subclass from parameters
3. **Multiple classes to learn** - Users must understand class hierarchy
4. **Polymorphism overhead** - Generic code must use `isinstance()` checks

### 2.4 Migration Complexity

```python
# Before
problem = MFGProblem(Nx=100, xmin=0, xmax=1, ...)

# After - Option 1: Direct instantiation
problem = CartesianMFGProblem(bounds=[(0, 1)], discretization=[100], ...)

# After - Option 2: Factory (recommended)
problem = create_mfg_problem(Nx=100, xmin=0, xmax=1, ...)  # Returns CartesianMFGProblem
```

---

## 3. Option B: Explicit Domain Container

### 3.1 Design

```python
from dataclasses import dataclass
from typing import Union

@dataclass
class CartesianDomain:
    bounds: tuple[tuple[float, float], ...]
    discretization: tuple[int, ...]
    grid: TensorProductGrid

@dataclass
class NetworkDomain:
    network: NetworkGraph
    num_nodes: int
    adjacency_matrix: NDArray

@dataclass
class MeshDomain:
    mesh: UnstructuredMesh
    mesh_data: MeshData
    collocation_points: NDArray

class MFGProblem:
    domain: CartesianDomain | NetworkDomain | MeshDomain
    T: float
    Nt: int
    sigma: float

    # Type-safe access via narrowing
    def get_grid_bounds(self) -> tuple[tuple[float, float], ...]:
        if isinstance(self.domain, CartesianDomain):
            return self.domain.bounds
        raise TypeError("Not a Cartesian domain")
```

### 3.2 Pros

1. **Single class** - No class hierarchy to learn
2. **Explicit domain type** - `problem.domain` clearly shows domain structure
3. **Type narrowing** - `isinstance(problem.domain, CartesianDomain)` enables IDE support
4. **Gradual migration** - Can coexist with legacy attributes during transition

### 3.3 Cons

1. **Extra indirection** - `problem.domain.bounds` vs `problem.bounds`
2. **New container classes** - Must define `CartesianDomain`, `NetworkDomain`, etc.
3. **Union type complexity** - Generic code must handle all domain types
4. **Partial type safety** - Still need runtime checks for domain type

---

## 4. Recommendation: Modified Option B

### 4.1 Unify Around `geometry` Attribute

The codebase already has:
1. `GeometryProtocol` abstraction in `mfg_pde.geometry`
2. Deprecation warnings pushing toward `geometry=` parameter
3. Internal creation of `TensorProductGrid` for all grid modes

**Proposal**: Make `geometry` always non-None, covering all domain types.

```python
class MFGProblem:
    # Single source of truth - always set, never None
    geometry: GeometryProtocol  # TensorProductGrid | NetworkGeometry | UnstructuredMesh

    # Legacy properties become computed, delegating to geometry
    @property
    def xmin(self) -> float | None:
        """Deprecated: Use geometry.bounds[0][0] instead."""
        if isinstance(self.geometry, TensorProductGrid) and self.dimension == 1:
            warnings.warn("xmin is deprecated, use geometry.bounds", DeprecationWarning)
            return self.geometry.bounds[0][0]
        return None

    @property
    def spatial_bounds(self) -> tuple[tuple[float, float], ...] | None:
        if hasattr(self.geometry, 'bounds'):
            return self.geometry.bounds
        return None

    @property
    def network(self) -> NetworkGraph | None:
        if isinstance(self.geometry, NetworkGeometry):
            return self.geometry.network
        return None
```

### 4.2 New Class: NetworkGeometry

```python
class NetworkGeometry:
    """Geometry implementation for graph/network domains."""

    def __init__(self, network: NetworkGraph):
        self.network = network
        self._dimension = 0  # Or network embedding dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def num_nodes(self) -> int:
        return self.network.num_nodes

    @property
    def adjacency_matrix(self) -> NDArray:
        return self.network.adjacency_matrix

    def get_spatial_grid(self) -> NDArray:
        """Return node positions or indices."""
        return self.network.node_positions

    # Implement GeometryProtocol methods...
```

### 4.3 Implementation Phases

| Phase | Description | Breaking Change |
|-------|-------------|-----------------|
| **Phase 1** | Create `NetworkGeometry`, make `geometry` always set | No |
| **Phase 2** | Add deprecation warnings to legacy attributes | No |
| **Phase 3** | Convert legacy attributes to computed properties | No |
| **Phase 4** | Update type hints throughout codebase | No |
| **Phase 5** | Remove deprecated attributes (v1.0.0) | Yes |

### 4.4 Rationale

1. **Aligns with existing deprecation** - Already pushing users toward `geometry=`
2. **Minimal breaking change** - Legacy code continues working
3. **Type-safe narrowing** - `isinstance(problem.geometry, TensorProductGrid)`
4. **Single point of truth** - `geometry` is THE domain representation
5. **No new user-facing classes** - Users don't need to learn subclass hierarchy

---

## 5. Comparison Matrix

| Criterion | Option A (Subclasses) | Option B (Domain Container) | Modified B (Geometry) |
|-----------|:---------------------:|:---------------------------:|:---------------------:|
| **Type Safety** | Full | Partial | Partial + Narrowing |
| **Breaking Change** | High | Medium | Low |
| **Learning Curve** | New classes | New container | Existing pattern |
| **Migration Effort** | High | Medium | Low |
| **Aligns with Deprecation** | No | No | Yes |
| **IDE Support** | Excellent | Good | Good |
| **Future Extensibility** | Excellent | Good | Good |

---

## 6. Open Questions for Experts

### 6.1 Type System Questions

1. **Union vs Subclass**: For scientific computing libraries, is subclass hierarchy or union types more idiomatic for domain-specific behavior?

2. **Protocol vs ABC**: Should `GeometryProtocol` be a `Protocol` (structural typing) or `ABC` (nominal typing)? Current implementation uses `Protocol`.

3. **Type Narrowing UX**: Is `isinstance(problem.geometry, TensorProductGrid)` an acceptable pattern for users, or too verbose?

### 6.2 API Design Questions

4. **Attribute Access**: Should legacy attributes like `xmin` remain as deprecated properties, or be removed entirely in favor of `geometry.bounds[0][0]`?

5. **Factory Pattern**: If Option A, should we provide a factory function that auto-detects the domain type from parameters?

6. **NetworkGeometry Design**: What methods should `NetworkGeometry` implement to satisfy `GeometryProtocol`? Graph Laplacian? Shortest paths?

### 6.3 Migration Questions

7. **Deprecation Timeline**: How long should deprecated attributes remain functional? Current plan: until v1.0.0.

8. **Documentation Priority**: Should we document the new `geometry`-first API before or after implementation?

---

## 7. Code Examples

### 7.1 Current Usage (Problematic)

```python
from mfg_pde import MFGProblem

# User creates problem
problem = MFGProblem(Nx=100, xmin=0, xmax=1, Nt=50, T=1.0, sigma=0.1)

# Later in generic code - type checker can't help
def process_grid(problem: MFGProblem):
    # Type checker sees xSpace as NDArray | None
    # No way to guarantee it's not None without runtime check
    return problem.xSpace.mean()  # Potential AttributeError
```

### 7.2 After Modified Option B

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# User creates problem (same API)
problem = MFGProblem(Nx=100, xmin=0, xmax=1, Nt=50, T=1.0, sigma=0.1)

# Generic code with type narrowing
def process_grid(problem: MFGProblem):
    if isinstance(problem.geometry, TensorProductGrid):
        # Type checker knows geometry is TensorProductGrid
        # All grid methods available with full IDE support
        return problem.geometry.get_spatial_grid().mean()
    raise TypeError("Expected grid-based problem")

# Or use legacy (with deprecation warning)
def process_grid_legacy(problem: MFGProblem):
    if problem.xSpace is not None:  # Explicit None check
        return problem.xSpace.mean()
    raise TypeError("No spatial grid available")
```

### 7.3 Option A Alternative

```python
from mfg_pde import CartesianMFGProblem, NetworkMFGProblem

# User must choose correct class
problem = CartesianMFGProblem(
    bounds=[(0, 1)],
    discretization=[100],
    Nt=50, T=1.0, sigma=0.1
)

# Generic code uses base class
def process_any(problem: MFGProblem):
    points = problem.get_spatial_points()  # Abstract method
    return points.mean()

# Specialized code uses specific class
def process_cartesian(problem: CartesianMFGProblem):
    # Full type safety - bounds is guaranteed non-None
    return problem.bounds[0][1] - problem.bounds[0][0]  # Domain length
```

---

## 8. Appendix: Current Codebase Statistics

- **MFGProblem file**: `mfg_pde/core/mfg_problem.py` (1985 lines)
- **Initialization modes**: 4 (`_init_1d_legacy`, `_init_nd`, `_init_geometry`, `_init_network`)
- **Public attributes**: ~30
- **Conditional attributes**: ~20
- **GeometryProtocol implementations**: `TensorProductGrid`, `NetworkGeometry` (with subclasses), `Mesh2D`, `Mesh3D`, `ImplicitDomain`
- **Network geometry classes**: `NetworkGeometry`, `GridNetwork`, `RandomNetwork`, `ScaleFreeNetwork` (all in `mfg_pde/geometry/graph/network_geometry.py`)
- **Note**: `BaseNetworkGeometry` is a backward compatibility alias for `NetworkGeometry` (renamed in PR #434)

---

## 9. References

- Issue #417: https://github.com/derrring/MFG_PDE/issues/417
- GeometryProtocol: `mfg_pde/geometry/protocol.py`
- TensorProductGrid: `mfg_pde/geometry/grids/tensor_grid.py`
- NetworkGeometry: `mfg_pde/geometry/graph/network_geometry.py` (renamed from BaseNetworkGeometry in PR #434)
- MFGProblem: `mfg_pde/core/mfg_problem.py`
- Deprecation roadmap: `docs/development/DEPRECATION_ROADMAP.md`

---

## 10. Expert Review & Feedback

### 10.1 Decision Summary

**Modified Option B is approved** as the recommended approach.

### 10.2 Expert Analysis

#### Why Not Option A (Subclasses)?

Option A suffers from the **multiple orthogonality problem**:
- Domain types are one axis (Cartesian, Network, Mesh)
- Other potential axes exist (stochastic vs deterministic, time-dependent vs stationary, etc.)
- Multiple inheritance or complex hierarchies would be required
- Class explosion risk as the framework evolves

#### Additional Concerns Identified

1. **Interface Bloat**: Deprecated properties still appear in IDE autocomplete
   - **Solution**: Override `__dir__()` to exclude deprecated attributes from autocomplete
   - Users explicitly accessing deprecated attributes will still get DeprecationWarning

2. **Serialization/Pickling**: `geometry` as single source of truth affects pickling
   - **Solution**: Implement `__getstate__()` and `__setstate__()` for clean serialization
   - Ensure backward compatibility with pickled objects from older versions

### 10.3 Answers to Open Questions

| Question | Expert Answer |
|----------|---------------|
| **Q1: Union vs Subclass** | Use **Union + Protocol**. More Pythonic for scientific computing, avoids class hierarchy complexity. |
| **Q2: Protocol vs ABC** | Use **Protocol** (structural typing). Better for duck typing, no inheritance required. |
| **Q3: Type Narrowing UX** | Add **semantic helper properties** like `is_network`, `is_cartesian` to reduce verbosity. |
| **Q4: Deprecated Attributes** | Keep deprecated properties with **DeprecationWarning**. Use two-version buffer before removal. |
| **Q6: NetworkGeometry Design** | Implement **grid-like interface**: `get_laplacian_matrix()`, `get_mass_matrix()`, `get_stiffness_matrix()`. |
| **Q8: Documentation Priority** | **Documentation First**. Document the `geometry`-first API before implementation to validate design. |

### 10.4 Enhanced GeometryProtocol Design

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class GeometryProtocol(Protocol):
    """Unified interface for all spatial domain representations."""

    @property
    def dimension(self) -> int:
        """Spatial dimension of the domain."""
        ...

    def get_spatial_grid(self) -> NDArray:
        """Return spatial discretization points."""
        ...

    def get_mass_matrix(self) -> NDArray:
        """Return mass matrix for weak formulations."""
        ...

    def get_stiffness_matrix(self) -> NDArray:
        """Return stiffness/Laplacian matrix for diffusion."""
        ...


class NetworkGeometry:
    """Geometry implementation for graph/network domains."""

    def __init__(self, network: NetworkGraph):
        self.network = network

    @property
    def dimension(self) -> int:
        return 0  # Or embedding dimension

    def get_spatial_grid(self) -> NDArray:
        return self.network.node_positions

    def get_mass_matrix(self) -> NDArray:
        """Diagonal mass matrix (node weights)."""
        return np.diag(self.network.node_weights)

    def get_stiffness_matrix(self) -> NDArray:
        """Graph Laplacian for diffusion on networks."""
        return self.network.get_laplacian_matrix()
```

### 10.5 Implementation Recommendations

1. **Prototype First**: Create `NetworkGeometry` prototype and validate against existing solvers
2. **Type Checker Configuration**: Configure MyPy/Pyright for strict geometry type checking
3. **IDE Friendliness**: Test autocomplete behavior with `__dir__()` override
4. **Serialization Tests**: Verify pickle/unpickle works correctly with geometry-based design

### 10.6 Updated Implementation Phases

| Phase | Description | Breaking Change | Priority |
|-------|-------------|-----------------|----------|
| **Phase 0** | Document `geometry`-first API (design validation) | No | High |
| **Phase 1** | Wire `NetworkGeometry` into `_init_network()`, make `geometry` always non-None, **update `geometry` type hint to non-Optional** | No | High |
| **Phase 2** | Add `is_network`, `is_cartesian` helper properties | No | Medium |
| **Phase 3** | Override `__dir__()` to hide deprecated attributes | No | Medium |
| **Phase 4** | Add deprecation warnings to legacy attributes | No | Medium |
| **Phase 5** | Implement `__getstate__`/`__setstate__` for serialization | No | Low |
| **Phase 6** | Update remaining type hints throughout codebase | No | Low |
| **Phase 7** | Remove deprecated attributes (v1.0.0) | Yes | Future |

---

## 11. Blueprint Audit (2025-12-11)

### 11.1 Critical Finding: NetworkGeometry Already Exists

The original report stated "NetworkGeometry needs to be created". This was **incorrect**.

**Actual state of codebase** (updated after PR #434):
- `NetworkGeometry` exists at `mfg_pde/geometry/graph/network_geometry.py` (renamed from `BaseNetworkGeometry`)
- It now properly inherits from `GraphGeometry` (fixed in PR #434)
- It implements full `GeometryProtocol` (including all boundary methods)
- Concrete subclasses exist: `GridNetwork`, `RandomNetwork`, `ScaleFreeNetwork`
- `BaseNetworkGeometry` is a backward compatibility alias for `NetworkGeometry`

### 11.2 The Real Problem

The issue is not missing classes but **missing wiring**:

```python
# mfg_pde/core/mfg_problem.py:939
def _init_network(...):
    ...
    self.geometry = None  # <-- THE PROBLEM: sets geometry to None instead of NetworkGeometry
```

**Fix required**: `_init_network()` should wrap the network in a `NetworkGeometry` subclass.

### 11.3 Corrected Phase 1 Implementation

Instead of "Create NetworkGeometry", Phase 1 should:

1. Modify `_init_network()` to create appropriate `NetworkGeometry` subclass
2. Set `self.geometry` to the created network geometry
3. Populate legacy attributes from `geometry.get_problem_config()`

```python
# Corrected _init_network() logic
def _init_network(self, network, ...):
    # Wrap raw network in geometry object
    if isinstance(network, NetworkGeometry):  # Or BaseNetworkGeometry (alias)
        self.geometry = network
    elif hasattr(network, 'adjacency_matrix'):
        # Raw network - wrap in GridNetwork or similar
        from mfg_pde.geometry import create_network
        self.geometry = create_network('custom', num_nodes=len(network.nodes), ...)
    else:
        # NetworkX graph
        self.geometry = GridNetwork.from_networkx(network)

    # Use geometry as source of truth
    config = self.geometry.get_problem_config()
    self.spatial_shape = config['spatial_shape']
    self.spatial_bounds = config['spatial_bounds']
    # ... etc
```

### 11.4 Enhanced Protocol Methods (Future Work)

The expert-proposed methods (`get_mass_matrix()`, `get_stiffness_matrix()`) are **not currently in GeometryProtocol**.

Adding them would require:
1. Updating `GeometryProtocol` in `protocol.py`
2. Implementing in `TensorProductGrid`, `NetworkGeometry`, `ImplicitDomain`, etc.
3. This is **separate work** from the `geometry` unification goal

**Recommendation**: Defer enhanced protocol methods to a future issue.

**Additional Concern (Protocol Bloat)**: `GeometryProtocol` risks becoming a "god interface" if physical methods are added. Geometry objects should remain responsible only for geometric information (points, edges, distances). Physical matrices (mass, stiffness) should be handled by dedicated `Discretization` or `Operator` classes.

### 11.5 Serialization Migration Risk

**Risk**: Old pickle files containing scattered attributes (`xmin`, `Nx`, etc.) but `geometry=None` will crash when loaded into new code expecting non-null `geometry`.

**Mitigation**: Phase 5 must include migration logic in `__setstate__()`:

```python
def __setstate__(self, state):
    """Restore state with legacy pickle migration."""
    # Detect legacy format: geometry=None but has legacy 1D attrs
    if state.get('geometry') is None and state.get('xmin') is not None:
        try:
            from mfg_pde.geometry import TensorProductGrid
            # Reconstruct geometry from legacy attributes
            xmin = state.get('xmin')
            xmax = state.get('xmax')
            Nx = state.get('Nx')
            if xmin is None or xmax is None or Nx is None:
                raise KeyError("Missing required legacy fields")
            state['geometry'] = TensorProductGrid(
                bounds=[(xmin, xmax)],
                num_points=[Nx + 1]
            )
        except KeyError as e:
            raise RuntimeError(
                f"Unable to migrate legacy pickle: {e}. "
                "This pickle file may be from an incompatible version."
            ) from e
    self.__dict__.update(state)
```

### 11.6 Additional Blind Spots (Expert Feedback Round 2)

#### 11.6.1 Setter Trap - Read vs Write Asymmetry

The proposal covers **reading** legacy attributes via `@property`, but not **writing**.

**Risk**: If user executes `problem.xmin = -5.0`:
- **Case A (no setter)**: Python creates new instance attribute shadowing the property. Result: `xmin` changes but `geometry` is unchanged - **data inconsistency**.
- **Case B (getter only)**: Raises `AttributeError: can't set attribute` - breaks backward compatibility.

**Mitigation**: Implement explicit setters for all deprecated attributes:

```python
@xmin.setter
def xmin(self, value: float):
    warnings.warn(
        "Setting xmin is deprecated. Recreate problem with new geometry.",
        DeprecationWarning,
        stacklevel=2
    )
    # Option 1: Fail explicitly (recommended for immutable design)
    raise AttributeError(
        "Legacy attribute 'xmin' is read-only. "
        "Please recreate the problem with new geometry."
    )
    # Option 2: If mutation is required, update geometry (complex)
```

#### 11.6.2 Init Order Dependencies

**Risk**: `MFGProblem.__init__` may have code that accesses `self.spatial_bounds` or `self.dimension` **before** `_init_network()` sets `geometry`. This would crash.

**Mitigation**:
1. **Audit `__init__`**: Ensure `geometry` construction happens before any code accessing spatial attributes
2. **Fail Fast**: Add early validation at `__init__` start:
```python
def __init__(self, ...):
    # Set geometry FIRST, before any derived calculations
    self._setup_geometry(...)  # Must come before other logic
    # Now safe to use self.dimension, self.spatial_bounds, etc.
```

#### 11.6.3 `__repr__` Debug Experience

**Risk**: If `__repr__` prints legacy attributes like `xmin`, it triggers `DeprecationWarning` spam in Jupyter notebooks.

**Mitigation**: Update `__repr__` to be architecture-aware:

```python
def __repr__(self) -> str:
    # Delegate spatial info to geometry, don't access xmin/Nx directly
    geom_info = f"geometry={self.geometry!r}" if self.geometry else "geometry=None"
    return f"MFGProblem({geom_info}, T={self.T}, Nt={self.Nt})"
```

#### 11.6.4 Deepcopy & Multiprocessing

**Risk**: `NetworkGeometry` with large adjacency matrices may:
- Shallow copy incorrectly (shared state bugs)
- Deepcopy inefficiently (memory explosion)

**Mitigation**: Add explicit deepcopy test in Phase 1:

```python
def test_deepcopy_independence():
    import copy
    p1 = MFGProblem(network=my_network, ...)
    p2 = copy.deepcopy(p1)

    # Modify p2's geometry
    p2.geometry.network_data.node_weights[0] = 999.0

    # p1 should be unchanged
    assert p1.geometry.network_data.node_weights[0] != 999.0
```

### 11.7 Updated Implementation Checklist

| Task | Phase | Priority |
|------|-------|----------|
| Implement read-only setters for deprecated attributes | 3 | High |
| Audit `__init__` dependency chain | 1 | High |
| Add deepcopy test for all geometry types | 1 | Medium |
| Update `__repr__` to use geometry delegation | 4 | Medium |
| Test Jupyter notebook warning behavior | 4 | Low |

### 11.8 Geometry Module Consistency Analysis

Full audit of geometry module to verify blueprint consistency.

#### 11.8.1 Geometry Class Hierarchy

**Main Hierarchy** (inheriting from `Geometry` ABC in `base.py:27`):
```
Geometry (ABC)
├── CartesianGrid (ABC, line 489)
│   └── TensorProductGrid (grids/tensor_grid.py:35)
├── UnstructuredMesh (ABC, line 540)
│   └── Mesh1D, Mesh2D, Mesh3D (meshes/)
├── ImplicitGeometry (ABC, line 1038)
│   └── ImplicitDomain (implicit/implicit_domain.py:31)
│       └── Hyperrectangle, Hypersphere, etc.
└── GraphGeometry (ABC, line 1245)
    └── MazeGeometry (graph/maze_generator.py:236)
```

**Network Hierarchy** (now properly inherits from `GraphGeometry` as of PR #434):
```
NetworkGeometry (ABC, network_geometry.py) - inherits from GraphGeometry
├── GridNetwork
├── RandomNetwork
└── ScaleFreeNetwork
```

**Note**: `BaseNetworkGeometry` is a backward compatibility alias for `NetworkGeometry`.

#### 11.8.2 ~~Critical Finding: BaseNetworkGeometry Design Debt~~ [RESOLVED in PR #434]

~~`BaseNetworkGeometry` implements `GeometryProtocol` methods (dimension, geometry_type, num_spatial_points, get_spatial_grid) but **does NOT inherit from `Geometry` or `GraphGeometry`**.~~

**UPDATE (PR #434)**: This design debt has been resolved. `NetworkGeometry` now properly inherits from `GraphGeometry`:

**Current state**:
- `isinstance(network, Geometry)` → **True**
- `isinstance(network, GraphGeometry)` → **True**
- `isinstance(network, GeometryProtocol)` → **True** (structural typing)

**Blueprint Impact**: Phase 1 fix will wire `NetworkGeometry` into `_init_network()`. This works because:
1. `NetworkGeometry` now inherits from `GraphGeometry` (proper class hierarchy)
2. `GeometryProtocol` is `@runtime_checkable` (structural typing also works)
3. `MFGProblem.geometry` is typed as `GeometryProtocol | None`

#### 11.8.3 Protocol Compliance Matrix

| Geometry Type | Inherits `Geometry` | Implements `GeometryProtocol` | Blueprint Compatible |
|---------------|---------------------|-------------------------------|---------------------|
| `TensorProductGrid` | ✅ via CartesianGrid | ✅ | ✅ |
| `Mesh2D`, `Mesh3D` | ✅ via UnstructuredMesh | ✅ | ✅ |
| `ImplicitDomain` | ✅ via ImplicitGeometry | ✅ | ✅ |
| `MazeGeometry` | ✅ via GraphGeometry | ✅ | ✅ |
| `GridNetwork` | ✅ via NetworkGeometry→GraphGeometry | ✅ | ✅ |
| `RandomNetwork` | ✅ via NetworkGeometry→GraphGeometry | ✅ | ✅ |
| `ScaleFreeNetwork` | ✅ via NetworkGeometry→GraphGeometry | ✅ | ✅ |

**Conclusion**: All geometry types now implement `GeometryProtocol` methods AND inherit from `Geometry`. The `NetworkGeometry` hierarchy design debt was resolved in PR #434.

---

## 12. Final Expert Approval (2025-12-11)

### 12.1 Approval Status

**Decision Status**: ✅ **PASSED** - Ready for implementation

The architecture decision report has passed final expert review. The document is now approved for implementation assignment.

### 12.2 Final Polish Items Incorporated

Two minor refinements were added based on final expert review:

#### 12.2.1 Serialization Fallback Mechanism

The `__setstate__` migration logic now includes proper error handling for very old pickle files that may be missing required legacy fields. Uses `try-except` with clear `RuntimeError` instead of raw `KeyError`.

#### 12.2.2 Type Hint Vacuum Prevention

Phase 1 now explicitly includes updating the `geometry` type hint to non-Optional. This ensures IDE autocomplete benefits are available immediately after the code change, rather than waiting until Phase 6.

### 12.3 Conclusion

This ADR answers:
- **What**: Unify around `geometry` attribute (Modified Option B)
- **Why**: Avoid class hierarchy complexity, leverage existing `NetworkGeometry`
- **How**: Wire existing classes, not build new ones
- **How not to fail**: Setter guards, init order audit, serialization migration, deepcopy tests

---

**Implementation Complete** (2025-12-11):

| Phase | PR | Status |
|-------|-----|--------|
| Phase 1: Wire NetworkGeometry | #436 | ✅ |
| Phase 2: Helper properties | #436 | ✅ |
| Phase 3: Attribute hiding | #437 | ✅ |
| Phase 4: Deprecation warnings | #438 | ✅ |
| Phase 5: Serialization | #439, #441 | ✅ |
| Phase 6: Type hints | #440, #441 | ✅ |
| Phase 7: Removal | - | Future (v1.0.0) |

**Remaining Work**:
- Phase 0: User-facing documentation (deferred)
- Phase 7: Remove deprecated attributes in v1.0.0
