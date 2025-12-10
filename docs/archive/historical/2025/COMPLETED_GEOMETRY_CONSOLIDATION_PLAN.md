# Geometry Abstraction Consolidation Plan

**Issue**: #245 Phase 2 - Geometry Architecture Unification
**Date**: 2025-11-09
**Status**: PLANNING

---

## Executive Summary

**Problem**: Current geometry system has fragmented abstractions:
- `GeometryProtocol` (Protocol) - interface definition
- `BaseGeometry` (ABC) - for Gmsh mesh-based geometries only
- Some classes inherit `BaseGeometry`, others implement `GeometryProtocol` directly
- Solvers access geometry through fragile `hasattr()` patterns

**Solution**: Consolidate into single `Geometry` ABC that:
1. Defines universal data interface (dimension, spatial grid, etc.)
2. Adds solver operation interface (Laplacian, gradient, interpolation, BCs)
3. ALL geometry classes inherit from it
4. Eliminates `hasattr()` duck typing in solvers

**Benefits**:
- ✅ Single source of truth for geometry interface
- ✅ Type-safe solver initialization
- ✅ Polymorphic operator dispatch (FD vs FEM vs graph Laplacian)
- ✅ Clear contracts between geometries and solvers
- ✅ Easy to extend with new geometry types

---

## Current State Analysis

### Existing Abstractions

```
GeometryProtocol (Protocol)
    ├── Domain1D (direct implementation)
    ├── TensorProductGrid (direct implementation)
    ├── NetworkGeometry (direct implementation)
    └── BaseGeometry (ABC for mesh-based)
            ├── Domain2D
            ├── Domain3D
            ├── SimpleGrid2D
            └── SimpleGrid3D
```

**Issues**:
1. **Split hierarchy**: Some inherit `BaseGeometry`, others don't
2. **Protocol vs ABC confusion**: `GeometryProtocol` is runtime-checkable Protocol, not ABC
3. **Missing solver operations**: No standard way to get discretization operators
4. **Fragile solver code**: Solvers use `hasattr()` to detect geometry capabilities

### Current GeometryProtocol Interface

```python
class GeometryProtocol(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def geometry_type(self) -> GeometryType: ...

    @property
    def num_spatial_points(self) -> int: ...

    def get_spatial_grid(self) -> np.ndarray | list[np.ndarray]: ...

    def get_problem_config(self) -> dict: ...
```

### Current BaseGeometry Interface

```python
class BaseGeometry(ABC):
    dimension: int
    mesh_data: MeshData | None

    @abstractmethod
    def create_gmsh_geometry(self) -> Any: ...

    @abstractmethod
    def generate_mesh(self) -> MeshData: ...

    @abstractmethod
    def set_mesh_parameters(self, **kwargs) -> None: ...

    # GeometryProtocol methods
    def get_spatial_grid(self) -> NDArray: ...
    @property
    def num_spatial_points(self) -> int: ...
```

**Key observation**: `BaseGeometry` is Gmsh-specific (mesh generation), but tries to satisfy `GeometryProtocol`.

---

## Target Architecture

### Unified Geometry ABC

```python
# mfg_pde/geometry/base.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .geometry_protocol import GeometryType  # Keep enum


class Geometry(ABC):
    """
    Unified abstract base class for all MFG geometries.

    Provides both data interface and solver operation interface,
    eliminating the need for separate protocol/adapter layers.

    All geometry types inherit from this class:
    - Cartesian grids: Domain1D, TensorProductGrid, SimpleGrid2D/3D
    - Unstructured meshes: Domain2D, Domain3D
    - Networks: NetworkGeometry
    - Adaptive meshes: AMR classes
    - Implicit domains: Level set, SDF

    Design Principles:
    1. Geometry objects are responsible for their own discretization
    2. Solvers request operations (Laplacian, gradient), not raw data
    3. Type system enforces solver-geometry compatibility
    """

    # ============================================================================
    # Data Interface (what GeometryProtocol had)
    # ============================================================================

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Spatial dimension of the geometry.

        Returns:
            int: Dimension (1, 2, 3, ..., or 0 for networks)
        """
        ...

    @property
    @abstractmethod
    def geometry_type(self) -> GeometryType:
        """
        Type of geometry (enum).

        Returns:
            GeometryType: CARTESIAN_GRID, NETWORK, DOMAIN_2D, etc.
        """
        ...

    @property
    @abstractmethod
    def num_spatial_points(self) -> int:
        """
        Total number of discrete spatial points.

        Returns:
            int: Number of grid points / mesh vertices / graph nodes
        """
        ...

    @abstractmethod
    def get_spatial_grid(self) -> NDArray:
        """
        Get spatial grid representation.

        Returns:
            For Cartesian grids: (N, d) array of grid points
            For meshes: (N_vertices, d) array of mesh vertices
            For networks: (N_nodes,) array or adjacency representation
        """
        ...

    @abstractmethod
    def get_bounds(self) -> tuple[NDArray, NDArray] | None:
        """
        Get bounding box of geometry.

        Returns:
            (min_coords, max_coords) or None if unbounded/not applicable
        """
        ...

    # ============================================================================
    # Solver Operation Interface (NEW - eliminates hasattr patterns)
    # ============================================================================

    @abstractmethod
    def get_laplacian_operator(self) -> Callable:
        """
        Return discretized Laplacian operator for this geometry.

        Returns:
            Function with signature: (u: NDArray, point_idx) -> float

        Example:
            >>> laplacian = geometry.get_laplacian_operator()
            >>> lap_value = laplacian(u_array, (i, j))  # 2D grid index

        Implementation notes:
            - Cartesian grids: Finite difference Laplacian
            - Unstructured meshes: Finite element Laplacian (mass matrix)
            - Networks: Graph Laplacian
        """
        ...

    @abstractmethod
    def get_gradient_operator(self) -> Callable:
        """
        Return discretized gradient operator for this geometry.

        Returns:
            Function with signature: (u: NDArray, point_idx) -> NDArray

        Example:
            >>> gradient = geometry.get_gradient_operator()
            >>> grad_u = gradient(u_array, (i, j))  # Returns [du/dx, du/dy]

        Implementation notes:
            - Cartesian grids: Finite difference gradient
            - Unstructured meshes: Finite element gradient
            - Networks: Discrete gradient along edges
        """
        ...

    @abstractmethod
    def get_interpolator(self) -> Callable:
        """
        Return interpolation function for this geometry.

        Returns:
            Function with signature: (u: NDArray, point: NDArray) -> float

        Example:
            >>> interpolate = geometry.get_interpolator()
            >>> value = interpolate(u_array, np.array([0.5, 0.3]))

        Implementation notes:
            - Cartesian grids: Linear/bilinear/trilinear interpolation
            - Unstructured meshes: Barycentric interpolation
            - Networks: Nearest node or graph-based interpolation
        """
        ...

    @abstractmethod
    def get_boundary_handler(self):
        """
        Return boundary condition handler for this geometry.

        Returns:
            BoundaryHandler: Object that applies boundary conditions

        Implementation notes:
            - Handles Dirichlet, Neumann, periodic, Robin BCs
            - Geometry-specific boundary detection and application
        """
        ...

    # ============================================================================
    # Grid/Mesh Utilities (for geometries that need them)
    # ============================================================================

    def get_grid_spacing(self) -> list[float] | None:
        """
        Get grid spacing for regular Cartesian grids.

        Returns:
            [dx1, dx2, ...] or None if not a regular grid

        Default implementation returns None (override in Cartesian grid classes).
        """
        return None

    def get_grid_shape(self) -> tuple[int, ...] | None:
        """
        Get grid shape for regular Cartesian grids.

        Returns:
            (Nx, Ny, ...) or None if not a regular grid

        Default implementation returns None (override in Cartesian grid classes).
        """
        return None

    # ============================================================================
    # MFGProblem Integration
    # ============================================================================

    @abstractmethod
    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        Polymorphic method allowing each geometry type to specify how it
        configures MFGProblem, avoiding hasattr checks and duck typing.

        Returns:
            Dictionary with keys (geometry-dependent):
                - num_spatial_points: int
                - spatial_shape: tuple
                - spatial_bounds: list[tuple] or None
                - spatial_discretization: list[int] or None
                - Additional geometry-specific data
        """
        ...
```

### Inheritance Structure After Consolidation

```
Geometry (ABC)
    ├── CartesianGrid (ABC for regular grids)
    │       ├── Domain1D
    │       ├── TensorProductGrid
    │       ├── SimpleGrid2D
    │       └── SimpleGrid3D
    │
    ├── UnstructuredMesh (ABC for FEM-style meshes)
    │       ├── Domain2D
    │       └── Domain3D
    │
    ├── NetworkGraph (ABC for graph-based)
    │       └── NetworkGeometry
    │
    └── AdaptiveMesh (ABC for AMR)
            ├── AMR1D
            ├── AMRQuadtree2D
            └── AMRTetrahedral3D
```

**Benefits of intermediate ABCs**:
- `CartesianGrid` provides `get_grid_spacing()`, `get_grid_shape()` implementations
- `UnstructuredMesh` provides Gmsh pipeline methods
- `NetworkGraph` provides graph Laplacian methods
- Shared code lives in appropriate layer

---

## Migration Plan

### Phase 1: Create Unified Base (Week 1)

**Step 1.1**: Create new `mfg_pde/geometry/base.py` with `Geometry` ABC
- Copy `GeometryProtocol` interface
- Add solver operation abstract methods
- Add grid/mesh utility methods with default `None` returns

**Step 1.2**: Create intermediate ABCs
- `CartesianGrid(Geometry)` - for regular grids
- `UnstructuredMesh(Geometry)` - for FEM meshes
- `NetworkGraph(Geometry)` - for graph geometries

**Step 1.3**: Update imports in `mfg_pde/geometry/__init__.py`
```python
# Old
from .geometry_protocol import GeometryProtocol, GeometryType
from .base_geometry import BaseGeometry, MeshData

# New
from .base import Geometry, GeometryType, MeshData
from .base import CartesianGrid, UnstructuredMesh, NetworkGraph
```

**Deliverable**: New base classes defined, no existing code broken yet

---

### Phase 2: Migrate Cartesian Grid Classes (Week 1-2)

**Step 2.1**: Migrate `TensorProductGrid`
```python
# Before
class TensorProductGrid:
    # Implements GeometryProtocol manually
    ...

# After
class TensorProductGrid(CartesianGrid):
    # Inherits from CartesianGrid

    def get_laplacian_operator(self) -> Callable:
        def laplacian_fd(u: NDArray, idx: tuple) -> float:
            # Finite difference Laplacian
            return self._compute_fd_laplacian(u, idx)
        return laplacian_fd

    def get_gradient_operator(self) -> Callable:
        def gradient_fd(u: NDArray, idx: tuple) -> NDArray:
            # Finite difference gradient
            return self._compute_fd_gradient(u, idx)
        return gradient_fd

    # ... other methods
```

**Step 2.2**: Migrate `Domain1D`, `SimpleGrid2D`, `SimpleGrid3D`
- Change from implementing `GeometryProtocol` → inheriting `CartesianGrid`
- Implement solver operation methods
- Add finite difference operators

**Deliverable**: All Cartesian grid classes use unified base

---

### Phase 3: Migrate Mesh Classes (Week 2)

**Step 3.1**: Refactor `BaseGeometry` → `UnstructuredMesh(Geometry)`
```python
# mfg_pde/geometry/base.py

class UnstructuredMesh(Geometry):
    """ABC for unstructured FEM-style meshes."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.mesh_data: MeshData | None = None
        self._gmsh_model = None

    # Gmsh pipeline methods (from old BaseGeometry)
    @abstractmethod
    def create_gmsh_geometry(self) -> Any: ...

    @abstractmethod
    def generate_mesh(self) -> MeshData: ...

    # Solver operations (NEW)
    def get_laplacian_operator(self) -> Callable:
        """FEM Laplacian using mass/stiffness matrices."""
        def laplacian_fem(u: NDArray, vertex_idx: int) -> float:
            # Finite element Laplacian
            return self._compute_fem_laplacian(u, vertex_idx)
        return laplacian_fem

    # ... other methods
```

**Step 3.2**: Update `Domain2D`, `Domain3D` to inherit `UnstructuredMesh`
```python
# Before
class Domain2D(BaseGeometry):
    ...

# After
class Domain2D(UnstructuredMesh):
    ...
```

**Deliverable**: Mesh classes migrated, old `BaseGeometry` deprecated

---

### Phase 4: Migrate Network Classes (Week 2)

**Step 4.1**: Create `NetworkGraph(Geometry)` intermediate ABC

**Step 4.2**: Migrate `NetworkGeometry`
```python
class NetworkGeometry(NetworkGraph):
    def get_laplacian_operator(self) -> Callable:
        """Graph Laplacian."""
        L = self.network_data.laplacian_matrix
        def laplacian_graph(u: NDArray, node_idx: int) -> float:
            return (L @ u)[node_idx]
        return laplacian_graph

    def get_gradient_operator(self) -> Callable:
        """Discrete gradient along edges."""
        def gradient_graph(u: NDArray, node_idx: int) -> NDArray:
            adj = self.network_data.adjacency_matrix
            neighbors = adj[node_idx].nonzero()[1]
            return u[neighbors] - u[node_idx]
        return gradient_graph
```

**Deliverable**: Network classes migrated

---

### Phase 5: Update Solvers (Week 3)

**Step 5.1**: Update type hints in solvers
```python
# Before
class HJBFDMSolver:
    def __init__(self, problem: MFGProblem):
        # hasattr() checks...
        if hasattr(problem, "dimension"):
            d = problem.dimension
        ...

# After
from mfg_pde.geometry import CartesianGrid

class HJBFDMSolver:
    def __init__(self, problem: MFGProblemProtocol):
        # Type-safe geometry check
        if not isinstance(problem.geometry, CartesianGrid):
            raise TypeError(
                f"HJB FDM requires CartesianGrid, got {type(problem.geometry)}"
            )

        geom: CartesianGrid = problem.geometry
        self.laplacian = geom.get_laplacian_operator()
        self.gradient = geom.get_gradient_operator()
        self.dx = geom.get_grid_spacing()  # Guaranteed to exist
        self.dimension = geom.dimension
```

**Step 5.2**: Update 1-2 solvers as proof of concept
- `HJBFDMSolver` (requires `CartesianGrid`)
- `HJBGFDMSolver` (can work with any `Geometry`)

**Deliverable**: Solver pattern established

---

### Phase 6: Deprecation & Cleanup (Week 3)

**Step 6.1**: Add deprecation warnings
```python
# mfg_pde/geometry/geometry_protocol.py

import warnings

class GeometryProtocol(Protocol):
    """DEPRECATED: Use Geometry ABC instead."""

    def __init_subclass__(cls):
        warnings.warn(
            f"{cls.__name__} implements deprecated GeometryProtocol. "
            f"Inherit from mfg_pde.geometry.Geometry instead. "
            f"GeometryProtocol will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2
        )
    ...

# mfg_pde/geometry/base_geometry.py

class BaseGeometry(UnstructuredMesh):
    """DEPRECATED: Use UnstructuredMesh instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BaseGeometry is deprecated. Use UnstructuredMesh instead. "
            "BaseGeometry will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

**Step 6.2**: Update all documentation
- README examples
- Tutorial notebooks
- API reference

**Step 6.3**: Remove deprecated classes in v1.0.0

**Deliverable**: Clean migration path documented

---

## Backward Compatibility Strategy

### Import Aliases (Temporary)

```python
# mfg_pde/geometry/__init__.py

from .base import Geometry, CartesianGrid, UnstructuredMesh, NetworkGraph

# Backward compatibility (deprecated)
GeometryProtocol = Geometry  # Alias
BaseGeometry = UnstructuredMesh  # Alias

__all__ = [
    "Geometry",
    "CartesianGrid",
    "UnstructuredMesh",
    "NetworkGraph",
    # Deprecated (remove in v1.0.0)
    "GeometryProtocol",
    "BaseGeometry",
]
```

### Versioning Timeline

| Version | Phase | Changes |
|:--------|:------|:--------|
| **v0.11.0** | Phase 1-2 | New `Geometry` ABC, Cartesian grids migrated |
| **v0.12.0** | Phase 3-4 | Mesh and network classes migrated |
| **v0.13.0** | Phase 5 | Solvers updated, deprecation warnings |
| **v0.14.0** | Phase 6 | Documentation updated, cleanup |
| **v1.0.0** | Final | Remove `GeometryProtocol`, `BaseGeometry` |

---

## Testing Strategy

### Unit Tests

**Test 1**: Verify all geometry classes satisfy `Geometry` ABC
```python
def test_geometry_protocol_compliance():
    from mfg_pde.geometry import (
        Domain1D, TensorProductGrid, Domain2D, NetworkGeometry, Geometry
    )

    # 1D
    domain = Domain1D(xmin=0, xmax=1, boundary_conditions="periodic")
    assert isinstance(domain, Geometry)

    # nD grid
    grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], num_points=[10, 10])
    assert isinstance(grid, Geometry)

    # Mesh
    mesh = Domain2D(domain_type="rectangle", bounds=(0, 1, 0, 1))
    assert isinstance(mesh, Geometry)

    # Network
    network = NetworkGeometry(topology="grid", n_nodes=100)
    assert isinstance(network, Geometry)
```

**Test 2**: Verify solver operation methods exist
```python
def test_solver_operations():
    grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], num_points=[10, 10])

    # Check required methods
    assert callable(grid.get_laplacian_operator())
    assert callable(grid.get_gradient_operator())
    assert callable(grid.get_interpolator())
    assert grid.get_boundary_handler() is not None
```

**Test 3**: Verify backward compatibility
```python
def test_backward_compatibility():
    from mfg_pde.geometry import GeometryProtocol, BaseGeometry

    # Should work but emit deprecation warning
    with pytest.warns(DeprecationWarning):
        class OldStyleGeometry(BaseGeometry):
            ...
```

### Integration Tests

**Test 4**: Verify solver-geometry integration
```python
def test_fdm_solver_with_cartesian_grid():
    from mfg_pde import MFGProblem
    from mfg_pde.geometry import TensorProductGrid
    from mfg_pde.solvers import HJBFDMSolver

    grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], num_points=[50, 50])
    problem = MFGProblem(geometry=grid, T=1.0, Nt=100)

    solver = HJBFDMSolver(problem)  # Should work
    assert solver.dimension == 2
    assert solver.dx is not None
```

**Test 5**: Verify type checking catches incompatible geometries
```python
def test_fdm_rejects_network():
    from mfg_pde import MFGProblem
    from mfg_pde.geometry import NetworkGeometry
    from mfg_pde.solvers import HJBFDMSolver

    network = NetworkGeometry(topology="random", n_nodes=100)
    problem = MFGProblem(geometry=network, T=1.0, Nt=100)

    with pytest.raises(TypeError, match="requires CartesianGrid"):
        HJBFDMSolver(problem)
```

---

## File Structure After Migration

```
mfg_pde/geometry/
├── __init__.py                  # Exports Geometry, intermediate ABCs
├── base.py                      # NEW: Geometry ABC + intermediate ABCs
│                                # (replaces geometry_protocol.py + base_geometry.py)
├── boundary_handler.py          # NEW: Boundary condition interface
│
├── cartesian/                   # Cartesian grid implementations
│   ├── domain_1d.py            # Domain1D(CartesianGrid)
│   ├── tensor_product_grid.py  # TensorProductGrid(CartesianGrid)
│   ├── simple_grid.py          # SimpleGrid2D/3D(CartesianGrid)
│   └── amr_1d.py               # AMR1D(CartesianGrid)
│
├── mesh/                        # Unstructured mesh implementations
│   ├── domain_2d.py            # Domain2D(UnstructuredMesh)
│   ├── domain_3d.py            # Domain3D(UnstructuredMesh)
│   ├── amr_quadtree_2d.py      # AMRQuadtree2D(UnstructuredMesh)
│   └── amr_tetrahedral_3d.py   # AMRTetrahedral3D(UnstructuredMesh)
│
├── network/                     # Network/graph implementations
│   ├── network_geometry.py     # NetworkGeometry(NetworkGraph)
│   └── network_backend.py      # Backend abstraction
│
├── implicit/                    # Implicit geometry (level set, SDF)
│   ├── hyperrectangle.py
│   ├── hypersphere.py
│   └── csg_operations.py
│
└── deprecated/                  # Deprecated classes (remove in v1.0.0)
    ├── geometry_protocol.py    # DEPRECATED: Use Geometry
    └── base_geometry.py        # DEPRECATED: Use UnstructuredMesh
```

---

## Risk Assessment

### Low Risk
- ✅ New `Geometry` ABC doesn't break existing code initially
- ✅ Import aliases maintain backward compatibility
- ✅ Gradual migration reduces regression risk

### Medium Risk
- ⚠️ Solvers need updates to use geometry operations
- ⚠️ Tests need updates for new interfaces
- ⚠️ Examples/docs need updates

### High Risk
- ❌ External code depending on `GeometryProtocol` will break in v1.0.0
- ❌ Solver operation implementation bugs could affect correctness

### Mitigation
1. **Extensive testing**: Unit + integration tests for all geometry types
2. **Gradual rollout**: 5-6 minor versions before v1.0.0 breaking change
3. **Clear deprecation warnings**: Users have time to migrate
4. **Documentation**: Migration guide with examples

---

## Success Criteria

| Criterion | Metric | Target |
|:----------|:-------|:-------|
| All geometries inherit `Geometry` | Class count | 100% |
| Solver operations implemented | Method coverage | 100% |
| Backward compatibility maintained | Import aliases work | Until v1.0.0 |
| Solvers use type-safe checks | `hasattr()` count | 0 |
| Tests pass | Test coverage | >95% |
| Documentation updated | Outdated examples | 0 |

---

## Next Steps

1. **Review this plan** - Get approval on architecture
2. **Create feature branch**: `feature/geometry-consolidation`
3. **Phase 1 implementation**: Create `Geometry` ABC
4. **Proof of concept**: Migrate `TensorProductGrid` + update one solver
5. **Iterate**: Complete remaining phases

---

**Last Updated**: 2025-11-09
**Related**: Issue #245, PHASE1_COMPLETION_SUMMARY.md, PROTOCOL_REVISION_GEOMETRY_AGNOSTIC.md
