# Unified Geometry Parameter for MFGProblem

**Created**: 2025-11-05
**Status**: Design Phase
**Related**: Phase 3 dimension-agnostic architecture

## Motivation

Current MFGProblem only supports rectangular Cartesian grids via `spatial_bounds` + `spatial_discretization`. Complex geometries (mazes, graphs, irregular domains) require specialized classes.

**User Request**: "use geometry, which will accept all geometry, including rectangle grids, maze, graph, etc... if hard to be self-aware then use flag?"

## Design Goals

1. **Unified interface**: Single `geometry` parameter accepting all geometry types
2. **Self-awareness**: Automatic geometry type detection where possible
3. **Optional flag**: `geometry_type` parameter for disambiguation when needed
4. **Backward compatibility**: Existing `spatial_bounds`/`spatial_discretization` interface preserved
5. **Solver compatibility**: Automatic detection based on geometry type

## Geometry Type Hierarchy

```python
from typing import Protocol, Union
from enum import Enum

class GeometryType(Enum):
    """Geometry type enumeration."""
    CARTESIAN_GRID = "cartesian_grid"  # Rectangular tensor product grid
    NETWORK = "network"                 # Graph/network structure
    MAZE = "maze"                       # Maze with obstacles
    DOMAIN_2D = "domain_2d"             # Complex 2D domain (gmsh)
    DOMAIN_3D = "domain_3d"             # Complex 3D domain (gmsh)
    IMPLICIT = "implicit"               # Implicit geometry (level set, SDF)
    CUSTOM = "custom"                   # User-defined geometry

class GeometryProtocol(Protocol):
    """Protocol that all geometry objects must satisfy."""

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        ...

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry for solver compatibility."""
        ...

    def sample_points(self, n_points: int) -> NDArray:
        """Sample n_points uniformly from geometry."""
        ...

    def contains(self, points: NDArray) -> NDArray:
        """Check if points are inside geometry."""
        ...
```

## API Design

### Option 1: Single `geometry` Parameter (Recommended)

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import CartesianGrid, NetworkGeometry, MazeGeometry

# 1. Cartesian grid (NEW: explicit geometry object)
grid = CartesianGrid(
    bounds=[(0, 1), (0, 1)],
    discretization=[50, 50]
)
problem = MFGProblem(
    geometry=grid,
    T=1.0, Nt=100,
    sigma=0.1
)

# 2. Network/graph (class-based, consistent with other geometries)
network = NetworkGeometry(
    topology="scale_free",
    n_nodes=100,
    avg_degree=4
)
problem = MFGProblem(
    geometry=network,
    time_interval=(0.0, 1.0),  # Support arbitrary intervals (t_0, t_T)
    Nt=100,
    sigma=0.1
)

# 3. Maze with obstacles
maze = MazeGeometry(
    bounds=[(0, 10), (0, 10)],
    walls=[(3, 3, 3, 7), (7, 3, 7, 7)],  # (x1, y1, x2, y2)
    discretization=[100, 100]
)
problem = MFGProblem(
    geometry=maze,
    time_interval=(0.0, 1.0),
    Nt=100,
    sigma=0.1
)

# 4. Complex 2D domain (gmsh)
from mfg_pde.geometry import Domain2D
domain = Domain2D.from_gmsh("building.msh")
problem = MFGProblem(
    geometry=domain,
    time_interval=(0.0, 1.0),
    Nt=100,
    sigma=0.1
)

# 5. Implicit geometry (level set / signed distance function)
from mfg_pde.geometry import ImplicitGeometry

def sphere_sdf(x):
    """Signed distance function for sphere of radius R=1."""
    import numpy as np
    return np.linalg.norm(x) - 1.0

implicit_geom = ImplicitGeometry(
    sdf=sphere_sdf,
    bounds=[(-2, 2), (-2, 2), (-2, 2)],  # Bounding box
    dimension=3
)
problem = MFGProblem(
    geometry=implicit_geom,
    time_interval=(0.0, 2.0),
    Nt=200,
    sigma=0.05
)

# BACKWARD COMPATIBLE: Old interface still works
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    T=1.0, Nt=100,
    sigma=0.1
)
# Internally creates CartesianGrid automatically
```

### Option 2: With Optional Type Flag

```python
# For ambiguous cases where auto-detection might fail
problem = MFGProblem(
    geometry=custom_object,
    geometry_type=GeometryType.NETWORK,  # Explicit type hint
    T=1.0, Nt=100,
    sigma=0.1
)
```

## Implementation Strategy

### Phase 1: Geometry Abstraction Layer

**File**: `mfg_pde/geometry/base_geometry.py`

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable

class GeometryType(Enum):
    CARTESIAN_GRID = "cartesian_grid"
    NETWORK = "network"
    MAZE = "maze"
    DOMAIN_2D = "domain_2d"
    DOMAIN_3D = "domain_3d"
    CUSTOM = "custom"

@runtime_checkable
class GeometryProtocol(Protocol):
    """Minimal protocol for geometry objects."""
    dimension: int
    geometry_type: GeometryType

    def sample_points(self, n_points: int) -> NDArray: ...
    def contains(self, points: NDArray) -> NDArray: ...

class BaseGeometry(ABC):
    """Base class for all geometry types."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension."""
        pass

    @property
    @abstractmethod
    def geometry_type(self) -> GeometryType:
        """Type identifier for solver compatibility."""
        pass

    @abstractmethod
    def sample_points(self, n_points: int) -> NDArray:
        """Uniform sampling from geometry."""
        pass

    @abstractmethod
    def contains(self, points: NDArray) -> NDArray:
        """Boundary check."""
        pass

    @abstractmethod
    def get_solver_compatibility(self) -> dict[str, bool]:
        """Which solvers work with this geometry."""
        pass
```

### Phase 2: Cartesian Grid Wrapper

**File**: `mfg_pde/geometry/cartesian_grid.py`

```python
class CartesianGrid(BaseGeometry):
    """Rectangular tensor product grid."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        discretization: list[int]
    ):
        self.bounds = bounds
        self.discretization = discretization
        self._dimension = len(bounds)
        self._build_grid()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.CARTESIAN_GRID

    def sample_points(self, n_points: int) -> NDArray:
        """Uniform random sampling from grid."""
        samples = []
        for (xmin, xmax) in self.bounds:
            samples.append(np.random.uniform(xmin, xmax, n_points))
        return np.column_stack(samples)

    def contains(self, points: NDArray) -> NDArray:
        """Check if points are in bounding box."""
        inside = np.ones(len(points), dtype=bool)
        for i, (xmin, xmax) in enumerate(self.bounds):
            inside &= (points[:, i] >= xmin) & (points[:, i] <= xmax)
        return inside

    def get_solver_compatibility(self) -> dict[str, bool]:
        return {
            "fdm": True,
            "semi_lagrangian": True,
            "particle": True,
            "gfdm": True,
            "dgm": self.dimension <= 3,
            "network_solver": False
        }
```

### Phase 3: MFGProblem Integration

**File**: `mfg_pde/core/mfg_problem.py`

```python
from typing import Union
from mfg_pde.geometry.base_geometry import GeometryProtocol, GeometryType, CartesianGrid

class MFGProblem(BaseMFGProblem):
    def __init__(
        self,
        # NEW: Unified geometry parameter
        geometry: GeometryProtocol | None = None,
        geometry_type: GeometryType | None = None,  # Optional hint

        # OLD: Backward compatible parameters
        spatial_bounds: list[tuple[float, float]] | None = None,
        spatial_discretization: list[int] | None = None,

        # NEW: Time interval support (t_0, t_T)
        time_interval: tuple[float, float] | None = None,  # (t_0, t_T)

        # OLD: Backward compatible time parameters
        T: float | None = None,  # Terminal time (assumes t_0 = 0)
        Nt: int | None = None,  # Number of timesteps

        # ... other parameters
        **kwargs
    ):
        # Geometry mode detection
        if geometry is not None:
            # NEW MODE: Explicit geometry object
            self._init_from_geometry(geometry, geometry_type)
        elif spatial_bounds is not None or spatial_discretization is not None:
            # OLD MODE: Backward compatible Cartesian grid
            self._init_from_grid_params(spatial_bounds, spatial_discretization)
        else:
            # DEFAULT: Use defaults and create CartesianGrid
            self._init_default_geometry()

        # Time interval processing
        if time_interval is not None:
            # NEW MODE: Explicit time interval (t_0, t_T)
            self.t_initial, self.t_terminal = time_interval
            self.T = self.t_terminal - self.t_initial  # Time horizon length
        elif T is not None:
            # OLD MODE: Assume t_0 = 0
            self.t_initial = 0.0
            self.t_terminal = T
            self.T = T
        else:
            # DEFAULT: [0, 1]
            self.t_initial = 0.0
            self.t_terminal = 1.0
            self.T = 1.0

        self.Nt = Nt if Nt is not None else 100  # Default timesteps
        self.Dt = self.T / self.Nt  # Timestep size

        # ... rest of initialization

    def _init_from_geometry(
        self,
        geometry: GeometryProtocol,
        geometry_type: GeometryType | None
    ):
        """Initialize from explicit geometry object."""
        # Type detection
        if not isinstance(geometry, GeometryProtocol):
            # Try to infer type from object attributes
            detected_type = self._detect_geometry_type(geometry)
            if geometry_type is not None and geometry_type != detected_type:
                warnings.warn(
                    f"Provided geometry_type={geometry_type} differs from "
                    f"detected type={detected_type}. Using provided type."
                )
                self.geometry_type = geometry_type
            else:
                self.geometry_type = detected_type
        else:
            self.geometry_type = geometry.geometry_type

        self.geometry = geometry
        self.dimension = geometry.dimension

        # Extract spatial_bounds/discretization if available (for Cartesian grids)
        if hasattr(geometry, 'bounds'):
            self.spatial_bounds = geometry.bounds
        if hasattr(geometry, 'discretization'):
            self.spatial_discretization = geometry.discretization

        # Update solver compatibility
        self._update_solver_compatibility()

    def _detect_geometry_type(self, obj) -> GeometryType:
        """Heuristic geometry type detection."""
        # Check for known types
        if hasattr(obj, 'geometry_type'):
            return obj.geometry_type

        # Check for NetworkGeometry attributes
        if hasattr(obj, 'adjacency_matrix') or hasattr(obj, 'edges'):
            return GeometryType.NETWORK

        # Check for maze attributes
        if hasattr(obj, 'walls') or hasattr(obj, 'obstacles'):
            return GeometryType.MAZE

        # Check for gmsh/mesh attributes
        if hasattr(obj, 'mesh') or hasattr(obj, 'nodes'):
            if obj.dimension == 2:
                return GeometryType.DOMAIN_2D
            elif obj.dimension == 3:
                return GeometryType.DOMAIN_3D

        # Default to custom
        return GeometryType.CUSTOM

    def _init_from_grid_params(
        self,
        bounds: list[tuple[float, float]] | None,
        discretization: list[int] | None
    ):
        """Backward compatible: Create CartesianGrid from old parameters."""
        # Fill in defaults if needed
        if bounds is None:
            bounds = [(0, 1)]
        if discretization is None:
            discretization = [100]

        # Create CartesianGrid internally
        grid = CartesianGrid(bounds=bounds, discretization=discretization)
        self._init_from_geometry(grid, geometry_type=None)

    def _update_solver_compatibility(self):
        """Update solver recommendations based on geometry."""
        if hasattr(self.geometry, 'get_solver_compatibility'):
            compat = self.geometry.get_solver_compatibility()
            self.solver_compatible = [k for k, v in compat.items() if v]
        else:
            # Default compatibility (particle works with everything)
            self.solver_compatible = ["particle"]

        # Update recommendations
        if self.geometry_type == GeometryType.CARTESIAN_GRID:
            self.solver_recommendations["default"] = "fdm"
        elif self.geometry_type == GeometryType.NETWORK:
            self.solver_recommendations["default"] = "network_solver"
        elif self.geometry_type in [GeometryType.MAZE, GeometryType.DOMAIN_2D, GeometryType.DOMAIN_3D]:
            self.solver_recommendations["default"] = "particle"
        else:
            self.solver_recommendations["default"] = "particle"
```

### Phase 4: Geometry Implementations

#### MazeGeometry

**File**: `mfg_pde/geometry/maze_geometry.py`

```python
class MazeGeometry(BaseGeometry):
    """Maze with obstacles on Cartesian grid."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        walls: list[tuple[float, float, float, float]],  # (x1, y1, x2, y2)
        discretization: list[int]
    ):
        self.bounds = bounds
        self.walls = walls
        self.discretization = discretization
        self._dimension = len(bounds)
        self._build_occupancy_grid()

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.MAZE

    def contains(self, points: NDArray) -> NDArray:
        """Check if points are in free space (not in walls)."""
        inside_bounds = super().contains(points)  # Bounding box check
        not_in_walls = ~self._intersects_walls(points)
        return inside_bounds & not_in_walls

    def get_solver_compatibility(self) -> dict[str, bool]:
        return {
            "fdm": False,  # Standard FDM doesn't handle obstacles well
            "particle": True,  # Particle methods excel here
            "gfdm": True,  # GFDM can handle irregular domains
            "network_solver": False
        }
```

#### NetworkGeometry (class-based consistent with other geometries)

**File**: `mfg_pde/geometry/network_geometry.py`

```python
class NetworkGeometry(BaseGeometry):
    """Graph/network geometry for MFG on graphs."""

    def __init__(
        self,
        topology: str = "scale_free",
        n_nodes: int = 100,
        avg_degree: float = 4.0,
        adjacency_matrix: np.ndarray | None = None,
        **kwargs
    ):
        """
        Initialize network geometry.

        Args:
            topology: Network type ("scale_free", "erdos_renyi", "small_world", "custom")
            n_nodes: Number of nodes
            avg_degree: Average degree (edges per node)
            adjacency_matrix: Optional custom adjacency matrix
            **kwargs: Additional parameters for network generation
        """
        self.topology = topology
        self.n_nodes = n_nodes
        self.avg_degree = avg_degree

        if adjacency_matrix is not None:
            self.adjacency = adjacency_matrix
        else:
            self.adjacency = self._generate_network()

        self._dimension = 0  # Networks are zero-dimensional in Euclidean sense

    @property
    def dimension(self) -> int:
        """Return 0 for graph (non-Euclidean)."""
        return 0

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.NETWORK

    def sample_points(self, n_points: int) -> NDArray:
        """Sample random nodes from network."""
        return np.random.choice(self.n_nodes, size=n_points, replace=True)

    def contains(self, points: NDArray) -> NDArray:
        """Check if node indices are valid."""
        return (points >= 0) & (points < self.n_nodes)

    def get_solver_compatibility(self) -> dict[str, bool]:
        return {
            "fdm": False,
            "particle": True,  # Particle methods can handle graphs
            "network_solver": True,  # Specialized network MFG solver
            "gfdm": False,
        }

    def _generate_network(self) -> np.ndarray:
        """Generate adjacency matrix based on topology."""
        if self.topology == "scale_free":
            # Use preferential attachment
            import networkx as nx
            G = nx.barabasi_albert_graph(self.n_nodes, int(self.avg_degree / 2))
            return nx.to_numpy_array(G)
        elif self.topology == "erdos_renyi":
            # Random graph with fixed edge probability
            import networkx as nx
            p = self.avg_degree / (self.n_nodes - 1)
            G = nx.erdos_renyi_graph(self.n_nodes, p)
            return nx.to_numpy_array(G)
        else:
            raise ValueError(f"Unknown network topology: {self.topology}")

# Factory function for backward compatibility
def create_network(topology: str, **kwargs) -> NetworkGeometry:
    """
    Factory function for creating network geometries.

    Backward compatible with existing create_network() usage.
    """
    return NetworkGeometry(topology=topology, **kwargs)
```

#### ImplicitGeometry (level sets and signed distance functions)

**File**: `mfg_pde/geometry/implicit_geometry.py`

```python
class ImplicitGeometry(BaseGeometry):
    """
    Implicit geometry defined by level set or signed distance function.

    The geometry is defined by the zero level set of a function φ(x):
        Ω = {x ∈ ℝ^d : φ(x) ≤ 0}

    For signed distance functions (SDFs):
        φ(x) = signed distance to boundary ∂Ω
        φ(x) < 0 inside, φ(x) = 0 on boundary, φ(x) > 0 outside

    Examples:
        - Sphere: φ(x) = ||x - c|| - R
        - Torus: φ(x,y,z) = (√(x²+y²) - R)² + z² - r²
        - Union/intersection via min/max operations
    """

    def __init__(
        self,
        sdf: Callable[[np.ndarray], float | np.ndarray],
        bounds: list[tuple[float, float]],
        dimension: int | None = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize implicit geometry.

        Args:
            sdf: Signed distance function φ(x) or level set function
            bounds: Bounding box for sampling/discretization
            dimension: Spatial dimension (inferred from bounds if None)
            tolerance: Tolerance for boundary detection (|φ| < tol)
        """
        self.sdf = sdf
        self.bounds = bounds
        self._dimension = dimension if dimension is not None else len(bounds)
        self.tolerance = tolerance

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.IMPLICIT

    def sample_points(self, n_points: int) -> NDArray:
        """
        Sample points inside implicit geometry using rejection sampling.

        Generate random points in bounding box and keep only those
        satisfying φ(x) ≤ 0.
        """
        samples = []
        attempts = 0
        max_attempts = n_points * 100  # Prevent infinite loop

        while len(samples) < n_points and attempts < max_attempts:
            # Generate batch of candidate points
            batch_size = min(n_points * 10, max_attempts - attempts)
            candidates = np.column_stack([
                np.random.uniform(low, high, batch_size)
                for (low, high) in self.bounds
            ])

            # Filter points inside geometry
            inside = self.sdf(candidates.T) <= self.tolerance
            samples.extend(candidates[inside])
            attempts += batch_size

        if len(samples) < n_points:
            raise RuntimeError(
                f"Failed to sample {n_points} points from implicit geometry. "
                f"Only found {len(samples)} after {attempts} attempts. "
                f"Consider enlarging bounds or checking SDF."
            )

        return np.array(samples[:n_points])

    def contains(self, points: NDArray) -> NDArray:
        """Check if points are inside geometry (φ(x) ≤ 0)."""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return self.sdf(points.T) <= self.tolerance

    def get_solver_compatibility(self) -> dict[str, bool]:
        return {
            "fdm": False,  # Standard FDM requires structured grid
            "particle": True,  # Particle methods excel with implicit geometries
            "gfdm": True,  # GFDM can handle point clouds
            "dgm": True,  # DG methods can work with implicit boundaries
            "network_solver": False
        }

    def project_to_boundary(self, points: NDArray) -> NDArray:
        """
        Project points to boundary ∂Ω using gradient descent on |φ(x)|.

        Useful for initialization and boundary condition enforcement.
        """
        # Simple gradient descent implementation
        x = points.copy()
        learning_rate = 0.1
        max_iter = 100

        for _ in range(max_iter):
            phi_val = self.sdf(x.T)
            if np.abs(phi_val).max() < self.tolerance:
                break

            # Finite difference gradient
            eps = 1e-6
            grad = np.zeros_like(x)
            for i in range(x.shape[1]):
                x_plus = x.copy()
                x_plus[:, i] += eps
                grad[:, i] = (self.sdf(x_plus.T) - phi_val) / eps

            # Gradient descent step
            x -= learning_rate * grad * np.sign(phi_val)[:, None]

        return x
```

**Example Usage**:

```python
# Sphere
def sphere_sdf(x):
    return np.linalg.norm(x, axis=0) - 1.0

sphere = ImplicitGeometry(
    sdf=sphere_sdf,
    bounds=[(-2, 2), (-2, 2), (-2, 2)],
    dimension=3
)

# Torus
def torus_sdf(x):
    R, r = 2.0, 0.5  # Major/minor radii
    x_vals, y_vals, z_vals = x[0], x[1], x[2]
    q = np.sqrt(x_vals**2 + y_vals**2) - R
    return np.sqrt(q**2 + z_vals**2) - r

torus = ImplicitGeometry(
    sdf=torus_sdf,
    bounds=[(-3, 3), (-3, 3), (-1, 1)],
    dimension=3
)

# Union of geometries (via min)
def union_sdf(x):
    sphere1 = np.linalg.norm(x - np.array([[0], [0], [0]]), axis=0) - 1.0
    sphere2 = np.linalg.norm(x - np.array([[1.5], [0], [0]]), axis=0) - 0.8
    return np.minimum(sphere1, sphere2)

union_geom = ImplicitGeometry(
    sdf=union_sdf,
    bounds=[(-2, 4), (-2, 2), (-2, 2)],
    dimension=3
)
```

## Solver Compatibility Matrix

| Geometry Type | FDM | Semi-Lagrangian | Particle | GFDM | DGM | Network Solver |
|:--------------|:----|:----------------|:---------|:-----|:----|:---------------|
| Cartesian Grid | ✓ | ✓ | ✓ | ✓ | ✓ (≤3D) | ✗ |
| Network | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Maze | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Domain2D | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| Domain3D | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| Implicit | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |

**Key Insight**: Particle methods are the most versatile, working with all geometry types.

## Testing Strategy

```python
# tests/unit/test_unified_geometry.py

def test_cartesian_grid_geometry():
    """Test CartesianGrid geometry object."""
    grid = CartesianGrid(bounds=[(0, 1), (0, 1)], discretization=[50, 50])
    problem = MFGProblem(geometry=grid, T=1.0, Nt=100, sigma=0.1)

    assert problem.dimension == 2
    assert problem.geometry_type == GeometryType.CARTESIAN_GRID
    assert "fdm" in problem.solver_compatible
    assert "particle" in problem.solver_compatible

def test_backward_compatible_grid():
    """Test old spatial_bounds/discretization interface."""
    problem = MFGProblem(
        spatial_bounds=[(0, 1), (0, 1)],
        spatial_discretization=[50, 50],
        T=1.0, Nt=100, sigma=0.1
    )

    assert problem.dimension == 2
    assert problem.geometry_type == GeometryType.CARTESIAN_GRID
    assert hasattr(problem, 'geometry')
    assert isinstance(problem.geometry, CartesianGrid)

def test_maze_geometry():
    """Test maze with obstacles."""
    maze = MazeGeometry(
        bounds=[(0, 10), (0, 10)],
        walls=[(3, 3, 3, 7), (7, 3, 7, 7)],
        discretization=[100, 100]
    )
    problem = MFGProblem(geometry=maze, T=1.0, Nt=100, sigma=0.1)

    assert problem.geometry_type == GeometryType.MAZE
    assert "particle" in problem.solver_compatible
    assert "fdm" not in problem.solver_compatible

def test_network_geometry():
    """Test network/graph geometry."""
    network = create_network("scale_free", n_nodes=100, avg_degree=4)
    problem = MFGProblem(geometry=network, T=1.0, Nt=100, sigma=0.1)

    assert problem.geometry_type == GeometryType.NETWORK
    assert "network_solver" in problem.solver_compatible
    assert "fdm" not in problem.solver_compatible

def test_geometry_type_flag():
    """Test explicit geometry_type flag for disambiguation."""
    custom_obj = CustomGeometry()
    problem = MFGProblem(
        geometry=custom_obj,
        geometry_type=GeometryType.CUSTOM,
        T=1.0, Nt=100, sigma=0.1
    )

    assert problem.geometry_type == GeometryType.CUSTOM
```

## Migration Path

**Existing code continues to work**:
```python
# OLD API (still works)
problem = MFGProblem(
    spatial_bounds=[(0, 1)],
    spatial_discretization=[100],
    T=1.0, Nt=100, sigma=0.1
)

# Internally converts to:
# geometry = CartesianGrid(bounds=[(0, 1)], discretization=[100])
```

**New API for complex geometries**:
```python
# NEW API (complex geometries)
problem = MFGProblem(
    geometry=maze_geometry,
    T=1.0, Nt=100, sigma=0.1
)
```

## Benefits

1. **Unified interface**: Single `geometry` parameter for all types
2. **Particle method advantage**: Naturally enables particle methods for complex geometries
3. **Automatic compatibility**: Solver recommendations adapt to geometry
4. **Extensible**: Easy to add new geometry types
5. **Backward compatible**: No breaking changes to existing code
6. **Self-aware**: Automatic type detection where possible
7. **Explicit when needed**: Optional `geometry_type` flag for disambiguation

## Implementation Phases

**Phase 1** (This PR): Design document and base geometry protocol
**Phase 2**: CartesianGrid wrapper and backward compatibility
**Phase 3**: MFGProblem integration with geometry parameter
**Phase 4**: MazeGeometry implementation
**Phase 5**: NetworkGeometry protocol compliance
**Phase 6**: Domain2D/Domain3D integration (optional dependency)

## References

- NetworkMFGProblem: `mfg_pde/core/network_mfg_problem.py`
- NetworkGeometry: `mfg_pde/geometry/network_geometry.py`
- Domain2D/Domain3D: `mfg_pde/geometry/` (optional gmsh dependency)
- Maze utilities: `mfg_pde/geometry/maze.py`

---

**Status**: Ready for implementation pending PR #246 merge
