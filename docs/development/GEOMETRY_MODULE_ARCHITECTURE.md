# Geometry Module Architecture

**Date**: 2025-12-02
**Status**: Reference
**Location**: `mfg_pde/geometry/`

---

## 1. Overview

The geometry module provides unified spatial domain management for MFG solvers. It defines the physical space where agents evolve, boundary conditions are applied, and solutions are computed.

### Design Principles

1. **Protocol-based polymorphism**: All geometries implement `GeometryProtocol`
2. **Separation of concerns**: Geometry defines space, BC applicators enforce constraints
3. **Solver-agnostic**: Same geometry works with FDM, GFDM, particle, network solvers
4. **Dimension-independent**: Core abstractions work for any dimension d >= 1

---

## 2. Module Structure

```
geometry/
├── __init__.py              # Public API exports
├── protocol.py              # GeometryProtocol definition
├── base.py                  # Base geometry classes
├── masks.py                 # Mask generation utilities
│
├── grids/                   # Structured grids (FDM)
│   ├── grid_1d.py          # SimpleGrid1D
│   ├── grid_2d.py          # SimpleGrid2D
│   ├── grid_3d.py          # SimpleGrid3D
│   └── tensor_grid.py      # TensorProductGrid (any dimension)
│
├── implicit/                # SDF-based domains (meshfree)
│   ├── implicit_domain.py  # Base ImplicitDomain
│   ├── hyperrectangle.py   # Axis-aligned boxes
│   ├── hypersphere.py      # Balls/circles
│   ├── csg_operations.py   # Union, Intersection, Difference
│   └── point_cloud.py      # Scattered point geometry
│
├── meshes/                  # Unstructured meshes (FEM)
│   ├── mesh_data.py        # Universal mesh container
│   ├── mesh_1d.py          # Mesh1D
│   ├── mesh_2d.py          # Mesh2D (triangular)
│   ├── mesh_3d.py          # Mesh3D (tetrahedral)
│   └── mesh_manager.py     # Multi-geometry management
│
├── graph/                   # Network/graph geometries
│   ├── network_geometry.py # NetworkGeometry base
│   ├── network_backend.py  # NetworkX/igraph backends
│   ├── maze_generator.py   # Maze generation algorithms
│   └── maze_*.py           # Various maze algorithms
│
├── amr/                     # Adaptive mesh refinement
│   ├── amr_1d.py           # 1D adaptive intervals
│   ├── amr_triangular_2d.py# 2D triangular AMR
│   ├── amr_quadtree_2d.py  # 2D quadtree AMR
│   └── amr_tetrahedral_3d.py# 3D tetrahedral AMR
│
├── boundary/                # Boundary condition management
│   ├── types.py            # BCType, BCSegment, BoundaryConditions
│   ├── conditions.py       # MixedBoundaryConditions
│   ├── applicator_base.py  # Base applicator classes
│   ├── applicator_fdm.py   # FDM ghost cell BC
│   ├── applicator_fem.py   # FEM weak form BC
│   ├── applicator_meshfree.py # Particle/collocation BC
│   ├── applicator_graph.py # Network BC
│   └── fem_bc_*.py         # FEM-specific BC classes
│
└── operators/               # Geometry operations
    ├── projection.py       # GeometryProjector, ProjectionRegistry
    └── __init__.py
```

---

## 3. Core Protocol

All geometry types implement `GeometryProtocol` (defined in `protocol.py`):

```
┌─────────────────────────────────────────────────────────────────────┐
│                       GeometryProtocol                               │
├─────────────────────────────────────────────────────────────────────┤
│ Properties:                                                          │
│   dimension: int              # Spatial dimension (1, 2, 3, ...)     │
│   geometry_type: GeometryType # CARTESIAN_GRID, NETWORK, IMPLICIT... │
│   num_spatial_points: int     # Total discrete points                │
├─────────────────────────────────────────────────────────────────────┤
│ Core Methods:                                                        │
│   get_spatial_grid()          # Grid/mesh representation             │
│   get_bounds()                # Bounding box (min, max)              │
│   get_problem_config()        # Config dict for MFGProblem           │
├─────────────────────────────────────────────────────────────────────┤
│ Boundary Methods (mandatory):                                        │
│   is_on_boundary(points)      # Boolean mask of boundary points      │
│   get_boundary_normal(points) # Outward unit normals                 │
│   project_to_boundary(points) # Project points to boundary           │
│   project_to_interior(points) # Keep points inside domain            │
│   get_boundary_regions()      # Named regions for mixed BC           │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decision

**Boundary methods are mandatory in GeometryProtocol**, not optional. Rationale:
- Every domain has a boundary (even "unbounded" is a boundary type)
- BC applicators need boundary info from geometry
- Unified handling simplifies solver implementation

---

## 4. Geometry Types

### 4.1 Structured Grids (`grids/`)

**Purpose**: Regular Cartesian grids for finite difference methods

**Classes**:
- `SimpleGrid1D`: 1D interval [xmin, xmax] with Nx points
- `SimpleGrid2D`: 2D rectangle with (Nx, Ny) points
- `SimpleGrid3D`: 3D box with (Nx, Ny, Nz) points
- `TensorProductGrid`: Arbitrary dimension tensor product grid

**Characteristics**:
- Axis-aligned bounding box
- Uniform spacing per axis
- Grid points stored as coordinate arrays
- O(1) point-to-index lookup
- Boundary: edges/faces at coordinate extrema

**Usage**:
```python
from mfg_pde.geometry import SimpleGrid2D, TensorProductGrid

# 2D grid on [0,10] x [0,10]
grid = SimpleGrid2D(bounds=(0, 10, 0, 10), resolution=(50, 50))

# nD tensor product grid
grid_nd = TensorProductGrid(
    dimension=4,
    bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    num_points=[20, 20, 20, 20]
)
```

### 4.2 Implicit Domains (`implicit/`)

**Purpose**: SDF-based domains for meshfree methods, complex shapes

**Classes**:
- `ImplicitDomain`: Base class with signed distance function
- `Hyperrectangle`: Axis-aligned box (any dimension)
- `Hypersphere`: Ball/circle (any dimension)
- CSG operations: `UnionDomain`, `IntersectionDomain`, `DifferenceDomain`
- `PointCloudGeometry`: Scattered point cloud

**Characteristics**:
- Domain defined by `signed_distance(x)`: negative inside, positive outside
- No explicit mesh - samples generated on demand
- Supports any dimension (d >= 1)
- Complex shapes via CSG composition
- Boundary: zero level set of SDF

**SDF Properties**:
- `|nabla(phi)| = 1` almost everywhere (exact SDF)
- Outward normal: `n = nabla(phi) / |nabla(phi)|`
- Distance to boundary: `|phi(x)|`

**Usage**:
```python
from mfg_pde.geometry import Hyperrectangle, Hypersphere, DifferenceDomain

# Room with circular obstacle
room = Hyperrectangle([[0, 10], [0, 10]])
obstacle = Hypersphere(center=[5, 5], radius=2)
domain = DifferenceDomain(room, obstacle)

# Sample interior points
points = domain.sample_interior(1000)
```

### 4.3 Unstructured Meshes (`meshes/`)

**Purpose**: Triangular/tetrahedral meshes for FEM

**Classes**:
- `MeshData`: Universal mesh data container
- `Mesh1D`, `Mesh2D`, `Mesh3D`: Dimension-specific mesh classes
- `MeshManager`: Multi-mesh management
- `MeshPipeline`: Gmsh -> Meshio -> PyVista workflow

**Characteristics**:
- Elements: segments (1D), triangles (2D), tetrahedra (3D)
- Nodes at arbitrary positions
- Boundary marked by element tags
- Requires external mesh generator (Gmsh)

**Usage**:
```python
from mfg_pde.geometry import Mesh2D, MeshPipeline

# Generate mesh from geometry
mesh = MeshPipeline.from_rectangle(
    bounds=[0, 10, 0, 10],
    mesh_size=0.5
)
```

### 4.4 Network Geometries (`graph/`)

**Purpose**: Graph/network topologies for discrete agent spaces

**Classes**:
- `BaseNetworkGeometry`: Abstract base for networks
- `GridNetwork`: Regular lattice network
- `RandomNetwork`: Erdos-Renyi random graph
- `ScaleFreeNetwork`: Barabasi-Albert model
- Maze generators: `PerfectMazeGenerator`, `VoronoiMazeGenerator`, etc.

**Characteristics**:
- Nodes represent discrete locations
- Edges represent possible transitions
- Embedding in continuous space (optional)
- Boundary: terminal nodes or graph boundary

**Usage**:
```python
from mfg_pde.geometry import GridNetwork, create_network

# 10x10 grid network
network = GridNetwork(nx=10, ny=10, bounds=(0, 10, 0, 10))

# Scale-free network
network = create_network("scale_free", n_nodes=100, m_edges=3)
```

---

## 5. Boundary Condition System

### 5.1 Abstraction Layers

```
┌───────────────────────────────────────────────────────────────────┐
│                    User API (Declarative)                          │
│  BoundaryConditions, BCSegment, factory functions                  │
├───────────────────────────────────────────────────────────────────┤
│                    BC Specification Layer                          │
│  BCType enum, MixedBoundaryConditions, region matching             │
├───────────────────────────────────────────────────────────────────┤
│                    BC Applicator Layer                             │
│  FDMApplicator, FEMApplicator, MeshfreeApplicator, GraphApplicator │
├───────────────────────────────────────────────────────────────────┤
│                    Geometry Protocol                               │
│  is_on_boundary, get_boundary_normal, project_to_interior          │
└───────────────────────────────────────────────────────────────────┘
```

### 5.2 BC Types

```python
class BCType(Enum):
    DIRICHLET = "dirichlet"  # u = g on boundary
    NEUMANN = "neumann"      # du/dn = g on boundary
    ROBIN = "robin"          # alpha*u + beta*du/dn = g
    PERIODIC = "periodic"    # u(x_min) = u(x_max)
    ABSORBING = "absorbing"  # Particles removed at boundary
    REFLECTING = "reflecting"# Particles bounce at boundary
```

### 5.3 BCSegment: Region-Specific BC

```python
@dataclass
class BCSegment:
    name: str                    # Human-readable name
    bc_type: BCType              # Type of BC
    value: float | Callable      # BC value (scalar or function)

    # Region specification (multiple methods, checked in order):
    boundary: str | None         # Method 1: Named boundary ("left", "x_min")
    region: dict | None          # Method 2: Coordinate range {0: (0, 5)}
    sdf_region: Callable | None  # Method 3: SDF-based region
    mask: np.ndarray | None      # Method 0: Direct mask (O(1) lookup)
    normal_direction: np.ndarray | None  # Method 4: Normal-based matching

    # Additional parameters
    alpha: float = 1.0           # Robin BC coefficient
    beta: float = 0.0            # Robin BC coefficient
    priority: int = 0            # Matching priority (higher = first)
```

### 5.4 BC Applicator Selection

| Geometry Type | Applicator | Method |
|:--------------|:-----------|:-------|
| SimpleGrid* | `FDMApplicator` | Ghost cell reflection |
| TensorProductGrid | `FDMApplicator` | Ghost cell reflection |
| Mesh2D/3D | `FEMApplicator` | Weak form / penalty |
| ImplicitDomain | `MeshfreeApplicator` | Particle reflection / penalty |
| NetworkGeometry | `GraphApplicator` | Node value enforcement |

---

## 6. Geometry Projection

### 6.1 Purpose

Enable dual-geometry MFG solving: different discretizations for HJB and FP.

### 6.2 `GeometryProjector` Class

```python
class GeometryProjector:
    """Project fields between different geometry representations."""

    def __init__(self, hjb_geometry, fp_geometry, method="auto"):
        self.hjb_geometry = hjb_geometry
        self.fp_geometry = fp_geometry
        self._select_methods(method)

    def project_hjb_to_fp(self, field):
        """Project HJB solution to FP geometry."""
        return self._hjb_to_fp_func(field)

    def project_fp_to_hjb(self, field):
        """Project FP density to HJB geometry."""
        return self._fp_to_hjb_func(field)
```

### 6.3 Projection Methods

| Source | Target | Method |
|:-------|:-------|:-------|
| Grid | Grid (different resolution) | Interpolation |
| Grid | Particles | Interpolation at particle locations |
| Particles | Grid | Kernel Density Estimation (KDE) |
| Grid | Network | Interpolation at node locations |
| Network | Grid | KDE spreading |

### 6.4 Custom Projections

```python
from mfg_pde.geometry import ProjectionRegistry

@ProjectionRegistry.register(MyGeometry, SimpleGrid2D, "hjb_to_fp")
def custom_projection(source_geo, target_geo, values, **kwargs):
    # Custom projection logic
    return projected_values
```

---

## 7. Usage Patterns

### 7.1 Standard MFG Problem Setup

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import SimpleGrid2D
from mfg_pde.geometry.boundary import dirichlet_bc

# Geometry
grid = SimpleGrid2D(bounds=(0, 10, 0, 10), resolution=(50, 50))

# Uniform BC
bc = dirichlet_bc(dimension=2, value=0.0)

# Problem
problem = MFGProblem(geometry=grid, T=2.0, Nt=100, sigma=0.3)
```

### 7.2 Mixed BC on Rectangular Domain

```python
from mfg_pde.geometry.boundary import (
    MixedBoundaryConditions, BCSegment, BCType
)

bc = MixedBoundaryConditions(
    dimension=2,
    segments=[
        BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
                  boundary="right", region={1: (4.0, 6.0)}),
        BCSegment(name="walls", bc_type=BCType.NEUMANN, value=0.0),
    ],
    default_bc=BCType.NEUMANN,
)
```

### 7.3 Complex Domain with Obstacle

```python
from mfg_pde.geometry import Hyperrectangle, Hypersphere, DifferenceDomain
from mfg_pde.geometry.boundary import MeshfreeApplicator

# Room minus obstacle
room = Hyperrectangle([[0, 10], [0, 10]])
obstacle = Hypersphere(center=[5, 5], radius=2)
domain = DifferenceDomain(room, obstacle)

# BC applicator for particles
bc_applicator = MeshfreeApplicator(domain)

# Apply reflecting BC to particles
particles = bc_applicator.apply_particle_bc(particles, "reflecting")
```

### 7.4 Dual Geometry (Multi-Resolution)

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import SimpleGrid2D

# Fine grid for HJB (value function needs accuracy)
hjb_grid = SimpleGrid2D(bounds=(0, 10, 0, 10), resolution=(200, 200))

# Coarse grid for FP (density is smooth)
fp_grid = SimpleGrid2D(bounds=(0, 10, 0, 10), resolution=(50, 50))

problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_grid,
    T=2.0, Nt=100, sigma=0.3
)

# Automatic projection between geometries
projector = problem.geometry_projector
```

---

## 8. Extension Points

### 8.1 Adding New Geometry Type

1. Implement `GeometryProtocol` in new class
2. Add to appropriate subdirectory
3. Export from `geometry/__init__.py`
4. Register projections if needed

### 8.2 Adding New BC Applicator

1. Subclass appropriate base (`BaseFDMApplicator`, `BaseMeshfreeApplicator`)
2. Implement `apply_boundary_conditions_*` methods
3. Add to `boundary/__init__.py` exports

### 8.3 Adding New Projection Method

1. Define projection function
2. Register with `ProjectionRegistry`
3. Specify source/target geometry types

---

## 9. Design Decisions

### Why Protocol-Based?

- **Flexibility**: Any class implementing protocol works with solvers
- **Duck typing**: Python-native approach
- **Runtime checking**: `isinstance(obj, GeometryProtocol)` works
- **Documentation**: Protocol defines clear interface

### Why Boundary Methods in GeometryProtocol?

- Every domain has boundary (mathematical fact)
- BC applicators need geometry info
- Avoids separate "boundary provider" abstraction
- Single point of truth for domain shape

### Why Separate BC Specification from Application?

- **Separation of concerns**: What BC vs how to apply
- **Reusability**: Same BC spec with different applicators
- **Testing**: Can test specification logic independently
- **Flexibility**: Mix-and-match BC types with applicator types

---

## 10. Related Documentation

- `docs/development/LIPSCHITZ_BC_IMPLEMENTATION_PLAN.md` - Curved boundary support
- `docs/development/BC_GEOMETRY_ROADMAP_GAP_ANALYSIS.md` - Implementation status
- `docs/user/dual_geometry_usage.md` - User guide for geometry projection
- `docs/theory/geometry_projection_mathematical_formulation.md` - Math details
