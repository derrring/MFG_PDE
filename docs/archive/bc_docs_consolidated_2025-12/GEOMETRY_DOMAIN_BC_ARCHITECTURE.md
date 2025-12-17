# Geometry, Domain, and Boundary Conditions: Architecture Report

## Overview

This document describes the relationship between geometry, domain, and boundary conditions in MFG_PDE at both the implementation and application levels.

---

## 1. Conceptual Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MFGProblem                                  │
│  (Physics: Hamiltonian, costs, initial density)                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ uses
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Geometry                                   │
│  (Spatial structure: points, coordinates, topology)                 │
│  - TensorProductGrid (Cartesian)                                    │
│  - ImplicitDomain (SDF-based)                                       │
│  - NetworkGeometry (Graph)                                          │
│  - Mesh2D/3D (Unstructured FEM)                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ has
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Boundary                                    │
│  (Where domain ends: detection, normals, regions)                   │
│  - is_on_boundary()                                                 │
│  - get_boundary_normal()                                            │
│  - get_boundary_regions()                                           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ constrained by
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Boundary Conditions                              │
│  (Physics at boundary: what happens when agents reach edge)         │
│  - BCType: Dirichlet, Neumann, Robin, Periodic, NoFlux, Reflecting  │
│  - BCSegment: Condition on specific boundary region                 │
│  - BoundaryConditions: Collection of segments                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ applied by
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BC Applicator                                  │
│  (Discretization-specific enforcement)                              │
│  - FDM: Ghost cells, matrix modification                            │
│  - FEM: Weak form integration                                       │
│  - Particle: Reflection algorithms                                  │
│  - Graph: Node constraints                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Implementation Architecture

### 2.1 Geometry Types (`mfg_pde/geometry/`)

| Type | Class | File | Use Case |
|:-----|:------|:-----|:---------|
| Cartesian Grid | `TensorProductGrid` | `grids/tensor_grid.py` | FDM, WENO, structured solvers |
| Dimension-specific | `Grid1D`, `Grid2D`, `Grid3D` | `grids/grid_{1d,2d,3d}.py` | Legacy compatibility |
| Implicit Domain | `ImplicitDomain` | `implicit/implicit_domain.py` | High-D particle methods |
| Implicit Primitives | `Hyperrectangle`, `Hypersphere` | `implicit/hyperrectangle.py`, `hypersphere.py` | SDF building blocks |
| CSG Operations | `UnionDomain`, `IntersectionDomain`, `DifferenceDomain` | `implicit/csg_operations.py` | Composite domains |
| Point Cloud | `PointCloudDomain` | `implicit/point_cloud.py` | Scattered data domains |
| Network | `BaseNetworkGeometry`, `GridNetwork` | `graph/network_geometry.py` | Graph MFG, traffic networks |
| Maze Generation | `MazeGenerator`, `HybridMazeGenerator` | `graph/maze_*.py` | Maze-based MFG problems |
| Unstructured Mesh | `Mesh1D`, `Mesh2D`, `Mesh3D` | `meshes/mesh_{1d,2d,3d}.py` | FEM, complex geometries |
| AMR | `AdaptiveGeometry` (Protocol) | `amr/` | Future external library integration (pyAMReX, etc.) |

### 2.2 GeometryProtocol (`protocol.py`)

All geometries implement `GeometryProtocol`:

```python
class GeometryProtocol(Protocol):
    # Core geometry
    dimension: int
    geometry_type: GeometryType
    num_spatial_points: int

    def get_spatial_grid(self) -> np.ndarray | list[np.ndarray]: ...
    def get_bounds(self) -> tuple[NDArray, NDArray] | None: ...
    def get_problem_config(self) -> dict: ...

    # Boundary methods (mandatory)
    def is_on_boundary(self, points, tolerance) -> NDArray[bool]: ...
    def get_boundary_normal(self, points) -> NDArray: ...
    def project_to_boundary(self, points) -> NDArray: ...
    def get_boundary_regions(self) -> dict[str, Callable]: ...
```

### 2.3 Boundary Condition Types (`boundary/types.py`)

```python
class BCType(Enum):
    DIRICHLET = "dirichlet"   # u = g (fixed value)
    NEUMANN = "neumann"       # du/dn = g (fixed flux)
    ROBIN = "robin"           # alpha*u + beta*du/dn = g (mixed type)
    PERIODIC = "periodic"     # u(x_min) = u(x_max)
    NO_FLUX = "no_flux"       # J dot n = 0 (field methods)
    REFLECTING = "reflecting" # Elastic reflection (particle methods)
```

### 2.4 Mixed Boundary Conditions

"Mixed BC" has two meanings in the codebase:

1. **Robin BC** (`BCType.ROBIN`): A single condition mixing Dirichlet and Neumann terms
   ```python
   # alpha*u + beta*du/dn = g
   BCSegment("absorbing", BCType.ROBIN, value=0.0, alpha=1.0, beta=0.1)
   ```

2. **Spatially mixed BCs**: Different types on different boundary regions
   ```python
   bc = BoundaryConditions(dimension=2)
   bc.add_segment(BCSegment("walls", BCType.NEUMANN, 0.0, boundary="all"))
   bc.add_segment(BCSegment("exit", BCType.DIRICHLET, 0.0,
                            boundary="right", region={"y": (2, 3)}, priority=1))
   ```

`MixedBoundaryConditions` is an alias for `BoundaryConditions` (backward compatibility).

### 2.5 Specialized Boundary Conditions

| BC Type | Physical Meaning | Implementation | Use Case |
|:--------|:-----------------|:---------------|:---------|
| **Absorbing** | Agents exit domain | `BCType.DIRICHLET` (value=0) for fields; `"absorbing"` for particles | Exits, sinks |
| **Reflecting** | Elastic bounce | `BCType.NO_FLUX` for fields; `"reflecting"` for particles | Walls |
| **Transparent** | Zero-gradient outflow | `BCType.NEUMANN` (value=0) | Open boundaries |
| **Periodic** | Wrap-around | `BCType.PERIODIC` | Torus topology |

**Lipschitz Domains**: Non-smooth boundaries (corners, edges) are supported via SDF:

```python
# BoundaryConditions with SDF for non-rectangular domains
bc = BoundaryConditions(dimension=2, domain_sdf=my_sdf_function)
bc.add_segment(BCSegment("exit", BCType.DIRICHLET, 0.0,
                         normal_direction=np.array([1, 0]),  # Match by normal
                         normal_tolerance=0.7))
```

**Particle-specific BCs** (`applicator_meshfree.py`):

```python
# Three modes for particle methods
applicator.apply_boundary_conditions(particles, bc_type="reflecting")  # Bounce
applicator.apply_boundary_conditions(particles, bc_type="absorbing")   # Remove
applicator.apply_boundary_conditions(particles, bc_type="periodic")    # Wrap
```

### 2.6 Boundary Segments (`BCSegment`)

Specify conditions on specific boundary regions:

```python
@dataclass
class BCSegment:
    name: str              # Human-readable identifier
    bc_type: BCType        # Type of condition
    value: float | Callable  # BC value (constant or function)

    # Matching methods (multiple supported)
    boundary: str | None   # "left", "right", "top", etc.
    region: dict | None    # {"y": (4.25, 5.75)} for partial walls
    sdf_region: Callable   # SDF for complex regions
    normal_direction: np.ndarray  # Match by outward normal

    priority: int = 0      # For overlapping segments
```

---

## 3. BC Applicator Architecture

```
                    BaseBCApplicator (ABC)
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    BaseStructured   BaseUnstructured  BaseMeshfree  BaseGraph
           │               │               │           │
           ▼               ▼               ▼           ▼
    FDMApplicator    FEMApplicator   ParticleApplicator  GraphApplicator
```

### 3.1 Applicator Files

| File | Purpose |
|:-----|:--------|
| `applicator_base.py` | Abstract base classes and protocols |
| `applicator_fdm.py` | Finite difference (ghost cells, matrix mods) |
| `applicator_fem.py` | Finite element (weak form) |
| `applicator_meshfree.py` | Particle/collocation methods |
| `applicator_graph.py` | Network/graph constraints |

### 3.2 Discretization-Specific BCs

| Method | Dirichlet | Neumann | NoFlux | Periodic |
|:-------|:----------|:--------|:-------|:---------|
| **FDM** | Set boundary values | Ghost cells | `np.pad(mode="edge")` | Wrap indices |
| **FEM** | Penalty/lifting | Natural BC | Natural BC | DOF coupling |
| **Particle** | Absorbing/reset | N/A | Reflecting | Wrap positions |
| **Graph** | Fix node values | Set edge weights | Zero edge flux | Connect nodes |

---

## 4. Application Paths

### 4.1 Path 1: Simple Rectangular Domain (Most Common)

```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BoundaryConditions, BCType, BCSegment

# 1. Create geometry
grid = TensorProductGrid(
    dimension=2,
    bounds=[(0, 10), (0, 5)],
    num_points=[101, 51]
)

# 2. Define BCs (uniform Neumann walls + Dirichlet exit)
bc = BoundaryConditions(dimension=2)
bc.add_segment(BCSegment("walls", BCType.NEUMANN, value=0.0, boundary="all"))
bc.add_segment(BCSegment("exit", BCType.DIRICHLET, value=0.0,
                         boundary="right", region={"y": (2.0, 3.0)}, priority=1))

# 3. Pass to problem
problem = MFGProblem(geometry=grid, boundary_conditions=bc, ...)
```

### 4.2 Path 2: Complex Domain (SDF-based)

```python
from mfg_pde.geometry.implicit import Hyperrectangle, Hypersphere
from mfg_pde.geometry.implicit.csg_operations import DifferenceDomain

# 1. Create domain with obstacle
room = Hyperrectangle(bounds=[(0, 10), (0, 5)])
obstacle = Hypersphere(center=[5, 2.5], radius=1.0)
domain = DifferenceDomain(room, obstacle)

# 2. BCs using normal matching (for curved boundaries)
bc = BoundaryConditions(dimension=2)
bc.add_segment(BCSegment("walls", BCType.NO_FLUX, value=0.0))
bc.add_segment(BCSegment("exit", BCType.DIRICHLET, value=0.0,
                         normal_direction=np.array([1.0, 0.0]),
                         normal_tolerance=0.9))
```

### 4.3 Path 3: Network Domain

```python
from mfg_pde.geometry import GridNetwork

# 1. Create network geometry
network = GridNetwork(rows=10, cols=10, spacing=1.0)

# 2. BCs on specific nodes
bc = BoundaryConditions(dimension="network")
bc.add_segment(BCSegment("source", BCType.DIRICHLET, value=1.0, nodes=[0, 1]))
bc.add_segment(BCSegment("sink", BCType.DIRICHLET, value=0.0, nodes=[99]))
```

---

## 5. Key Design Principles

### 5.1 Separation of Concerns

| Layer | Responsibility | Does NOT handle |
|:------|:---------------|:----------------|
| **Geometry** | Where points are, boundary detection | Physics, BC values |
| **BoundaryConditions** | What conditions apply where | How to enforce them |
| **BCApplicator** | How to enforce BCs numerically | Physics meaning |
| **Solver** | Time-stepping, coupling | Geometry details |

### 5.2 Polymorphic Configuration

Geometries provide `get_problem_config()` to configure MFGProblem without type-checking:

```python
# Geometry knows how to configure problem
config = geometry.get_problem_config()
# Returns: {
#     "num_spatial_points": 5151,
#     "spatial_shape": (101, 51),
#     "spatial_bounds": ((0, 10), (0, 5)),
#     "spatial_discretization": (101, 51),
# }
```

### 5.3 Boundary Methods Are Mandatory

Every domain has a boundary (even "unbounded" uses computational truncation). The `GeometryProtocol` requires:

- `is_on_boundary()` - Detection
- `get_boundary_normal()` - For flux/reflection
- `project_to_boundary()` - For constraint enforcement
- `get_boundary_regions()` - For mixed BCs

---

## 6. File Reference

```
mfg_pde/geometry/
├── __init__.py              # Public API exports
├── protocol.py              # GeometryProtocol, GeometryType
├── base.py                  # Geometry ABC
├── masks.py                 # Domain masking utilities
│
├── grids/
│   ├── __init__.py          # Grid exports
│   ├── tensor_grid.py       # TensorProductGrid (unified nD, recommended)
│   ├── grid_1d.py           # Grid1D (legacy)
│   ├── grid_2d.py           # Grid2D (legacy)
│   └── grid_3d.py           # Grid3D (legacy)
│
├── boundary/
│   ├── __init__.py          # Boundary exports
│   ├── types.py             # BCType, BCSegment
│   ├── conditions.py        # BoundaryConditions class (modern, unified)
│   ├── applicator_base.py   # BaseBCApplicator
│   ├── applicator_fdm.py    # FDM applicator
│   ├── applicator_fem.py    # FEM applicator
│   ├── applicator_meshfree.py # Particle applicator
│   ├── applicator_graph.py  # Graph applicator
│   ├── fdm_bc_1d.py         # 1D FDM BC utilities (deprecated, use conditions.py)
│   ├── fem_bc_1d.py         # 1D FEM BC classes
│   ├── fem_bc_2d.py         # 2D FEM BC classes
│   └── fem_bc_3d.py         # 3D FEM BC classes
│
├── implicit/
│   ├── __init__.py          # Implicit domain exports
│   ├── implicit_domain.py   # ImplicitDomain base class
│   ├── hyperrectangle.py    # Hyperrectangle SDF primitive
│   ├── hypersphere.py       # Hypersphere SDF primitive
│   ├── csg_operations.py    # CSG: Union, Intersection, Difference
│   ├── point_cloud.py       # PointCloudDomain
│   └── README.md            # Implicit geometry documentation
│
├── graph/
│   ├── __init__.py          # Graph/network exports
│   ├── network_geometry.py  # BaseNetworkGeometry, GridNetwork
│   ├── network_backend.py   # Graph backend utilities
│   ├── maze_generator.py    # Core maze generation
│   ├── maze_config.py       # Maze configuration
│   ├── maze_hybrid.py       # Hybrid maze generation
│   ├── maze_recursive_division.py  # Recursive division algorithm
│   ├── maze_voronoi.py      # Voronoi-based mazes
│   ├── maze_cellular_automata.py   # Cellular automata mazes
│   ├── maze_postprocessing.py      # Maze post-processing
│   └── maze_utils.py        # Maze utilities
│
├── meshes/
│   ├── __init__.py          # Mesh exports
│   ├── mesh_data.py         # Universal mesh container (MeshData)
│   ├── mesh_manager.py      # Mesh lifecycle management
│   ├── mesh_1d.py           # 1D unstructured mesh
│   ├── mesh_2d.py           # 2D unstructured mesh
│   └── mesh_3d.py           # 3D unstructured mesh
│
├── amr/
│   └── __init__.py          # AMR stub for future external library integration (pyAMReX, etc.)
│
└── operators/
    ├── __init__.py          # Operator exports
    └── projection.py        # Projection operators
```

---

## 7. Summary

| Question | Answer |
|:---------|:-------|
| What is a Geometry? | Spatial structure (points, topology, bounds) |
| What is a Domain? | Geometry + physical interpretation (interior, boundary) |
| What are BCs? | Physical constraints at boundary |
| How are BCs specified? | `BCSegment` objects in `BoundaryConditions` |
| How are BCs enforced? | `BCApplicator` (discretization-specific) |
| Who owns what? | Geometry owns structure, BCs own physics at edges, Applicators own numerics |

---

*Generated: 2025-12-09*
*Last audited: 2025-12-10*
*Part of: MFG_PDE Deprecation Cleanup Phase 2*
