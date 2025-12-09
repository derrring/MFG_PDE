# Geometry Module

Comprehensive geometry and mesh generation support for Mean Field Games.

## Overview

The geometry module provides domain definitions, boundary conditions, and mesh generation capabilities for 2D and 3D MFG problems through the **Gmsh → Meshio → PyVista** pipeline.

## Module Organization

The geometry module is organized into specialized subdirectories:

### Subdirectory Structure

- **`meshes/`** - Mesh generation and data structures
  - `mesh_data.py` - Universal mesh data containers (`MeshData`, `MeshVisualizationMode`)
  - `mesh_1d.py`, `mesh_2d.py`, `mesh_3d.py` - Dimension-specific mesh implementations
  - `mesh_manager.py` - Mesh lifecycle management
  - `mesh_pipeline.py` - Gmsh → Meshio → PyVista integration

- **`grids/`** - Cartesian grid geometries (no external dependencies)
  - `tensor_grid.py` - Unified tensor product grids (`TensorProductGrid`) for all dimensions

- **`graph/`** - Network and maze geometries
  - **Network geometries**: `network_geometry.py` - Graph-based domains (`BaseNetworkGeometry`, `GridNetwork`)
  - **Maze generators**:
    - `maze_cellular_automata.py` - Cellular automata maze generation
    - `maze_recursive_division.py` - Recursive division algorithm
    - `maze_hybrid.py` - Hybrid maze strategies
    - `maze_voronoi.py` - Voronoi-based maze generation

- **`boundary/`** - Boundary condition management
  - `bc_1d.py`, `bc_2d.py`, `bc_3d.py` - Dimension-specific boundary conditions
  - `bc_manager.py` - Unified boundary condition management
  - `mfg_bc_handler_2d.py`, `mfg_bc_handler_3d.py` - MFG-specific handlers

- **`implicit/`** - Implicit geometry definitions
  - `implicit_geometry.py` - Level-set based geometry representation
  - Support for complex geometries defined by implicit functions

- **`amr/`** - Adaptive mesh refinement
  - `amr_1d.py`, `amr_triangular_2d.py`, `amr_tetrahedral_3d.py` - AMR implementations
  - `amr_manager.py` - AMR lifecycle management

### File Naming Conventions

- **Grid files**: `tensor_grid.py` provides unified `TensorProductGrid` for all dimensions
- **Maze files**: Prefixed with `maze_` (`maze_cellular_automata.py`, `maze_hybrid.py`)
- **Network files**: Prefixed with `network_` (`network_geometry.py`)
- **Boundary files**: Prefixed with `bc_` (`bc_2d.py`, `bc_manager.py`)

### Import Patterns

```python
# Public API imports (recommended)
from mfg_pde.geometry import (
    TensorProductGrid,  # Unified Cartesian grids for all dimensions
    BaseNetworkGeometry,
    GridNetwork,
    MeshData,
    MeshVisualizationMode,
)

# Direct submodule imports (when needed)
from mfg_pde.geometry.grids import TensorProductGrid
from mfg_pde.geometry.graph import BaseNetworkGeometry
from mfg_pde.geometry.meshes import MeshData
from mfg_pde.geometry.boundary import BoundaryConditionManager2D
```

## Domain Types

### 2D Domains (`Domain2D`)

- **Rectangle**: Axis-aligned rectangular domains
- **Circle**: Circular domains
- **Polygon**: Arbitrary polygonal domains (including non-convex)
- **Custom**: User-defined geometries via Gmsh geometry objects

### 3D Domains (`Domain3D`)

- **Box**: Axis-aligned box domains
- **Sphere**: Spherical domains
- **Cylinder**: Cylindrical domains
- **Polyhedron**: Arbitrary polyhedral domains (including non-convex)
- **Custom**: User-defined 3D geometries via Gmsh geometry objects

## Lipschitz Bounded Domains

The `polygon` (2D) and `polyhedron` (3D) domain types support **arbitrary Lipschitz continuous boundaries**, including non-convex domains.

### Example: L-Shaped Domain (Non-Convex)

```python
import numpy as np
from mfg_pde.geometry import Domain2D

# Define L-shaped domain vertices
vertices = np.array([
    [0.0, 0.0], [1.0, 0.0], [1.0, 0.5],
    [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]
])

domain = Domain2D(
    domain_type="polygon",
    vertices=vertices,
    mesh_size=0.05
)

# Generate mesh
mesh = domain.generate_mesh()
```

### Example: Star-Shaped Domain

```python
# Define star-shaped domain
theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
r = 1.0 + 0.3 * np.sin(5 * theta)  # 5-pointed star
vertices = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

domain = Domain2D(
    domain_type="polygon",
    vertices=vertices,
    mesh_size=0.1
)
```

## Unbounded Domains

For problems on unbounded domains (e.g., $\mathbb{R}^d$, half-spaces), users must apply **computational truncation** or **domain transformation** techniques.

### Strategy 1: Large Bounded Domain with Truncation

```python
from mfg_pde.geometry import Domain2D
from mfg_pde.geometry.boundary_conditions_2d import NeumannBC2D, BoundaryConditionManager2D

# Truncate R^2 to large box [-L, L]^2
L = 100.0
domain = Domain2D(
    domain_type="rectangle",
    bounds=(-L, L, -L, L),
    mesh_size=0.5
)

# Add absorbing or transparent boundary conditions
manager = BoundaryConditionManager2D()
absorbing_bc = NeumannBC2D(0.0, name="Absorbing")  # or custom transparent BC
for edge in range(4):
    manager.add_condition(absorbing_bc, edge)
```

**Notes:**
- Choose $L$ large enough that boundary effects are negligible
- Verify solution decay near boundaries
- Document truncation radius in analysis

### Strategy 2: Domain Transformation

Map unbounded domain to bounded domain via analytic transformation:

```python
import numpy as np
from mfg_pde.geometry import Domain2D

# Example: Map R^2 to disk via transformation
# x = r * cos(θ), y = r * sin(θ)
# where r = tan(ρ * π/2) for ρ ∈ [0, 1)

# Define unit disk domain
domain = Domain2D(
    domain_type="circle",
    center=(0.0, 0.0),
    radius=1.0,
    mesh_size=0.05
)

# User applies transformation in problem setup
def transform_from_unbounded(r_unbounded):
    """Map R^+ to [0, 1) via r = tan(ρ * π/2)."""
    return (2.0 / np.pi) * np.arctan(r_unbounded)

def transform_to_unbounded(rho_bounded):
    """Map [0, 1) to R^+ via r = tan(ρ * π/2)."""
    return np.tan(rho_bounded * np.pi / 2.0)
```

**Common Transformations:**
- **Half-space** $\mathbb{R}^+$: $x = \log(\xi / (1-\xi))$ for $\xi \in (0,1)$
- **Whole space** $\mathbb{R}$: $x = \tan(\pi(\xi - 0.5))$ for $\xi \in (0,1)$
- **Stereographic projection**: Maps $\mathbb{R}^d$ to $\mathbb{S}^d \setminus \{point\}$

### Strategy 3: Localized Solutions

For problems with localized solutions, use **adaptive mesh refinement** near support:

```python
# Create large domain with fine mesh near origin
domain = Domain2D(
    domain_type="rectangle",
    bounds=(-50.0, 50.0, -50.0, 50.0),
    mesh_size=1.0  # Coarse far from origin
)

# User implements mesh refinement callback (future feature)
# or uses Gmsh fields for size control
```

## Bounding Box Property

The `bounds` property returns an **axis-aligned bounding box** (AABB), not the actual domain geometry:

```python
domain = Domain2D(
    domain_type="circle",
    center=(0.0, 0.0),
    radius=1.0
)

min_coords, max_coords = domain.bounds
# Returns: ([-1.0, -1.0], [1.0, 1.0])
# This is the bounding box, NOT the circle itself
```

**Usage:**
- Spatial indexing (quadtree/octree structures)
- Algorithm initialization (grid spacing, etc.)
- Visualization bounds

The actual domain geometry is preserved in the mesh representation generated by Gmsh.

## Boundary Conditions

The geometry module provides comprehensive boundary condition support:

### 2D Boundary Conditions

- **DirichletBC2D**: Fixed values $u = g(x,y,t)$
- **NeumannBC2D**: Flux conditions $\nabla u \cdot n = g(x,y,t)$
- **RobinBC2D**: Mixed conditions $\alpha u + \beta \nabla u \cdot n = g(x,y,t)$
- **PeriodicBC2D**: Periodic conditions $u(x_1,y) = u(x_2,y)$

### 3D Boundary Conditions

Analogous classes for 3D: `DirichletBC3D`, `NeumannBC3D`, `RobinBC3D`, `PeriodicBC3D`

### MFG-Specific Handlers

- **MFGBoundaryHandler2D**: HJB and Fokker-Planck equation boundary conditions
- **MFGBoundaryHandler3D**: 3D MFG boundary condition management

## Mesh Generation Pipeline

The module uses the standard computational geometry pipeline:

1. **Gmsh**: Geometry definition and mesh generation
2. **Meshio**: Format conversion and I/O
3. **PyVista**: Visualization and post-processing

### Example: Complete Workflow

```python
from mfg_pde.geometry import Domain2D
from mfg_pde.geometry.boundary_conditions_2d import create_rectangle_boundary_conditions

# 1. Define geometry
domain = Domain2D(
    domain_type="rectangle",
    bounds=(0.0, 1.0, 0.0, 1.0),
    mesh_size=0.05
)

# 2. Generate mesh
mesh = domain.generate_mesh()

# 3. Setup boundary conditions
bc_manager = create_rectangle_boundary_conditions(
    domain_bounds=(0.0, 1.0, 0.0, 1.0),
    condition_type="dirichlet_zero"
)

# 4. Apply to system matrix and RHS (in solver)
# matrix_mod, rhs_mod = bc_manager.apply_all_conditions(matrix, rhs, mesh)
```

## Advanced Features

### Domains with Holes

```python
# 2D domain with circular hole
outer_vertices = np.array([[0,0], [2,0], [2,2], [0,2]])
hole_center = (1.0, 1.0)
hole_radius = 0.3

domain = Domain2D(
    domain_type="polygon",
    vertices=outer_vertices,
    holes=[(hole_center, hole_radius)]
)
```

### Multi-Region Domains

```python
# Define multiple regions with different material properties
# (Implementation uses Gmsh physical groups)
```

### Mesh Quality Metrics

```python
# Compute mesh quality
mesh = domain.generate_mesh()

quality = mesh.quality_metrics
print(f"Min element quality: {quality['min_quality']}")
print(f"Average element quality: {quality['avg_quality']}")
```

## References

- **Gmsh Documentation**: https://gmsh.info/doc/texinfo/gmsh.html
- **Meshio Documentation**: https://github.com/nschloe/meshio
- **PyVista Documentation**: https://docs.pyvista.org/

## See Also

### Module References

- **Meshes**: `mfg_pde.geometry.meshes` - Mesh data structures and generation
  - `meshes.mesh_data`: Universal mesh containers
  - `meshes.mesh_manager`: Mesh lifecycle management
  - `meshes.mesh_pipeline`: Gmsh integration pipeline

- **Grids**: `mfg_pde.geometry.grids` - Cartesian grid geometries
  - `grids.tensor_grid`: Unified TensorProductGrid for all dimensions

- **Graphs**: `mfg_pde.geometry.graph` - Network and maze geometries
  - `graph.network_geometry`: Network-based domains
  - `graph.maze_*`: Maze generation algorithms

- **Boundaries**: `mfg_pde.geometry.boundary` - Boundary condition management
  - `boundary.bc_2d`, `boundary.bc_3d`: Dimension-specific boundary conditions
  - `boundary.bc_manager`: Unified boundary management
  - `boundary.mfg_bc_handler_*`: MFG-specific handlers

- **Implicit**: `mfg_pde.geometry.implicit` - Level-set based geometries
  - `implicit.implicit_geometry`: Implicit geometry definitions

- **AMR**: `mfg_pde.geometry.amr` - Adaptive mesh refinement
  - `amr.amr_*`: Dimension-specific AMR implementations

### Examples

- `examples/basic/geometry/`: Simple geometry demonstrations
- `examples/advanced/geometry/`: Complex multi-domain examples
- `examples/tutorials/`: Step-by-step geometry tutorials
