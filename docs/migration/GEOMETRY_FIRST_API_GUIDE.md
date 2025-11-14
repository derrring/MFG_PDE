# Geometry-First API Guide

**Status**: ✅ Complete as of v0.10.0
**Date**: 2025-11-05

## Overview

The geometry-first API is the new recommended way to construct MFG problems in MFG_PDE. Instead of manually specifying grid parameters in `MFGProblem`, you first create a geometry object and pass it to `MFGProblem`.

## Key Benefits

1. **Type Safety**: Geometry objects are validated at construction time
2. **Reusability**: Same geometry can be used for multiple problems
3. **Clarity**: Separation of spatial discretization from temporal/diffusion parameters
4. **Flexibility**: Supports all geometry types through unified protocol

## Available Geometry Types

| Geometry Class | Type | Description | Dimensions |
|:---------------|:-----|:------------|:-----------|
| `TensorProductGrid` | CARTESIAN_GRID | Structured regular grid | 1D-nD |
| `Domain1D` | DOMAIN_1D | 1D domain with BC | 1D |
| `Domain2D`, `Domain3D` | DOMAIN_2D/3D | Unstructured mesh via Gmsh | 2D, 3D |
| `Hyperrectangle` | IMPLICIT | Box domain via SDF | nD |
| `Hypersphere` | IMPLICIT | Sphere/ball via SDF | nD |
| `Grid` (mazes) | MAZE | Maze-based grid | 2D |
| `NetworkGeometry` | NETWORK | Graph-based network | 2D |

## Quick Start Examples

### 1. TensorProductGrid (Structured Grid)

**New API** (recommended):
```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.core import MFGProblem

# Create 2D grid: [0,10] × [0,5]
geometry = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, 10.0), (0.0, 5.0)],
    num_points=[101, 51]
)

# Create problem
problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

**Old API** (deprecated, still works):
```python
problem = MFGProblem(
    spatial_bounds=[(0.0, 10.0), (0.0, 5.0)],
    spatial_discretization=[101, 51],
    T=1.0,
    Nt=100,
    sigma=0.1
)
```

### 2. Domain1D (1D with Boundary Conditions)

**New API**:
```python
from mfg_pde.geometry import Domain1D
from mfg_pde.core import MFGProblem

# Create 1D periodic domain
domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions="periodic")
domain.create_grid(Nx=101)

problem = MFGProblem(geometry=domain, T=1.0, Nt=100, sigma=0.1)
```

**Old API** (deprecated):
```python
problem = MFGProblem(
    xmin=[0.0], xmax=[1.0], Nx=[101],
    T=1.0, Nt=100, sigma=0.1
)
```

### 3. Implicit Domains (Meshfree)

**Hyperrectangle** (box domain via signed distance function):
```python
from mfg_pde.geometry.implicit import Hyperrectangle
from mfg_pde.core import MFGProblem
import numpy as np

# Create [0,1]² box
geometry = Hyperrectangle(bounds=np.array([[0, 1], [0, 1]]))

problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

**Hypersphere** (sphere via signed distance function):
```python
from mfg_pde.geometry.implicit import Hypersphere
from mfg_pde.core import MFGProblem

# Create unit sphere centered at origin
geometry = Hypersphere(center=[0, 0], radius=1.0)

problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

### 4. Maze Geometry

```python
from mfg_pde.geometry.graph import PerfectMazeGenerator
from mfg_pde.core import MFGProblem

# Generate 10×10 maze
maze_gen = PerfectMazeGenerator(rows=10, cols=10)
geometry = maze_gen.generate()

problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

### 5. High-Dimensional Grids

```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.core import MFGProblem

# Create 4D grid (will emit performance warning)
geometry = TensorProductGrid(
    dimension=4,
    bounds=[(0.0, 1.0)] * 4,
    num_points=[10, 10, 10, 10]  # 10^4 = 10,000 points
)

problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

## GeometryProtocol

All geometry objects implement the `GeometryProtocol`:

```python
from typing import Protocol
from numpy.typing import NDArray
from mfg_pde.geometry.protocol import GeometryType

class GeometryProtocol(Protocol):
    """Protocol for geometry objects usable in MFGProblem."""

    @property
    def dimension(self) -> int:
        """Spatial dimension (1, 2, 3, ...)."""
        ...

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (CARTESIAN_GRID, IMPLICIT, etc.)."""
        ...

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points."""
        ...

    def get_spatial_grid(self) -> NDArray:
        """Get spatial grid representation as (N, dimension) array."""
        ...
```

## Migration Strategy

### Phase 1: Update Examples (v0.10.0)
- Convert `examples/basic/` to use geometry-first API
- Keep old API examples in `examples/legacy/` for reference

### Phase 2: Deprecation Period (v0.10.x - v0.99.x)
- Old API emits `DeprecationWarning`
- Both APIs fully functional
- Update documentation to show new API first

### Phase 3: Restriction (v1.0.0+)
- Old manual grid construction restricted (requires explicit flag)
- Geometry-first API becomes mandatory for new code

## Testing Your Migration

Run this script to test geometry-first API with your geometry type:

```python
#!/usr/bin/env python
"""Test geometry-first API."""
import warnings
from mfg_pde.core import MFGProblem
from mfg_pde.geometry import TensorProductGrid  # or your geometry

# Suppress deprecation warnings (we're using new API)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create geometry
geometry = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    num_points=[51, 51]
)

# Create problem
problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)

# Verify
assert problem.dimension == 2
assert problem.num_spatial_points == 2601
assert problem.domain_type == "grid"

print("✓ Geometry-first API working correctly!")
```

## Advanced: Custom Geometry Types

You can create custom geometry types by implementing `GeometryProtocol`:

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from mfg_pde.geometry.protocol import GeometryType

@dataclass
class MyCustomGeometry:
    """Custom geometry implementing GeometryProtocol."""

    points: NDArray  # Your spatial discretization
    dim: int

    @property
    def dimension(self) -> int:
        return self.dim

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.CUSTOM

    @property
    def num_spatial_points(self) -> int:
        return len(self.points)

    def get_spatial_grid(self) -> NDArray:
        return self.points

# Use it
geometry = MyCustomGeometry(
    points=np.random.rand(100, 2),
    dim=2
)

problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)
```

## Common Patterns

### Reusing Geometry

```python
# Create geometry once
geometry = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], num_points=[51, 51])

# Use for multiple problems with different parameters
problem1 = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)
problem2 = MFGProblem(geometry=geometry, T=2.0, Nt=20, sigma=0.2)
```

### Geometry Refinement

```python
# Create coarse grid
coarse = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], num_points=[11, 11])

# Refine by factor of 2
fine = coarse.refine(factor=2)  # Now 21×21

# Use refined grid
problem = MFGProblem(geometry=fine, T=1.0, Nt=10, sigma=0.1)
```

### Composing Geometries

```python
# Create multiple domains for multi-population problems
domain1 = Hyperrectangle(bounds=np.array([[0, 1], [0, 1]]))
domain2 = Hypersphere(center=[2, 0], radius=0.5)

# Use in multi-population problem (future feature)
# problem = MultiPopulationMFG(geometries=[domain1, domain2], ...)
```

## Performance Considerations

### High-Dimensional Grids
- **Grid-based** (`TensorProductGrid`): O(N^d) memory/computation
  - Practical limit: d ≤ 3 for dense grids
  - For d > 3: Use coarse grids or sparse methods

- **Implicit domains** (`Hyperrectangle`, `Hypersphere`): O(N) memory
  - No mesh generation required
  - Suitable for high dimensions with meshfree solvers

### Mesh Generation
- **Structured grids** (`TensorProductGrid`): Instant creation
- **Unstructured meshes** (`Domain2D/3D`): Requires Gmsh (can be slow)
- **Implicit domains**: No mesh generation (SDF-based)

## FAQ

**Q: Do I need to update my code immediately?**
A: No. Old API continues to work with deprecation warnings through v0.99.x.

**Q: Will my old code break?**
A: No. 100% backward compatibility maintained until v1.0.0.

**Q: What if I need manual grid construction?**
A: Use `TensorProductGrid` - it provides the same flexibility with better type safety.

**Q: Can I mix old and new APIs?**
A: Yes, but not recommended. Use one pattern consistently per project.

**Q: When should I use implicit domains vs structured grids?**
A: Use implicit domains for:
  - High dimensions (d > 3)
  - Complex shapes (circles, arbitrary SDFs)
  - Meshfree methods (particle-based solvers)

Use structured grids for:
  - Low dimensions (d ≤ 3)
  - Rectangular domains
  - Finite difference methods

## See Also

- `docs/api/geometry_protocol.md` - Full protocol specification
- `docs/theory/implicit_domains.md` - Mathematical foundation for SDFs
- `examples/basic/geometry_demo.py` - Complete working examples
- `CLAUDE.md` - Repository conventions and standards
