# Boundary Conditions Guide

This guide explains how to specify and apply boundary conditions in MFG_PDE.

## Quick Start

```python
from mfg_pde.geometry.boundary import (
    dirichlet_bc,
    neumann_bc,
    periodic_bc,
    no_flux_bc,
    mixed_bc,
    BCSegment,
    BCType,
)

# Uniform BC (same on all boundaries)
bc = neumann_bc(dimension=2)  # Zero-flux everywhere

# Dirichlet with specific value
bc = dirichlet_bc(dimension=2, value=0.0)

# Periodic BC
bc = periodic_bc(dimension=2)

# Time-dependent BC
bc = dirichlet_bc(dimension=2, value=lambda t: np.sin(t))
```

## BC Types

| Type | Mathematical Form | Typical Use |
|------|-------------------|-------------|
| `DIRICHLET` | u = g | Fixed value at boundary |
| `NEUMANN` | du/dn = g | Fixed flux at boundary |
| `NO_FLUX` | du/dn = 0 | Reflecting/insulating boundary |
| `ROBIN` | alpha*u + beta*du/dn = g | Mixed condition |
| `PERIODIC` | u(x_min) = u(x_max) | Wrap-around domain |

## Mixed Boundary Conditions

For different BC types on different boundaries:

```python
from mfg_pde.geometry.boundary import mixed_bc, BCSegment, BCType
import numpy as np

# Define domain
bounds = np.array([[0, 1], [0, 1]])  # Unit square

# Exit at top (Dirichlet), walls elsewhere (Neumann)
exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=0.0,
    boundary="y_max",  # Top boundary
)
wall_bc = BCSegment(
    name="wall",
    bc_type=BCType.NEUMANN,
    value=0.0,
    # No boundary specified = default for remaining boundaries
)

bc = mixed_bc([exit_bc, wall_bc], dimension=2, domain_bounds=bounds)
```

## Region-Based Boundary Conditions

For complex geometries, you can define BCs using **regions** marked on the geometry rather than boundary identifiers:

```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import mixed_bc_from_regions, BCSegment, BCType

# Create geometry
geometry = TensorProductGrid(dimension=2, bounds=[(0, 2), (0, 1)], Nx_points=[41, 21])

# Mark regions using predicates
geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)  # Left 10%
geometry.mark_region("outlet", predicate=lambda x: x[:, 0] > 1.9)  # Right 10%
geometry.mark_region("walls", boundary="y_min")  # Bottom wall
geometry.mark_region("walls", boundary="y_max")  # Top wall (merged with bottom)

# Define BCs referencing regions
bc_config = {
    "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
    "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
    "walls": BCSegment(name="wall_bc", bc_type=BCType.NO_FLUX),
    "default": BCSegment(name="periodic_bc", bc_type=BCType.PERIODIC),
}

bc = mixed_bc_from_regions(geometry, bc_config)

# Apply BCs (must pass geometry parameter)
from mfg_pde.geometry.boundary import FDMApplicator
applicator = FDMApplicator(dimension=2)
padded = applicator.apply(field, bc, domain_bounds=geometry.bounds, geometry=geometry)
```

### Region Marking Methods

Regions can be marked using boundaries or custom predicates:

```python
# Method 1: Mark using boundary identifier
geometry.mark_region("left_wall", boundary="x_min")

# Method 2: Mark using predicate function
geometry.mark_region("inlet", predicate=lambda x: (x[:, 0] < 0.1) & (x[:, 1] > 0.5))

# Method 3: Combine multiple boundaries in one region
geometry.mark_region("walls", boundary="y_min")
geometry.mark_region("walls", boundary="y_max")  # Adds to existing "walls" region
```

Predicates receive grid points as `(N, dimension)` array and must return boolean mask of length `N`.

### Priority Resolution for Overlapping Regions

When regions overlap, **priority** determines which BC wins (higher number = higher precedence):

```python
# Mark overlapping regions
geometry.mark_region("broad", predicate=lambda x: x[:, 0] < 0.5)
geometry.mark_region("narrow", predicate=lambda x: x[:, 0] < 0.1)

# Create segments with priorities
bc_narrow = BCSegment(
    name="narrow_bc",
    bc_type=BCType.DIRICHLET,
    value=1.0,
    region_name="narrow",
    priority=2,  # Higher number = higher precedence
)
bc_broad = BCSegment(
    name="broad_bc",
    bc_type=BCType.NEUMANN,
    value=0.0,
    region_name="broad",
    priority=1,  # Lower number = lower precedence
)

bc = BoundaryConditions(
    dimension=2,
    segments=[bc_narrow, bc_broad],  # Order doesn't matter
    default_bc=BCType.PERIODIC,
    domain_bounds=geometry.bounds,
)

# At x=0.05 (in both regions): narrow_bc wins due to higher priority
# At x=0.3 (only in broad): broad_bc applies
```

### Boundary-Based vs Region-Based

| Approach | Use When | Example |
|----------|----------|---------|
| **Boundary-based** | Simple geometries, entire boundaries | `boundary="x_min"` |
| **Region-based** | Complex geometries, partial boundaries | `predicate=lambda x: x[:, 0] < 0.1` |

Both approaches can be mixed in the same BC specification.

### Performance Notes

- Region lookup overhead: <5% compared to standard boundary-based BCs
- Region masks are cached after first call to `mark_region()`
- For best performance, mark all regions before solver iteration

## Boundary Naming Convention

| Dimension | Boundaries |
|-----------|------------|
| 1D | `x_min`, `x_max` |
| 2D | `x_min`, `x_max`, `y_min`, `y_max` |
| 3D | `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max` |
| 4D+ | `dim0_min`, `dim0_max`, `dim1_min`, ... |

## Ghost Cell Method (FDM)

For finite difference methods, BCs are enforced via ghost cells:

```python
from mfg_pde.geometry.boundary import apply_boundary_conditions_2d

# field has shape (Ny, Nx) - interior points only
padded = apply_boundary_conditions_2d(field, bc, domain_bounds)
# padded has shape (Ny+2, Nx+2) - includes ghost cells
```

### Ghost Cell Formulas

For cell-centered grids with boundary at cell face:

- **Dirichlet** (u = g): `u_ghost = 2*g - u_interior`
- **Neumann** (du/dn = g): `u_ghost = u_interior + 2*dx*g` (outward normal)
- **No-flux** (du/dn = 0): `u_ghost = u_interior`

## Corner Handling

When different BCs meet at corners (e.g., Dirichlet on one edge, Neumann on adjacent edge), the ghost cell value is computed using **averaging**:

```
corner_ghost = 0.5 * (adjacent_edge_ghost_1 + adjacent_edge_ghost_2)
```

### Why Averaging?

1. **Numerical stability**: Avoids discontinuities at corners
2. **BC agnostic**: Works for any combination of BC types
3. **Smooth solutions**: Produces well-behaved ghost values

### Corner Strategies (Advanced)

For special cases, alternative strategies may be more appropriate:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Average** (default) | Mean of adjacent ghost values | General purpose |
| **Priority** | Dirichlet takes precedence over Neumann | Sharp BC interfaces |
| **Extrapolate** | From interior points | When BCs don't dominate |

The default averaging strategy is recommended for most MFG applications.

## Time-Dependent BCs

BCs can vary with time:

```python
# Time-varying Dirichlet
def inlet_profile(t):
    return 1.0 - np.exp(-t)

bc = dirichlet_bc(dimension=2, value=inlet_profile)

# Apply at specific time
padded = apply_boundary_conditions_2d(field, bc, bounds, time=0.5)
```

For spatially-varying BCs:

```python
# Function of position and time
def inlet_profile(point, time):
    x, y = point
    return np.sin(np.pi * y) * np.exp(-time)

exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=inlet_profile,
    boundary="x_max",
)
```

## Lazy Dimension Binding

BCs can be created without specifying dimension, then bound when attached to geometry:

```python
# Create BC without dimension
bc = neumann_bc()  # dimension=None

# Dimension auto-detected when used with geometry
grid = TensorProductGrid(bounds=[[0,1], [0,1]], num_points=[51, 51])
grid.boundary_conditions = bc  # Automatically binds to dimension=2
```

## Solver-Specific Notes

### HJB Solvers (FDM, WENO, GFDM)

Ghost values are used for upwind gradient computation:
- At left boundary with rightward flow: backward difference uses ghost
- At right boundary with leftward flow: forward difference uses ghost

### FP Solvers (FDM)

Ghost values ensure mass conservation at boundaries:
- No-flux BC: ghost = interior (zero gradient)
- Dirichlet BC: ghost reflects the boundary value

### Particle Methods

Particles are reflected at boundaries:
- Normal reflection for no-flux
- Absorption for Dirichlet (particle removed or reset)

## Performance Tips

1. **Pre-compute masks**: For mixed BCs, use `create_boundary_mask_2d()` to identify segments once
2. **Avoid callable BCs when possible**: Constant values are faster
3. **Use uniform BCs**: Simpler path, less overhead than mixed BCs

## Common Issues

### "BC dimension not set"

```python
# Wrong: using unbound BC
bc = neumann_bc()  # dimension=None
apply_boundary_conditions_2d(field, bc, bounds)  # Error!

# Correct: bind dimension first
bc = bc.bind_dimension(2)
apply_boundary_conditions_2d(field, bc, bounds)  # Works
```

### "domain_bounds required for mixed BC"

```python
# Wrong: mixed BC without bounds
bc = mixed_bc([seg1, seg2], dimension=2)
apply_boundary_conditions_2d(field, bc)  # Error!

# Correct: provide bounds
apply_boundary_conditions_2d(field, bc, domain_bounds=bounds)
```

## See Also

- `mfg_pde.geometry.boundary` module documentation
- `docs/development/reports/BC_CORNER_LOGIC_AUDIT.md` - Technical details on corner handling
- `docs/development/reports/BC_UNIFICATION_TECHNICAL_REPORT.md` - Architecture overview
