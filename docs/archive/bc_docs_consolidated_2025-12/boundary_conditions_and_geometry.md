# Boundary Conditions and Geometry: Algorithm Design Guide

This document provides guidance for choosing discretization methods based on domain geometry and boundary condition requirements in MFG problems.

## Architecture Overview

MFG_PDE separates boundary condition handling into three layers:

```
Layer 1: BC Specification (geometry/boundary/conditions.py)
         - BoundaryConditions class
         - BCType enum: DIRICHLET, NEUMANN, ROBIN, PERIODIC, NO_FLUX
         - Supports domain_bounds (rectangular) or domain_sdf (general)

Layer 2: BC Application (geometry/boundary/applicator_*.py)
         - FDMApplicator: Ghost cell padding for structured grids
         - FEMApplicator: DOF modification for mesh-based methods
         - MeshfreeApplicator: Particle reflection at boundaries
         - GraphApplicator: Maze/network boundary handling

Layer 3: Solver Integration (alg/numerical/)
         - Solvers call applicators to enforce BCs
         - BC-specific matrix modifications (row elimination, stencil adjustment)
```

## Method Selection by Geometry

### Quick Reference

| Domain Type | FDM | FEM | Meshfree/GFDM |
|-------------|:---:|:---:|:-------------:|
| Rectangle (1D/2D/3D) | Recommended | OK | OK |
| Rectangle + rectangular obstacles | OK (masking) | OK | OK |
| Circle, ellipse | Not recommended | Recommended | OK |
| L-shape, polygon | Not recommended | Recommended | OK |
| Arbitrary SDF boundary | Not feasible | OK | Recommended |
| Moving boundaries | Not feasible | Difficult | Recommended |

### Detailed Analysis

#### Finite Difference Method (FDM)

**Best for**: Hyperrectangular domains with axis-aligned boundaries.

**Why FDM works well on rectangles**:
- Grid points naturally align with boundaries
- Ghost point formulas are simple and well-understood
- Sparse matrix structure is optimal
- Fast solvers available (FFT, multigrid)

**FDM on rectangles with rectangular obstacles**:
```
+------------------+
|                  |
|    +----+        |   Grid masking approach:
|    |hole|        |   - Mark obstacle points
|    +----+        |   - Apply interior BCs at obstacle edges
|                  |
+------------------+
```
This works because obstacles are still axis-aligned.

**FDM on complex geometry - why it fails**:

1. **Boundary-grid misalignment**: When boundary doesn't pass through grid points,
   ghost point formulas introduce O(dx) errors instead of O(dx^2).

2. **Staircase approximation**: Curved boundaries become jagged steps:
   ```
   Actual boundary:     FDM approximation:
        ___                +--+
      /     \             +    +
     |       |           +      +
      \_____/            +--+--+
   ```
   This causes:
   - Mass leakage at corners
   - Spurious boundary layer effects
   - O(dx) convergence instead of O(dx^2)

3. **Normal direction ambiguity**: At staircase corners, which direction is "outward normal"?

**When FDM is still acceptable for non-rectangular domains**:
- Accuracy requirements are moderate (O(dx) acceptable)
- Grid resolution is high enough that staircase error is small
- Boundary layer behavior is not critical
- Fast prototyping is more valuable than precision

#### Finite Element Method (FEM)

**Best for**: Complex geometry with accurate boundary representation.

**Advantages**:
- Mesh conforms to boundary exactly
- Natural handling of Neumann/Robin BCs via weak form
- Higher-order elements available
- Well-established error theory

**Disadvantages**:
- Mesh generation required (external tools like gmsh, Triangle)
- Matrix assembly more complex than FDM
- Mesh quality affects solution quality

**Implementation status in MFG_PDE**:
- `FEMApplicator` available
- 1D/2D/3D boundary condition classes implemented
- Less tested than FDM solvers

#### Meshfree / GFDM Methods

**Best for**: Arbitrary domains, moving boundaries, scattered data.

**Advantages**:
- No mesh required - just point cloud
- Natural for Lagrangian/moving domains
- Easy to add/remove points adaptively
- SDF-based boundary handling is natural

**Disadvantages**:
- Stencil construction more expensive
- Consistency/stability analysis more complex
- Less mature solvers than FDM/FEM

**Implementation status in MFG_PDE**:
- `MeshfreeApplicator` with `ParticleReflector`
- GFDM solvers available
- Good for particle-based MFG methods

## Boundary Condition Types

### Mathematical Formulation

| BC Type | Equation | Physical Meaning |
|---------|----------|------------------|
| Dirichlet | $u = g$ | Fixed value at boundary |
| Neumann | $\partial u/\partial n = g$ | Fixed flux at boundary |
| Robin | $\alpha u + \beta \partial u/\partial n = g$ | Mixed condition |
| No-flux | $\partial u/\partial n = 0$ | Reflecting boundary |
| Periodic | $u(x_{min}) = u(x_{max})$ | Wrap-around |

### Ghost Point Formulas (FDM)

For cell-centered grid with boundary at cell face:

**Dirichlet** ($u = g$ at boundary):
$$u_{ghost} = 2g - u_{interior}$$

**Neumann** ($\partial u/\partial n = g$ at boundary):
$$u_{ghost} = u_{interior} \mp 2 \Delta x \cdot g$$
(sign depends on boundary orientation)

**No-flux** (Neumann with $g=0$):
$$u_{ghost} = u_{interior}$$
This preserves conservation: row sum of stencil remains zero.

### Conservation Properties

For Fokker-Planck equations, boundary conditions affect mass conservation:

| BC Type | Mass Conserving? | Notes |
|---------|------------------|-------|
| No-flux (Neumann g=0) | Yes | Standard for FP |
| Dirichlet | No | Mass absorbed at boundary |
| Robin | Depends on coefficients | |
| Periodic | Yes | Mass wraps around |

## Implementation Guidelines

### When to Use Each Applicator

```python
from mfg_pde.geometry.boundary import (
    FDMApplicator,
    FEMApplicator,
    MeshfreeApplicator,
    BoundaryConditions,
)

# Rectangular domain - use FDM
bc = BoundaryConditions(dimension=2, bc_type=BCType.NEUMANN,
                        domain_bounds=np.array([[0, 1], [0, 1]]))
applicator = FDMApplicator(dimension=2)

# Complex domain with mesh - use FEM
bc = BoundaryConditions(dimension=2, bc_type=BCType.DIRICHLET,
                        domain_sdf=my_sdf_function)
applicator = FEMApplicator(dimension=2)

# Particle-based method - use Meshfree
applicator = MeshfreeApplicator(dimension=2)
reflector = ParticleReflector(domain_sdf=my_sdf_function)
```

### SDF-Based Domains

The `BoundaryConditions` class supports SDF specification:

```python
# Circle domain
def circle_sdf(x):
    return np.linalg.norm(x - center) - radius

bc = BoundaryConditions(
    dimension=2,
    bc_type=BCType.NEUMANN,
    domain_sdf=circle_sdf
)

# Query boundary properties
bc.is_on_boundary(point)       # |phi| < tolerance
bc.get_outward_normal(point)   # gradient of SDF
bc.get_boundary_id(point)      # "x_min", "x_max", etc.
```

**Important**: The SDF machinery in `BoundaryConditions` is fully implemented,
but `FDMApplicator` does NOT use it. For SDF-based domains, use FEM or Meshfree.

## Lessons Learned

### BCTransforms Effort (Issue #379, Closed)

An attempt was made to create generic BC stencil transforms that would modify
any operator stencil at boundaries. This was closed because:

1. **Narrow scope**: Only handled 1D hyperrectangle FDM
2. **Existing infrastructure sufficient**: `apply_boundary_conditions_*d()` functions already work
3. **Premature abstraction**: The "combinatorial explosion" (N_operators x N_bcs) doesn't exist in practice

**Takeaway**: Don't over-engineer BC handling. Each solver has ~10 lines of BC code;
abstracting this provides marginal value.

### Design Principles

1. **Separation of concerns**: BC specification (what) vs. BC application (how)
2. **Method-appropriate applicators**: FDM, FEM, Meshfree each have their own
3. **Don't fight the geometry**: Use FDM for rectangles, FEM/Meshfree for complex domains

## Code References

- BC specification: `mfg_pde/geometry/boundary/conditions.py`
- BC types: `mfg_pde/geometry/boundary/types.py`
- FDM applicator: `mfg_pde/geometry/boundary/applicator_fdm.py`
- FEM applicator: `mfg_pde/geometry/boundary/applicator_fem.py`
- Meshfree applicator: `mfg_pde/geometry/boundary/applicator_meshfree.py`
- Graph applicator: `mfg_pde/geometry/boundary/applicator_graph.py`

## Related Documents

- `geometry_projection_mathematical_formulation.md` - Projection methods
- `fp_hjb_coupling.md` - FP-HJB system boundary treatment
- GitHub Issue #379 (closed) - BCTransforms design discussion
