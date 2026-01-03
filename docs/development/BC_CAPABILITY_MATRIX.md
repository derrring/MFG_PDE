# Boundary Condition Capability Matrix

**Created**: 2025-12-18
**Status**: Active reference document
**Related**:
- Issue #527 (BC Infrastructure Integration)
- Issue #535 (BC Framework Enhancement: L-S Validation, Boundary Matrices, Neural Interface)
- Issue #536 (Particle Absorbing BC)
- [boundary_framework_implementation_design.md](./boundary_framework_implementation_design.md) (implementation roadmap)
- [boundary_framework_mathematical_foundation.md](../theory/boundary_framework_mathematical_foundation.md) (theory)

---

## Overview

This document maps boundary condition support across all numerical solvers in MFG_PDE. It serves as:
1. Reference for users selecting appropriate solvers
2. Gap analysis for infrastructure integration
3. Guide for new solver development

---

## BC Type Support by Solver

| Solver | Dirichlet | Neumann | Robin | Periodic | No-flux | Reflecting | Extrapolation |
|--------|:---------:|:-------:|:-----:|:--------:|:-------:|:----------:|:-------------:|
| **HJB FDM** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **HJB GFDM** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **HJB Semi-Lagrangian** | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **HJB WENO** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **FP FDM** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **FP Particle** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **FP GFDM** | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |

**Legend**: ✅ Supported | ❌ Not supported | ⚠️ Partial/Limited

---

## Infrastructure Integration Level

| Solver | Uses `geometry/boundary/` | BoundaryCapable | Implementation | Integration Gap |
|--------|:-------------------------:|:---------------:|----------------|-----------------|
| **HJB FDM** | ✅ `get_ghost_values_nd` | ❌ | Ghost cells via PreallocatedGhostBuffer | Low |
| **HJB GFDM** | ⚠️ Type annotations only | ✅ | Custom normal-based weights | **Medium** |
| **HJB Semi-Lagrangian** | ⚠️ `periodic_bc` only | ❌ | Characteristic clamping | Medium |
| **HJB WENO** | ✅ `PreallocatedGhostBuffer` | ❌ | Ghost cells (depth=2) | Low |
| **FP FDM** | ✅ Factory functions | ❌ | Delegates to `fp_fdm_time_stepping` | Low |
| **FP Particle** | ⚠️ `periodic_bc` factory | ❌ | Custom particle reflection | Medium |
| **FP GFDM** | ❌ None | ❌ | Custom `boundary_type` string | **High** |

### Integration Status (Issue #527)

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | BoundaryCapable protocol, dispatch.py, documentation | ✅ Complete |
| Phase 2 | Central application point (`Geometry.apply_bc()`) | ❌ Not started |
| Phase 3 | GFDM/Solver delegation to applicators | ❌ Not started |
| Phase 4 | Advanced BC types (Robin, Mixed, High-order) | ❌ Not started |

**Note**: `dispatch.py` contains `apply_bc()`, `get_applicator_for_geometry()` but these are
not yet wired into solvers. Phase 2-4 will complete the integration.

---

## Detailed Solver Analysis

### HJB FDM Solver

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`

| Aspect | Details |
|--------|---------|
| **Dimensions** | 1D (optimized), nD (general) |
| **Ghost Depth** | 1 cell per side |
| **BC Source** | `geometry.boundary_conditions` or `problem.boundary_conditions` |
| **Fallback** | One-sided stencils (warns once) |
| **Key Import** | `from mfg_pde.geometry.boundary import get_ghost_values_nd` |

**BC Application Pattern**:
```python
# Ghost values computed per timestep
ghost_values = get_ghost_values_nd(u, bc, domain_bounds)
# Then used in gradient computation
```

### HJB GFDM Solver

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`

| Aspect | Details |
|--------|---------|
| **Dimensions** | Meshfree (any dimension) |
| **Ghost Depth** | N/A (meshfree) |
| **BC Source** | `BoundaryConditions` parameter |
| **Normal Computation** | `BoundaryConditions.get_outward_normal()` (SDF-based) |
| **Key Import** | `from mfg_pde.geometry import BoundaryConditions` |

**BC Application Pattern**:
```python
# Custom Neumann weight construction
# Uses outward normals for directional derivatives
normal = bc.get_outward_normal(point)
# Builds weights satisfying du/dn = g
```

**Gap**: Does not use `MeshfreeApplicator` from `applicator_meshfree.py`.

### HJB Semi-Lagrangian Solver

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py`

| Aspect | Details |
|--------|---------|
| **Dimensions** | 1D (production), nD (supported) |
| **BC Mechanism** | Characteristic clamping at boundaries |
| **BC Source** | `problem.get_boundary_conditions()` |
| **Key Import** | `from mfg_pde.geometry.boundary.conditions import periodic_bc` |

**BC Application Pattern**:
```python
# Characteristics traced backward; clamped at domain boundaries
x_foot = trace_characteristic(x, dt, velocity)
x_foot = np.clip(x_foot, xmin, xmax)  # Simple clamping
```

**Gap**: Only periodic/no-flux; no Dirichlet or mixed BC support.

### HJB WENO Solver

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`

| Aspect | Details |
|--------|---------|
| **Dimensions** | 1D direct; nD via dimensional splitting |
| **Ghost Depth** | 2 cells per side (WENO5 stencil) |
| **BC Source** | `problem.geometry.boundary_conditions` |
| **Default BC** | `neumann_bc()` (no-flux) |
| **Key Import** | `from mfg_pde.geometry.boundary import PreallocatedGhostBuffer, neumann_bc` |

**BC Application Pattern**:
```python
self.ghost_buffer = PreallocatedGhostBuffer(
    interior_shape=shape,
    boundary_conditions=bc,
    ghost_depth=2  # WENO5 requirement
)
# Update ghosts before each WENO reconstruction
ghost_buffer.update_ghosts()
```

### FP FDM Solver

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

| Aspect | Details |
|--------|---------|
| **Dimensions** | 1D, 2D, 3D, nD |
| **Advection Schemes** | `gradient_upwind` (default), `divergence_upwind`, centered variants |
| **BC Source** | `BoundaryConditions` parameter |
| **Default BC** | `no_flux_bc()` |
| **Key Import** | `from mfg_pde.geometry import BoundaryConditions` |

**BC Application Pattern**:
```python
# Delegates to fp_fdm_time_stepping module
# BC applied during matrix assembly
bc_type = self._get_bc_type()
bc_value = self._get_bc_value(t_idx)
```

### FP Particle Solver

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`

| Aspect | Details |
|--------|---------|
| **Dimensions** | Any (SDE-based) |
| **BC Mechanism** | Particle reflection at boundaries |
| **BC Source** | Optional `BoundaryConditions` parameter |
| **Default BC** | `periodic_bc()` |
| **Key Import** | `from mfg_pde.geometry.boundary.conditions import periodic_bc` |

**BC Application Pattern**:
```python
# Particles evolved via SDE
# Reflection at boundaries (elastic)
particles = np.clip(particles, xmin, xmax)  # Simple version
# Or elastic reflection: v -> -v at boundary
```

**Gap**: Does not use `ParticleReflector` from `applicator_meshfree.py`.

### FP GFDM Solver

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_gfdm.py`

| Aspect | Details |
|--------|---------|
| **Dimensions** | Meshfree (any dimension) |
| **BC Source** | `boundary_type` string parameter |
| **Current Support** | `"no_flux"` only |
| **Key Import** | None from `geometry/boundary/` |

**BC Application Pattern**:
```python
# Constructor accepts boundary_type string
fp_solver = FPGFDMSolver(
    problem,
    collocation_points=points,
    boundary_type="no_flux"  # Only option
)
```

**Gap**: No integration with unified BC infrastructure.

---

## Geometry/Boundary Infrastructure

### Available BC Types (`geometry/boundary/types.py`)

```python
class BCType(Enum):
    DIRICHLET = "dirichlet"              # u = g
    NEUMANN = "neumann"                  # du/dn = g
    ROBIN = "robin"                      # alpha*u + beta*du/dn = g
    PERIODIC = "periodic"                # u(x_min) = u(x_max)
    REFLECTING = "reflecting"            # Particle reflection
    NO_FLUX = "no_flux"                  # J·n = 0 (mass conservation)
    EXTRAPOLATION_LINEAR = "linear"      # d²u/dx² = 0
    EXTRAPOLATION_QUADRATIC = "quadratic" # d³u/dx³ = 0
```

### Applicator Classes

| Applicator | Target Method | File | LOC |
|------------|---------------|------|-----|
| `FDMApplicator` | FDM (ghost cells) | `applicator_fdm.py` | 2,427 |
| `FEMApplicator` | FEM (matrix assembly) | `applicator_fem.py` | 614 |
| `MeshfreeApplicator` | Collocation/Particles | `applicator_meshfree.py` | 614 |
| `GraphApplicator` | Network domains | `applicator_graph.py` | 829 |

### Factory Functions (`geometry/boundary/conditions.py`)

```python
# Create uniform BCs
bc = periodic_bc(dimension=2)
bc = neumann_bc(dimension=2, value=0.0)
bc = dirichlet_bc(dimension=2, value=lambda x: np.sin(x[0]))
bc = no_flux_bc(dimension=2)

# Create mixed BCs
bc = mixed_bc(dimension=2, segments=[
    BCSegment(BCType.DIRICHLET, region="left", value=0.0),
    BCSegment(BCType.NEUMANN, region="right", value=1.0),
])
```

---

## Integration Recommendations

### For New Solvers

1. Accept `BoundaryConditions` parameter (or extract from `problem.geometry`)
2. Default to `no_flux_bc(dimension)` for field methods
3. Use appropriate applicator:
   - FDM: `PreallocatedGhostBuffer`
   - Meshfree: `MeshfreeApplicator`
   - Particle: `ParticleReflector`

### Priority Gaps to Address

| Priority | Solver | Gap | Solution |
|----------|--------|-----|----------|
| **High** | HJB GFDM | Custom BC, no applicator | Integrate `MeshfreeApplicator` |
| **High** | FP GFDM | No BC infrastructure | Add `BoundaryConditions` support |
| **Medium** | FP Particle | Custom reflection | Use `ParticleReflector` |
| **Medium** | HJB Semi-Lag | Clamping only | Add characteristic reflection |
| **Low** | All | Robin BC unused | Enable in FP solvers |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-18 | Initial capability matrix |
