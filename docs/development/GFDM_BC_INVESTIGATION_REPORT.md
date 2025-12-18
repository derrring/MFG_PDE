# GFDM Boundary Condition Handling Investigation Report

**Date**: 2025-12-18 (Updated: Integration with MFG_PDE Infrastructure)
**Status**: In Progress - Infrastructure Integration Phase
**Authors**: Claude Code Investigation

---

## Executive Summary

Investigation of HJB-GFDM solver producing catastrophic negative values (U(0) min = -225.6 vs FDM min = -0.06) revealed two distinct architectural issues:

1. **Geometric Issue (Fixed)**: Ghost particles corrupting derivative computation
2. **Algebraic Issue (Partially Implemented)**: Missing Row Replacement for Neumann BC
3. **Infrastructure Integration (New)**: GFDM should use existing MFG_PDE BC infrastructure

Both issues stem from conflating **derivative computation** with **boundary condition enforcement**.

**Key Finding**: The MFG_PDE codebase already has mature BC infrastructure (`BoundaryConditions`, `MeshfreeApplicator`, Calculator/Topology pattern) that GFDM should integrate with rather than bypass. See **Section 11** for integration details.

---

## 1. Problem Statement

### Observed Behavior
- GFDM solver producing `U(0) min = -225.6` (catastrophic negatives)
- FDM solver producing `U(0) min = -0.06` (physically reasonable)
- Both solvers using same MFG problem (2D crowd evacuation with no-flux BC)

### Expected Behavior
- GFDM should produce results comparable to FDM
- No negative values for this value function problem

---

## 2. Root Cause Analysis

### 2.1 The "Coupling Fallacy"

The original implementation attempted to make a **single operator matrix** responsible for both:
1. **Geometric approximation** (high-order derivatives)
2. **Constraint enforcement** (physics at boundary)

This is mathematically incorrect.

### 2.2 Ghost Particle Mechanism (Original Design)

For a boundary point at x=0 with no-flux BC (∂u/∂n = 0):

1. GFDM created a ghost particle at x = -h (outside domain)
2. Set `u(ghost) = u(center)` to enforce symmetry
3. Taylor matrix A included ghost position
4. Derivative weights computed assuming ghost value = center value

**The Problem**: This approach only works for functions that already satisfy ∂u/∂n = 0. For arbitrary functions (e.g., during Newton iteration), the operator returns biased derivatives:

| Function | Exact ∂u/∂x at x=0 | Ghost-Based Computed | Error |
|----------|-------------------|---------------------|-------|
| u = 10x² - 20x | -20.0 | -7.25 | 64% |
| u = 10x² (satisfies BC) | 0.0 | 0.19 | ~0% |

### 2.3 Mathematical Derivation

Taylor expansion at x=0 with neighbors at x=h and ghost at x=-h:

```
u(x) ≈ u(0) + u'(0)·x + (1/2)·u''(0)·x² + ...

For neighbor at x=h:    u(h) - u(0) = u'(0)·h + (1/2)·u''(0)·h²
For ghost at x=-h:      u(-h) - u(0) = -u'(0)·h + (1/2)·u''(0)·h²

Ghost constraint: u(-h) = u(0)
→ u(-h) - u(0) = 0 = -u'(0)·h + (1/2)·u''(0)·h²
→ u'(0) = (h/2)·u''(0) ≈ 0 (for small h)
```

The ghost constraint **forces** the computed derivative toward zero, regardless of the actual function values.

---

## 3. The Correct Architecture: Operator-Constraint Separation

### 3.1 Principle

Split responsibilities into two distinct components:

| Component | Responsibility | BC Knowledge |
|-----------|----------------|--------------|
| **Differential Operator** | Compute ∇u, Δu from values | None (purely geometric) |
| **BC Enforcement** | Populate system with BC equations | Full BC specification |

### 3.2 Implementation Pattern: Direct Collocation with Row Replacement

**Step 1: Geometry (bc_type = None)**
- No ghost particles
- One-sided stencils at boundaries (natural GFDM behavior)
- Derivative operator works for ANY smooth function

**Step 2: Algebra (Row Replacement)**
- Build PDE operator matrix L for ALL points (including boundaries)
- Identify boundary rows
- **Overwrite** boundary rows with BC operator B

For system Au = f:
```
Interior rows:  L·u = f  (PDE equation)
Boundary rows:  B·u = g  (BC equation)
```

For Neumann BC (∂u/∂n = 0):
```
B[i, :] = [normal derivative weights]
g[i] = 0
```

---

## 4. Implementation Status

### 4.1 Geometric Fix (Completed)

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`

**Change**: Removed ghost particles from GFDMOperator initialization

```python
# Before (WRONG):
bc_type = "no_flux" if bc_name.lower() == "neumann" else ...
self._gfdm_operator = GFDMOperator(..., boundary_type=bc_type, ...)

# After (CORRECT):
self._gfdm_operator = GFDMOperator(..., boundary_type=None, ...)
```

**Result**:
- Gradient error reduced by **8.5×** (2.99 → 0.35)
- Laplacian error reduced by **8.3×** (61.8 → 7.4)
- U(0) min: -225.6 → +0.046 (no more negatives)

### 4.2 Algebraic Fix (Implemented but Not Activated)

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`

**Added Methods**:
1. `_compute_outward_normal(point_idx)` - Compute normal vector at boundary
2. `_build_neumann_bc_weights()` - Compute GFDM weights for ∂u/∂n
3. Modified `_apply_boundary_conditions_to_sparse_system()` - Row Replacement

**Issue**: BC type detection is inconsistent:
```python
# Line 1304 (sparse system): default = "neumann"
bc_type = self._get_boundary_condition_property("type", "neumann")

# Line 1818 (solution): default = "dirichlet"
bc_type_val = self._get_boundary_condition_property("type", "dirichlet")
```

### 4.3 Remaining Gap

Even with both fixes, FDM and GFDM results differ (mean difference ~10.5):

| Metric | FDM | GFDM | Difference |
|--------|-----|------|------------|
| U(0) min | -0.0625 | +0.0460 | 0.11 |
| U(0) max | 108.99 | 145.78 | 36.79 |
| Mean diff | - | - | 10.54 |

Possible causes:
1. Different solver types (FDM: fixed_point, GFDM: newton)
2. Interpolation between collocation points and grid
3. BC not being properly passed to GFDM solver (diagnostic doesn't set BC)

---

## 5. Expert Critique and Revised Analysis

### 5.1 The "Missing 10.5" is Almost Certainly Upwinding

**Original hypothesis** (Section 4.3): Interpolation or solver types cause the discrepancy.

**Expert correction**: This is too optimistic. The discrepancy is most likely due to **Numerical Dissipation (or lack thereof)**.

| Solver | Advection Handling | Result |
|--------|-------------------|--------|
| **FDM** | Implicit upwinding (forward/backward based on drift) | Artificial viscosity, stable, slight smearing |
| **GFDM** | Central differences (standard least-squares) | Non-dissipative, oscillations around true solution |

**The Risk**: GFDM solution is "oscillating" (Gibbs phenomenon) while FDM is "smearing" slightly to stay stable.

**Critical Insight**: Cannot match FDM results without implementing **Upwind GFDM**. This is mathematically required for convergence to the viscosity solution of HJB.

### 5.2 Normal Vector Estimation is Fragile

The current `_compute_outward_normal()` implementation:
```python
def _compute_outward_normal(self, point_idx):
    point = self.collocation_points[point_idx]
    for d in range(self.dimension):
        if abs(point[d] - low) < tol:
            normal[d] -= 1.0
        if abs(point[d] - high) < tol:
            normal[d] += 1.0
```

**Problem**: This only works for rectangular domains aligned with coordinate axes.

**Expert warning**: If computed normal deviates by even 5° from true geometry, the Neumann condition `∇u · n = 0` effectively pumps flux into the domain, destroying conservation.

**Required fix for unstructured meshes**:
```python
def _compute_normal_pca(self, point_idx):
    """Robust normal estimation using PCA on boundary neighbors."""
    # Get k nearest boundary neighbors
    boundary_neighbors = self._get_boundary_neighbors(point_idx, k=10)
    points = self.collocation_points[boundary_neighbors]

    # PCA to find tangent plane
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Normal is eigenvector with smallest eigenvalue
    normal = eigenvectors[:, 0]

    # Orient outward (away from domain center)
    center = self.collocation_points.mean(axis=0)
    if np.dot(normal, self.collocation_points[point_idx] - center) < 0:
        normal = -normal

    return normal
```

### 5.3 RBF-FD Does Not Fix Physics

**Misconception**: RBF-FD is a "magic bullet" for the discrepancy.

**Reality**: RBF-FD fixes **numerical conditioning** (prevents singular matrices), but:
- RBF-FD weights are typically "centered" (isotropic)
- Applying RBF-FD to HJB without "upwind RBF" techniques → same oscillations

**Correct approach**:
1. Fix **physics** (Upwinding + BCs) in Polynomial framework first
2. Verify < 1% error vs FDM
3. *Then* switch to RBF-FD for better geometric flexibility

### 5.4 Configuration Drift is a "Silent Killer"

**The problem** (Section 4.2):
```python
# Line 1304 (_apply_boundary_conditions_to_sparse_system):
bc_type = self._get_boundary_condition_property("type", "neumann")

# Line 1818 (_apply_boundary_conditions_to_solution):
bc_type_val = self._get_boundary_condition_property("type", "dirichlet")
```

**Why this is dangerous**: If matrix builder thinks `bc='neumann'` but residual computer thinks `bc='dirichlet'`, Newton solver diverges because **Jacobian does not match residual**.

**Required fix**: Single Source of Truth via `BoundaryConditionManager`:

```python
class BoundaryConditionManager:
    """Single source of truth for boundary condition handling."""

    def __init__(self, config, geometry, gfdm_operator):
        self.config = config
        self.geometry = geometry
        self.gfdm_operator = gfdm_operator
        self.boundary_indices = self._identify_boundary()
        self.normals = self._compute_robust_normals()  # PCA-based

    def get_bc_type(self, point_idx: int) -> str:
        """Get BC type for a specific point."""
        # Check for spatially-varying BCs
        if hasattr(self.config, 'get_bc_at_point'):
            return self.config.get_bc_at_point(point_idx)
        return self.config.default_bc

    def apply_to_matrix(self, A, b):
        """Row Replacement pattern - algebraic BC enforcement."""
        for idx in self.boundary_indices:
            bc_type = self.get_bc_type(idx)

            if bc_type == 'neumann':
                # Clear row
                A[idx, :] = 0

                # Get GFDM weights
                weights = self.gfdm_operator.get_derivative_weights(idx)
                grad_w = weights['grad_weights']
                n = self.normals[idx]

                # Normal derivative: ∂u/∂n = n · ∇u
                for k, j in enumerate(weights['neighbor_indices']):
                    if j >= 0:
                        A[idx, j] = sum(n[d] * grad_w[d, k] for d in range(self.dim))

                # Center contribution
                A[idx, idx] = -A[idx, :].sum()
                b[idx] = 0.0

            elif bc_type == 'dirichlet':
                A[idx, :] = 0
                A[idx, idx] = 1.0
                b[idx] = self.config.get_bc_value(idx)

    def apply_to_residual(self, residual, u):
        """Compute BC residual - must match apply_to_matrix."""
        for idx in self.boundary_indices:
            bc_type = self.get_bc_type(idx)

            if bc_type == 'neumann':
                # Compute ∂u/∂n
                grad_u = self.gfdm_operator.gradient(u)[idx]
                n = self.normals[idx]
                residual[idx] = np.dot(grad_u, n)  # Should be 0

            elif bc_type == 'dirichlet':
                residual[idx] = u[idx] - self.config.get_bc_value(idx)
```

### 5.5 Revised Action Plan

**Phase 1: Close the "10.5 Gap" (Physics Consistency)**

1. **Implement Upwinding**: Modify `GFDMOperator` to accept `velocity_field`
   ```python
   def get_upwind_neighbors(self, point_idx, velocity):
       """Select neighbors upstream of velocity field."""
       neighbors = self.neighborhoods[point_idx]['indices']
       center = self.points[point_idx]
       directions = self.points[neighbors] - center
       # Keep neighbors where direction opposes velocity
       upwind_mask = np.dot(directions, -velocity) > 0
       return neighbors[upwind_mask]
   ```

2. **Verify Normals**: Add visualization debug
   ```python
   def plot_boundary_normals(self):
       """Debug: visualize computed normals."""
       plt.quiver(boundary_pts[:, 0], boundary_pts[:, 1],
                  normals[:, 0], normals[:, 1])
   ```

**Phase 2: Solidify BC Architecture**

1. Implement `BoundaryConditionManager` class
2. Pass single manager to both matrix builder AND residual computer
3. Add assertion: `assert matrix_bc_type == residual_bc_type`

**Phase 3: RBF-FD (Optimization)**

Only after Phase 1 & 2 pass with < 1% error vs FDM:
1. Implement `RBFFDOperator`
2. Add PHS + polynomial augmentation
3. Verify same results as Taylor-GFDM

---

## 6. Diagnostic Code

### 5.1 Ghost Particle Test

```python
from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator
import numpy as np

# Test function that does NOT satisfy no-flux BC
def test_func(pts):
    return 10 * (pts[:, 0]**2 + pts[:, 1]**2) - 20 * pts[:, 0] - 5 * pts[:, 1]

# WITH ghost particles (WRONG)
gfdm_ghost = GFDMOperator(points, boundary_type="no_flux", ...)
grad_ghost = gfdm_ghost.gradient(u)  # BIASED at boundaries

# WITHOUT ghost particles (CORRECT)
gfdm_no_ghost = GFDMOperator(points, boundary_type=None, ...)
grad_no_ghost = gfdm_no_ghost.gradient(u)  # ACCURATE everywhere
```

### 5.2 Validation Results

```
Grid: 11x11 = 121 points, 40 boundary points

Test 1: WITH ghost particles (boundary_type='no_flux')
  Gradient error (boundary): mean=9.04, max=21.07
  Laplacian error (boundary): mean=187.0, max=590.0

Test 2: WITHOUT ghost particles (boundary_type=None)
  Gradient error (boundary): mean=1.06, max=1.47
  Laplacian error (boundary): mean=22.5, max=41.6

Corner point (0,0):
  Exact gradient: (-20.00, -5.00)
  With ghost:     (-3.17, -0.67)  [error: 17.38]
  Without ghost:  (-18.95, -3.99) [error: 1.46]
```

---

## 7. Recommendations for Complete BC Handling

### 7.1 Unified BC Detection

Create consistent BC type resolution:

```python
def _resolve_bc_type(self) -> str:
    """Resolve BC type from multiple sources with consistent priority."""
    # 1. Check explicit boundary_conditions parameter
    if self.boundary_conditions:
        if hasattr(self.boundary_conditions, "default_bc"):
            return self.boundary_conditions.default_bc.value.lower()
        if "type" in self.boundary_conditions:
            return self.boundary_conditions["type"].lower()

    # 2. Check problem.geometry.boundary_conditions
    if hasattr(self.problem, "geometry") and hasattr(self.problem.geometry, "boundary_conditions"):
        bc = self.problem.geometry.boundary_conditions
        if hasattr(bc, "default_bc"):
            return bc.default_bc.value.lower()

    # 3. Default: neumann (no-flux) for MFG problems
    return "neumann"
```

### 7.2 BC-Aware Residual Computation

For Neumann BC, the residual at boundary should be ∂u/∂n (not PDE):

```python
def _compute_hjb_residual(self, u, time_idx, ...):
    residual = self._compute_pde_residual(u, ...)  # PDE at all points

    # Override boundary residuals with BC residual
    if self.bc_type == "neumann":
        for i in self.boundary_indices:
            normal = self._compute_outward_normal(i)
            grad_u = self._compute_gradient_at_point(u, i)
            residual[i] = np.dot(normal, grad_u)  # Should be 0 for BC satisfaction

    return residual
```

### 7.3 Complete Row Replacement Matrix

```python
def _build_system_matrix(self, u, time_idx):
    # Build PDE operator for all points
    jacobian = self._build_pde_jacobian(u, time_idx)

    # Row Replacement for boundary
    if self.bc_type == "neumann":
        for i in self.boundary_indices:
            jacobian[i, :] = 0  # Clear row
            normal = self._compute_outward_normal(i)
            weights = self._get_derivative_weights(i)
            for j, w in zip(weights["indices"], weights["grad"]):
                jacobian[i, j] = np.dot(normal, w)
    elif self.bc_type == "dirichlet":
        for i in self.boundary_indices:
            jacobian[i, :] = 0
            jacobian[i, i] = 1

    return jacobian
```

---

## 8. Testing Strategy

### 8.1 Unit Tests for Derivative Accuracy

```python
def test_gfdm_derivatives_without_ghost():
    """Verify GFDM computes accurate derivatives without ghost particles."""
    # Test with function that does NOT satisfy no-flux BC
    u = 10 * x**2 - 20 * x
    gfdm = GFDMOperator(points, boundary_type=None)
    grad = gfdm.gradient(u)

    # Should match exact derivatives at ALL points including boundary
    assert np.allclose(grad[:, 0], 20*x - 20, atol=0.5)
```

### 8.2 Integration Tests for BC Enforcement

```python
def test_neumann_bc_row_replacement():
    """Verify Row Replacement correctly enforces Neumann BC."""
    solver = HJBGFDMSolver(problem, boundary_conditions=neumann_bc)

    # Solve and check boundary condition satisfaction
    U = solver.solve()

    for i in solver.boundary_indices:
        normal = solver._compute_outward_normal(i)
        grad_u = compute_gradient(U[-1], i)
        assert abs(np.dot(normal, grad_u)) < 1e-3, "Neumann BC not satisfied"
```

---

## 9. Files Modified

| File | Changes |
|------|---------|
| `hjb_gfdm.py` | Removed ghost particles, added Row Replacement, added normal computation |
| `gfdm_operators.py` | (No changes - correctly supports bc_type=None) |

---

## 10. Next Steps

1. **Fix BC type inconsistency** - Unify defaults across all methods
2. **Ensure BC propagation** - Pass BoundaryConditions from problem to solver
3. **Test end-to-end** - Verify FDM/GFDM match with identical BC handling
4. **Document pattern** - Update architecture docs with Direct Collocation + Row Replacement

---

## 11. Existing MFG_PDE Boundary Condition Infrastructure

### 11.1 Architecture Overview

The MFG_PDE codebase already has a mature BC infrastructure that GFDM should integrate with rather than bypass:

```
mfg_pde/geometry/boundary/
├── conditions.py           # BoundaryConditions class (unified BC specification)
├── types.py               # BCType enum, BCSegment dataclass
├── applicator_base.py     # Base applicators + Calculator/Topology pattern
├── applicator_meshfree.py # MeshfreeApplicator for particle/collocation methods
├── applicator_fdm_*.py    # FDM-specific applicators
└── applicator_fem.py      # FEM-specific applicators
```

**Key Design Principle**: Separation of BC **specification** (what BC) from BC **application** (how to apply it).

### 11.2 Existing Normal Vector Computation

The codebase already provides robust normal vector computation in `conditions.py:450-488`:

```python
# mfg_pde/geometry/boundary/conditions.py

class BoundaryConditions:
    def get_outward_normal(self, point: np.ndarray, epsilon: float = 1e-5) -> np.ndarray | None:
        """
        Get the outward normal at a boundary point.

        Args:
            point: Spatial coordinates on the boundary
            epsilon: Finite difference step for SDF gradient

        Returns:
            Unit outward normal vector, or None if not available
        """
        dimension = self._require_dimension("get_outward_normal")
        point = np.asarray(point, dtype=float)

        # SDF domain: use gradient (general geometry)
        if self.domain_sdf is not None:
            normal = _compute_sdf_gradient(point, self.domain_sdf, epsilon=epsilon)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                return normal / normal_norm
            return None

        # Rectangular domain: use boundary ID (axis-aligned)
        if self.domain_bounds is not None:
            boundary_id = self.identify_boundary_id(point)
            if boundary_id is None:
                return None

            normal = np.zeros(dimension)
            for axis_idx in range(dimension):
                axis_name = ["x", "y", "z", "w"][axis_idx] if axis_idx < 4 else f"axis{axis_idx}"
                if boundary_id == f"{axis_name}_min":
                    normal[axis_idx] = -1.0
                    return normal
                elif boundary_id == f"{axis_name}_max":
                    normal[axis_idx] = 1.0
                    return normal

        return None
```

**Implication for GFDM**: The custom `_compute_outward_normal()` in `hjb_gfdm.py` should be **replaced** with calls to the existing `BoundaryConditions.get_outward_normal()`.

### 11.3 MeshfreeApplicator Integration

The `MeshfreeApplicator` class in `applicator_meshfree.py:41-316` provides:

```python
class MeshfreeApplicator(BaseMeshfreeApplicator):
    """
    Boundary condition applicator for meshfree methods.
    Works with any geometry implementing GeometryProtocol.
    """

    def get_boundary_mask(self, points, tolerance=1e-10) -> NDArray[np.bool_]:
        """Get boolean mask of boundary points."""
        return self.geometry.is_on_boundary(points, tolerance=tolerance)

    def get_boundary_normals(self, points) -> NDArray[np.floating]:
        """Get outward normal vectors at points."""
        return self.geometry.get_boundary_normal(points)

    def apply_field_bc(self, field, points, bc_type="dirichlet", bc_value=0.0, ...):
        """Apply boundary conditions to field values at collocation points."""
        # ... implementation
```

**Integration Path**: GFDM should use `MeshfreeApplicator` for:
1. Boundary detection: `get_boundary_mask(collocation_points)`
2. Normal computation: `get_boundary_normals(boundary_points)`
3. Field BC (if applicable): `apply_field_bc(...)`

### 11.4 SDFParticleBCHandler for Complex Geometry

For SDF-based domains, `SDFParticleBCHandler` (applicator_meshfree.py:318-517) provides:

- Bisection-based boundary point detection
- SDF gradient-based normal computation
- Physics-based elastic reflection

This is the correct approach for unstructured meshes / complex geometry.

### 11.5 Calculator/Topology Pattern (applicator_base.py)

The 2-layer BC architecture separates:

| Layer | Responsibility | GFDM Usage |
|-------|---------------|------------|
| **Topology** | Memory/indexing (periodic wrap vs bounded) | Boundary point identification |
| **Calculator** | Physics/values (ghost cell formulas) | Row Replacement values |

**Key Calculators available:**

| Calculator | Purpose | GFDM Equivalent |
|------------|---------|-----------------|
| `DirichletCalculator` | u = g | Row: `A[i, i] = 1, b[i] = g` |
| `NeumannCalculator` | du/dn = g | Row: Normal derivative weights |
| `ZeroGradientCalculator` | du/dn = 0 | Row: Normal derivative = 0 |
| `ZeroFluxCalculator` | J·n = 0 (mass conserving) | For FP: flux weights = 0 |

### 11.6 Recommended Integration for GFDM Solver

**Current (WRONG)**: Custom BC logic in `hjb_gfdm.py`

```python
def _compute_outward_normal(self, point_idx: int) -> np.ndarray:
    # Custom axis-aligned logic only
    ...
```

**Corrected (INTEGRATED)**:

```python
# In HJBGFDMSolver.__init__
from mfg_pde.geometry.boundary import MeshfreeApplicator

class HJBGFDMSolver:
    def __init__(self, problem, ...):
        ...
        # Use existing infrastructure
        self._bc_applicator = MeshfreeApplicator(problem.geometry)
        self._boundary_conditions = problem.geometry.boundary_conditions

    def _get_outward_normal(self, point_idx: int) -> np.ndarray:
        """Use existing infrastructure for normal computation."""
        point = self.collocation_points[point_idx]

        # Method 1: Direct from BoundaryConditions (if available)
        if hasattr(self._boundary_conditions, 'get_outward_normal'):
            normal = self._boundary_conditions.get_outward_normal(point)
            if normal is not None:
                return normal

        # Method 2: Via MeshfreeApplicator (batch operation)
        # For single point, reshape appropriately
        normals = self._bc_applicator.get_boundary_normals(
            point.reshape(1, -1)
        )
        return normals[0]
```

### 11.7 BoundaryConditionManager Design (New Component)

Based on the existing infrastructure, here's the recommended `BoundaryConditionManager`:

```python
# mfg_pde/alg/numerical/bc_manager.py

from mfg_pde.geometry.boundary import BoundaryConditions, MeshfreeApplicator
from mfg_pde.geometry.boundary.applicator_base import (
    DirichletCalculator,
    NeumannCalculator,
    ZeroFluxCalculator,
    LinearConstraint,
    calculator_to_constraint,
)

class BoundaryConditionManager:
    """
    Single source of truth for GFDM boundary condition handling.

    Integrates with existing MFG_PDE infrastructure to provide:
    - Consistent BC type resolution
    - Normal vector computation (via BoundaryConditions/MeshfreeApplicator)
    - Row Replacement matrix assembly
    - Residual computation at boundary
    """

    def __init__(
        self,
        boundary_conditions: BoundaryConditions,
        geometry,
        gfdm_operator,
        collocation_points: np.ndarray,
    ):
        self.boundary_conditions = boundary_conditions
        self.geometry = geometry
        self.gfdm_operator = gfdm_operator
        self.collocation_points = collocation_points
        self.dimension = collocation_points.shape[1]

        # Use existing applicator for boundary operations
        self._applicator = MeshfreeApplicator(geometry)

        # Cache boundary indices and normals
        self._identify_boundary_points()
        self._compute_boundary_normals()

    def _identify_boundary_points(self):
        """Use MeshfreeApplicator for boundary detection."""
        mask = self._applicator.get_boundary_mask(self.collocation_points)
        self.boundary_indices = np.where(mask)[0]

    def _compute_boundary_normals(self):
        """Use existing infrastructure for normal computation."""
        boundary_points = self.collocation_points[self.boundary_indices]
        self.normals = self._applicator.get_boundary_normals(boundary_points)

    def get_bc_type(self, point_idx: int) -> str:
        """
        Get BC type at a point (single source of truth).

        Uses BoundaryConditions.get_bc_at_point() for mixed BC support.
        """
        point = self.collocation_points[point_idx]

        if self.boundary_conditions.is_uniform:
            return self.boundary_conditions.bc_type.value

        # For mixed BCs, use segment-based lookup
        segment = self.boundary_conditions.get_bc_at_point(point)
        return segment.bc_type.value

    def apply_to_matrix(self, A, b, time_idx: int = 0):
        """
        Row Replacement pattern using GFDM derivative weights.

        Uses existing Calculator->LinearConstraint conversion for consistency.
        """
        for local_idx, global_idx in enumerate(self.boundary_indices):
            bc_type = self.get_bc_type(global_idx)
            normal = self.normals[local_idx]

            # Clear the row
            A[global_idx, :] = 0.0

            if bc_type in ("neumann", "no_flux"):
                # Get GFDM weights for normal derivative
                weights = self.gfdm_operator.get_derivative_weights(global_idx)
                neighbor_indices = weights['neighbor_indices']
                grad_weights = weights['grad_weights']  # Shape: (dim, k)

                # Normal derivative: ∂u/∂n = n · ∇u
                center_weight = 0.0
                for k, j in enumerate(neighbor_indices):
                    if j >= 0 and j != global_idx:
                        weight = sum(normal[d] * grad_weights[d, k]
                                    for d in range(self.dimension))
                        A[global_idx, j] = weight
                        center_weight -= weight

                A[global_idx, global_idx] = center_weight
                b[global_idx] = 0.0  # Neumann: ∂u/∂n = 0

            elif bc_type == "dirichlet":
                A[global_idx, global_idx] = 1.0
                # Get BC value from segment
                point = self.collocation_points[global_idx]
                segment = self.boundary_conditions.get_bc_at_point(point)
                b[global_idx] = segment.get_value(point, time_idx)

    def apply_to_residual(self, residual: np.ndarray, u: np.ndarray, time_idx: int = 0):
        """
        Compute BC residual (MUST match apply_to_matrix for Newton consistency).
        """
        for local_idx, global_idx in enumerate(self.boundary_indices):
            bc_type = self.get_bc_type(global_idx)

            if bc_type in ("neumann", "no_flux"):
                # Compute ∂u/∂n
                grad_u = self.gfdm_operator.gradient(u)[global_idx]
                normal = self.normals[local_idx]
                residual[global_idx] = np.dot(grad_u, normal)

            elif bc_type == "dirichlet":
                point = self.collocation_points[global_idx]
                segment = self.boundary_conditions.get_bc_at_point(point)
                target_value = segment.get_value(point, time_idx)
                residual[global_idx] = u[global_idx] - target_value
```

### 11.8 Key Takeaways

1. **Don't reinvent the wheel**: The codebase has mature BC infrastructure
2. **Use `BoundaryConditions.get_outward_normal()`** instead of custom implementation
3. **Use `MeshfreeApplicator`** for boundary detection and normal computation
4. **Follow the Topology/Calculator pattern** for consistency with FDM applicators
5. **Single source of truth**: `BoundaryConditionManager` should be the only place BC decisions are made

---

## 12. Strategy Pattern Refactoring for GFDMOperator

### 12.1 The Problem: GFDMOperator is a "God Object"

The current `GFDMOperator` class in `gfdm_operators.py` handles too many responsibilities:
- **Geometry**: KDTree neighborhood search
- **Physics**: Taylor polynomial weights computation
- **Boundary Architecture**: Ghost particle creation and handling

This violates the Single Responsibility Principle and makes it difficult to support different methods (`direct`, `upwind`, `rbf`) or boundary handling strategies.

### 12.2 The Solution: Strategy Pattern

Split `GFDMOperator` into three distinct concepts:

| Component | Responsibility | Examples |
|-----------|----------------|----------|
| **DifferentialOperator** | Geometric derivative weights (L) | `TaylorOperator`, `UpwindOperator`, `RBFOperator` |
| **BoundaryHandler** | Matrix enforcement (L → A) | `RowReplacementHandler` (New), `GhostNodeHandler` (Legacy) |
| **Solver** | Time-stepping & Orchestration | `HJBGFDMSolver` |

### 12.3 Abstract Interfaces

```python
from abc import ABC, abstractmethod
import numpy as np

class DifferentialOperator(ABC):
    """Abstract Strategy for computing spatial derivatives."""

    @abstractmethod
    def build_system(self, u: np.ndarray) -> dict:
        """
        Returns the geometric data structures.
        Must return: {'laplacian': matrix, 'gradient_x': matrix, ...}
        """
        pass

class BoundaryHandler(ABC):
    """Abstract Strategy for enforcing BCs."""

    @abstractmethod
    def apply(self, system_matrix, rhs, boundary_indices, operator):
        """Modifies A and b in-place."""
        pass
```

### 12.4 Concrete Implementations

#### TaylorOperator (Direct Collocation - Current GFDM)

```python
class TaylorOperator(DifferentialOperator):
    """
    Standard GFDM with Central/One-Sided differences.
    (Refactored from GFDMOperator, WITHOUT ghost particles)
    """
    def __init__(self, points, delta, taylor_order=2):
        self.points = points
        self.delta = delta
        self._build_neighborhoods()  # Standard radius search (NO ghosts)
        self._build_taylor_matrices()

    def build_system(self, u):
        return {
            'laplacian': self.get_laplacian_matrix(),
            'gradient': self.get_gradient_weights(),
        }
```

#### UpwindOperator (Critical for MFG)

```python
class UpwindOperator(TaylorOperator):
    """
    GFDM with Flow-Biased Stencils.
    Use this when method='upwind'.
    """
    def __init__(self, points, delta, velocity_field, **kwargs):
        self.velocity_field = velocity_field
        super().__init__(points, delta, **kwargs)

    def _build_neighborhoods(self):
        """Override: Filters neighbors to ensure they are 'upstream'."""
        tree = cKDTree(self.points)
        for i in range(self.n_points):
            indices = tree.query_ball_point(self.points[i], self.delta)

            # Filter based on velocity v_i
            v = self.velocity_field[i]
            valid_neighbors = []
            for j in indices:
                dx = self.points[j] - self.points[i]
                # Keep if neighbor is "behind" the flow (dot product < 0)
                if np.dot(dx, v) <= 1e-5:
                    valid_neighbors.append(j)

            self.neighborhoods.append({'indices': np.array(valid_neighbors)})
```

> **Expert Note**: This is a **strict upwind** scheme (only upstream neighbors). For
> smoother solutions, consider a "biased central" variant that includes some downwind
> neighbors with lower weights. However, strict upwinding is the correct starting point
> to fix oscillations.

### 12.5 DirectCollocationHandler (Row Replacement)

```python
class DirectCollocationHandler(BoundaryHandler):
    """
    The 'Row Replacement' method.
    Robust, no ghost nodes, works for Neumann/Dirichlet.
    """
    def apply(self, A, rhs, boundary_indices, operator, bc_config):
        for idx in boundary_indices:
            bc = bc_config.get(idx, default='neumann')

            # Clear Row (Wipe the PDE)
            A[idx, :] = 0.0

            if bc['type'] == 'dirichlet':
                A[idx, idx] = 1.0
                rhs[idx] = bc['value']

            elif bc['type'] == 'neumann':
                # Enforce dU/dn = 0
                w = operator.get_derivative_weights(idx)
                normal = bc['normal']

                # Project gradients: n_x * D_x + n_y * D_y
                flux_row = normal[0] * w['grad_x'] + normal[1] * w['grad_y']

                A[idx, w['indices']] = flux_row
                rhs[idx] = bc['value']  # Usually 0 for no-flux
```

### 12.6 Factory Pattern for Method Selection

```python
def create_operator(config, points):
    if config.method == 'direct':
        return TaylorOperator(points, delta=config.delta)
    elif config.method == 'upwind':
        return UpwindOperator(points, delta=config.delta, velocity_field=config.velocity)
    elif config.method == 'rbf':
        return RBFOperator(points, rbf_type='phs3')
    else:
        raise ValueError(f"Unknown method: {config.method}")

def create_bc_handler(config):
    if config.bc_method == 'direct':
        return DirectCollocationHandler()
    elif config.bc_method == 'ghost':
        return GhostNodeHandler()  # Legacy, wrapped in a class
    else:
        raise ValueError(f"Unknown BC method: {config.bc_method}")
```

### 12.7 DifferentialOperator: Scope and Limitations

The `DifferentialOperator` class is **generic for all Collocation Methods** but **not universal** for all numerical methods:

| Method Category | Examples | Uses DifferentialOperator? |
|-----------------|----------|---------------------------|
| **Strong Form (Collocation)** | FDM, GFDM, RBF-FD, Spectral | ✅ YES |
| **Weak Form (Galerkin)** | FEM, FVM | ❌ NO |

**Why this distinction matters:**
- **Collocation methods** approximate derivatives as weighted sums: `Lu(xᵢ) ≈ Σⱼ wⱼ u(xⱼ)`
- **Galerkin methods** require element stiffness matrices, flux reconstruction, quadrature rules

**Recommendation**:
- Keep the name `DifferentialOperator` (or use `CollocationOperator` for precision)
- Do not try to make this class handle Finite Elements
- If FEM is needed, create a separate `HJBFEMSolver` class

### 12.8 Implementation Roadmap

> **Expert Recommendation**: Refactoring (Strategy Pattern) should be done *simultaneously*
> with Row Replacement implementation. It is much harder to implement Row Replacement cleanly
> inside the old "God Object" class. Breaking the class apart first makes adding Row
> Replacement trivial.

**Immediate Actions (Do Together):**

1. **Modify `gfdm_operators.py`**: Remove `boundary_type="no_flux"` and `_create_ghost_particles` logic. Rename class to `TaylorOperator` (alias `GFDMOperator` for backward compatibility).

2. **Create `DirectCollocationHandler`**: Implement Row Replacement in a separate class, use `BoundaryConditionManager` from Section 11.7.

3. **Update Solver `__init__`**: Use factory pattern to instantiate `self.operator` and `self.bc_handler` based on config flags.

**Configuration Interface:**

```python
# In HJBGFDMSolver
class HJBGFDMSolver:
    def __init__(
        self,
        problem,
        method: str = "direct",      # "direct", "upwind", "rbf"
        bc_method: str = "direct",   # "direct" (Row Replacement), "ghost" (Legacy)
        ...
    ):
        self.operator = create_operator(config, collocation_points)
        self.bc_handler = create_bc_handler(config)
```

---

## 13. RBF-FD: Recommended Alternative to Taylor-GFDM

### 13.1 Current Implementation Gap

The codebase currently has:
- **GFDM** (`GFDMOperator`): Taylor polynomial + weighted least squares
- **RBF** (`RBFOperator`): Global RBF interpolation

But **NOT** the modern hybrid approach:
- **RBF-FD**: Local stencils + RBF interpolation

### 13.2 Why RBF-FD is Superior

| Aspect | Taylor-GFDM | RBF-FD |
|--------|-------------|--------|
| **Basis** | Polynomial: 1, x, y, x², ... | RBF: φ(‖x - xⱼ‖) |
| **Fitting** | Weighted least squares | Exact interpolation |
| **Conditioning** | Singular for collinear nodes | Robust to node arrangement |
| **Parameter Tuning** | Weight function selection | PHS is parameter-free |
| **Accuracy** | O(h^p) for order-p polynomial | Often higher for same stencil |

### 13.3 RBF-FD Mathematical Formulation

For local stencil around point xᵢ with k neighbors {x₁, ..., xₖ}:

**Step 1: Build local interpolation matrix**
```
A = [φ(‖x₁ - x₁‖)  φ(‖x₁ - x₂‖)  ...  φ(‖x₁ - xₖ‖)]
    [φ(‖x₂ - x₁‖)  φ(‖x₂ - x₂‖)  ...  φ(‖x₂ - xₖ‖)]
    [     ...            ...      ...       ...      ]
    [φ(‖xₖ - x₁‖)  φ(‖xₖ - x₂‖)  ...  φ(‖xₖ - xₖ‖)]
```

**Step 2: Differentiation weights**
For operator L (e.g., Laplacian):
```
b = [L φ(‖xᵢ - x₁‖)]
    [L φ(‖xᵢ - x₂‖)]
    [      ...      ]
    [L φ(‖xᵢ - xₖ‖)]

w = A⁻¹ b   (differentiation weights)
```

**Step 3: Apply operator**
```
L u(xᵢ) ≈ Σⱼ wⱼ u(xⱼ)
```

### 13.4 PHS + Polynomial Augmentation

**Polyharmonic Splines (PHS)**: φ(r) = r^m (odd m) or r^m log(r) (even m)

- **PHS-3**: φ(r) = r³ (most common)
- **PHS-5**: φ(r) = r⁵ (higher accuracy)

**Polynomial augmentation** ensures polynomial reproduction:
```
[A  P][λ]   [u]
[Pᵀ 0][c] = [0]

where P = polynomial basis evaluated at stencil points
```

This is **nearly parameter-free** (no shape parameter tuning).

### 13.5 Recommended Implementation

```python
class RBFFDOperator:
    """RBF-FD differential operator with local stencils."""

    def __init__(
        self,
        points: np.ndarray,
        k_neighbors: int = 15,
        rbf_type: str = "phs3",  # Polyharmonic spline r³
        poly_degree: int = 2,     # Polynomial augmentation degree
        boundary_indices: set[int] | None = None,
    ):
        self.points = points
        self.k_neighbors = k_neighbors
        self.rbf_type = rbf_type
        self.poly_degree = poly_degree
        self.boundary_indices = boundary_indices or set()

        self._build_neighborhoods()
        self._build_differentiation_matrices()

    def _compute_rbf_fd_weights(self, center_idx: int) -> dict:
        """Compute RBF-FD weights for derivatives at a single point."""
        neighbors = self.neighborhoods[center_idx]
        n = len(neighbors)

        # Build local RBF matrix
        A = self._build_rbf_matrix(neighbors)

        # Build polynomial augmentation matrix
        P = self._build_polynomial_matrix(neighbors)

        # Augmented system
        m = P.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = P
        M[n:, :n] = P.T

        # RHS for gradient and Laplacian
        center = self.points[center_idx]

        # Gradient weights: d/dx_d φ(‖x - xⱼ‖) evaluated at center
        grad_rhs = self._rbf_gradient_rhs(center, neighbors)

        # Laplacian weights: Δ φ(‖x - xⱼ‖) evaluated at center
        lap_rhs = self._rbf_laplacian_rhs(center, neighbors)

        # Solve for weights
        grad_weights = np.linalg.solve(M, np.vstack([grad_rhs, np.zeros((m, grad_rhs.shape[1]))]))[:n]
        lap_weights = np.linalg.solve(M, np.concatenate([lap_rhs, np.zeros(m)]))[:n]

        return {
            "neighbor_indices": neighbors,
            "grad_weights": grad_weights,
            "lap_weights": lap_weights,
        }
```

### 13.6 Boundary Handling with RBF-FD

RBF-FD handles boundaries more naturally:

1. **One-sided stencils**: Work exactly as with GFDM
2. **Row Replacement**: Same pattern applies
3. **No ghost particles needed**: RBF interpolation doesn't require symmetry

The key advantage: RBF-FD with PHS is **robust to stencil geometry**, so one-sided boundary stencils don't cause conditioning issues.

### 13.7 Implementation Priority

**Recommendation**: Add RBF-FD as an alternative to Taylor-GFDM

```python
class HJBGFDMSolver:
    def __init__(
        self,
        problem,
        derivative_method: str = "taylor",  # "taylor" or "rbf_fd"
        ...
    ):
        if derivative_method == "taylor":
            self._operator = GFDMOperator(...)  # Current implementation
        elif derivative_method == "rbf_fd":
            self._operator = RBFFDOperator(...)  # New implementation
```

This provides:
1. Backward compatibility (default = Taylor)
2. Better option for irregular meshes (RBF-FD)
3. Parameter-free operation with PHS

---

## 14. GFDM Variations: Complete Family Overview

### 14.1 The GFDM Family

GFDM is a broad family of methods. The variations address specific weaknesses:

| Method | Basis Function | Stability (Advection) | BC Handling | Best For |
|--------|----------------|----------------------|-------------|----------|
| **Standard GFDM** | Taylor Polynomial | Low (Oscillatory) | Row Replacement | Diffusion / Poisson |
| **RBF-FD** | Gaussian / PHS | Medium | Row Replacement | Irregular Geometries |
| **Upwind GFDM** | Taylor (Biased) | **High** | Row Replacement | **MFG / HJB / Fluid** |
| **Hermite GFDM** | Hermite Poly | Medium | **Trivial** | High Accuracy Req. |
| **Explicit GFDM** | Any | Depends on CFL | Row Replacement | GPU / Pseudo-time |

### 14.2 Upwind GFDM (Critical for MFG)

**Why it matters**: Standard GFDM uses "central" weights (minimizing error equally in all directions). For the Hamiltonian term in HJB or the drift term in FP, this causes **non-physical oscillations**.

**The Problem**:
- HJB: `∂u/∂t + H(∇u) = 0` has advection character
- FP: `∂m/∂t + ∇·(v·m) = σΔm` has drift term
- Central differences → Gibbs phenomenon, negative densities

**Two Approaches**:

1. **Stencil Bias**: Select neighbors only in "upwind" direction relative to flow
```python
def get_upwind_neighbors(point_idx, velocity):
    """Select neighbors upstream of velocity field."""
    all_neighbors = self.get_neighbors(point_idx)
    upwind_mask = np.dot(neighbors - center, -velocity) > 0
    return all_neighbors[upwind_mask]
```

2. **Weight Bias**: Use centered stencil but penalize downwind error more heavily
```python
def compute_upwind_weights(distances, velocity_direction):
    """Add upwinding to least squares weights."""
    base_weights = wendland_kernel(distances)
    upwind_factor = 1 + alpha * np.dot(directions, -velocity_direction)
    return base_weights * np.maximum(upwind_factor, 0)
```

**Relevance**: Without upwinding:
- Density `m` may become negative
- Value function `u` develops spurious oscillations near steep gradients

### 14.3 Hermite GFDM (H-GFDM)

**The Idea**: Include derivatives as unknowns: `[u, u_x, u_y]` at each node.

**BC Handling Advantage**:
- Neumann BC becomes trivial: just set `u_n = 0` directly
- No Row Replacement needed - derivatives are explicit variables

**Tradeoff**: System size triples (`3N × 3N` instead of `N × N`)

**Mathematical Formulation**:
```
For 2D, unknowns at each node: U_i = [u_i, (u_x)_i, (u_y)_i]

Consistency equations:
  u_j = u_i + (u_x)_i·Δx + (u_y)_i·Δy + O(h²)
  (u_x)_j = (u_x)_i + (u_xx)_i·Δx + (u_xy)_i·Δy + O(h)
  (u_y)_j = (u_y)_i + (u_xy)_i·Δx + (u_yy)_i·Δy + O(h)
```

**When to use**: High accuracy requirements, complex BCs, not iterative MFG

### 14.4 Explicit GFDM (E-GFDM)

**Time-stepping strategy**, not spatial discretization:
```
u^{n+1} = u^n + Δt · L(u^n)
```

**Advantages**:
- No global matrix solve needed
- Trivially parallelizable (GPU-friendly)
- Works with pseudo-time stepping for steady-state

**Constraint**: CFL condition limits timestep

**Relevance for MFG**:
- Many MFG algorithms use pseudo-time stepping
- JAX/PyTorch implementations benefit from explicit methods

### 14.5 Current Implementation Status

| Variation | Implemented? | Location |
|-----------|--------------|----------|
| Standard GFDM | ✅ Yes | `GFDMOperator` |
| RBF-FD | ❌ No | Recommended addition |
| Upwind GFDM | ⚠️ Partial | `hjb_fdm.py` has upwind, not `hjb_gfdm.py` |
| Hermite GFDM | ❌ No | Low priority for MFG |
| Explicit GFDM | ✅ Yes | Time-stepping in solvers |

### 14.6 Recommendations for MFG_PDE

**Priority Order**:

1. **Fix BC handling** (this report) - Row Replacement for Neumann ✅ In Progress
2. **Add Upwind GFDM** - Critical for HJB stability
3. **Add RBF-FD** - Better conditioning for irregular meshes
4. **Skip Hermite** - Too expensive for iterative MFG

**Implementation Strategy**:
```python
class HJBGFDMSolver:
    def __init__(
        self,
        problem,
        derivative_method: str = "taylor",  # "taylor", "rbf_fd"
        advection_scheme: str = "centered",  # "centered", "upwind"
        ...
    ):
        # Select operator
        if derivative_method == "taylor":
            self._operator = GFDMOperator(...)
        elif derivative_method == "rbf_fd":
            self._operator = RBFFDOperator(...)

        # Configure upwinding
        self._use_upwind = (advection_scheme == "upwind")
```

---

## 15. References

### Original GFDM
1. Liszka, T., & Orkisz, J. (1980). "The finite difference method at arbitrary irregular grids." *Computers & Structures*.

### RBF-FD
2. Wright, G.B., & Fornberg, B. (2006). "Scattered node compact finite difference-type formulas from RBFs." *J. Comp. Physics*.

### Upwind GFDM
3. Gavete, L., et al. (2017). "Upwind scheme for GFDM to solve advection-diffusion." *Int. J. Numer. Methods Fluids*.

### Hermite GFDM
4. Fan, C.M., et al. (2020). "Hermite GFDM for solving PDEs." *Eng. Analysis with Boundary Elements*.

### Boundary Handling
5. Benito, J.J., et al. (2001). "Influence of several factors in GFDM." *Applied Mathematical Modelling*.
6. Atluri, S.N., & Zhu, T. (1998). "MLPG approach in computational mechanics." *Computational Mechanics*.

---

**Appendix: Key Insight**

The fundamental error was treating ghost particles as a "trick" to enforce BC. In reality:

- **Ghost particles** = Artificial data that corrupts the operator
- **One-sided stencils** = Natural GFDM geometry (no artificial data)
- **Row Replacement** = Standard technique for constraint enforcement

The correct architecture is:
```
[Pure Derivative Operator] + [Explicit BC Equations] = [Correct Solution]
```

Not:
```
[BC-Contaminated Operator] = [Wrong Solution]
```
