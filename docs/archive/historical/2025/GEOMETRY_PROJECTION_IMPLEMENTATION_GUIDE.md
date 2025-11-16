# Geometry Projection Implementation Guide

**Status**: ✅ COMPLETED (Issue #257 Phases 1-3)
**Created**: 2025-11-10
**Last Updated**: 2025-11-10

## Overview

This guide covers the implementation details of the geometry projection system for MFG_PDE developers who need to:
- Understand the codebase structure
- Add new projection methods
- Extend to new geometry types
- Debug projection issues
- Integrate with solvers

## Codebase Structure

### Core Files

#### `mfg_pde/geometry/projection.py` (500+ lines)
Main implementation file containing:

```
projection.py
├── ProjectionRegistry (lines 41-88)
│   ├── _registry: ClassVar[dict]
│   ├── register(source_type, target_type, direction)
│   ├── get_projector(source, target, direction)
│   ├── list_registered()
│   └── clear_registry()
├── GeometryProjector (lines 90-386)
│   ├── __init__(hjb_geometry, fp_geometry, projection_method)
│   ├── _setup_projection_methods()
│   ├── project_hjb_to_fp(U_on_hjb_geometry)
│   ├── project_fp_to_hjb(M_on_fp_geometry)
│   ├── _interpolate_grid_to_points(values, points)
│   ├── _interpolate_grid_to_grid(values)
│   ├── _restrict_grid_to_grid(values)
│   ├── _project_particles_to_grid_kde(values, bandwidth)
│   ├── _nearest_neighbor_projection(values, points)
│   └── _histogram_density_nd(positions, values, shape)
└── Helper Functions
    ├── estimate_bandwidth(positions, method)
    └── validate_projection_shapes(source_shape, target_shape)
```

#### `mfg_pde/core/mfg_problem.py` (modified lines 89-120, 685-689)
Integration points:

```python
def __init__(self, ...):
    # Lines 89-120: Dual geometry parameter handling
    if hjb_geometry is not None and fp_geometry is not None:
        # Dual geometry mode
        self.hjb_geometry = hjb_geometry
        self.fp_geometry = fp_geometry
        self.geometry_projector = GeometryProjector(...)
    elif geometry is not None:
        # Unified mode (backward compatible)
        self.hjb_geometry = geometry
        self.fp_geometry = geometry
        self.geometry_projector = None

    # Lines 685-689: Ensure dual geometry attributes set
    if not hasattr(self, "hjb_geometry"):
        self.hjb_geometry = getattr(self, "geometry", None)
        self.fp_geometry = getattr(self, "geometry", None)
```

#### `mfg_pde/geometry/__init__.py` (lines 98-99, 139-140)
Public API exports:

```python
from .projection import GeometryProjector, ProjectionRegistry

__all__ = [
    # ...
    "GeometryProjector",
    "ProjectionRegistry",
    # ...
]
```

### Modified Interpolators

#### `mfg_pde/geometry/simple_grid_1d.py` (lines 110-140)
```python
def get_interpolator(self) -> Callable:
    """Return vectorized linear interpolator."""
    x_grid = np.array(self._cached_grid)

    def interpolate_1d(u: NDArray, points: NDArray) -> NDArray | float:
        # Handle scalar, single point, or array of points
        if points.ndim == 0:
            return float(np.interp(points, x_grid, u))
        elif points.shape == (1,):
            return float(np.interp(points[0], x_grid, u))
        else:
            x = points[:, 0] if points.ndim == 2 else points
            return np.interp(x, x_grid, u)

    return interpolate_1d
```

#### `mfg_pde/geometry/simple_grid.py` (lines 85-110, 240-265)
```python
# SimpleGrid2D
def get_interpolator(self) -> Callable:
    """Return vectorized bilinear interpolator."""
    def interpolate_2d(u: NDArray, points: NDArray) -> NDArray | float:
        from scipy.interpolate import RegularGridInterpolator

        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)

        interpolator = RegularGridInterpolator(
            (x, y), u, method="linear", bounds_error=False, fill_value=0.0
        )

        result = interpolator(points)
        return float(result) if result.shape == () else result

    return interpolate_2d

# SimpleGrid3D: similar pattern with trilinear
```

### Test Suite

#### `tests/unit/geometry/test_geometry_projection.py` (440+ lines)
```
test_geometry_projection.py
├── TestGeometryProjectorBasics (5 tests)
│   ├── test_same_geometry_no_projection
│   ├── test_projection_method_detection_grid_to_particles
│   └── test_projection_method_detection_grid_to_grid
├── TestGrid1DProjections (3 tests)
│   ├── test_grid_to_grid_1d_interpolation
│   ├── test_1d_grid_shape_consistency
│   └── test_1d_particles_to_grid_kde
├── TestGrid2DProjections (3 tests)
│   ├── test_grid_to_grid_2d_interpolation
│   ├── test_2d_grid_shape_consistency
│   └── test_2d_bilinear_accuracy
├── TestGrid3DProjections (2 tests)
│   ├── test_grid_to_grid_3d_interpolation
│   └── test_3d_grid_shape_consistency
├── TestProjectionRegistry (7 tests)
│   ├── test_registry_registration
│   ├── test_registry_lookup_exact_match
│   ├── test_registry_lookup_category_match
│   ├── test_registry_fallback_to_auto
│   ├── test_registry_integration_with_projector
│   ├── test_registry_clear
│   └── test_multiple_registrations
└── TestMFGProblemIntegration (7 tests) [in test_mfg_problem.py]
    ├── test_dual_geometry_specification
    ├── test_dual_geometry_backward_compatibility
    ├── test_dual_geometry_error_on_partial_specification
    ├── test_dual_geometry_error_on_conflict
    ├── test_dual_geometry_projector_attributes
    ├── test_dual_geometry_different_resolutions
    └── test_dual_geometry_auto_detection
```

## Implementation Patterns

### Pattern 1: Adding a New Projection Method

**Scenario:** Add spectral projection for high-accuracy grid-to-grid.

**Step 1: Implement projection function**
```python
# In mfg_pde/geometry/projection.py

def _spectral_projection(self, values_on_source_grid: NDArray) -> NDArray:
    """
    Spectral projection using FFT for periodic grids.

    Args:
        values_on_source_grid: Values on source grid

    Returns:
        Values on target grid
    """
    from scipy.fft import fft, ifft

    # Transform to frequency domain
    coeffs = fft(values_on_source_grid, axis=-1)

    # Truncate/pad to target resolution
    target_shape = self.fp_geometry.get_grid_shape()
    coeffs_resized = self._resize_frequency_domain(coeffs, target_shape)

    # Transform back
    values_target = ifft(coeffs_resized, axis=-1).real

    return values_target
```

**Step 2: Add to auto-detection**
```python
def _setup_projection_methods(self) -> None:
    """Auto-detect appropriate projection methods."""
    # ... existing code ...

    # Check if both geometries are periodic grids
    hjb_is_periodic = (
        isinstance(self.hjb_geometry, CartesianGrid)
        and getattr(self.hjb_geometry, "periodic", False)
    )
    fp_is_periodic = (
        isinstance(self.fp_geometry, CartesianGrid)
        and getattr(self.fp_geometry, "periodic", False)
    )

    if hjb_is_periodic and fp_is_periodic:
        self.hjb_to_fp_method = "spectral"  # NEW
```

**Step 3: Add to projection dispatcher**
```python
def project_hjb_to_fp(self, U_on_hjb_geometry: NDArray, backend=None) -> NDArray:
    """Project HJB value function to FP geometry."""
    # ... existing code ...

    if self.hjb_to_fp_method == "spectral":
        return self._spectral_projection(U_on_hjb_geometry)
    # ... existing methods ...
```

**Step 4: Add tests**
```python
# In tests/unit/geometry/test_geometry_projection.py

def test_spectral_projection_accuracy():
    """Test spectral projection on smooth periodic function."""
    from mfg_pde.geometry import SimpleGrid1D, BoundaryConditions

    bc = BoundaryConditions(type="periodic")
    coarse = SimpleGrid1D(xmin=0.0, xmax=2*np.pi, boundary_conditions=bc)
    coarse.create_grid(num_points=32)

    fine = SimpleGrid1D(xmin=0.0, xmax=2*np.pi, boundary_conditions=bc)
    fine.create_grid(num_points=128)

    projector = GeometryProjector(hjb_geometry=coarse, fp_geometry=fine)

    # Smooth function: u(x) = sin(x) + 0.5*sin(2x)
    x_coarse = coarse.get_spatial_grid()
    u_coarse = np.sin(x_coarse) + 0.5 * np.sin(2 * x_coarse)

    u_fine = projector.project_hjb_to_fp(u_coarse)

    # Spectral should be exact for low-frequency content
    x_fine = fine.get_spatial_grid()
    expected = np.sin(x_fine) + 0.5 * np.sin(2 * x_fine)

    np.testing.assert_allclose(u_fine, expected, atol=1e-10)
```

### Pattern 2: Registering Custom Geometry Projection

**Scenario:** Add custom projection for your new geometry type.

**Example: Unstructured Mesh → Grid**

```python
from mfg_pde.geometry import ProjectionRegistry
from scipy.interpolate import LinearNDInterpolator

@ProjectionRegistry.register(UnstructuredMesh, SimpleGrid2D, "hjb_to_fp")
def mesh_to_grid_projection(mesh_geo, grid_geo, mesh_values, **kwargs):
    """
    Project from unstructured mesh to regular grid using Delaunay interpolation.

    Args:
        mesh_geo: Source unstructured mesh
        grid_geo: Target regular grid
        mesh_values: Values at mesh vertices (N_vertices,)
        **kwargs: Additional arguments (unused)

    Returns:
        Values on grid (nx+1, ny+1)
    """
    # Get mesh vertex positions
    vertices = mesh_geo.get_vertices()  # (N_vertices, 2)

    # Create Delaunay interpolator
    interpolator = LinearNDInterpolator(vertices, mesh_values)

    # Evaluate at grid points
    grid_points = grid_geo.get_spatial_grid()  # (N_grid, 2)
    grid_values_flat = interpolator(grid_points)

    # Handle extrapolation (fill NaN with nearest)
    nan_mask = np.isnan(grid_values_flat)
    if np.any(nan_mask):
        from scipy.spatial import KDTree
        tree = KDTree(vertices)
        _, nearest_indices = tree.query(grid_points[nan_mask])
        grid_values_flat[nan_mask] = mesh_values[nearest_indices]

    # Reshape to grid
    grid_shape = grid_geo.get_grid_shape()
    return grid_values_flat.reshape(grid_shape)


@ProjectionRegistry.register(SimpleGrid2D, UnstructuredMesh, "fp_to_hjb")
def grid_to_mesh_projection(grid_geo, mesh_geo, grid_values, **kwargs):
    """
    Project from regular grid to unstructured mesh using bilinear interpolation.

    Args:
        grid_geo: Source regular grid
        mesh_geo: Target unstructured mesh
        grid_values: Values on grid (nx+1, ny+1)
        **kwargs: Additional arguments (unused)

    Returns:
        Values at mesh vertices (N_vertices,)
    """
    # Get mesh vertex positions
    vertices = mesh_geo.get_vertices()  # (N_vertices, 2)

    # Use grid's built-in interpolator
    interpolator = grid_geo.get_interpolator()
    mesh_values = interpolator(grid_values, vertices)

    return mesh_values
```

**Usage:**
```python
from mfg_pde import MFGProblem

mesh = UnstructuredMesh(...)  # Your mesh geometry
grid = SimpleGrid2D(...)      # Regular grid

problem = MFGProblem(
    hjb_geometry=mesh,   # HJB on unstructured mesh
    fp_geometry=grid,    # FP on regular grid
    time_domain=(1.0, 50),
    sigma=0.1
)

# Projector automatically uses registered functions
projector = problem.geometry_projector
assert projector.hjb_to_fp_method == "registry"
assert projector.fp_to_hjb_method == "registry"
```

### Pattern 3: Hierarchical Category Registration

**Scenario:** Register projection for all CartesianGrid types (SimpleGrid1D/2D/3D).

```python
from mfg_pde.geometry.base_geometry import CartesianGrid

@ProjectionRegistry.register(CartesianGrid, CartesianGrid, "hjb_to_fp")
def conservative_grid_projection(source_grid, target_grid, values, **kwargs):
    """
    Conservative projection for any CartesianGrid type.
    Uses category match (isinstance check).
    """
    dim = source_grid.get_dimension()

    if dim == 1:
        return conservative_project_1d(source_grid, target_grid, values)
    elif dim == 2:
        return conservative_project_2d(source_grid, target_grid, values)
    elif dim == 3:
        return conservative_project_3d(source_grid, target_grid, values)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


# This works for SimpleGrid1D, SimpleGrid2D, SimpleGrid3D, TensorProductGrid, etc.
```

### Pattern 4: Integrating with Solver

**Scenario:** Use dual geometries in your solver.

```python
def solve_mfg_with_dual_geometries(problem: MFGProblem, max_iterations: int = 100):
    """
    Fixed point iteration with dual geometries.

    Args:
        problem: MFGProblem with hjb_geometry and fp_geometry
        max_iterations: Maximum number of iterations

    Returns:
        (U_hjb, M_fp): Solution on respective geometries
    """
    from mfg_pde.solvers import create_hjb_solver, create_fp_solver

    # Initialize on FP geometry
    M_fp = problem.m0  # Assumed on fp_geometry

    # Create solvers
    hjb_solver = create_hjb_solver(problem, geometry=problem.hjb_geometry)
    fp_solver = create_fp_solver(problem, geometry=problem.fp_geometry)

    for iteration in range(max_iterations):
        # 1. Project density: FP → HJB
        if problem.geometry_projector is not None:
            M_hjb = problem.geometry_projector.project_fp_to_hjb(M_fp, bandwidth="scott")
        else:
            M_hjb = M_fp  # Unified geometry

        # 2. Solve HJB on HJB geometry
        U_hjb = hjb_solver.solve(M_hjb)

        # 3. Project value: HJB → FP
        if problem.geometry_projector is not None:
            U_fp = problem.geometry_projector.project_hjb_to_fp(U_hjb)
        else:
            U_fp = U_hjb  # Unified geometry

        # 4. Solve FP on FP geometry
        M_fp_new = fp_solver.solve(U_fp)

        # 5. Check convergence
        error_M = np.linalg.norm(M_fp_new - M_fp) / np.linalg.norm(M_fp)
        if error_M < 1e-6:
            print(f"Converged in {iteration+1} iterations")
            return U_hjb, M_fp_new

        M_fp = M_fp_new

    print("Warning: Maximum iterations reached")
    return U_hjb, M_fp
```

## Common Issues and Solutions

### Issue 1: Shape Mismatch After Projection

**Problem:**
```python
U_projected = projector.project_hjb_to_fp(U_grid)
# Expected: (21, 21), Got: (441,)
```

**Cause:** Interpolators return flattened arrays by default.

**Solution:** Use `_interpolate_grid_to_grid()` which handles reshaping:
```python
def _interpolate_grid_to_grid(self, values_on_source_grid: NDArray) -> NDArray:
    target_points = self.fp_geometry.get_spatial_grid()
    values_flat = self._interpolate_grid_to_points(values_on_source_grid, target_points)

    # Reshape if target is Cartesian grid
    if isinstance(self.fp_geometry, CartesianGrid):
        target_shape = self.fp_geometry.get_grid_shape()
        return values_flat.reshape(target_shape)

    return values_flat
```

### Issue 2: Mass Not Conserved in KDE

**Problem:**
```python
total_mass_particles = np.sum(particle_masses)
grid_density = projector.project_fp_to_hjb(particle_masses, bandwidth="scott")
total_mass_grid = np.sum(grid_density) * cell_volume
# total_mass_grid ≠ total_mass_particles
```

**Cause:** KDE normalization doesn't account for cell volumes.

**Solution:** Normalize explicitly:
```python
def _project_particles_to_grid_kde(self, particle_values, bandwidth, backend):
    # ... KDE computation ...

    # Normalize to conserve mass
    cell_volume = np.prod(self.hjb_geometry.get_cell_sizes())
    total_mass_particles = np.sum(particle_values)
    total_mass_grid_unnormalized = np.sum(density) * cell_volume

    if total_mass_grid_unnormalized > 1e-12:
        density *= total_mass_particles / total_mass_grid_unnormalized

    return density.reshape(grid_shape)
```

### Issue 3: Interpolation Outside Domain Returns 0

**Problem:**
```python
# Particle slightly outside domain [0,1]²
particle_pos = np.array([[1.05, 0.5]])
u_particle = projector.project_hjb_to_fp(U_grid)
# u_particle = 0.0 (fill_value)
```

**Cause:** `RegularGridInterpolator` uses `fill_value=0.0` for out-of-bounds.

**Solution:** Clamp particles to domain or use extrapolation:
```python
# Option 1: Clamp to domain
particle_pos_clamped = np.clip(particle_pos, [xmin, ymin], [xmax, ymax])

# Option 2: Enable extrapolation (nearest value)
interpolator = RegularGridInterpolator(
    (x, y), u, method="linear",
    bounds_error=False,
    fill_value=None  # Use nearest extrapolation
)
```

### Issue 4: KDE Too Slow for 2D/3D

**Problem:**
```python
# 10,000 particles → 100×100 grid takes 10 seconds
density = projector.project_fp_to_hjb(particle_values)
```

**Cause:** Naive O(N*M) KDE computation.

**Solution 1:** Use histogram fallback (current implementation):
```python
def _histogram_density_nd(self, positions, values, grid_shape):
    """Fast histogram-based density (piecewise constant)."""
    # ... histogram computation ...
    return density
```

**Solution 2:** Implement fast summation (future):
```python
def _fast_multipole_kde(self, positions, values, grid_points, bandwidth):
    """O((N+M)log(N+M)) using tree codes."""
    from sklearn.neighbors import KernelDensity
    # Use tree-based acceleration
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', algorithm='ball_tree')
    kde.fit(positions, sample_weight=values)
    log_density = kde.score_samples(grid_points)
    return np.exp(log_density)
```

### Issue 5: Registry Not Found

**Problem:**
```python
@ProjectionRegistry.register(CustomGeometry, SimpleGrid2D, "hjb_to_fp")
def custom_proj(...): ...

projector = GeometryProjector(hjb_geometry=custom_geo, fp_geometry=grid)
# Still uses fallback, not custom_proj
```

**Cause:** Registration happens AFTER GeometryProjector initialization.

**Solution:** Register BEFORE creating projector:
```python
# Register first
@ProjectionRegistry.register(CustomGeometry, SimpleGrid2D, "hjb_to_fp")
def custom_proj(...): ...

# Then create projector
projector = GeometryProjector(hjb_geometry=custom_geo, fp_geometry=grid)
```

Or use manual registration:
```python
def custom_proj(...): ...

# Register manually
ProjectionRegistry.register(CustomGeometry, SimpleGrid2D, "hjb_to_fp")(custom_proj)

# Then create projector
projector = GeometryProjector(hjb_geometry=custom_geo, fp_geometry=grid)
```

## Debugging Tips

### Debugging Projection Selection

```python
projector = GeometryProjector(hjb_geometry=..., fp_geometry=...)

print("HJB → FP method:", projector.hjb_to_fp_method)
print("FP → HJB method:", projector.fp_to_hjb_method)

# Check if using registry
if projector.hjb_to_fp_method == "registry":
    print("Using registered projector:", projector._hjb_to_fp_func)

# List all registered projectors
print("Registered projectors:")
for key in ProjectionRegistry.list_registered():
    print(f"  {key}")
```

### Debugging Shape Mismatches

```python
U_hjb = ...  # Shape (101, 101)

print("HJB geometry shape:", problem.hjb_geometry.get_grid_shape())
print("FP geometry shape:", problem.fp_geometry.get_grid_shape())
print("U_hjb shape:", U_hjb.shape)

U_fp = projector.project_hjb_to_fp(U_hjb)
print("U_fp shape:", U_fp.shape)
print("Expected shape:", problem.fp_geometry.get_grid_shape())
```

### Debugging KDE Bandwidth

```python
from mfg_pde.geometry.projection import estimate_bandwidth

particles = fp_geometry.get_spatial_grid()

h_scott = estimate_bandwidth(particles, method="scott")
h_silverman = estimate_bandwidth(particles, method="silverman")

print(f"Scott's bandwidth: {h_scott}")
print(f"Silverman's bandwidth: {h_silverman}")

# Try different bandwidths
for h in [0.01, 0.05, 0.1, 0.5]:
    density = projector.project_fp_to_hjb(particle_values, bandwidth=h)
    print(f"h={h}: min={density.min():.3e}, max={density.max():.3e}, sum={density.sum():.3e}")
```

## Performance Optimization

### Optimization 1: Cache Spatial Grids

```python
class GeometryProjector:
    def __init__(self, hjb_geometry, fp_geometry, projection_method="auto"):
        # ... existing code ...

        # Cache spatial grids (computed once)
        self._hjb_points = None
        self._fp_points = None

    def project_hjb_to_fp(self, U_on_hjb_geometry, backend=None):
        # Compute FP points only once
        if self._fp_points is None:
            self._fp_points = self.fp_geometry.get_spatial_grid()

        # Use cached points
        return self._interpolate_grid_to_points(U_on_hjb_geometry, self._fp_points)
```

### Optimization 2: Reuse KD-Trees

```python
class GeometryProjector:
    def __init__(self, ...):
        # ... existing code ...
        self._kdtree_cache = {}

    def _get_kdtree(self, geometry_key):
        if geometry_key not in self._kdtree_cache:
            points = self.hjb_geometry.get_spatial_grid()
            self._kdtree_cache[geometry_key] = KDTree(points)
        return self._kdtree_cache[geometry_key]
```

### Optimization 3: Parallel KDE

```python
def _project_particles_to_grid_kde_parallel(self, particle_values, bandwidth, backend):
    """Parallel KDE using joblib."""
    from joblib import Parallel, delayed

    particle_positions = self.fp_geometry.get_spatial_grid()
    grid_points = self.hjb_geometry.get_spatial_grid()

    # Split grid into chunks
    n_jobs = 4
    chunks = np.array_split(grid_points, n_jobs)

    def compute_chunk(chunk):
        distances = cdist(particle_positions, chunk)
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        return particle_values @ weights

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(delayed(compute_chunk)(chunk) for chunk in chunks)

    density_flat = np.concatenate(results)
    grid_shape = self.hjb_geometry.get_grid_shape()
    return density_flat.reshape(grid_shape)
```

## Testing Patterns

### Test Pattern 1: Accuracy Test

```python
def test_projection_accuracy():
    """Test projection accuracy with known function."""
    # Use function that matches interpolation order
    # Linear interpolation → linear function is exact
    # Quadratic interpolation → quadratic function is exact

    U_coarse = f(x_coarse)  # Known function
    U_fine = projector.project_hjb_to_fp(U_coarse)
    expected = f(x_fine)

    np.testing.assert_allclose(U_fine, expected, rtol=1e-10)
```

### Test Pattern 2: Conservation Test

```python
def test_mass_conservation():
    """Test that projection conserves total mass."""
    particle_values = np.random.rand(N_particles)
    total_mass_particles = np.sum(particle_values)

    grid_density = projector.project_fp_to_hjb(particle_values)
    cell_volume = np.prod(grid.get_cell_sizes())
    total_mass_grid = np.sum(grid_density) * cell_volume

    rel_error = abs(total_mass_grid - total_mass_particles) / total_mass_particles
    assert rel_error < 0.05  # Within 5%
```

### Test Pattern 3: Consistency Test

```python
def test_round_trip_consistency():
    """Test that grid → particles → grid is approximately identity."""
    U_grid_original = np.random.rand(*grid.get_grid_shape())

    # Grid → Particles
    U_particles = projector.project_hjb_to_fp(U_grid_original)

    # Particles → Grid (via KDE)
    U_grid_reconstructed = projector.project_fp_to_hjb(U_particles)

    # Should be similar (not exact due to KDE smoothing)
    rel_error = np.linalg.norm(U_grid_reconstructed - U_grid_original) / np.linalg.norm(U_grid_original)
    assert rel_error < 0.1  # Within 10%
```

### Test Pattern 4: Shape Test

```python
def test_projection_shapes():
    """Test that projection returns correct shapes."""
    U_hjb = np.random.rand(*problem.hjb_geometry.get_grid_shape())
    U_fp = projector.project_hjb_to_fp(U_hjb)

    expected_shape = problem.fp_geometry.get_grid_shape()
    assert U_fp.shape == expected_shape
```

## Next Steps for Development

### Phase 4: Solver Integration (Next)
- Update `solve_mfg()` to use dual geometries
- Modify `create_*_solver()` factory functions
- Add dual geometry examples
- Update documentation

### Phase 5: Advanced Projections (Future)
- Conservative grid restriction
- Edge-aware network projection
- Adaptive bandwidth KDE
- GPU acceleration for 2D/3D KDE
- Spectral methods for periodic domains

### Phase 6: High-Dimensional Support (Future)
- Implicit geometry projections
- Dimensionality reduction
- Sparse grid projections
- Monte Carlo methods

## References

### Implementation Files
- `mfg_pde/geometry/projection.py`: Core implementation
- `mfg_pde/core/mfg_problem.py`: Integration
- `tests/unit/geometry/test_geometry_projection.py`: Test suite

### Related Documentation
- `docs/theory/geometry_projection_mathematical_formulation.md`: Math theory
- `docs/development/CONSISTENCY_GUIDE.md`: Code standards
- Issue #257: Dual geometry architecture

---

**Document Version**: 1.0
**Implementation Status**: ✅ Core complete, solver integration next
