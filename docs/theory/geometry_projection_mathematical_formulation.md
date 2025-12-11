# Geometry Projection Mathematical Formulation

**Status**: ✅ COMPLETED (Issue #257 Phases 1-3)
**Created**: 2025-11-10
**Last Updated**: 2025-11-10

## Overview

The geometry projection system enables **hybrid MFG solvers** where the HJB and FP equations are solved on different discretizations. This is essential for:

1. **Particle-based FP + Grid-based HJB**: Lagrangian particle methods with Eulerian value iteration
2. **Multi-resolution methods**: Fine grid for HJB, coarse grid for FP
3. **Network-based FP + Grid-based HJB**: Discrete agent movement on graphs with continuous value functions
4. **Implicit geometry FP + Grid-based HJB**: High-dimensional particle methods with low-dimensional projections

## Mathematical Framework

### Dual Geometry Formulation

For a Mean Field Game on spatial domain $\Omega \subset \mathbb{R}^d$ with time horizon $[0,T]$:

**Standard (Unified) Formulation:**
- Both HJB and FP solved on same discretization $\mathcal{G}$
- $u: [0,T] \times \mathcal{G} \to \mathbb{R}$ (value function)
- $m: [0,T] \times \mathcal{G} \to \mathbb{R}_+$ (density)

**Dual Geometry Formulation:**
- HJB solved on discretization $\mathcal{G}_{\text{HJB}}$ (e.g., grid)
- FP solved on discretization $\mathcal{G}_{\text{FP}}$ (e.g., particles)
- Projection operators:
  - $\Pi_{\text{H→F}}: \mathcal{G}_{\text{HJB}} \to \mathcal{G}_{\text{FP}}$ (value function projection)
  - $\Pi_{\text{F→H}}: \mathcal{G}_{\text{FP}} \to \mathcal{G}_{\text{HJB}}$ (density projection)

### Fixed Point Iteration with Projections

The MFG fixed point iteration becomes:

```
Given: m^0 on G_FP
For k = 0, 1, 2, ...:
    1. Project density: m^k_HJB = Π_F→H(m^k_FP)
    2. Solve HJB: u^{k+1}_HJB = SolveHJB(m^k_HJB)  on G_HJB
    3. Project value: u^{k+1}_FP = Π_H→F(u^{k+1}_HJB)
    4. Solve FP: m^{k+1}_FP = SolveFP(u^{k+1}_FP)  on G_FP
    5. Check convergence
```

## Projection Methods

### 1. Grid → Points Interpolation (HJB → Particle FP)

**Mathematical Formulation:**

For value function $u_h: \mathcal{G}_{\text{grid}} \to \mathbb{R}$ on regular grid with spacing $h$, evaluate at particle positions $\{x_i\}_{i=1}^N$:

$$u(x_i) = \sum_{j \in \text{neighbors}(x_i)} w_j(x_i) \cdot u_h(x_j)$$

where $w_j(x_i)$ are interpolation weights.

**1D Linear Interpolation:**
$$u(x) = u_h(x_j) + \frac{x - x_j}{h}(u_h(x_{j+1}) - u_h(x_j))$$
where $x_j \leq x < x_{j+1}$.

**2D Bilinear Interpolation:**
For point $(x, y)$ in cell $[x_i, x_{i+1}] \times [y_j, y_{j+1}]$:
$$u(x,y) = (1-s)(1-t) u_{i,j} + s(1-t) u_{i+1,j} + (1-s)t u_{i,j+1} + st u_{i+1,j+1}$$
where $s = (x - x_i)/h_x$, $t = (y - y_j)/h_y$.

**3D Trilinear Interpolation:**
Extension of bilinear with 8 neighbors.

**Implementation:** `mfg_pde/geometry/projection.py:158`
```python
def _interpolate_grid_to_points(self, values_on_grid: NDArray, target_points: NDArray) -> NDArray:
    """Linear/bilinear/trilinear interpolation."""
    interpolator = self.hjb_geometry.get_interpolator()
    return interpolator(values_on_grid, target_points)
```

### 2. Particles → Grid KDE (Particle FP → HJB)

**Mathematical Formulation:**

Given particle distribution $\{(x_i, m_i)\}_{i=1}^N$ with positions $x_i \in \mathbb{R}^d$ and masses $m_i > 0$, reconstruct density on grid $\{y_j\}$ using Kernel Density Estimation:

$$m_h(y) = \sum_{i=1}^N m_i \cdot K_h(y - x_i)$$

where $K_h(z) = \frac{1}{h^d} K(z/h)$ is the scaled kernel with bandwidth $h$.

**Gaussian Kernel:**
$$K(z) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{\|z\|^2}{2}\right)$$

**Bandwidth Selection:**

- **Scott's Rule:** $h = n^{-1/(d+4)} \sigma$ where $\sigma$ is standard deviation
- **Silverman's Rule:** $h = \left(\frac{n(d+2)}{4}\right)^{-1/(d+4)} \sigma$
- **Manual:** User-specified constant

**Properties:**
- Mass conservation: $\int_\Omega m_h(y) dy = \sum_{i=1}^N m_i$ (with appropriate normalization)
- Smoothness: $m_h \in C^\infty(\mathbb{R}^d)$
- Convergence: $m_h \to m$ as $h \to 0$ and $n \to \infty$

**Implementation:** `mfg_pde/geometry/projection.py:234`
```python
def _project_particles_to_grid_kde(self, particle_values: NDArray,
                                   bandwidth: str | float,
                                   backend: BaseBackend | None) -> NDArray:
    """KDE projection with GPU acceleration for 1D."""
    from mfg_pde.alg.numerical.density_estimation import gaussian_kde_gpu
    # ... bandwidth computation ...
    if dimension == 1:
        density = gaussian_kde_gpu(particles, grid, bandwidth, backend)
    else:
        density = self._histogram_density_nd(...)  # Fallback
    return density.reshape(grid_shape)
```

### 3. Grid → Grid Interpolation (Multi-resolution)

**Mathematical Formulation:**

Project from coarse grid $\mathcal{G}_{\text{coarse}}$ with spacing $H$ to fine grid $\mathcal{G}_{\text{fine}}$ with spacing $h < H$:

$$u_h(y) = \mathcal{I}(u_H)(y)$$

where $\mathcal{I}$ is prolongation operator (linear/bilinear/trilinear interpolation).

**Conservative Property (Desired):**
For density projection (fine → coarse), should satisfy:
$$\int_{\Omega_j} m_h(x) dx = M_j$$
where $\Omega_j$ is the coarse cell and $M_j$ is the coarse cell mass.

**Implementation:** `mfg_pde/geometry/projection.py:171`
```python
def _interpolate_grid_to_grid(self, values_on_source_grid: NDArray) -> NDArray:
    """Grid → Grid with automatic reshaping."""
    target_points = self.fp_geometry.get_spatial_grid()
    values_flat = self._interpolate_grid_to_points(values_on_source_grid, target_points)

    if isinstance(self.fp_geometry, CartesianGrid):
        target_shape = self.fp_geometry.get_grid_shape()
        return values_flat.reshape(target_shape)
    return values_flat
```

### 4. Grid → Grid Restriction (Mass Conservation)

**Mathematical Formulation:**

Project from fine grid $\mathcal{G}_{\text{fine}}$ to coarse grid $\mathcal{G}_{\text{coarse}}$ preserving mass:

$$M_j = \frac{1}{|\Omega_j|} \int_{\Omega_j} m_h(x) dx \approx \frac{1}{N_j} \sum_{i \in \Omega_j} m_h(x_i)$$

where $N_j$ is number of fine grid points in coarse cell $\Omega_j$.

**Current Implementation:** Uses interpolation (not fully conservative)
**Future Enhancement:** True conservative restriction using averaging

**Implementation:** `mfg_pde/geometry/projection.py:187`
```python
def _restrict_grid_to_grid(self, values_on_source_grid: NDArray) -> NDArray:
    """Grid restriction (currently via interpolation)."""
    # TODO: Implement true conservative restriction
    return self._interpolate_grid_to_grid(values_on_source_grid)
```

### 5. Network → Grid Projection

**Mathematical Formulation:**

Given density on network nodes $\{m_i\}_{i=1}^{N_{\text{nodes}}}$ at positions $\{x_i\}$, project to grid using Gaussian kernel spreading:

$$m_h(y) = \sum_{i=1}^{N_{\text{nodes}}} m_i \cdot K_h(y - x_i)$$

This treats network nodes as point masses and spreads them spatially.

**Edge-Aware Alternative:**
For edges $(i,j)$ with flow $f_{ij}$, spread along edge:
$$m_h(y) = \sum_{(i,j) \in E} f_{ij} \int_0^1 K_h(y - ((1-s)x_i + s x_j)) ds$$

**Implementation:** Not yet in codebase (uses nearest neighbor fallback)

### 6. Grid → Network Projection

**Mathematical Formulation:**

Sample grid values at network node positions:
$$u_{\text{node}_i} = \mathcal{I}(u_h)(x_i)$$

where $\mathcal{I}$ is the grid interpolator.

**Implementation:** Straightforward using existing grid interpolators

### 7. Nearest Neighbor Fallback

**Mathematical Formulation:**

For arbitrary geometries without specialized projectors, use KD-tree nearest neighbor:

$$u(y) = u_h(x_{i^*}) \quad \text{where} \quad i^* = \arg\min_i \|y - x_i\|$$

**Implementation:** `mfg_pde/geometry/projection.py:201`
```python
def _nearest_neighbor_projection(self, values: NDArray, target_points: NDArray) -> NDArray:
    """Generic fallback using KD-tree."""
    from scipy.spatial import KDTree
    source_points = self.hjb_geometry.get_spatial_grid()
    tree = KDTree(source_points)
    _, indices = tree.query(target_points)
    return values.ravel()[indices]
```

## Architecture

### GeometryProjector Class

**Location:** `mfg_pde/geometry/projection.py:90`

**Core Methods:**
- `project_hjb_to_fp(U_on_hjb_geometry)`: $\Pi_{\text{H→F}}(u)$
- `project_fp_to_hjb(M_on_fp_geometry)`: $\Pi_{\text{F→H}}(m)$

**Auto-Detection Logic:**

```python
if hjb_is_grid and fp_is_particles:
    hjb_to_fp_method = "interpolation"  # Grid → Particles
    fp_to_hjb_method = "kde"            # Particles → Grid (KDE)
elif hjb_is_grid and fp_is_grid:
    hjb_to_fp_method = "grid_interpolation"  # Multi-resolution
    fp_to_hjb_method = "grid_restriction"    # Conservative restriction
elif hjb_is_grid and fp_is_network:
    hjb_to_fp_method = "interpolation"       # Grid → Network nodes
    fp_to_hjb_method = "network_to_grid_kde" # Network → Grid (KDE)
else:
    # Fallback to generic methods
    hjb_to_fp_method = "interpolation"
    fp_to_hjb_method = "nearest"
```

### ProjectionRegistry Pattern

**Location:** `mfg_pde/geometry/projection.py:41`

**Design:** Decorator-based registration with hierarchical fallback

**Registration:**
```python
@ProjectionRegistry.register(SourceType, TargetType, "hjb_to_fp")
def custom_projector(source_geo, target_geo, values, **kwargs):
    # Custom projection logic
    return projected_values
```

**Lookup Hierarchy:**
1. **Exact type match**: `(type(source), type(target), direction)`
2. **Category match**: `isinstance(source, registered_type)`
3. **Auto-detection fallback**: Generic methods

**Complexity:** O(N) specialized projectors, not O(N²) all pairs

## Error Analysis

### Interpolation Error

For value function projection with interpolation order $p$ and grid spacing $h$:
$$\|u - u_h\|_{L^\infty} \leq C h^p \|\nabla^p u\|_{L^\infty}$$

- Linear (1D): $p=1$
- Bilinear (2D): $p=1$ per dimension
- Trilinear (3D): $p=1$ per dimension

### KDE Error

For density projection with bandwidth $h$ and $n$ particles:
$$\mathbb{E}[\|m_h - m\|_{L^2}^2] = O(h^4) + O(n^{-1}h^{-d})$$

**Optimal bandwidth:** Balances bias ($h^4$) and variance ($n^{-1}h^{-d}$)
$$h^* \sim n^{-1/(d+4)}$$

### Mass Conservation Error

For KDE projection:
$$\left|\int_\Omega m_h(x) dx - \sum_{i=1}^N m_i\right| \leq C h^2$$

where $C$ depends on boundary effects and kernel normalization.

## Performance Considerations

### Grid → Particles (N particles, M grid points)
- **1D**: $O(N \log M)$ with binary search
- **2D/3D**: $O(N)$ with RegularGridInterpolator (tensor product structure)

### Particles → Grid KDE
- **Naive**: $O(NM)$ for $N$ particles, $M$ grid points
- **GPU (1D)**: $O(NM)$ but 10-100× faster with parallelization
- **Fast summation (future)**: $O((N+M)\log(N+M))$ with tree codes

### Network Projections
- **Grid → Network**: $O(N_{\text{nodes}})$ (just interpolation)
- **Network → Grid**: $O(N_{\text{nodes}} \cdot N_{\text{grid}})$ (kernel spreading, can be expensive)

## Usage Examples

### Example 1: Particle FP + Grid HJB

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Grid for HJB (50×50)
hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[51, 51])

# Particles for FP (1000 agents)
class ParticleGeometry:
    def __init__(self, num_particles, bounds):
        self.num_particles = num_particles
        self.bounds = bounds

    def get_spatial_grid(self):
        # Return particle positions (N, 2)
        return np.random.rand(self.num_particles, 2)

fp_particles = ParticleGeometry(num_particles=1000, bounds=(0, 1, 0, 1))

# Create MFG problem with dual geometries
problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_particles,
    time_domain=(1.0, 50),
    sigma=0.1
)

# Projector automatically created
projector = problem.geometry_projector

# In solver iteration:
# 1. Project particles → grid
m_grid = projector.project_fp_to_hjb(m_particles, bandwidth="scott")

# 2. Solve HJB on grid
u_grid = solve_hjb(problem, m_grid)

# 3. Project grid → particles
u_particles = projector.project_hjb_to_fp(u_grid)

# 4. Update particles using u_particles
m_particles_new = update_particles(u_particles)
```

### Example 2: Multi-Resolution

```python
# Coarse grid for FP (20×20)
fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[21, 21])

# Fine grid for HJB (100×100)
hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[101, 101])

problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_grid,
    time_domain=(1.0, 50),
    sigma=0.1
)

# Uses grid_interpolation and grid_restriction automatically
```

### Example 3: Custom Network Projection

```python
from mfg_pde.geometry import GeometryProjector, ProjectionRegistry, GridNetwork

# Register custom network → grid projector
@ProjectionRegistry.register(GridNetwork, TensorProductGrid, "fp_to_hjb")
def network_to_grid_custom(network, grid, node_density, **kwargs):
    """Custom projection with edge-aware spreading."""
    node_positions = network.get_node_positions()
    grid_points = grid.get_spatial_grid()

    # Custom logic: spread density along edges
    # ... implementation ...

    return grid_density

# Now GeometryProjector will use this for GridNetwork → TensorProductGrid
```

## Integration with MFGProblem

**Location:** `mfg_pde/core/mfg_problem.py:89-120`

### Dual Geometry Parameters

```python
problem = MFGProblem(
    hjb_geometry=...,  # Geometry for HJB solver
    fp_geometry=...,   # Geometry for FP solver
    # ... other parameters ...
)
```

**Backward Compatibility:**
```python
# Unified mode (still works)
problem = MFGProblem(geometry=grid, ...)
# Both HJB and FP use same geometry
```

**Error Handling:**
- Specifying both `geometry` and `hjb_geometry`/`fp_geometry` raises `ValueError`
- Specifying only one of `hjb_geometry`/`fp_geometry` raises `ValueError`

### Automatic Projector Creation

When dual geometries are provided:
```python
if hjb_geometry is not None and fp_geometry is not None:
    self.geometry_projector = GeometryProjector(
        hjb_geometry=hjb_geometry,
        fp_geometry=fp_geometry,
        projection_method="auto"
    )
```

### Unified Access

```python
problem.hjb_geometry        # Always available (unified or dual)
problem.fp_geometry         # Always available (unified or dual)
problem.geometry_projector  # None if unified, GeometryProjector if dual
```

## Testing

**Location:** `tests/unit/geometry/test_geometry_projection.py`

### Test Coverage (27 tests total)

**Phase 1 - Basic Projections (13 tests):**
- 1D grid projections (linear interpolation accuracy)
- 2D grid projections (bilinear interpolation accuracy)
- 3D grid projections (trilinear interpolation accuracy)
- Shape verification
- Same-geometry identity projection

**Phase 2 - Registry Pattern (7 tests):**
- Registration mechanics
- Lookup with exact match
- Lookup with category match (isinstance)
- Integration with GeometryProjector
- Fallback behavior

**Phase 3 - MFGProblem Integration (7 tests):**
- Dual geometry specification
- Backward compatibility with unified geometry
- Error handling (partial specification, conflicts)
- Projector attribute access

### Key Test Patterns

**Accuracy Test:**
```python
def test_grid_to_grid_1d_interpolation(self):
    coarse_grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx_points=[11])
    fine_grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx_points=[21])
    projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

    # Linear function u(x) = x (exact for linear interpolation)
    U_coarse = coarse_grid.get_spatial_grid()
    U_fine = projector.project_hjb_to_fp(U_coarse)

    expected = fine_grid.get_spatial_grid()
    np.testing.assert_allclose(U_fine, expected, rtol=1e-10)
```

**Conservation Test:**
```python
def test_particles_to_grid_conservation(self):
    # KDE should approximately conserve mass
    total_mass_particles = np.sum(particle_values)
    grid_density = projector.project_fp_to_hjb(particle_values)
    total_mass_grid = np.sum(grid_density) * cell_volume

    assert np.abs(total_mass_grid - total_mass_particles) < 0.01 * total_mass_particles
```

## Future Enhancements

### 1. Conservative Grid Restriction
Replace interpolation-based restriction with true conservative averaging:
```python
def conservative_restrict(fine_values, fine_grid, coarse_grid):
    """Mass-conserving restriction using averaging."""
    for coarse_cell in coarse_grid:
        fine_cells = fine_grid.get_cells_in(coarse_cell)
        coarse_values[cell] = np.mean(fine_values[fine_cells])
```

### 2. Edge-Aware Network Projection
Spread density along network edges, not just at nodes:
```python
def network_to_grid_edge_aware(network, grid, edge_flows):
    """Spread flow along edges using line integrals."""
    for edge in network.edges:
        integrate_along_edge(edge, edge_flows[edge], grid)
```

### 3. Adaptive Bandwidth Selection
Choose KDE bandwidth based on local density:
```python
def adaptive_kde(particles, grid, method="knn"):
    """Adaptive bandwidth based on k-nearest neighbors."""
    bandwidths = estimate_local_bandwidth(particles, k=5)
    return gaussian_kde_variable_bandwidth(particles, grid, bandwidths)
```

### 4. GPU Acceleration for Multi-D KDE
Extend GPU KDE from 1D to 2D/3D:
```python
def gaussian_kde_gpu_2d(particles, grid, bandwidth, backend):
    """GPU-accelerated 2D KDE using CUDA."""
    # Vectorized distance computation on GPU
    # Parallel reduction for kernel summation
```

### 5. Implicit Geometry Support
Add projections for high-dimensional implicit domains:
```python
@ProjectionRegistry.register(ImplicitDomain, TensorProductGrid, "fp_to_hjb")
def implicit_to_grid_3d(implicit_geo, grid, values, **kwargs):
    """Project from high-D implicit to nD grid via slicing."""
    # Extract nD slice from high-dimensional domain
    # Apply dimensionality reduction projection
```

## References

### Implemented
- **Issue #257**: Dual geometry architecture
- `mfg_pde/geometry/projection.py`: Core implementation
- `mfg_pde/core/mfg_problem.py:89-120`: Integration
- `tests/unit/geometry/test_geometry_projection.py`: Comprehensive tests

### Related Work
- Kernel Density Estimation: Silverman (1986), Scott (1992)
- Multigrid Methods: Briggs et al. (2000)
- Particle-Grid Methods: Cottet & Koumoutsakos (2000)
- Conservative Restriction: AMR literature

---

**Document Version**: 1.0
**Implementation Status**: ✅ Phases 1-3 complete, Phase 4 (API integration) in progress
