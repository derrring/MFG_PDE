# AMR Performance Architecture: JAX + Numba Strategy

**Date**: August 1, 2025  
**Status**: Architecture Recommendation  
**Purpose**: Optimal performance architecture for AMR implementation

## Core Principle

> **Use JAX for your core, high-performance, differentiable computations that can be expressed as pure functions.**
> 
> **Use Numba as a surgical tool to speed up specific, imperative bottlenecks, especially in helper functions or legacy code that would be difficult to rewrite for JAX.**

## Current AMR Architecture Analysis

### What Should Use JAX (Pure Functional Operations)

✅ **JAX-Optimal Components:**

1. **Error Indicator Computation** - Pure mathematical functions
2. **Conservative Interpolation** - Differentiable tensor operations
3. **Gradient Calculations** - Built-in JAX gradients
4. **Solution Transfer** - Pure array transformations
5. **Mass Conservation Enforcement** - Mathematical constraints

### What Should Use Numba (Imperative Bottlenecks)

🔧 **Numba-Optimal Components:**

1. **Quadtree Traversal** - Complex tree navigation logic
2. **Cell Index Mapping** - Coordinate-to-cell lookups
3. **Adaptive Mesh Bookkeeping** - Parent-child relationships
4. **Cell Subdivision Logic** - Imperative tree construction
5. **Mesh Statistics Collection** - Data structure traversal

## Proposed Architecture Refactoring

### JAX Core Computational Kernels

```python
# mfg_pde/geometry/amr_jax_kernels.py
"""JAX-accelerated core AMR computations."""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from typing import Tuple

@jax.jit
def compute_error_indicators_vectorized(
    U: jnp.ndarray, 
    M: jnp.ndarray, 
    dx: float, 
    dy: float
) -> jnp.ndarray:
    """
    Vectorized error indicator computation using JAX.
    
    Pure function - ideal for JAX JIT compilation and GPU acceleration.
    """
    # First-order gradients using JAX
    dU_dx = jnp.gradient(U, dx, axis=0)
    dU_dy = jnp.gradient(U, dy, axis=1)
    dM_dx = jnp.gradient(M, dx, axis=0) 
    dM_dy = jnp.gradient(M, dy, axis=1)
    
    # Second-order gradients (curvature)
    d2U_dx2 = jnp.gradient(dU_dx, dx, axis=0)
    d2U_dy2 = jnp.gradient(dU_dy, dy, axis=1)
    d2M_dx2 = jnp.gradient(dM_dx, dx, axis=0)
    d2M_dy2 = jnp.gradient(dM_dy, dy, axis=1)
    
    # Combined error indicators
    grad_magnitude_U = jnp.sqrt(dU_dx**2 + dU_dy**2)
    grad_magnitude_M = jnp.sqrt(dM_dx**2 + dM_dy**2)
    
    curvature_U = jnp.abs(d2U_dx2) + jnp.abs(d2U_dy2)
    curvature_M = jnp.abs(d2M_dx2) + jnp.abs(d2M_dy2)
    
    # Adaptive weighting
    error_indicator = jnp.maximum(grad_magnitude_U, grad_magnitude_M)
    curvature_indicator = jnp.maximum(curvature_U, curvature_M)
    
    return error_indicator + 0.1 * curvature_indicator


@jax.jit
def conservative_mass_transfer_2d(
    old_density: jnp.ndarray,
    old_coords: Tuple[jnp.ndarray, jnp.ndarray],
    new_coords: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    Conservative density transfer between mesh levels.
    
    Pure function with automatic differentiation support.
    """
    from jax.scipy.ndimage import map_coordinates
    
    # Compute total mass (for conservation)
    old_dx = old_coords[0][1] - old_coords[0][0]
    old_dy = old_coords[1][1] - old_coords[1][0]
    total_mass = jnp.sum(old_density) * old_dx * old_dy
    
    # Create coordinate mapping
    old_shape = old_density.shape
    new_x, new_y = new_coords
    new_shape = (len(new_x), len(new_y))
    
    # Map new coordinates to old grid indices
    x_indices = jnp.interp(new_x, old_coords[0], jnp.arange(old_shape[0]))
    y_indices = jnp.interp(new_y, old_coords[1], jnp.arange(old_shape[1])) 
    
    X_indices, Y_indices = jnp.meshgrid(x_indices, y_indices, indexing='ij')
    coords = jnp.stack([X_indices.ravel(), Y_indices.ravel()])
    
    # Conservative interpolation
    new_density = map_coordinates(old_density, coords, order=1, mode='nearest')
    new_density = new_density.reshape(new_shape)
    
    # Enforce mass conservation
    new_dx = new_x[1] - new_x[0] if len(new_x) > 1 else old_dx
    new_dy = new_y[1] - new_y[0] if len(new_y) > 1 else old_dy
    new_mass = jnp.sum(new_density) * new_dx * new_dy
    
    conservation_factor = total_mass / (new_mass + 1e-12)
    return new_density * conservation_factor


@jax.jit
def gradient_preserving_interpolation_2d(
    values: jnp.ndarray,
    old_coords: Tuple[jnp.ndarray, jnp.ndarray],
    new_coords: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    Gradient-preserving value function interpolation.
    
    Uses higher-order interpolation to maintain smoothness.
    """
    from jax.scipy.ndimage import map_coordinates
    
    old_shape = values.shape
    new_x, new_y = new_coords
    new_shape = (len(new_x), len(new_y))
    
    # Map coordinates
    x_indices = jnp.interp(new_x, old_coords[0], jnp.arange(old_shape[0]))
    y_indices = jnp.interp(new_y, old_coords[1], jnp.arange(old_shape[1]))
    
    X_indices, Y_indices = jnp.meshgrid(x_indices, y_indices, indexing='ij')
    coords = jnp.stack([X_indices.ravel(), Y_indices.ravel()])
    
    # Higher-order interpolation for gradient preservation
    interpolated = map_coordinates(values, coords, order=3, mode='nearest')
    return interpolated.reshape(new_shape)


@jax.jit
def adaptive_error_threshold_scaling(
    base_threshold: float,
    mesh_level: int,
    solution_variance: float,
    adaptive_factor: float = 0.1
) -> float:
    """
    Adaptively scale error threshold based on mesh level and solution characteristics.
    """
    level_scaling = jnp.power(0.5, mesh_level)  # Stricter thresholds for finer levels
    variance_scaling = 1.0 + adaptive_factor * jnp.log1p(solution_variance)
    
    return base_threshold * level_scaling * variance_scaling


# Vectorized operations for batch processing
compute_error_batch = vmap(compute_error_indicators_vectorized, in_axes=(0, 0, None, None))
conservative_transfer_batch = vmap(conservative_mass_transfer_2d, in_axes=(0, 0, 0))
```

### Numba-Accelerated Imperative Operations

```python
# mfg_pde/geometry/amr_numba_kernels.py
"""Numba-accelerated imperative AMR operations."""

import numba
from numba import jit, types
from numba.typed import Dict, List
import numpy as np
from typing import Tuple, List as PyList, Dict as PyDict

@numba.jit(nopython=True, cache=True)
def find_containing_cell_numba(
    x: float, 
    y: float,
    cell_bounds: np.ndarray,  # Shape: (N, 4) for (x_min, x_max, y_min, y_max)
    cell_levels: np.ndarray,  # Shape: (N,) for level of each cell
    leaf_mask: np.ndarray     # Shape: (N,) boolean mask for leaf cells
) -> int:
    """
    Fast cell lookup using Numba.
    
    This is imperative logic that's hard to vectorize efficiently in JAX.
    """
    best_cell = -1
    best_level = -1
    
    for i in range(len(cell_bounds)):
        if not leaf_mask[i]:
            continue
            
        x_min, x_max, y_min, y_max = cell_bounds[i]
        
        if x_min <= x <= x_max and y_min <= y <= y_max:
            if cell_levels[i] > best_level:
                best_level = cell_levels[i]
                best_cell = i
    
    return best_cell


@numba.jit(nopython=True, cache=True)
def collect_refinement_candidates(
    error_indicators: np.ndarray,
    cell_bounds: np.ndarray,
    cell_levels: np.ndarray,
    leaf_mask: np.ndarray,
    error_threshold: float,
    max_level: int,
    min_cell_size: float
) -> np.ndarray:
    """
    Collect cells that should be refined.
    
    Complex conditional logic - good fit for Numba.
    """
    candidates = []
    
    for i in range(len(error_indicators)):
        if not leaf_mask[i]:
            continue
            
        # Check refinement criteria
        if error_indicators[i] <= error_threshold:
            continue
            
        if cell_levels[i] >= max_level:
            continue
            
        x_min, x_max, y_min, y_max = cell_bounds[i]
        dx = x_max - x_min
        dy = y_max - y_min
        
        if min(dx, dy) / 2.0 <= min_cell_size:
            continue
            
        candidates.append(i)
    
    return np.array(candidates, dtype=np.int64)


@numba.jit(nopython=True, cache=True)
def collect_coarsening_candidates(
    error_indicators: np.ndarray,
    parent_child_map: np.ndarray,  # Shape: (N, 4) mapping parent to 4 children
    cell_levels: np.ndarray,
    leaf_mask: np.ndarray,
    coarsening_threshold: float
) -> np.ndarray:
    """
    Collect parent cells whose children can be coarsened.
    
    Complex parent-child relationship logic - ideal for Numba.
    """
    candidates = []
    
    for parent_idx in range(len(parent_child_map)):
        children = parent_child_map[parent_idx]
        
        # Check if all children are leaves
        all_leaves = True
        for child_idx in children:
            if child_idx >= 0 and not leaf_mask[child_idx]:
                all_leaves = False
                break
        
        if not all_leaves:
            continue
            
        # Check if all children have low error
        all_low_error = True
        for child_idx in children:
            if child_idx >= 0 and error_indicators[child_idx] > coarsening_threshold:
                all_low_error = False
                break
        
        if all_low_error:
            candidates.append(parent_idx)
    
    return np.array(candidates, dtype=np.int64)


@numba.jit(nopython=True, cache=True)
def update_mesh_statistics(
    cell_bounds: np.ndarray,
    cell_levels: np.ndarray,
    leaf_mask: np.ndarray
) -> Tuple[int, int, float, float, float]:
    """
    Efficiently compute mesh statistics.
    
    Iteration-heavy computation - good for Numba.
    """
    total_cells = 0
    leaf_cells = 0
    max_level = 0
    total_area = 0.0
    min_cell_size = np.inf
    max_cell_size = 0.0
    
    for i in range(len(cell_bounds)):
        total_cells += 1
        
        if leaf_mask[i]:
            leaf_cells += 1
            
        level = cell_levels[i]
        if level > max_level:
            max_level = level
            
        x_min, x_max, y_min, y_max = cell_bounds[i]
        area = (x_max - x_min) * (y_max - y_min)
        total_area += area
        
        cell_size = min(x_max - x_min, y_max - y_min)
        if cell_size < min_cell_size:
            min_cell_size = cell_size
        if cell_size > max_cell_size:
            max_cell_size = cell_size
    
    return total_cells, leaf_cells, max_level, total_area, min_cell_size, max_cell_size


@numba.jit(nopython=True, cache=True)
def map_coordinates_to_cells(
    coordinates: np.ndarray,  # Shape: (N, 2) for (x, y) coordinates
    cell_bounds: np.ndarray,  # Shape: (M, 4) for cell bounds
    leaf_mask: np.ndarray     # Shape: (M,) for leaf cells
) -> np.ndarray:
    """
    Map a batch of coordinates to their containing cells.
    
    Nested loop structure - efficient with Numba.
    """
    result = np.full(len(coordinates), -1, dtype=np.int64)
    
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        
        for j in range(len(cell_bounds)):
            if not leaf_mask[j]:
                continue
                
            x_min, x_max, y_min, y_max = cell_bounds[j]
            
            if x_min <= x <= x_max and y_min <= y <= y_max:
                result[i] = j
                break
    
    return result
```

### Hybrid Architecture Implementation

```python
# mfg_pde/geometry/amr_mesh_optimized.py
"""Optimized AMR mesh using JAX + Numba hybrid architecture."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

try:
    import jax.numpy as jnp
    from .amr_jax_kernels import (
        compute_error_indicators_vectorized,
        conservative_mass_transfer_2d,
        gradient_preserving_interpolation_2d,
        adaptive_error_threshold_scaling
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from .amr_numba_kernels import (
        find_containing_cell_numba,
        collect_refinement_candidates,
        collect_coarsening_candidates,
        update_mesh_statistics,
        map_coordinates_to_cells
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from .amr_mesh import QuadTreeNode, AMRRefinementCriteria, BaseErrorEstimator


class OptimizedGradientErrorEstimator(BaseErrorEstimator):
    """JAX-accelerated error estimator for vectorized computation."""
    
    def __init__(self, backend=None):
        self.backend = backend
        self.use_jax = JAX_AVAILABLE and (backend is None or backend.name == "jax")
    
    def estimate_error_vectorized(
        self, 
        solution_data: Dict[str, NDArray],
        dx: float,
        dy: float
    ) -> NDArray:
        """
        Vectorized error estimation using JAX.
        
        This is the main computational kernel - perfect for JAX.
        """
        if not self.use_jax:
            return self._estimate_error_numpy(solution_data, dx, dy)
            
        U = jnp.array(solution_data['U'])
        M = jnp.array(solution_data['M'])
        
        # Use JAX-compiled kernel for pure computation
        error_indicators = compute_error_indicators_vectorized(U, M, dx, dy)
        
        return np.array(error_indicators)
    
    def estimate_error(self, node: QuadTreeNode, solution_data: Dict[str, NDArray]) -> float:
        """Single-cell error estimation (fallback for compatibility)."""
        # Use vectorized version and extract single value
        error_field = self.estimate_error_vectorized(solution_data, node.dx, node.dy)
        
        # Map node to grid indices (using Numba if available)
        i, j = self._get_cell_indices(node, solution_data['U'].shape)
        
        if 0 <= i < error_field.shape[0] and 0 <= j < error_field.shape[1]:
            return float(error_field[i, j])
        else:
            return 0.0


class OptimizedAdaptiveMesh:
    """
    Optimized adaptive mesh using JAX + Numba hybrid architecture.
    
    JAX: Pure computational kernels (error estimation, interpolation)
    Numba: Imperative mesh operations (traversal, bookkeeping)
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float, float, float],
        refinement_criteria: Optional[AMRRefinementCriteria] = None,
        error_estimator: Optional[BaseErrorEstimator] = None,
        backend=None
    ):
        self.domain_bounds = domain_bounds
        self.criteria = refinement_criteria or AMRRefinementCriteria()
        self.error_estimator = error_estimator or OptimizedGradientErrorEstimator(backend)
        self.backend = backend
        
        # Initialize mesh data structures for efficient access
        self._initialize_optimized_structures()
        
        # Performance tracking
        self.performance_stats = {
            'jax_time': 0.0,
            'numba_time': 0.0,
            'numpy_time': 0.0
        }
    
    def _initialize_optimized_structures(self):
        """Initialize data structures optimized for Numba operations."""
        # Flat arrays for efficient Numba processing
        self.cell_bounds = np.array([
            [self.domain_bounds[0], self.domain_bounds[1], 
             self.domain_bounds[2], self.domain_bounds[3]]
        ], dtype=np.float64)
        
        self.cell_levels = np.array([0], dtype=np.int32)
        self.leaf_mask = np.array([True], dtype=np.bool_)
        self.parent_child_map = np.full((1, 4), -1, dtype=np.int64)
        
        # Keep quadtree for compatibility
        self.root = QuadTreeNode(
            level=0,
            x_min=self.domain_bounds[0], x_max=self.domain_bounds[1],
            y_min=self.domain_bounds[2], y_max=self.domain_bounds[3]
        )
        self.leaf_nodes = [self.root]
    
    def adapt_mesh_optimized(
        self, 
        solution_data: Dict[str, NDArray]
    ) -> Dict[str, int]:
        """
        Optimized mesh adaptation using JAX + Numba hybrid approach.
        """
        import time
        
        stats = {'total_refined': 0, 'total_coarsened': 0}
        
        # Phase 1: JAX-accelerated error computation
        start_time = time.time()
        if isinstance(self.error_estimator, OptimizedGradientErrorEstimator):
            dx = (self.domain_bounds[1] - self.domain_bounds[0]) / solution_data['U'].shape[0]
            dy = (self.domain_bounds[3] - self.domain_bounds[2]) / solution_data['U'].shape[1]
            
            error_indicators = self.error_estimator.estimate_error_vectorized(
                solution_data, dx, dy
            )
        else:
            # Fallback to per-cell estimation
            error_indicators = self._compute_error_indicators_fallback(solution_data)
        
        self.performance_stats['jax_time'] += time.time() - start_time
        
        # Phase 2: Numba-accelerated mesh operations
        start_time = time.time()
        
        if NUMBA_AVAILABLE:
            # Use Numba for imperative mesh operations
            refined_count = self._refine_mesh_numba(error_indicators)
            coarsened_count = self._coarsen_mesh_numba(error_indicators)
        else:
            # Fallback to Python implementation
            refined_count = self._refine_mesh_python(error_indicators)
            coarsened_count = self._coarsen_mesh_python(error_indicators)
        
        self.performance_stats['numba_time'] += time.time() - start_time
        
        stats['total_refined'] = refined_count
        stats['total_coarsened'] = coarsened_count
        
        return stats
    
    def _refine_mesh_numba(self, error_indicators: NDArray) -> int:
        """Numba-accelerated mesh refinement."""
        if not NUMBA_AVAILABLE:
            return self._refine_mesh_python(error_indicators)
        
        # Map error indicators to cells (using spatial mapping)
        cell_error_values = self._map_errors_to_cells(error_indicators)
        
        # Use Numba to collect refinement candidates
        candidates = collect_refinement_candidates(
            cell_error_values,
            self.cell_bounds,
            self.cell_levels,
            self.leaf_mask,
            self.criteria.error_threshold,
            self.criteria.max_refinement_levels,
            self.criteria.min_cell_size
        )
        
        # Execute refinements (this updates data structures)
        refined_count = 0
        for candidate_idx in candidates:
            if self._execute_refinement(candidate_idx):
                refined_count += 1
        
        return refined_count
    
    def _coarsen_mesh_numba(self, error_indicators: NDArray) -> int:
        """Numba-accelerated mesh coarsening."""
        if not NUMBA_AVAILABLE:
            return self._coarsen_mesh_python(error_indicators)
        
        cell_error_values = self._map_errors_to_cells(error_indicators)
        coarsening_threshold = (self.criteria.error_threshold * 
                              self.criteria.coarsening_threshold)
        
        # Use Numba to collect coarsening candidates
        candidates = collect_coarsening_candidates(
            cell_error_values,
            self.parent_child_map,
            self.cell_levels,
            self.leaf_mask,
            coarsening_threshold
        )
        
        # Execute coarsening
        coarsened_count = 0
        for candidate_idx in candidates:
            if self._execute_coarsening(candidate_idx):
                coarsened_count += 1
        
        return coarsened_count
    
    def conservative_solution_transfer(
        self, 
        old_solution: Dict[str, NDArray],
        old_coords: Tuple[NDArray, NDArray],
        new_coords: Tuple[NDArray, NDArray]
    ) -> Dict[str, NDArray]:
        """
        JAX-accelerated conservative solution transfer.
        
        This is pure computation - perfect for JAX.
        """
        import time
        start_time = time.time()
        
        new_solution = {}
        
        if JAX_AVAILABLE and self.backend and self.backend.name == "jax":
            # Use JAX kernels for solution transfer
            if 'M' in old_solution:
                new_solution['M'] = np.array(conservative_mass_transfer_2d(
                    jnp.array(old_solution['M']), old_coords, new_coords
                ))
            
            if 'U' in old_solution:
                new_solution['U'] = np.array(gradient_preserving_interpolation_2d(
                    jnp.array(old_solution['U']), old_coords, new_coords
                ))
        else:
            # Fallback to NumPy implementation
            new_solution = self._transfer_solution_numpy(
                old_solution, old_coords, new_coords
            )
        
        self.performance_stats['jax_time'] += time.time() - start_time
        return new_solution
    
    def get_performance_statistics(self) -> Dict[str, float]:
        """Get detailed performance breakdown."""
        total_time = sum(self.performance_stats.values())
        
        if total_time > 0:
            return {
                'jax_fraction': self.performance_stats['jax_time'] / total_time,
                'numba_fraction': self.performance_stats['numba_time'] / total_time,
                'numpy_fraction': self.performance_stats['numpy_time'] / total_time,
                'total_time': total_time,
                **self.performance_stats
            }
        else:
            return self.performance_stats.copy()
```

## Architecture Benefits

### JAX Strengths Applied Correctly:

1. **Error Computation**: Pure mathematical functions → Perfect for `@jax.jit`
2. **Conservative Interpolation**: Differentiable operations → Automatic gradients
3. **Vectorized Operations**: Batch processing → GPU acceleration
4. **Functional Purity**: No side effects → Optimal compilation

### Numba Strengths Applied Correctly:

1. **Tree Traversal**: Complex navigation logic → Imperative efficiency
2. **Conditional Logic**: Refinement decisions → Branch-heavy code
3. **Data Structure Updates**: Mesh bookkeeping → Mutable operations
4. **Index Mapping**: Coordinate lookups → Cache-friendly loops

### Performance Expected:

- **JAX Components**: 5-20× speedup on GPU, 2-5× on CPU
- **Numba Components**: 10-100× speedup over pure Python
- **Hybrid Efficiency**: Best of both worlds without architectural compromises

This architecture follows your principle perfectly: JAX for pure, differentiable computations and Numba for imperative bottlenecks that are hard to vectorize.
