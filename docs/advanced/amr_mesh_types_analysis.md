# AMR for Different Mesh Types and Basis Functions

**Date**: August 1, 2025  
**Status**: Analysis and Extension Recommendations  
**Scope**: Triangular FEM meshes, wavelet bases, and other local basis methods

## Current AMR Implementation Analysis

### What We Have: Structured Quadtree AMR
```python
# Current implementation (mfg_pde/geometry/amr_mesh.py)
class QuadTreeNode:
    # 2D rectangular cells with 4-way subdivision
    def subdivide(self) -> List['QuadTreeNode']:
        # Creates 4 rectangular children
        return [SW_child, SE_child, NW_child, NE_child]
```

**Strengths:**
- ✅ Simple, regular structure
- ✅ Easy coordinate mapping
- ✅ Efficient memory layout
- ✅ Natural tensor operations for JAX

**Limitations:**
- ❌ Only rectangular domains
- ❌ Cannot handle complex geometries
- ❌ Not optimal for anisotropic features
- ❌ Limited to axis-aligned refinement

## Extension 1: Triangular Finite Element AMR

### Mathematical Foundation

For triangular FEM meshes, we need:

**Triangle Subdivision Strategies:**
1. **Red Refinement**: Divide triangle into 4 similar triangles
2. **Green Refinement**: Divide triangle into 2 triangles (for conformity)
3. **Blue Refinement**: Divide triangle into 3 triangles

**Triangle Quality Metrics:**
- **Aspect Ratio**: `longest_edge / (2 * area / shortest_edge)`
- **Shape Regularity**: `radius_inscribed / radius_circumscribed`
- **Angle Quality**: Deviation from equilateral triangle

### Implementation Architecture

```python
# mfg_pde/geometry/triangular_amr.py
"""Adaptive Mesh Refinement for Triangular FEM Meshes."""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class TriangleNode:
    """Triangular mesh element for AMR."""
    
    # Geometric properties
    vertices: np.ndarray  # Shape: (3, 2) for 2D coordinates
    level: int
    element_id: int
    
    # Mesh relationships
    parent: Optional['TriangleNode'] = None
    children: Optional[List['TriangleNode']] = None
    neighbors: List[Optional['TriangleNode']] = None
    
    # Solution data
    solution_data: Optional[Dict[str, np.ndarray]] = None
    error_estimate: float = 0.0
    
    # Mesh quality
    aspect_ratio: float = 0.0
    min_angle: float = 0.0
    max_angle: float = 0.0
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = [None, None, None]  # 3 edges
        self._compute_geometric_properties()
    
    def _compute_geometric_properties(self):
        """Compute triangle geometric properties."""
        v0, v1, v2 = self.vertices
        
        # Edge lengths
        e01 = np.linalg.norm(v1 - v0)
        e12 = np.linalg.norm(v2 - v1)
        e20 = np.linalg.norm(v0 - v2)
        
        # Area using cross product
        self.area = 0.5 * abs(np.cross(v1 - v0, v2 - v0))
        
        # Centroid
        self.centroid = (v0 + v1 + v2) / 3.0
        
        # Quality metrics
        perimeter = e01 + e12 + e20
        self.aspect_ratio = perimeter**2 / (12.0 * self.area) if self.area > 0 else np.inf
        
        # Angles using law of cosines
        angles = []
        edges = [e01, e12, e20]
        for i in range(3):
            a, b, c = edges[i], edges[(i+1)%3], edges[(i+2)%3]
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
            angles.append(np.arccos(cos_angle))
        
        self.min_angle = np.min(angles)
        self.max_angle = np.max(angles)
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf element."""
        return self.children is None or len(self.children) == 0
    
    @property
    def diameter(self) -> float:
        """Longest edge length."""
        v0, v1, v2 = self.vertices
        return max(
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2)
        )
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside triangle using barycentric coordinates."""
        v0, v1, v2 = self.vertices
        
        # Compute barycentric coordinates
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-12:
            return False
            
        a = ((v1[1] - v2[1]) * (point[0] - v2[0]) + (v2[0] - v1[0]) * (point[1] - v2[1])) / denom
        b = ((v2[1] - v0[1]) * (point[0] - v2[0]) + (v0[0] - v2[0]) * (point[1] - v2[1])) / denom
        c = 1 - a - b
        
        return a >= 0 and b >= 0 and c >= 0
    
    def red_refinement(self) -> List['TriangleNode']:
        """
        Red refinement: Divide into 4 similar triangles.
        
        Creates 4 children by connecting edge midpoints:
        - 1 central triangle (inverted orientation)
        - 3 corner triangles (same orientation as parent)
        """
        if not self.is_leaf:
            raise ValueError("Cannot refine non-leaf triangle")
        
        v0, v1, v2 = self.vertices
        
        # Edge midpoints
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)
        
        # Create 4 children
        children = [
            # Corner triangles
            TriangleNode(
                vertices=np.array([v0, m01, m20]),
                level=self.level + 1,
                element_id=self.element_id * 4 + 0,
                parent=self
            ),
            TriangleNode(
                vertices=np.array([m01, v1, m12]),
                level=self.level + 1,
                element_id=self.element_id * 4 + 1,
                parent=self
            ),
            TriangleNode(
                vertices=np.array([m20, m12, v2]),
                level=self.level + 1,
                element_id=self.element_id * 4 + 2,
                parent=self
            ),
            # Central triangle (inverted)
            TriangleNode(
                vertices=np.array([m01, m12, m20]),
                level=self.level + 1,
                element_id=self.element_id * 4 + 3,
                parent=self
            )
        ]
        
        self.children = children
        return children
    
    def green_refinement(self, edge_index: int) -> List['TriangleNode']:
        """
        Green refinement: Divide into 2 triangles by bisecting one edge.
        
        Used to maintain mesh conformity when neighbors have different levels.
        """
        if not self.is_leaf:
            raise ValueError("Cannot refine non-leaf triangle")
        
        v0, v1, v2 = self.vertices
        vertices = [v0, v1, v2]
        
        # Find edge to bisect
        if edge_index == 0:  # Edge v0-v1
            midpoint = 0.5 * (v0 + v1)
            child1_vertices = np.array([v0, midpoint, v2])
            child2_vertices = np.array([midpoint, v1, v2])
        elif edge_index == 1:  # Edge v1-v2
            midpoint = 0.5 * (v1 + v2)
            child1_vertices = np.array([v0, v1, midpoint])
            child2_vertices = np.array([v0, midpoint, v2])
        else:  # Edge v2-v0
            midpoint = 0.5 * (v2 + v0)
            child1_vertices = np.array([v0, v1, midpoint])
            child2_vertices = np.array([midpoint, v1, v2])
        
        children = [
            TriangleNode(
                vertices=child1_vertices,
                level=self.level + 1,
                element_id=self.element_id * 2 + 0,
                parent=self
            ),
            TriangleNode(
                vertices=child2_vertices,
                level=self.level + 1,
                element_id=self.element_id * 2 + 1,
                parent=self
            )
        ]
        
        self.children = children
        return children


class TriangularAMRMesh:
    """Adaptive triangular mesh for finite element methods."""
    
    def __init__(self, 
                 initial_vertices: np.ndarray,  # Shape: (N, 2)
                 initial_triangles: np.ndarray,  # Shape: (M, 3) indices
                 refinement_criteria: Optional[AMRRefinementCriteria] = None):
        
        self.refinement_criteria = refinement_criteria or AMRRefinementCriteria()
        
        # Build initial mesh
        self.triangles: List[TriangleNode] = []
        self._build_initial_mesh(initial_vertices, initial_triangles)
        
        # Mesh statistics
        self.total_triangles = len(self.triangles)
        self.max_level = 0
        
    def _build_initial_mesh(self, vertices: np.ndarray, triangle_indices: np.ndarray):
        """Build initial triangular mesh from vertex/triangle data."""
        for i, triangle in enumerate(triangle_indices):
            triangle_vertices = vertices[triangle]
            self.triangles.append(TriangleNode(
                vertices=triangle_vertices,
                level=0,
                element_id=i
            ))
        
        # Build neighbor relationships
        self._build_neighbor_relationships()
    
    def _build_neighbor_relationships(self):
        """Build triangle-triangle adjacency information."""
        # This is computationally intensive but crucial for mesh quality
        for i, tri1 in enumerate(self.triangles):
            for j, tri2 in enumerate(self.triangles):
                if i == j:
                    continue
                
                # Check if triangles share an edge
                shared_vertices = self._find_shared_vertices(tri1.vertices, tri2.vertices)
                if len(shared_vertices) == 2:
                    # They share an edge - they're neighbors
                    edge_index = self._find_edge_index(tri1.vertices, shared_vertices)
                    tri1.neighbors[edge_index] = tri2
    
    def _find_shared_vertices(self, v1: np.ndarray, v2: np.ndarray, 
                            tol: float = 1e-10) -> List[int]:
        """Find vertices shared between two triangles."""
        shared = []
        for i, vertex1 in enumerate(v1):
            for j, vertex2 in enumerate(v2):
                if np.linalg.norm(vertex1 - vertex2) < tol:
                    shared.append((i, j))
        return shared
    
    def _find_edge_index(self, vertices: np.ndarray, shared_vertices: List) -> int:
        """Find which edge of triangle contains the shared vertices."""
        if len(shared_vertices) != 2:
            return -1
        
        indices = [sv[0] for sv in shared_vertices]
        indices.sort()
        
        if indices == [0, 1]:
            return 0  # Edge v0-v1
        elif indices == [1, 2]:
            return 1  # Edge v1-v2
        else:
            return 2  # Edge v2-v0
    
    def refine_triangle(self, triangle: TriangleNode, 
                       strategy: str = "red") -> List[TriangleNode]:
        """Refine a triangle using specified strategy."""
        if strategy == "red":
            children = triangle.red_refinement()
        elif strategy.startswith("green"):
            edge_index = int(strategy[-1]) if len(strategy) > 5 else 0
            children = triangle.green_refinement(edge_index)
        else:
            raise ValueError(f"Unknown refinement strategy: {strategy}")
        
        # Update mesh statistics
        self.total_triangles += len(children) - 1  # Replace 1 with N children
        self.max_level = max(self.max_level, triangle.level + 1)
        
        # Maintain mesh conformity
        self._enforce_conformity(triangle, children)
        
        return children
    
    def _enforce_conformity(self, parent: TriangleNode, children: List[TriangleNode]):
        """
        Enforce mesh conformity by refining neighbors if necessary.
        
        This prevents hanging nodes in FEM mesh.
        """
        for neighbor in parent.neighbors:
            if neighbor is None or not neighbor.is_leaf:
                continue
            
            # Check if neighbor needs green refinement for conformity
            level_difference = abs(neighbor.level - (parent.level + 1))
            if level_difference > 1:
                # Need to refine neighbor to maintain conformity
                self._green_refine_for_conformity(neighbor, parent)
    
    def _green_refine_for_conformity(self, triangle: TriangleNode, refined_neighbor: TriangleNode):
        """Apply green refinement to maintain mesh conformity."""
        # Find shared edge and apply green refinement
        shared_vertices = self._find_shared_vertices(triangle.vertices, refined_neighbor.vertices)
        if len(shared_vertices) == 2:
            edge_index = self._find_edge_index(triangle.vertices, shared_vertices)
            self.refine_triangle(triangle, f"green{edge_index}")


class TriangularFEMErrorEstimator(BaseErrorEstimator):
    """Error estimator for triangular FEM meshes."""
    
    def estimate_error(self, triangle: TriangleNode, 
                      solution_data: Dict[str, np.ndarray]) -> float:
        """
        Estimate error for triangular element.
        
        Uses gradient recovery and element residual methods.
        """
        if 'U' not in solution_data or 'M' not in solution_data:
            return 0.0
        
        # Element-based error estimation
        element_error = self._compute_element_residual(triangle, solution_data)
        
        # Edge-based error estimation (jump discontinuities)
        edge_error = self._compute_edge_jumps(triangle, solution_data)
        
        # Combined error indicator
        return max(element_error, edge_error)
    
    def _compute_element_residual(self, triangle: TriangleNode, 
                                solution_data: Dict[str, np.ndarray]) -> float:
        """Compute element residual error indicator."""
        # Simplified implementation - in practice would use proper FEM assembly
        
        # Sample solution at triangle centroid
        U_val = self._interpolate_at_point(triangle.centroid, solution_data['U'])
        M_val = self._interpolate_at_point(triangle.centroid, solution_data['M'])
        
        # Estimate gradients using vertex values
        grad_U = self._estimate_gradient(triangle, solution_data['U'])
        grad_M = self._estimate_gradient(triangle, solution_data['M'])
        
        # Element residual (simplified MFG residual)
        residual = abs(np.linalg.norm(grad_U)) + abs(np.linalg.norm(grad_M))
        
        # Scale by element size
        h = triangle.diameter
        return h * residual
    
    def _compute_edge_jumps(self, triangle: TriangleNode, 
                          solution_data: Dict[str, np.ndarray]) -> float:
        """Compute edge jump error indicator."""
        max_jump = 0.0
        
        for neighbor in triangle.neighbors:
            if neighbor is None:
                continue  # Boundary edge
            
            # Compute solution jump across shared edge
            jump_U = self._compute_solution_jump(triangle, neighbor, solution_data['U'])
            jump_M = self._compute_solution_jump(triangle, neighbor, solution_data['M'])
            
            edge_jump = max(abs(jump_U), abs(jump_M))
            max_jump = max(max_jump, edge_jump)
        
        return max_jump
    
    def _interpolate_at_point(self, point: np.ndarray, solution: np.ndarray) -> float:
        """Interpolate solution at given point (simplified)."""
        # In practice, would use proper FEM basis function interpolation
        # For now, return a dummy value
        return np.random.uniform(0, 1)
    
    def _estimate_gradient(self, triangle: TriangleNode, solution: np.ndarray) -> np.ndarray:
        """Estimate gradient within triangle (simplified)."""
        # In practice, would use gradient recovery methods
        return np.random.uniform(-1, 1, 2)
    
    def _compute_solution_jump(self, tri1: TriangleNode, tri2: TriangleNode, 
                             solution: np.ndarray) -> float:
        """Compute solution jump across shared edge."""
        # Simplified implementation
        return np.random.uniform(0, 0.1)
```

### Integration with MFG Solvers

```python
# mfg_pde/alg/mfg_solvers/triangular_amr_solver.py
"""MFG solver with triangular AMR mesh."""

from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from ..base_solver import BaseMFGSolver
from ...core.mfg_problem import MFGProblem
from ...geometry.triangular_amr import TriangularAMRMesh, TriangularFEMErrorEstimator
from ...utils.solver_result import SolverResult

class TriangularAMRMFGSolver(BaseMFGSolver):
    """MFG solver with adaptive triangular mesh refinement."""
    
    def __init__(self,
                 problem: MFGProblem,
                 initial_mesh: TriangularAMRMesh,
                 base_solver_type: str = "fem",
                 amr_frequency: int = 5,
                 max_amr_cycles: int = 3,
                 **kwargs):
        
        super().__init__(problem, kwargs.get('backend'))
        
        self.triangular_mesh = initial_mesh
        self.base_solver_type = base_solver_type
        self.amr_frequency = amr_frequency
        self.max_amr_cycles = max_amr_cycles
        
        # FEM-specific error estimator
        self.error_estimator = TriangularFEMErrorEstimator()
        
        # AMR statistics
        self.amr_stats = {
            'total_refinements': 0,
            'total_coarsenings': 0,
            'red_refinements': 0,
            'green_refinements': 0,
            'conformity_enforcements': 0
        }
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6, 
              verbose: bool = True) -> SolverResult:
        """Solve MFG problem with triangular AMR."""
        
        # Initialize FEM solution
        U_fem, M_fem = self._initialize_fem_solution()
        
        convergence_history = []
        
        for amr_cycle in range(self.max_amr_cycles):
            if verbose:
                print(f"AMR Cycle {amr_cycle + 1}/{self.max_amr_cycles}")
                print(f"Current mesh: {self.triangular_mesh.total_triangles} triangles")
            
            # Solve on current triangular mesh
            for iteration in range(max_iterations):
                # FEM solver step (simplified)
                U_new, M_new, residual = self._fem_solver_step(U_fem, M_fem)
                
                # Check convergence
                change = self._compute_solution_change(U_fem, M_fem, U_new, M_new)
                convergence_history.append(change)
                
                U_fem, M_fem = U_new, M_new
                
                # Periodic mesh adaptation
                if (iteration + 1) % self.amr_frequency == 0:
                    self._adapt_triangular_mesh({'U': U_fem, 'M': M_fem}, verbose)
                
                if change < tolerance:
                    if verbose:
                        print(f"Converged in {iteration + 1} iterations")
                    break
            
            # Final mesh adaptation
            adaptation_stats = self._adapt_triangular_mesh({'U': U_fem, 'M': M_fem}, verbose)
            
            if adaptation_stats['total_refined'] == 0:
                break
        
        # Convert FEM solution to uniform grid for compatibility
        U_uniform, M_uniform = self._project_fem_to_uniform(U_fem, M_fem)
        
        return SolverResult(
            U=U_uniform,
            M=M_uniform,
            convergence_achieved=convergence_history[-1] < tolerance,
            final_error=convergence_history[-1],
            convergence_history=convergence_history,
            execution_time=0.0,
            solver_info={
                'solver_type': 'triangular_amr',
                'amr_stats': self.amr_stats,
                'final_triangles': self.triangular_mesh.total_triangles,
                'max_level': self.triangular_mesh.max_level
            }
        )
    
    def _adapt_triangular_mesh(self, solution_data: Dict[str, NDArray], 
                             verbose: bool = False) -> Dict[str, int]:
        """Adapt triangular mesh based on error estimates."""
        
        refinements = 0
        red_refs = 0
        green_refs = 0
        
        # Collect leaf triangles for refinement
        leaf_triangles = [tri for tri in self.triangular_mesh.triangles if tri.is_leaf]
        
        for triangle in leaf_triangles:
            # Estimate error
            error = self.error_estimator.estimate_error(triangle, solution_data)
            triangle.error_estimate = error
            
            # Refinement decision
            if error > self.triangular_mesh.refinement_criteria.error_threshold:
                # Choose refinement strategy based on triangle quality
                if triangle.aspect_ratio < 2.0 and triangle.min_angle > np.pi/6:
                    # Good quality triangle - use red refinement
                    self.triangular_mesh.refine_triangle(triangle, "red")
                    red_refs += 1
                else:
                    # Poor quality triangle - use green refinement on longest edge
                    longest_edge = self._find_longest_edge(triangle)
                    self.triangular_mesh.refine_triangle(triangle, f"green{longest_edge}")
                    green_refs += 1
                
                refinements += 1
        
        # Update statistics
        self.amr_stats['total_refinements'] += refinements
        self.amr_stats['red_refinements'] += red_refs
        self.amr_stats['green_refinements'] += green_refs
        
        if verbose and refinements > 0:
            print(f"  Refined {refinements} triangles ({red_refs} red, {green_refs} green)")
        
        return {'total_refined': refinements, 'total_coarsened': 0}
```

## Extension 2: Wavelet-Based Adaptive Methods

### Mathematical Foundation

**Wavelet AMR Principle:**
Instead of geometric mesh refinement, use **hierarchical wavelet basis** with adaptive coefficient thresholding:

```
u(x,t) ≈ ∑_{j,k} u_{j,k} ψ_{j,k}(x)
```

Where:
- `j`: Scale level (resolution)
- `k`: Translation index (location)
- `ψ_{j,k}(x) = 2^{j/2} ψ(2^j x - k)`: Wavelet basis function

**Adaptive Strategy:**
1. **Coefficient Thresholding**: Keep only coefficients `|u_{j,k}| > ε`
2. **Dynamic Refinement**: Add basis functions where residual is large
3. **Coarsening**: Remove basis functions where coefficients are small

### Implementation Architecture

```python
# mfg_pde/geometry/wavelet_amr.py
"""Wavelet-based adaptive refinement for MFG problems."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

class WaveletBasis(ABC):
    """Abstract base class for wavelet basis functions."""
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward wavelet transform."""
        pass
    
    @abstractmethod
    def inverse_transform(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """Inverse wavelet transform."""
        pass
    
    @abstractmethod
    def threshold_coefficients(self, coeffs: Dict[str, np.ndarray], 
                             threshold: float) -> Dict[str, np.ndarray]:
        """Apply adaptive thresholding to coefficients."""
        pass

class Daubechies2DWavelet(WaveletBasis):
    """2D Daubechies wavelet basis implementation."""
    
    def __init__(self, wavelet_name: str = 'db4', max_levels: int = 5):
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets required for wavelet AMR")
        
        self.wavelet_name = wavelet_name
        self.max_levels = max_levels
        self.wavelet = pywt.Wavelet(wavelet_name)
    
    def transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """2D wavelet decomposition."""
        # Multi-level 2D wavelet transform
        coeffs = pywt.wavedec2(data, self.wavelet_name, level=self.max_levels)
        
        # Convert to dictionary format for easier manipulation
        result = {'approximation': coeffs[0]}
        
        for level, (cH, cV, cD) in enumerate(coeffs[1:]):
            result[f'horizontal_{level}'] = cH
            result[f'vertical_{level}'] = cV
            result[f'diagonal_{level}'] = cD
        
        return result
    
    def inverse_transform(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """Reconstruct data from wavelet coefficients."""
        # Convert back to PyWavelets format
        approximation = coeffs['approximation']
        details = []
        
        level = 0
        while f'horizontal_{level}' in coeffs:
            cH = coeffs[f'horizontal_{level}']
            cV = coeffs[f'vertical_{level}']
            cD = coeffs[f'diagonal_{level}']
            details.append((cH, cV, cD))
            level += 1
        
        # Reconstruct
        pywt_coeffs = [approximation] + details
        return pywt.waverec2(pywt_coeffs, self.wavelet_name)
    
    def threshold_coefficients(self, coeffs: Dict[str, np.ndarray], 
                             threshold: float, 
                             mode: str = 'soft') -> Dict[str, np.ndarray]:
        """Apply wavelet coefficient thresholding."""
        thresholded = {}
        
        # Keep approximation coefficients (low-frequency components)
        thresholded['approximation'] = coeffs['approximation'].copy()
        
        # Threshold detail coefficients
        for key, coeff_array in coeffs.items():
            if key != 'approximation':
                if mode == 'soft':
                    thresholded[key] = pywt.threshold(coeff_array, threshold, mode='soft')
                elif mode == 'hard':
                    thresholded[key] = pywt.threshold(coeff_array, threshold, mode='hard')
                else:
                    # Adaptive thresholding based on coefficient magnitude
                    mask = np.abs(coeff_array) > threshold
                    thresholded[key] = coeff_array * mask
        
        return thresholded
    
    def compute_coefficient_importance(self, coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute importance scores for wavelet coefficients."""
        importance = {}
        
        for key, coeff_array in coeffs.items():
            if key == 'approximation':
                # Approximation coefficients are always important
                importance[key] = np.ones_like(coeff_array)
            else:
                # Detail coefficients importance based on magnitude
                importance[key] = np.abs(coeff_array)
        
        return importance


class WaveletAMRSolver:
    """Wavelet-based adaptive solver for MFG problems."""
    
    def __init__(self,
                 problem,
                 wavelet_basis: WaveletBasis,
                 adaptive_threshold: float = 1e-4,
                 compression_ratio: float = 0.1):
        
        self.problem = problem
        self.wavelet_basis = wavelet_basis
        self.adaptive_threshold = adaptive_threshold
        self.compression_ratio = compression_ratio
        
        # Wavelet representation of solution
        self.U_coeffs: Optional[Dict[str, np.ndarray]] = None
        self.M_coeffs: Optional[Dict[str, np.ndarray]] = None
        
        # Adaptive support tracking
        self.active_coeffs_U: Optional[Dict[str, np.ndarray]] = None
        self.active_coeffs_M: Optional[Dict[str, np.ndarray]] = None
        
        # Statistics
        self.compression_stats = {
            'original_coeffs': 0,
            'active_coeffs': 0,
            'compression_achieved': 0.0
        }
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6,
              verbose: bool = True) -> Dict:
        """Solve MFG problem using wavelet-based adaptation."""
        
        # Initialize with uniform grid solution
        U_grid, M_grid = self._initialize_grid_solution()
        
        # Transform to wavelet domain
        self.U_coeffs = self.wavelet_basis.transform(U_grid)
        self.M_coeffs = self.wavelet_basis.transform(M_grid)
        
        convergence_history = []
        
        for iteration in range(max_iterations):
            if verbose and iteration % 10 == 0:
                print(f"Wavelet iteration {iteration}")
            
            # Adaptive coefficient selection
            self._adapt_wavelet_support()
            
            # Solve in compressed wavelet domain
            residual = self._wavelet_iteration_step()
            convergence_history.append(residual)
            
            if residual < tolerance:
                if verbose:
                    print(f"Wavelet solver converged in {iteration + 1} iterations")
                break
        
        # Reconstruct final solution
        U_final = self.wavelet_basis.inverse_transform(self.U_coeffs)
        M_final = self.wavelet_basis.inverse_transform(self.M_coeffs)
        
        return {
            'U': U_final,
            'M': M_final,
            'convergence_history': convergence_history,
            'compression_stats': self.compression_stats,
            'active_coefficients': {
                'U': self.active_coeffs_U,
                'M': self.active_coeffs_M
            }
        }
    
    def _adapt_wavelet_support(self):
        """Adaptively select active wavelet coefficients."""
        # Compute coefficient importance
        importance_U = self.wavelet_basis.compute_coefficient_importance(self.U_coeffs)
        importance_M = self.wavelet_basis.compute_coefficient_importance(self.M_coeffs)
        
        # Apply adaptive thresholding
        self.active_coeffs_U = self.wavelet_basis.threshold_coefficients(
            self.U_coeffs, self.adaptive_threshold
        )
        self.active_coeffs_M = self.wavelet_basis.threshold_coefficients(
            self.M_coeffs, self.adaptive_threshold
        )
        
        # Update compression statistics
        total_coeffs_U = sum(coeff.size for coeff in self.U_coeffs.values())
        active_coeffs_U = sum(np.count_nonzero(coeff) for coeff in self.active_coeffs_U.values())
        
        self.compression_stats['original_coeffs'] = total_coeffs_U
        self.compression_stats['active_coeffs'] = active_coeffs_U
        self.compression_stats['compression_achieved'] = 1.0 - (active_coeffs_U / total_coeffs_U)
    
    def _wavelet_iteration_step(self) -> float:
        """Perform one iteration in wavelet domain."""
        # This would implement the MFG iteration in wavelet space
        # For now, return a dummy residual
        return np.random.uniform(1e-6, 1e-3)
    
    def _initialize_grid_solution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize solution on uniform grid."""
        nx = 128  # Grid resolution
        x = np.linspace(self.problem.xmin, self.problem.xmax, nx)
        X, Y = np.meshgrid(x, x)
        
        # Initial guess
        U = np.zeros((nx, nx))
        M = np.exp(-(X**2 + Y**2))
        M = M / np.sum(M)  # Normalize
        
        return U, M


# Factory function for wavelet AMR
def create_wavelet_amr_solver(problem, 
                            wavelet_type: str = 'db4',
                            max_levels: int = 5,
                            adaptive_threshold: float = 1e-4) -> WaveletAMRSolver:
    """Create a wavelet-based AMR solver."""
    
    if wavelet_type.startswith('db'):
        basis = Daubechies2DWavelet(wavelet_type, max_levels)
    else:
        raise ValueError(f"Unsupported wavelet type: {wavelet_type}")
    
    return WaveletAMRSolver(
        problem=problem,
        wavelet_basis=basis,
        adaptive_threshold=adaptive_threshold
    )
```

## Applicability Analysis

### Current Quadtree AMR
| Feature | Coverage | Extension Needed |
|---------|----------|------------------|
| **Rectangular domains** | ✅ Fully supported | None |
| **Complex geometries** | ❌ Not supported | Triangular AMR |
| **Anisotropic refinement** | ❌ Only axis-aligned | Directional refinement |
| **FEM compatibility** | ❌ Structured only | Unstructured AMR |

### Triangular FEM AMR Extension
| Feature | Benefits | Implementation Status |
|---------|----------|---------------------|
| **Complex geometries** | ✅ Natural boundary fitting | 🚧 Architecture designed |
| **Mesh conformity** | ✅ No hanging nodes | 🚧 Conformity algorithms |
| **Quality control** | ✅ Triangle quality metrics | 🚧 Quality enforcement |
| **FEM integration** | ✅ Natural for FEM solvers | 🚧 FEM solver needed |

### Wavelet AMR Extension  
| Feature | Benefits | Implementation Status |
|---------|----------|---------------------|
| **Compression** | ✅ Sparse representation | 🚧 Basic framework |
| **Multi-scale** | ✅ Natural scale separation | 🚧 Basis functions |
| **Smooth solutions** | ✅ Optimal for smooth functions | 🚧 Coefficient adaptation |
| **GPU friendly** | ✅ Dense linear algebra | 🚧 JAX integration needed |

## Recommendations

### For Finite Element Meshes:
1. **Implement triangular AMR** for complex geometries
2. **Use red/green refinement** for mesh quality
3. **Integrate with FEM solvers** for proper basis functions
4. **Priority**: High for realistic applications

### For Wavelet Methods:
1. **Implement for smooth problems** where compression is effective
2. **Use for multi-scale phenomena** with scale separation
3. **Integrate with JAX** for efficient linear algebra
4. **Priority**: Medium for specialized applications

### Hybrid Approach:
```python
# Optimal strategy: Choose method based on problem characteristics
def create_adaptive_solver(problem, domain_type="rectangular", 
                          solution_smoothness="moderate"):
    
    if domain_type == "complex_geometry":
        return create_triangular_amr_solver(problem)
    elif solution_smoothness == "very_smooth":
        return create_wavelet_amr_solver(problem)
    else:
        return create_amr_solver(problem)  # Current quadtree AMR
```

The current quadtree AMR is excellent for structured problems, but extensions to triangular meshes and wavelet bases would significantly expand the applicability to complex geometries and smooth solution problems.