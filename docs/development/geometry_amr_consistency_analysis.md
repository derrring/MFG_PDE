# Geometry Module AMR Consistency Analysis

**Date**: August 1, 2025  
**Status**: Complete architectural review after AMR implementation  
**Purpose**: Ensure consistency, integrity, and extensibility across all dimensions

## Current Geometry Module Architecture

### Domain Coverage Analysis

| Dimension | Domain Class | AMR Support | Status | Missing Components |
|-----------|--------------|-------------|--------|-------------------|
| **1D** | `Domain1D` | ❌ **MISSING** | Incomplete | 1D AMR implementation |
| **2D Structured** | `Domain2D` → QuadTree | ✅ Complete | Production | None |
| **2D Unstructured** | `Domain2D` → Triangular | ✅ Complete | Production | FEM solver integration |
| **Network** | `NetworkGeometry` | ❌ Not applicable | Complete | N/A (discrete structure) |

### Critical Gap Identified: 1D AMR Missing

The geometry module lacks **1D AMR support**, creating an inconsistency in the architecture:

```python
# Current inconsistency:
# 2D: Full AMR support available
amr_2d = create_amr_solver(problem_2d)           # ✅ Works
triangular_amr = create_triangular_amr_solver()  # ✅ Works

# 1D: No AMR support
amr_1d = create_amr_solver(problem_1d)           # ❌ FAILS - no 1D AMR
```

## Required 1D AMR Implementation

### 1D AMR Mathematical Foundation

**1D Adaptive Mesh:**
- **Intervals**: [x₀, x₁], [x₁, x₂], ..., [xₙ₋₁, xₙ]
- **Refinement**: Split interval [xᵢ, xᵢ₊₁] → [xᵢ, xₘ], [xₘ, xᵢ₊₁]
- **Error Estimation**: |u'(xᵢ)| Δxᵢ + |u''(xᵢ)| (Δxᵢ)²

**Conservative Transfer:**
```
∫[old_interval] m(x) dx = ∫[child1] m(x) dx + ∫[child2] m(x) dx
```

### Implementation Requirements

#### 1. 1D Interval Tree Structure
```python
@dataclass
class Interval1D:
    """1D interval for adaptive refinement."""
    x_min: float
    x_max: float
    level: int
    parent: Optional['Interval1D'] = None
    children: Optional[List['Interval1D']] = None
    
    @property
    def center(self) -> float:
        return 0.5 * (self.x_min + self.x_max)
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    def subdivide(self) -> List['Interval1D']:
        """Split interval into 2 children."""
        mid = self.center
        return [
            Interval1D(self.x_min, mid, self.level + 1, parent=self),
            Interval1D(mid, self.x_max, self.level + 1, parent=self)
        ]
```

#### 2. 1D AMR Mesh Manager
```python
class OneDimensionalAMRMesh:
    """1D adaptive mesh for MFG problems."""
    
    def __init__(self, domain: Domain1D, 
                 refinement_criteria: AMRRefinementCriteria):
        self.domain = domain
        self.criteria = refinement_criteria
        self.intervals: Dict[int, Interval1D] = {}
        self.leaf_intervals: List[int] = []
        
    def refine_interval(self, interval_id: int) -> List[int]:
        """Refine 1D interval."""
        
    def adapt_mesh_1d(self, solution_data: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Adapt 1D mesh based on solution gradients."""
```

#### 3. Integration with Existing Domain1D
```python
def create_1d_amr_mesh(domain_1d: Domain1D, 
                       error_threshold: float = 1e-4) -> OneDimensionalAMRMesh:
    """Create 1D AMR mesh from existing Domain1D."""
```

## Consistency Issues Analysis

### 1. **Architectural Inconsistency**

**Problem:** Different AMR approaches for different dimensions
```python
# Current inconsistent interfaces:
quadtree_amr = AdaptiveMesh(domain_bounds_2d)      # 2D structured
triangular_amr = TriangularAMRMesh(mesh_data_2d)   # 2D unstructured  
# Missing: 1D AMR interface
```

**Solution:** Unified AMR interface
```python
# Proposed consistent interface:
amr_1d = create_amr_mesh(domain_1d, dimension=1)
amr_2d_structured = create_amr_mesh(domain_2d, dimension=2, mesh_type="structured") 
amr_2d_triangular = create_amr_mesh(mesh_data_2d, dimension=2, mesh_type="triangular")
```

### 2. **Factory Pattern Inconsistency**

**Current State:**
```python
# 2D AMR: Multiple factory functions
from mfg_pde.factory import create_amr_solver  # Only works for 2D structured
from mfg_pde.geometry import create_triangular_amr_mesh  # 2D unstructured
# Missing: 1D AMR factory
```

**Should Be:**
```python
# Unified factory supporting all dimensions
create_amr_solver(problem_1d)  # Should work for 1D
create_amr_solver(problem_2d)  # Should work for 2D  
create_amr_solver(problem_2d, mesh_type="triangular")  # Should work for triangular
```

### 3. **Error Estimator Inconsistency**

**Current:**
- ✅ `BaseErrorEstimator` (abstract base)
- ✅ `GradientErrorEstimator` (for 2D structured)
- ✅ `TriangularMeshErrorEstimator` (for 2D triangular)
- ❌ **Missing**: 1D error estimator

### 4. **Backend Integration Gaps**

**JAX Acceleration Coverage:**
- ✅ 2D structured AMR: JAX-accelerated error computation
- ✅ 2D triangular AMR: Ready for JAX integration  
- ❌ 1D AMR: Not implemented

**Numba Optimization Coverage:**
- ✅ 2D structured: Numba tree traversal
- ✅ 2D triangular: Ready for Numba integration
- ❌ 1D: Not implemented

## Extensibility Analysis

### Current Extension Points

#### ✅ **Well-Designed Extension Points:**
1. **`BaseErrorEstimator`**: Clean abstract interface for new error estimators
2. **`AMRRefinementCriteria`**: Configurable refinement parameters
3. **`MeshData`**: Universal mesh container supporting any element type
4. **Backend system**: Pluggable JAX/NumPy/Numba backends

#### ✅ **Good Polymorphism:**
```python
# Works for any error estimator implementation
error_estimator: BaseErrorEstimator = GradientErrorEstimator()  # or TriangularMeshErrorEstimator()
amr_mesh.adapt_mesh(solution_data, error_estimator)
```

#### ❌ **Missing Extension Points:**
1. **Dimension-agnostic AMR interface**: No common base class for 1D/2D AMR
2. **Element-type factory**: No unified way to create AMR for different element types
3. **Refinement strategy plugin system**: Hard-coded red/green refinement

### Proposed Extension Architecture

#### Universal AMR Base Class
```python
class BaseAMRMesh(ABC):
    """Abstract base class for all AMR mesh types."""
    
    @abstractmethod
    def adapt_mesh(self, solution_data: Dict[str, np.ndarray], 
                   error_estimator: BaseErrorEstimator) -> Dict[str, int]:
        """Adapt mesh based on solution data."""
        
    @abstractmethod
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mesh statistics."""
        
    @abstractmethod
    def export_mesh_data(self) -> MeshData:
        """Export to universal MeshData format."""
```

#### Dimension-Specific Implementations
```python
class OneDimensionalAMRMesh(BaseAMRMesh):
    """1D AMR implementation."""
    
class TwoDimensionalStructuredAMRMesh(BaseAMRMesh): 
    """2D structured (quadtree) AMR."""
    
class TwoDimensionalTriangularAMRMesh(BaseAMRMesh):
    """2D unstructured (triangular) AMR."""
```

## Integrity Analysis

### Data Flow Consistency

#### ✅ **Consistent Data Formats:**
```python
# All use same solution data format
solution_data = {'U': np.ndarray, 'M': np.ndarray}

# All export to same format  
mesh_data: MeshData = amr_mesh.export_mesh_data()
```

#### ✅ **Consistent Configuration:**
```python
# Same refinement criteria across dimensions
criteria = AMRRefinementCriteria(
    error_threshold=1e-4,
    max_refinement_levels=5
)
```

#### ❌ **Inconsistent Interfaces:**
```python
# Different function signatures
quadtree_stats = quadtree_amr.get_mesh_statistics()           # ✅ Implemented
triangular_stats = triangular_amr.get_mesh_statistics()       # ✅ Implemented  
# Missing: one_d_stats = one_d_amr.get_mesh_statistics()      # ❌ Not implemented
```

### Memory Management Consistency

#### ✅ **Consistent Memory Patterns:**
- All AMR implementations use similar data structures
- Consistent parent-child relationships
- Similar statistics tracking

#### ⚠️ **Potential Issues:**
- No memory pooling across different AMR types
- Each AMR type manages memory independently

## Recommendations for Complete Consistency

### Priority 1: Implement 1D AMR (HIGH)

#### 1.1 Create 1D AMR Implementation
```python
# mfg_pde/geometry/one_dimensional_amr.py
class OneDimensionalAMRMesh(BaseAMRMesh):
    """Complete 1D AMR implementation."""
```

#### 1.2 1D Error Estimator
```python
class OneDimensionalErrorEstimator(BaseErrorEstimator):
    """1D gradient-based error estimation."""
    
    def estimate_error(self, interval: Interval1D, 
                      solution_data: Dict[str, np.ndarray]) -> float:
        # 1D finite difference gradient estimation
        pass
```

#### 1.3 Factory Integration
```python
# Update mfg_pde/factory/solver_factory.py
def create_amr_solver(problem, **kwargs):
    if problem.dimension == 1:
        return create_1d_amr_solver(problem, **kwargs)
    elif problem.dimension == 2:
        return create_2d_amr_solver(problem, **kwargs)
```

### Priority 2: Unified AMR Interface (MEDIUM)

#### 2.1 Abstract Base Class
```python
# mfg_pde/geometry/base_amr.py
class BaseAMRMesh(ABC):
    """Universal AMR interface for all dimensions."""
```

#### 2.2 Dimension Detection
```python
def create_amr_mesh(domain_or_problem, **kwargs) -> BaseAMRMesh:
    """Universal AMR mesh factory."""
    if hasattr(domain_or_problem, 'dimension'):
        dim = domain_or_problem.dimension
    else:
        dim = infer_dimension(domain_or_problem)
    
    if dim == 1:
        return OneDimensionalAMRMesh(...)
    elif dim == 2:
        return create_2d_amr_mesh(...)
```

### Priority 3: Extension Point Improvements (LOW)

#### 3.1 Refinement Strategy Plugin System
```python
class RefinementStrategy(ABC):
    @abstractmethod
    def refine_element(self, element, criteria) -> List[Element]:
        pass

class RedRefinementStrategy(RefinementStrategy): pass
class GreenRefinementStrategy(RefinementStrategy): pass
class AdaptiveRefinementStrategy(RefinementStrategy): pass
```

#### 3.2 Performance Backend Integration
```python
class AMRBackendManager:
    """Manage JAX/Numba backends for all AMR types."""
    
    def get_error_computer(self, dimension: int, backend: str):
        if dimension == 1:
            return get_1d_error_computer(backend)
        elif dimension == 2:
            return get_2d_error_computer(backend)
```

## Implementation Plan

### Phase 1: 1D AMR Implementation (2-3 weeks)
- [ ] Implement `Interval1D` data structure
- [ ] Create `OneDimensionalAMRMesh` class
- [ ] Develop `OneDimensionalErrorEstimator`
- [ ] Add JAX acceleration for 1D operations
- [ ] Create factory integration
- [ ] Write comprehensive tests

### Phase 2: Interface Unification (1-2 weeks)  
- [ ] Create `BaseAMRMesh` abstract class
- [ ] Refactor existing 2D AMR to inherit from base
- [ ] Update factory functions for consistency
- [ ] Unify configuration and statistics interfaces

### Phase 3: Extension System (1 week)
- [ ] Create refinement strategy plugin system
- [ ] Enhance backend management
- [ ] Add performance monitoring across all AMR types
- [ ] Create extension documentation

## Summary

### Current State Assessment:
- ✅ **2D AMR**: Complete and well-designed
- ❌ **1D AMR**: Critical gap - missing entirely
- ⚠️ **Architecture**: Inconsistent interfaces between dimensions
- ✅ **Extensibility**: Good foundation, needs completion

### Critical Action Required:
**1D AMR implementation is essential** for:
1. **Architectural consistency** across all dimensions
2. **User expectations** - AMR should work for all MFG problems
3. **Code completeness** - geometry module should handle all cases
4. **Factory pattern integrity** - `create_amr_solver()` should work universally

### Success Metrics:
```python
# This should work after implementation:
problem_1d = create_1d_problem()
problem_2d = create_2d_problem()

amr_1d = create_amr_solver(problem_1d)  # ✅ Should work
amr_2d = create_amr_solver(problem_2d)  # ✅ Already works

result_1d = amr_1d.solve()  # ✅ Should work  
result_2d = amr_2d.solve()  # ✅ Already works
```

The geometry module will achieve full consistency and integrity once 1D AMR is implemented and interfaces are unified.