# Adaptive Mesh Refinement (AMR) System Design for MFG_PDE
## Dynamic 2D/3D Mesh Adaptation for Complex Domains

**Document Version**: 1.0  
**Date**: 2025-07-30  
**Author**: MFG_PDE Development Team  
**Status**: Design Specification

---

## 📋 Executive Summary

This document presents the design for an **Adaptive Mesh Refinement (AMR) system** for MFG_PDE that dynamically refines and coarsens meshes based on solution features, error estimates, and geometric requirements. The system integrates seamlessly with the existing **Gmsh → Meshio → PyVista pipeline** to provide research-grade computational capabilities for complex Mean Field Games problems.

**Key Features:**
- **Error-Based Refinement**: Automatic mesh adaptation based on solution gradients and error indicators
- **Feature-Driven Refinement**: Targeted refinement around critical solution features (shocks, boundaries, etc.)
- **Multi-Level Hierarchy**: Tree-based mesh structure supporting multiple refinement levels
- **MFG-Specific Criteria**: Specialized refinement for density concentrations and value function gradients
- **Real-Time Adaptation**: Dynamic refinement during solver iterations

---

## 🎯 Design Objectives

### Primary Goals
- **Automated Mesh Adaptation**: Intelligent refinement/coarsening without manual intervention
- **MFG-Optimized Criteria**: Refinement strategies tailored to Mean Field Games characteristics
- **Computational Efficiency**: Minimize computational cost while maximizing solution accuracy
- **Seamless Integration**: Work transparently with existing Gmsh/Meshio/PyVista pipeline
- **Multi-Dimensional Support**: Full 2D and 3D adaptive refinement capabilities

### Secondary Goals
- **Parallel Scalability**: Distributed mesh adaptation for large-scale problems
- **Memory Optimization**: Efficient data structures for multi-level meshes
- **Visualization Support**: Real-time mesh adaptation visualization with PyVista
- **Quality Preservation**: Maintain mesh quality during refinement operations

---

## 🏗️ System Architecture

### Core Components

```python
# Adaptive Mesh Refinement Architecture
mfg_pde/
└── geometry/
    ├── adaptive/
    │   ├── __init__.py              # AMR exports
    │   ├── refinement_criteria.py   # Error indicators and refinement criteria
    │   ├── mesh_hierarchy.py        # Multi-level mesh management
    │   ├── amr_manager.py           # Main AMR orchestration
    │   ├── solution_transfer.py     # Inter-mesh solution interpolation
    │   └── quality_metrics.py       # Mesh quality analysis
    └── mesh_manager.py              # Extended with AMR capabilities
```

### Integration Points

1. **Solver Integration**: AMR hooks into solver iteration loops
2. **Geometry Pipeline**: Extends existing Gmsh → Meshio → PyVista workflow
3. **Visualization**: Real-time mesh adaptation display
4. **Problem Definition**: AMR parameters in GeneralMFGProblem

---

## 🔧 Technical Implementation

### 1. Refinement Criteria System

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..base_geometry import MeshData

class RefinementCriterion(ABC):
    """Abstract base class for mesh refinement criteria."""
    
    @abstractmethod
    def compute_indicators(self, 
                          mesh_data: MeshData, 
                          solution: Dict[str, np.ndarray],
                          problem_context: Any) -> np.ndarray:
        """Compute refinement indicators for each element."""
        pass

class GradientRefinementCriterion(RefinementCriterion):
    """Refine based on solution gradient magnitude."""
    
    def __init__(self, threshold: float = 0.1, variable: str = "density"):
        self.threshold = threshold
        self.variable = variable
    
    def compute_indicators(self, mesh_data, solution, problem_context):
        """Compute gradient-based refinement indicators."""
        # Implementation for gradient-based refinement
        pass

class MFGDensityRefinementCriterion(RefinementCriterion):
    """MFG-specific refinement based on density concentrations."""
    
    def __init__(self, density_threshold: float = 0.05, 
                 gradient_threshold: float = 0.1):
        self.density_threshold = density_threshold
        self.gradient_threshold = gradient_threshold
    
    def compute_indicators(self, mesh_data, solution, problem_context):
        """Refine where density is high or changing rapidly."""
        density = solution.get("density", np.zeros(mesh_data.num_elements))
        
        # High density regions
        high_density_mask = density > self.density_threshold
        
        # High gradient regions
        density_gradient = self._compute_element_gradients(mesh_data, density)
        high_gradient_mask = np.linalg.norm(density_gradient, axis=1) > self.gradient_threshold
        
        # Combine criteria
        refinement_indicators = (high_density_mask | high_gradient_mask).astype(float)
        return refinement_indicators

class ValueFunctionRefinementCriterion(RefinementCriterion):
    """Refine based on value function characteristics."""
    
    def __init__(self, curvature_threshold: float = 0.2):
        self.curvature_threshold = curvature_threshold
    
    def compute_indicators(self, mesh_data, solution, problem_context):
        """Refine in regions of high value function curvature."""
        # Implementation for value function-based refinement
        pass
```

### 2. Multi-Level Mesh Hierarchy

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import numpy as np

@dataclass
class MeshLevel:
    """Represents a single level in the mesh hierarchy."""
    level: int
    mesh_data: MeshData
    parent_elements: Optional[np.ndarray] = None  # Mapping to parent level
    child_elements: Optional[Dict[int, List[int]]] = None  # Children of each element
    refinement_history: List[str] = field(default_factory=list)
    
class AdaptiveMeshHierarchy:
    """Manages multi-level adaptive mesh hierarchy."""
    
    def __init__(self, base_mesh: MeshData, max_levels: int = 5):
        self.base_mesh = base_mesh
        self.max_levels = max_levels
        self.levels = [MeshLevel(level=0, mesh_data=base_mesh)]
        self.current_level = 0
    
    def refine_elements(self, element_ids: np.ndarray, level: int = None) -> MeshLevel:
        """Refine specified elements to create new level."""
        if level is None:
            level = self.current_level
        
        if level >= self.max_levels - 1:
            raise ValueError(f"Maximum refinement level {self.max_levels} reached")
        
        current_mesh = self.levels[level].mesh_data
        
        # Generate refined mesh using Gmsh
        refined_mesh = self._gmsh_local_refinement(current_mesh, element_ids)
        
        # Create new mesh level
        new_level = MeshLevel(
            level=level + 1,
            mesh_data=refined_mesh,
            parent_elements=self._compute_parent_mapping(element_ids),
            refinement_history=self.levels[level].refinement_history + [f"refine_{len(element_ids)}_elements"]
        )
        
        # Add to hierarchy
        if len(self.levels) <= level + 1:
            self.levels.append(new_level)
        else:
            self.levels[level + 1] = new_level
        
        return new_level
    
    def coarsen_elements(self, element_ids: np.ndarray, level: int = None) -> Optional[MeshLevel]:
        """Coarsen specified elements by removing refinement."""
        # Implementation for mesh coarsening
        pass
    
    def _gmsh_local_refinement(self, mesh_data: MeshData, element_ids: np.ndarray) -> MeshData:
        """Use Gmsh for local mesh refinement."""
        import gmsh
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
        
        try:
            # Import current mesh
            gmsh.open(mesh_data.file_path)
            
            # Mark elements for refinement
            for elem_id in element_ids:
                gmsh.model.mesh.setSize(
                    gmsh.model.getBoundary([(mesh_data.dimension, elem_id)], False, False, True),
                    mesh_data.characteristic_length * 0.5  # Halve element size
                )
            
            # Generate refined mesh
            gmsh.model.mesh.generate(mesh_data.dimension)
            
            # Export refined mesh
            refined_file = mesh_data.file_path.replace('.msh', '_refined.msh')
            gmsh.write(refined_file)
            
            # Convert to MeshData
            import meshio
            refined_meshio = meshio.read(refined_file)
            refined_mesh_data = MeshData.from_meshio(refined_meshio, refined_file)
            
            return refined_mesh_data
            
        finally:
            gmsh.finalize()
```

### 3. AMR Manager - Main Orchestration

```python
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from ..base_geometry import MeshData
import logging

logger = logging.getLogger(__name__)

class AMRManager:
    """Main orchestrator for adaptive mesh refinement."""
    
    def __init__(self, 
                 initial_mesh: MeshData,
                 refinement_criteria: List[RefinementCriterion],
                 refinement_threshold: float = 0.3,
                 coarsening_threshold: float = 0.1,
                 max_levels: int = 5,
                 min_elements: int = 100,
                 max_elements: int = 100000):
        
        self.hierarchy = AdaptiveMeshHierarchy(initial_mesh, max_levels)
        self.criteria = refinement_criteria
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        self.min_elements = min_elements
        self.max_elements = max_elements
        
        # Statistics
        self.adaptation_history = []
        self.quality_metrics = []
    
    def adapt_mesh(self, 
                   solution: Dict[str, np.ndarray], 
                   problem_context: Any,
                   target_level: Optional[int] = None) -> Tuple[MeshData, Dict[str, Any]]:
        """
        Perform mesh adaptation based on current solution.
        
        Args:
            solution: Current solution fields (density, value_function, etc.)
            problem_context: MFG problem instance for context
            target_level: Specific level to adapt (None for automatic)
        
        Returns:
            Tuple of (adapted_mesh, adaptation_info)
        """
        
        current_level = target_level if target_level is not None else self.hierarchy.current_level
        current_mesh = self.hierarchy.levels[current_level].mesh_data
        
        logger.info(f"Starting mesh adaptation at level {current_level}")
        logger.info(f"Current mesh: {current_mesh.num_elements} elements, {current_mesh.num_nodes} nodes")
        
        # Compute refinement indicators from all criteria
        refinement_indicators = self._compute_combined_indicators(
            current_mesh, solution, problem_context
        )
        
        # Decide refinement/coarsening actions
        adaptation_actions = self._determine_adaptation_actions(
            refinement_indicators, current_mesh
        )
        
        # Perform adaptations
        adapted_mesh, adaptation_info = self._execute_adaptations(
            adaptation_actions, current_level, solution
        )
        
        # Update statistics
        self._update_statistics(adaptation_info)
        
        logger.info(f"Adaptation complete: {adapted_mesh.num_elements} elements")
        
        return adapted_mesh, adaptation_info
    
    def _compute_combined_indicators(self, 
                                   mesh_data: MeshData, 
                                   solution: Dict[str, np.ndarray], 
                                   problem_context: Any) -> np.ndarray:
        """Combine indicators from all refinement criteria."""
        
        combined_indicators = np.zeros(mesh_data.num_elements)
        
        for criterion in self.criteria:
            indicators = criterion.compute_indicators(mesh_data, solution, problem_context)
            combined_indicators = np.maximum(combined_indicators, indicators)
        
        return combined_indicators
    
    def _determine_adaptation_actions(self, 
                                    indicators: np.ndarray, 
                                    mesh_data: MeshData) -> Dict[str, np.ndarray]:
        """Determine which elements to refine or coarsen."""
        
        actions = {
            'refine': np.array([], dtype=int),
            'coarsen': np.array([], dtype=int),
            'keep': np.array([], dtype=int)
        }
        
        # Check element count constraints
        if mesh_data.num_elements >= self.max_elements:
            # Only allow coarsening if at maximum
            actions['coarsen'] = np.where(indicators < self.coarsening_threshold)[0]
            actions['keep'] = np.where(indicators >= self.coarsening_threshold)[0]
            
        elif mesh_data.num_elements <= self.min_elements:
            # Only allow refinement if at minimum
            actions['refine'] = np.where(indicators > self.refinement_threshold)[0]
            actions['keep'] = np.where(indicators <= self.refinement_threshold)[0]
            
        else:
            # Normal operation
            actions['refine'] = np.where(indicators > self.refinement_threshold)[0]
            actions['coarsen'] = np.where(indicators < self.coarsening_threshold)[0]
            actions['keep'] = np.where(
                (indicators >= self.coarsening_threshold) & 
                (indicators <= self.refinement_threshold)
            )[0]
        
        return actions
    
    def _execute_adaptations(self, 
                           actions: Dict[str, np.ndarray], 
                           level: int,
                           solution: Dict[str, np.ndarray]) -> Tuple[MeshData, Dict[str, Any]]:
        """Execute the determined adaptation actions."""
        
        adaptation_info = {
            'refined_elements': len(actions['refine']),
            'coarsened_elements': len(actions['coarsen']),
            'unchanged_elements': len(actions['keep']),
            'level_changed': False,
            'new_level': level
        }
        
        current_mesh = self.hierarchy.levels[level].mesh_data
        
        # Execute refinement
        if len(actions['refine']) > 0:
            logger.info(f"Refining {len(actions['refine'])} elements")
            new_level = self.hierarchy.refine_elements(actions['refine'], level)
            current_mesh = new_level.mesh_data
            adaptation_info['level_changed'] = True
            adaptation_info['new_level'] = new_level.level
            
            # Transfer solution to new mesh
            self._transfer_solution_to_mesh(solution, current_mesh, new_level.mesh_data)
        
        # Execute coarsening (if no refinement occurred)
        elif len(actions['coarsen']) > 0 and level > 0:
            logger.info(f"Coarsening {len(actions['coarsen'])} elements")
            # Implementation for coarsening would go here
            pass
        
        return current_mesh, adaptation_info
    
    def _transfer_solution_to_mesh(self, 
                                 solution: Dict[str, np.ndarray], 
                                 old_mesh: MeshData, 
                                 new_mesh: MeshData):
        """Transfer solution fields between meshes using interpolation."""
        # Implementation for solution transfer
        # This would use techniques like:
        # - Conservative interpolation for density
        # - Higher-order interpolation for value function
        # - Projection methods for maintaining physical properties
        pass
    
    def _update_statistics(self, adaptation_info: Dict[str, Any]):
        """Update adaptation statistics."""
        self.adaptation_history.append(adaptation_info)
        
        # Compute mesh quality metrics
        current_mesh = self.get_current_mesh()
        quality = self._compute_mesh_quality(current_mesh)
        self.quality_metrics.append(quality)
    
    def _compute_mesh_quality(self, mesh_data: MeshData) -> Dict[str, float]:
        """Compute mesh quality metrics."""
        # Implementation for mesh quality analysis
        return {
            'min_angle': 0.0,
            'max_angle': 0.0,
            'aspect_ratio_mean': 0.0,
            'skewness_mean': 0.0,
            'orthogonal_quality_mean': 0.0
        }
    
    def get_current_mesh(self) -> MeshData:
        """Get the current active mesh."""
        return self.hierarchy.levels[self.hierarchy.current_level].mesh_data
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation process."""
        return {
            'total_adaptations': len(self.adaptation_history),
            'current_level': self.hierarchy.current_level,
            'current_elements': self.get_current_mesh().num_elements,
            'quality_evolution': self.quality_metrics,
            'adaptation_history': self.adaptation_history
        }
```

### 4. Integration with GeneralMFGProblem

```python
# Extension to GeneralMFGProblem for AMR support

class AMRConfiguration:
    """Configuration for adaptive mesh refinement."""
    
    def __init__(self,
                 enabled: bool = False,
                 refinement_criteria: List[str] = None,
                 refinement_threshold: float = 0.3,
                 coarsening_threshold: float = 0.1,
                 max_levels: int = 5,
                 adaptation_frequency: int = 10,  # Adapt every N solver iterations
                 min_elements: int = 100,
                 max_elements: int = 100000):
        
        self.enabled = enabled
        self.refinement_criteria = refinement_criteria or ["gradient", "mfg_density"]
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        self.max_levels = max_levels
        self.adaptation_frequency = adaptation_frequency
        self.min_elements = min_elements
        self.max_elements = max_elements

# Add AMR support to MFGComponents
@dataclass
class MFGComponents:
    # ... existing fields ...
    
    # AMR configuration
    amr_config: Optional[AMRConfiguration] = None
    
    # AMR-specific functions
    custom_refinement_criterion: Optional[Callable] = None
```

---

## 🎯 MFG-Specific Refinement Strategies

### 1. Density Gradient Refinement
- **Target**: Regions where agent density changes rapidly
- **Criterion**: `|∇m(x,t)| > threshold`
- **Application**: Crowd dynamics, traffic flow, evacuation scenarios

### 2. Value Function Curvature Refinement
- **Target**: Areas of high value function curvature
- **Criterion**: `|∇²u(x,t)| > threshold`
- **Application**: Financial optimization, optimal control problems

### 3. Hamilton-Jacobi Shock Refinement
- **Target**: Discontinuities in value function gradients
- **Criterion**: Jump conditions in `∇u`
- **Application**: Non-smooth HJ equations, front propagation

### 4. Coupling Strength Refinement
- **Target**: Regions where mean field coupling is strong
- **Criterion**: `|dH/dm| > threshold`
- **Application**: Strong interaction problems, congestion effects

---

## 📊 Performance Considerations

### Computational Complexity
- **Memory**: O(N × L) where N = base elements, L = refinement levels
- **Refinement Cost**: O(R log R) where R = elements to refine
- **Solution Transfer**: O(N_old + N_new) for interpolation

### Optimization Strategies
1. **Lazy Evaluation**: Only compute refinement indicators when needed
2. **Incremental Updates**: Track changes since last adaptation
3. **Parallel Refinement**: Distribute refinement operations across cores
4. **Memory Pooling**: Reuse memory for temporary refinement operations

---

## 🔍 Quality Metrics and Validation

### Mesh Quality Indicators
- **Aspect Ratio**: Element elongation measure
- **Skewness**: Deviation from ideal element shape
- **Orthogonal Quality**: Mesh orthogonality measure
- **Growth Rate**: Element size variation smoothness

### Adaptation Validation
- **Solution Convergence**: Monitor solution error reduction
- **Conservation Properties**: Ensure mass conservation during adaptation
- **Stability Analysis**: Check solver stability on adapted meshes

---

## 🚀 Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
1. ✅ Implement basic refinement criteria classes
2. ✅ Create mesh hierarchy management
3. ✅ Develop AMR manager framework

### Phase 2: MFG Integration (Week 3-4)
1. Integrate with GeneralMFGProblem
2. Implement MFG-specific refinement criteria
3. Add solution transfer capabilities

### Phase 3: Advanced Features (Week 5-6)
1. Implement coarsening algorithms
2. Add parallel refinement support
3. Develop quality metrics and validation

### Phase 4: Testing and Optimization (Week 7-8)
1. Comprehensive testing with various MFG problems
2. Performance optimization and profiling
3. Documentation and examples

---

## 📚 Usage Examples

### Basic AMR Configuration
```python
from mfg_pde import GeneralMFGProblem, MFGProblemBuilder
from mfg_pde.geometry.adaptive import AMRConfiguration

amr_config = AMRConfiguration(
    enabled=True,
    refinement_criteria=["gradient", "mfg_density"],
    refinement_threshold=0.3,
    max_levels=4,
    adaptation_frequency=5  # Adapt every 5 solver iterations
)

problem = (MFGProblemBuilder()
          .hamiltonian(my_hamiltonian, my_hamiltonian_dm)
          .domain_2d(width=2.0, height=1.0, refinement_level=0)
          .amr_configuration(amr_config)
          .build())

# Solver automatically uses AMR during iterations
solver = create_fast_solver()
result = solver.solve(problem)

# Access adaptation information
print(f"Final mesh elements: {result.mesh_info['final_elements']}")
print(f"Adaptation levels used: {result.mesh_info['max_level_reached']}")
```

### Custom Refinement Criterion
```python
def custom_mfg_refinement(mesh_data, solution, problem_context):
    """Custom refinement for specific MFG problem."""
    density = solution["density"]
    value_func = solution["value_function"]
    
    # Refine where density * value gradient is high
    density_value_product = density * np.linalg.norm(
        problem_context.compute_gradient(value_func), axis=1
    )
    
    return (density_value_product > 0.1).astype(float)

problem = (MFGProblemBuilder()
          .hamiltonian(my_hamiltonian, my_hamiltonian_dm)
          .amr_configuration(AMRConfiguration(
              enabled=True,
              custom_refinement_criterion=custom_mfg_refinement
          ))
          .build())
```

---

## 🎉 Expected Benefits

### Research Benefits
- **Higher Accuracy**: Automatic refinement in solution-critical regions
- **Computational Efficiency**: Fewer elements in smooth regions
- **Problem Adaptability**: Mesh automatically adapts to problem characteristics
- **Research Productivity**: Less manual mesh tuning required

### Technical Benefits
- **Robust Framework**: Professional-grade AMR implementation
- **Scalability**: Handles problems from small 2D to large 3D domains
- **Integration**: Seamless integration with existing MFG_PDE infrastructure
- **Extensibility**: Easy to add new refinement criteria and strategies

This adaptive mesh refinement system will significantly enhance MFG_PDE's capabilities for solving complex 2D and 3D Mean Field Games problems with optimal computational efficiency and accuracy.