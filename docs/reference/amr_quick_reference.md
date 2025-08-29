# AMR Quick Reference Guide

**Date**: August 1, 2025  
**Module**: `mfg_pde.geometry.amr_mesh`, `mfg_pde.alg.mfg_solvers.amr_mfg_solver`

## Quick Start

### Basic Usage
```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_amr_solver

# Create problem
problem = ExampleMFGProblem(Nx=32, Nt=50, T=1.0)

# Create AMR solver
amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-4,    # Refine when error > threshold
    max_levels=4,            # Maximum refinement levels
    base_solver="fixed_point"
)

# Solve
result = amr_solver.solve(max_iterations=100, tolerance=1e-6, verbose=True)
```

### Factory Function Parameters

```python
create_amr_solver(
    problem: MFGProblem,
    error_threshold: float = 1e-4,     # Error threshold for refinement
    max_levels: int = 5,               # Maximum refinement levels
    base_solver: str = "fixed_point",  # Underlying solver type
    domain_bounds: Tuple = None,       # Custom domain bounds
    amr_frequency: int = 5,            # Adapt every N iterations
    max_amr_cycles: int = 3,           # Maximum AMR cycles
    backend: str = "auto",             # Backend selection
    **kwargs                           # Additional solver parameters
) -> AMRMFGSolver
```

## Configuration Presets

### Speed-Optimized
```python
amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-3,    # Less aggressive refinement
    max_levels=3,            # Fewer levels
    amr_frequency=10,        # Less frequent adaptation
    max_amr_cycles=2,        # Fewer cycles
    backend="jax"            # GPU acceleration
)
```

### Accuracy-Optimized
```python
amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-6,    # Very aggressive refinement
    max_levels=6,            # Many levels
    amr_frequency=3,         # Frequent adaptation
    max_amr_cycles=5,        # Many cycles
    base_solver="particle_collocation"
)
```

### Balanced (Recommended)
```python
amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-4,    # Moderate refinement
    max_levels=4,            # Reasonable levels
    amr_frequency=5,         # Default frequency
    max_amr_cycles=3,        # Default cycles
    backend="auto"
)
```

## Advanced Configuration

### Custom Refinement Criteria
```python
from mfg_pde.geometry.amr_mesh import AMRRefinementCriteria, create_amr_mesh
from mfg_pde.alg.mfg_solvers.amr_mfg_solver import AMRMFGSolver

# Custom criteria
criteria = AMRRefinementCriteria(
    error_threshold=1e-5,
    gradient_threshold=0.05,
    max_refinement_levels=6,
    min_cell_size=1e-7,
    coarsening_threshold=0.02,
    adaptive_error_scaling=True
)

# Custom mesh
amr_mesh = create_amr_mesh(
    domain_bounds=(-3.0, 3.0, -2.0, 2.0),
    error_threshold=criteria.error_threshold,
    max_levels=criteria.max_refinement_levels,
    backend="jax"
)

# Custom solver
amr_solver = AMRMFGSolver(
    problem=problem,
    adaptive_mesh=amr_mesh,
    base_solver_type="particle_collocation",
    amr_frequency=3,
    max_amr_cycles=5
)
```

## Key Classes and Methods

### AMRMFGSolver
```python
class AMRMFGSolver(BaseMFGSolver):
    # Main solve method
    def solve(self, max_iterations=100, tolerance=1e-6, verbose=True) -> SolverResult
    
    # Get comprehensive AMR statistics
    def get_amr_statistics(self) -> Dict[str, Any]
```

### AdaptiveMesh
```python
class AdaptiveMesh:
    # Refine mesh based on solution data
    def refine_mesh(self, solution_data: Dict[str, NDArray]) -> int
    
    # Coarsen mesh where appropriate
    def coarsen_mesh(self, solution_data: Dict[str, NDArray]) -> int
    
    # Complete adaptation cycle
    def adapt_mesh(self, solution_data: Dict[str, NDArray]) -> Dict[str, int]
    
    # Get mesh statistics
    def get_mesh_statistics(self) -> Dict[str, Any]
```

### QuadTreeNode
```python
@dataclass
class QuadTreeNode:
    level: int              # Refinement level
    x_min, x_max: float    # Cell boundaries
    y_min, y_max: float
    is_leaf: bool = True   # Leaf node flag
    
    # Properties
    @property
    def center_x(self) -> float     # Cell center X
    @property
    def center_y(self) -> float     # Cell center Y
    @property
    def dx(self) -> float           # Cell width
    @property
    def dy(self) -> float           # Cell height
    @property
    def area(self) -> float         # Cell area
    
    # Methods
    def contains_point(self, x: float, y: float) -> bool
    def subdivide(self) -> List['QuadTreeNode']
```

## Result Analysis

### AMR Statistics
```python
result = amr_solver.solve()

# Get AMR statistics
amr_stats = amr_solver.get_amr_statistics()

# Mesh statistics
mesh_stats = amr_stats['mesh_statistics']
print(f"Total cells: {mesh_stats['total_cells']}")
print(f"Max level: {mesh_stats['max_level']}")
print(f"Level distribution: {mesh_stats['level_distribution']}")
print(f"Refinement ratio: {mesh_stats['refinement_ratio']:.2f}")

# Adaptation statistics
adapt_stats = amr_stats['adaptation_statistics']  
print(f"Total refinements: {adapt_stats['total_refinements']}")
print(f"Total coarsenings: {adapt_stats['total_coarsenings']}")
print(f"Adaptation cycles: {adapt_stats['adaptation_cycles']}")

# Efficiency metrics
efficiency = amr_stats['efficiency_metrics']
print(f"Average efficiency: {efficiency['average_efficiency']:.3f}")
```

### Result Data
```python
# Solution arrays
U = result.U              # Value function
M = result.M              # Density function

# Convergence information
print(f"Converged: {result.convergence_achieved}")
print(f"Final error: {result.final_error:.2e}")
print(f"Iterations: {len(result.convergence_history)}")

# AMR-specific solver info
solver_info = result.solver_info
if 'amr_stats' in solver_info:
    print(f"Refinements: {solver_info['amr_stats']['total_refinements']}")
if 'final_mesh_stats' in solver_info:
    print(f"Final cells: {solver_info['final_mesh_stats']['total_cells']}")
```

## Parameter Guidelines

### Error Threshold Selection
| Problem Type | Recommended Threshold | Description |
|--------------|----------------------|-------------|
| Smooth solutions | 1e-3 | Conservative refinement |
| Moderate gradients | 1e-4 | Balanced refinement |
| Sharp features | 1e-5 | Aggressive refinement |
| Very sharp/discontinuous | 1e-6 | Very aggressive |

### Max Levels Selection
| Target Resolution Factor | Max Levels | Memory Impact |
|-------------------------|------------|---------------|
| 2× base resolution | 1 | Low |
| 4× base resolution | 2 | Low-Medium |
| 8× base resolution | 3 | Medium |
| 16× base resolution | 4 | Medium-High |
| 32× base resolution | 5 | High |
| 64× base resolution | 6 | Very High |

### AMR Frequency Guidelines
| Adaptation Pattern | AMR Frequency | Use Case |
|-------------------|---------------|----------|
| Frequent tracking | 3-4 | Rapidly evolving solutions |
| Balanced adaptation | 5-7 | Most problems |
| Infrequent updates | 8-12 | Slowly evolving solutions |

## Common Issues and Solutions

### Issue: No Refinements Occurring
```python
# Check: Error threshold too high?
amr_solver = create_amr_solver(problem, error_threshold=1e-5)  # Lower threshold

# Check: Max levels too restrictive?
amr_solver = create_amr_solver(problem, max_levels=5)  # Allow more levels

# Check: Solution too smooth?
problem = ExampleMFGProblem(sigma=0.05)  # Reduce diffusion for sharper features
```

### Issue: Excessive Memory Usage
```python
# Solution: Increase error threshold
amr_solver = create_amr_solver(problem, error_threshold=1e-3)

# Solution: Limit max levels
amr_solver = create_amr_solver(problem, max_levels=3)

# Solution: Set minimum cell size
from mfg_pde.geometry.amr_mesh import AMRRefinementCriteria
criteria = AMRRefinementCriteria(min_cell_size=1e-6)
```

### Issue: Poor Performance
```python
# Solution: Reduce adaptation frequency
amr_solver = create_amr_solver(problem, amr_frequency=10)

# Solution: Use JAX backend
amr_solver = create_amr_solver(problem, backend="jax")

# Solution: Fewer AMR cycles
amr_solver = create_amr_solver(problem, max_amr_cycles=1)
```

## JAX Acceleration

### Enable JAX Backend
```python
# Automatic backend selection (recommended)
amr_solver = create_amr_solver(problem, backend="auto")

# Force JAX backend (for GPU acceleration)
amr_solver = create_amr_solver(problem, backend="jax")

# Force NumPy backend (for CPU-only)
amr_solver = create_amr_solver(problem, backend="numpy")
```

### JAX-Specific Functions
```python
from mfg_pde.alg.mfg_solvers.amr_mfg_solver import JAXAcceleratedAMR

# GPU-accelerated error computation
error_indicators = JAXAcceleratedAMR.compute_error_indicators(U, M, dx, dy)

# Conservative mass interpolation
density_new = JAXAcceleratedAMR.conservative_mass_interpolation_2d(
    density, old_dx, old_dy, new_dx, new_dy
)

# Gradient-preserving interpolation
values_new = JAXAcceleratedAMR.gradient_preserving_interpolation_2d(
    values, old_dx, old_dy, new_dx, new_dy
)
```

## Integration with MFG_PDE

### Factory Integration
```python
# Available through main factory
from mfg_pde.factory import create_amr_solver

# Also available as solver type
from mfg_pde.factory import create_solver
amr_solver = create_solver(problem, solver_type="amr", error_threshold=1e-4)

# Integration with other factory functions
from mfg_pde.factory import create_accurate_solver
amr_solver = create_accurate_solver(problem, solver_type="amr")
```

### Backend Integration
```python
# Works with existing backend system
from mfg_pde.backends import create_backend

backend = create_backend("jax")
amr_solver = create_amr_solver(problem, backend=backend)
```

## Mathematical Background

### Error Estimation
AMR uses gradient-based error indicators:
```
η_cell = max(||∇U||, ||∇M||) + 0.1 * max(||∇²U||, ||∇²M||)
```

### Refinement Criterion
```
Refine if: η_cell > error_threshold
Coarsen if: η_cell < coarsening_threshold * error_threshold
```

### Conservative Interpolation
Mass conservation enforced:
```
∫_Ω M_new(x) dx = ∫_Ω M_old(x) dx
```

## Performance Expectations

### Typical Performance Gains
- **Accuracy**: 2-5× better for same computational cost
- **Memory**: 30-70% reduction for localized features
- **Speed**: 1.5-3× faster convergence
- **GPU Acceleration**: 5-15× speedup with JAX

### Recommended Use Cases
- ✅ Sharp gradients, boundary layers
- ✅ Localized phenomena (shocks, fronts, congestion)  
- ✅ Multi-scale problems
- ✅ Resource-constrained high-accuracy simulations
- ❌ Globally smooth solutions
- ❌ Small problems where uniform grids suffice

---

**Last Updated**: August 1, 2025  
**Version**: MFG_PDE 2.0+
